############################################################
# nbme_deberta_v3_small.py  –– good for kaggle competition (offline rule)
############################################################
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # enable/disable parallelism
import ast, random, json
from pathlib import Path
import glob
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

import torch, torch.nn as nn

from torchcrf import CRF
import io, logging  # for tqdm→logger bridge

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModel,
    get_cosine_schedule_with_warmup,
)
from torch.cuda.amp import autocast
from torch.amp import GradScaler

# -------------- 1. 全域設定 -------------------------------
class CFG:
    data_dir      = Path("/home/hjc/codeSpace/NLP_Final/nbme_data")  # Local raw‑data directory
    competition   = "nbme-score-clinical-patient-notes"  # kept for reference
    model_name    = "/home/hjc/codeSpace/NLP_Final/deberta-v3-small"  # local checkpoint
    ensemble_model_name = "/home/hjc/codeSpace/NLP_Final/Bio_ClinicalBERT_offline"
    max_len       = 384          # 由final_EDA 得知 443的長度其實已經足夠，預留餘度
    batch_size    = 4
    gradient_accum = 4
    epochs        = 4
    lr            = 2e-5
    weight_decay  = 0.01
    scheduler     = "cosine"
    warmup_ratio  = 0.1
    n_folds       = 5
    seed          = 42
    output_dir    = "./NLP_Final/nbme_ckpt"
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    inference_fold = 3
    checkpoint_dir = Path("/home/hjc/codeSpace/NLP_Final/fold_checkpoints")

# -------------- Utils: logger & seed --------------------
OUTPUT_DIR = CFG.output_dir  # For log file path

def get_logger(filename=OUTPUT_DIR + '/train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    log_path = Path(filename).parent
    log_path.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    logger = getLogger(__name__)
    if logger.handlers:  # avoid duplicate handlers
        return logger
    logger.setLevel(INFO)

    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))

    handler2 = FileHandler(f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

class _TqdmToLogger(io.StringIO):
    """File‑like object that redirects tqdm output to our LOGGER."""
    def __init__(self, logger: logging.Logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf):
        buf = buf.strip()
        if buf:
            self.logger.log(self.level, buf)

    def flush(self):
        pass  # tqdm expects this method.

TQDM_LOGGER = _TqdmToLogger(logging.getLogger(__name__))

LOGGER = get_logger()

def seed_everything(seed: int = 42):
    """Seed all RNGs for full reproducibility (deterministic cuDNN)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 可選：將 CFG 輸出成 JSON 便於日後追蹤

from pathlib import Path
(Path(CFG.output_dir).mkdir(exist_ok=True, parents=True))
cfg_dict = {
    k: (str(v) if isinstance(v, Path) else v)
    for k, v in CFG.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
json.dump(cfg_dict, open(Path(CFG.output_dir)/"cfg.json", "w"), indent=2)

def main():
    seed_everything(CFG.seed)

    # 混合精度設定
    scaler = GradScaler()
    # -------------- 2. 資料讀取與預處理 -----------------------
    BASE = CFG.data_dir
    # 支援任意檔名前綴；以 *train*, *test* 等關鍵字辨識
    def _find_csv(keyword: str) -> Path:
        patt = str(BASE / "*.csv")
        files = [Path(f) for f in glob.glob(patt) if keyword.lower() in Path(f).stem.lower()]
        if not files:
            raise FileNotFoundError(f"[Error] CSV containing '{keyword}' not found under {BASE.resolve()}")
        return files[0]

    train  = pd.read_csv(_find_csv("train"))
    test   = pd.read_csv(_find_csv("test"))
    feats  = pd.read_csv(_find_csv("features"))
    pnotes = pd.read_csv(_find_csv("patient_notes"))

    # ---------- 修補 incorrect annotations -----------------
    def _parse_list(x: str):
        """Safe eval to list; returns [] on empty/malformed."""
        try:
            return ast.literal_eval(x) if isinstance(x, str) and x else []
        except Exception:
            return []

    def _find_all_spans(note: str, phrase: str) -> List[Tuple[int, int]]:
        """Return all (start,end) spans (inclusive-exclusive) of phrase in note, case-insensitive."""
        spans = []
        for m in re.finditer(re.escape(phrase), note, flags=re.I):
            spans.append((m.start(), m.end()))
        return spans

    def apply_annotation_fixes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix known bad rows and attempt auto‑repair when location missing.
        Returns a *new* DataFrame (copy).
        """
        df = df.copy()

        # 1) manual whitelist based on public notebooks
        MANUAL_FIX: dict[int, tuple[str, str]] = {
            338:  ("father heart attack", "764 783"),
            621:  ("for the last 2-3 months", "77 100"),
            1262: ("mother thyroid problem", "551 572"),
        }
        for idx, (ann, loc) in MANUAL_FIX.items():
            if idx in df.index:
                df.at[idx, "annotation"] = f'["{ann}"]'
                df.at[idx, "location"]   = f'["{loc}"]'

        # 2) auto repair for rows where len(annotation_list) != len(location_list)
        auto_fixed = 0
        for i, row in df.iterrows():
            anns = _parse_list(row["annotation"])
            locs = _parse_list(row["location"])
            if len(anns) == len(locs) and len(locs) > 0:
                continue  # already aligned
            note = row["pn_history"]
            new_locs = []
            used = set()
            for ann in anns:
                spans = _find_all_spans(note, ann)
                # pick the first span not yet used
                chosen = None
                for s, e in spans:
                    if (s, e) not in used:
                        chosen = (s, e); break
                if chosen is None:
                    break  # cannot find unique span
                used.add(chosen)
                new_locs.append(f"{chosen[0]} {chosen[1]}")
            if len(new_locs) == len(anns):  # successful repair
                df.at[i, "location"] = str([*new_locs])
                auto_fixed += 1
        LOGGER.info(f"[annotation fix] manual={len(MANUAL_FIX)}, auto={auto_fixed}")
        return df

    #
    # merge
    train = (train.merge(feats, on=["feature_num","case_num"], how="left")
                   .merge(pnotes, on=["pn_num","case_num"],    how="left"))
    test  = (test .merge(feats, on=["feature_num","case_num"], how="left")
                   .merge(pnotes, on=["pn_num","case_num"],    how="left"))

    # apply annotation fixes now that pn_history is present
    train = apply_annotation_fixes(train)

    # 解析 annotation / location 成 list⇢list[int]
    def str2list(x): 
        return ast.literal_eval(x) if isinstance(x,str) and x!="" else []
    train["annotation_list"] = train["annotation"].apply(str2list)
    train["location_list"]   = train["location"].apply(str2list)

    tok = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
    tok_clinical = AutoTokenizer.from_pretrained(CFG.ensemble_model_name, use_fast=True)
    def create_char_targets(text: str, spans: List[str]) -> np.ndarray:
        """Return a 0/1 vector marking characters belonging to any span."""
        targets = np.zeros(len(text), dtype=np.int8)
        for span in spans:
            if not span:
                continue
            for loc in span.split(";"):
                if loc == "":
                    continue
                try:
                    start, end = map(int, loc.split())
                except ValueError:
                    continue  # skip malformed
                if start >= len(text):
                    continue
                end = min(end, len(text))
                targets[start:end] = 1
        return targets

    def encode_example(note: str, feature: str, targets: np.ndarray | None):
        """
        Tokenise (feature, note) pair using two tokenizers and return a dict of tensors compatible
        with the model.  `labels` tensor is int64 for CRF; others are long.
        """
        enc1 = tok(
            feature,
            note,
            truncation="only_second",
            padding="max_length",
            max_length=CFG.max_len,
            return_offsets_mapping=True
        )
        enc2 = tok_clinical(
            feature,
            note,
            truncation="only_second",
            padding="max_length",
            max_length=CFG.max_len
        )

        if targets is not None:
            # 0: O, 1: B/I
            labels = np.zeros(len(enc1["input_ids"]), dtype=np.int64)
            seq_ids = enc1.sequence_ids()
            for idx, (s, e) in enumerate(enc1["offset_mapping"]):
                if seq_ids[idx] != 1 or s == e:
                    continue
                if targets[s:e].max() > 0:
                    labels[idx] = 1
        else:
            labels = None

        # Remove offset_mapping from enc1
        enc1.pop("offset_mapping")

        tensor_dict = {
            "input_ids": torch.tensor(enc1["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc1["attention_mask"], dtype=torch.long),
            "input_ids2": torch.tensor(enc2["input_ids"], dtype=torch.long),
            "attention_mask2": torch.tensor(enc2["attention_mask"], dtype=torch.long),
        }
        if labels is not None:
            tensor_dict["labels"] = torch.tensor(labels, dtype=torch.long)
        return tensor_dict

    # -------------- 4. 自訂 Dataset ---------------------------
    class NBMEDataset(Dataset):
        def __init__(self, df:pd.DataFrame, is_train=True):
            self.df = df
            self.is_train = is_train
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            if self.is_train:
                # row.location_list is already List[str] like ["12 20", "32 40"]
                # just pass it directly
                char_targets = create_char_targets(row.pn_history, row.location_list)
            else:
                char_targets = None
            return encode_example(row.pn_history, row.feature_text, char_targets)

    # -------------- 5. 評分工具 (rule.md 規則) ----------------
    def span_to_char_set(spans: List[str]) -> set[int]:
        """
        Convert list of span strings ["12 20", "30 35;40 45", ...] to a set of char indices.
        Any malformed span tokens are skipped gracefully.
        """
        char_set = set()
        for sp in spans:
            if not sp:
                continue
            for loc in sp.split(";"):
                if not loc.strip():
                    continue
                parts = loc.split()
                if len(parts) != 2:
                    continue  # skip malformed "start end"
                try:
                    a, b = map(int, parts)
                except ValueError:
                    continue
                if a >= b:
                    continue
                char_set.update(range(a, b))
        return char_set
    def compute_micro_f1(pred_df:pd.DataFrame) -> float:
        """pred_df 需含 columns: id, ground (list[str] or str), pred (list[str] or str)"""
        tp = fp = fn = 0
        for g, p in zip(pred_df.ground, pred_df.pred):
            # 將 ground 及 pred 統一為 list[str]
            if isinstance(g, str):
                spans_g = [s for s in g.split(";") if s.strip()]
            else:
                spans_g = g
            if isinstance(p, str):
                spans_p = [s for s in p.split(";") if s.strip()]
            else:
                spans_p = p
            # 轉 char index set
            gset = span_to_char_set(spans_g)
            pset = span_to_char_set(spans_p)
            tp += len(gset & pset)
            fp += len(pset - gset)
            fn += len(gset - pset)
        # 計算並回傳 micro F1
        return 2 * tp / (2 * tp + fp + fn + 1e-8)
    def span_micro_f1(y_true, y_pred):
        """Alias wrapper so external utils can call the same scorer name."""
        return compute_micro_f1(pd.DataFrame({"ground": y_true, "pred": y_pred}))
    def get_score(y_true, y_pred):
        return span_micro_f1(y_true, y_pred)

    # -------------- 6. 模型 -------------------------------

    class DebertaWithCRF(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(CFG.model_name)
            self.dropout  = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.backbone.config.hidden_size, 2)  # 2 tags: O / B-I
            self.crf = CRF(2, batch_first=True)
            # load fine-tuned fold weights
            fold_path = Path(CFG.checkpoint_dir) / f"fold{CFG.inference_fold}.pt"
            if fold_path.exists():
                checkpoint = torch.load(fold_path, map_location=CFG.device)
                self.load_state_dict(checkpoint)
            else:
                LOGGER.warning(f"Fold checkpoint {fold_path} not found, using pretrained backbone only")
        def forward(self, input_ids, attention_mask, labels=None):
            # backbone
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)  # (B, L, H)
            emissions = self.classifier(sequence_output)  # (B, L, 2)
            loss = None
            if labels is not None:
                # CRF loss: negative log-likelihood
                loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            # decode best path
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return {"loss": loss, "predictions": predictions}


    class EnsembleWithCRF(nn.Module):
        def __init__(self):
            super().__init__()
            # 兩個 backbone：DeBERTa 與 ClinicalBERT
            self.backbone1 = AutoModel.from_pretrained(CFG.model_name)
            self.backbone2 = AutoModel.from_pretrained(CFG.ensemble_model_name)
            self.dropout   = nn.Dropout(0.1)
            # 共用分類層輸出 2 個 tag
            self.classifier = nn.Linear(self.backbone1.config.hidden_size, 2)
            self.crf = CRF(2, batch_first=True)
        def forward(self, input_ids, attention_mask, input_ids2, attention_mask2, labels=None):
            out1 = self.backbone1(input_ids=input_ids, attention_mask=attention_mask)
            out2 = self.backbone2(input_ids=input_ids2, attention_mask=attention_mask2)
            seq1 = self.dropout(out1.last_hidden_state)
            seq2 = self.dropout(out2.last_hidden_state)
            # 平均 emissions
            emissions1 = self.classifier(seq1)
            emissions2 = self.classifier(seq2)
            emissions = (emissions1 + emissions2) / 2
            loss = None
            if labels is not None:
                loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return {"loss": loss, "predictions": predictions}

    def generate_pseudo_labels(model, test_loader, threshold=0.9):
        model.eval()
        pseudo_data = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(CFG.device)
                attention_mask = batch["attention_mask"].to(CFG.device)
                input_ids2 = batch["input_ids2"].to(CFG.device)
                attention_mask2 = batch["attention_mask2"].to(CFG.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                input_ids2=input_ids2, attention_mask2=attention_mask2)
                predictions = outputs["predictions"]  # list of lists of tags
                for idx, tags in enumerate(predictions):
                    prob = sum([1 for t in tags if t!=0]) / len(tags)
                    if prob >= threshold:
                        # 將 tags 轉成 spans 字串
                        spans = []
                        start = None
                        for i, t in enumerate(tags):
                            if t != 0 and start is None:
                                start = i
                            elif t == 0 and start is not None:
                                spans.append(f"{start} {i}")
                                start = None
                        if start is not None:
                            spans.append(f"{start} {len(tags)}")
                        pseudo_data.append((batch["id"][idx].item() if hasattr(batch["id"][idx], "item") else batch["id"][idx], spans))
        return pseudo_data

    # -------------- 7. Pseudolabeling + 交叉驗證訓練 ------------------------
    # 先進行一輪初步訓練
    oof_preds, oof_gts = [], []
    gkf = GroupKFold(n_splits=CFG.n_folds)
    for_pseudo_model = EnsembleWithCRF().to(CFG.device)
    # 啟用梯度檢查點
    for_pseudo_model.backbone1.gradient_checkpointing_enable()
    for_pseudo_model.backbone2.gradient_checkpointing_enable()
    # 用全部訓練資料訓練一小輪
    trn_ds = NBMEDataset(train)
    trn_loader = DataLoader(trn_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2, pin_memory=(CFG.device=="cuda"))
    optimizer = torch.optim.AdamW(for_pseudo_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    num_training_steps = 1 * len(trn_loader) // CFG.gradient_accum  # 只跑1 epoch
    num_warmup = int(CFG.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_training_steps)
    for epoch in range(1):
        for_pseudo_model.train(); running=0
        pbar = tqdm(trn_loader, total=len(trn_loader), desc=f"Pseudolabel Pretrain", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(CFG.device) for k, v in batch.items()}
            with autocast():
                out = for_pseudo_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    input_ids2=batch["input_ids2"],
                    attention_mask2=batch["attention_mask2"],
                    labels=batch["labels"]
                )
                loss = out["loss"] / CFG.gradient_accum
            scaler.scale(loss).backward()
            running += loss.item()
            if (step+1)%CFG.gradient_accum==0 or step+1==len(trn_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(loss=running/((step+1)//CFG.gradient_accum+1e-6))
    # 偽標籤產生
    test_ds = NBMEDataset(test, is_train=False)
    pin_memory = CFG.device == "cuda"
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    pseudo_data = generate_pseudo_labels(for_pseudo_model, test_loader, threshold=0.9)
    # 加入偽標籤到 train DataFrame
    pseudo_rows = []
    for pseudo_id, spans in pseudo_data:
        # 找到 test row
        test_row = test[test['id']==pseudo_id]
        if test_row.empty: continue
        row = test_row.iloc[0].copy()
        row["annotation_list"] = ["pseudo"]*len(spans) if spans else []
        row["location_list"] = spans
        pseudo_rows.append(row)
    if pseudo_rows:
        train = pd.concat([train, pd.DataFrame(pseudo_rows)], ignore_index=True)
    # 重新構造 dataset
    oof_preds, oof_gts = [], []
    gkf = GroupKFold(n_splits=CFG.n_folds)
    for fold,(trn_idx,val_idx) in enumerate(gkf.split(train, groups=train.pn_num)):
        LOGGER.info(f"\n========== FOLD {fold} ==========")
        trn_ds = NBMEDataset(train.iloc[trn_idx])
        val_ds = NBMEDataset(train.iloc[val_idx])
        pin_memory = CFG.device == "cuda"
        trn_loader = DataLoader(trn_ds, batch_size=CFG.batch_size,
                                shuffle=True, num_workers=2, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=CFG.batch_size,
                                shuffle=False,num_workers=2, pin_memory=pin_memory)

        model = EnsembleWithCRF().to(CFG.device)
        # 啟用梯度檢查點
        model.backbone1.gradient_checkpointing_enable()
        model.backbone2.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                      weight_decay=CFG.weight_decay)
        num_training_steps = CFG.epochs * len(trn_loader) // CFG.gradient_accum
        num_warmup = int(CFG.warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup, num_training_steps)

        best_f1 = -1; best_path = Path(CFG.output_dir)/f"fold{fold}.pt"
        for epoch in range(CFG.epochs):
            # ---- train
            model.train(); running=0
            pbar = tqdm(
                trn_loader,
                total=len(trn_loader),
                desc=f"Train E{epoch}",
                dynamic_ncols=True,
                leave=False,
            )
            for step,batch in enumerate(pbar):
                batch = {k:v.to(CFG.device) for k,v in batch.items()}
                with autocast():
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        input_ids2=batch["input_ids2"],
                        attention_mask2=batch["attention_mask2"],
                        labels=batch["labels"]
                    )
                    loss = out["loss"] / CFG.gradient_accum
                scaler.scale(loss).backward()
                running += loss.item()
                if (step+1)%CFG.gradient_accum==0 or step+1==len(trn_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.set_postfix(loss=running/((step+1)//CFG.gradient_accum+1e-6))
            # ---- valid
            model.eval(); preds=[]; gts=[]; ids=[]
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Valid", dynamic_ncols=True, leave=False):
                    input_ids = batch["input_ids"].to(CFG.device)
                    attention_mask = batch["attention_mask"].to(CFG.device)
                    input_ids2 = batch["input_ids2"].to(CFG.device)
                    attention_mask2 = batch["attention_mask2"].to(CFG.device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_ids2=input_ids2,
                        attention_mask2=attention_mask2
                    )
                    predictions = outputs["predictions"]
                    preds.extend(predictions)
            # 將predictions轉回spans
            for i, row in enumerate(train.iloc[val_idx].itertuples()):
                pred_tags = preds[i]
                enc = tok(row.feature_text, row.pn_history,
                          truncation="only_second", max_length=CFG.max_len,
                          return_offsets_mapping=True)
                seq_ids_val = enc.sequence_ids()
                offsets = enc["offset_mapping"]
                # Collect char-level predictions from tags
                char_pred = np.zeros(len(row.pn_history), dtype=np.uint8)
                for t, tag in enumerate(pred_tags):
                    if seq_ids_val[t]==1 and tag==1:
                        s, e = offsets[t]
                        if s < e and s < len(char_pred):
                            char_pred[s:min(e, len(char_pred))] = 1
                # 根據char_pred重建spans
                spans = []
                start = None
                for idx, pv in enumerate(char_pred):
                    if pv == 1 and start is None:
                        start = idx
                    elif (pv == 0 or idx == len(char_pred)-1) and start is not None:
                        end = idx if pv==0 else idx+1
                        spans.append(f"{start} {end}")
                        start = None
                pred_span = ";".join(spans)
                oof_preds.append(pred_span)
                oof_gts.append(";".join(row.location_list))
            f1 = compute_micro_f1(pd.DataFrame({"ground":oof_gts[-len(val_idx):],
                                                "pred":  oof_preds[-len(val_idx):]}))
            LOGGER.info(f"Fold {fold} Epoch {epoch} F1={f1:.4f}")
            if f1>best_f1:
                best_f1=f1
                torch.save(model.state_dict(), best_path)
        LOGGER.info(f"Fold {fold} best F1={best_f1:.4f}")

    # -------------- 8. 整體 OOF 分數 ------------------------
    overall_f1 = compute_micro_f1(pd.DataFrame({"ground":oof_gts,"pred":oof_preds}))
    LOGGER.info(f"\n========== CV micro-F1: {overall_f1:.4f} ==========")

    # -------------- 9. 測試推論 & 提交 ----------------------
    test_ds = NBMEDataset(test, is_train=False)
    pin_memory = CFG.device == "cuda"
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size,
                             shuffle=False, num_workers=2, pin_memory=pin_memory)

    all_preds=[]
    for fold in range(CFG.n_folds):
        model = EnsembleWithCRF().to(CFG.device)
        # 啟用梯度檢查點
        model.backbone1.gradient_checkpointing_enable()
        model.backbone2.gradient_checkpointing_enable()
        model.load_state_dict(torch.load(Path(CFG.output_dir)/f"fold{fold}.pt",
                                         map_location=CFG.device))
        model.eval(); fold_pred=[]
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(CFG.device)
                attention_mask = batch["attention_mask"].to(CFG.device)
                input_ids2 = batch["input_ids2"].to(CFG.device)
                attention_mask2 = batch["attention_mask2"].to(CFG.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2
                )
                predictions = outputs["predictions"]
                fold_pred.extend(predictions)
        all_preds.append(fold_pred)
    # K-fold majority voting
    final_preds = []
    for i in range(len(test)):
        votes = [all_preds[fold][i] for fold in range(CFG.n_folds)]
        # majority vote per token
        # Pad all to maxlen
        maxlen = max(len(v) for v in votes)
        votes_pad = [v + [0]*(maxlen-len(v)) for v in votes]
        arr = np.array(votes_pad)
        maj = np.round(np.mean(arr, axis=0)).astype(int)
        final_preds.append(maj.tolist())
    subs=[]
    for i,row in enumerate(test.itertuples()):
        pred_tags = final_preds[i]
        enc = tok(row.feature_text,row.pn_history,
                  truncation="only_second", max_length=CFG.max_len,
                  return_offsets_mapping=True)
        seq_ids_test = enc.sequence_ids()
        offsets = enc["offset_mapping"]
        char_pred = np.zeros(len(row.pn_history), dtype=np.uint8)
        for t, tag in enumerate(pred_tags):
            if seq_ids_test[t]==1 and tag==1:
                s, e = offsets[t]
                if s < e and s < len(char_pred):
                    char_pred[s:min(e, len(char_pred))] = 1
        spans=[];start=None
        for idx,pv in enumerate(char_pred):
            if pv==1 and start is None:
                start=idx
            elif (pv==0 or idx==len(char_pred)-1) and start is not None:
                end=idx if pv==0 else idx+1
                spans.append(f"{start} {end}")
                start=None
        subs.append({"id":row.id, "location":";".join(spans)})

    sub_df = pd.DataFrame(subs)
    sub_df.to_csv("submission.csv", index=False)
    LOGGER.info("submission.csv saved!")


if __name__ == "__main__":
    main()