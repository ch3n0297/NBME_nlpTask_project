# Codebase: /home/hjc/codeSpace/NLP_Final
"""
final_EDA.py

此腳本會對以下資料集進行探索性資料分析 (EDA)：
  - train.csv: 包含每筆病人紀錄及標註的 span 位置
  - test.csv: 待預測的測試資料
  - sample_submission.csv: 提交結果範本
  - patient_notes.csv: 病歷文字說明
  - features.csv: 特徵編號與名稱對應表

主要功能：
1. 資料載入與基本摘要：自動讀取 CSV 檔並列印資料維度、欄位型態、缺失值統計
2. Span 分析：解析 train.csv 中 location 欄位，計算每筆紀錄的 span 數量與長度，並繪製對應直方圖
3. 病歷文字分析：統計 patient_notes.csv 中文本的字元數與詞數分布，並繪出直方圖
4. 特徵分佈：檢視 features.csv 中各 feature_name 的出現次數，並繪製長條圖
5. 合併後分析：將 train、patient_notes、features 三者合併，計算不同 feature 的平均 span 長度，繪製長條圖
6. 進階探勘：包含唯一鍵與紀錄數統計、文字標記詞頻統計、數值關聯係數矩陣、train/test 差異比較、缺失值百分比分析、關係散佈圖與散佈矩陣
7. 結果輸出：所有圖表均輸出至 eda_results/ 資料夾，並在終端印出摘要統計資訊

使用方式：
將本檔與所有 CSV 檔置於同一目錄，執行：
    python final_EDA.py
即可自動生成完整的 EDA 圖表與統計摘要，作為後續資料前處理與模型開發的參考依據。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pandas.plotting import scatter_matrix
import re
import sys

class Logger:
    """
    Logger to write messages both to stdout and to a log file.
    """
    def __init__(self, stdout, log_file):
        self.stdout = stdout
        self.log_file = log_file

    def write(self, message):
        self.stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

os.chdir("NLP_Final")
OUTPUT_DIR = "eda_results"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_data():
    """Load all CSV files into pandas DataFrames."""
    train = pd.read_csv("nbme_data/train.csv")
    test = pd.read_csv("nbme_data/test.csv")
    submission = pd.read_csv("nbme_data/sample_submission.csv")
    patient = pd.read_csv("nbme_data/patient_notes.csv")
    features = pd.read_csv("nbme_data/features.csv")
    return train, test, submission, patient, features

def basic_summary(df: pd.DataFrame, name: str):
    """Print basic shape, dtypes, and missing value summary."""
    print(f"===== {name} =====")
    print(f"Shape: {df.shape}")
    print("Data types:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isna().sum())
    print()

def parse_spans(location_str: str):
    """
    Robustly parse the 'location' field into spans.
    Accept formats like "start end; start end" or "start,end;start,end".
    Returns list of (start, end) tuples, skipping invalid entries.
    """
    if pd.isna(location_str):
        return []
    loc = str(location_str).strip()
    if not loc:
        return []
    spans = []
    # split on semicolons
    for part in re.split(r'[;]', loc):
        part = part.strip()
        if not part:
            continue
        # replace commas with spaces for uniform splitting
        part = part.replace(',', ' ')
        tokens = part.split()
        if len(tokens) != 2:
            continue
        try:
            start, end = int(tokens[0]), int(tokens[1])
        except ValueError:
            continue
        spans.append((start, end))
    return spans

def analyze_spans(train: pd.DataFrame):
    """Analyze span counts and lengths from train data."""
    train = train.copy()
    train["spans"] = train["location"].apply(parse_spans)
    train["n_spans"] = train["spans"].apply(len)
    train["span_lengths"] = train["spans"].apply(lambda lst: [e - s for s, e in lst] if lst else [])
    train["mean_span_len"] = train["span_lengths"].apply(lambda L: np.mean(L) if L else 0)

    # Histogram: number of spans per record
    plt.figure()
    plt.hist(train["n_spans"], bins=range(train["n_spans"].max() + 2), edgecolor="black")
    plt.xlabel("Number of Spans")
    plt.ylabel("Count")
    plt.title("Distribution of Span Counts")
    plt.savefig(f"{OUTPUT_DIR}/span_counts.png")
    plt.close()

    # Histogram: all span lengths
    all_lengths = np.concatenate(train["span_lengths"].values) if train["span_lengths"].any() else []
    plt.figure()
    plt.hist(all_lengths, bins=50, edgecolor="black")
    plt.xlabel("Span Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Span Lengths")
    plt.savefig(f"{OUTPUT_DIR}/span_lengths.png")
    plt.close()

    print(f"Average spans per record: {train['n_spans'].mean():.2f}")
    print(f"Average span length: {np.mean(all_lengths) if len(all_lengths)>0 else 0:.2f}")
    print()

    return train

def analyze_patient_notes(patient: pd.DataFrame):
    """Compute and plot character and word count distributions for patient notes."""
    patient = patient.copy()
    patient["char_count"] = patient["pn_history"].astype(str).apply(len)
    patient["word_count"] = patient["pn_history"].astype(str).apply(lambda txt: len(txt.split()))

    # Character count distribution
    plt.figure()
    plt.hist(patient["char_count"], bins=50, edgecolor="black")
    plt.xlabel("Character Count")
    plt.ylabel("Number of Records")
    plt.title("Patient Notes Character Count")
    plt.savefig(f"{OUTPUT_DIR}/patient_char_counts.png")
    plt.close()

    # Word count distribution
    plt.figure()
    plt.hist(patient["word_count"], bins=50, edgecolor="black")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Records")
    plt.title("Patient Notes Word Count")
    plt.savefig(f"{OUTPUT_DIR}/patient_word_counts.png")
    plt.close()

    print(f"Average note length: {patient['char_count'].mean():.1f} chars, {patient['word_count'].mean():.1f} words")
    print()

    return patient

def analyze_features(features: pd.DataFrame):
    """Plot distribution of feature names."""
    counts = features["feature_text"].value_counts()
    plt.figure(figsize=(8, 4))
    counts.plot.bar()
    plt.xlabel("Feature Name")
    plt.ylabel("Count")
    plt.title("Feature Name Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_distribution.png")
    plt.close()

    print("Top 10 feature counts:")
    print(counts.head(10))
    print()

    return counts

def merge_analysis(train: pd.DataFrame, patient: pd.DataFrame, features: pd.DataFrame):
    """
    Merge train, patient_notes, and features on pn_num and feature_num,
    then analyze mean span length by feature.
    """
    df = train.merge(patient, on="pn_num", how="left") \
              .merge(features, on="feature_num", how="left")
    summary = df.groupby("feature_text")["mean_span_len"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    summary.plot.bar()
    plt.xlabel("Feature Name")
    plt.ylabel("Mean Span Length")
    plt.title("Mean Span Length per Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/span_length_by_feature.png")
    plt.close()

    print("Mean span length by feature (top 10):")
    print(summary.head(10))
    print()


# ========== New Functions ==========
def analyze_unique_counts(train, test, submission, patient, features):
    """Print and plot counts of unique keys across datasets."""
    unique_info = {
        'train_pn_num': train['pn_num'].nunique(),
        'train_feature_num': train['feature_num'].nunique(),
        'train_records': len(train),
        'test_records': len(test),
        'submission_records': len(submission),
        'patient_notes': patient['pn_num'].nunique(),
        'features': features['feature_num'].nunique()
    }
    print("Unique counts across datasets:")
    for k, v in unique_info.items():
        print(f"  {k}: {v}")
    # bar chart
    plt.figure(figsize=(6,4))
    plt.bar(list(unique_info.keys()), list(unique_info.values()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Unique Key and Record Counts")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/unique_counts.png")
    plt.close()

def analyze_text_tokens(patient):
    """Token frequency analysis on patient notes."""
    texts = patient['pn_history'].astype(str).str.lower().str.split()
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)
    top_tokens = counter.most_common(20)
    words, freqs = zip(*top_tokens)
    plt.figure(figsize=(8,4))
    plt.bar(words, freqs)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 20 Tokens in Patient Notes")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/text_token_frequency.png")
    plt.close()
    print("Top 20 tokens in patient notes:")
    for word, freq in top_tokens:
        print(f"  {word}: {freq}")
    print()

def analyze_correlations(train, patient):
    """Compute correlation matrix for numeric EDA features."""
    # ensure spans and counts exist
    df_corr = pd.DataFrame({
        'n_spans': train.get('n_spans', []),
        'mean_span_len': train.get('mean_span_len', []),
        'char_count': patient.get('char_count', []),
        'word_count': patient.get('word_count', [])
    }).dropna()
    corr = df_corr.corr()
    plt.figure(figsize=(5,4))
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    plt.close()
    print("Correlation matrix:")
    print(corr)
    print()

def analyze_train_test_overlap(train, test):
    """Compare feature_num overlap between train and test."""
    train_set = set(train['feature_num'].unique())
    test_set = set(test['feature_num'].unique())
    only_train = train_set - test_set
    only_test = test_set - train_set
    print(f"Features only in train: {len(only_train)}")
    print(f"Features only in test: {len(only_test)}")
    print()

def analyze_missing(df: pd.DataFrame, name: str):
    """Plot percentage of missing values per column."""
    missing_pct = df.isna().mean() * 100
    plt.figure(figsize=(6,4))
    missing_pct.sort_values(ascending=False).plot.bar()
    plt.ylabel("Missing %")
    plt.title(f"Missing Values in {name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/missing_{name}.png")
    plt.close()
    print(f"Missing percentages for {name}:")
    print(missing_pct[missing_pct > 0])
    print()

def analyze_word_span_scatter(train, patient):
    """Scatter plot of patient word count vs. mean span length."""
    df = pd.merge(train, patient[['pn_num', 'word_count']], on='pn_num', how='inner')
    plt.figure(figsize=(6,6))
    plt.scatter(df['word_count'], df['mean_span_len'], alpha=0.5)
    plt.xlabel("Patient Note Word Count")
    plt.ylabel("Mean Span Length")
    plt.title("Word Count vs. Mean Span Length")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/word_vs_span.png")
    plt.close()
    corr = df['word_count'].corr(df['mean_span_len'])
    print(f"Correlation between word_count and mean_span_len: {corr:.3f}")
    print()

def analyze_scatter_matrix(train, patient):
    """Scatter matrix for numeric EDA features."""
    df = pd.DataFrame({
        'n_spans': train['n_spans'],
        'mean_span_len': train['mean_span_len'],
        'char_count': patient['char_count'],
        'word_count': patient['word_count']
    }).dropna()
    plt.figure(figsize=(8,8))
    scatter_matrix(df, alpha=0.2, diagonal='kde')
    plt.suptitle("Scatter Matrix of Core Metrics")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_matrix.png")
    plt.close()

def main():
    ensure_output_dir()
    # Redirect stdout to both console and log file
    log_path = os.path.join(OUTPUT_DIR, "eda_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Logger(sys.stdout, log_file)
    train, test, submission, patient, features = load_data()

    basic_summary(train, "train.csv")
    basic_summary(test, "test.csv")
    basic_summary(submission, "sample_submission.csv")
    basic_summary(patient, "patient_notes.csv")
    basic_summary(features, "features.csv")

    train = analyze_spans(train)
    patient = analyze_patient_notes(patient)
    _ = analyze_features(features)
    merge_analysis(train, patient, features)
    analyze_unique_counts(train, test, submission, patient, features)
    analyze_text_tokens(patient)
    analyze_correlations(train, patient)
    analyze_train_test_overlap(train, test)

    # Missing value analysis for each dataset
    analyze_missing(train, "train")
    analyze_missing(test, "test")
    analyze_missing(patient, "patient_notes")
    analyze_missing(features, "features")

    # Relationship scatter and matrix
    analyze_word_span_scatter(train, patient)
    analyze_scatter_matrix(train, patient)

if __name__ == "__main__":
    main()