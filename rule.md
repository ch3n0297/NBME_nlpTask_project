## Evaluation
This competition is evaluated by a micro-averaged F1 score.

For each instance, we predict a set of character spans. A character span is a pair of indexes representing a range of characters within a text. A span i j represents the characters with indices i through j, inclusive of i and exclusive of j. In Python notation, a span i j is equivalent to a slice i:j.

For each instance there is a collection of ground-truth spans and a collection of predicted spans. The spans we delimit with a semicolon, like: 0 3; 5 9.

We score each character index as:

- TP if it is within both a ground-truth and a prediction,
- FN if it is within a ground-truth but not a prediction, and,
- FP if it is within a prediction but not a ground truth.

## Finally, we compute an overall F1 score from the TPs, FNs, and FPs aggregated across all instances.

**Example**</br>
Suppose we have an instance:

| ground-truth | prediction    |
|--------------|---------------|
| 0 3; 3 5     | 2 5; 7 9; 2 3 |
These spans give the sets of indices:

| ground-truth | prediction |
|--------------|------------|
| 0 1 2 3 4    | 2 3 4 7 8  |
We therefore compute:

`TP = size of {2, 3, 4} = 3`</br>
`FN = size of {0, 1} = 2`</br>
`FP = size of {7, 8} = 2`</br>

Repeat for all instances, collect the TPs, FNs, and FPs, and compute the final F1 score.

**Sample Submission**</br>
For each id in the test set, you must predict zero or more spans delimited by a semicolon. The file should contain a header and have the following format:
```
id,location
00016_000,0 100
00016_001,
00016_002,200 250;300 500
...
```
For 00016_000 you should give predictions for feature 000 in patient note 00016.