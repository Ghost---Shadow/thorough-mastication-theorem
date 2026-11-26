# Results of the chapters test, 26 Nov 2026

The test of [../experiments/PREDICTIONS-chapters.md](../experiments/PREDICTIONS-chapters.md) was
run once, on the processor, in 3 seconds. The predictions were committed before the code
existed. Every number is from `python experiments/chapters.py`; the full output is in
`experiments/results/chapters-summary.txt`. All mastications reached their target.

The question came from the author's reading of Sal Khan: errors compound, so without chapter 3
there is no chapter 4; and school chapters are supersets, chapter N holding the concepts of
chapters 1 to N. The test asks what that structure does in a model that is linear in its
weights. It does not test a model that learns its own features.

## The verdicts

| | Prediction | Verdict | Measured |
|---|---|---|---|
| S1 | A later chapter teaches all the earlier ones | holds, 5 of 5 | lowest entry 0.9999 |
| S2 | An earlier chapter teaches only part of a later one | holds, 5 of 5 | 0.23 to 0.85 |
| S3 | The table gives the order 1, 2, 3, 4 | holds, 5 of 5 | all six pairs, none the wrong way |
| S4 | Mastery in order forgets nothing | holds, 5 of 5 | at most 0.003% of a chapter's loss left |
| S5 | The squared travels add up to the squared distance to the teacher | holds, 5 of 5 | 99.6% to 99.8% |
| S6 | Time-boxing is forgiven: chapter 1 is better at the end than after its own steps | **fails**, 3 of 5 | better in 3 seeds, worse in 2 |
| S7 | Disjoint chapters: chapter 1 is exactly as its own steps left it | holds, 5 of 5 | no change at all |

The mean table for superset chapters (row = chapter masticated, column = chapter measured):

| | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **1** | | 0.50 | 0.41 | 0.29 |
| **2** | 1.00 | | 0.71 | 0.49 |
| **3** | 1.00 | 1.00 | | 0.74 |
| **4** | 1.00 | 1.00 | 1.00 | |

Below the diagonal everything is 1: a later chapter contains the earlier ones. Above it the
entries are close to what the theory says, chapter `k` teaching `k / j` of chapter `j`.

## What this says

- **Mastery in order has a guarantee, and time-boxing does not.** After mastering the chapters
  in order, every chapter was still learned (S4). After giving each chapter 5 steps and moving
  on, chapter 1 ended better in three seeds and worse in two (S6).
- **Why S6 failed, as far as I can tell.** I had argued that later chapters contain chapter
  1's concepts, so they keep teaching it. That left something out: a later chapter that is
  only half learned has large errors in its new material, and its steps push some of that
  error into the old material, because old and new concepts sit in the same examples. This
  explanation was made after seeing the result and is not tested.
- **Each chapter costs only what is new in it.** In the mastery pass the squared travels
  added up to the whole distance and no more (S5). Nothing was paid twice.
- **The table finds the curriculum.** From the entries alone, the rule fixed in advance put
  the four chapters in their order in every seed (S3).

## What this does not say

- **It does not show compounding.** In this setting the last chapter alone teaches everything
  (S1), so chapter 4 is not out of reach without chapter 3. Whether errors compound for a
  learner that builds on what it has learned is a question about a chain model, where the
  signal for a later stage passes through the earlier ones. That is not built or tested.
- **S6 is one setting.** Five steps per chapter, four chapters, one size. The split of 3 to 2
  says the effect can go either way here, not how often.
- **The model is linear and the labels are clean.**
