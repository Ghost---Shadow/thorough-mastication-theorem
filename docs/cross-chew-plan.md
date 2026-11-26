# Plan: cross-mastication, a table of what each batch teaches about the others

Written on 26 Nov 2026, before any code for it existed. The predictions and the stop rule are
in `experiments/PREDICTIONS-cross-chew.md`.

## The idea

Use a chew to measure a dataset, not to train on it. Start every chew from the same weights.
Chew batch `i` until it fits, then measure the loss on every other batch `j`, with no step
taken on `j`. The entry of the table is the share of batch `j`'s loss that the chew removed:

    entry(i, j) = 1 − (loss on j after chewing i) / (loss on j at the start)

It is like cross-validation with the roles turned round: fit on one part, test on all the
others. The theorems in `docs/proofs.md` say what the two ends mean, for a model that is
linear in its weights: an entry is 1 when batch `i` spans batch `j` (section 8), and 0 when the
two share no features (section 9).

A second reading comes from one pass over the data without going back to the start: the
**travel** of each chew, the distance the weights move. It is 0 for a batch that already fits
(section 7).

## The test: datasets where the answer is planted

The table is useful only if it finds structure that is known to be there. So the data are
built with the structure put in by hand.

- **Weights:** 288, in three groups of 96. An example of a group has random inputs in the 96
  coordinates of its group and zeros elsewhere. So two groups share no features.
- **Labels:** from a linear teacher, with no noise, except in the bad batch.
- **20 batches of 32 examples:**

| Batches | Group | What they are |
|---|---|---|
| 0 to 5 | 1 | six batches of new examples |
| 6 | 1 | an exact duplicate of batch 0, rows in another order |
| 7 | 1 | covered by batch 1: eight combinations of batch 1's examples, each four times |
| 8 to 12 | 2 | five batches of new examples |
| 13 | 2 | the bad batch: new examples with every label multiplied by −1 |
| 14 to 19 | 3 | six batches of new examples |

- **Models:** `linear` (288 weights, starting at zero, squared error), where the theorems
  apply; and `mlp` (288 inputs, one hidden layer of 256 `tanh` units, squared error), where
  they are only a guide.
- **The chew:** gradient descent on the whole batch from the common start, until the loss on
  the batch is at most one millionth of what it was (one ten-thousandth for `mlp`), or 20,000
  steps. For `linear` the learning rate is one over the largest strength of the batch's own
  kernel, which is always stable. For `mlp` it is the largest of 0.1, 0.03, 0.01, 0.003 at
  which that chew does not diverge. (First written as "the chew of batch 0"; changed before
  any run, with the scaling of batch 7's combinations by the root of 32: see the amendment in
  the predictions file.) Whether each chew reached its target is
  recorded.
- **The pass for travel:** from the common start, chew the batches in the order 0, 6, 1, 7,
  2, 3, 4, 5, 8, 9, ..., 19, without going back. The duplicate comes right after its
  original, and the covered batch right after the batch that covers it.
- **Seeds:** 0 to 4, each with its own data, teacher and starting weights.

## What the theory says the table should look like (`linear`)

- **Duplicates (0 and 6):** 1 both ways.
- **Covered (1 covers 7):** 1 from batch 1 to batch 7; much less the other way, because batch
  7 holds only 8 of the 32 directions of batch 1.
- **Different groups:** exactly 0.
- **Same group, different examples:** about 1/3 on average, because a batch of 32 spans a
  third of the 96 directions of its group. A numeric check before the plan gave a mean of
  0.319, with single entries from −0.47 to 0.64. So single entries are noisy, and only
  averages are predicted.
- **The bad batch:** chewing it teaches the opposite of the teacher, so its entries on its
  group mates average about −1.

## Cost

Everything is small: 20 chews and 400 measurements per model and seed. It runs on the
processor in a few minutes and does not need the graphics card.

## What this does not test

Real data, large models, and the dependence on where in training the common start is taken.
The table shows relatedness as this model sees it, not a property of the data alone.
