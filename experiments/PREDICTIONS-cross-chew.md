# Cross-mastication: predictions and stop rule, written before any run

Written on 26 Nov 2026. No code for this test existed when this file was committed. What is
run is in `docs/cross-chew-plan.md`. The batch numbers below are the ones of that plan.

## How a prediction is scored

Each prediction is checked in each of the 5 seeds. It **holds** when it is true in at least 4
of them. `entry(i, j)` is the share of batch `j`'s loss removed by chewing batch `i`.

## Predictions for `linear`, where the theorems apply

- **Q1, duplicates.** `entry(0, 6)` and `entry(6, 0)` are both at least 0.99.
- **Q2, covered.** `entry(1, 7)` is at least 0.99, and `entry(7, 1)` is at most 0.9.
- **Q3, separate groups.** Every entry between batches of different groups is at most 0.001
  in size.
- **Q4, same group.** Over the pairs of different batches in one group, leaving out batches
  6, 7 and 13, the mean entry is between 0.2 and 0.45. (The theory says 1/3.)
- **Q5, the bad batch.** The mean of `entry(13, j)` over its group mates `j` = 8 to 12 is at
  most −0.5. And among all 20 batches, batch 13 has the lowest mean entry on its own group.
- **Q6, travel.** In the pass for travel, the travel of batch 6 is at most 1% of the travel
  of batch 0, and the travel of batch 7 is at most 1% of the travel of batch 1.
- **Q7, a new group arrives untouched.** In the same pass, the loss on batch 8 just before
  its chew is within 0.1% of its loss at the common start, and the same for batch 14.
- **Q8, the structure is recovered by simple rules.**
  - Groups: join two batches when either entry between them is above 0.01 in size. The joined
    sets are exactly the three planted groups.
  - Duplicates: the pairs with both entries at least 0.99 are exactly {0, 6}.
  - Covering: the pairs with one entry at least 0.99 and the other at most 0.9 are exactly
    "1 covers 7".

## Predictions for `mlp`, a guess, reported apart

- **M1.** `entry(0, 6)` and `entry(6, 0)` are both at least 0.9.
- **M2.** The mean size of the entries between different groups is at most half of the mean
  entry inside a group (clean, different batches).
- **M3.** Batch 13 has the lowest mean entry on its own group of all 20 batches.

## Stop rule

One run of the plan: two models, five seeds. Nothing is changed and no seed is added after the
results are seen. If a chew does not reach its target within 20,000 steps, that is reported
with the result and the entry is used as it is.

A test of a few steps may be run first, to see that the code does not crash. Its numbers are
thrown away.

## What would count as the probe working

Q1 to Q3 are the theorems of `docs/proofs.md` seen in numbers; if they fail, the code is
wrong. Q5 and Q8 are the test of usefulness: the table finds the bad batch and the planted
structure with rules fixed in advance. M1 to M3 say whether any of it survives in a network.
