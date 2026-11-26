# Results of the chain pacing test, 26 Nov 2026

The test of [../experiments/PREDICTIONS-chain.md](../experiments/PREDICTIONS-chain.md) was run
once. Nothing in it is random, so it is four runs, one for each slowness of the hard chapter.
The predictions were committed before the code existed. Every number is from
`python experiments/chain.py`; the full output is in `experiments/results/chain-summary.txt`.

The chain has 6 chapters with one weight each, all starting at 0.1. Chapter `k` is right when
the product of weights 1 to `k` is 1, so each chapter uses all the ones before it.

## The verdicts

| | Prediction | Verdict | Measured |
|---|---|---|---|
| K1 | The last chapter cannot be learned first | **holds** | "last chapter only" ended at an error of 1.0000 in all four runs; mastery at 0.00001 |
| K2 | With no hard chapter, time-boxing is as good as mastery | **holds** | time-boxed ended lower than mastery |
| K3 | With a hard chapter, mastery beats time-boxing | **fails** | time-boxed ended lower than mastery in both runs (errors below 1e-26) |
| K4 | The chain forgets, even under mastery | **holds** | after the mastery pass, earlier chapters had errors up to 2.8 |

## What this says

- **The order matters enormously.** Given the same steps, going through the chapters in order
  solved chapter 6, and working on chapter 6 alone left it exactly where it started: its
  weights moved from 0.100 to at most 0.102. This is section 13 of `docs/proofs.md` seen in
  numbers. With every weight at 0.1, the signal for each one is the product of five others.
  In this model you cannot learn chapter 6 without the chapters before it.
- **The pace did not matter.** I guessed that a hard chapter, left half learned by its time
  box, would hold back everything after it. It did not. The later chapters contain the hard
  chapter and finished the job, and time-boxing ended with a lower error on the goal than
  mastery, in every run. So this test gives no support to mastery pacing over time-boxing.
  (Time-boxed ended lower only because it kept stepping after the error passed 0.0001, where
  mastery stops; both solved chapter 6.)
- **Mastery did not protect the earlier chapters.** After the pass, chapters 2 to 5 were far
  from right, with errors of 0.8 to 2.8, under both ways of pacing. Training a later chapter
  moves the earlier weights too: the weights ended spread from 1.69 down to 0.53, with the
  right product only for chapter 6. In the linear model with superset chapters, mastery in
  order forgot nothing (`docs/results-chapters-20261126.md`, S4). In the chain it forgets a
  lot.

## What this means for the idea

Of the two halves of the mastery-learning argument, this toy supports one and not the
other:

- **"You cannot do chapter 4 without chapter 3"** is true here, as a statement about the
  order of learning (K1).
- **"Stay on a chapter until it is mastered"** is not shown to help here (K3). What helped was
  having met the earlier chapters at all, in order. How long one stayed on each did not
  change the result.

In the words used before: the curriculum (the order) carried the result, and the pace did not.

## What this does not say

- **It is the smallest possible chain.** One weight per chapter, no data, no noise. A chapter
  was hard only in that it learned slowly.
- **The budget was generous.** Every pacing got as many steps as mastery needed. A test with
  too few steps to finish might separate the two. That was not in this plan; it was run
  afterwards as a second test, with its own predictions, and is reported at the end of this
  page.
- **The goal was the last chapter alone.** If every chapter has to be right at the end, both
  pacings failed here (K4), and neither was built to prevent that.
- **K3 was a guess**, and it was marked as one. Its failure is a result about this toy, not a
  proof that pacing never matters.

# A second test: too few steps to finish

The first test gave every pacing as many steps as mastery needed. This one gives less: 25%,
50% and 75% of them, with chapter 2 learning 64 times more slowly. Its predictions were
committed before its code, in
[../experiments/PREDICTIONS-chain-tight-budget.md](../experiments/PREDICTIONS-chain-tight-budget.md).
The output is in `experiments/results/chain-tight-budget-summary.txt`
(`python experiments/chain.py --tight`).

| | Prediction | Verdict | Measured |
|---|---|---|---|
| T1 | Mastery never leaves the hard chapter, and ends with an error of at least 0.5 on chapter 6 | **holds** | still on chapter 2 at all three budgets; error on chapter 6 of 0.9998 or more |
| T2 | (a guess) Time-boxed reaches the goal at 50% and 75% | **holds** | error on chapter 6 below 1e-31 at both |

At 25%, where no prediction was made, time-boxed also reached the goal.

**What this says.** With limited time, mastery pacing did not leave the chapter it could not
finish. It needed 1,631 steps to leave chapter 2 and never had them, so it never met chapters
3 to 6 and made no progress on the goal. A fixed time per chapter moved on, and the later
chapters made up for the hard one. Mastery learning objects to the fixed time per lesson; in this toy
the fixed time is what got the learner to the end.

**What it does not say.** It is the same smallest chain, with one weight per chapter. The
goal was chapter 6 alone: time-boxed got there with the earlier chapters left wrong (errors
of 0.4 to 2.7), which a teacher would call learning the last chapter by rote. Mastery left
chapters 1 and 2 nearly right and everything after untouched. Which of those is better
depends on what is asked for at the end, and this test asked only for chapter 6.
