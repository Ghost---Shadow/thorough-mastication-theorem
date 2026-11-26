# Pacing a chain of chapters: plan, predictions and stop rule, written before any run

Written on 26 Nov 2026. No code for this test existed when this file was committed.

## The question

Sal Khan's point, as the author put it: each lesson gets a fixed time, the class moves on
before a lesson is learned, and the error compounds, because the next chapter needs this one.
The theorems of `docs/proofs.md`, sections 11 to 13, say where compounding can come from: in a
chain, the learning signal for a chapter is the product of the earlier weights.

This test compares three ways to spend the same number of gradient steps on a chain.

## What is run

- **The chain:** 6 chapters, one weight each, all starting at 0.1. The teacher has every
  weight equal to 1. The output after chapter `k` is the product of weights 1 to `k`, and the
  error of chapter `k` is (that product − 1) squared. So chapter `k` uses chapters 1 to `k`.
- **A step on chapter `k`:** one gradient step on `hardness_k × (product − 1)²`, at learning
  rate 0.05. It moves weights 1 to `k`. A chapter with a small `hardness_k` learns slowly.
- **One hard chapter:** chapter 2 has `hardness = 1 / h`, with `h` = 1, 4, 16, 64. Every
  other chapter has hardness 1. `h = 1` means no chapter is hard.
- **Three ways to pace, with the same number of steps `B`:**
  - **Mastery:** stay on chapter 1 until its error is at most 0.0001, then chapter 2, and so
    on to chapter 6. `B` is the number of steps this takes.
  - **Time-boxed:** `B / 6` steps on each chapter, in order.
  - **Last chapter only:** all `B` steps on chapter 6.
- **Measured at the end:** the error of chapter 6, which is the goal, and the error of every
  chapter. Nothing is random, so there are no seeds: four runs, one for each `h`.

## Predictions

- **K1, the last chapter cannot be learned first.** For every `h`, "last chapter only" ends
  with an error of at least 0.5 on chapter 6, while mastery ends at 0.0001 or below. (With all
  weights at 0.1, the signal for each weight is the product of five others, about 0.00001.)
- **K2, with no hard chapter, time-boxing is as good as mastery.** For `h = 1`, the time-boxed
  error on chapter 6 is at most 10 times the mastery error.
- **K3, with a hard chapter, mastery beats time-boxing.** For `h = 16` and `h = 64`, the
  time-boxed error on chapter 6 is at least 10 times the mastery error. This one is a guess:
  later chapters contain chapter 2 and may repair it, as S6 hoped for the linear model.
- **K4, the chain forgets, even under mastery.** For every `h`, at the end of the mastery pass
  at least one earlier chapter has an error above 0.0001, the level it was mastered to.
  (Training a later chapter moves the earlier weights too.)

## What the predictions say about the question, if they hold

K1 is Khan's premise in this model: chapter 6 is out of reach without the chapters before it.
K2 and K3 together say when the pace matters: not when every chapter takes the same time,
only when some chapter needs more than its box. K4 says mastery is not a guarantee against
forgetting once a model builds on what it has learned.

## Stop rule

The four runs above and nothing more. If a run's weights stop being finite numbers, that is
reported as the result for that run. A first run with `h = 1` may be made to see that the code
does not crash; its numbers are thrown away.
