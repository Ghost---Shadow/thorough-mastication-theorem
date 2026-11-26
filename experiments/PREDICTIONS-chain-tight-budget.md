# Pacing a chain with too few steps: predictions and stop rule, written before any run

Written on 26 Nov 2026, after the first chain test (`docs/results-chain-20261126.md`) and
before any code or run for this one. That test gave every pacing as many steps as mastery
needed, and its write-up said a tighter budget was not tested. This is that test. It is a new
test with its own predictions; the first one is closed and unchanged.

## What is run

The same chain as before: 6 chapters, weights starting at 0.1, learning rate 0.05, chapter 2
learning 64 times more slowly. Mastery needed 1,677 steps there, 1,631 of them before it
could leave chapter 2.

- **Budgets:** 25%, 50% and 75% of those 1,677 steps: 419, 838 and 1,257.
- **Mastery:** stay on each chapter until its error is at most 0.0001, as before, and stop
  wherever the budget runs out.
- **Time-boxed:** a sixth of the budget on each chapter, in order.
- **Measured:** the error of chapter 6 at the end, and the chapter mastery was on when its
  budget ran out.

Nothing is random: six runs.

## Predictions

- **T1, mastery never leaves the hard chapter.** At all three budgets, mastery is still on
  chapter 2 when the steps run out, and its error on chapter 6 is at least 0.5. (It needs
  1,631 steps to leave chapter 2, and the largest budget is 1,257.)
- **T2, time-boxing reaches the goal (a guess).** At 50% and at 75%, time-boxed ends with an
  error on chapter 6 of at most 0.01. The reasoning: the later chapters contain chapter 2 and
  can make up for it with their own weights, as they did in the first test. What could go
  wrong: making up for a small weight needs large ones, and the steps may become unstable.
- **At 25%** no prediction is made; the result is reported.

## What the predictions say, if they hold

Mastery pacing has a failure of its own: with limited time it does not leave the chapter it
cannot finish, and so never meets the chapters that could have helped. A fixed time per
chapter is what Sal Khan objects to, and here it would be what saves the learner. That would
be a point against the idea as first stated, in this toy.

## Stop rule

The six runs above and nothing more. If a run's weights stop being finite numbers, that is
reported as the result of that run.
