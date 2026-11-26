# Predictions and stop rule, written before any run

Written on 26 Nov 2026. No run of the experiment had been made when this file was committed,
and the experiment script did not exist yet. The predictions come from the theorems in
`docs/proofs.md`, and they are not all in favour of the idea. What is run is described in
`docs/experiment-plan.md`.

## How a setting is scored

For each task, noise level, data mode and chew, the learning rate with the lowest median
validation loss at the end is used. A run whose loss stops being a number counts as the
worst. Chew A **beats** chew B when its test loss is lower in at least 4 of the 5 seeds. The
same rule is used for every comparison below.

## Predictions

- **P1, per gradient step.** With clean labels and new data (`stream`), no chew above 1 beats
  plain SGD at the end of the 10,000 steps. *Fails if* any of 4, 16, 64 or `until` beats 1,
  in `linear` or in `mlp`.
- **P2, per example.** With clean labels and new data, chew 64 beats plain SGD when both have
  seen the same number of examples (the number chew 64 has seen at the end of its run).
  *Fails if* it does not, in the `linear` task. For `mlp` the same is predicted with less
  confidence, and it is reported apart.
- **P3, noise.** With noisy labels and new data, plain SGD beats chew 64 and `until` at the
  end of the 10,000 steps. *Fails if* it does not beat both, in `linear` or in `mlp`.
- **P4, forgetting.** With clean labels, the mean rise in the loss on the previous batch is
  larger for chew 64 than for chew 1, in every task and data mode. *Fails if* it is smaller
  or equal in any of them.
- **P5, a fixed pool and the digits (a guess).** With a pool, plain SGD also returns to each
  example, so the theorems say less. The guess: P1 holds for the pool runs of all three
  tasks, and with noisy labels plain SGD beats chew 64 on `digits`. *Fails if* a chew above 1
  beats plain SGD in a clean pool run, or if plain SGD does not beat chew 64 on noisy digits.

## Stop rule

The 750 runs of the plan are all that is run. No run is made longer, and no seed is added,
after the results are seen. One change is allowed, and it is applied to every chew of that
task at once: if the best learning rate of plain SGD is at an end of its list, one more rate
is added on that side (three times larger or smaller).

A test of a few steps on the processor may be run first, to see that the code does not crash.
Its numbers are thrown away.

## Amendment of 26 Nov 2026, before any run of the experiment

The learning rates of `mlp` were 0.3, 0.1, 0.03. A check on the processor, 300 steps of chew 1
and chew 64 at each rate of each task, looked at one thing only: whether the loss stayed a
number. For `mlp` at 0.3 it did not, for both chews. Every other rate of every task stayed
finite. So the list for `mlp` is now 0.1, 0.03, 0.01. No loss from that check was looked at
or kept, and no prediction was changed.

## What would count as the idea working

P2 holding is the idea working in the sense the theorems support: fewer examples loaded for
the same loss. P1 failing would be a stronger result than the theorems predict. P3 holding is
the cost, and it says where not to use the method.
