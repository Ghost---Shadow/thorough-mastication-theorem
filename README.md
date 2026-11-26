# The Thorough Mastication Theorem

*Chew each batch before you swallow it.*

**Status, 26 Nov 2026.** This is an idea and an argument made by hand and checked with
numbers. There are no experiments yet. Nothing here is proved in Lean yet, so
"theorem" in the name is a hope, not a claim. The marks used below: **proved** (in the
squiggly line library), **checked by hand** (derived here, checked with numbers by
`scripts/check_claims.py`), **my reading**, **recalled** (from memory, not checked against the
source).

## The idea

Plain SGD takes a batch, makes one step, and draws the next batch.

TMT-SGD takes a batch and keeps stepping on it until the model has learned it. Then it draws
the next batch. One such run of steps on one batch is called a *chew*.

```
for batch in data:
    repeat:
        one gradient step on batch
    until the loss on batch is small, or a cap on steps is reached
```

With one step per chew this is plain SGD. So the question is how many steps a batch should
get.

## Why it might help: the author's two reasons

1. **Locality of reference.** The batch stays in video memory. If the model fits there too,
   everything stays on the card until the chew is done, and the bus between the card and the
   rest of the machine is idle.
2. **A naive curriculum.** If you cannot do page 8 of a book, you cannot do the whole book.
   Repeat page 8 before you move to page 9.

**The risk** is forgetting: a chew on page 9 can undo page 8. The goal is to find out whether
the risk is worth taking. Plain SGD forgets too, one step at a time.

## What the squiggly line library says

The [squiggly line theorem](https://github.com/Ghost---Shadow/squiggly-line-theorem) is a
Lean library of 604 theorems about training. Every one of them is listed in
[docs/theorem-table.md](docs/theorem-table.md) with a verdict: does it help here, and why.
The first columns of that table are generated, so new theorems show up as "not judged".

The main theorem itself does not help: it says which target a model starts closer to, not in
which order to take steps. The lemmas about the kernel do help. They hold for a model that is
linear in its weights, with squared loss. A real network is not linear in its weights, so for
a network they are a guide, not a guarantee.

**Proved in the library:**

- **A chew is a projection.** If gradient descent on a batch arrives at a fit of that batch,
  no other fit of the batch is nearer to where the chew started
  (`gradient_descent_fit_is_the_nearest_fit`). The chew moves only along the features of its
  batch (`gradient_descent_stays_in_the_directions_of_the_features`).
- **A ceiling on forgetting.** A chew changes the errors on an earlier batch by at most the
  size of that batch's predictor times the travel of the chew
  (`forgetting_is_at_most_predict_size_times_travel`). If the two batches share no features,
  the earlier batch is left exactly as it was (`separate_features_forget_nothing`).
- **Order matters only through the kernel between batches.** Two steps in a row differ from
  one step on both batches by a term that passes through the kernel between them
  (`two_small_steps_against_one_big_step`). Chewing changes only the order of steps.
- **Returns from a chew fall geometrically.** Each further step on the same batch shrinks the
  error along a direction by a fixed factor (`one_step_shrinks_one_direction`), strong
  directions first (`strong_directions_first`). Weak directions take very long, which is why
  "until convergence" needs a cap.

**Checked by hand, not yet in Lean.** Suppose some weights fit every batch (more weights
than data, no label noise). Call any such weights a *full fit*.

- **C1. A chew never moves away from a full fit.** After a chew, the squared distance to a
  full fit has fallen by exactly the squared travel of the chew. (Pythagoras: the travel is
  along the batch's features, and the rest of the way to the full fit is across them.)
- **C2. No use of one batch does better.** Among all weights that can be reached from the
  start using only this batch's gradients, the chewed weights are the nearest to every full
  fit.
- **C3. The travel runs out.** By C1, the squared travels of all chews add up to at most the
  squared distance from the start to a full fit. So the chews get shorter, and by the ceiling
  above so does the forgetting.
- **C4. With noise, a chew is only as good as its batch.** Take the simplest model, one
  number fitted to noisy measurements. A chew sets the number to the mean of its batch, so
  after any number of batches the error is that of one batch. Plain SGD with a falling
  learning rate keeps the running mean, and its error falls with every batch.

**My reading.**

- Chewing is the best use of a batch **per batch loaded** (C2). No statement says it is
  better **per gradient step**, and the geometric fall says a step on a fresh batch should
  gain more than one more step on a chewed batch. So I expect chewing to tie or lose per
  step and to win per example. It pays when loading a batch costs more than a step. That is
  reason 1, with its condition.
- Reason 2 holds in the form of C1 when a full fit exists. It fails in the form of C4 when
  the labels are noisy and the model is small.
- Forgetting is real but bounded, and it shrinks as the chews get shorter (C3).

## This is not new for linear models

For a model that is linear in its weights, chewing every batch to convergence is the block
Kaczmarz method, also known as projections onto convex sets (Kaczmarz, 1937; **recalled**).
Learning tasks one after another by such projections has been studied as a model of
catastrophic forgetting (Evron and others, 2022; **recalled**). Several steps on one batch
also appear in the local steps of federated averaging and in the epochs of PPO (**recalled**).
What this repository adds is the reading through the squiggly line library, the table, and an
honest test on small models.

## Experiments

**None yet, and no experiment code yet.** The theory comes first. What the argument above
predicts, so that a test can prove it wrong:

- **Per gradient step**, with clean labels and fresh data, chewing does not beat plain SGD.
- **Per example loaded**, with clean labels, chewing beats plain SGD.
- **With noisy labels**, plain SGD beats long chews.
- **Forgetting** of the previous batch grows with the length of the chew.

The exact runs, the criterion and the stop rule will be written and committed before the
first run. The hardware is one 8 GB graphics card, so the tests will be toy models on toy
data. Whether any of this carries over to large models is not known.

## Looking for funding

This work has no funding and one modest graphics card. If you can fund compute for tests on
real models, or want to run them yourself, please open an issue.

## arXiv endorsement

The author is unaffiliated and has no endorsement for arXiv cs.LG. If you are able to
endorse and think this is worth posting, an endorsement would be a great help. Please open an
issue.

## Thanks

The idea comes from Sal Khan of Khan Academy. In one of his videos he says (from memory, not
a quote) that the biggest problem with schooling is that each lesson gets a fixed time, so a
class moves on before every doubt and mistake is cleared. Plain SGD makes the same mistake:
one step per batch, then on to the next. TMT-SGD waits.

## Citation

```bibtex
@misc{nanda2026mastication,
  author       = {Nanda, Souradeep},
  title        = {The Thorough Mastication Theorem: training on one batch until it is learned},
  year         = {2026},
  howpublished = {\url{https://github.com/Ghost---Shadow/thorough-mastication-theorem}},
  note         = {Unaffiliated}
}
```
