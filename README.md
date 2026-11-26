# The Thorough Mastication Theorem

*Masticate each batch before you swallow it.*

**Status, 26 Nov 2026.** The idea, nineteen theorems about it proved in Lean for a model that
is linear in its weights, and four tests on toy models. Whether it helps a large network is not
known. The marks used below: **proved** (in Lean, here or in the squiggly line library),
**checked with numbers** (by `scripts/check_claims.py`, not proved), **my reading**,
**recalled** (from memory, not checked against the source).

## The idea

Plain SGD takes a batch, makes one step, and draws the next batch.

TMT-SGD takes a batch and keeps stepping on it until the model has learned it. Then it draws
the next batch. One such run of steps on one batch is called a *mastication*.

```
for batch in data:
    repeat:
        one gradient step on batch
    until the loss on batch is small, or a cap on steps is reached
```

With one step per mastication this is plain SGD. So the question is how many steps a batch should
get.

## Why it might help: the author's two reasons

1. **Locality of reference.** The batch stays in video memory. If the model fits there too,
   everything stays on the card until the mastication is done, and the bus between the card and the
   rest of the machine is idle.
2. **A naive curriculum.** If you cannot do page 8 of a book, you cannot do the whole book.
   Repeat page 8 before you move to page 9.

**The risk** is forgetting: a mastication on page 9 can undo page 8. The goal is to find out whether
the risk is worth taking. Plain SGD forgets too, one step at a time.

## What the squiggly line library says

The [squiggly line theorem](https://github.com/Ghost---Shadow/squiggly-line-theorem) is a
Lean library of 604 theorems about training. Every one of them is listed in
[docs/theorem-table.md](docs/theorem-table.md) with a verdict: does it help here, and why.
The first columns of that table are generated, so new theorems show up as "not judged".
How to bring it up to date is in
[docs/updating-the-theorem-table.md](docs/updating-the-theorem-table.md).

The main theorem itself does not help: it says which target a model starts closer to, not in
which order to take steps. The lemmas about the kernel do help. They hold for a model that is
linear in its weights, with squared loss. A real network is not linear in its weights, so for
a network they are a guide, not a guarantee.

**Proved in the library:**

- **A mastication is a projection.** If gradient descent on a batch arrives at a fit of that batch,
  no other fit of the batch is nearer to where the mastication started
  (`gradient_descent_fit_is_the_nearest_fit`). The mastication moves only along the features of its
  batch (`gradient_descent_stays_in_the_directions_of_the_features`).
- **A ceiling on forgetting.** A mastication changes the errors on an earlier batch by at most the
  size of that batch's predictor times the travel of the mastication
  (`forgetting_is_at_most_predict_size_times_travel`). If the two batches share no features,
  the earlier batch is left exactly as it was (`separate_features_forget_nothing`).
- **Order matters only through the kernel between batches.** Two steps in a row differ from
  one step on both batches by a term that passes through the kernel between them
  (`two_small_steps_against_one_big_step`). Masticating changes only the order of steps.
- **Returns from a mastication fall geometrically.** Each further step on the same batch shrinks the
  error along a direction by a fixed factor (`one_step_shrinks_one_direction`), strong
  directions first (`strong_directions_first`). Weak directions take very long, which is why
  "until convergence" needs a cap.

**Proved here**, in `lean/ThoroughMastication/Mastication.lean`. The plain English of each proof is
in [docs/proofs.md](docs/proofs.md). Suppose some weights fit every batch (more weights than
data, no label noise). Call any such weights a *full fit*.

- **C1. A mastication never moves away from a full fit.** After a mastication, the squared distance to a
  full fit has fallen by exactly the squared travel of the mastication
  (`mastication_brings_weights_closer_by_exactly_its_travel`). This is Pythagoras: the travel is
  along the batch's features, and the rest of the way to the full fit is across them.
- **C2. No use of one batch does better.** Among all weights that can be reached from the
  start by moving along this batch's features, the masticated weights are the nearest to every
  full fit (`no_use_of_the_batch_gets_nearer_than_the_mastication`). One gradient step and a mastication
  stopped early are such moves.
- **C3. The travel runs out.** The squared travels of all mastications add up to at most the squared
  distance from the start to a full fit
  (`total_squared_travel_is_at_most_the_first_squared_distance`). So does the forgetting of
  an old batch, up to the size of its predictor (`total_squared_forgetting_is_bounded`).
- **C4. With noise, a mastication is only as good as its batch.** Take the simplest model, one
  number fitted to noisy labels. The best fit to a batch is the batch mean, whatever the
  start was (`batch_mean_is_the_best_single_number`), so a mastication keeps nothing of earlier
  batches. **Checked with numbers**, not proved: its error stays at the noise of one batch,
  while plain SGD with a falling learning rate keeps the running mean and improves with
  every batch.

One thing is assumed and not proved, as in the library: that gradient descent on a batch
arrives at a fit. In general it only comes as close as you like.

To check the Lean file, from a built checkout of the library:

```bash
cd squiggly-line-theorem/lean
lake env lean -DautoImplicit=false ../../thorough-mastication-theorem/lean/ThoroughMastication/Mastication.lean
```

This is how it was checked on 26 Nov 2026: no errors, no warnings, and every theorem rests
only on Lean's three standard axioms. `lean/lakefile.toml` also sets this folder up as its own
Lake project that fetches the library by itself, so a clone of this repository alone is meant
to be enough (`lake exe cache get`, then `lake build`, in `lean/`). That build has not been run
yet. `.github/workflows/ci.yml` runs it, and it is switched off while the repository is
private.

**My reading.**

- Masticating is the best use of a batch **per batch loaded** (C2). No statement says it is
  better **per gradient step**, and the geometric fall says a step on a fresh batch should
  gain more than one more step on a masticated batch. So I expect masticating to tie or lose per
  step and to win per example. It pays when loading a batch costs more than a step. That is
  reason 1, with its condition.
- Reason 2 holds in the form of C1 when a full fit exists. It fails in the form of C4 when
  the labels are noisy and the model is small.
- Forgetting is real but bounded, and it shrinks as the mastications get shorter (C3).

## A second use: a mastication as a probe of a dataset

The toy runs say masticating is a poor way to train. The theorems say it is an exact way to read
a batch: by C2 a mastication takes everything the batch has to give, so what is still wrong
afterwards is what the batch did not contain. Six more theorems make that precise
(`docs/proofs.md`, sections 7 to 10): a duplicate needs no travel, a batch that is spanned is
learned for free, a batch with separate features is left alone, and in between the error is
bounded by how far an example sticks out.

**Cross-mastication** uses this. Masticate each batch from the same start and measure the loss on
every other batch. The table of results is meant to show duplicates, groups, batches that
cover others, and batches with wrong labels. The plan, with datasets where the answer is
planted, is in [docs/cross-mastication-plan.md](docs/cross-mastication-plan.md), and its predictions are in
[experiments/PREDICTIONS-cross-mastication.md](experiments/PREDICTIONS-cross-mastication.md). It was run on 26 Nov 2026 and all
eleven predictions held in every seed; the results and their limits are in
[docs/results-cross-mastication-20261126.md](docs/results-cross-mastication-20261126.md).

**Chapters that are supersets.** The author's point from Sal Khan is that errors compound:
without chapter 3 there is no chapter 4, and chapter N holds the concepts of chapters 1 to N.
A third test asked what that structure does in a model with fixed features
([docs/results-chapters-20261126.md](docs/results-chapters-20261126.md)). Mastering the
chapters in order forgot nothing, and each chapter cost only its new material. Moving on
after a fixed number of steps had no such guarantee: the later chapters made chapter 1
better in 3 seeds and worse in 2, against my prediction. The last chapter alone taught all
the earlier ones, so nothing compounds in this setting. Compounding needs a model that builds
on what it has learned.

**A chain of chapters.** In a chain, each chapter is built on the ones before it, and the
learning signal for a chapter is the product of the earlier weights (three more theorems,
`docs/proofs.md`, sections 11 to 13). A fourth test compared ways to pace such a chain
([docs/results-chain-20261126.md](docs/results-chain-20261126.md)). The order mattered
enormously: with the same steps, going through the chapters in order solved the last
chapter, and working on the last chapter alone did not move it. The pace did not matter:
moving on after a fixed number of steps did as well as staying until each chapter was
mastered, against my guess. So this toy supports "no chapter 4 without chapter 3" and gives
no support to mastery pacing.

## This is not new for linear models

For a model that is linear in its weights, masticating every batch to convergence is the block
Kaczmarz method, also known as projections onto convex sets (Kaczmarz, 1937; **recalled**).
Learning tasks one after another by such projections has been studied as a model of
catastrophic forgetting (Evron and others, 2022; **recalled**). Several steps on one batch
also appear in the local steps of federated averaging and in the epochs of PPO (**recalled**).
What this repository adds is the reading through the squiggly line library, the table, and an
honest test on small models.

## Experiments

**Run on 26 Nov 2026: 1,150 toy runs.** The results are in
[docs/results-20261126.md](docs/results-20261126.md). What the argument above predicted, and
what happened:

- **Per gradient step**, with clean labels and fresh data, masticating does not beat plain SGD.
  *Held:* plain SGD was lower in every seed, and by a wide margin.
- **Per example loaded**, with clean labels, masticating beats plain SGD. *Held:* mastication 64 was
  lower in 5 of 5 seeds, for 64 times the gradient steps.
- **With noisy labels**, plain SGD beats long mastications. *Held* in 5 of 5 seeds.
- **Forgetting** of the previous batch grows with the length of the mastication. *Held.*
- A guess that no mastication helps on a fixed pool of data *failed*: on the clean digits, a mastication of
  4 steps ended with a slightly lower test loss than plain SGD in 4 of 5 seeds.

So on these toys the method is the better use of each batch and the worse use of each
gradient step. The stop rule's extra learning rates were run as well (400 of the 1,150 runs), and no
verdict changed.

The models, the data and the training loop are in
[docs/experiment-plan.md](docs/experiment-plan.md). The exact predictions, the criterion and
the stop rule are in [experiments/PREDICTIONS.md](experiments/PREDICTIONS.md), committed
before any run. The hardware is one 8 GB graphics card, so the tests are toy models on toy
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
