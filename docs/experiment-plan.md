# Experiment plan: plain SGD against TMT-SGD on toy tasks

Written on 26 Nov 2026, before any code was run. The predictions and the stop rule are in
`experiments/PREDICTIONS.md`. Nothing runs on the graphics card until the author says so.

## The training loop

The only thing that changes between the methods is how many steps a batch gets.

```python
for inputs, labels in batches:
    arrival_loss = None
    for _ in range(step_cap):            # 1 for plain SGD; 4, 16, 64; 256 for "until"
        loss = loss_function(model(inputs), labels)
        if arrival_loss is None:
            arrival_loss = loss.item()
        elif chew == "until" and loss.item() <= 0.01 * arrival_loss:
            break                        # the batch is learned: move on
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
```

- **Optimiser:** `torch.optim.SGD` with no momentum and no weight decay, so that the loop is
  the one the theorems are about.
- **Batch:** 32 examples.
- **Budget:** 10,000 gradient steps per run, the same for every method. A chew of 64 sees 64
  times fewer examples in that budget. Both counts are logged, so the methods can be compared
  at equal steps and at equal examples.

## The three tasks

| Task | Model | Data | Loss | Why it is here |
|---|---|---|---|---|
| `linear` | 2,000 weights, no hidden layer, starts at zero | inputs with features of unequal strength (feature `i` scaled by `1/sqrt(i)`), labels from a linear teacher | squared error | The home ground of the theorems: linear in its weights, a full fit exists when the labels are clean |
| `mlp` | 20 inputs, one hidden layer of 256 `tanh` units | random inputs, labels from a teacher network of 16 units | squared error | A real network, where the theorems are only a guide |
| `digits` | 64 inputs, one hidden layer of 128 `tanh` units, 10 outputs | the 1,797 handwritten digits of 8 by 8 pixels that come with scikit-learn | cross-entropy | Real data and a loss the theorems do not cover |

No task downloads anything. `linear` and `mlp` make their data from a seed. The digits come
with scikit-learn and are saved once to `experiments/data/digits.npz`, so that the training
script needs only torch.

**Data modes.** `stream`: every batch is new (only `linear` and `mlp` can do this). `pool`: a
fixed set of examples, gone through in a new order each pass. The pool is 2,048 examples for
`linear` and `mlp`, and 1,297 training digits for `digits` (250 for validation and 250 for
test, split once with seed 0).

**Noise.** For `linear` and `mlp`: none, or noise added to the training labels of 0.5 times
the spread of the labels. For `digits`: none, or 20% of the training labels replaced by a
wrong digit. Validation and test labels are always clean.

## What is varied

- Steps per batch: 1, 4, 16, 64, `until`.
- Learning rate: 3, 1, 0.3 for `linear`; 0.3, 0.1, 0.03 for `mlp` and `digits`.
- Seeds: 0 to 4.

That is 750 runs of 10,000 steps: 600 for `linear` and `mlp` (2 tasks, 2 noise levels, 2 data
modes, 5 chews, 3 rates, 5 seeds) and 150 for `digits`.

## What is measured

- Test loss and validation loss, every 10 steps for the first 500 steps and every 250 steps
  after that, with the number of examples seen so far. For `digits`, test accuracy too.
- The rise in the loss on the previous batch, measured right after the next chew. This is the
  forgetting.
- The steps each `until` chew took.

The author's first reason (the batch stays on the card) is measured by counting examples
loaded, not with a clock. On toy data the loading time is too small to measure.

## Cost

Every model here is far below 8 GB: the largest tensor is the `linear` test set, 4,096 by
2,000 numbers, about 33 MB. The time is not known until the first runs are timed. My guess is
about an hour for all 750 runs on the card. The script writes one line per finished run, so
it can be stopped and started again.

## What is not tested

Large models, real language data, momentum and Adam, and learning rate schedules. A result
here says nothing certain about them.
