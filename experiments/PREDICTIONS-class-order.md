# One class at a time: theory and predictions, written before any code or run

Written on 26 Nov 2026. No code for this test exists yet, and nothing has been run.

## The question

Masticate one batch of the digit 0, then one batch of the digit 1, then 2, and so on. Is that
better than plain SGD on shuffled data? The hoped-for gains are locality of reference and a
curriculum.

## What the theory says

- **A batch of one class can be fitted by a wrong rule.** With every label the same, the
  loss on the batch goes down by answering that digit for every input. Nothing in the batch
  says otherwise.
- **The classes share their features, so this is undone and redone at every batch.** The
  cross-mastication test showed what one batch does to the others through shared features:
  a batch that pulls the wrong way gives negative entries (`docs/proofs.md`, section 15). A
  batch of only 3s pulls every other digit towards "3".
- **The order 0, 1, 2, ... is not a curriculum.** In the chain, a later chapter is built on
  the earlier ones (sections 11 to 13). The digit 2 is not built on the digit 1. So the result
  that order matters does not apply.
- **Locality of reference is a cost of loading, not of learning.** It cannot show up in
  accuracy at equal steps.

This setting is known as class-incremental learning, and its failure as catastrophic
forgetting (recalled, not checked against a source).

## What would be run

- **Data:** the 1,797 scikit-learn digits already in `experiments/data/digits.npz`, which need
  no download. The same test on MNIST needs a download of about 11 MB and the author's yes.
- **Model:** 64 inputs, 128 `tanh` units, 10 outputs, cross-entropy, plain SGD at rate 0.1,
  batches of 32, 2,000 gradient steps, seeds 0 to 4.
- **Shuffled:** plain SGD on shuffled batches.
- **One class at a time, round and round:** a batch of 0s for `m` steps, then a batch of 1s,
  ..., a batch of 9s, and again from 0, with `m` = 1, 4, 16, 64.
- **One class at a time, once through:** 200 steps on 0s, then 200 on 1s, ..., then 200 on 9s.

## Predictions

A prediction holds when it is true in at least 4 of the 5 seeds.

- **C1.** Every one-class-at-a-time run ends with a lower test accuracy than shuffled SGD.
- **C2.** Round and round, the test accuracy with `m` = 64 is lower than with `m` = 1.
- **C3.** Once through, the test accuracy at the end is at most 30%, and at least 90% of the
  test digits are answered as 9, the last class seen.

## Amendment of 26 Nov 2026, before any code or run: the test is run on MNIST

The author asked for MNIST, so MNIST replaces the scikit-learn digits. Nothing had been run.
The predictions C1 to C3 are unchanged. What changes:

- **Data:** MNIST, 60,000 training and 10,000 test images of 28 by 28 pixels, pixels divided
  by 255. It is downloaded once (about 11 MB) and is not kept in the repository.
- **Model:** 784 inputs, 128 `tanh` units, 10 outputs. The rest is as above: cross-entropy,
  plain SGD at rate 0.1, batches of 32, 2,000 gradient steps, seeds 0 to 4.
- **A batch of one class** is 32 training images of that digit, drawn at random. Each visit
  to a class draws a new batch and masticates it for `m` steps. Once through, each class gets
  one batch and 200 steps on it.
- **Test accuracy** is measured on all 10,000 test images at the end of the run.

## Stop rule

The runs above and nothing more; no change after the results are seen.
