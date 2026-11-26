# Results of the class-order test on MNIST, 26 Nov 2026

The test of [../experiments/PREDICTIONS-class-order.md](../experiments/PREDICTIONS-class-order.md)
was run once on MNIST, on the processor, in 26 seconds. The predictions were committed before
the code existed. Every number is from `python experiments/class_order.py`; the full output
is in `experiments/results/class-order-summary.txt`. No run had a loss that stopped being a
number. The script was written and run by a Sonnet agent.

## The question

Masticate one batch of the digit 0, then one batch of 1s, then 2s, and so on. Is that better
than plain SGD on shuffled data? Every run had 2,000 gradient steps on a network of 784
inputs, 128 `tanh` units and 10 outputs.

## The numbers

Test accuracy on the 10,000 test images, mean of five seeds:

| Schedule | Test accuracy |
|---|---|
| Shuffled SGD | 93.4% |
| One class at a time, 1 step per batch, round and round | 93.3% |
| One class at a time, 4 steps per batch | 83.5% |
| One class at a time, 16 steps per batch | 55.9% |
| One class at a time, 64 steps per batch | 32.2% |
| One class at a time, once through, 200 steps per class | 10.1% |

## The verdicts

| | Prediction | Verdict | Measured |
|---|---|---|---|
| C1 | Every one-class-at-a-time run ends below shuffled SGD | holds, 4 of 5 seeds | in one seed, 1 step per batch ended 0.03 points above shuffled |
| C2 | Round and round, 64 steps per batch is worse than 1 | holds, 5 of 5 | 32.2% against 93.3% |
| C3 | Once through, the model answers "9" for at least 90% of test images, with accuracy at most 30% | holds, 5 of 5 | it answered "9" for every test image; accuracy 10.09%, the share of 9s in the test set |

## What this says

- **The longer the stay on one class, the worse the result.** Accuracy fell from 93% to 32% as
  the steps per batch went from 1 to 64, and to chance when each class was done once and left.
- **Once through, the model kept only the last class.** It answered "9" for all 10,000 test
  images, in every seed. This is the forgetting the theory describes: a batch of one class is
  fitted by answering that class for everything, and nothing in the batch says otherwise.
- **With one step per batch, the order hardly mattered.** Going round the classes one step at
  a time was within 0.1 points of shuffled SGD. That is why C1 holds by only 4 seeds of 5: the
  difference between those two schedules is about the size of the noise. Read C1 as "staying
  on one class never helped", not as "one class at a time is always worse".
- **This is the time-box verdict again.** The shortest stay, one step and move on, was the
  best of the one-class schedules.

## What this does not say

- One model, one learning rate, 2,000 steps, plain SGD with no momentum.
- The order 0, 1, 2, ... is not a curriculum of prerequisites: the digit 2 is not built on the
  digit 1. This test says nothing about data where later material is built on earlier
  material; the chain tests are about that.
- Locality of reference was not measured. It is a cost of loading, and it cannot show up in
  accuracy at equal steps.
