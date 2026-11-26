"""Save scikit-learn's 8 by 8 digits to experiments/data/digits.npz, once.

    python experiments/make_digits.py

Pixels are scaled to [0, 1]. The split is made once with seed 0: 1,297 for training, 250 for
validation, 250 for test. The training script reads the file with numpy only.
"""

from pathlib import Path

import numpy
from sklearn.datasets import load_digits

OUTPUT = Path(__file__).resolve().parent / "data" / "digits.npz"
SEED = 0
VALIDATION_SIZE = 250
TEST_SIZE = 250
MAX_PIXEL = 16.0


def main() -> None:
    digits = load_digits()
    inputs = (digits.data / MAX_PIXEL).astype(numpy.float32)
    labels = digits.target.astype(numpy.int64)
    order = numpy.random.RandomState(SEED).permutation(len(labels))
    test_part = order[:TEST_SIZE]
    validation_part = order[TEST_SIZE:TEST_SIZE + VALIDATION_SIZE]
    train_part = order[TEST_SIZE + VALIDATION_SIZE:]
    OUTPUT.parent.mkdir(exist_ok=True)
    numpy.savez_compressed(
        OUTPUT,
        train_inputs=inputs[train_part], train_labels=labels[train_part],
        validation_inputs=inputs[validation_part], validation_labels=labels[validation_part],
        test_inputs=inputs[test_part], test_labels=labels[test_part])
    print(f"saved {OUTPUT}: {len(train_part)} train, {len(validation_part)} validation, "
          f"{len(test_part)} test")


if __name__ == "__main__":
    main()
