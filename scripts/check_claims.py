"""Check claims C1 to C4 of the README with numbers. Runs on the processor in a second.

This is a check of formulas, not the experiment. Everything printed is computed here.

Setting: a model that is linear in its weights, with more weights than data, and targets
made by one set of weights (a "full fit"). A mastication is gradient descent on one batch until it
fits, which is the projection onto the fits of that batch.
"""

import numpy as np

WEIGHT_COUNT = 400
BATCH_SIZE = 16
BATCH_COUNT = 60
TRIAL_COUNT = 2000


def masticate(weights: np.ndarray, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The nearest fit of one batch."""
    return weights + np.linalg.pinv(features) @ (targets - features @ weights)


def masticate_by_gradient_descent(weights, features, targets, step_count=20000):
    strength = np.linalg.eigvalsh(features @ features.T).max()
    for _ in range(step_count):
        weights = weights - (1.0 / strength) * features.T @ (features @ weights - targets)
    return weights


def main() -> None:
    random = np.random.default_rng(0)
    full_fit = random.normal(size=WEIGHT_COUNT)
    batches = []
    for _ in range(BATCH_COUNT):
        features = random.normal(size=(BATCH_SIZE, WEIGHT_COUNT)) / np.sqrt(WEIGHT_COUNT)
        batches.append((features, features @ full_fit))

    start = np.zeros(WEIGHT_COUNT)
    first_features, first_targets = batches[0]
    gap = np.abs(masticate_by_gradient_descent(start, first_features, first_targets)
                 - masticate(start, first_features, first_targets)).max()
    print(f"A mastication is a projection: gradient descent and the projection differ by at most {gap:.2e}")

    weights = start
    worst_pythagoras = 0.0
    nearer_count = 0
    travel_squared_sum = 0.0
    worst_forgetting_ratio = 0.0
    forgetting_ratios = []
    travels = []
    previous = None
    for features, targets in batches:
        masticated = masticate(weights, features, targets)
        travel = np.linalg.norm(masticated - weights)
        before = np.linalg.norm(weights - full_fit) ** 2
        after = np.linalg.norm(masticated - full_fit) ** 2
        worst_pythagoras = max(worst_pythagoras, abs(before - after - travel ** 2))
        for _ in range(TRIAL_COUNT // BATCH_COUNT):
            reachable = weights + features.T @ random.normal(size=BATCH_SIZE) * random.uniform(0, 3)
            if np.linalg.norm(reachable - full_fit) ** 2 < after - 1e-12:
                nearer_count += 1
        if previous is not None:
            old_features, _ = previous
            change = np.linalg.norm(old_features @ (masticated - weights))
            ceiling = np.linalg.norm(old_features, 2) * travel
            forgetting_ratios.append(change / ceiling)
            worst_forgetting_ratio = max(worst_forgetting_ratio, change / ceiling)
        travel_squared_sum += travel ** 2
        travels.append(travel)
        previous = (features, targets)
        weights = masticated

    start_distance_squared = np.linalg.norm(start - full_fit) ** 2
    print(f"C1: largest error in 'distance squared falls by travel squared': {worst_pythagoras:.2e}")
    print(f"C2: reachable points nearer to the full fit than the masticated point: {nearer_count} of "
          f"{(TRIAL_COUNT // BATCH_COUNT) * BATCH_COUNT} tried")
    print(f"C3: sum of squared travel {travel_squared_sum:.3f}, squared distance from start to "
          f"the full fit {start_distance_squared:.3f}")
    print(f"C3: travel of mastication 1, 20, 40, 60: "
          + ", ".join(f"{travels[i]:.3f}" for i in (0, 19, 39, 59)))
    print(f"Ceiling on forgetting: change in old errors over the ceiling, worst "
          f"{worst_forgetting_ratio:.3f}, mean {np.mean(forgetting_ratios):.3f} (at most 1)")
    print(f"Distance to the full fit: start {np.sqrt(start_distance_squared):.3f}, "
          f"after {BATCH_COUNT} mastications {np.linalg.norm(weights - full_fit):.3f}")

    # C4: one number fitted to noisy measurements of 0 with noise 1.
    run_count, batch_count = 4000, 100
    measurements = random.normal(size=(run_count, batch_count, BATCH_SIZE))
    batch_means = measurements.mean(axis=2)
    running_means = np.cumsum(batch_means, axis=1) / np.arange(1, batch_count + 1)
    print("C4: mean squared error of one number after 1, 10, 100 batches of 16 (noise 1)")
    for count in (1, 10, 100):
        print(f"    batches {count:3d}: mastication {np.mean(batch_means[:, count - 1] ** 2):.5f}   "
              f"running mean {np.mean(running_means[:, count - 1] ** 2):.5f}   "
              f"1/(16 x batches) = {1 / (BATCH_SIZE * count):.5f}")


if __name__ == "__main__":
    main()
