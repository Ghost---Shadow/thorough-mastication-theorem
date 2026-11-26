"""One class at a time versus shuffled SGD on MNIST (spec: experiments/PREDICTIONS-class-order.md).

How to run, from the repository root:
    python experiments/class_order.py              run everything, write results/class-order.json, print summary
    python experiments/class_order.py --summarise  print the summary from the results file
    python experiments/class_order.py --smoke      seed 0 only, 100 steps, writes class-order-smoke.json
"""

import json
import math
import sys
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mnist"
RESULTS_DIR = ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "class-order.json"
SMOKE_FILE = RESULTS_DIR / "class-order-smoke.json"

SEEDS = [0, 1, 2, 3, 4]
NUMBER_OF_STEPS = 2000
BATCH_SIZE = 32
LEARNING_RATE = 0.1
HIDDEN_UNITS = 128
NUMBER_OF_CLASSES = 10
ROUND_STEPS_PER_VISIT = [1, 4, 16, 64]
ONCE_STEPS_PER_CLASS = 200
SCHEDULES = ["shuffled"] + [f"round_{m}" for m in ROUND_STEPS_PER_VISIT] + ["once"]
ONE_CLASS_SCHEDULES = SCHEDULES[1:]
MINIMUM_SEEDS_FOR_HOLDS = 4
C3_MAXIMUM_ACCURACY = 0.30
C3_MINIMUM_SHARE_OF_NINE = 0.90


def load_mnist():
    """Return (train_images, train_labels, test_images, test_labels), images flat in [0, 1]."""
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    train_images = train.data.float().div(255.0).reshape(-1, 784)
    test_images = test.data.float().div(255.0).reshape(-1, 784)
    return train_images, train.targets.long(), test_images, test.targets.long()


def make_initial_weights(seed):
    """Initial weights depend on the seed only (PyTorch Linear default uniform init)."""
    generator = torch.Generator().manual_seed(seed)
    bound_first = 1.0 / math.sqrt(784)
    bound_second = 1.0 / math.sqrt(HIDDEN_UNITS)
    first_weight = (torch.rand(HIDDEN_UNITS, 784, generator=generator) * 2 - 1) * bound_first
    first_bias = (torch.rand(HIDDEN_UNITS, generator=generator) * 2 - 1) * bound_first
    second_weight = (torch.rand(NUMBER_OF_CLASSES, HIDDEN_UNITS, generator=generator) * 2 - 1) * bound_second
    second_bias = (torch.rand(NUMBER_OF_CLASSES, generator=generator) * 2 - 1) * bound_second
    return [first_weight, first_bias, second_weight, second_bias]


def forward(parameters, images):
    first_weight, first_bias, second_weight, second_bias = parameters
    hidden = torch.tanh(images @ first_weight.T + first_bias)
    return hidden @ second_weight.T + second_bias


def batch_plan(schedule, class_indices, number_of_train_images, generator):
    """Yield one tensor of training indices per gradient step, exactly NUMBER_OF_STEPS of them."""

    def draw_from_class(digit):
        pool = class_indices[digit]
        return pool[torch.randperm(len(pool), generator=generator)[:BATCH_SIZE]]

    if schedule == "shuffled":
        for _ in range(NUMBER_OF_STEPS):
            yield torch.randperm(number_of_train_images, generator=generator)[:BATCH_SIZE]
    elif schedule.startswith("round_"):
        steps_per_visit = int(schedule.split("_")[1])
        steps_taken = 0
        digit = 0
        while steps_taken < NUMBER_OF_STEPS:
            batch = draw_from_class(digit)
            for _ in range(min(steps_per_visit, NUMBER_OF_STEPS - steps_taken)):
                yield batch
                steps_taken += 1
            digit = (digit + 1) % NUMBER_OF_CLASSES
    elif schedule == "once":
        for digit in range(NUMBER_OF_CLASSES):
            batch = draw_from_class(digit)
            for _ in range(ONCE_STEPS_PER_CLASS):
                yield batch
    else:
        raise ValueError(schedule)


def run_one(schedule, seed, data, number_of_steps):
    train_images, train_labels, test_images, test_labels = data
    class_indices = [(train_labels == digit).nonzero().squeeze(1) for digit in range(NUMBER_OF_CLASSES)]
    parameters = [tensor.clone().requires_grad_(True) for tensor in make_initial_weights(seed)]
    generator = torch.Generator().manual_seed(1000 + seed)
    any_non_finite = False
    step_count = 0
    for batch in batch_plan(schedule, class_indices, len(train_labels), generator):
        if step_count >= number_of_steps:
            break
        logits = forward(parameters, train_images[batch])
        loss = torch.nn.functional.cross_entropy(logits, train_labels[batch])
        if not math.isfinite(loss.item()):
            any_non_finite = True
        gradients = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter -= LEARNING_RATE * gradient
        step_count += 1
    with torch.no_grad():
        predictions = forward(parameters, test_images).argmax(dim=1)
    correct = predictions == test_labels
    return {
        "test_accuracy": correct.float().mean().item(),
        "answered_share": [(predictions == digit).float().mean().item() for digit in range(NUMBER_OF_CLASSES)],
        "accuracy_per_true_digit": [correct[test_labels == digit].float().mean().item() for digit in range(NUMBER_OF_CLASSES)],
        "non_finite_loss": any_non_finite,
        "steps": step_count,
    }


def run_everything(seeds, number_of_steps, output_file):
    data = load_mnist()
    results = {schedule: {} for schedule in SCHEDULES}
    for seed in seeds:
        for schedule in SCHEDULES:
            results[schedule][str(seed)] = run_one(schedule, seed, data, number_of_steps)
            print(f"done {schedule} seed {seed}: accuracy {results[schedule][str(seed)]['test_accuracy']:.4f}", file=sys.stderr)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=1))


def verdict(per_seed_flags):
    count = sum(per_seed_flags)
    return f"{'holds' if count >= MINIMUM_SEEDS_FOR_HOLDS else 'fails'} ({count} of {len(per_seed_flags)} seeds)"


def summarise(input_file, label):
    results = json.loads(input_file.read_text())
    seeds = sorted(results["shuffled"].keys(), key=int)
    print(label)
    print(f"{'schedule':<10} {'mean':>7}  per-seed accuracy")
    for schedule in SCHEDULES:
        accuracies = [results[schedule][seed]["test_accuracy"] for seed in seeds]
        print(f"{schedule:<10} {sum(accuracies) / len(accuracies):7.4f}  " + " ".join(f"{a:.4f}" for a in accuracies))
    non_finite = [(s, seed) for s in SCHEDULES for seed in seeds if results[s][seed]["non_finite_loss"]]
    print(f"runs with a non-finite loss: {non_finite if non_finite else 'none'}")

    def accuracy(schedule, seed):
        return results[schedule][seed]["test_accuracy"]

    flags = [all(accuracy(s, seed) < accuracy("shuffled", seed) for s in ONE_CLASS_SCHEDULES) for seed in seeds]
    print(f"C1 every one-class schedule below shuffled, per seed {flags}: {verdict(flags)}")
    flags = [accuracy("round_64", seed) < accuracy("round_1", seed) for seed in seeds]
    print(f"C2 round_64 below round_1, per seed {flags}: {verdict(flags)}")
    pairs = [(accuracy("once", seed), results["once"][seed]["answered_share"][9]) for seed in seeds]
    flags = [acc <= C3_MAXIMUM_ACCURACY and share >= C3_MINIMUM_SHARE_OF_NINE for acc, share in pairs]
    print("C3 once: accuracy at most 0.30 and share answered as 9 at least 0.90")
    for seed, (acc, share) in zip(seeds, pairs):
        print(f"  seed {seed}: accuracy {acc:.4f}, share answered as 9 {share:.4f}")
    print(f"C3 {verdict(flags)}")


def main():
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        run_everything([0], 100, SMOKE_FILE)
        summarise(SMOKE_FILE, "SMOKE TEST ONLY (seed 0, 100 steps): not the experiment")
    elif "--summarise" in arguments:
        summarise(RESULTS_FILE, "Summary from class-order.json")
    else:
        run_everything(SEEDS, NUMBER_OF_STEPS, RESULTS_FILE)
        summarise(RESULTS_FILE, "Summary from class-order.json")


if __name__ == "__main__":
    main()
