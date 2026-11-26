"""Chapters that are supersets, in a model that is linear in its weights.

Plan, predictions and stop rule: experiments/PREDICTIONS-chapters.md. Processor only, float64.

    python experiments/chapters.py              run seeds 0 to 4, write experiments/results/chapters.json,
                                                then print the scores of S1 to S7
    python experiments/chapters.py --summarise  print the scores from that file
    python experiments/chapters.py --smoke      one seed, written to chapters-smoke.json; numbers mean nothing

Everything the summary prints is computed from the results file.
"""

import argparse
import json
import math
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_FOLDER / "chapters.json"
SMOKE_PATH = RESULTS_FOLDER / "chapters-smoke.json"

SEEDS = [0, 1, 2, 3, 4]
LAYOUTS = ["superset", "disjoint"]
CHAPTER_COUNT = 4
BLOCK_SIZE = 24  # the new concepts of one chapter
WEIGHT_COUNT = CHAPTER_COUNT * BLOCK_SIZE
EXAMPLES_PER_BLOCK = 32
TARGET_SHARE = 1e-6  # a mastication stops at this share of the loss it started with
MAX_STEPS = 200_000
TIME_BOX_STEPS = 5

BOUND_TAUGHT = 0.99
BOUND_PART = 0.9
BOUND_FORGOTTEN = 1e-4
BOUND_TRAVEL = 0.01
BOUND_SAME = 1e-9
HOLD_SHARE = 0.8  # a prediction holds when it is true in at least 4 of 5 seeds


# ---------------------------------------------------------------- data and steps

def build_chapters(layout, seed):
    """The teacher and the four chapters, each a pair (inputs, labels)."""
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(WEIGHT_COUNT, generator=generator)
    chapters = []
    for chapter in range(CHAPTER_COUNT):
        last = BLOCK_SIZE * (chapter + 1)
        if layout == "superset":
            first, example_count = 0, EXAMPLES_PER_BLOCK * (chapter + 1)
        else:
            first, example_count = BLOCK_SIZE * chapter, EXAMPLES_PER_BLOCK
        inputs = torch.zeros(example_count, WEIGHT_COUNT)
        inputs[:, first:last] = torch.randn(example_count, last - first, generator=generator)
        chapters.append((inputs, inputs @ teacher))
    return teacher, chapters


def loss_of(weights, chapter):
    inputs, labels = chapter
    return torch.mean((inputs @ weights - labels) ** 2).item()


def gradient_of(weights, chapter):
    inputs, labels = chapter
    return (2.0 / len(labels)) * inputs.T @ (inputs @ weights - labels)


def safe_rate(chapter):
    """One over the largest strength of the chapter's kernel: always stable."""
    inputs, _ = chapter
    return 1.0 / torch.linalg.eigvalsh((2.0 / len(inputs)) * inputs.T @ inputs).max().item()


def take_steps(weights, chapter, step_count):
    rate = safe_rate(chapter)
    for _ in range(step_count):
        weights = weights - rate * gradient_of(weights, chapter)
    return weights


def masticate(weights, chapter, max_steps):
    """Gradient descent on one chapter until it fits. Returns (weights, steps, reached target)."""
    rate = safe_rate(chapter)
    target = TARGET_SHARE * loss_of(weights, chapter)
    steps = 0
    while loss_of(weights, chapter) > target and steps < max_steps:
        weights = weights - rate * gradient_of(weights, chapter)
        steps += 1
    return weights, steps, loss_of(weights, chapter) <= target


# ---------------------------------------------------------------- one run

def run_one(layout, seed, max_steps):
    teacher, chapters = build_chapters(layout, seed)
    start = torch.zeros(WEIGHT_COUNT)
    start_losses = [loss_of(start, chapter) for chapter in chapters]

    table = [[None] * CHAPTER_COUNT for _ in range(CHAPTER_COUNT)]
    table_steps, table_reached = [], []
    for taught, chapter in enumerate(chapters):
        weights, steps, reached = masticate(start, chapter, max_steps)
        table_steps.append(steps)
        table_reached.append(reached)
        for measured, other in enumerate(chapters):
            if measured != taught:
                table[taught][measured] = 1.0 - loss_of(weights, other) / start_losses[measured]

    weights = start
    travels, mastery_reached = [], []
    for chapter in chapters:
        before = weights
        weights, _, reached = masticate(weights, chapter, max_steps)
        travels.append(torch.linalg.vector_norm(weights - before).item())
        mastery_reached.append(reached)
    loss_share_after_mastery = [loss_of(weights, chapter) / start_losses[index]
                                for index, chapter in enumerate(chapters)]

    weights = take_steps(start, chapters[0], TIME_BOX_STEPS)
    first_chapter_loss_after_its_steps = loss_of(weights, chapters[0])
    for chapter in chapters[1:]:
        weights = take_steps(weights, chapter, TIME_BOX_STEPS)
    first_chapter_loss_at_the_end = loss_of(weights, chapters[0])

    return {
        "table": table,
        "table_steps": table_steps,
        "table_reached_target": table_reached,
        "travels": travels,
        "mastery_reached_target": mastery_reached,
        "loss_share_after_mastery": loss_share_after_mastery,
        "squared_distance_to_teacher": torch.sum((teacher - start) ** 2).item(),
        "first_chapter_loss_after_its_steps": first_chapter_loss_after_its_steps,
        "first_chapter_loss_at_the_end": first_chapter_loss_at_the_end,
        "first_chapter_loss_at_the_start": start_losses[0],
    }


def run_all(seeds, max_steps, path):
    results = {"seeds": seeds, "layouts": {}}
    for layout in LAYOUTS:
        results["layouts"][layout] = {}
        for seed in seeds:
            run = run_one(layout, seed, max_steps)
            results["layouts"][layout][str(seed)] = run
            print(f"{layout} seed {seed}: most steps in a mastication {max(run['table_steps'])}, "
                  f"all reached their target: {all(run['table_reached_target'] + run['mastery_reached_target'])}",
                  flush=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path.write_text(json.dumps(results), encoding="utf-8")
    return results


# ---------------------------------------------------------------- scores

def recovered_order(table):
    """Pairs (i, j), numbered from 1, that the rule of S3 puts as "i before j"."""
    return sorted((i + 1, j + 1) for i in range(CHAPTER_COUNT) for j in range(CHAPTER_COUNT)
                  if i != j and table[j][i] >= BOUND_TAUGHT and table[i][j] <= BOUND_PART)


def score(name, sentence, runs, measure):
    """Print the measured numbers of each seed and whether the prediction holds."""
    print(f"\n{name}: {sentence}")
    true_count = 0
    for seed, run in runs.items():
        is_true, numbers = measure(run)
        true_count += is_true
        print(f"  seed {seed}: {str(is_true).lower():5}  {numbers}")
    needed = math.ceil(HOLD_SHARE * len(runs))
    print(f"{name}: {'holds' if true_count >= needed else 'fails'}   "
          f"(true in {true_count} of {len(runs)} seeds, {needed} needed)")


def summarise(results) -> None:
    superset, disjoint = results["layouts"]["superset"], results["layouts"]["disjoint"]
    print(f"\nseeds {results['seeds']}")
    print("\nSuperset chapters, mean table over the seeds (row = chapter masticated, column = chapter measured)")
    for taught in range(CHAPTER_COUNT):
        cells = []
        for measured in range(CHAPTER_COUNT):
            values = [run["table"][taught][measured] for run in superset.values()]
            cells.append("   --  " if taught == measured else f"{sum(values) / len(values):7.3f}")
        print(f"  chapter {taught + 1}: " + " ".join(cells))

    later_to_earlier = [(k, j) for k in range(CHAPTER_COUNT) for j in range(k)]
    earlier_to_later = [(k, j) for k in range(CHAPTER_COUNT) for j in range(k + 1, CHAPTER_COUNT)]
    expected_order = sorted((i + 1, j + 1) for i in range(CHAPTER_COUNT) for j in range(i + 1, CHAPTER_COUNT))

    def s1(run):
        lowest = min(run["table"][k][j] for k, j in later_to_earlier)
        return lowest >= BOUND_TAUGHT, f"lowest entry from a later chapter to an earlier one = {lowest:.6f}"

    def s2(run):
        entries = [run["table"][k][j] for k, j in earlier_to_later]
        return (min(entries) > 0 and max(entries) <= BOUND_PART,
                "entries 1>2, 1>3, 1>4, 2>3, 2>4, 3>4 = " + ", ".join(f"{entry:.3f}" for entry in entries))

    def s3(run):
        order = recovered_order(run["table"])
        return order == expected_order, f"pairs 'i before j' = {order}"

    def s4(run):
        worst = max(run["loss_share_after_mastery"])
        return worst <= BOUND_FORGOTTEN, f"largest loss left on a chapter, as a share of its start = {worst:.3e}"

    def s5(run):
        ratio = sum(travel ** 2 for travel in run["travels"]) / run["squared_distance_to_teacher"]
        shares = [travel ** 2 / run["squared_distance_to_teacher"] for travel in run["travels"]]
        return (abs(ratio - 1) <= BOUND_TRAVEL,
                f"sum of squared travels / squared distance to the teacher = {ratio:.6f}; "
                "share of each chapter = " + ", ".join(f"{share:.3f}" for share in shares))

    def s6(run):
        after, end = run["first_chapter_loss_after_its_steps"], run["first_chapter_loss_at_the_end"]
        start = run["first_chapter_loss_at_the_start"]
        return end < after, (f"loss on chapter 1 as a share of its start: after its own steps "
                             f"{after / start:.4f}, at the end of the pass {end / start:.4f}")

    def s7(run):
        after, end = run["first_chapter_loss_after_its_steps"], run["first_chapter_loss_at_the_end"]
        start = run["first_chapter_loss_at_the_start"]
        return abs(end / after - 1) <= BOUND_SAME, (
            f"loss on chapter 1 as a share of its start: after its own steps {after / start:.4f}, "
            f"at the end of the pass {end / start:.4f}; relative change {abs(end / after - 1):.1e}")

    score("S1", "a later chapter teaches all the earlier ones (entry at least 0.99)", superset, s1)
    score("S2", "an earlier chapter teaches only part of a later one (above 0, at most 0.9)", superset, s2)
    score("S3", "the table gives the order 1, 2, 3, 4", superset, s3)
    score("S4", "mastery in order forgets nothing (at most 1e-4 of the start loss is left)", superset, s4)
    score("S5", "the squared travels add up to the squared distance to the teacher, within 1%", superset, s5)
    score("S6", "time-boxing is forgiven: chapter 1 is lower at the end than after its own steps", superset, s6)
    score("S7", "disjoint chapters: chapter 1 at the end equals chapter 1 after its own steps", disjoint, s7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print("=== SMOKE TEST: one seed, the numbers mean nothing ===")
        summarise(run_all([0], MAX_STEPS, SMOKE_PATH))
    elif arguments.summarise:
        summarise(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
    else:
        summarise(run_all(SEEDS, MAX_STEPS, RESULTS_PATH))


if __name__ == "__main__":
    main()
