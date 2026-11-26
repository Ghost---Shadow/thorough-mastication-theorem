"""Pacing a chain of chapters: mastery, time-boxed, or the last chapter only.

Plan, predictions and stop rule: experiments/PREDICTIONS-chain.md. Plain Python, nothing random.

    python experiments/chain.py              run the four settings, write experiments/results/chain.json,
                                             then print the scores of K1 to K4
    python experiments/chain.py --summarise  print the scores from that file
    python experiments/chain.py --smoke      the setting with no hard chapter only; numbers mean nothing

Everything the summary prints is computed from the results file.
"""

import argparse
import json
import math
from pathlib import Path

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_FOLDER / "chain.json"
SMOKE_PATH = RESULTS_FOLDER / "chain-smoke.json"

CHAPTER_COUNT = 6
START_WEIGHT = 0.1
RATE = 0.05
MASTERY_ERROR = 1e-4
HARD_CHAPTER = 2  # counted from 1
HARDNESS_FACTORS = [1, 4, 16, 64]  # the hard chapter learns this many times more slowly
MAX_STEPS = 10_000_000

BOUND_OUT_OF_REACH = 0.5
BOUND_FACTOR = 10.0


def product_after(weights, chapter):
    """The output after the first `chapter` chapters, for an input of one."""
    return math.prod(weights[:chapter])


def error_of(weights, chapter):
    return (product_after(weights, chapter) - 1.0) ** 2


def take_step(weights, chapter, hardness):
    """One gradient step on hardness * (product - 1)^2. It moves weights 1 to `chapter`."""
    gap = product_after(weights, chapter) - 1.0
    new_weights = list(weights)
    for index in range(chapter):
        others = math.prod(weights[:index] + weights[index + 1:chapter])
        new_weights[index] = weights[index] - RATE * hardness * 2.0 * gap * others
    return new_weights


def hardness_of(chapter, factor):
    return 1.0 / factor if chapter == HARD_CHAPTER else 1.0


def is_finite(weights):
    return all(math.isfinite(weight) for weight in weights)


def errors_of_all(weights):
    return [error_of(weights, chapter) for chapter in range(1, CHAPTER_COUNT + 1)]


def run_mastery(factor):
    """Stay on each chapter until it is mastered. Returns (weights, steps spent on each chapter)."""
    weights = [START_WEIGHT] * CHAPTER_COUNT
    steps_per_chapter = []
    for chapter in range(1, CHAPTER_COUNT + 1):
        steps = 0
        while error_of(weights, chapter) > MASTERY_ERROR and steps < MAX_STEPS and is_finite(weights):
            weights = take_step(weights, chapter, hardness_of(chapter, factor))
            steps += 1
        steps_per_chapter.append(steps)
    return weights, steps_per_chapter


def run_schedule(factor, schedule):
    """Run a list of (chapter, step count) from the start."""
    weights = [START_WEIGHT] * CHAPTER_COUNT
    for chapter, step_count in schedule:
        for _ in range(step_count):
            if not is_finite(weights):
                return weights
            weights = take_step(weights, chapter, hardness_of(chapter, factor))
    return weights


def run_one(factor):
    mastery_weights, steps_per_chapter = run_mastery(factor)
    budget = sum(steps_per_chapter)
    box = budget // CHAPTER_COUNT
    boxes = [(chapter, box) for chapter in range(1, CHAPTER_COUNT + 1)]
    boxes[-1] = (CHAPTER_COUNT, box + budget - box * CHAPTER_COUNT)  # leftover steps go to the last chapter
    time_boxed_weights = run_schedule(factor, boxes)
    last_only_weights = run_schedule(factor, [(CHAPTER_COUNT, budget)])
    return {
        "budget": budget,
        "mastery_steps_per_chapter": steps_per_chapter,
        "steps_per_box": box,
        "weights": {"mastery": mastery_weights, "time_boxed": time_boxed_weights,
                    "last_only": last_only_weights},
        "errors": {"mastery": errors_of_all(mastery_weights),
                   "time_boxed": errors_of_all(time_boxed_weights),
                   "last_only": errors_of_all(last_only_weights)},
    }


def run_all(factors, path):
    results = {"factors": factors, "runs": {str(factor): run_one(factor) for factor in factors}}
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path.write_text(json.dumps(results), encoding="utf-8")
    return results


def verdict(name, is_true):
    print(f"{name}: {'holds' if is_true else 'fails'}")


def summarise(results) -> None:
    runs = results["runs"]
    for factor, run in runs.items():
        print(f"\nhard chapter learns {factor} times more slowly: budget {run['budget']} steps, "
              f"mastery spent {run['mastery_steps_per_chapter']} per chapter, "
              f"a box is {run['steps_per_box']} steps")
        for pace, errors in run["errors"].items():
            print(f"  {pace:10} error of chapters 1 to {CHAPTER_COUNT}: "
                  + "  ".join(f"{error:.2e}" for error in errors))
            print(f"  {'':10} weights: " + "  ".join(f"{weight:.3f}" for weight in run["weights"][pace]))

    def goal_error(run, pace):
        return run["errors"][pace][-1]

    print("\nK1: 'last chapter only' ends at 0.5 or above on the last chapter; mastery at 0.0001 or below")
    k1 = True
    for factor, run in runs.items():
        is_true = (goal_error(run, "last_only") >= BOUND_OUT_OF_REACH
                   and goal_error(run, "mastery") <= MASTERY_ERROR)
        k1 = k1 and is_true
        print(f"  slowness {factor}: {str(is_true).lower():5}  last only {goal_error(run, 'last_only'):.4f}, "
              f"mastery {goal_error(run, 'mastery'):.2e}")
    verdict("K1", k1)

    print("\nK2: with no hard chapter, the time-boxed error on the last chapter is at most 10 times mastery's")
    if "1" in runs:
        ratio = goal_error(runs["1"], "time_boxed") / goal_error(runs["1"], "mastery")
        print(f"  slowness 1: time-boxed / mastery = {ratio:.3g}")
        verdict("K2", ratio <= BOUND_FACTOR)

    print("\nK3: with a hard chapter (16 and 64), the time-boxed error on the last chapter is at least 10 times mastery's")
    hard = [factor for factor in ("16", "64") if factor in runs]
    k3 = bool(hard)
    for factor in hard:
        ratio = goal_error(runs[factor], "time_boxed") / goal_error(runs[factor], "mastery")
        k3 = k3 and ratio >= BOUND_FACTOR
        print(f"  slowness {factor}: time-boxed / mastery = {ratio:.3g}")
    if hard:
        verdict("K3", k3)

    print("\nK4: after the mastery pass, some earlier chapter has an error above 0.0001")
    k4 = True
    for factor, run in runs.items():
        worst = max(run["errors"]["mastery"][:-1])
        k4 = k4 and worst > MASTERY_ERROR
        print(f"  slowness {factor}: largest error of chapters 1 to {CHAPTER_COUNT - 1} = {worst:.2e}")
    verdict("K4", k4)


# ---------------------------------------------------------------- the test with too few steps
# Plan and predictions: experiments/PREDICTIONS-chain-tight-budget.md.

TIGHT_PATH = RESULTS_FOLDER / "chain-tight-budget.json"
TIGHT_FACTOR = 64
TIGHT_SHARES = [0.25, 0.5, 0.75]
BOUND_REACHED = 0.01


def run_mastery_within(factor, budget):
    """Mastery pacing that stops when the budget runs out.

    Returns (weights, the chapter it was on, counted from 1; one past the last if it finished)."""
    weights = [START_WEIGHT] * CHAPTER_COUNT
    steps = 0
    for chapter in range(1, CHAPTER_COUNT + 1):
        while error_of(weights, chapter) > MASTERY_ERROR:
            if steps >= budget or not is_finite(weights):
                return weights, chapter
            weights = take_step(weights, chapter, hardness_of(chapter, factor))
            steps += 1
    return weights, CHAPTER_COUNT + 1


def run_tight(path):
    _, steps_per_chapter = run_mastery(TIGHT_FACTOR)
    full_budget = sum(steps_per_chapter)
    runs = {}
    for share in TIGHT_SHARES:
        budget = int(share * full_budget)
        mastery_weights, chapter_reached = run_mastery_within(TIGHT_FACTOR, budget)
        box = budget // CHAPTER_COUNT
        boxes = [(chapter, box) for chapter in range(1, CHAPTER_COUNT + 1)]
        boxes[-1] = (CHAPTER_COUNT, box + budget - box * CHAPTER_COUNT)
        time_boxed_weights = run_schedule(TIGHT_FACTOR, boxes)
        runs[str(share)] = {
            "budget": budget,
            "steps_per_box": box,
            "mastery_was_on_chapter": chapter_reached,
            "weights": {"mastery": mastery_weights, "time_boxed": time_boxed_weights},
            "errors": {"mastery": errors_of_all(mastery_weights),
                       "time_boxed": errors_of_all(time_boxed_weights)},
        }
    results = {"full_budget": full_budget, "steps_to_leave_the_hard_chapter": sum(steps_per_chapter[:HARD_CHAPTER]),
               "runs": runs}
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path.write_text(json.dumps(results), encoding="utf-8")
    return results


def summarise_tight(results) -> None:
    runs = results["runs"]
    print(f"\nmastery needs {results['full_budget']} steps in all, "
          f"{results['steps_to_leave_the_hard_chapter']} of them to leave chapter {HARD_CHAPTER}")
    for share, run in runs.items():
        print(f"\nbudget {run['budget']} steps ({float(share):.0%}), a box is {run['steps_per_box']} steps; "
              f"mastery was on chapter {run['mastery_was_on_chapter']} when the steps ran out")
        for pace, errors in run["errors"].items():
            print(f"  {pace:10} error of chapters 1 to {CHAPTER_COUNT}: "
                  + "  ".join(f"{error:.2e}" for error in errors))
            print(f"  {'':10} weights: " + "  ".join(f"{weight:.3f}" for weight in run["weights"][pace]))

    print("\nT1: at every budget mastery is still on the hard chapter, with an error of at least 0.5 on the last chapter")
    t1 = True
    for share, run in runs.items():
        is_true = (run["mastery_was_on_chapter"] == HARD_CHAPTER
                   and run["errors"]["mastery"][-1] >= BOUND_OUT_OF_REACH)
        t1 = t1 and is_true
        print(f"  budget {float(share):.0%}: {str(is_true).lower():5}  on chapter {run['mastery_was_on_chapter']}, "
              f"error on the last chapter {run['errors']['mastery'][-1]:.4f}")
    verdict("T1", t1)

    print("\nT2: at 50% and 75%, time-boxed ends with an error of at most 0.01 on the last chapter")
    t2 = True
    for share in ("0.5", "0.75"):
        error = runs[share]["errors"]["time_boxed"][-1]
        is_true = math.isfinite(error) and error <= BOUND_REACHED
        t2 = t2 and is_true
        print(f"  budget {float(share):.0%}: {str(is_true).lower():5}  error on the last chapter {error:.3e}")
    verdict("T2", t2)
    print(f"\nAt 25% (no prediction): time-boxed error on the last chapter "
          f"{runs['0.25']['errors']['time_boxed'][-1]:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tight", action="store_true",
                        help="the test with too few steps (PREDICTIONS-chain-tight-budget.md)")
    arguments = parser.parse_args()
    if arguments.tight:
        if arguments.summarise:
            summarise_tight(json.loads(TIGHT_PATH.read_text(encoding="utf-8")))
        else:
            summarise_tight(run_tight(TIGHT_PATH))
    elif arguments.smoke:
        print("=== SMOKE TEST: no hard chapter only, the numbers mean nothing ===")
        summarise(run_all([1], SMOKE_PATH))
    elif arguments.summarise:
        summarise(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
    else:
        summarise(run_all(HARDNESS_FACTORS, RESULTS_PATH))


if __name__ == "__main__":
    main()
