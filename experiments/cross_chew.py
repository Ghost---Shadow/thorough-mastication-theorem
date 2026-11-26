"""Cross-mastication: a table of what chewing one batch teaches about every other batch.

Plan: docs/cross-chew-plan.md. Predictions and stop rule: experiments/PREDICTIONS-cross-chew.md.
CPU only, float64.

    python experiments/cross_chew.py              run seeds 0 to 4, both models,
                                                  write experiments/results/cross-chew.json
    python experiments/cross_chew.py --summarise  print the tables and Q1..Q8, M1..M3 from that file
    python experiments/cross_chew.py --smoke      one seed, 300 steps per chew, writes
                                                  experiments/results/cross-chew-smoke.json,
                                                  numbers mean nothing
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_FOLDER / "cross-chew.json"
SMOKE_PATH = RESULTS_FOLDER / "cross-chew-smoke.json"

SEEDS = [0, 1, 2, 3, 4]
MODELS = ["linear", "mlp"]
GROUP_COUNT = 3
GROUP_SIZE = 96
WEIGHT_COUNT = GROUP_COUNT * GROUP_SIZE
BATCH_SIZE = 32
BATCH_COUNT = 20
HIDDEN_UNITS = 256
MAX_STEPS = 20000
SMOKE_MAX_STEPS = 300
TARGET_SHARE = {"linear": 1e-6, "mlp": 1e-4}
MLP_RATES = [0.1, 0.03, 0.01, 0.003]

DUPLICATE_OF_ZERO = 6
COVERED_BATCH = 7
COVERING_BATCH = 1
BAD_BATCH = 13
COMBINATION_COUNT = 8
TRAVEL_ORDER = [0, 6, 1, 7, 2, 3, 4, 5] + list(range(8, 20))

# Group of each batch (0, 1, 2) and the planted groups as sets.
BATCH_GROUP = [0] * 8 + [1] * 6 + [2] * 6
PLANTED_GROUPS = [set(range(0, 8)), set(range(8, 14)), set(range(14, 20))]

BOUND_DUPLICATE = 0.99
BOUND_COVERED_REVERSE = 0.9
BOUND_SEPARATE = 0.001
BOUND_JOIN = 0.01
HOLD_FRACTION_NUMERATOR = 4  # holds when true in ceil(0.8 * seeds) seeds
HOLD_FRACTION_DENOMINATOR = 5


# ---------------------------------------------------------------- data

def make_group_inputs(generator, group, row_count):
    inputs = torch.zeros(row_count, WEIGHT_COUNT)
    start = GROUP_SIZE * group
    inputs[:, start:start + GROUP_SIZE] = torch.randn(
        row_count, GROUP_SIZE, generator=generator)
    return inputs


def build_batches(seed):
    """Return a list of 20 (inputs, labels) pairs. Depends on the seed only, not on the model."""
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(WEIGHT_COUNT, generator=generator)
    inputs_of = {}
    for batch in range(BATCH_COUNT):
        if batch in (DUPLICATE_OF_ZERO, COVERED_BATCH):
            continue  # made from other batches below
        inputs_of[batch] = make_group_inputs(generator, BATCH_GROUP[batch], BATCH_SIZE)
    order = torch.randperm(BATCH_SIZE, generator=generator)
    inputs_of[DUPLICATE_OF_ZERO] = inputs_of[0][order]
    # Divided by the root of the batch size, so that a combination is as large as one example.
    coefficients = torch.randn(COMBINATION_COUNT, BATCH_SIZE, generator=generator) / math.sqrt(BATCH_SIZE)
    combinations = coefficients @ inputs_of[COVERING_BATCH]
    repeats = BATCH_SIZE // COMBINATION_COUNT
    inputs_of[COVERED_BATCH] = combinations.repeat_interleave(repeats, dim=0)
    batches = []
    for batch in range(BATCH_COUNT):
        labels = inputs_of[batch] @ teacher
        if batch == BAD_BATCH:
            labels = -labels
        batches.append((inputs_of[batch], labels))
    return batches


# ---------------------------------------------------------------- models

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(WEIGHT_COUNT))

    def forward(self, inputs):
        return inputs @ self.weights


def make_model(model_name, seed):
    if model_name == "linear":
        return LinearModel()
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(WEIGHT_COUNT, HIDDEN_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_UNITS, 1),
    )


def predict(model, inputs):
    output = model(inputs)
    return output.reshape(-1)


def batch_loss(model, inputs, labels):
    return torch.mean((predict(model, inputs) - labels) ** 2)


def flat_parameters(model):
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


# ---------------------------------------------------------------- the chew

def linear_rate(inputs):
    kernel = (2.0 / BATCH_SIZE) * (inputs @ inputs.T)
    return 1.0 / torch.linalg.eigvalsh(kernel).max().item()


def chew(model, inputs, labels, rate, target_share, max_steps):
    """Full-batch gradient descent. Returns (reached target, steps taken, final loss)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    start_loss = batch_loss(model, inputs, labels).item()
    target = target_share * start_loss
    steps = 0
    loss = start_loss
    while True:
        optimizer.zero_grad()
        loss_tensor = batch_loss(model, inputs, labels)
        loss = loss_tensor.item()
        if loss <= target or not math.isfinite(loss) or steps >= max_steps:
            break
        loss_tensor.backward()
        optimizer.step()
        steps += 1
    return loss <= target, steps, loss


def chew_at_a_safe_rate(model, model_name, inputs, labels, max_steps):
    """Chew one batch from the model's present weights. Returns (reached target, steps, rate).

    For linear the rate is one over the largest strength of the batch's kernel. For mlp it is
    the largest of MLP_RATES at which this chew does not diverge: the loss stays finite and
    ends below where it started. If every rate diverges, the smallest is used as it is.
    """
    target_share = TARGET_SHARE[model_name]
    if model_name == "linear":
        rate = linear_rate(inputs)
        did_reach, steps, _ = chew(model, inputs, labels, rate, target_share, max_steps)
        return did_reach, steps, rate
    state_before = copy.deepcopy(model.state_dict())
    start_loss = batch_loss(model, inputs, labels).item()
    for rate in MLP_RATES:
        model.load_state_dict(state_before)
        did_reach, steps, final_loss = chew(model, inputs, labels, rate, target_share, max_steps)
        if math.isfinite(final_loss) and final_loss < start_loss:
            break
    return did_reach, steps, rate


# ---------------------------------------------------------------- one run

def run_one(model_name, seed, max_steps):
    batches = build_batches(seed)
    model = make_model(model_name, seed)
    start_state = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        model.load_state_dict(start_state)
        start_losses = [batch_loss(model, inputs, labels).item() for inputs, labels in batches]

    table = [[None] * BATCH_COUNT for _ in range(BATCH_COUNT)]
    reached = []
    steps_taken = []
    rates = []
    for i, (inputs, labels) in enumerate(batches):
        model.load_state_dict(start_state)
        did_reach, steps, rate = chew_at_a_safe_rate(model, model_name, inputs, labels, max_steps)
        reached.append(did_reach)
        steps_taken.append(steps)
        rates.append(rate)
        with torch.no_grad():
            for j, (other_inputs, other_labels) in enumerate(batches):
                if j != i:
                    loss_after = batch_loss(model, other_inputs, other_labels).item()
                    table[i][j] = 1.0 - loss_after / start_losses[j]

    # The pass for travel: no reset between chews.
    model.load_state_dict(start_state)
    travel = {}
    loss_ratio_before = {}
    pass_reached = {}
    pass_steps = {}
    for i in TRAVEL_ORDER:
        inputs, labels = batches[i]
        with torch.no_grad():
            loss_before = batch_loss(model, inputs, labels).item()
        loss_ratio_before[i] = loss_before / start_losses[i]
        before = flat_parameters(model)
        did_reach, steps, _ = chew_at_a_safe_rate(model, model_name, inputs, labels, max_steps)
        travel[i] = torch.linalg.vector_norm(flat_parameters(model) - before).item()
        pass_reached[i] = did_reach
        pass_steps[i] = steps

    return {
        "table": table,
        "reached_target": reached,
        "steps": steps_taken,
        "rate_per_chew": rates,
        "start_losses": start_losses,
        "travel_order": TRAVEL_ORDER,
        "travel": [travel[i] for i in range(BATCH_COUNT)],
        "loss_before_over_loss_at_start": [loss_ratio_before[i] for i in range(BATCH_COUNT)],
        "travel_pass_reached_target": [pass_reached[i] for i in range(BATCH_COUNT)],
        "travel_pass_steps": [pass_steps[i] for i in range(BATCH_COUNT)],
    }


def run_all(seeds, max_steps, path):
    results = {"seeds": seeds, "max_steps": max_steps, "models": {}}
    for model_name in MODELS:
        results["models"][model_name] = {}
        for seed in seeds:
            started = time.time()
            run = run_one(model_name, seed, max_steps)
            results["models"][model_name][str(seed)] = run
            missed = sum(1 for flag in run["reached_target"] if not flag)
            print(f"{model_name} seed {seed}: done in {time.time() - started:.1f} s, "
                  f"chews not at target: {missed}, max steps in a chew: {max(run['steps'])}",
                  flush=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    with open(path, "w") as handle:
        json.dump(results, handle)
    return results


# ---------------------------------------------------------------- summary helpers

def other_batches_in_group(batch):
    return [j for j in range(BATCH_COUNT) if j != batch and BATCH_GROUP[j] == BATCH_GROUP[batch]]


def clean_same_group_pairs():
    skipped = {DUPLICATE_OF_ZERO, COVERED_BATCH, BAD_BATCH}
    return [(i, j) for i in range(BATCH_COUNT) for j in range(BATCH_COUNT)
            if i != j and BATCH_GROUP[i] == BATCH_GROUP[j]
            and i not in skipped and j not in skipped]


def different_group_pairs():
    return [(i, j) for i in range(BATCH_COUNT) for j in range(BATCH_COUNT)
            if BATCH_GROUP[i] != BATCH_GROUP[j]]


def mean(values):
    return sum(values) / len(values)


def mean_on_own_group(table, batch):
    return mean([table[batch][j] for j in other_batches_in_group(batch)])


def lowest_on_own_group(table):
    means = [mean_on_own_group(table, i) for i in range(BATCH_COUNT)]
    others = [means[i] for i in range(BATCH_COUNT) if i != BAD_BATCH]
    return means[BAD_BATCH] < min(others), means[BAD_BATCH], min(others)


def connected_groups(table):
    parent = list(range(BATCH_COUNT))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for i in range(BATCH_COUNT):
        for j in range(i + 1, BATCH_COUNT):
            if abs(table[i][j]) > BOUND_JOIN or abs(table[j][i]) > BOUND_JOIN:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(BATCH_COUNT):
        groups.setdefault(find(i), set()).add(i)
    return sorted(groups.values(), key=min)


def duplicate_pairs(table):
    return [(i, j) for i in range(BATCH_COUNT) for j in range(i + 1, BATCH_COUNT)
            if table[i][j] >= BOUND_DUPLICATE and table[j][i] >= BOUND_DUPLICATE]


def covering_pairs(table):
    """Ordered pairs (i, j): chewing i covers j, while j does not cover i."""
    return [(i, j) for i in range(BATCH_COUNT) for j in range(BATCH_COUNT)
            if i != j and table[i][j] >= BOUND_DUPLICATE
            and table[j][i] <= BOUND_COVERED_REVERSE]


# ---------------------------------------------------------------- predictions

def evaluate_linear(run):
    """Return {name: (holds in this seed, text of the measured numbers)}."""
    table = run["table"]
    travel = run["travel"]
    ratio = run["loss_before_over_loss_at_start"]
    out = {}

    a, b = table[0][6], table[6][0]
    out["Q1"] = (a >= BOUND_DUPLICATE and b >= BOUND_DUPLICATE,
                 f"entry(0,6)={a:.4f} entry(6,0)={b:.4f}")

    a, b = table[1][7], table[7][1]
    out["Q2"] = (a >= BOUND_DUPLICATE and b <= BOUND_COVERED_REVERSE,
                 f"entry(1,7)={a:.4f} entry(7,1)={b:.4f}")

    largest = max(abs(table[i][j]) for i, j in different_group_pairs())
    out["Q3"] = (largest <= BOUND_SEPARATE, f"largest cross-group size={largest:.3e}")

    value = mean([table[i][j] for i, j in clean_same_group_pairs()])
    out["Q4"] = (0.2 <= value <= 0.45, f"mean same-group entry={value:.4f}")

    bad_mean = mean([table[BAD_BATCH][j] for j in range(8, 13)])
    is_lowest, own, best_other = lowest_on_own_group(table)
    out["Q5"] = (bad_mean <= -0.5 and is_lowest,
                 f"mean entry(13, 8..12)={bad_mean:.4f} own-group mean of 13={own:.4f} "
                 f"lowest of the others={best_other:.4f}")

    share_6 = travel[6] / travel[0]
    share_7 = travel[7] / travel[1]
    out["Q6"] = (share_6 <= 0.01 and share_7 <= 0.01,
                 f"travel6/travel0={share_6:.3e} travel7/travel1={share_7:.3e}")

    gap_8, gap_14 = abs(ratio[8] - 1.0), abs(ratio[14] - 1.0)
    out["Q7"] = (gap_8 <= 0.001 and gap_14 <= 0.001,
                 f"|loss8_before/loss8_start-1|={gap_8:.3e} same for 14={gap_14:.3e}")

    groups = connected_groups(table)
    duplicates = duplicate_pairs(table)
    covering = covering_pairs(table)
    groups_ok = groups == PLANTED_GROUPS
    duplicates_ok = duplicates == [(0, 6)]
    covering_ok = covering == [(1, 7)]
    out["Q8"] = (groups_ok and duplicates_ok and covering_ok,
                 f"groups={[sorted(g) for g in groups]} duplicate pairs={duplicates} "
                 f"covering pairs={covering}")
    return out


def evaluate_mlp(run):
    table = run["table"]
    out = {}
    a, b = table[0][6], table[6][0]
    out["M1"] = (a >= 0.9 and b >= 0.9, f"entry(0,6)={a:.4f} entry(6,0)={b:.4f}")

    across = mean([abs(table[i][j]) for i, j in different_group_pairs()])
    inside = mean([table[i][j] for i, j in clean_same_group_pairs()])
    out["M2"] = (across <= 0.5 * inside,
                 f"mean size across groups={across:.4f} mean inside a group={inside:.4f}")

    is_lowest, own, best_other = lowest_on_own_group(table)
    out["M3"] = (is_lowest,
                 f"own-group mean of 13={own:.4f} lowest of the others={best_other:.4f}")
    return out


PREDICTION_TEXT = {
    "Q1": "duplicates: entry(0,6) and entry(6,0) both >= 0.99",
    "Q2": "covered: entry(1,7) >= 0.99 and entry(7,1) <= 0.9",
    "Q3": "separate groups: every cross-group entry at most 0.001 in size",
    "Q4": "same group (leaving out 6, 7, 13): mean entry between 0.2 and 0.45",
    "Q5": "bad batch: mean entry(13, 8..12) <= -0.5 and 13 has the lowest own-group mean of 20",
    "Q6": "travel: batch 6 <= 1% of batch 0, batch 7 <= 1% of batch 1",
    "Q7": "new group untouched: loss on 8 and on 14 before their chew within 0.1% of the start",
    "Q8": "structure recovered: groups are the planted three, duplicates exactly {0,6}, "
          "covering exactly 1 covers 7",
    "M1": "entry(0,6) and entry(6,0) both >= 0.9",
    "M2": "mean size across groups <= half of the mean entry inside a group",
    "M3": "batch 13 has the lowest own-group mean entry of all 20",
}


# ---------------------------------------------------------------- summary printing

def format_cell(value):
    """Two decimals; a diverged chew gives huge or infinite values, shown in short form."""
    if math.isfinite(value) and abs(value) < 1000:
        return f"{value:6.2f}"
    return f"{value:6.0e}" if math.isfinite(value) else ("  -inf" if value < 0 else "  +inf")


def print_mean_table(model_name, runs):
    seeds = list(runs)
    print(f"\nMean table over seeds {', '.join(seeds)}, model {model_name} "
          f"(row = batch chewed, column = batch measured)")
    print("      " + "".join(f"{j:6d}" for j in range(BATCH_COUNT)))
    for i in range(BATCH_COUNT):
        cells = []
        for j in range(BATCH_COUNT):
            if i == j:
                cells.append("     .")
            else:
                cells.append(format_cell(mean([runs[s]['table'][i][j] for s in seeds])))
        print(f"{i:4d}  " + "".join(cells) + f"  {i:4d}")
    print("      " + "".join(f"{j:6d}" for j in range(BATCH_COUNT)))


def print_not_reached(model_name, runs):
    count = sum(1 for run in runs.values() for flag in run["reached_target"] if not flag)
    total = sum(len(run["reached_target"]) for run in runs.values())
    pass_count = sum(1 for run in runs.values() for flag in run["travel_pass_reached_target"]
                     if not flag)
    print(f"{model_name}: chews that did not reach their target: {count} of {total} "
          f"(travel pass: {pass_count} of {total})")
    for seed, run in runs.items():
        missed = [i for i, flag in enumerate(run["reached_target"]) if not flag]
        if missed:
            print(f"  seed {seed}: batches {missed}")


def print_predictions(model_name, runs, evaluate):
    seeds = list(runs)
    needed = (HOLD_FRACTION_NUMERATOR * len(seeds) + HOLD_FRACTION_DENOMINATOR - 1) \
        // HOLD_FRACTION_DENOMINATOR
    per_seed = {seed: evaluate(runs[seed]) for seed in seeds}
    for name in per_seed[seeds[0]]:
        print(f"\n{name}: {PREDICTION_TEXT[name]}")
        for seed in seeds:
            flag, text = per_seed[seed][name]
            print(f"  seed {seed}: {'true ' if flag else 'false'}  {text}")
        true_count = sum(1 for seed in seeds if per_seed[seed][name][0])
        verdict = "holds" if true_count >= needed else "fails"
        print(f"{name}: {verdict}   (true in {true_count} of {len(seeds)} seeds, "
              f"{needed} needed)")


def summarise(path, label):
    with open(path) as handle:
        results = json.load(handle)
    if label:
        print(label)
    print(f"seeds {results['seeds']}, cap of {results['max_steps']} steps per chew")
    for model_name in MODELS:
        runs = results["models"][model_name]
        print(f"\n===== model {model_name} =====")
        print_mean_table(model_name, runs)
        print()
        print_not_reached(model_name, runs)
        if model_name == "mlp":
            for seed, run in runs.items():
                print(f"  seed {seed}: rates used by the chews: {sorted(set(run['rate_per_chew']))}")
        evaluate = evaluate_linear if model_name == "linear" else evaluate_mlp
        print_predictions(model_name, runs, evaluate)
    if label:
        print("\n" + label)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summarise", action="store_true",
                        help="print the summary computed from experiments/results/cross-chew.json")
    parser.add_argument("--smoke", action="store_true",
                        help="one seed, 300 steps per chew, writes cross-chew-smoke.json")
    arguments = parser.parse_args()

    if arguments.summarise:
        summarise(RESULTS_PATH, label=None)
    elif arguments.smoke:
        run_all([0], SMOKE_MAX_STEPS, SMOKE_PATH)
        summarise(SMOKE_PATH, label="SMOKE TEST ONLY: the numbers below mean nothing "
                                    "(one seed, 300 steps per chew)")
    else:
        run_all(SEEDS, MAX_STEPS, RESULTS_PATH)
        print(f"written {RESULTS_PATH}; run with --summarise to read it")


if __name__ == "__main__":
    sys.exit(main())
