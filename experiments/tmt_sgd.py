"""Plain SGD against TMT-SGD on toy tasks. The plan is docs/experiment-plan.md, the predictions
and the stop rule are experiments/PREDICTIONS.md.

    python experiments/make_digits.py                  # once: saves experiments/data/digits.npz
    python experiments/tmt_sgd.py --list               # how many runs there are, how many are done
    python experiments/tmt_sgd.py --smoke              # a few seconds on the processor, numbers mean nothing
    python experiments/tmt_sgd.py                      # run what is not done yet (default device: cpu)
    python experiments/tmt_sgd.py --device cuda        # only when the owner has cleared the card
    python experiments/tmt_sgd.py --summarise          # score the predictions from the results file

Results go to experiments/results/runs.jsonl, one line per run, so a stopped run can be
started again. Everything the summary prints is computed from that file.
"""

import argparse
import dataclasses
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import numpy
import torch

FOLDER = Path(__file__).resolve().parent
RESULTS_PATH = FOLDER / "results" / "runs.jsonl"
SMOKE_PATH = FOLDER / "results" / "smoke.jsonl"
DIGITS_PATH = FOLDER / "data" / "digits.npz"

BATCH_SIZE = 32
POOL_SIZE = 2048  # for linear and mlp; the digits pool is the training set of the file
HELD_OUT_SIZE = 4096
STEP_BUDGET = 10_000
STEP_CAP = 256  # for the chew "until"
TOLERANCE = 0.01  # "until": stop at this share of the loss the batch arrived with
LOG_EVERY_EARLY = 10  # for the first EARLY_STEPS steps
EARLY_STEPS = 500
LOG_EVERY_LATE = 250
LINEAR_INPUTS = 2000
MLP_INPUTS = 20
MLP_WIDTH = 256
TEACHER_WIDTH = 16
DIGITS_WIDTH = 128
DIGITS_INPUTS = 64
DIGIT_CLASSES = 10
TEACHER_SEED_OFFSET = 1000
BEAT_SHARE = 0.8  # "lower in at least 4 of the 5 seeds"

TASKS = ["linear", "mlp", "digits"]
NOISES = {"linear": [0.0, 0.5], "mlp": [0.0, 0.5], "digits": [0.0, 0.2]}
DATA_MODES = {"linear": ["stream", "pool"], "mlp": ["stream", "pool"], "digits": ["pool"]}
CHEWS = ["1", "4", "16", "64", "until"]
RATES = {"linear": [3.0, 1.0, 0.3], "mlp": [0.1, 0.03, 0.01], "digits": [0.3, 0.1, 0.03]}
SEEDS = [0, 1, 2, 3, 4]

# Columns of a log row: steps, examples seen, test loss, validation loss, test accuracy.
STEPS, EXAMPLES, TEST_LOSS, VALIDATION_LOSS, TEST_ACCURACY = range(5)


@dataclasses.dataclass
class Plan:
    """Everything that differs between the real experiment and the smoke test."""
    results_path: Path = RESULTS_PATH
    seeds: tuple = tuple(SEEDS)
    rates: dict = dataclasses.field(default_factory=lambda: dict(RATES))
    budget: int = STEP_BUDGET
    held_out_size: int = HELD_OUT_SIZE
    linear_inputs: int = LINEAR_INPUTS
    device: str = "cpu"
    worker_index: int = 0  # this process takes every worker_count-th run, starting here
    worker_count: int = 1


def make_smoke_plan() -> Plan:
    return Plan(results_path=SMOKE_PATH, seeds=(0,), rates={task: RATES[task][:1] for task in TASKS},
                budget=60, held_out_size=256, linear_inputs=200, device="cpu")


# ---------------------------------------------------------------- models and tasks

class LinearModel(torch.nn.Module):
    def __init__(self, input_count: int):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(input_count))

    def forward(self, inputs):
        return inputs @ self.weights


class Network(torch.nn.Module):
    def __init__(self, input_count: int, width: int, output_count: int = 1):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(input_count, width), torch.nn.Tanh(), torch.nn.Linear(width, output_count))
        self.squeeze = output_count == 1

    def forward(self, inputs):
        outputs = self.body(inputs)
        return outputs.squeeze(-1) if self.squeeze else outputs


def mean_square(model, inputs, targets):
    return ((model(inputs) - targets) ** 2).mean()


def cross_entropy(model, inputs, targets):
    return torch.nn.functional.cross_entropy(model(inputs), targets)


@dataclasses.dataclass
class Task:
    """What one run needs: the model, the loss, the held-out sets and a source of batches."""
    model: torch.nn.Module
    loss_function: object
    test: tuple
    validation: tuple
    batches: object  # function from a data mode to an endless iterator of (inputs, targets)
    is_classification: bool


def make_regression_task(task, noise, seed, plan):
    """The data come from their own generator, so no choice of model changes them."""
    generator = torch.Generator().manual_seed(seed)
    if task == "linear":
        input_count = plan.linear_inputs
        strengths = torch.arange(1, input_count + 1) ** -0.5
        strengths = strengths / strengths.norm()
        teacher_weights = torch.randn(input_count, generator=generator)

        def draw(count):
            inputs = torch.randn(count, input_count, generator=generator) * strengths
            return inputs, inputs @ teacher_weights
    else:
        input_count = MLP_INPUTS
        torch.manual_seed(seed + TEACHER_SEED_OFFSET)
        teacher = Network(input_count, TEACHER_WIDTH)

        def draw(count):
            inputs = torch.randn(count, input_count, generator=generator)
            with torch.no_grad():
                return inputs, teacher(inputs)

    test = draw(plan.held_out_size)
    validation = draw(plan.held_out_size)
    noise_size = noise * test[1].std().item()

    def draw_noisy(count):
        inputs, targets = draw(count)
        noise_draw = torch.randn(count, generator=generator)  # drawn even when noise is 0
        return inputs, targets + noise_size * noise_draw

    def batches(data_mode):
        if data_mode == "stream":
            while True:
                yield draw_noisy(BATCH_SIZE)
        pool_inputs, pool_targets = draw_noisy(POOL_SIZE)
        while True:
            order = torch.randperm(POOL_SIZE, generator=generator)
            for first in range(0, POOL_SIZE, BATCH_SIZE):
                chosen = order[first:first + BATCH_SIZE]
                yield pool_inputs[chosen], pool_targets[chosen]

    torch.manual_seed(seed)
    model = LinearModel(input_count) if task == "linear" else Network(input_count, MLP_WIDTH)
    return Task(model, mean_square, test, validation, batches, False)


def wrong_labels(labels, share, generator):
    """Replace each label, with probability `share`, by a different digit chosen at random."""
    replaced = torch.rand(len(labels), generator=generator) < share
    shift = torch.randint(1, DIGIT_CLASSES, (len(labels),), generator=generator)  # drawn even when share is 0
    return torch.where(replaced, (labels + shift) % DIGIT_CLASSES, labels)


def make_digits_task(noise, seed):
    data = numpy.load(DIGITS_PATH)
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.from_numpy(data["train_inputs"])
    train_labels = wrong_labels(torch.from_numpy(data["train_labels"]), noise, generator)
    test = (torch.from_numpy(data["test_inputs"]), torch.from_numpy(data["test_labels"]))
    validation = (torch.from_numpy(data["validation_inputs"]), torch.from_numpy(data["validation_labels"]))

    def batches(data_mode):
        assert data_mode == "pool", "digits has no stream mode"
        while True:
            order = torch.randperm(len(train_labels), generator=generator)
            for first in range(0, len(order), BATCH_SIZE):
                chosen = order[first:first + BATCH_SIZE]
                yield train_inputs[chosen], train_labels[chosen]

    torch.manual_seed(seed)
    model = Network(DIGITS_INPUTS, DIGITS_WIDTH, DIGIT_CLASSES)
    return Task(model, cross_entropy, test, validation, batches, True)


def make_task(task, noise, seed, plan):
    if task == "digits":
        return make_digits_task(noise, seed)
    return make_regression_task(task, noise, seed, plan)


# ---------------------------------------------------------------- one run

def to_device(pair, device):
    return pair[0].to(device), pair[1].to(device)


def test_accuracy(model, inputs, labels):
    return (model(inputs).argmax(dim=1) == labels).float().mean().item()


def log_row(task, step_count, example_count):
    with torch.no_grad():
        test_loss = task.loss_function(task.model, *task.test).item()
        validation_loss = task.loss_function(task.model, *task.validation).item()
        accuracy = test_accuracy(task.model, *task.test) if task.is_classification else None
    return [step_count, example_count, test_loss, validation_loss, accuracy]


def is_log_step(step_count):
    return step_count % LOG_EVERY_LATE == 0 or (
        step_count < EARLY_STEPS and step_count % LOG_EVERY_EARLY == 0)


def run(task_name, noise, data_mode, chew, rate, seed, plan):
    """Train one model. Returns the record of the run, without its key."""
    started = time.perf_counter()
    task = make_task(task_name, noise, seed, plan)
    task.model.to(plan.device)
    task.test = to_device(task.test, plan.device)
    task.validation = to_device(task.validation, plan.device)
    optimiser = torch.optim.SGD(task.model.parameters(), lr=rate)
    step_cap = STEP_CAP if chew == "until" else int(chew)
    step_count = example_count = chew_count = 0
    log, rises, previous, diverged = [], [], None, False
    source = task.batches(data_mode)
    while step_count < plan.budget and not diverged:
        inputs, targets = to_device(next(source), plan.device)
        example_count += len(inputs)
        chew_count += 1
        if previous is not None:
            with torch.no_grad():
                before = task.loss_function(task.model, *previous).item()
        arrival_loss = None
        for _ in range(step_cap):
            loss = task.loss_function(task.model, inputs, targets)
            value = loss.item()
            if not math.isfinite(value):
                diverged = True
                break
            if arrival_loss is None:
                arrival_loss = value
            elif chew == "until" and value <= TOLERANCE * arrival_loss:
                break
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step_count += 1
            if is_log_step(step_count):
                log.append(log_row(task, step_count, example_count))
                diverged = not math.isfinite(log[-1][TEST_LOSS])
            if step_count >= plan.budget or diverged:
                break
        if previous is not None and not diverged:
            with torch.no_grad():
                rises.append(task.loss_function(task.model, *previous).item() - before)
        previous = (inputs, targets)
    return {"task": task_name, "noise": noise, "data_mode": data_mode, "chew": chew, "rate": rate,
            "seed": seed, "diverged": diverged, "log": log, "steps": step_count,
            "examples": example_count, "chews": chew_count,
            "mean_steps_per_chew": step_count / chew_count,
            "mean_rise": statistics.fmean(rises) if rises else None,
            "seconds": time.perf_counter() - started}


# ---------------------------------------------------------------- the list of runs and the results file

def make_key(task, noise, data_mode, chew, rate, seed):
    return f"{task}|{noise}|{data_mode}|{chew}|{rate}|{seed}"


def all_runs(plan):
    """Every run of the plan, as (task, noise, data_mode, chew, rate, seed)."""
    for task in TASKS:
        for noise, data_mode, chew, seed in itertools.product(
                NOISES[task], DATA_MODES[task], CHEWS, plan.seeds):
            for rate in plan.rates[task]:
                yield task, noise, data_mode, chew, rate, seed


def worker_path(path, worker_index) -> Path:
    return path.with_name(f"{path.stem}-w{worker_index}.jsonl")


def read_results(path) -> dict:
    """Every finished run: the results file and the files of the workers beside it."""
    records = {}
    for file in [path] + sorted(path.parent.glob(f"{path.stem}-w*.jsonl")):
        if file.exists():
            for line in filter(None, file.read_text(encoding="utf-8").splitlines()):
                record = json.loads(line)
                records[record["key"]] = record
    return records


def count_runs(plan):
    done = read_results(plan.results_path)
    keys = [make_key(*one_run) for one_run in all_runs(plan)]
    return len(keys), sum(key in done for key in keys)


def print_list(plan) -> None:
    total, finished = count_runs(plan)
    print(f"runs in the plan: {total}")
    for task in TASKS:
        print(f"  {task}: {sum(1 for one_run in all_runs(plan) if one_run[0] == task)}")
    print(f"runs done: {finished}")
    print(f"runs left: {total - finished}")


def run_all(plan) -> None:
    plan.results_path.parent.mkdir(exist_ok=True)
    done = read_results(plan.results_path)
    # Several workers can share the card: each takes its own share of the runs and writes
    # its own file, so no two processes write to one file.
    write_path = (plan.results_path if plan.worker_count == 1
                  else worker_path(plan.results_path, plan.worker_index))
    for index, one_run in enumerate(all_runs(plan)):
        key = make_key(*one_run)
        if index % plan.worker_count != plan.worker_index or key in done:
            continue
        record = run(*one_run, plan)
        record["key"] = key
        with write_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        last = record["log"][-1][TEST_LOSS] if record["log"] else float("nan")
        print(f"{key}  test loss {last:.5f}  diverged {record['diverged']}  "
              f"{record['seconds']:.1f}s", flush=True)


# ---------------------------------------------------------------- the summary

def final_loss(record, column):
    """The last logged value of a run; the worst possible if it diverged."""
    if record["diverged"] or not record["log"] or not math.isfinite(record["log"][-1][column]):
        return math.inf
    return record["log"][-1][column]


def test_loss_at_examples(record, example_count):
    """Test loss at the first logged moment with at least this many examples seen."""
    if record["diverged"]:
        return math.inf
    for row in record["log"]:
        if row[EXAMPLES] >= example_count:
            return row[TEST_LOSS]
    return record["log"][-1][TEST_LOSS]


def choose_rate(results, plan, task, noise, data_mode, chew):
    """(rate, runs) for the rate with the lowest median validation loss; None if runs are missing."""
    choices = []
    for rate in plan.rates[task]:
        runs = [results.get(make_key(task, noise, data_mode, chew, rate, seed)) for seed in plan.seeds]
        if not all(runs):
            return None
        median = statistics.median(final_loss(record, VALIDATION_LOSS) for record in runs)
        choices.append((median, rate, runs))
    _, rate, runs = min(choices, key=lambda choice: choice[0])
    return rate, runs


def all_settings():
    for task in TASKS:
        for noise, data_mode in itertools.product(NOISES[task], DATA_MODES[task]):
            yield task, noise, data_mode


def choose_everything(results, plan):
    return {(task, noise, data_mode): {
                chew: choose_rate(results, plan, task, noise, data_mode, chew) for chew in CHEWS}
            for task, noise, data_mode in all_settings()}


def final_test_losses(runs):
    return [final_loss(record, TEST_LOSS) for record in runs]


def mean_of(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else float("nan")


def print_table(setting, chosen) -> None:
    task, noise, data_mode = setting
    print(f"\n{task}, noise {noise}, {data_mode}")
    if any(entry is None for entry in chosen.values()):
        print("  not all runs are done")
        return
    accuracy_header = "  median test accuracy" if task == "digits" else ""
    print(f"  chew  rate   median test loss  examples seen  mean rise  steps per chew{accuracy_header}")
    for chew, (rate, runs) in chosen.items():
        line = (f"  {chew:>5}  {rate:<5}  {statistics.median(final_test_losses(runs)):<16.6f}  "
                f"{mean_of(record['examples'] for record in runs):<13.0f}  "
                f"{mean_of(record['mean_rise'] for record in runs):<9.6f}  "
                f"{mean_of(record['mean_steps_per_chew'] for record in runs):<14.2f}")
        if task == "digits":
            accuracies = [record["log"][-1][TEST_ACCURACY] for record in runs if record["log"]]
            line += f"  {statistics.median(accuracies):.4f}" if accuracies else "  nan"
        print(line)


def needed_seeds(plan) -> int:
    return math.ceil(BEAT_SHARE * len(plan.seeds))


def lower_count(losses_a, losses_b) -> int:
    return sum(a < b for a, b in zip(losses_a, losses_b))


def losses_of(chosen_setting, chew):
    entry = chosen_setting[chew]
    return None if entry is None else final_test_losses(entry[1])


def say_count(plan, name_a, name_b, count, how) -> None:
    print(f"  {how}, {name_a} is lower than {name_b} in {count} of {len(plan.seeds)} seeds")


def chews_beating_plain(plan, chosen, setting, other_chews, how):
    """Print, for each chew, in how many seeds it is lower than chew 1. True if any beats it."""
    plain = losses_of(chosen[setting], "1")
    beaten = False
    for chew in other_chews:
        losses = losses_of(chosen[setting], chew)
        count = lower_count(losses, plain)
        say_count(plan, f"chew {chew}", "chew 1", count, f"{setting[0]} noise {setting[1]} {setting[2]} {how}")
        beaten = beaten or count >= needed_seeds(plan)
    return beaten


def plain_beats(plan, chosen, setting, chew_names, how):
    """Print, for each chew, in how many seeds chew 1 is lower. True if chew 1 beats all of them."""
    plain = losses_of(chosen[setting], "1")
    beats_all = True
    for chew in chew_names:
        count = lower_count(plain, losses_of(chosen[setting], chew))
        say_count(plan, "chew 1", f"chew {chew}", count, f"{setting[0]} noise {setting[1]} {setting[2]} {how}")
        beats_all = beats_all and count >= needed_seeds(plan)
    return beats_all


def complete(chosen, settings) -> bool:
    return all(entry is not None for setting in settings for entry in chosen[setting].values())


def verdict(name, all_done, holds) -> None:
    print(f"{name}: " + ("not all runs are done" if not all_done else "holds" if holds else "fails"))


def equal_example_losses(chosen_setting):
    """Per seed: chew 1's test loss at the examples chew 64 saw in total, and chew 64's final test loss."""
    plain_runs, chewed_runs = chosen_setting["1"][1], chosen_setting["64"][1]
    plain = [test_loss_at_examples(plain_run, chewed_run["examples"])
             for plain_run, chewed_run in zip(plain_runs, chewed_runs)]
    return final_test_losses(chewed_runs), plain


def score_p1(plan, chosen) -> None:
    settings = [("linear", 0.0, "stream"), ("mlp", 0.0, "stream")]
    print("\nP1, per gradient step: clean labels, new data")
    if not complete(chosen, settings):
        return verdict("P1", False, False)
    beaten = [chews_beating_plain(plan, chosen, setting, CHEWS[1:], "at equal steps") for setting in settings]
    verdict("P1", True, not any(beaten))


def score_p2(plan, chosen) -> None:
    print("\nP2, per example: clean labels, new data, equal examples seen")
    results = {}
    for task in ["linear", "mlp"]:
        setting = (task, 0.0, "stream")
        if not complete(chosen, [setting]):
            results[task] = None
            continue
        chewed, plain = equal_example_losses(chosen[setting])
        count = lower_count(chewed, plain)
        say_count(plan, "chew 64", "chew 1", count, f"{task} at equal examples seen")
        results[task] = count >= needed_seeds(plan)
    verdict("P2", results["linear"] is not None, results["linear"])
    verdict("P2 (mlp, reported apart)", results["mlp"] is not None, results["mlp"])


def score_p3(plan, chosen) -> None:
    settings = [("linear", 0.5, "stream"), ("mlp", 0.5, "stream")]
    print("\nP3, noise: noisy labels, new data")
    if not complete(chosen, settings):
        return verdict("P3", False, False)
    plain_wins = [plain_beats(plan, chosen, setting, ["64", "until"], "at equal steps") for setting in settings]
    verdict("P3", True, all(plain_wins))


def rise_per_seed(runs):
    return [record["mean_rise"] for record in runs]


def score_p4(plan, chosen) -> None:
    settings = [(task, 0.0, data_mode) for task in TASKS for data_mode in DATA_MODES[task]]
    print("\nP4, forgetting: clean labels, mean rise in the loss on the previous batch")
    if not complete(chosen, settings):
        return verdict("P4", False, False)
    larger_everywhere = True
    for task, noise, data_mode in settings:
        rises = {chew: rise_per_seed(chosen[(task, noise, data_mode)][chew][1]) for chew in ["1", "64"]}
        pairs = [(big, small) for big, small in zip(rises["64"], rises["1"])
                 if big is not None and small is not None]
        count = sum(big > small for big, small in pairs)
        mean_64, mean_1 = mean_of(rises["64"]), mean_of(rises["1"])
        print(f"  {task} {data_mode}: rise is larger for chew 64 than for chew 1 in {count} of "
              f"{len(plan.seeds)} seeds; means {mean_64:.6f} and {mean_1:.6f}")
        larger_everywhere = larger_everywhere and mean_64 > mean_1
    verdict("P4", True, larger_everywhere)


def score_p5(plan, chosen) -> None:
    clean_pools = [(task, 0.0, "pool") for task in TASKS]
    noisy_digits = ("digits", 0.2, "pool")
    print("\nP5, a fixed pool and the digits (a guess)")
    if not complete(chosen, clean_pools + [noisy_digits]):
        return verdict("P5", False, False)
    beaten = [chews_beating_plain(plan, chosen, setting, CHEWS[1:], "at equal steps") for setting in clean_pools]
    plain_wins = plain_beats(plan, chosen, noisy_digits, ["64"], "at equal steps")
    verdict("P5", True, not any(beaten) and plain_wins)


def print_stop_rule(chosen, plan) -> None:
    print("\nStop rule: is the best rate of plain SGD (chew 1) at an end of its list?")
    for task in TASKS:
        edges = []
        for setting in all_settings():
            entry = chosen[setting]["1"] if setting[0] == task else None
            if entry is not None:
                edges.append((setting, entry[0] in (plan.rates[task][0], plan.rates[task][-1])))
        for (_, noise, data_mode), at_end in edges:
            print(f"  {task} noise {noise} {data_mode}: {at_end}")
        if edges:
            print(f"  {task}: at an end in {sum(flag for _, flag in edges)} of {len(edges)} settings")
        else:
            print(f"  {task}: not all runs are done")


def summarise(plan) -> None:
    results = read_results(plan.results_path)
    total, finished = count_runs(plan)
    print(f"results file: {plan.results_path}; runs read: {len(results)}; "
          f"runs of the plan done: {finished} of {total}")
    chosen = choose_everything(results, plan)
    for setting in all_settings():
        print_table(setting, chosen[setting])
    for score in (score_p1, score_p2, score_p3, score_p4, score_p5):
        score(plan, chosen)
    print_stop_rule(chosen, plan)


def smoke() -> None:
    plan = make_smoke_plan()
    plan.results_path.parent.mkdir(exist_ok=True)
    plan.results_path.unlink(missing_ok=True)
    run_all(plan)
    print("\n=== SMOKE TEST SUMMARY: one seed, 60 steps, the numbers mean nothing ===")
    summarise(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda, only when cleared")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--list", action="store_true", help="count the runs, run nothing")
    parser.add_argument("--smoke", action="store_true", help="a tiny run on the processor")
    parser.add_argument("--worker", default="0/1", metavar="INDEX/COUNT",
                        help="take every COUNT-th run starting at INDEX, e.g. 0/4 ... 3/4 for four workers")
    arguments = parser.parse_args()
    worker_index, worker_count = map(int, arguments.worker.split("/"))
    if worker_count > 1:
        torch.set_num_threads(1)  # the workers share the processor too
    plan = Plan(device=arguments.device, worker_index=worker_index, worker_count=worker_count)
    if arguments.smoke:
        smoke()
    elif arguments.list:
        print_list(plan)
    elif arguments.summarise:
        summarise(plan)
    else:
        run_all(plan)


if __name__ == "__main__":
    main()
