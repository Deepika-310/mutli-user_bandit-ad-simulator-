"""Reproducible benchmark: how much better is each algorithm, and did the v2 changes help?

    python benchmark.py                 # full run, writes results/benchmark.md + .csv
    python benchmark.py --quick         # small smoke run (used in CI)

Part 1 compares every algorithm on four scenarios against a uniform-random baseline.
Part 2 measures the effect of each engineering change made in v2 (ablations).
Everything is seeded; only the timing table varies between machines.
"""
import argparse
import csv
import dataclasses
import platform
import time
from pathlib import Path

import numpy as np

from analysis import (
    converged_runs,
    cumulative_regret,
    cumulative_reward,
    mean_ci,
    steps_to_converge,
)
from bandit import EpsilonGreedy
from simulation import run_experiment

# label, algorithm name, parameters
BASELINE = "Uniform random (baseline)"
REFERENCE = "Epsilon-Greedy (e=0.1)"
ALGOS = [
    (BASELINE, "Epsilon-Greedy", dict(epsilon=1.0)),
    (REFERENCE, "Epsilon-Greedy", dict(epsilon=0.1)),
    ("Decaying Epsilon", "Decaying Epsilon", dict(epsilon=0.1)),
    ("UCB1 (c=2)", "UCB1", dict(ucb_c=2.0)),
    ("UCB1 (c=0.25)", "UCB1", dict(ucb_c=0.25)),
    ("Thompson Sampling", "Thompson Sampling", {}),
]
ADAPTIVE = [  # only meaningful when click rates change
    ("Epsilon-Greedy (e=0.1, step 0.1)", "Epsilon-Greedy", dict(epsilon=0.1, step_size=0.1)),
    ("UCB1 (c=0.25, step 0.1)", "UCB1", dict(ucb_c=0.25, step_size=0.1)),
    ("Thompson Sampling (discount 0.99)", "Thompson Sampling", dict(discount=0.99)),
]
SCENARIOS = [
    dict(key="easy", title="Easy: 3 ads with clear gaps", probs=[0.3, 0.5, 0.7],
         steps=5000, shift=False),
    dict(key="low_ctr", title="Realistic: 4 ads with 2-5% click rates",
         probs=[0.02, 0.03, 0.04, 0.05], steps=10000, shift=False),
    dict(key="many_arms", title="Many ads: 10 ads, small gaps",
         probs=[round(float(p), 2) for p in np.linspace(0.10, 0.55, 10)], steps=10000, shift=False),
    dict(key="shift", title="Changing world: rates reverse halfway",
         probs=[0.3, 0.5, 0.7], steps=4000, shift=True),
]


# ---- Part 1: algorithm comparison -------------------------------------------------------

def run_scenario(scenario, n_runs, seed, steps_scale=1.0):
    steps = max(200, int(scenario["steps"] * steps_scale))
    shift_at = steps // 2 if scenario["shift"] else None
    algos = ALGOS + (ADAPTIVE if scenario["shift"] else [])
    experiments = []
    for label, name, params in algos:
        exp = run_experiment(name, scenario["probs"], steps, n_runs, seed, shift_at, **params)
        experiments.append(dataclasses.replace(exp, name=label))
    return experiments


def scenario_rows(scenario, experiments):
    """One row per algorithm; 'vs' columns are relative to the baseline/reference."""
    final = {e.name: cumulative_regret(e)[:, -1] for e in experiments}
    random_regret = final[BASELINE].mean()
    ref_regret = final[REFERENCE].mean()
    rows = []
    for exp in experiments:
        reg = final[exp.name]
        _, ci = mean_ci(reg[:, None])
        start = exp.shift_at or 0
        t = steps_to_converge(exp, start=start)
        reached = t[t >= 0]
        rows.append({
            "Scenario": scenario["title"],
            "Algorithm": exp.name,
            "Final regret": round(float(reg.mean()), 1),
            "+/- 95% CI": round(float(ci[0]), 1),
            "Less regret than random": f"{1 - reg.mean() / random_regret:.0%}",
            "Less regret than e-greedy": f"{1 - reg.mean() / ref_regret:+.0%}",
            "Best ad found (runs)": f"{converged_runs(exp).mean():.0%}",
            "Median users to converge": int(np.median(reached)) if reached.size else "never",
        })
    return rows


# ---- Part 2: ablations of the v2 changes --------------------------------------------------

class _FirstIndexTies:
    """v1 behaviour: np.argmax returns the lowest index on ties."""

    def _argmax(self, values):
        return np.argmax(values, axis=1)


class _V1Decay:
    """v1 schedule: epsilon / t."""

    def current_epsilon(self):
        return self.epsilon / (self.t + 1)


class _V1Full(_FirstIndexTies, _V1Decay, EpsilonGreedy):
    pass


class _V1DecayRandomTies(_V1Decay, EpsilonGreedy):
    pass


def ablation_decay(n_runs, seed):
    """v1 -> v2 for Decaying Epsilon, changing one thing at a time."""
    probs, steps = [0.3, 0.5, 0.7], 2000
    variants = [
        ("v1: eps/t, lowest-index ties", lambda k, r, g: _V1Full(k, r, g, epsilon=0.1, decay=True)),
        ("+ random tie-breaking", lambda k, r, g: _V1DecayRandomTies(k, r, g, epsilon=0.1, decay=True)),
        ("+ new schedule eps/(1+t/50)  (= v2)",
         lambda k, r, g: EpsilonGreedy(k, r, g, epsilon=0.1, decay=True, decay_scale=50)),
    ]
    rows = []
    for label, factory in variants:
        exp = run_experiment(label, probs, steps, n_runs, seed, agent_factory=factory)
        reg = cumulative_regret(exp)[:, -1]
        _, ci = mean_ci(reg[:, None])
        rows.append({"Variant": label, "Runs locked onto a bad ad": f"{1 - converged_runs(exp).mean():.1%}",
                     "Final regret": f"{reg.mean():.1f} +/- {ci[0]:.1f}"})
    return rows


def ablation_regret_estimator(n_runs, seed):
    """Realised regret (v1) vs pseudo-regret (v2) for the same runs."""
    probs, steps = [0.3, 0.5, 0.7], 2000
    exp = run_experiment("Epsilon-Greedy", probs, steps, n_runs, seed, epsilon=0.1)
    realised = np.cumsum(max(probs) - exp.rewards, axis=1)
    pseudo = cumulative_regret(exp)
    sd_real, sd_pseudo = realised[:, -1].std(ddof=1), pseudo[:, -1].std(ddof=1)
    return [{
        "Estimator": "Realised (v1): best rate - actual click",
        "Std of final regret": f"{sd_real:.1f}",
        "95% CI half-width (50 runs)": f"{1.96 * sd_real / np.sqrt(50):.1f}",
        "Runs dipping below 0": f"{(realised.min(axis=1) < 0).mean():.0%}",
    }, {
        "Estimator": "Pseudo (v2): best rate - shown ad's rate",
        "Std of final regret": f"{sd_pseudo:.1f}",
        "95% CI half-width (50 runs)": f"{1.96 * sd_pseudo / np.sqrt(50):.1f}",
        "Runs dipping below 0": f"{(pseudo.min(axis=1) < 0).mean():.0%}",
    }], sd_real / sd_pseudo


def ablation_common_random_numbers(n_runs, seed):
    """Precision of 'Thompson minus e-greedy' with shared vs independent user randomness.

    Tested on two metrics: total clicks (depends on the click draws) and final
    pseudo-regret (depends only on which ads were chosen).
    """
    probs, steps = [0.3, 0.5, 0.7], 2000

    def run(name, s, **p):
        return run_experiment(name, probs, steps, n_runs, s, **p)

    ts = run("Thompson Sampling", seed)
    eps_same = run("Epsilon-Greedy", seed)      # same user randomness as ts
    eps_other = run("Epsilon-Greedy", seed + 1)  # independent user randomness
    metrics = {
        "Total clicks": lambda e: cumulative_reward(e)[:, -1].astype(float),
        "Final pseudo-regret": lambda e: cumulative_regret(e)[:, -1],
    }
    rows, ratios = [], {}
    for label, f in metrics.items():
        d_same, d_other = f(ts) - f(eps_same), f(ts) - f(eps_other)
        ratios[label] = d_other.var(ddof=1) / d_same.var(ddof=1)
        half = lambda d: 1.96 * d.std(ddof=1) / np.sqrt(len(d))  # noqa: E731
        rows.append({
            "Metric": label,
            "CI half-width, independent randomness": f"{half(d_other):.2f}",
            "CI half-width, shared randomness (v2)": f"{half(d_same):.2f}",
            "Variance reduction": f"{ratios[label]:.2f}x",
        })
    return rows, ratios


def ablation_vectorisation(seed, loop_runs=20, batch_runs=200):
    """Per-run cost of one Python loop per run vs one batched loop for all runs."""
    steps = 2000
    t0 = time.perf_counter()
    for i in range(loop_runs):
        run_experiment("Thompson Sampling", [0.3, 0.5, 0.7], steps, 1, seed + i)
    loop_per_run = (time.perf_counter() - t0) / loop_runs
    t0 = time.perf_counter()
    run_experiment("Thompson Sampling", [0.3, 0.5, 0.7], steps, batch_runs, seed)
    batch_per_run = (time.perf_counter() - t0) / batch_runs
    return [{
        "Approach": "One run at a time (loop over runs)",
        "Time per run (ms)": f"{loop_per_run * 1000:.1f}",
    }, {
        "Approach": f"Batched: {batch_runs} runs in one loop (v2)",
        "Time per run (ms)": f"{batch_per_run * 1000:.2f}",
    }], loop_per_run / batch_per_run


# ---- output --------------------------------------------------------------------------------

def md_table(rows):
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def build_report(n_runs, seed, quick):
    scale = 0.25 if quick else 1.0
    ab_runs = 100 if quick else 1000
    out = ["# Benchmark results", ""]
    out.append(f"Seed {seed}, {n_runs} runs per algorithm per scenario, Python "
               f"{platform.python_version()}, NumPy {np.__version__}."
               + (" **Quick mode: reduced size, numbers are indicative only.**" if quick else ""))
    out.append("Regret = expected clicks lost vs always showing the best ad (lower is better). "
               "'Best ad found' = share of runs where the best ad got >= 50% of impressions in the "
               "last 20% of users. 'Users to converge' = users until the best ad is shown >= 90% of "
               "the time (median over runs that get there; after the shift in the last scenario).")
    out.append("")
    all_rows = []
    out += ["## Part 1: algorithm comparison", ""]
    for sc in SCENARIOS:
        exps = run_scenario(sc, n_runs, seed, scale)
        rows = scenario_rows(sc, exps)
        all_rows += rows
        out += [f"### {sc['title']}",
                f"click rates {sc['probs']}, {exps[0].steps} users"
                + (f", rates reverse at user {exps[0].shift_at}" if sc["shift"] else ""), "",
                md_table([{k: v for k, v in r.items() if k != "Scenario"} for r in rows]), ""]

    out += ["## Part 2: what each v2 change bought", ""]
    rows = ablation_decay(ab_runs, seed)
    out += [f"### Decaying Epsilon: original vs fixed ({ab_runs} runs, easy scenario, 2000 users)", "",
            md_table(rows), ""]
    rows, ratio = ablation_regret_estimator(ab_runs, seed)
    out += [f"### Regret estimator ({ab_runs} runs)", "", md_table(rows), "",
            f"Pseudo-regret has {ratio:.1f}x lower spread, so the same confidence needs "
            f"{ratio ** 2:.1f}x fewer runs.", ""]
    rows, ratios = ablation_common_random_numbers(ab_runs, seed)
    out += [f"### Common random numbers: Thompson minus e-greedy ({ab_runs} runs)", "",
            md_table(rows), "",
            "Sharing the simulated users' randomness clearly helps click-based comparisons. "
            "For regret-based ones the effect is small (about 1.0-1.3x in my runs), because "
            "pseudo-regret ignores the click draws and depends only on the ads chosen.", ""]
    rows, speedup = ablation_vectorisation(seed)
    out += ["### Vectorising across runs (timing, varies by machine)", "", md_table(rows), "",
            f"Batching is {speedup:.0f}x faster per run.", ""]
    return "\n".join(out), all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true", help="small run for smoke testing")
    args = parser.parse_args()
    if args.quick:
        args.runs = min(args.runs, 20)

    report, rows = build_report(args.runs, args.seed, args.quick)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark.md").write_text(report, encoding="utf-8")
    with open(out_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(report)
    print(f"\nWrote {out_dir / 'benchmark.md'} and {out_dir / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
