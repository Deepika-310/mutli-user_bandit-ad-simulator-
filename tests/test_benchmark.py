import numpy as np
import pytest

import benchmark
from analysis import converged_runs, random_play_regret, steps_to_converge, summarize
from simulation import run_experiment

PROBS = [0.3, 0.5, 0.7]


def test_random_play_regret_formula():
    exp = run_experiment("Epsilon-Greedy", PROBS, 100, 2, seed=0)
    assert random_play_regret(exp) == pytest.approx(100 * (0.7 - 0.5))


def test_uniform_random_baseline_has_random_play_regret():
    exp = run_experiment("Epsilon-Greedy", PROBS, 2000, 50, seed=0, epsilon=1.0)
    final = exp.inst_regret.sum(axis=1).mean()
    assert final == pytest.approx(random_play_regret(exp), rel=0.05)


def test_summary_reports_regret_avoided_for_a_good_learner():
    exp = run_experiment("Thompson Sampling", PROBS, 2000, 30, seed=0)
    row = summarize([exp])[0]
    assert int(row["Regret avoided vs random"].rstrip("%")) > 90
    assert row["Converged runs"] == "100%"


def test_converged_runs_flags_a_greedy_agent_stuck_on_a_bad_ad():
    # epsilon=0 and v1-style first-index ties can lock onto arm 0 forever
    exp = run_experiment("Epsilon-Greedy", PROBS, 1000, 200, seed=0, epsilon=0.0)
    assert 0 < converged_runs(exp).mean() < 1


def test_steps_to_converge_marks_never_and_orders_algorithms():
    random_agent = run_experiment("Epsilon-Greedy", PROBS, 1000, 20, seed=0, epsilon=1.0)
    assert (steps_to_converge(random_agent) == -1).all()
    learner = run_experiment("Thompson Sampling", PROBS, 3000, 20, seed=0)
    t = steps_to_converge(learner)
    assert (t >= 0).all() and np.median(t) < 1500


def test_steps_to_converge_measures_recovery_after_a_shift():
    exp = run_experiment("Epsilon-Greedy", PROBS, 2000, 30, seed=0, shift_at=1000, step_size=0.1)
    t = steps_to_converge(exp, start=1000)
    assert (t >= 0).mean() > 0.8 and np.median(t[t >= 0]) < 500


def test_quick_benchmark_is_reproducible_and_complete():
    first, rows = benchmark.build_report(n_runs=10, seed=3, quick=True)
    second, _ = benchmark.build_report(n_runs=10, seed=3, quick=True)

    def strip_timing(text):  # the vectorisation table is wall-clock time
        return text.split("### Vectorising")[0]

    assert strip_timing(first) == strip_timing(second)
    scenarios = {r["Scenario"] for r in rows}
    assert len(scenarios) == len(benchmark.SCENARIOS)
    assert "Part 2" in first and "Decaying Epsilon: original vs fixed" in first
    baseline_rows = [r for r in rows if r["Algorithm"] == benchmark.BASELINE]
    assert all(r["Less regret than random"] == "0%" for r in baseline_rows)
