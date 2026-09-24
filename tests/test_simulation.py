import numpy as np
import pytest

from analysis import (
    arm_counts,
    cumulative_regret,
    cumulative_reward,
    mean_ci,
    rolling_mean,
    summarize,
)
from bandit import ALGORITHMS
from simulation import BernoulliEnvironment, run_experiment

PROBS = [0.3, 0.5, 0.7]


def test_same_seed_is_reproducible():
    a = run_experiment("UCB1", PROBS, 300, 5, seed=7)
    b = run_experiment("UCB1", PROBS, 300, 5, seed=7)
    assert np.array_equal(a.arms, b.arms) and np.array_equal(a.rewards, b.rewards)


def test_different_seed_differs():
    a = run_experiment("Thompson Sampling", PROBS, 300, 5, seed=1)
    b = run_experiment("Thompson Sampling", PROBS, 300, 5, seed=2)
    assert not np.array_equal(a.arms, b.arms)


def test_common_random_numbers_across_algorithms():
    # One uniform draw per run per step, whatever arm is played: with equal click rates,
    # the reward stream must not depend on the arms chosen.
    def rewards_for(arm):
        env = BernoulliEnvironment([0.5, 0.5], 4, np.random.default_rng(3))
        return [env.step(np.full(4, arm), t)[0] for t in range(20)]
    assert np.array_equal(rewards_for(0), rewards_for(1))


def test_pseudo_regret_is_nonnegative_and_monotone():
    exp = run_experiment("Epsilon-Greedy", PROBS, 500, 4, seed=0)
    reg = cumulative_regret(exp)
    assert (exp.inst_regret >= 0).all()
    assert (np.diff(reg, axis=1) >= 0).all()


def test_regret_matches_arm_gaps():
    exp = run_experiment("Epsilon-Greedy", PROBS, 400, 3, seed=0)
    gaps = np.array(PROBS).max() - np.array(PROBS)
    assert np.allclose(exp.inst_regret, gaps[exp.arms])


def test_shift_reverses_click_rates():
    env = BernoulliEnvironment(PROBS, shift_at=10)
    assert env.probs_at(9).tolist() == PROBS
    assert env.probs_at(10).tolist() == PROBS[::-1]


def test_shift_changes_which_arm_is_optimal():
    exp = run_experiment("Epsilon-Greedy", PROBS, 100, 1, seed=0, shift_at=50)
    before = exp.arms[0, :50]
    after = exp.arms[0, 50:]
    gaps_before = 0.7 - np.array(PROBS)[before]
    gaps_after = 0.7 - np.array(PROBS[::-1])[after]
    assert np.allclose(exp.inst_regret[0, :50], gaps_before)
    assert np.allclose(exp.inst_regret[0, 50:], gaps_after)


def test_rewards_are_binary_and_cumulative_reward_matches():
    exp = run_experiment("UCB1", PROBS, 200, 2, seed=3)
    assert set(np.unique(exp.rewards)) <= {0, 1}
    assert cumulative_reward(exp)[:, -1].tolist() == exp.rewards.sum(axis=1).tolist()


def test_arm_counts_sum_to_steps():
    exp = run_experiment("Decaying Epsilon", PROBS, 250, 3, seed=0)
    assert (arm_counts(exp).sum(axis=1) == 250).all()


@pytest.mark.parametrize("name", ["Epsilon-Greedy", "UCB1", "Thompson Sampling"])
def test_learners_mostly_play_best_ad(name):
    exp = run_experiment(name, PROBS, 3000, 30, seed=0)
    share = arm_counts(exp).mean(axis=0) / 3000
    assert share.argmax() == 2 and share[2] > 0.7


def test_thompson_beats_random_play_by_a_wide_margin():
    exp = run_experiment("Thompson Sampling", PROBS, 3000, 30, seed=0)
    random_play_regret = 3000 * (0.7 - np.mean(PROBS))
    assert cumulative_regret(exp)[:, -1].mean() < 0.15 * random_play_regret


def test_fixed_epsilon_regret_grows_linearly():
    exp = run_experiment("Epsilon-Greedy", PROBS, 4000, 50, seed=0, epsilon=0.2)
    mean_reg = cumulative_regret(exp).mean(axis=0)
    first_half = mean_reg[1999] - mean_reg[999]
    second_half = mean_reg[3999] - mean_reg[2999]
    assert second_half == pytest.approx(first_half, rel=0.25)  # constant slope


def test_thompson_regret_flattens():
    exp = run_experiment("Thompson Sampling", PROBS, 4000, 50, seed=0)
    mean_reg = cumulative_regret(exp).mean(axis=0)
    assert (mean_reg[3999] - mean_reg[2999]) < 0.5 * (mean_reg[999] - mean_reg[0])


def test_constant_step_size_helps_after_a_shift():
    kwargs = dict(steps=2000, n_runs=40, seed=0, shift_at=1000)
    slow = run_experiment("Epsilon-Greedy", PROBS, **kwargs)
    fast = run_experiment("Epsilon-Greedy", PROBS, step_size=0.1, **kwargs)
    assert cumulative_regret(fast)[:, -1].mean() < 0.5 * cumulative_regret(slow)[:, -1].mean()


def test_rolling_mean_and_ci_helpers():
    x = np.array([[0, 1, 1, 1, 0, 0]], dtype=float)
    assert rolling_mean(x, 2)[0].tolist() == pytest.approx([0, 0.5, 1, 1, 0.5, 0])
    mean, ci = mean_ci(np.array([[1.0], [3.0]]))
    assert mean[0] == 2.0 and ci[0] > 0
    assert mean_ci(np.array([[5.0]]))[1][0] == 0


@pytest.mark.parametrize("bad", [[0.5], [0.2, 1.4], [[0.1, 0.2]]])
def test_environment_validates_probs(bad):
    with pytest.raises(ValueError):
        BernoulliEnvironment(bad)


def test_summarize_has_one_row_per_experiment():
    exps = [run_experiment(n, PROBS, 100, 3, seed=0) for n in ALGORITHMS]
    assert [r["Algorithm"] for r in summarize(exps)] == [e.name for e in exps]
