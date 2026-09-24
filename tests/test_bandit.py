import numpy as np
import pytest

from bandit import ALGORITHMS, UCB, EpsilonGreedy, ThompsonSampling, make_agent


def rng(seed=0):
    return np.random.default_rng(seed)


def test_sample_average_matches_mean():
    agent = EpsilonGreedy(2, rng=rng())
    rewards = [1, 0, 0, 1, 1, 0, 1]
    for r in rewards:
        agent.update(np.array([0]), np.array([r]))
    assert agent.Q[0, 0] == pytest.approx(np.mean(rewards))
    assert agent.N[0, 0] == len(rewards)
    assert agent.Q[0, 1] == 0


def test_constant_step_size_is_recency_weighted():
    agent = EpsilonGreedy(2, rng=rng(), step_size=0.5)
    for r in [1, 1, 0]:
        agent.update(np.array([0]), np.array([r]))
    # 0 -> .5 -> .75 -> .375
    assert agent.Q[0, 0] == pytest.approx(0.375)


def test_runs_are_independent():
    agent = EpsilonGreedy(3, n_runs=2, rng=rng())
    agent.update(np.array([0, 2]), np.array([1, 1]))
    assert agent.Q[0].tolist() == [1, 0, 0]
    assert agent.Q[1].tolist() == [0, 0, 1]


def test_greedy_picks_argmax():
    agent = EpsilonGreedy(3, epsilon=0.0, rng=rng())
    agent.Q[0] = [0.1, 0.9, 0.5]
    assert all(agent.select_arm()[0] == 1 for _ in range(20))


def test_greedy_tie_break_is_random_not_always_arm_zero():
    agent = EpsilonGreedy(3, n_runs=500, epsilon=0.0, rng=rng())
    picks = agent.select_arm()  # all Q equal -> tie
    assert set(np.unique(picks)) == {0, 1, 2}


def test_epsilon_one_explores_uniformly():
    agent = EpsilonGreedy(3, n_runs=6000, epsilon=1.0, rng=rng())
    agent.Q[:] = [0, 0, 1]
    share = np.bincount(agent.select_arm(), minlength=3) / 6000
    assert share == pytest.approx([1 / 3] * 3, abs=0.03)


def test_decaying_epsilon_schedule():
    agent = EpsilonGreedy(3, epsilon=0.5, decay=True, decay_scale=100)
    assert agent.current_epsilon() == pytest.approx(0.5)
    agent.t = 100
    assert agent.current_epsilon() == pytest.approx(0.25)
    agent.t = 900
    assert agent.current_epsilon() == pytest.approx(0.05)


def test_ucb_pulls_every_arm_once_first():
    agent = UCB(4, n_runs=50, rng=rng())
    seen = np.zeros((50, 4), dtype=bool)
    for _ in range(4):
        arms = agent.select_arm()
        seen[np.arange(50), arms] = True
        agent.update(arms, np.zeros(50, dtype=int))
    assert seen.all()


def test_ucb_prefers_less_explored_arm_when_means_tie():
    agent = UCB(2, rng=rng())
    agent.N[0] = [50, 5]
    agent.Q[0] = [0.5, 0.5]
    agent.t = 55
    assert agent.select_arm()[0] == 1


def test_thompson_posterior_counts():
    agent = ThompsonSampling(2, rng=rng())
    for r in [1, 1, 0]:
        agent.update(np.array([0]), np.array([r]))
    assert agent.alpha[0].tolist() == [3, 1]   # 1 + two clicks
    assert agent.beta[0].tolist() == [2, 1]    # 1 + one miss


def test_thompson_discount_forgets_toward_prior():
    agent = ThompsonSampling(2, rng=rng(), discount=0.5)
    agent.update(np.array([0]), np.array([1]))
    agent.update(np.array([1]), np.array([0]))
    assert agent.alpha[0, 0] == pytest.approx(1 + 0.5)  # 1 click, halved once


def test_thompson_concentrates_on_clearly_better_arm():
    agent = ThompsonSampling(2, n_runs=200, rng=rng())
    agent.alpha[:] = [[80, 5]]
    agent.beta[:] = [[20, 95]]
    assert (agent.select_arm() == 0).mean() > 0.99


@pytest.mark.parametrize("name", ALGORITHMS)
def test_factory_builds_every_algorithm(name):
    agent = make_agent(name, 3, 4, rng())
    arms = agent.select_arm()
    assert arms.shape == (4,) and arms.min() >= 0 and arms.max() < 3


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_agent("Nope", 3, 1, rng())


@pytest.mark.parametrize("bad", [dict(epsilon=1.5), dict(epsilon=-0.1), dict(decay_scale=0)])
def test_invalid_epsilon_greedy_params(bad):
    with pytest.raises(ValueError):
        EpsilonGreedy(3, **bad)
