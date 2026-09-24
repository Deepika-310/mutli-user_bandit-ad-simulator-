"""Ad-click environment and the simulation loop."""
from dataclasses import dataclass

import numpy as np

from bandit import make_agent


class BernoulliEnvironment:
    """K ads, each with a hidden click probability, for ``n_runs`` parallel runs.

    With ``shift_at`` set, the click rates are reversed from that step on (the best
    ad becomes the worst), the simplest change that is guaranteed to move the optimum.
    Exactly one uniform draw per run is consumed each step regardless of which arm is
    played, so two algorithms simulated with the same seed face the *same* random
    user behaviour (common random numbers). That cuts the variance of comparisons
    based on clicks; it does not help regret comparisons (see benchmark.py).
    """

    def __init__(self, probs, n_runs=1, rng=None, shift_at=None):
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 1 or probs.size < 2:
            raise ValueError("Need at least two ads")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("Click probabilities must be in [0, 1]")
        self.probs = probs
        self.n_runs = n_runs
        self.rng = rng if rng is not None else np.random.default_rng()
        self.shift_at = shift_at

    def probs_at(self, t):
        if self.shift_at is not None and t >= self.shift_at:
            return self.probs[::-1]
        return self.probs

    def step(self, arms, t):
        """Return (rewards, instantaneous pseudo-regret) for the chosen arms."""
        p = self.probs_at(t)
        rewards = (self.rng.random(self.n_runs) < p[arms]).astype(np.int8)
        return rewards, p.max() - p[arms]


@dataclass(frozen=True)
class Experiment:
    """Raw per-step history of ``n_runs`` independent runs of one algorithm.

    Arrays have shape (n_runs, steps). Everything else (cumulative regret, optimal-ad
    rate, ...) is derived from these in analysis.py.
    """

    name: str
    probs: np.ndarray
    shift_at: int | None
    arms: np.ndarray          # arm chosen at each step
    rewards: np.ndarray       # 1 = click, 0 = no click
    inst_regret: np.ndarray   # best expected click rate - chosen ad's expected click rate

    @property
    def n_runs(self):
        return self.arms.shape[0]

    @property
    def steps(self):
        return self.arms.shape[1]


def run_experiment(name, probs, steps=1000, n_runs=1, seed=None, shift_at=None,
                   agent_factory=None, **agent_params):
    """Simulate ``n_runs`` independent runs of the algorithm called ``name``.

    ``seed`` makes the whole experiment reproducible. The agent and the environment
    get independent random streams spawned from it. ``agent_factory(n_arms, n_runs,
    rng)`` can replace the built-in agent (used by the benchmark's ablations).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    agent_seq, env_seq = np.random.SeedSequence(seed).spawn(2)
    env = BernoulliEnvironment(probs, n_runs, np.random.default_rng(env_seq), shift_at)
    agent_rng = np.random.default_rng(agent_seq)
    if agent_factory is not None:
        agent = agent_factory(len(env.probs), n_runs, agent_rng)
    else:
        agent = make_agent(name, len(env.probs), n_runs, agent_rng, **agent_params)

    arms = np.empty((n_runs, steps), dtype=np.int16)
    rewards = np.empty((n_runs, steps), dtype=np.int8)
    inst_regret = np.empty((n_runs, steps))

    for t in range(steps):
        a = agent.select_arm()
        r, reg = env.step(a, t)
        agent.update(a, r)
        arms[:, t] = a
        rewards[:, t] = r
        inst_regret[:, t] = reg

    return Experiment(agent.name, env.probs, shift_at, arms, rewards, inst_regret)
