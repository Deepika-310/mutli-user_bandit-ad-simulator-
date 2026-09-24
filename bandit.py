"""Multi-armed bandit agents.

Every agent is *batched*: one object drives ``n_runs`` independent bandit problems
at once, so its state arrays have shape (n_runs, n_arms). That lets the simulator
average over many seeds with a handful of NumPy calls per step instead of a
Python loop per run.
"""
import numpy as np

ALGORITHMS = ["Epsilon-Greedy", "Decaying Epsilon", "UCB1", "Thompson Sampling"]


class Bandit:
    """Base agent: keeps Q (value estimates), N (pull counts) and the clock t.

    Subclasses only implement ``_select``. With ``step_size=None`` the update is the
    sample average (Q = mean of all rewards, right for stationary arms). A constant
    ``step_size`` gives an exponential recency-weighted average, which can track arms
    whose click rates drift.
    """

    name = "Bandit"

    def __init__(self, n_arms, n_runs=1, rng=None, step_size=None):
        if n_arms < 1 or n_runs < 1:
            raise ValueError("n_arms and n_runs must be >= 1")
        if step_size is not None and not 0 < step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        self.n_arms = n_arms
        self.n_runs = n_runs
        self.rng = rng if rng is not None else np.random.default_rng()
        self.step_size = step_size
        self.Q = np.zeros((n_runs, n_arms))
        self.N = np.zeros((n_runs, n_arms))
        self.t = 0  # number of pulls made so far in each run
        self._rows = np.arange(n_runs)

    def select_arm(self):
        arms = self._select()
        self.t += 1
        return arms

    def _select(self):
        raise NotImplementedError

    def update(self, arms, rewards):
        r = self._rows
        self.N[r, arms] += 1
        alpha = 1.0 / self.N[r, arms] if self.step_size is None else self.step_size
        self.Q[r, arms] += alpha * (rewards - self.Q[r, arms])

    def load_stats(self, pulls, clicks):
        """Rebuild the agent's state from per-arm impression and click counts.

        Lets a stateless service reconstruct the agent from SQL aggregates. Only valid
        for the sample-average update (step_size=None), since a constant step size
        depends on the order of rewards, not just the counts. Single-run agents only.
        """
        pulls = np.asarray(pulls, dtype=float)
        clicks = np.asarray(clicks, dtype=float)
        if self.n_runs != 1 or self.step_size is not None:
            raise ValueError("load_stats needs n_runs=1 and the sample-average update")
        if pulls.shape != (self.n_arms,) or clicks.shape != (self.n_arms,):
            raise ValueError("pulls and clicks must have one entry per arm")
        if np.any(clicks < 0) or np.any(clicks > pulls):
            raise ValueError("need 0 <= clicks <= pulls for every arm")
        self.N[0] = pulls
        self.Q[0] = np.divide(clicks, pulls, out=np.zeros(self.n_arms), where=pulls > 0)
        self.t = int(pulls.sum())

    def _argmax(self, values):
        """Row-wise argmax with *random* tie-breaking.

        np.argmax always returns the lowest index on ties, which silently favours
        arm 0 while every estimate is still 0.
        """
        is_best = values == values.max(axis=1, keepdims=True)
        noise = self.rng.random(values.shape)
        return np.argmax(np.where(is_best, noise, -1.0), axis=1)


class EpsilonGreedy(Bandit):
    """Explore uniformly with probability eps, otherwise exploit argmax(Q).

    ``decay=True`` uses eps_t = epsilon / (1 + t / decay_scale): near ``epsilon`` at
    the start and ~1/t later, with decay_scale controlling how long exploration lasts.
    """

    def __init__(self, n_arms, n_runs=1, rng=None, epsilon=0.1, decay=False,
                 decay_scale=50.0, step_size=None):
        super().__init__(n_arms, n_runs, rng, step_size)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if decay_scale <= 0:
            raise ValueError("decay_scale must be > 0")
        self.epsilon = epsilon
        self.decay = decay
        self.decay_scale = decay_scale
        self.name = "Decaying Epsilon" if decay else "Epsilon-Greedy"

    def current_epsilon(self):
        if self.decay:
            return self.epsilon / (1.0 + self.t / self.decay_scale)
        return self.epsilon

    def _select(self):
        explore = self.rng.random(self.n_runs) < self.current_epsilon()
        random_arms = self.rng.integers(0, self.n_arms, size=self.n_runs)
        return np.where(explore, random_arms, self._argmax(self.Q))


class UCB(Bandit):
    """UCB1: pick argmax of Q + sqrt(c * ln(t) / N).

    The bonus is large for rarely pulled arms and shrinks as N grows. Untried arms
    get an infinite bonus, so each arm is pulled once before the formula takes over.
    c = 2 is the classic UCB1 constant.
    """

    name = "UCB1"

    def __init__(self, n_arms, n_runs=1, rng=None, c=2.0, step_size=None):
        super().__init__(n_arms, n_runs, rng, step_size)
        if c < 0:
            raise ValueError("c must be >= 0")
        self.c = c

    def _select(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = np.sqrt(self.c * np.log(max(self.t, 1)) / self.N)
        bonus = np.where(self.N == 0, np.inf, bonus)
        return self._argmax(self.Q + bonus)


class ThompsonSampling(Bandit):
    """Bayesian bandit: sample a click rate from each arm's Beta posterior, play the max.

    Beta(1, 1) prior; each click adds 1 to alpha, each miss adds 1 to beta.
    ``discount < 1`` shrinks the evidence towards the prior on every step so old
    data is gradually forgotten (useful when click rates drift).
    """

    name = "Thompson Sampling"

    def __init__(self, n_arms, n_runs=1, rng=None, discount=1.0):
        super().__init__(n_arms, n_runs, rng)
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        self.discount = discount
        self.alpha = np.ones((n_runs, n_arms))
        self.beta = np.ones((n_runs, n_arms))

    def _select(self):
        return np.argmax(self.rng.beta(self.alpha, self.beta), axis=1)

    def load_stats(self, pulls, clicks):
        if self.discount < 1.0:
            raise ValueError("load_stats cannot rebuild a discounted posterior")
        super().load_stats(pulls, clicks)
        self.alpha[0] = 1.0 + np.asarray(clicks, dtype=float)
        self.beta[0] = 1.0 + np.asarray(pulls, dtype=float) - np.asarray(clicks, dtype=float)

    def update(self, arms, rewards):
        super().update(arms, rewards)
        if self.discount < 1.0:
            self.alpha = 1.0 + self.discount * (self.alpha - 1.0)
            self.beta = 1.0 + self.discount * (self.beta - 1.0)
        r = self._rows
        self.alpha[r, arms] += rewards
        self.beta[r, arms] += 1 - rewards


def make_agent(name, n_arms, n_runs, rng, *, epsilon=0.1, decay_scale=50.0,
               ucb_c=2.0, step_size=None, discount=1.0):
    """Build an agent by its display name (the names in ALGORITHMS)."""
    if name == "Epsilon-Greedy":
        return EpsilonGreedy(n_arms, n_runs, rng, epsilon=epsilon, step_size=step_size)
    if name == "Decaying Epsilon":
        return EpsilonGreedy(n_arms, n_runs, rng, epsilon=epsilon, decay=True,
                             decay_scale=decay_scale, step_size=step_size)
    if name == "UCB1":
        return UCB(n_arms, n_runs, rng, c=ucb_c, step_size=step_size)
    if name == "Thompson Sampling":
        return ThompsonSampling(n_arms, n_runs, rng, discount=discount)
    raise ValueError(f"Unknown algorithm: {name!r}")
