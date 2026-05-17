import numpy as np
import random

class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.Q = np.zeros(n_arms)  # estimated rewards
        self.N = np.zeros(n_arms)  # counts

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class EpsilonGreedy(Bandit):
    def __init__(self, n_arms, epsilon=0.1, decay=False):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.decay = decay
        self.t = 1

    def select_arm(self):
        eps = self.epsilon / self.t if self.decay else self.epsilon
        self.t += 1

        if random.random() < eps:
            return random.randint(0, self.n_arms - 1)
        return np.argmax(self.Q)


class UCB(Bandit):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.t = 1

    def select_arm(self):
        self.t += 1
        ucb_values = self.Q + np.sqrt(2 * np.log(self.t) / (self.N + 1e-5))
        return np.argmax(ucb_values)