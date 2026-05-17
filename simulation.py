import numpy as np
import random
def simulate(ads, algo, steps=1000, non_stationary=False):
    rewards = []
    regret = []
    selections = []   
    total_reward = 0
    cumulative_optimal = 0

    for t in range(steps):

        if non_stationary and t == steps // 2:
            ads = [max(0, min(1, p + random.uniform(-0.2, 0.2))) for p in ads]

        optimal_mean = max(ads)
        cumulative_optimal += optimal_mean

        arm = algo.select_arm()
        selections.append(arm)   
        reward = 1 if random.random() < ads[arm] else 0
        algo.update(arm, reward)

        total_reward += reward

        rewards.append(total_reward)
        regret.append(cumulative_optimal - total_reward)

    return rewards, regret, algo.N, selections 