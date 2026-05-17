import matplotlib.pyplot as plt

def plot_results(rewards, regret):
    plt.figure()
    plt.plot(rewards)
    plt.title("Cumulative Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.show()

    plt.figure()
    plt.plot(regret)
    plt.title("Regret Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Regret")
    plt.show()