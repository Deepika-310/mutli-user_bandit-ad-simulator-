import streamlit as st
from bandit import EpsilonGreedy, UCB
from simulation import simulate
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bandit Simulator", layout="centered")

# TITLE + PURPOSE


st.title("Ad Selection using Bandit Algorithms")

st.markdown("""
**Goal:** Understand how machines make decisions under uncertainty.

We simulate a system that:
- Learns which ad performs best
- Balances **exploration (trying new ads)** vs **exploitation (using best ad)**

We measure:
- **Cumulative Reward** → total clicks gained  
- **Regret** → missed opportunity compared to optimal choice  
""")


# HOW IT WORKS


with st.expander("How the simulation works"):
    st.markdown("""
1. Initialize ads with unknown success probabilities (e.g., [0.3, 0.5, 0.7])  
2. At each step:
   - Algorithm selects an ad  
   - User clicks (or not) based on probability  
3. Algorithm updates its belief  
4. Process repeats over many users  

Over time → the system should learn the best ad  
""")


#  USER INPUT


algo_choice = st.selectbox(
    "Choose an Algorithm",
    ["Epsilon-Greedy", "Decaying Epsilon", "UCB"]
)

epsilon = st.slider("Epsilon (exploration rate)", 0.0, 1.0, 0.1)
steps = st.slider("Number of Users", 100, 5000, 1000)
non_stationary = st.checkbox("Enable changing environment")

ads = [0.3, 0.5, 0.7]


#  RUN SIMULATION


if st.button("Run the Simulation"):

    if algo_choice == "Epsilon-Greedy":
        algo = EpsilonGreedy(3, epsilon)

    elif algo_choice == "Decaying Epsilon":
        algo = EpsilonGreedy(3, epsilon, decay=True)

    else:
        algo = UCB(3)

    rewards, regret, counts = simulate(ads, algo, steps, non_stationary)

    
    # RESULTS
    

    st.subheader(" Results")

    st.write("Ad Selection Counts:", counts)

    fig1, ax1 = plt.subplots()
    ax1.plot(rewards)
    ax1.set_title("Cumulative Reward")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(regret)
    ax2.set_title("Regret")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Regret")
    st.pyplot(fig2)

    
    # FINAL METRICS
    

    st.subheader(" Final Metrics")

    st.write(f"Total reward after {steps} users: **{rewards[-1]}**")
    st.write(f"Final regret: **{int(regret[-1])}**")

    
    # INTELLIGENT INTERPRETATION
    

    st.subheader(" Trial Interpretation")

    reward_growth_rate = (rewards[-1] - rewards[0]) / len(rewards)
    regret_growth_rate = (regret[-1] - regret[0]) / len(regret)

    last_100_reward = rewards[-1] - rewards[-100] if len(rewards) > 100 else rewards[-1]
    last_100_regret = regret[-1] - regret[-100] if len(regret) > 100 else regret[-1]

    # Convergence insight
    if last_100_regret < 10:
        st.write("The algorithm has mostly converged — very few mistakes recently.")
    elif regret_growth_rate > 0.4:
        st.write("The algorithm is still making frequent mistakes — weak convergence.")

    # Reward insight
    if last_100_reward > 50:
        st.write(" Strong reward gain in the final phase — effective exploitation.")
    elif reward_growth_rate < 0.2:
        st.write(" Slow reward growth — too much exploration or poor learning.")

    # Overall judgment
    if last_100_regret < 10 and last_100_reward > 50:
        st.success(" Overall: Good balance between exploration and exploitation.")
    else:
        st.warning(" Overall: Needs tuning (epsilon / algorithm choice).")

    
    # TREND ANALYSIS (KEY UPGRADE)
    

    st.subheader(" Trend Insight")

    if len(regret) > 200 and last_100_regret < (regret[100] * 0.2):
        st.write(" Regret growth has slowed → algorithm is learning effectively.")

    if len(rewards) > 200 and last_100_reward > reward_growth_rate * 100:
        st.write(" Reward accumulation is accelerating → strong ad identified.")
