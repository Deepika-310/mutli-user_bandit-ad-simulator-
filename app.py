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

# USER INPUT


algo_choice = st.selectbox(
    "Choose an Algorithm",
    ["Epsilon-Greedy", "Decaying Epsilon", "UCB"]
)

epsilon = st.slider("Epsilon (exploration rate)", 0.0, 1.0, 0.1)
steps = st.slider("Number of Users", 100, 5000, 1000)
non_stationary = st.checkbox("Enable changing environment")

ads = [0.3, 0.5, 0.7]


# RUN SIMULATION


if st.button("Run the Simulation"):

    if algo_choice == "Epsilon-Greedy":
        algo = EpsilonGreedy(3, epsilon)

    elif algo_choice == "Decaying Epsilon":
        algo = EpsilonGreedy(3, epsilon, decay=True)

    else:
        algo = UCB(3)

    rewards, regret, counts, selections = simulate(ads, algo, steps, non_stationary)

    
    # RESULTS
    

    st.subheader("Results")

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

    
    # AD SELECTION GRAPH (NEW)
    

    st.subheader("Ad Selection Over Time")

    fig3, ax3 = plt.subplots()
    ax3.plot(selections, alpha=0.6)
    ax3.set_title("Which Ad is Selected Over Time")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Ad Index (0,1,2)")
    st.pyplot(fig3)

   
    # FINAL METRICS
   

    st.subheader("Final Metrics")

    st.write(f"Total reward after {steps} users: **{rewards[-1]}**")
    st.write(f"Final regret: **{int(regret[-1])}**")

    st.subheader("Trial Interpretation")

    reward_growth_rate = (rewards[-1] - rewards[0]) / len(rewards)
    regret_growth_rate = (regret[-1] - regret[0]) / len(regret)

    last_100_reward = rewards[-1] - rewards[-100] if len(rewards) > 100 else rewards[-1]
    last_100_regret = regret[-1] - regret[-100] if len(regret) > 100 else regret[-1]

    if last_100_regret < 10:
        st.write("In the final phase, the algorithm is making very few mistakes — it has likely identified the best ad.")
    elif regret_growth_rate > 0.4:
        st.write("Regret is still increasing quickly, meaning the algorithm hasn't settled on a good strategy yet.")

    if last_100_reward > 50:
        st.write("The system is gaining strong rewards toward the end — it is confidently exploiting a good ad.")
    elif reward_growth_rate < 0.2:
        st.write("Reward growth is slow, suggesting too much exploration or difficulty identifying the best ad.")

    if last_100_regret < 10 and last_100_reward > 50:
        st.success("Overall, the algorithm found a good balance and learned effectively.")
    else:
        st.warning("The algorithm may need tuning (try changing epsilon or algorithm type).")

    
    #INSIGHT

    st.subheader("Trend Insight")

    if len(regret) > 200 and last_100_regret < (regret[100] * 0.2):
        st.write("Mistakes are slowing down over time — learning is stabilizing.")

    if len(rewards) > 200 and last_100_reward > reward_growth_rate * 100:
        st.write("Reward accumulation is picking up speed — a strong ad is being favored.")

    #SELECTION

    st.subheader("Selection Behavior")

    last_200 = selections[-200:] if len(selections) > 200 else selections
    dominant_ad = max(set(last_200), key=last_200.count)
    dominance_ratio = last_200.count(dominant_ad) / len(last_200)

    if dominance_ratio > 0.8:
        st.success(f"The algorithm has clearly settled on Ad {dominant_ad} ({int(dominance_ratio*100)}% of recent choices).")
    elif dominance_ratio > 0.5:
        st.write(f"Ad {dominant_ad} is preferred, but the algorithm is still testing others occasionally.")
    else:
        st.warning("No single ad dominates yet — the algorithm is still exploring heavily.")