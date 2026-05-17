Multi-Armed Bandit Ad Simulator

A simple interactive project to understand how machines make decisions under uncertainty using multi-armed bandit algorithms.

Our objective is to simulate how a system learns to choose the best ad over time by balancing:

Exploration → trying different ads
Exploitation → selecting the best-known ad
Algorithms : Epsilon-Greedy → fixed exploration
Decaying Epsilon → exploration reduces over time
UCB (Upper Confidence Bound) → uses uncertainty to guide exploration

How the System works
Ads have hidden success probabilities (e.g., [0.3, 0.5, 0.7])
At each step:
Algorithm selects an ad
A click is simulated
Algorithm updates its estimates
Over time → it learns the best ad

Observations and Analysis 
Cumulative Reward → total clicks (higher is better)
Regret → missed opportunities (lower is better)
Ad Selection Trend → shows learning behavior over time

Run the Project
git clone https://github.com/your-username/bandit-ad-simulator.git
cd bandit-ad-simulator
pip install streamlit numpy matplotlib
streamlit run app.py

Key Insight

Good algorithms:

Increase reward steadily
Reduce regret growth
Converge to the best ad over time
Future Scope
Add Thompson Sampling
Use real-world datasets
Extend to contextual bandits (personalization)
Add accuracy metrics for best ad selection
 Structure
app.py          # UI (Streamlit)
simulation.py   # Simulation logic
bandit.py       # Algorithms

This project focuses on understanding learning behavior, not just implementation — similar to how real recommendation systems evolve over time.