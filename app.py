from pathlib import Path

import streamlit as st

from analysis import interpret, summarize
from bandit import ALGORITHMS
from plots import (
    color_map,
    optimal_rate_plot,
    pull_share_plot,
    regret_plot,
    reward_plot,
    selection_plot,
)
from simulation import run_experiment

st.set_page_config(page_title="Bandit Ad Simulator", layout="wide")

st.title("Ad Selection with Multi-Armed Bandits")
st.markdown(
    "Which ad should you show when you don't know which one people click? "
    "Compare four bandit algorithms on a simulated ad-click problem, averaged over "
    "many independent runs so you can tell real differences from luck."
)

with st.expander("How the simulation works"):
    st.markdown("""
Each ad has a **hidden click probability**. For every user, the algorithm picks an ad,
the user clicks with that ad's probability, and the algorithm updates its estimate.

- **Exploration** = showing an ad to learn about it. **Exploitation** = showing the ad that looks best.
- **Regret** = expected clicks lost versus always showing the best ad. A flattening regret curve means the algorithm has learned.
- Every setting is run over **many seeds**; lines show the mean and the shaded band a 95% confidence interval.
- **Changing environment** reverses the click rates halfway through (best ad becomes worst) to test adaptation.
""")

# ---- controls ----------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")
    probs_text = st.text_input("True click rates (comma-separated)", "0.3, 0.5, 0.7")
    steps = st.slider("Users per run", 100, 10000, 2000, step=100)
    n_runs = st.slider("Independent runs", 1, 200, 50)
    seed = st.number_input("Random seed", 0, 1_000_000, 42)
    shifting = st.checkbox("Changing environment (rates reverse halfway)")

    st.header("Algorithms")
    chosen = st.multiselect("Compare", ALGORITHMS, default=ALGORITHMS)
    epsilon = st.slider("Epsilon (initial epsilon for Decaying)", 0.0, 1.0, 0.1, 0.01)
    decay_scale = st.slider("Decay scale (Decaying Epsilon)", 10, 1000, 50)
    ucb_c = st.slider("UCB exploration constant c", 0.0, 4.0, 2.0, 0.1)

    with st.expander("Adaptation to change (advanced)"):
        step_size = st.slider("Constant step size for Epsilon/UCB (0 = sample average)",
                              0.0, 0.5, 0.0, 0.01)
        discount = st.slider("Thompson Sampling discount (1 = never forget)",
                             0.90, 1.0, 1.0, 0.005)

    run_clicked = st.button("Run simulation", type="primary", width="stretch")


def parse_probs(text):
    try:
        probs = [float(x) for x in text.split(",")]
    except ValueError:
        return None, "Click rates must be numbers separated by commas."
    if not 2 <= len(probs) <= 10:
        return None, "Enter between 2 and 10 click rates."
    if any(p < 0 or p > 1 for p in probs):
        return None, "Each click rate must be between 0 and 1."
    return probs, None


# ---- run ---------------------------------------------------------------------------

if run_clicked:
    probs, error = parse_probs(probs_text)
    if error:
        st.error(error)
    elif not chosen:
        st.error("Select at least one algorithm.")
    else:
        params = dict(epsilon=epsilon, decay_scale=decay_scale, ucb_c=ucb_c,
                      step_size=step_size or None, discount=discount)
        shift_at = steps // 2 if shifting else None
        with st.spinner("Simulating..."):
            experiments = [run_experiment(name, probs, steps, n_runs, int(seed), shift_at,
                                          **params) for name in chosen]
        # Keep results in session_state so moving a widget doesn't wipe the charts
        st.session_state["results"] = dict(
            experiments=experiments, probs=probs, steps=steps, n_runs=n_runs,
            shifting=shifting)

# ---- results -----------------------------------------------------------------------

results = st.session_state.get("results")
if results is None:
    st.info("Set things up in the sidebar and press **Run simulation**.")
    st.stop()

experiments = results["experiments"]
colors = color_map([e.name for e in experiments])
st.caption(
    f"Showing: {results['n_runs']} runs x {results['steps']} users, click rates "
    f"{results['probs']}" + (", rates reversed halfway" if results["shifting"] else "")
    + ". Change settings and press Run again to update."
)

st.subheader("Summary")
st.dataframe(summarize(experiments), hide_index=True, width="stretch")

tab_regret, tab_best, tab_reward, tab_ads, tab_read, tab_bench = st.tabs(
    ["Regret", "Best-ad rate", "Clicks", "Ad selection", "Interpretation", "Benchmark report"])

with tab_regret:
    st.pyplot(regret_plot(experiments, colors))
    st.caption("Regret = expected clicks lost vs. always showing the best ad. "
               "Linear growth = never stops exploring or got stuck; flattening = converged.")
with tab_best:
    st.pyplot(optimal_rate_plot(experiments, colors))
with tab_reward:
    st.pyplot(reward_plot(experiments, colors))
with tab_ads:
    st.pyplot(pull_share_plot(experiments, colors))
    pick = st.selectbox("Trace one run for", [e.name for e in experiments])
    exp = next(e for e in experiments if e.name == pick)
    run = st.number_input("Run number", 1, exp.n_runs, 1) - 1
    st.pyplot(selection_plot(exp, run))
with tab_read:
    for exp in experiments:
        st.markdown(f"**{exp.name}**")
        for level, message in interpret(exp):
            {"success": st.success, "warning": st.warning}.get(level, st.info)(message)
with tab_bench:
    report = Path(__file__).parent / "results" / "benchmark.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("Run `python benchmark.py` to generate the benchmark report.")
