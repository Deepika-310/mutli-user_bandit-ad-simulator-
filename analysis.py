"""Metrics and plain-English interpretation derived from an Experiment."""
import numpy as np

LATE_FRACTION = 0.2  # "late phase" = last 20% of the steps


def cumulative_regret(exp):
    return np.cumsum(exp.inst_regret, axis=1)


def cumulative_reward(exp):
    return np.cumsum(exp.rewards, axis=1, dtype=np.int64)


def optimal_choice(exp):
    """Boolean (runs, steps): did the agent play a best ad at this step?"""
    return exp.inst_regret <= 1e-12


def mean_ci(x):
    """Mean over runs (axis 0) and the half-width of a 95% normal confidence interval."""
    mean = x.mean(axis=0)
    if x.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, 1.96 * x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])


def rolling_mean(x, window):
    """Trailing moving average along the last axis (expanding window at the start)."""
    window = max(1, min(window, x.shape[-1]))
    c = np.cumsum(x, axis=-1, dtype=float)
    out = c / np.arange(1, x.shape[-1] + 1)
    out[..., window:] = (c[..., window:] - c[..., :-window]) / window
    return out


def arm_counts(exp):
    """(runs, n_arms): how many times each ad was shown in each run."""
    return (exp.arms[..., None] == np.arange(len(exp.probs))).sum(axis=1)


def _late_slice(steps):
    return slice(steps - max(1, int(steps * LATE_FRACTION)), steps)


def random_play_regret(exp):
    """Regret of showing ads uniformly at random for the whole run (the baseline)."""
    return exp.steps * (exp.probs.max() - exp.probs.mean())


def converged_runs(exp):
    """Boolean per run: was the best ad shown to at least half the users in the late phase?"""
    return optimal_choice(exp)[:, _late_slice(exp.steps)].mean(axis=1) >= 0.5


def steps_to_converge(exp, start=0, threshold=0.9):
    """Per run, users needed after ``start`` until the best ad is shown >= threshold of the time.

    "Shown" is measured with a rolling window of max(20, 2% of the remaining steps).
    Returns -1 for runs that never get there. Use start=shift_at to time recovery.
    """
    opt = optimal_choice(exp)[:, start:]
    window = max(20, opt.shape[1] // 50)
    rate = rolling_mean(opt, window)
    rate[:, : window - 1] = 0.0  # ignore the warm-up where the window isn't full yet
    hit = rate >= threshold
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def summarize(experiments):
    """One row of headline numbers per experiment."""
    rows = []
    for exp in experiments:
        final_regret = cumulative_regret(exp)[:, -1]
        _, ci = mean_ci(final_regret[:, None])
        late_opt = optimal_choice(exp)[:, _late_slice(exp.steps)].mean(axis=1)
        baseline = random_play_regret(exp)
        avoided = 1.0 - final_regret.mean() / baseline if baseline > 0 else float("nan")
        rows.append({
            "Algorithm": exp.name,
            "Final regret": round(float(final_regret.mean()), 1),
            "± 95% CI": round(float(ci[0]), 1),
            "Regret avoided vs random": f"{avoided:.0%}",
            "Converged runs": f"{converged_runs(exp).mean():.0%}",
            "Best ad in last 20%": f"{late_opt.mean():.0%}",
            "Avg clicks": round(float(cumulative_reward(exp)[:, -1].mean()), 1),
        })
    return rows


def interpret(exp):
    """Rule-based reading of one experiment as a list of (level, message).

    Levels are "success", "info" or "warning". Thresholds are relative to the
    problem (random-play regret rate, optimal-ad rate), not hard-coded click counts.
    """
    T = exp.steps
    late = _late_slice(T)
    early = slice(0, max(1, int(T * LATE_FRACTION)))
    out = []

    rate = exp.inst_regret.mean(axis=0)  # mean per-step regret across runs
    baseline = exp.probs.max() - exp.probs.mean()  # per-step regret of picking ads uniformly
    late_rate, early_rate = rate[late].mean(), rate[early].mean()
    late_opt = optimal_choice(exp)[:, late].mean(axis=1)  # per run

    if baseline > 0:
        ratio = late_rate / baseline
        if ratio < 0.05:
            out.append(("success", f"Late-phase regret is only {late_rate:.3f} per user "
                                   f"({ratio:.0%} of random play): it has essentially converged."))
        elif ratio < 0.25:
            out.append(("info", f"Late-phase regret is {late_rate:.3f} per user "
                                f"({ratio:.0%} of random play): low but not zero, which is "
                                "what constant exploration costs."))
        else:
            out.append(("warning", f"Late-phase regret is still {late_rate:.3f} per user "
                                   f"({ratio:.0%} of random play): learning has not settled."))
        if early_rate > 0 and late_rate < 0.5 * early_rate:
            out.append(("info", "Regret per user fell substantially after the early phase "
                                "(the curve bends toward flat)."))

    stuck = float((late_opt < 0.5).mean())
    if exp.n_runs > 1 and stuck > 0:
        out.append(("warning", f"In {stuck:.0%} of runs the algorithm was still mostly "
                               "showing a sub-optimal ad at the end (premature lock-in "
                               "or failure to re-adapt)."))
    elif exp.n_runs > 1:
        out.append(("success", "Every run ended up mostly showing the best ad."))

    if exp.shift_at is not None:
        after = late_opt.mean()
        if after > 0.8:
            out.append(("success", "It recovered after the click rates shifted."))
        else:
            out.append(("warning", "After the click rates shifted it kept favouring the old "
                                   "winner. Try a constant step size (or a discount for "
                                   "Thompson Sampling)."))
    return out
