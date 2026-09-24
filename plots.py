"""Matplotlib figures for the Streamlit app.

Figures are built with the object-oriented ``Figure`` API rather than ``pyplot``:
pyplot keeps global state that is not thread-safe (Streamlit serves each session in
its own thread) and leaks figures across reruns.
"""
import numpy as np
from matplotlib.figure import Figure

from analysis import (
    arm_counts,
    cumulative_regret,
    cumulative_reward,
    mean_ci,
    optimal_choice,
    rolling_mean,
)

PALETTE = ["#2f6fdb", "#e8833a", "#2a9d6f", "#b04fc4", "#d1495b", "#6b7280"]


def color_map(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def _axes(title, xlabel, ylabel):
    fig = Figure(figsize=(7, 3.6), layout="constrained")
    ax = fig.subplots()
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def _mark_shift(ax, exp):
    if exp.shift_at is not None:
        ax.axvline(exp.shift_at, color="#888", ls="--", lw=1)
        ax.annotate("click rates shift", (exp.shift_at, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color="#666")


def _band_plot(experiments, colors, curve, title, ylabel):
    fig, ax = _axes(title, "Users shown an ad", ylabel)
    for exp in experiments:
        mean, ci = mean_ci(curve(exp))
        x = np.arange(1, exp.steps + 1)
        ax.plot(x, mean, color=colors[exp.name], lw=1.8, label=exp.name)
        ax.fill_between(x, mean - ci, mean + ci, color=colors[exp.name], alpha=0.18, lw=0)
    _mark_shift(ax, experiments[0])
    ax.legend(frameon=False, fontsize=9)
    return fig


def regret_plot(experiments, colors):
    return _band_plot(experiments, colors, cumulative_regret,
                      "Cumulative regret (lower is better)", "Regret")


def reward_plot(experiments, colors):
    return _band_plot(experiments, colors, cumulative_reward,
                      "Cumulative clicks (higher is better)", "Clicks")


def optimal_rate_plot(experiments, colors):
    window = max(10, experiments[0].steps // 20)
    fig, ax = _axes(f"How often the best ad is shown (rolling {window}-user window)",
                    "Users shown an ad", "Share of users")
    for exp in experiments:
        mean, ci = mean_ci(rolling_mean(optimal_choice(exp), window))
        x = np.arange(1, exp.steps + 1)
        ax.plot(x, mean, color=colors[exp.name], lw=1.8, label=exp.name)
        ax.fill_between(x, np.clip(mean - ci, 0, 1), np.clip(mean + ci, 0, 1),
                        color=colors[exp.name], alpha=0.18, lw=0)
    ax.set_ylim(0, 1.02)
    _mark_shift(ax, experiments[0])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return fig


def selection_plot(exp, run=0):
    """Which ad was shown to each user in a single run."""
    fig, ax = _axes(f"{exp.name}: ad shown to each user (run {run + 1})",
                    "Users shown an ad", "Ad")
    ax.scatter(np.arange(1, exp.steps + 1), exp.arms[run], s=4, alpha=0.6,
               color=PALETTE[0], linewidths=0)
    ax.set_yticks(range(len(exp.probs)))
    ax.set_yticklabels([f"Ad {i} (p={p:.2f})" for i, p in enumerate(exp.probs)])
    ax.set_ylim(-0.5, len(exp.probs) - 0.5)
    _mark_shift(ax, exp)
    return fig


def pull_share_plot(experiments, colors):
    """Average share of impressions each ad received, per algorithm."""
    n_arms = len(experiments[0].probs)
    fig, ax = _axes("Share of impressions per ad (averaged over runs)", "", "Share of users")
    width = 0.8 / len(experiments)
    for i, exp in enumerate(experiments):
        share = arm_counts(exp).mean(axis=0) / exp.steps
        ax.bar(np.arange(n_arms) + i * width, share, width, label=exp.name,
               color=colors[exp.name])
    ax.set_xticks(np.arange(n_arms) + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels([f"Ad {i}\n(p={p:.2f})" for i, p in enumerate(experiments[0].probs)])
    ax.legend(frameon=False, fontsize=9)
    return fig
