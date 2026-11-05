"""
Phase 2: the ratchet put. How much does locking in gains cost, and how does
the price depend on the ratchet percentage and the reset schedule?

Produces three figures in outputs/:
  1. ratchet_vs_ratio.png     -- price vs ratchet percentage, monthly resets
  2. ratchet_vs_frequency.png -- price vs reset frequency, for three ratios
  3. ratchet_sample_path.png  -- one simulated path with its ratcheting strike

Run from the repo root:  python scripts/phase2_ratchet.py
"""

import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pricers.lookback import lookback_put_closed_form
from pricers.ratchet import ratchet_put_tree, ratchet_put_mc, strike_path

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

S0, K0, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.25, 1.0
N_STEPS = 252   # daily grid; reset counts must divide it (1, 2, 4, 12, 252 do; 52 does not)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def bs_put(S, K):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def main():
    vanilla = bs_put(S0, K0)
    lookback_cont = lookback_put_closed_form(S0, r, q, sigma, T)
    print(f"Vanilla ATM put (ratio = 0):           {vanilla:.4f}")
    print(f"Continuous lookback put (ceiling):     {lookback_cont:.4f}\n")

    # ------------------------------------------- 1. price vs ratchet percentage
    ratios = np.linspace(0.0, 1.0, 21)
    tree_px = [ratchet_put_tree(S0, K0, k, r, q, sigma, T, N_STEPS, 12) for k in ratios]
    mc_ratios = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    mc_px, mc_se = [], []
    print("Monthly resets, price by ratchet percentage (tree n=252 / MC 200k paths):")
    for k in mc_ratios:
        p, se = ratchet_put_mc(S0, K0, k, r, q, sigma, T, N_STEPS, 12, 200_000, seed=1)
        mc_px.append(p); mc_se.append(se)
        tp = ratchet_put_tree(S0, K0, k, r, q, sigma, T, N_STEPS, 12)
        print(f"  ratio={k:4.2f}  tree={tp:7.4f}  MC={p:7.4f} +/- {se:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(vanilla, color=MUTED, lw=1.2, ls="--")
    ax.text(0.01, vanilla + 0.15, f"vanilla ATM put {vanilla:.2f}", color=MUTED, fontsize=9)
    ax.plot(ratios * 100, tree_px, color=BLUE, label="tree, monthly resets")
    ax.errorbar(np.array(mc_ratios) * 100, mc_px, yerr=1.96 * np.array(mc_se),
                fmt="o", color=ORANGE, capsize=3, label="Monte Carlo check")
    ax.set_xlabel("ratchet percentage (strike resets to this % of the running high)")
    ax.set_ylabel("price")
    ax.set_title("Ratchet put: the cost of locking in gains")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ratchet_vs_ratio.png"), dpi=150)
    plt.close(fig)

    # --------------------------------------------- 2. price vs reset frequency
    freqs = [("annual", 1), ("semi-annual", 2), ("quarterly", 4), ("monthly", 12), ("daily", 252)]
    print("\nPrice by reset frequency (tree n=252):")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, color in ((1.0, BLUE), (0.9, ORANGE), (0.8, AQUA)):
        px = [ratchet_put_tree(S0, K0, k, r, q, sigma, T, N_STEPS, m) for _, m in freqs]
        print(f"  ratio={k:.2f}: " + "  ".join(f"{nm}={p:.2f}" for (nm, _), p in zip(freqs, px)))
        ax.plot([m for _, m in freqs], px, color=color, marker="o", label=f"ratchet {int(k*100)}%")
        ax.annotate(f"{int(k*100)}%", (freqs[-1][1], px[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=9, color=color)
    ax.axhline(vanilla, color=MUTED, lw=1.2, ls="--")
    ax.text(1.0, vanilla + 0.15, f"vanilla ATM put {vanilla:.2f}", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([m for _, m in freqs])
    ax.set_xticklabels([nm for nm, _ in freqs], fontsize=9)
    ax.set_xlabel("reset frequency")
    ax.set_ylabel("price")
    ax.set_title("More frequent resets catch more of the high")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ratchet_vs_frequency.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------ 3. sample path
    # pick a path that rallies then falls, so the ratchet is visible
    _, _, paths, resets = ratchet_put_mc(S0, K0, 0.9, r, q, sigma, T, N_STEPS, 12, 2_000,
                                         antithetic=False, seed=3, return_paths=True)
    peak = paths.max(axis=1)
    drop = peak - paths[:, -1]
    idx = int(np.argmax((peak > 125) * drop))
    S = paths[idx]
    K = strike_path(S, set(resets), K0, 0.9)
    t = np.arange(N_STEPS + 1) / N_STEPS

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t, S, color=BLUE, lw=1.6, label="stock")
    ax.step(t, K, where="post", color=ORANGE, lw=2, label="ratcheted strike (90% of observed high)")
    ax.axhline(K0, color=MUTED, lw=1.2, ls="--")
    ax.text(0.01, K0 - 3.5, "vanilla strike stays at 100", color=MUTED, fontsize=9)
    rt = np.array(resets) / N_STEPS
    ax.scatter(rt, S[resets], color=INK, s=18, zorder=5, label="reset dates")
    payoff = max(K[-1] - S[-1], 0.0)
    ax.annotate(f"payoff {payoff:.1f}\n(vanilla: {max(K0 - S[-1], 0):.1f})",
                (1.0, S[-1]), textcoords="offset points", xytext=(-150, 40), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("time (years)")
    ax.set_ylabel("price")
    ax.set_title("One path: the strike follows the stock up and stays there")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ratchet_sample_path.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
