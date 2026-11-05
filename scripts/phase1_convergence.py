"""
Phase 1: validate the three lookback pricers against each other.

Produces three figures in outputs/:
  1. tree_convergence.png    -- binomial tree price vs number of steps
  2. mc_convergence.png      -- Monte Carlo price vs number of paths, with 95% bands
  3. discrete_monitoring.png -- price vs monitoring frequency (the practical gap)

Run from the repo root:  python scripts/phase1_convergence.py
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pricers.lookback import (lookback_put_closed_form, lookback_put_tree,
                              lookback_put_mc, bgk_discrete_approx, TREE_BETA)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- base case --------------------------------------------------------------
S0, r, q, sigma, T = 100.0, 0.05, 0.0, 0.25, 1.0

# ---- chart styling ----------------------------------------------------------
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


def main():
    V_cf = lookback_put_closed_form(S0, r, q, sigma, T)
    print(f"Closed form (Goldman-Sosin-Gatto, continuous): {V_cf:.4f}\n")

    # ------------------------------------------------------------------ 1. tree
    steps = [10, 20, 50, 100, 200, 400, 800]
    tree_px, bgk_px = [], []
    print("Binomial tree with running-max state:")
    for n in steps:
        t0 = time.time()
        v = lookback_put_tree(S0, r, q, sigma, T, n)
        tree_px.append(v)
        bgk_px.append(bgk_discrete_approx(S0, r, q, sigma, T, n, beta=TREE_BETA))
        print(f"  n={n:4d}  price={v:.4f}  gap to closed form={V_cf - v:+.4f}  ({time.time()-t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(V_cf, color=INK, lw=1.2, ls="--")
    ax.text(steps[0], V_cf + 0.08, f"closed form (continuous) {V_cf:.3f}", color=INK, fontsize=9)
    ax.plot(steps, bgk_px, color=ORANGE, lw=1.5, ls=":", label="lattice-max correction (beta = 1/2)")
    ax.plot(steps, tree_px, color=BLUE, marker="o", label="binomial tree, n monitoring dates")
    ax.set_xscale("log")
    ax.set_xlabel("tree steps n (log scale)")
    ax.set_ylabel("price")
    ax.set_title("Lookback put: tree converges from below as monitoring gets denser")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tree_convergence.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------- 2. MC
    path_counts = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000]
    mc_px, mc_se = [], []
    print("\nMonte Carlo, continuous monitoring (Brownian bridge), 252 grid steps:")
    for N in path_counts:
        t0 = time.time()
        p, se = lookback_put_mc(S0, r, q, sigma, T, 252, N, bridge=True, seed=1)
        mc_px.append(p)
        mc_se.append(se)
        print(f"  paths={N:7d}  price={p:.4f} +/- {se:.4f}   ({time.time()-t0:.1f}s)")

    mc_px, mc_se = np.array(mc_px), np.array(mc_se)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(V_cf, color=INK, lw=1.2, ls="--")
    ax.text(path_counts[0], V_cf + 0.03, f"closed form {V_cf:.3f}", color=INK, fontsize=9)
    ax.fill_between(path_counts, mc_px - 1.96 * mc_se, mc_px + 1.96 * mc_se,
                    color=BLUE, alpha=0.18, lw=0, label="95% confidence band")
    ax.plot(path_counts, mc_px, color=BLUE, marker="o", label="Monte Carlo estimate")
    ax.set_xscale("log")
    ax.set_xlabel("simulated paths (log scale)")
    ax.set_ylabel("price")
    ax.set_title("Lookback put: Monte Carlo with bridge-sampled max hits the closed form")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mc_convergence.png"), dpi=150)
    plt.close(fig)

    # --------------------------------------------------- 3. discrete monitoring
    freqs = [("annual", 1), ("semi-annual", 2), ("quarterly", 4), ("monthly", 12),
             ("weekly", 52), ("daily", 252)]
    print("\nDiscrete monitoring (Monte Carlo, 400k paths) vs continuous closed form:")
    disc_px, disc_se, disc_bgk = [], [], []
    for name, m in freqs:
        p, se = lookback_put_mc(S0, r, q, sigma, T, m, 400_000, bridge=False, seed=2)
        disc_px.append(p)
        disc_se.append(se)
        disc_bgk.append(bgk_discrete_approx(S0, r, q, sigma, T, m))
        print(f"  {name:12s} m={m:5d}  price={p:.4f} +/- {se:.4f}  "
              f"below continuous by {100*(V_cf-p)/V_cf:5.1f}%   BGK approx={disc_bgk[-1]:.4f}")

    ms = [m for _, m in freqs]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(V_cf, color=INK, lw=1.2, ls="--")
    ax.text(ms[0], V_cf + 0.12, f"continuous monitoring (closed form) {V_cf:.3f}", color=INK, fontsize=9)
    # BGK is an asymptotic (large-m) result; only draw it where it's meaningful
    keep = [i for i, m in enumerate(ms) if m >= 12]
    ax.plot([ms[i] for i in keep], [disc_bgk[i] for i in keep], color=ORANGE, lw=1.5, ls=":",
            label="BGK first-order correction (m >= 12)")
    ax.errorbar(ms, disc_px, yerr=1.96 * np.array(disc_se), color=BLUE, marker="o",
                capsize=3, label="Monte Carlo, m fixings")
    for (name, m), p in zip(freqs, disc_px):
        ax.annotate(name, (m, p), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("monitoring dates per year (log scale)")
    ax.set_ylabel("price")
    ax.set_title("The practical gap: fewer fixings, cheaper option")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "discrete_monitoring.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
