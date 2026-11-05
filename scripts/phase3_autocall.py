"""
Phase 3: the autocallable note. What coupon can a structurer offer, where does
the coupon come from, and what does the hedge look like?

Produces four figures in outputs/:
  1. autocall_fair_coupon.png  -- fair annual coupon vs vol, for three knock-in barriers
  2. autocall_greeks.png       -- delta and gamma vs spot, day before an observation vs mid-period
  3. autocall_vega.png         -- vega vs spot on the trade date
  4. autocall_call_profile.png -- probability of being called on each observation date

Run from the repo root:  python scripts/phase3_autocall.py
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pricers.autocall import AutocallSpec, price_autocall, greeks

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- base case: 3y quarterly autocall on a single stock ---------------------
S0, r, q, sigma, SPREAD = 100.0, 0.05, 0.015, 0.25, 0.01
YEARS = 3.0
N_PATHS = 400_000

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def main():
    base = AutocallSpec.quarterly(YEARS)
    R = price_autocall(S0, S0, r, q, sigma, base, SPREAD, N_PATHS)
    c = R.fair_coupon
    print(f"Base case: S0={S0}, vol={sigma:.0%}, r={r:.1%}, q={q:.1%}, spread={SPREAD:.1%}")
    print(f"  autocall 100% / coupon barrier 70% / knock-in 60% European / quarterly / {YEARS:.0f}y")
    print(f"  fair coupon: {c:.3%} per quarter = {4*c:.2%} per year")
    print(f"  P(called on first date) = {R.call_prob[0]:.1%}, P(ever called early) = {R.call_prob[:-1].sum():.1%}")
    print(f"  P(loss at maturity) = {R.ki_prob:.1%}, expected life = {R.expected_life:.2f}y")
    for name, kw in (("memory coupons", dict(memory=True)),
                     ("American knock-in", dict(ki_type="american")),
                     ("1y non-call", dict(non_call_periods=4)),
                     ("knock-in 70%", dict(ki_barrier=0.70)),
                     ("knock-in 50%", dict(ki_barrier=0.50))):
        Rx = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(YEARS, **kw), SPREAD, 200_000)
        print(f"  with {name:18s}: fair coupon {4*Rx.fair_coupon:.2%}/yr")

    # decomposition: what the coupon is paying for
    R_rf = price_autocall(S0, S0, r, q, sigma, base, 0.0, N_PATHS)
    print(f"\n  funding spread of {SPREAD:.0%} adds {4*(c - R_rf.fair_coupon):.2%}/yr to the coupon")

    spec = AutocallSpec.quarterly(YEARS, coupon=c)   # the note as issued

    # ------------------------------------------------- 1. fair coupon vs vol
    vols = np.arange(0.10, 0.51, 0.025)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    print("\nFair annual coupon by vol (knock-in 60%):")
    for ki, color in ((0.70, ORANGE), (0.60, BLUE), (0.50, AQUA)):
        fc = []
        for v in vols:
            Rv = price_autocall(S0, S0, r, q, v, AutocallSpec.quarterly(YEARS, ki_barrier=ki), SPREAD, 100_000)
            fc.append(4 * Rv.fair_coupon * 100)
        if ki == 0.60:
            print("  " + "  ".join(f"{v:.0%}:{f:.1f}%" for v, f in zip(vols[::2], fc[::2])))
        ax.plot(vols * 100, fc, color=color, label=f"knock-in {int(ki*100)}%")
        ax.annotate(f"KI {int(ki*100)}%", (vols[-1] * 100, fc[-1]), textcoords="offset points",
                    xytext=(6, -3), fontsize=9, color=color)
    ax.axhline(100 * (r + SPREAD), color=MUTED, lw=1.2, ls="--")
    ax.text(30, 100 * (r + SPREAD) + 0.4, "bank's funding rate 6% (what the coupon collapses to at low vol)",
            color=MUTED, fontsize=9)
    ax.set_xlabel("volatility of the underlying (%)")
    ax.set_ylabel("fair annual coupon (%)")
    ax.set_title("The headline coupon is the price of the put you sold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "autocall_fair_coupon.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------- 2. delta & gamma vs spot
    spots = np.linspace(55, 130, 76)
    one_day = 1.0 / 252
    # situation A: one day before the first observation, note has been alive ~3 months
    spec_A = AutocallSpec.quarterly(YEARS, coupon=c)
    spec_A.obs_times = np.arange(0, 12) * 0.25 + one_day
    # situation B: half-way between observations
    spec_B = AutocallSpec.quarterly(YEARS, coupon=c)
    spec_B.obs_times = np.arange(0, 12) * 0.25 + 0.125

    print("\nComputing Greeks across spot (this is the slow part)...")
    t0 = time.time()
    GA = [greeks(s, S0, r, q, sigma, spec_A, SPREAD, 200_000) for s in spots]
    GB = [greeks(s, S0, r, q, sigma, spec_B, SPREAD, 200_000) for s in spots]
    print(f"  done in {time.time()-t0:.0f}s")
    dA, dB = [g["delta"] for g in GA], [g["delta"] for g in GB]
    gA, gB = [g["gamma"] for g in GA], [g["gamma"] for g in GB]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    for ax, A, B, lab in ((ax1, dA, dB, "delta (shares per note)"), (ax2, gA, gB, "gamma")):
        ax.plot(spots, B, color=BLUE, label="mid-way between observation dates")
        ax.plot(spots, A, color=ORANGE, label="one day before an observation date")
        for lvl, txt in ((100, "autocall"), (70, "coupon barrier"), (60, "knock-in")):
            ax.axvline(lvl, color=MUTED, lw=1, ls=":")
        ax.set_ylabel(lab)
    for lvl, txt in ((100, "autocall 100"), (70, "coupon 70"), (60, "KI 60")):
        ax1.text(lvl + 0.6, ax1.get_ylim()[1] * 0.95, txt, color=MUTED, fontsize=8, va="top")
    ax1.set_title("Hedging an autocall: the Greeks the day before a check date")
    ax2.legend(loc="lower right")
    ax2.set_xlabel("spot (initial fixing = 100)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "autocall_greeks.png"), dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------- 3. vega vs spot
    spots_v = np.linspace(55, 130, 31)
    vega = [greeks(s, S0, r, q, sigma, spec, SPREAD, 200_000)["vega"] for s in spots_v]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(0, color=INK, lw=1)
    ax.plot(spots_v, vega, color=BLUE, marker="o")
    for lvl, txt in ((100, "autocall"), (70, "coupon barrier"), (60, "knock-in")):
        ax.axvline(lvl, color=MUTED, lw=1, ls=":")
        ax.text(lvl + 0.6, -0.06, txt, color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("spot (initial fixing = 100)")
    ax.set_ylabel("vega ($ per note per 1 vol point)")
    ax.set_title("Vega on the trade date: the holder is short vol")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "autocall_vega.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------- 4. call profile
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = [f"{t:.2f}y" for t in base.obs_times]
    colors = [BLUE] * (len(labels) - 1) + [ORANGE]
    ax.bar(labels, R.call_prob * 100, color=colors, width=0.7)
    ax.text(len(labels) - 1, R.call_prob[-1] * 100 + 1.2,
            f"matures\n{R.ki_prob:.1%} with a loss", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("probability the note ends on this date (%)")
    ax.set_xlabel("observation date")
    ax.set_title(f"When does it end? Half the notes are called at the first check")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "autocall_call_profile.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
