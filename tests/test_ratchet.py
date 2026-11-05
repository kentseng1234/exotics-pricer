"""Run with:  python -m pytest tests/   (or just  python tests/test_ratchet.py)"""

import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pricers.lookback import lookback_put_tree, lookback_put_mc
from pricers.ratchet import ratchet_put_tree, ratchet_put_mc

S0, r, q, sigma, T = 100.0, 0.05, 0.0, 0.25, 1.0


def bs_put(S, K):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def test_full_ratchet_every_step_is_lookback_tree():
    n = 200
    assert abs(ratchet_put_tree(S0, S0, 1.0, r, q, sigma, T, n, n)
               - lookback_put_tree(S0, r, q, sigma, T, n)) < 1e-9


def test_zero_ratchet_is_vanilla_put():
    tree = ratchet_put_tree(S0, S0, 0.0, r, q, sigma, T, 480, 12)
    assert abs(tree - bs_put(S0, S0)) < 0.03
    p, se = ratchet_put_mc(S0, S0, 0.0, r, q, sigma, T, 252, 12, 200_000, seed=4)
    assert abs(p - bs_put(S0, S0)) < 3 * se + 0.02


def test_full_ratchet_monthly_mc_is_monthly_lookback():
    p1, se1 = ratchet_put_mc(S0, S0, 1.0, r, q, sigma, T, 252, 12, 200_000, seed=6)
    p2, se2 = lookback_put_mc(S0, r, q, sigma, T, 12, 200_000, bridge=False, seed=6)
    assert abs(p1 - p2) < 3 * (se1 + se2) + 0.02


def test_tree_and_mc_agree_base_case():
    tree = ratchet_put_tree(S0, S0, 0.9, r, q, sigma, T, 252, 12)
    p, se = ratchet_put_mc(S0, S0, 0.9, r, q, sigma, T, 252, 12, 400_000, seed=8)
    assert abs(tree - p) < 3 * se + 0.03, (tree, p, se)


def test_price_increases_in_ratio_and_in_reset_frequency():
    by_ratio = [ratchet_put_tree(S0, S0, k, r, q, sigma, T, 252, 12) for k in (0.0, 0.5, 0.8, 0.9, 1.0)]
    assert all(np.diff(by_ratio) >= -1e-12)
    by_freq = [ratchet_put_tree(S0, S0, 0.9, r, q, sigma, T, 252, m) for m in (1, 4, 12, 252)]
    assert all(np.diff(by_freq) >= -1e-12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
