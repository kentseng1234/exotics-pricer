"""Run with:  python -m pytest tests/   (or just  python tests/test_lookback.py)"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pricers.lookback import (lookback_put_closed_form, lookback_put_tree,
                              lookback_put_mc, bgk_discrete_approx, TREE_BETA)

S0, r, q, sigma, T = 100.0, 0.05, 0.0, 0.25, 1.0


def test_bridge_mc_matches_closed_form():
    cf = lookback_put_closed_form(S0, r, q, sigma, T)
    p, se = lookback_put_mc(S0, r, q, sigma, T, 252, 200_000, bridge=True, seed=7)
    assert abs(p - cf) < 3 * se + 0.02, (p, cf, se)


def test_tree_tracks_lattice_correction():
    # An n-step CRR tree is NOT an n-fixing contract: its +/-1 lattice max is
    # biased above a Gaussian walk's max (offset constant 1/2 vs 0.5826). So the
    # tree should track the closed form shifted by beta = 1/2, not BGK's beta.
    for n in (200, 800):
        tree = lookback_put_tree(S0, r, q, sigma, T, n)
        lattice = bgk_discrete_approx(S0, r, q, sigma, T, n, beta=TREE_BETA)
        assert abs(tree - lattice) < 0.05, (n, tree, lattice)


def test_tree_overprices_an_n_fixing_contract():
    # ...and therefore sits above Monte Carlo with the same number of fixings.
    n = 52
    tree = lookback_put_tree(S0, r, q, sigma, T, n)
    p, se = lookback_put_mc(S0, r, q, sigma, T, n, 200_000, bridge=False, seed=3)
    assert tree > p + 3 * se, (tree, p, se)


def test_one_fixing_is_a_vanilla_put():
    # With a single fixing at T, M_T = max(S0, S_T) and payoff = max(S0 - S_T, 0):
    # an ATM vanilla put. Black-Scholes put with K = S0.
    from scipy.stats import norm
    d1 = (np.log(S0 / S0) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_put = S0 * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    p, se = lookback_put_mc(S0, r, q, sigma, T, 1, 400_000, bridge=False, seed=5)
    assert abs(p - bs_put) < 3 * se + 0.02, (p, bs_put, se)


def test_discrete_prices_increase_with_monitoring():
    prices = [lookback_put_mc(S0, r, q, sigma, T, m, 100_000, seed=9)[0] for m in (4, 12, 52, 252)]
    assert all(np.diff(prices) > 0)
    assert prices[-1] < lookback_put_closed_form(S0, r, q, sigma, T)


def test_bgk_close_for_dense_monitoring():
    p, se = lookback_put_mc(S0, r, q, sigma, T, 252, 400_000, seed=11)
    assert abs(p - bgk_discrete_approx(S0, r, q, sigma, T, 252)) < 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
