"""Run with:  python -m pytest tests/   (or just  python tests/test_autocall.py)"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pricers.autocall import AutocallSpec, price_autocall, greeks

S0, r, q, sigma, SPREAD = 100.0, 0.05, 0.015, 0.25, 0.01


def test_reprice_at_fair_coupon_is_par():
    R = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0), SPREAD, 200_000, seed=1)
    R2 = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0, coupon=R.fair_coupon),
                        SPREAD, 200_000, seed=1)
    assert abs(R2.price - 100.0) < 1e-6          # same paths, exact by linearity
    R3 = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0, coupon=R.fair_coupon),
                        SPREAD, 200_000, seed=99)
    assert abs(R3.price - 100.0) < 3 * R3.se + 0.02   # fresh paths, within MC error


def test_zero_vol_called_on_first_date():
    spec = AutocallSpec.quarterly(3.0, coupon=0.02)
    R = price_autocall(S0, S0, r, q, 1e-4, spec, SPREAD, 2_000)
    expected = 100 * (1 + 0.02) * np.exp(-(r + SPREAD) * 0.25)
    assert abs(R.price - expected) < 1e-3
    assert R.call_prob[0] > 0.999


def test_zero_vol_far_below_barriers_is_a_forward():
    # spot at 40, never pays a coupon, never called, redeems at S_T/S_ref at maturity
    spec = AutocallSpec.quarterly(3.0, coupon=0.02)
    R = price_autocall(40.0, S0, r, q, 1e-4, spec, SPREAD, 2_000)
    S_T = 40.0 * np.exp((r - q) * 3.0)
    expected = 100 * S_T / S0 * np.exp(-(r + SPREAD) * 3.0)
    assert abs(R.price - expected) < 1e-2
    assert R.ki_prob > 0.999


def test_fair_coupon_rises_with_vol_and_with_ki_barrier():
    fc_vol = [price_autocall(S0, S0, r, q, v, AutocallSpec.quarterly(3.0), SPREAD, 50_000).fair_coupon
              for v in (0.15, 0.25, 0.35)]
    assert all(np.diff(fc_vol) > 0)
    fc_ki = [price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0, ki_barrier=b), SPREAD, 50_000).fair_coupon
             for b in (0.50, 0.60, 0.70)]
    assert all(np.diff(fc_ki) > 0)


def test_memory_and_non_call_lower_the_fair_coupon():
    base = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0), SPREAD, 100_000).fair_coupon
    mem = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0, memory=True), SPREAD, 100_000).fair_coupon
    nc = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0, non_call_periods=4), SPREAD, 100_000).fair_coupon
    assert mem < base and nc < base


def test_american_ki_more_expensive_than_european():
    eu = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0), SPREAD, 50_000).fair_coupon
    am = price_autocall(S0, S0, r, q, sigma, AutocallSpec.quarterly(3.0, ki_type="american"), SPREAD, 50_000).fair_coupon
    assert am > eu


def test_holder_is_short_vega_at_issue():
    spec = AutocallSpec.quarterly(3.0, coupon=0.0288)
    G = greeks(S0, S0, r, q, sigma, spec, SPREAD, 100_000)
    assert G["vega"] < 0
    assert 0 < G["delta"] < 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
