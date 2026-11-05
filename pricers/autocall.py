"""
Autocallable contingent-coupon note -- the structured-products desk's bread and butter.

Mechanics (per $notional of face):
  On each observation date t_i (quarterly, say):
    - if the note is alive and S_i >= coupon_barrier * S_ref, pay a coupon
      (with memory=True, also pay any coupons missed earlier)
    - if the note is alive, i >= non_call_periods, and S_i >= autocall_barrier * S_ref,
      redeem at par immediately ("autocall"); the note is over
  At maturity (the last observation date), if still alive:
    - European knock-in: if S_T < ki_barrier * S_ref, repay notional * S_T / S_ref
      (the investor eats the full loss); otherwise repay par
    - American knock-in: if the path ever closed below ki_barrier * S_ref AND
      S_T < S_ref, repay notional * S_T / S_ref; otherwise par
  Cash flows are discounted at r + funding_spread, because the note is unsecured
  bank debt, not a collateralised derivative.

The "if it's Tuesday it pays off" stack of conditions makes this Monte Carlo
territory: each path is a sequence of yes/no checks on fixed dates.

Key trick: the price is LINEAR in the coupon.
    price = PV(principal flows) + coupon * PV(number of coupons paid)
so the fair coupon (price = par) is one division, no root-finding:
    fair_coupon = (notional - PV_principal) / PV_coupon_units
Both PVs come out of the same simulation.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AutocallSpec:
    obs_times: np.ndarray                 # years from now; last entry is maturity
    coupon: float = 0.02                  # per observation period, fraction of notional
    autocall_barrier: float = 1.00        # fraction of S_ref
    coupon_barrier: float = 0.70
    ki_barrier: float = 0.60
    ki_type: str = "european"             # or "american"
    memory: bool = False
    non_call_periods: int = 0             # first k observation dates cannot autocall
    notional: float = 100.0

    @staticmethod
    def quarterly(years=3.0, **kw):
        n = int(round(years * 4))
        return AutocallSpec(obs_times=np.arange(1, n + 1) * 0.25, **kw)


@dataclass
class AutocallResult:
    price: float
    se: float
    pv_principal: float        # PV of par / loss-adjusted redemption, per note
    pv_coupon_units: float     # PV of "one unit of coupon" per coupon paid
    fair_coupon: float         # per-period coupon that prices the note at par
    call_prob: np.ndarray      # P(autocalled on date i)
    ki_prob: float             # P(knocked in at maturity, i.e. a loss)
    expected_life: float       # years
    coupon: float = 0.0

    @property
    def fair_coupon_annual(self):
        return self.fair_coupon * 4  # assumes quarterly; adjust if not


def _simulate(S, S_ref, r, q, sigma, spec, n_paths, steps_per_period, seed, antithetic):
    """Simulate GBM on a grid that hits every observation date. Returns (S_obs, path_min)."""
    rng = np.random.default_rng(seed)
    obs = np.asarray(spec.obs_times, dtype=float)
    # time grid: steps_per_period substeps inside each observation period
    grid = [0.0]
    for i, t in enumerate(obs):
        prev = grid[-1]
        grid.extend(np.linspace(prev, t, steps_per_period + 1)[1:])
    grid = np.array(grid)
    dts = np.diff(grid)
    n_steps = len(dts)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    incr = (r - q - 0.5 * sigma**2) * dts + sigma * np.sqrt(dts) * Z
    logS = np.log(S) + np.cumsum(incr, axis=1)
    obs_idx = np.arange(1, len(obs) + 1) * steps_per_period - 1   # index into logS columns
    S_obs = np.exp(logS[:, obs_idx])
    path_min = np.exp(logS.min(axis=1)) if spec.ki_type == "american" else None
    return S_obs, path_min


def price_autocall(S, S_ref, r, q, sigma, spec, funding_spread=0.0,
                   n_paths=200_000, steps_per_period=1, seed=0, antithetic=True):
    """
    S      : current spot
    S_ref  : the initial fixing the barriers are struck against (S_ref = S on trade date)
    Returns an AutocallResult.
    """
    if spec.ki_type == "american" and steps_per_period < 2:
        steps_per_period = 63  # need the path, not just the fixings
    S_obs, path_min = _simulate(S, S_ref, r, q, sigma, spec, n_paths, steps_per_period, seed, antithetic)
    n_paths = S_obs.shape[0]
    obs = np.asarray(spec.obs_times)
    n_obs = len(obs)
    df = np.exp(-(r + funding_spread) * obs)

    alive = np.ones(n_paths, dtype=bool)
    missed = np.zeros(n_paths)                      # memory-coupon counter
    pv_principal = np.zeros(n_paths)
    pv_coupon_units = np.zeros(n_paths)
    called_on = np.full(n_paths, -1)

    for i in range(n_obs):
        Si = S_obs[:, i]
        pays = alive & (Si >= spec.coupon_barrier * S_ref)
        units = np.where(pays, 1.0 + (missed if spec.memory else 0.0), 0.0)
        pv_coupon_units += units * df[i]
        if spec.memory:
            missed = np.where(pays, 0.0, missed + alive)

        if i < n_obs - 1:
            call = alive & (i >= spec.non_call_periods) & (Si >= spec.autocall_barrier * S_ref)
            pv_principal += np.where(call, spec.notional * df[i], 0.0)
            called_on[call] = i
            alive &= ~call
        else:
            # maturity: everyone still alive is redeemed here
            ratio = Si / S_ref
            if spec.ki_type == "european":
                loss = ratio < spec.ki_barrier
            else:
                loss = (path_min < spec.ki_barrier * S_ref) & (ratio < 1.0)
            redemption = np.where(loss, spec.notional * ratio, spec.notional)
            pv_principal += np.where(alive, redemption * df[i], 0.0)
            ki_flag = alive & loss
            called_on[alive] = i

    coupon_cash = spec.coupon * spec.notional
    pv = pv_principal + coupon_cash * pv_coupon_units
    if antithetic:
        half = n_paths // 2
        paired = 0.5 * (pv[:half] + pv[half:])
        price, se = paired.mean(), paired.std(ddof=1) / np.sqrt(half)
    else:
        price, se = pv.mean(), pv.std(ddof=1) / np.sqrt(n_paths)

    pvP = pv_principal.mean()
    pvC = pv_coupon_units.mean()
    fair = (spec.notional - pvP) / (spec.notional * pvC) if pvC > 0 else float("nan")
    call_prob = np.array([(called_on == i).mean() for i in range(n_obs)])
    life = float((obs[called_on]).mean())
    return AutocallResult(price=float(price), se=float(se), pv_principal=float(pvP),
                          pv_coupon_units=float(pvC), fair_coupon=float(fair),
                          call_prob=call_prob, ki_prob=float(ki_flag.mean()),
                          expected_life=life, coupon=spec.coupon)


def greeks(S, S_ref, r, q, sigma, spec, funding_spread=0.0, n_paths=200_000,
           dS=0.01, dvol=0.01, seed=0, **kw):
    """
    Bump-and-reprice with common random numbers (same seed each call), so the
    noise largely cancels in the differences.
      delta = dV/dS            (per 1 unit of spot)
      gamma = d2V/dS2
      vega  = dV/dsigma        (per 1 vol point, i.e. per 0.01)
    """
    h = dS * S
    f = lambda s, v: price_autocall(s, S_ref, r, q, v, spec, funding_spread, n_paths, seed=seed, **kw).price
    V0 = f(S, sigma)
    Vu, Vd = f(S + h, sigma), f(S - h, sigma)
    Vv = f(S, sigma + dvol)
    return {
        "price": V0,
        "delta": (Vu - Vd) / (2 * h),
        "gamma": (Vu - 2 * V0 + Vd) / h**2,
        "vega": (Vv - V0),          # per 1 vol point
    }
