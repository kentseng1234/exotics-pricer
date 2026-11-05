"""
Floating-strike lookback put, priced three ways.

Payoff at maturity:  M_T - S_T,  where M_T = max_{0<=t<=T} S_t.
(The strike "looks back" to the highest price the stock reached, so the put
always finishes in the money. This is the textbook version of the ratchet put
from the Goldman coffee chat; the ratchet is Phase 2.)

Three pricers:
  1. lookback_put_closed_form  -- Goldman, Sosin & Gatto (1979). Continuous monitoring.
  2. lookback_put_tree         -- CRR binomial tree with the running max carried as a
                                  second state variable. Discrete monitoring (n dates).
  3. lookback_put_mc           -- Monte Carlo. Discrete monitoring by default; with
                                  bridge=True it samples the continuous max between
                                  grid points (Brownian bridge), so it should match the
                                  closed form directly.

Model: Black-Scholes world, GBM with constant vol, continuous dividend yield q.
"""

import numpy as np
from scipy.stats import norm

# Broadie-Glasserman-Kou constant: -zeta(1/2)/sqrt(2*pi). The max of a Gaussian
# random walk sits below the continuous max by ~beta*sigma*sqrt(dt).
BGK_BETA = 0.5825971579
# The same offset for a symmetric +/-1 lattice walk (what a CRR tree does) is 1/2,
# so a tree with n steps is NOT the same thing as a contract with n fixings.
TREE_BETA = 0.5


# --------------------------------------------------------------------------- #
# 1. Closed form
# --------------------------------------------------------------------------- #
def lookback_put_closed_form(S0, r, q, sigma, T, M0=None):
    """
    Goldman-Sosin-Gatto (1979) floating-strike lookback put, continuous monitoring.
    Formula as written in Hull, "Options, Futures and Other Derivatives".

    M0 is the running maximum observed so far (defaults to S0 for a new option).
    """
    if M0 is None:
        M0 = S0
    if M0 < S0:
        raise ValueError("running max M0 cannot be below the current spot S0")
    if abs(r - q) < 1e-10:
        # formula has a (r - q) in the denominator; nudge to avoid the singularity
        q = r - 1e-8

    sqT = sigma * np.sqrt(T)
    lnMS = np.log(M0 / S0)
    b1 = (lnMS + (-r + q + 0.5 * sigma**2) * T) / sqT
    b2 = b1 - sqT
    b3 = (lnMS + (r - q - 0.5 * sigma**2) * T) / sqT
    Y2 = 2.0 * (r - q - 0.5 * sigma**2) * lnMS / sigma**2
    k = sigma**2 / (2.0 * (r - q))

    return (M0 * np.exp(-r * T) * (norm.cdf(b1) - k * np.exp(Y2) * norm.cdf(-b3))
            + S0 * np.exp(-q * T) * k * norm.cdf(-b2)
            - S0 * np.exp(-q * T) * norm.cdf(b2))


def bgk_discrete_approx(S0, r, q, sigma, T, m, beta=BGK_BETA):
    """
    First-order Broadie-Glasserman-Kou correction: the max of a path observed on
    m equally spaced dates sits below the continuous max by roughly a factor
    exp(-beta * sigma * sqrt(T/m)). Applying that shift to the payoff gives

        V_discrete ~= e^{-c} V_cont + (e^{-c} - 1) S0 e^{-qT},   c = beta*sigma*sqrt(T/m)

    beta=BGK_BETA  -> a real contract with m fixings (Gaussian increments)
    beta=TREE_BETA -> what an m-step CRR tree converges to (lattice increments)

    Asymptotic in m; useful as a sanity line, not exact for small m.
    """
    c = beta * sigma * np.sqrt(T / m)
    V = lookback_put_closed_form(S0, r, q, sigma, T)
    return np.exp(-c) * V + (np.exp(-c) - 1.0) * S0 * np.exp(-q * T)


# --------------------------------------------------------------------------- #
# 2. Binomial tree with the running max as a state variable
# --------------------------------------------------------------------------- #
def lookback_put_tree(S0, r, q, sigma, T, n):
    """
    CRR tree, n steps. Node state is (j, k):
        j = number of up-moves so far      -> S = S0 * u^(2j - i) at step i
        k = max exponent reached so far    -> M = S0 * u^k

    A plain recombining tree can't price this because the payoff depends on the
    path. Carrying k as a second state variable is the standard fix: the tree
    stays recombining in S, and k only needs to be tracked over its feasible
    range max(2j-i, 0) <= k <= j. That keeps the state count O(n^3) instead of
    the O(2^n) you'd get from tracking every path.

    Backward induction from the terminal payoff M - S:
        V[i][j, k] = e^{-r dt} * ( p * V[i+1][j+1, max(k, 2j+1-i)] + (1-p) * V[i+1][j, k] )
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    if not (0.0 < p < 1.0):
        raise ValueError("risk-neutral probability out of (0,1); increase n")

    # Terminal layer, shape (n+1, n+1): rows j, cols k. Infeasible (j,k) cells
    # get a harmless value; they are never reached from feasible cells.
    j = np.arange(n + 1)[:, None]
    k = np.arange(n + 1)[None, :]
    V = S0 * u ** k - S0 * u ** (2 * j - n)

    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)[:, None]
        k = np.arange(i + 1)[None, :]
        k_up = np.maximum(k, 2 * j + 1 - i)          # up-move may set a new max
        V = disc * (p * V[j + 1, k_up] + (1.0 - p) * V[j, k])

    return float(V[0, 0])


# --------------------------------------------------------------------------- #
# 3. Monte Carlo
# --------------------------------------------------------------------------- #
def lookback_put_mc(S0, r, q, sigma, T, n_steps, n_paths,
                    bridge=False, antithetic=True, seed=0):
    """
    Monte Carlo under GBM.

    bridge=False : discrete monitoring on n_steps equally spaced dates
                   (what a real contract with daily/weekly/monthly fixings pays).
    bridge=True  : continuous monitoring. Between each pair of grid points the
                   max of the log-price is sampled from the Brownian-bridge
                   maximum distribution, so the estimator targets the GSG price
                   exactly, with no discretization bias.

    Returns (price, standard_error).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])
        n_paths = 2 * half
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    logS = np.empty((n_paths, n_steps + 1))
    logS[:, 0] = np.log(S0)
    logS[:, 1:] = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)

    if bridge:
        a = logS[:, :-1]
        b = logS[:, 1:]
        U = rng.uniform(size=a.shape)
        # max of a Brownian bridge from a to b over an interval of length dt
        seg_max = 0.5 * (a + b + np.sqrt((b - a) ** 2 - 2.0 * sigma**2 * dt * np.log(U)))
        logM = seg_max.max(axis=1)
    else:
        logM = logS.max(axis=1)

    payoff = np.exp(logM) - np.exp(logS[:, -1])
    disc_payoff = np.exp(-r * T) * payoff

    if antithetic:
        # pair each path with its mirror before taking the standard error
        paired = 0.5 * (disc_payoff[:half] + disc_payoff[half:])
        return float(paired.mean()), float(paired.std(ddof=1) / np.sqrt(half))
    return float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))
