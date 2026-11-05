"""
Ratchet (reset) put -- the product from the Goldman coffee chat.

A vanilla put has a fixed strike K0. Here the strike ratchets up with the stock:
on each reset date t_i the strike becomes

    K_i = max( K_{i-1},  ratio * M_i ),     M_i = highest price observed on reset dates so far

and the payoff at maturity is  max(K_T - S_T, 0).

With ratio = 1 you lock in 100% of the high: if resets happen on every monitoring
date this is exactly the discrete lookback put from Phase 1 (always in the money).
With ratio = 0.9 you lock in 90% of the high, so the option can still expire
worthless if the stock never falls more than 10% from its observed peak. With
ratio = 0 the strike never moves and you have a vanilla put with strike K0.

Two pricers:
  ratchet_put_tree -- CRR tree carrying the observed max as a second state variable.
                      Same machinery as the lookback tree, but the max state only
                      updates on reset steps. Note: on a reset step a DOWN move can
                      still set a new observed high (the stock may have risen
                      unobserved and then dipped), so both branches update.
  ratchet_put_mc   -- Monte Carlo; M is the max of the path sampled on reset dates.

No closed form exists for discrete resets. Validation is by the two limits above.
"""

import numpy as np


def _reset_steps(n_steps, n_resets):
    if n_steps % n_resets:
        raise ValueError(f"n_steps={n_steps} must be a multiple of n_resets={n_resets}")
    every = n_steps // n_resets
    return set(range(every, n_steps + 1, every))   # reset at steps every, 2*every, ..., n_steps


def ratchet_put_tree(S0, K0, ratio, r, q, sigma, T, n, n_resets):
    """CRR tree, n steps, strike resets on n_resets equally spaced dates (last one at T)."""
    resets = _reset_steps(n, n_resets)
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    if not (0.0 < p < 1.0):
        raise ValueError("risk-neutral probability out of (0,1); increase n")

    # terminal layer: rows j (up-moves), cols k (max exponent observed on reset dates)
    j = np.arange(n + 1)[:, None]
    k = np.arange(n + 1)[None, :]
    K_T = np.maximum(K0, ratio * S0 * u ** k)
    V = np.maximum(K_T - S0 * u ** (2 * j - n), 0.0)

    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)[:, None]
        k = np.arange(i + 1)[None, :]
        if (i + 1) in resets:
            k_up = np.maximum(k, 2 * j + 1 - i)
            k_dn = np.maximum(k, 2 * j - 1 - i)
        else:
            k_up = np.broadcast_to(k, (i + 1, i + 1))
            k_dn = k_up
        V = disc * (p * V[j + 1, k_up] + (1.0 - p) * V[j, k_dn])

    return float(V[0, 0])


def ratchet_put_mc(S0, K0, ratio, r, q, sigma, T, n_steps, n_resets, n_paths,
                   antithetic=True, seed=0, return_paths=False):
    """Monte Carlo under GBM. Returns (price, standard_error)."""
    resets = sorted(_reset_steps(n_steps, n_resets))
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

    obs = logS[:, [0] + resets]              # S0 counts as the first observation
    M = np.exp(obs.max(axis=1))
    K_T = np.maximum(K0, ratio * M)
    S_T = np.exp(logS[:, -1])
    disc_payoff = np.exp(-r * T) * np.maximum(K_T - S_T, 0.0)

    if antithetic:
        paired = 0.5 * (disc_payoff[:half] + disc_payoff[half:])
        price, se = float(paired.mean()), float(paired.std(ddof=1) / np.sqrt(half))
    else:
        price, se = float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))

    if return_paths:
        return price, se, np.exp(logS), resets
    return price, se


def strike_path(S, resets, K0, ratio):
    """For one simulated path S (length n_steps+1), the ratcheted strike at every step.
    Used for illustration."""
    K = np.full_like(S, K0, dtype=float)
    M = S[0]
    current = max(K0, ratio * M)
    for t in range(1, len(S)):
        if t in resets:
            M = max(M, S[t])
            current = max(current, ratio * M)
        K[t] = current
    return K
