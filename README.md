# exotics-pricer

Pricing and hedging path-dependent equity derivatives, from binomial trees to autocalls.

The idea: a vanilla option's value depends only on where the stock finishes, which is why Black-Scholes gives a clean formula. Once the payoff depends on the path the stock took (its running max, whether it crossed a barrier, what it did on a set of check dates), the clean formula usually disappears and you need numerical methods. This repo builds those methods up one product at a time and shows where each one stops working.

## Status

- Phase 1 (done): floating-strike lookback put, priced three ways and cross-checked
- Phase 2 (done): ratchet / reset put, strike resets to k% of the running high on a schedule
- Phase 3 (done): autocallable note, Monte Carlo pricing, fair-coupon solver, Greeks by bump-and-reprice
- Phase 4 (next): repricing under stochastic vol (Heston) and a delta-hedging P&L simulation

## Phase 1: the lookback put

Payoff at maturity is M_T minus S_T, where M_T is the highest price the stock reached over the life of the option. The option always finishes in the money; what you pay for is the "look back" to the peak.

Three pricers, all in `pricers/lookback.py`:

1. **Closed form.** Goldman, Sosin and Gatto (1979), continuous monitoring. The benchmark.
2. **Binomial tree.** CRR tree with the running max carried as a second state variable. Tracking every path would cost 2^n; tracking (price, running max) costs about n^3/6 states and keeps the tree recombining.
3. **Monte Carlo.** Discrete monitoring on m fixing dates by default. With `bridge=True` the max between grid points is sampled from the Brownian-bridge maximum distribution, which removes the discretization bias entirely, so the estimator targets the closed form exactly.

### Results (S0 = 100, r = 5%, q = 0, vol = 25%, T = 1y)

- Closed form: 18.723
- Monte Carlo with bridge-sampled max, 400k paths: 18.722 ± 0.012
- Tree, 800 steps: 18.207, still climbing toward 18.723 at roughly 1/sqrt(n)

The three methods agree. That agreement is the validation step for everything that follows.

### Three things the plots show

**`outputs/mc_convergence.png`.** The Monte Carlo estimate with the bridge correction lands on the closed form and the 95% band shrinks like 1/sqrt(paths). Without the bridge, Monte Carlo on a daily grid lands 5.6% below the closed form, because a daily sample of the path misses the true peak.

**`outputs/discrete_monitoring.png`.** Real contracts fix on a schedule, and the schedule is a big deal. Same option, same vol, same maturity:

- 1 fixing per year: 7.46 (with one fixing at expiry this is exactly an ATM vanilla put, and it prices as one)
- quarterly: 11.72
- monthly: 14.29
- weekly: 16.46
- daily: 17.67
- continuous: 18.72

Monthly monitoring is worth 24% less than continuous. This is the "messier stuff in practice" that makes the textbook formula only a starting point. The dotted line is the Broadie-Glasserman-Kou first-order correction, which shifts the continuous max down by 0.5826 * vol * sqrt(dt); it's good once you have a dozen or more fixings and poor below that.

**`outputs/tree_convergence.png`.** The tree converges from below, and it converges slowly. One subtlety that cost me a failed test: an n-step tree is not the same product as an n-fixing contract. A binomial lattice can only move up or down by exactly one tick each step, and the running max of that walk sits slightly higher than the running max of a Gaussian walk with the same variance (the offset constant is 1/2 for a ±1 walk versus 0.5826 for Gaussian steps). So a 52-step tree prices at 16.79 while a true 52-fixing contract prices at 16.47 by Monte Carlo. The tree tracks the closed form shifted by beta = 1/2, which is the dotted line on the plot.

The practical lesson: use the tree to understand the mechanics and to price small path-dependent problems, use Monte Carlo for anything with a real fixing schedule, and use the closed form to check both.

## Phase 2: the ratchet put

This is the product described on the desk. A vanilla put has a fixed strike. Here, on each reset date the strike becomes the larger of its current level and k% of the highest price observed on reset dates so far. Payoff is max(K_T minus S_T, 0). If the stock runs from 100 to 150 and k is 90%, the strike ratchets to 135 and stays there even if the stock falls back, so the holder has locked in most of the gain.

Code is in `pricers/ratchet.py`: the same tree-with-running-max machinery as Phase 1, except the max state only updates on reset steps (and on a reset step a down move can still set a new observed high, because the stock may have risen unobserved and then dipped), plus a Monte Carlo pricer. There is no closed form for discrete resets, so validation is by the two limits:

- k = 100% with a reset on every step reproduces the Phase 1 lookback tree to machine precision
- k = 0 reproduces the Black-Scholes put with strike K0

Both pass, and for the base case (k = 90%, monthly resets) tree and Monte Carlo agree at 9.03 versus 9.02 ± 0.01.

### Results (S0 = K0 = 100, r = 5%, vol = 25%, T = 1y, monthly resets)

- vanilla ATM put: 7.46
- ratchet 80%: 7.61
- ratchet 90%: 9.03
- ratchet 95%: 11.05
- ratchet 100%: 14.33 (this is the monthly discrete lookback from Phase 1)

### What the plots show

**`outputs/ratchet_vs_ratio.png`.** The price is flat at the vanilla level until k gets to about 75%, then rises steeply. The reason is mechanical: the ratchet only does anything once the observed high exceeds K0 divided by k. At 80% the stock has to be up 25% before the strike moves at all, and under 25% vol that is uncommon within a year. At 90% it moves once the stock is up 11%, which happens often. So "lock in 80% of gains" sounds like most of the protection and costs almost nothing extra, while "lock in 100%" doubles the premium. That trade-off is what a structurer is pricing when they pick the ratchet level for a client.

**`outputs/ratchet_vs_frequency.png`.** Same story as Phase 1 but now interacting with k. For the full ratchet (100%), going from annual to daily resets takes the price from 7.45 to 17.81. For the 90% ratchet the same change only moves it from 7.45 to 10.14, and for 80% it barely moves. The reset schedule matters most when the ratchet percentage is high, because that is when the strike is close to the stock and small differences in the observed high change the payoff.

**`outputs/ratchet_sample_path.png`.** One simulated path. The stock peaks above 140 between reset dates, but the monthly schedule only ever observes 137, so the strike locks at 123.4 rather than 128. The option still pays 51.9 at expiry against 28.4 for a vanilla put. Worth looking at before an interview: it is the whole product in one picture.

### Where the tree stops being enough

The tree handles the ratchet fine because the running max is one extra state variable. An autocall (Phase 3) adds an early-redemption decision on each observation date plus a knock-in barrier, and when the underlying is a basket or the model has stochastic vol the state space stops fitting in a lattice. That is where the project switches to Monte Carlo only, which is also what desks do.

## Phase 3: the autocallable note

This is the structured-products desk's core retail product. Per $100 of face:

- on each quarterly observation date, if the stock is at or above the coupon barrier (70% of the initial fixing), the note pays a coupon
- if the stock is at or above the autocall level (100%), the note redeems at par and ends
- at maturity (3 years), if it was never called: if the stock is below the knock-in barrier (60%), the investor gets $100 times S_T/S_0, i.e. takes the full loss; otherwise par
- everything is discounted at the bank's funding rate (risk-free plus a spread), because the note is unsecured bank debt

Code is in `pricers/autocall.py`. Monte Carlo only: each path is a sequence of yes/no checks on fixed dates, which is exactly the "if it's Tuesday it pays off" stack. Options supported: memory coupons (missed coupons paid later if the condition is met), a non-call period, and European (maturity only) or American (any close) knock-in.

One thing worth knowing: the price is linear in the coupon, so price = PV(principal) + coupon times PV(number of coupons paid). The fair coupon that prices the note at par is therefore a single division on the same simulated paths, not a root-finding problem. That is what `fair_coupon` returns.

### Results (S0 = 100, vol = 25%, r = 5%, div yield 1.5%, funding spread 1%)

- fair coupon: 2.88% per quarter, 11.5% per year
- probability of being called on the first observation date: 50%
- probability of ever being called early: 84%
- probability of taking a loss at maturity: 6.3%
- expected life: just under one year, on a three-year note

Variations, all from the same code:

- memory coupons: 10.5% per year (the investor gets more, so the coupon is lower)
- one-year non-call period: 10.1%
- knock-in at 70% instead of 60%: 12.9%
- knock-in at 50%: 9.9%
- American knock-in instead of European: 13.2%
- the 1% funding spread alone adds about 1.2% a year to the coupon

### What the plots show

**`outputs/autocall_fair_coupon.png`.** Fair annual coupon against the volatility of the underlying, for three knock-in levels. At 10% vol the coupon collapses to the bank's funding rate of 6%, because there is almost no chance of a loss and the note is just a bond. At 25% vol it is 11.5%, at 40% vol it is 23%. The headline coupon is the premium on the put the investor sold, plus the bank's funding rate. When a 12% coupon looks attractive, this chart is what it is actually paying you for.

**`outputs/autocall_greeks.png`.** Delta and gamma against spot, with the same note priced at two moments: mid-way between observation dates, and one day before one. Mid-period the Greeks are smooth. One day before a check date, delta spikes to over 2 shares per note at the coupon barrier (a digital payoff: $2.88 either arrives tomorrow or it doesn't), gamma flips sign across the barrier, and delta drops to zero just above the autocall level (if the note is called tomorrow, its value is a fixed cash amount and the stock no longer matters). This is the hedging problem in one picture. A desk short this note has to buy stock at 69 and sell it at 71 the day before every observation, and the hedge errors concentrate at the barriers.

**`outputs/autocall_vega.png`.** Vega on the trade date, against spot. Negative everywhere: the holder is short volatility, most of all around 75 to 80 where the knock-in and the coupon barrier both bite. The bank holding the other side is long vol from a book of these notes, which is why large retail autocall issuance (Korea, Japan, and increasingly the US) shows up in index vol markets.

**`outputs/autocall_call_profile.png`.** Where the note ends. Half the notes are called at the first observation, and the distribution of lives is heavily front-loaded. The retail pitch of "three years of 12% coupons" is, on average, "one year of 12% coupons and then your money back", which is why these get reissued constantly.

### Caveats that matter for Part A of the project

The model is Black-Scholes with flat vol, so it ignores skew. The investor is short a downside put, and real-world downside implied vol is higher than ATM vol, which means this pricer understates the value of what the investor gave up and overstates the fair coupon. Phase 4 (stochastic vol) is the fix. The correct funding spread is the issuer's own senior unsecured spread, not a round number. And a note's bank-stated estimated value is for the full product including fees; comparing it to this pricer needs the fee structure from the pricing supplement.

## Running it

```
pip install -r requirements.txt
python scripts/phase1_convergence.py     # ~30 s, writes three PNGs to outputs/
python scripts/phase2_ratchet.py         # ~20 s, writes three more
python scripts/phase3_autocall.py        # ~60 s, writes four more
python tests/test_lookback.py            # or python -m pytest tests/
python tests/test_ratchet.py
python tests/test_autocall.py
```

## References

- Goldman, Sosin, Gatto (1979), "Path Dependent Options: Buy at the Low, Sell at the High", Journal of Finance. The original closed form.
- Broadie, Glasserman, Kou (1999), "Connecting discrete and continuous path-dependent options", Finance and Stochastics. The discrete-monitoring correction.
- Cheuk and Vorst (1997), "Currency lookback options and observation frequency: a binomial approach". A change-of-numeraire trick that collapses the floating-strike lookback to a one-dimensional tree; a good next step if you want the tree to be faster.
- Hull, Options, Futures and Other Derivatives, chapter on exotic options, for the formula as implemented here.
