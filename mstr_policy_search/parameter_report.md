# MSTR Options Protection — Parameter Report

**Date:** 2026-03-12
**Scope:** Parametric policy search — optimized vs. fixed parameters, with best policy result.

---

## Code Version Reference

| Field | Value |
|---|---|
| Repository branch | `main` |
| Commit hash | `419d31086d6e432f8a49b334893dc01b96615f16` |
| Commit date | 2026-03-12 13:52 +0100 |
| Commit message | `new score = CVaR_20_improvement - 1.0 x E_drag` |
| Entry point | `mstr_policy_search/main.py` |

### Runtime environment

| Package | Version |
|---|---|
| Python | 3.14.3 |
| numpy | 2.4.2 |
| scipy | 1.17.0 |
| yfinance | 1.2.0 |
| tabulate | 0.10.0 |

### Reproduce this run

```bash
git checkout 419d31086d6e432f8a49b334893dc01b96615f16
cd mstr_policy_search
python main.py --sanity    # quick sanity check (~30 sec)
python main.py             # full optimization with BTC-era returns (~17 min)
python main.py --refresh   # force live option chain fetch before running
```

Cached data files used (no live fetch required by default):

- `mstr_options_cache.csv` — option chain, spot = $133.53
- `mstr_returns_cache.csv` — MSTR log-returns, full history (6,976 days)

---

## Optimized Parameters — Policy Search Space

Parameters searched via three-phase coordinate ascent. Each parameter is independently varied
while all others are held at their current best, until no single-step improvement is found.

Two search runs were executed. Run 1 is the committed baseline. Run 2 extended the ranges of
parameters whose best value fell at a grid boundary in Run 1, to confirm the optimum is interior.
Both runs returned the identical best policy — see robustness note below.

| Parameter | Description | Run 1 grid (committed) | Run 2 grid (extended) | N (R1/R2) | **Best** | Boundary in R1? |
|---|---|---|---|:---:|:---:|:---:|
| `alpha_L` | Long put strike / S(t) | 0.90 … 1.25 (step 0.05) | 1.05, 1.10, 1.15, 1.20, 1.25 | 8 / 5 | **1.15** | No (interior) |
| `alpha_S1` | Short put strike / S(t) | 0.40 … 0.75 (step 0.05) | 0.40, 0.45, 0.50, 0.55 | 8 / 4 | **0.45** | No (2nd from low) |
| `T_L` | Long put expiry (days) | all liquid expiries | same | 20 / 20 | **34** | No |
| `T_S1` | Short put expiry (days) | all liquid expiries | same | 20 / 20 | **467** | No |
| `q1` | Short puts per 1 long put | 1, 2, 3 | **0**, 1, 2 | 3 / 3 | **1** | **Yes — was lower bound** |
| `beta` | HWM ratchet weight | 0.00, 0.25, 0.50, 0.75, 1.00 | same | 5 / 5 | **0.50** | No (centre) |
| `eta_pct` | Net premium bound / S | -20%, -15%, -10%, -5%, 0%, +5% | -20%, -15%, -10%, -5% | 6 / 4 | **-15%** | No (2nd from low) |
| `d_min_short` | Min DTE, short roll | 14, 21, 28, **35** | 28, **35**, 42, 60, 90 | 4 / 5 | **35** | **Yes — was upper bound** |
| `d_min_long` | Min DTE, long roll | 7, 14, **21** | 14, **21**, 28, 35, 42, 60, 90 | 3 / 7 | **21** | **Yes — was upper bound** |

### Robustness note — boundary confirmation

Three parameters were at a grid boundary in Run 1. Run 2 extended the grid beyond each boundary:

| Parameter | Run 1 best position | Extension added | Run 2 result |
|---|---|---|---|
| `q1 = 1` | Lower bound (grid started at 1) | Added q1 = 0 (no short put) below | q1 = 1 confirmed interior |
| `d_min_short = 35` | Upper bound (grid ended at 35) | Added 42, 60, 90 above | 35 confirmed interior |
| `d_min_long = 21` | Upper bound (grid ended at 21) | Added 28, 35, 42, 60, 90 above | 21 confirmed interior |

**Conclusion:** the optimal policy is a genuine interior optimum across all searched parameters.
It is not an artifact of a truncated grid.

### Liquid chain expiries available to T_L and T_S1

Expiries are filtered to those with bid-ask spread <= 25% of mid price (liquidity filter).
At fetch date, 20 expiries passed the filter:

```
6, 13, 20, 26, 34, 41, 69, 103, 132, 167, 195, 223, 286, 314, 467, 559, 650, 685, 832, 1014
              ▲                                                   ▲
          T_L = 34 (best)                                   T_S1 = 467 (best)
```

### Best policy translated to dollar strikes

Spot at fetch date: **$133.53**

| Leg | Formula | Strike | Interpretation |
|---|---|---|---|
| Long put K_L | 133.53 x 1.15 | **$153.56** | 15% in-the-money |
| Short put K_S1 | 133.53 x 0.45 | **$60.09** | 55% out-of-the-money |

---

## Fixed Model Parameters — Held Constant Throughout

These parameters are set in `main.py` as module-level globals and are not part of the search space.

| Parameter | Description | Value | Notes |
|---|---|---|---|
| `DELTA_FLOOR` | Floor fraction of reference price | **0.80** | Floor = 0.80 x [(1-beta) x S0 + beta x H(t)] |
| `LAMBDA` | Drag penalty weight in score formula | **1.0** | Score = CVaR_20_imp - 1.0 x E_drag |
| `HORIZON_DAYS` | Simulation length | **504 days** | Approximately 2 trading years |
| `ANNUAL_DEFAULT_PROB` | Company default probability | **0.1% / year** | ~0.20% of 504-day paths crash to $0.01 permanently |
| `COST_PER_LEG` | Transaction cost per contract leg | **$1.00** | Total per roll event = (1 + q1) x $1 |
| `SPREAD_PCT_MAX` | Liquidity filter cutoff | **25%** | Max bid-ask spread / mid price for a valid expiry |
| `BTC_ERA_ONLY` | Bootstrap pool restriction | **True** | Returns drawn only from 2020-08-11 onward |
| `BTC_ERA_CUTOFF` | Bootstrap start date | **2020-08-11** | MicroStrategy first BTC purchase date |

### Bootstrap return pool statistics (BTC-era)

| Statistic | Value |
|---|---|
| Days in pool | 1,399 |
| Mean daily log-return | +0.17% |
| Daily volatility | 5.87% |
| Min daily return | -29.5% |
| Max daily return | +25.6% |

---

## Optimizer Settings — Also Fixed

| Parameter | Value | Role |
|---|---|---|
| `N_INIT` | 50 | Random policies evaluated in Phase 1 |
| `K_STARTS` | 15 | Top Phase-1 seeds fed into Phase-2 coordinate ascent |
| `N_SEARCH` | 500 (seed = 42) | CRN paths shared by Phase 1 and Phase 2 |
| `N_FINE` | 2,000 (seed = 137) | Fresh paths for Phase-3 fine verification |
| `MIN_P_SUCCESS` | 0.0 | No hard floor-breach filter; score drives ranking |
| Score formula | CVaR_20_imp - LAMBDA x E_drag | Both terms in terminal-wealth improvement (USD) |

---

## Best Policy Summary

```
Structure
  Long put:   K_L  = S(t) x 1.15   expire in 34 trading days, rolling
  Short put:  K_S1 = S(t) x 0.45   expire in 467 trading days  (~1 roll at day 432)
  Quantity:   1 long  :  1 short

Floor
  Floor(t) = 0.80 x [0.50 x S0 + 0.50 x H(t)]   (partial high-watermark ratchet, beta=0.50)

Premium constraint
  Net option cash flow per roll >= -15% x S(t)    (slight net debit allowed)

Roll triggers
  Short: T_S1_remaining <= 35 days                (calendar, path-independent)
  Long:  T_L_remaining  <= 21 days                (calendar, path-independent)
     OR  K_L < Floor(t)  AND  S(t) x 1.15 > K_L  AND  cooldown >= 21 days
                                                   (floor trigger, path-dependent via H(t))
```

### Performance metrics (Phase-3 fine paths, N = 2,000, seed = 137)

| Metric | Value | Description |
|---|---|---|
| Score | **+134.11** | CVaR_20_imp - 1.0 x E_drag |
| CVaR_20_improvement | **+172.72** | Mean (W + Pi_T) in bottom 20% of paths by terminal S_T |
| E_drag | **38.61** | Mean max(-(W + Pi_T), 0) in top 80% of paths by terminal S_T |
| E_imp | **+302.61** | Unconditional mean improvement over buy-and-hold across all paths |
| E_W | **+206.59** | Mean net option cash flows only (stock excluded) |
| P_success | **90.1%** | Fraction of paths where floor is never breached |
| Mean rolls per path | **39.0** | Roll events (long calendar rolls dominate) |

### Benchmark comparison (same 2,000 fine paths)

| Strategy | CVaR_20_imp | E_drag | E_imp | Mean trades |
|---|---:|---:|---:|---:|
| Buy-and-hold | 0.00 | 0.00 | 0.00 | -- |
| Stop-loss (sell below floor, buy back above) | +10.45 | 113.26 | -88.52 | 4.1 |
| Options overlay (best policy) | **+172.72** | **38.61** | **+302.61** | 39.0 |

- **CVaR_20_imp:** mean improvement over BH in the 20% of paths with the worst terminal S_T (crash paths).
- **E_drag:** mean loss vs BH in the top 80% of paths (bull paths) — lower is better.
- **E_imp:** unconditional mean improvement over BH across all paths.
- Stop-loss E_imp = -$88.52 means stop-loss is unconditionally worse than buy-and-hold in the BTC era,
  due to whipsaw from MSTR's 5.87% daily volatility combined with positive drift.

---

---

## Version History

| Run | Commit | Date | Grid change | Result |
|---|---|---|---|---|
| Run 1 — baseline | `419d310` | 2026-03-12 | Initial grid | Best policy found |
| Run 2 — robustness | uncommitted | 2026-03-12 | Extended q1, d_min_short, d_min_long beyond prior bounds | Identical best policy confirmed |

*Report generated: 2026-03-12 | Baseline commit: `419d310`*
