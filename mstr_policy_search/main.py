#!/usr/bin/env python3
"""
main.py -- Entry point for MSTR parametric policy search (v4).

Run:
    cd mstr_policy_search
    python main.py             # uses cached chain + cached returns
    python main.py --refresh   # force live yfinance fetch
    python main.py --sanity    # single-policy sanity check only (fast)

ALL tunable parameters live here as module-level globals.

Drift sweep
-----------
When DRIFT_ADJ_VALUES contains more than one entry the optimiser runs once
per drift level and a cross-scenario summary table is printed at the end.
The BTC-era raw mean is +0.17%/day (+42.8%/yr).  Subtracting drift_adj
shifts the simulated mean without changing volatility or serial structure:

    drift_adj =  0.0000  ->  +0.17%/day  (+42.8%/yr)  raw historical
    drift_adj = -0.0005  ->  +0.12%/day  (+30.2%/yr)  mild headwind
    drift_adj = -0.0010  ->  +0.07%/day  (+17.6%/yr)  moderate headwind
    drift_adj = -0.0020  ->  -0.03%/day  ( -7.6%/yr)  severe stress

Results for each scenario are saved with a tag suffix, e.g.
    best_policy_drift_+0.00pct.json
    top_policies_drift_-0.10pct.csv
"""
import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import data_loader
from surface import build_surface
from bootstrap import BootstrapSampler
import policy_grid
from policy_grid import Policy
from simulator import simulate_policy, simulate_stop_loss
import optimizer as opt_module
import results as results_module

# ══════════════════════════════════════════════════════════════════════════════
#  ALL TUNABLE PARAMETERS  —  edit only this block
# ══════════════════════════════════════════════════════════════════════════════

# ── Simulation horizon ────────────────────────────────────────────────────────
HORIZON_DAYS        = 504             # 2 trading years

# ── Bootstrap return pool ─────────────────────────────────────────────────────
BTC_ERA_ONLY        = True            # True = BTC-era only (post-2020-08-11)
BTC_ERA_CUTOFF      = "2020-08-11"   # MicroStrategy's first BTC purchase date
                                      # BTC-era : N=1399 days, mean=+0.17%/day, std=5.87%/day
                                      # Full history: N=6976 days, includes -96% dot-com crash

# ── Floor specification ───────────────────────────────────────────────────────
DELTA_FLOOR         = 0.80            # floor = DELTA_FLOOR x reference_price
                                      # reference = (1-beta)*S0 + beta*H(t)
                                      # beta=0 -> fixed DELTA_FLOOR*S0 (never ratchets)
                                      # beta=1 -> trailing DELTA_FLOOR*H(t) (full HWM)

# ── Crash overlay ─────────────────────────────────────────────────────────────
ANNUAL_DEFAULT_PROB = 0.001           # 0.1% annual -> ~0.20% of 504d paths crash to $0.01

# ── Vol cap ────────────────────────────────────────────────────────────────────
# Caps EGARCH conditional sigma_t in simulation.
# Full-history max rolling 21d vol ever observed for MSTR: 391.7%/yr (dot-com era).
# Cap is set just above that absolute maximum to prevent super-unit-root (alpha+beta=1.165)
# variance explosions that have no empirical basis.
VOL_CAP_ANN = 400.0                   # %/yr  (None = no cap)

# ── Transaction costs ─────────────────────────────────────────────────────────
COST_PER_LEG        = 1.0             # $ per contract leg per roll

# ── Score parameters ───────────────────────────────────────────────────────────
LAMBDA              = 1.0             # drag penalty in: score = CVaR_20_imp - LAMBDA * E_drag
                                      # 1.0 = equal dollar weight on crash improvement vs bull drag

# ── Liquidity filter ──────────────────────────────────────────────────────────
SPREAD_PCT_MAX      = 25.0            # max bid-ask spread % for a leg to be considered liquid

# ── Bootstrap mode ────────────────────────────────────────────────────────────
BLOCK_SIZE          = 10              # circular block bootstrap block size (trading days)
                                      # 1  = i.i.d. (v0.10 baseline)
                                      # 10 = 2-week blocks (preserves GARCH vol clustering)
                                      # 21 = 1-month blocks

# ── EGARCH path simulation ─────────────────────────────────────────────────────
USE_EGARCH          = True            # True  = EGARCH(1,1) + t_5 paths (vol clustering persists
                                      #         across the entire path; more realistic regime dynamics)
                                      # False = block bootstrap (original behaviour; backward-compat)
VOL_SCALE_ROLL      = False           # True  = vol-scaled Black-Scholes at roll pricing events
                                      # False = frozen surface / dynamic IV (see USE_DYNAMIC_IV)
                                      # Must be False when USE_DYNAMIC_IV = True
USE_DYNAMIC_IV      = True            # True  = dynamic-IV BS pricing: IV(K,T,t) = IV_initial(K/S_t,T)
                                      #                                             + IV_BETA*log(S_t/S0)
                                      # False = frozen surface prices at all pricing points
IV_BETA             = -1.0            # Leverage coefficient (negative → IV rises in crashes).
                                      # Typical range for equities: -0.5 to -2.0.
                                      # -1.0 = 1% log-price fall → +1pp IV rise (annualised)

# ── Drift adjustment sweep ────────────────────────────────────────────────────
# Constant subtracted from every bootstrapped log-return (daily units).
# The raw BTC-era mean is +0.17%/day.  Each entry below shifts that mean:
#   0.0000  -> +0.17%/day  (+42.8%/yr)  raw historical (no adjustment)
#  -0.0005  -> +0.12%/day  (+30.2%/yr)  mild headwind
#  -0.0010  -> +0.07%/day  (+17.6%/yr)  moderate headwind
#  -0.0020  -> -0.03%/day  ( -7.6%/yr)  severe stress / net bear market
# Set to [0.0] to reproduce a single-scenario run (backward compatible).
DRIFT_ADJ_VALUES    = [0.0]             # 0.0 = raw BTC-era drift (+42.8%/yr); no scenario bias

# ── Search space (coord ascent cycles over these) ─────────────────────────────
# Ranges refined from drift-sweep boundary analysis (drift=-0.002, -7.5%/yr):
#   Interior params -> shrink to [best-1step, best, best+1step]
#   Boundary params -> extend to [best-1step, best, best+1step, best+2step]
#
# Parameter   | prev range                     | best | status    | new range
# ------------|--------------------------------|------|-----------|------------------------
# alpha_L     | [1.05,1.10,1.15,1.20,1.25]    | 1.15 | interior  | [1.10, 1.15, 1.20]
# alpha_S1    | [0.40,0.45,0.50,0.55]         | 0.55 | upper bnd | [0.50, 0.55, 0.60, 0.65]
# q1          | [0,1,2]                        | 2    | upper bnd | [1, 2, 3, 4]
# beta        | [0.0,0.25,0.50,0.75,1.0]      | 0.50 | interior  | [0.25, 0.50, 0.75]
# eta_pct     | [-0.20,-0.15,-0.10,-0.05]     |-0.10 | interior  | [-0.15,-0.10,-0.05]
# d_min_short | [28,35,42,60,90]               | 90   | upper bnd | fixed [90] (irrelevant:
#             |                                |      |           | T_S1=685>504d, never rolls)
# d_min_long  | [14,21,28,35,42,60,90]         | 28   | interior  | [21, 28, 35]
# T_L         | all chain                      | 34   | interior  | [26, 34, 41]  (see override)
# T_S1        | all chain                      | 685  | interior  | [650, 685, 832] (see override)
ALPHA_L_VALUES    = [0.90, 0.95, 1.00]               # run7: alpha_L=0.95 confirmed interior
ALPHA_S1_VALUES   = [0.10, 0.15, 0.20]              # run7: alpha_S1=0.15 confirmed interior
Q1_VALUES         = [1]                               # frozen
BETA_VALUES       = [0.75, 1.00]                      # 1.0 natural ceiling
ETA_PCT_VALUES    = [-0.50, -0.40, -0.30]            # run7: extend down to confirm degeneracy
DMIN_SHORT_VALUES = [1]                               # natural floor; frozen
DMIN_LONG_VALUES  = [1, 7, 14]                        # run7: d_min_long=7 confirmed interior

# ── T expiry overrides (None = use all liquid chain expiries) ──────────────────
T_L_VALUES_OVERRIDE  = [223, 286, 314]               # run7: T_L=286 confirmed interior
T_S1_VALUES_OVERRIDE = [223, 286, 314]               # run7: T_S1=286 confirmed interior


# ── Coord-ascent parameters ───────────────────────────────────────────────────
N_INIT            = 50               # random policies evaluated in Phase 1
K_STARTS          = 15               # top Phase-1 seeds fed into Phase 2 coord ascent
N_SEARCH          = 500              # paths for Phase 1+2 (CRN — same paths for all evals)
N_FINE            = 2_000            # paths for Phase 3 fine verification
MIN_P_SUCCESS     = 0.0              # 0.0 = no hard filter; score (CVaR_20_imp - E_drag) drives ranking

# ══════════════════════════════════════════════════════════════════════════════


def _push_config_to_policy_grid(chain: list[dict]) -> list[int]:
    """
    Propagate all search-space lists from main.py into the policy_grid module.
    T_L and T_S1 come from all liquid chain expiries unless T_L_VALUES_OVERRIDE
    or T_S1_VALUES_OVERRIDE are set, in which case those explicit lists are used.
    Returns the full frozen_expiries list (all liquid days) for the simulator.
    """
    import pandas as pd
    chain_df = pd.DataFrame(chain)
    all_expiries = sorted(
        chain_df[chain_df["spread_pct"] <= SPREAD_PCT_MAX]["days_out"].unique()
    )

    t_l_values  = T_L_VALUES_OVERRIDE  if T_L_VALUES_OVERRIDE  is not None \
                  else [int(d) for d in all_expiries]
    t_s1_values = T_S1_VALUES_OVERRIDE if T_S1_VALUES_OVERRIDE is not None \
                  else [int(d) for d in all_expiries]

    policy_grid.ALPHA_L_VALUES    = ALPHA_L_VALUES
    policy_grid.ALPHA_S1_VALUES   = ALPHA_S1_VALUES
    policy_grid.T_L_VALUES        = t_l_values
    policy_grid.T_S1_VALUES       = t_s1_values
    policy_grid.Q1_VALUES         = Q1_VALUES
    policy_grid.BETA_VALUES       = BETA_VALUES
    policy_grid.ETA_PCT_VALUES    = ETA_PCT_VALUES
    policy_grid.DMIN_SHORT_VALUES = DMIN_SHORT_VALUES
    policy_grid.DMIN_LONG_VALUES  = DMIN_LONG_VALUES
    policy_grid.SPREAD_PCT_MAX    = SPREAD_PCT_MAX

    override_note = ""
    if T_L_VALUES_OVERRIDE is not None:
        override_note += f"  T_L  overridden: {t_l_values}\n"
    if T_S1_VALUES_OVERRIDE is not None:
        override_note += f"  T_S1 overridden: {t_s1_values}\n"

    print(f"  Chain expiries (liquid, {len(all_expiries)} total): {all_expiries[:10]}...")
    if override_note:
        print(override_note, end="")

    return [int(d) for d in all_expiries]


def run_sanity_check(
    surface,
    spot:          float,
    returns:       np.ndarray,
    egarch_params: dict | None = None,
    rvol_base:     float | None = None,
) -> dict:
    """
    Single-policy sanity check on 100 paths with verbose trace.
    Uses a near-ATM 282d long put / 37d 75%-OTM short put (beta=1 trailing HWM).
    Always runs with drift_adj=0.0 (raw returns) — this is a surface/simulator
    validation step, not a scenario-specific optimization.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK -- single policy, 100 paths, verbose trace")
    mode = "EGARCH" if (USE_EGARCH and egarch_params is not None) else f"bootstrap(block={BLOCK_SIZE})"
    vs   = ("vol-scaled BS at roll" if VOL_SCALE_ROLL
            else f"dynamic-IV BS (beta={IV_BETA})" if USE_DYNAMIC_IV
            else "frozen surface only")
    print(f"  Path gen : {mode}   Pricing: {vs}")
    print("=" * 60)

    chain_days = sorted(surface._df["days_out"].unique())

    def snap(target: int) -> int:
        return min(chain_days, key=lambda d: abs(d - target))

    T_L_snap  = snap(282)
    T_S1_snap = snap(37)

    test_policy = Policy(
        alpha_L=0.97, alpha_S1=0.75,
        T_L=T_L_snap, T_S1=T_S1_snap,
        base_q1=1, beta=1.0,
        d_min_short=7, d_min_long=7,
        eta_pct=0.0,
    )

    sampler = BootstrapSampler(returns, seed=99)
    if USE_EGARCH and egarch_params is not None:
        sampler.set_egarch_params(egarch_params)

    paths_100 = sampler.sample_paths(
        100, HORIZON_DAYS, spot, annual_default_prob=ANNUAL_DEFAULT_PROB,
        block_size=BLOCK_SIZE,
        use_egarch=(USE_EGARCH and egarch_params is not None),
        vol_cap_ann=VOL_CAP_ANN,
        # drift_adj=0.0 intentionally (sanity = surface/simulator check, not scenario)
    )

    result = simulate_policy(
        test_policy, paths_100, surface, spot,
        delta_floor=DELTA_FLOOR, cost_per_leg=COST_PER_LEG, verbose=True,
        rvol_base=rvol_base if VOL_SCALE_ROLL else None,
        vol_scale_roll=VOL_SCALE_ROLL,
        use_dynamic_iv=USE_DYNAMIC_IV and not VOL_SCALE_ROLL,
        iv_beta=IV_BETA,
    )

    print(f"\n-- Sanity result (100 paths, seed=99) --")
    print(f"  P_success:    {result['P_success']*100:.1f}%")
    print(f"  E_W:         {result['E_W']:+.2f}")
    print(f"  CVaR_W:      {result['CVaR_W']:+.2f}")
    print(f"  P(W>0):       {result['P_W_positive']*100:.1f}%")
    print(f"  Mean rolls:   {result['n_rolls']:.1f}")
    print(f"  Breach depth: {result['E_breach_depth']:.2f}")
    print(f"  CVaR_20_imp:  {result['CVaR_20_improvement']:+.2f}")
    print(f"  E_drag:       {result['E_drag']:.2f}")
    print(f"  Score:        {result['CVaR_20_improvement'] - LAMBDA * result['E_drag']:+.4f}")
    return result


def _drift_tag(drift_adj: float) -> str:
    """Return a filesystem-safe tag string for a given drift_adj value."""
    return f"_drift_{drift_adj*100:+.2f}pct"


def _print_drift_summary(summary: list[dict], raw_mean_daily: float) -> None:
    """Print a compact cross-scenario comparison table."""
    from tabulate import tabulate

    print("\n" + "=" * 100)
    print("DRIFT SENSITIVITY SUMMARY")
    print(f"  Raw BTC-era mean: {raw_mean_daily*100:+.4f}%/day  "
          f"({raw_mean_daily*252*100:+.1f}%/yr)  |  "
          f"block_size={BLOCK_SIZE}  |  N_fine={N_FINE:,}")
    print("=" * 100)

    rows = []
    for s in summary:
        p = s["best_policy"]
        rows.append([
            f"{s['drift_adj']*100:+.4f}",
            f"{s['adj_mean_pct_daily']:+.4f}",
            f"{s['adj_mean_pct_annual']:+.1f}",
            f"{p.alpha_L:.2f}",
            p.base_q1,
            p.T_L,
            p.T_S1,
            f"{p.beta:.2f}",
            f"{s['P_success']*100:.1f}%",
            f"{s['CVaR_20_imp']:+.2f}",
            f"{s['E_drag']:.2f}",
            f"{s['score']:+.4f}",
            f"{s['sl_CVaR_imp']:+.2f}",
            f"{s['sl_E_drag']:.2f}",
        ])

    headers = [
        "drift_adj%/d", "adj_mean%/d", "adj_mean%/yr",
        "aL", "q1", "T_L", "T_S1", "beta",
        "P_succ", "CVaR20_imp", "E_drag", "Score",
        "SL_CVaR20", "SL_drag",
    ]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()
    print("  drift_adj%/d  : constant subtracted from each bootstrapped log-return (daily %)")
    print("  adj_mean%/yr  : resulting simulated drift, annualised")
    print("  CVaR20_imp    : options overlay mean improvement in bottom-20% crash paths")
    print("  E_drag        : options overlay mean cost in top-80% bull paths")
    print("  Score         : CVaR20_imp - LAMBDA * E_drag")
    print("  SL_CVaR20/drag: same metrics for the stop-loss benchmark")
    print("=" * 100)

    # Save to CSV
    import csv
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "drift_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  Drift summary saved to {csv_path}")


def main() -> None:
    refresh     = "--refresh" in sys.argv
    sanity_only = "--sanity"  in sys.argv

    # ── 1. Load data (once, shared across all drift scenarios) ────────────────
    chain, spot, fetch_date = data_loader.load_option_chain("MSTR", refresh=refresh)
    returns = data_loader.load_historical_returns(
        "MSTR", refresh=refresh,
        cutoff_date=BTC_ERA_CUTOFF if BTC_ERA_ONLY else None,
    )
    raw_mean_daily = float(returns.mean())
    raw_std_daily  = float(returns.std())
    print(f"\nReturn pool stats (BTC-era={BTC_ERA_ONLY}):")
    print(f"  N={len(returns)} days  "
          f"mean={raw_mean_daily*100:+.4f}%/d ({raw_mean_daily*252*100:+.1f}%/yr)  "
          f"std={raw_std_daily*100:.4f}%/d")

    # ── 1b. Fit EGARCH(1,1) once on the return pool (optional) ────────────────
    egarch_params: dict | None = None
    rvol_base:     float | None = None
    if USE_EGARCH:
        from egarch import fit_egarch
        print(f"\nFitting EGARCH(1,1) with t_5 innovations ...")
        egarch_params = fit_egarch(returns, verbose=True)
        rvol_base     = None   # will be set from surface ATM IV below (after surface build)

    # ── 2. Build frozen surface (once) ────────────────────────────────────────
    print(f"\nBuilding frozen option surface ...")
    surface = build_surface(chain, spot)
    print(f"  Surface built from {len(chain)} contracts  (spot=${spot})")

    # Compute vol-scaling reference from surface ATM IV (T=34d), NOT trailing realized vol.
    # vol_scale = rvol_21d / iv_base, so iv_base defines where vol_scale=1.0.
    # Using returns[-21:].std() (~128%/yr) would deflate all roll prices by ~27%
    # in normal EGARCH conditions. The surface embeds ~79%/yr ATM IV; that is the
    # correct baseline so that "no vol regime shift" maps to vol_scale=1.0.
    if USE_EGARCH and VOL_SCALE_ROLL and surface._has_iv_surface:
        _iv_atm   = surface.iv_vector(np.array([spot]), np.array([34.0]), np.array([spot]))[0]
        rvol_base = float(_iv_atm / np.sqrt(252))
        print(f"  Vol-scaling reference: ATM IV (T=34d) = {_iv_atm*100:.1f}%/yr"
              f"  = {rvol_base*100:.3f}%/day  [iv_base from surface, not trailing realized vol]")

    # ── 3. Push search space into policy_grid (once) ──────────────────────────
    print(f"\nConfiguring policy_grid search space ...")
    frozen_expiries = _push_config_to_policy_grid(chain)

    # ── 4. Sanity check (once, no drift adjustment) ───────────────────────────
    sanity = run_sanity_check(surface, spot, returns,
                              egarch_params=egarch_params, rvol_base=rvol_base)

    if sanity_only:
        print("\n--sanity flag set. Exiting before full optimization.")
        return

    if sanity["E_W"] == 0.0 and sanity["n_rolls"] == 0.0:
        print("\nERROR: Sanity check returned all-zero results. Surface lookup broken.")
        sys.exit(1)

    # ── 5. Drift sweep ────────────────────────────────────────────────────────
    n_scenarios = len(DRIFT_ADJ_VALUES)
    print(f"\n{'='*70}")
    print(f"DRIFT SWEEP: {n_scenarios} scenario(s)  "
          f"drift values = {[f'{d*100:+.4f}%' for d in DRIFT_ADJ_VALUES]}")
    print(f"{'='*70}")

    scenario_summary = []

    for scenario_idx, drift_adj in enumerate(DRIFT_ADJ_VALUES):
        adj_mean_daily  = raw_mean_daily + drift_adj
        adj_mean_annual = adj_mean_daily * 252
        tag = ""   # drift sweep retired — single scenario, no suffix in filenames

        print(f"\n{'#'*70}")
        print(f"SCENARIO {scenario_idx+1}/{n_scenarios}:  "
              f"drift_adj={drift_adj*100:+.4f}%/day  "
              f"-> simulated mean={adj_mean_daily*100:+.4f}%/day  "
              f"({adj_mean_annual*100:+.1f}%/yr)")
        print(f"{'#'*70}")

        # -- 5a. Generate paths for this drift level ---------------------------
        _use_eg = USE_EGARCH and egarch_params is not None
        mode_str = "EGARCH(1,1)+t5" if _use_eg else f"bootstrap(block={BLOCK_SIZE})"
        print(f"\nGenerating {mode_str} paths (drift_adj={drift_adj*100:+.4f}%/d) ...")

        sampler_search = BootstrapSampler(returns, seed=42)
        sampler_fine   = BootstrapSampler(returns, seed=137)
        if _use_eg:
            sampler_search.set_egarch_params(egarch_params)
            sampler_fine.set_egarch_params(egarch_params)

        paths_search = sampler_search.sample_paths(
            N_SEARCH, HORIZON_DAYS, spot,
            annual_default_prob=ANNUAL_DEFAULT_PROB,
            block_size=BLOCK_SIZE,
            drift_adj=drift_adj,
            use_egarch=_use_eg,
            vol_cap_ann=VOL_CAP_ANN,
        )
        paths_fine = sampler_fine.sample_paths(
            N_FINE, HORIZON_DAYS, spot,
            annual_default_prob=ANNUAL_DEFAULT_PROB,
            block_size=BLOCK_SIZE,
            drift_adj=drift_adj,
            use_egarch=_use_eg,
            vol_cap_ann=VOL_CAP_ANN,
        )
        print(f"  Search : {paths_search.shape}  ({N_SEARCH} paths x {HORIZON_DAYS}d, seed=42)")
        print(f"  Fine   : {paths_fine.shape}    ({N_FINE} paths x {HORIZON_DAYS}d, seed=137)")
        print(f"  Path gen: {mode_str}")
        pricing_label = ("vol-scaled BS" if VOL_SCALE_ROLL
                         else f"dynamic-IV BS (beta={IV_BETA})" if USE_DYNAMIC_IV
                         else "frozen surface")
        print(f"  Roll pricing: {pricing_label}")
        print(f"  Crash overlay: ANNUAL_DEFAULT_PROB={ANNUAL_DEFAULT_PROB*100:.3f}%"
              f"  (~{ANNUAL_DEFAULT_PROB*HORIZON_DAYS/252*100:.2f}% of fine paths crash)")

        # -- 5b. Run optimization ----------------------------------------------
        sim_kwargs = {}
        if VOL_SCALE_ROLL and rvol_base is not None:
            sim_kwargs["rvol_base"]      = rvol_base
            sim_kwargs["vol_scale_roll"] = True
        if USE_DYNAMIC_IV and not VOL_SCALE_ROLL:
            sim_kwargs["use_dynamic_iv"] = True
            sim_kwargs["iv_beta"]        = IV_BETA

        ranked = opt_module.run_optimization(
            surface          = surface,
            spot0            = spot,
            paths_search     = paths_search,
            paths_fine       = paths_fine,
            delta_floor      = DELTA_FLOOR,
            cost_per_leg     = COST_PER_LEG,
            lambda_          = LAMBDA,
            n_init           = N_INIT,
            k_starts         = K_STARTS,
            min_p_success    = MIN_P_SUCCESS,
            rng_seed         = 0,
            frozen_expiries  = frozen_expiries,
            sim_kwargs       = sim_kwargs,
        )

        # -- 5c. Benchmark: stop-loss on same fine paths -----------------------
        print(f"\nComputing stop-loss benchmark on {N_FINE:,} fine paths ...")
        sl_metrics = simulate_stop_loss(paths_fine, spot, delta_floor=DELTA_FLOOR)
        print(f"  Stop-loss: CVaR_20_imp={sl_metrics['CVaR_20_imp']:+.2f}  "
              f"E_drag={sl_metrics['E_drag']:.2f}  "
              f"E_imp={sl_metrics['E_improvement']:+.2f}  "
              f"mean_trades={sl_metrics['n_trades']:.1f}")

        benchmarks = {"stop_loss": sl_metrics}

        # -- 5d. Report + save (tagged per scenario) ---------------------------
        results_module.report(ranked, spot, delta_floor=DELTA_FLOOR,
                              lambda_=LAMBDA, benchmarks=benchmarks, tag=tag)

        # -- 5e. Collect summary row -------------------------------------------
        best = ranked[0]
        scenario_summary.append({
            "drift_adj":          drift_adj,
            "adj_mean_pct_daily": adj_mean_daily  * 100,
            "adj_mean_pct_annual": adj_mean_annual * 100,
            "best_policy":        best["policy"],
            "score":              best["score"],
            "CVaR_20_imp":        best["CVaR_20_improvement"],
            "E_drag":             best["E_drag"],
            "P_success":          best["P_success"],
            "sl_CVaR_imp":        sl_metrics["CVaR_20_imp"],
            "sl_E_drag":          sl_metrics["E_drag"],
        })

    # ── 6. Cross-scenario summary ─────────────────────────────────────────────
    if len(scenario_summary) > 1:
        _print_drift_summary(scenario_summary, raw_mean_daily)
    else:
        # Single scenario: just confirm which drift was used
        s = scenario_summary[0]
        print(f"\nSingle scenario completed: drift_adj={s['drift_adj']*100:+.4f}%/d  "
              f"adj_mean={s['adj_mean_pct_annual']:+.1f}%/yr")


if __name__ == "__main__":
    main()
