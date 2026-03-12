#!/usr/bin/env python3
"""
main.py -- Entry point for MSTR parametric policy search (v4).

Run:
    cd mstr_policy_search
    python main.py             # uses cached chain + cached returns
    python main.py --refresh   # force live yfinance fetch
    python main.py --sanity    # single-policy sanity check only (fast)

ALL tunable parameters live here as module-level globals.
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
DELTA_FLOOR         = 0.80            # floor = DELTA_FLOOR × reference_price
                                      # reference = (1-beta)×S0 + beta×H(t)
                                      # beta=0 → fixed $DELTA_FLOOR×S0 (never ratchets)
                                      # beta=1 → trailing DELTA_FLOOR×H(t) (full HWM)

# ── Crash overlay ─────────────────────────────────────────────────────────────
ANNUAL_DEFAULT_PROB = 0.001           # 0.1% annual → ~0.20% of 504d paths crash to $0.01

# ── Transaction costs ─────────────────────────────────────────────────────────
COST_PER_LEG        = 1.0             # $ per contract leg per roll

# ── Score parameters ───────────────────────────────────────────────────────────
LAMBDA              = 1.0             # drag penalty in: score = CVaR_20_imp - LAMBDA * E_drag
                                      # 1.0 = equal dollar weight on crash improvement vs bull drag

# ── Liquidity filter ──────────────────────────────────────────────────────────
SPREAD_PCT_MAX      = 25.0            # max bid-ask spread % for a leg to be considered liquid

# ── Search space (coord ascent cycles over these) ─────────────────────────────
ALPHA_L_VALUES    = [ 0.90, 0.95, 1.00, 1.05, 1.10, 1.15,1.20,1.25]
ALPHA_S1_VALUES   = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
Q1_VALUES         = [1, 2, 3]
BETA_VALUES       = [0.0, 0.25, 0.50, 0.75, 1.0]
ETA_PCT_VALUES    = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05]
DMIN_SHORT_VALUES = [ 14, 21,28,35]
DMIN_LONG_VALUES  = [7, 14, 21]
# T values come from the live chain — all liquid expiries are used

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
    T_L and T_S1 values come from ALL liquid chain expiries (no ordering constraint).
    Returns the frozen_expiries list for use in the simulator's T-snap logic.
    """
    import pandas as pd
    chain_df = pd.DataFrame(chain)
    all_expiries = sorted(
        chain_df[chain_df["spread_pct"] <= SPREAD_PCT_MAX]["days_out"].unique()
    )

    policy_grid.ALPHA_L_VALUES    = ALPHA_L_VALUES
    policy_grid.ALPHA_S1_VALUES   = ALPHA_S1_VALUES
    policy_grid.T_L_VALUES        = [int(d) for d in all_expiries]
    policy_grid.T_S1_VALUES       = [int(d) for d in all_expiries]
    policy_grid.Q1_VALUES         = Q1_VALUES
    policy_grid.BETA_VALUES       = BETA_VALUES
    policy_grid.ETA_PCT_VALUES    = ETA_PCT_VALUES
    policy_grid.DMIN_SHORT_VALUES = DMIN_SHORT_VALUES
    policy_grid.DMIN_LONG_VALUES  = DMIN_LONG_VALUES
    policy_grid.SPREAD_PCT_MAX    = SPREAD_PCT_MAX

    print(f"  Chain expiries (liquid, {len(all_expiries)} total): {all_expiries[:10]}...")
    return [int(d) for d in all_expiries]


def run_sanity_check(surface, spot: float, returns: np.ndarray) -> dict:
    """
    Single-policy sanity check on 100 paths with verbose trace.
    Uses a near-ATM 282d long put / 37d 75%-OTM short put (beta=1 trailing HWM).
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK -- single policy, 100 paths, verbose trace")
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

    sampler   = BootstrapSampler(returns, seed=99)
    paths_100 = sampler.sample_paths(
        100, HORIZON_DAYS, spot, annual_default_prob=ANNUAL_DEFAULT_PROB
    )

    result = simulate_policy(
        test_policy, paths_100, surface, spot,
        delta_floor=DELTA_FLOOR, cost_per_leg=COST_PER_LEG, verbose=True,
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


def main() -> None:
    refresh     = "--refresh" in sys.argv
    sanity_only = "--sanity"  in sys.argv

    # ── 1. Load data ──────────────────────────────────────────────────────────
    chain, spot, fetch_date = data_loader.load_option_chain("MSTR", refresh=refresh)
    returns = data_loader.load_historical_returns(
        "MSTR", refresh=refresh,
        cutoff_date=BTC_ERA_CUTOFF if BTC_ERA_ONLY else None,
    )

    # ── 2. Build frozen surface ───────────────────────────────────────────────
    print(f"\nBuilding frozen option surface ...")
    surface = build_surface(chain, spot)
    print(f"  Surface built from {len(chain)} contracts  (spot=${spot})")

    # ── 3. Push search space into policy_grid ─────────────────────────────────
    print(f"\nConfiguring policy_grid search space ...")
    frozen_expiries = _push_config_to_policy_grid(chain)

    # ── 4. Sanity check ───────────────────────────────────────────────────────
    sanity = run_sanity_check(surface, spot, returns)

    if sanity_only:
        print("\n--sanity flag set. Exiting before full optimization.")
        return

    if sanity["E_W"] == 0.0 and sanity["n_rolls"] == 0.0:
        print("\nERROR: Sanity check returned all-zero results. Surface lookup broken.")
        sys.exit(1)

    # ── 5. Generate bootstrap paths ───────────────────────────────────────────
    print(f"\nGenerating bootstrap paths ...")
    sampler_search = BootstrapSampler(returns, seed=42)
    sampler_fine   = BootstrapSampler(returns, seed=137)
    paths_search = sampler_search.sample_paths(
        N_SEARCH, HORIZON_DAYS, spot, annual_default_prob=ANNUAL_DEFAULT_PROB
    )
    paths_fine = sampler_fine.sample_paths(
        N_FINE, HORIZON_DAYS, spot, annual_default_prob=ANNUAL_DEFAULT_PROB
    )
    print(f"  Search : {paths_search.shape}  ({N_SEARCH} paths x {HORIZON_DAYS}d, seed=42)")
    print(f"  Fine   : {paths_fine.shape}    ({N_FINE} paths x {HORIZON_DAYS}d, seed=137)")
    print(f"  Crash overlay: ANNUAL_DEFAULT_PROB={ANNUAL_DEFAULT_PROB*100:.3f}%"
          f"  (~{ANNUAL_DEFAULT_PROB*HORIZON_DAYS/252*100:.2f}% of fine paths crash)")

    # ── 6. Run optimization ───────────────────────────────────────────────────
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
    )

    # ── 7. Benchmark: stop-loss on the same fine paths ────────────────────────
    print(f"\nComputing stop-loss benchmark on {N_FINE:,} fine paths ...")
    sl_metrics = simulate_stop_loss(paths_fine, spot, delta_floor=DELTA_FLOOR)
    print(f"  Stop-loss: CVaR_20_imp={sl_metrics['CVaR_20_imp']:+.2f}  "
          f"E_drag={sl_metrics['E_drag']:.2f}  "
          f"E_imp={sl_metrics['E_improvement']:+.2f}  "
          f"mean_trades={sl_metrics['n_trades']:.1f}")

    benchmarks = {"stop_loss": sl_metrics}

    # ── 8. Report ─────────────────────────────────────────────────────────────
    results_module.report(ranked, spot, delta_floor=DELTA_FLOOR,
                          lambda_=LAMBDA, benchmarks=benchmarks)


if __name__ == "__main__":
    main()
