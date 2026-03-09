#!/usr/bin/env python3
"""
main.py -- Entry point for MSTR parametric policy search.

Run:
    cd mstr_policy_search
    python main.py             # uses cached option chain + cached returns
    python main.py --refresh   # force live yfinance fetch for both caches
    python main.py --sanity    # single-policy sanity check only (fast)

Key parameters (edit here):
    ETA          -- max allowed initial net debit per share (default $15).
                    A long 130d ATM put costs ~$23; a single short 67d 75%-OTM
                    generates ~$6.  Most 1x1 structures cost $11-17 net; eta=15
                    admits the full single-short space while blocking naked longs.
    HORIZON_DAYS -- simulation horizon in trading days (504 = 2 years)
    N_COARSE     -- paths in Stage-1 coarse screen
    N_FINE       -- paths in Stage-2 fine evaluation
"""
import sys
import os

# Force UTF-8 stdout on Windows to handle Unicode in print statements
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

# Make imports work from this directory
sys.path.insert(0, os.path.dirname(__file__))

import data_loader
from surface import build_surface
from bootstrap import BootstrapSampler
import policy_grid
from simulator import simulate_policy
import optimizer as opt_module
import results as results_module
from policy_grid import Policy

# ── Tunable parameters ────────────────────────────────────────────────────────
ETA              = 15.0   # max initial net debit allowed per share (~11% of spot)
                          # Raise to 20+ to search more structures; lower to 5 for strict self-funding

HORIZON_DAYS     = 504    # 2 trading years (full evaluation horizon)

# Stage 1 (coarse): short horizon for speed — full grid screened in ~13 min
HORIZON_COARSE   = 63     # days for coarse screening (2 calendar months)
N_COARSE         = 200    # paths per policy in Stage 1

# Stage 2 (fine): full horizon on survivors
N_FINE           = 2_000  # paths per policy in Stage 2


def run_sanity_check(surface, spot, returns):
    """
    Single-policy sanity check on 100 paths -- prints trace for first 3 paths.
    Uses a near-ATM 270-day long put / 45-day 75%-OTM short put (N=1 structure).
    The sanity policy is intentionally NOT self-funding (it starts with a debit)
    so it exercises the roll mechanics in a realistic scenario.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK -- single policy, 100 paths, verbose trace")
    print("=" * 60)

    # Snap T values to nearest available chain expiries used by the grid
    chain_days = sorted(set(r["days_out"] for r in surface._df.to_dict("records")))
    def snap(target):
        return min(chain_days, key=lambda d: abs(d - target))

    T_L_snap  = snap(270)
    T_S1_snap = snap(45)

    test_policy = Policy(
        alpha_L=0.97, alpha_S1=0.75, alpha_S2=0.0,
        T_L=T_L_snap, T_S1=T_S1_snap, T_S2=0,
        base_q1=1, base_q2=0,
        gamma=0.0, d_min=14,
    )
    print(f"  Test policy: alpha_L=0.97  alpha_S1=0.75  T_L={T_L_snap}d  T_S1={T_S1_snap}d")

    sampler   = BootstrapSampler(returns, seed=99)
    paths_100 = sampler.sample_paths(100, HORIZON_DAYS, spot)

    result = simulate_policy(
        test_policy, paths_100, surface, spot, returns, verbose=True
    )

    print(f"\n-- Sanity result (100 paths) --")
    print(f"  P_success:    {result['P_success']*100:.1f}%")
    print(f"  E_W:         {result['E_W']:+.2f}")
    print(f"  CVaR_W:      {result['CVaR_W']:+.2f}")
    print(f"  P(W>0):       {result['P_W_positive']*100:.1f}%")
    print(f"  Mean rolls:   {result['n_rolls']:.1f}")
    print(f"  Breach depth: ${result['E_breach_depth']:.2f}")

    return result


def main():
    refresh     = "--refresh" in sys.argv
    sanity_only = "--sanity"  in sys.argv

    # ── 1. Load data ──────────────────────────────────────────────────────────
    chain, spot, fetch_date = data_loader.load_option_chain("MSTR", refresh=refresh)
    returns = data_loader.load_historical_returns("MSTR", refresh=refresh)

    # ── 2. Build frozen surface ───────────────────────────────────────────────
    print(f"\nBuilding frozen option surface ...")
    surface = build_surface(chain, spot)
    print(f"  Surface built from {len(chain)} contracts  (spot=${spot})")

    # ── 3. Sanity check ───────────────────────────────────────────────────────
    sanity = run_sanity_check(surface, spot, returns)

    if sanity_only:
        print("\n--sanity flag set. Exiting before full optimization.")
        return

    # Abort if surface returns all-zeros (build failure)
    if sanity["E_W"] == 0.0 and sanity["n_rolls"] == 0.0:
        print("\nERROR: Sanity check returned all-zero results. "
              "Surface lookup may be broken. Aborting.")
        sys.exit(1)

    # ── 4. Generate policy grid ───────────────────────────────────────────────
    print(f"\nGenerating feasible policy grid  (eta=${ETA:.1f} initial debit allowed) ...")
    feasible = policy_grid.generate(chain, spot, eta=ETA)

    if not feasible:
        print("No feasible policies found.")
        print("  - Try --refresh to fetch a fresh option chain.")
        print(f"  - Try increasing ETA (currently {ETA}) in main.py.")
        sys.exit(1)

    # ── 5. Pre-generate bootstrap paths ──────────────────────────────────────
    print(f"\nGenerating bootstrap paths ...")
    sampler      = BootstrapSampler(returns, seed=42)
    paths_coarse = sampler.sample_paths(N_COARSE, HORIZON_COARSE, spot)
    paths_fine   = sampler.sample_paths(N_FINE,   HORIZON_DAYS,   spot)
    print(f"  Coarse : {paths_coarse.shape}  ({N_COARSE} paths x {HORIZON_COARSE} days)")
    print(f"  Fine   : {paths_fine.shape}  ({N_FINE} paths x {HORIZON_DAYS} days)")

    # ── 6. Run optimization ───────────────────────────────────────────────────
    ranked = opt_module.run_optimization(
        feasible_policies = feasible,
        paths_coarse      = paths_coarse,
        paths_fine        = paths_fine,
        surface           = surface,
        spot0             = spot,
        returns           = returns,
    )

    # ── 7. Report ─────────────────────────────────────────────────────────────
    results_module.report(ranked, spot)


if __name__ == "__main__":
    main()
