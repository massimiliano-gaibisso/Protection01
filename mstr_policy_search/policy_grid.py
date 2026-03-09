"""
policy_grid.py — Generate the feasible policy parameter grid.

Three filters applied in sequence:
  C_structure  — structural ordering constraints
  C_liquidity  — each leg must have a liquid option in the chain
  C_self_fund  — inception net premium >= -eta

Key design: T_L, T_S1, T_S2 values are derived from the ACTUAL chain expiries
rather than being hardcoded, so the grid adapts to whatever options are listed.
"""
import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd


class Policy(NamedTuple):
    alpha_L:  float   # long put moneyness:    K_L  = spot × alpha_L
    alpha_S1: float   # short put 1 moneyness: K_S1 = spot × alpha_S1
    alpha_S2: float   # short put 2 moneyness: K_S2 = spot × alpha_S2  (0 = disabled)
    T_L:      int     # long put target expiry in days  (snapped to nearest chain expiry)
    T_S1:     int     # short put 1 target expiry in days
    T_S2:     int     # short put 2 target expiry in days (ignored if alpha_S2=0)
    base_q1:  int     # base quantity for short put 1
    base_q2:  int     # base quantity for short put 2 (0 if N=1)
    gamma:    float   # self-funding aggressiveness
    d_min:    int     # calendar roll trigger: days to expiry threshold


# Static alpha and quantity grids (same regardless of chain)
ALPHA_GRID = {
    "alpha_L":  [0.90, 0.95, 1.00, 1.05],
    "alpha_S1": [0.65, 0.70, 0.75, 0.80],
    "alpha_S2": [0.0, 0.50, 0.55, 0.60, 0.65],   # 0.0 = N=1 mode
    "base_q1":  [1, 2],
    "base_q2":  [0, 1, 2],                         # 0 when alpha_S2=0
    "gamma":    [0.0, 0.5, 1.0],
    "d_min":    [7, 14, 21],
}

# Liquidity filter parameters
SPREAD_PCT_MAX = 25.0    # max bid-ask spread % (MSTR has high IV → wider spreads)
STRIKE_TOL_PCT = 0.10    # ±10% of K_target acceptable (for strike snapping)
DAYS_TOL       = 7       # ±7 calendar days for expiry matching once T values come from chain


def _pick_expiries(chain_df: pd.DataFrame, lo_days: int, hi_days: int, n: int) -> list[int]:
    """
    Pick up to n representative actual expiry days_out values in [lo_days, hi_days]
    that have at least one liquid contract (spread_pct <= SPREAD_PCT_MAX).
    Spreads evenly across the range if more candidates exist.
    """
    available = (
        chain_df[(chain_df["days_out"] >= lo_days) & (chain_df["days_out"] <= hi_days)
                 & (chain_df["spread_pct"] <= SPREAD_PCT_MAX)]
        ["days_out"].unique()
    )
    available = sorted(available)
    if not available:
        return []
    if len(available) <= n:
        return [int(d) for d in available]
    # Evenly space: pick indices at 0, step, 2*step, ...
    idx = np.round(np.linspace(0, len(available) - 1, n)).astype(int)
    return [int(available[i]) for i in idx]


def generate(chain_rows: list[dict], spot: float, eta: float = 0.0) -> list[Policy]:
    """
    Generate all feasible policies after applying C_structure, C_liquidity, C_self_fund.
    """
    from surface import FrozenSurface

    surface  = FrozenSurface(chain_rows, spot)
    chain_df = pd.DataFrame(chain_rows)

    # ── Derive T values from actual chain expiries ───────────────────────────
    # Long put: 120 – 600 calendar days (roughly 4 months to 2 years)
    T_L_candidates = _pick_expiries(chain_df, lo_days=120, hi_days=600, n=4)
    # Short puts: 20 – 90 calendar days (roughly 3 weeks to 3 months)
    T_S_candidates = _pick_expiries(chain_df, lo_days=20,  hi_days=90,  n=4)

    if not T_L_candidates:
        # Fallback: take any expiry > 60 days
        T_L_candidates = _pick_expiries(chain_df, lo_days=60, hi_days=1200, n=4)
    if not T_S_candidates:
        T_S_candidates = _pick_expiries(chain_df, lo_days=5, hi_days=120, n=4)

    print(f"  T_L expiries available : {T_L_candidates}")
    print(f"  T_S expiries available : {T_S_candidates}")

    if not T_L_candidates or not T_S_candidates:
        print("ERROR: No suitable expiries found in chain.")
        return []

    # ── Build raw parameter grid ─────────────────────────────────────────────
    keys = list(ALPHA_GRID.keys()) + ["T_L", "T_S1", "T_S2"]
    all_vals = (
        list(ALPHA_GRID.values())
        + [T_L_candidates, T_S_candidates, T_S_candidates]
    )
    combos = list(itertools.product(*all_vals))
    raw = [Policy(**dict(zip(keys, c))) for c in combos]
    print(f"Raw grid:           {len(raw):>10,} policies")

    # ── C_structure ─────────────────────────────────────────────────────────
    def passes_structure(p: Policy) -> bool:
        # Strike ordering: K_S1 < K_L (short put below long put)
        if p.alpha_S1 >= p.alpha_L:
            return False
        # T_S1 < T_L (short put shorter-dated than long put)
        if p.T_S1 >= p.T_L:
            return False
        if p.alpha_S2 > 0:
            # Two-short mode: K_S2 < K_S1 and need valid q2
            if p.alpha_S2 >= p.alpha_S1:
                return False
            if p.base_q2 == 0:
                return False
            if p.T_S2 >= p.T_L:
                return False
        else:
            if p.base_q2 != 0:
                return False
        # d_min must be less than T_S1 (cooldown < short duration)
        if p.d_min >= p.T_S1:
            return False
        return True

    struct = [p for p in raw if passes_structure(p)]
    print(f"After C_structure:  {len(struct):>10,} policies")

    # ── C_liquidity ─────────────────────────────────────────────────────────
    # Since T values come from the actual chain, we use DAYS_TOL as a small buffer.
    def liquid_exists(alpha: float, T: int) -> bool:
        K_target = spot * alpha
        K_lo     = K_target * (1.0 - STRIKE_TOL_PCT)
        K_hi     = K_target * (1.0 + STRIKE_TOL_PCT)
        mask = (
            (chain_df["strike"]     >= K_lo) &
            (chain_df["strike"]     <= K_hi) &
            (chain_df["days_out"]   >= T - DAYS_TOL) &
            (chain_df["days_out"]   <= T + DAYS_TOL) &
            (chain_df["spread_pct"] <= SPREAD_PCT_MAX)
        )
        return bool(mask.any())

    leg_cache: dict[tuple, bool] = {}

    def leg_liquid(alpha: float, T: int) -> bool:
        key = (round(alpha, 4), int(T))
        if key not in leg_cache:
            leg_cache[key] = liquid_exists(alpha, T)
        return leg_cache[key]

    def passes_liquidity(p: Policy) -> bool:
        if not leg_liquid(p.alpha_L, p.T_L):
            return False
        if not leg_liquid(p.alpha_S1, p.T_S1):
            return False
        if p.alpha_S2 > 0 and not leg_liquid(p.alpha_S2, p.T_S2):
            return False
        return True

    liquid = [p for p in struct if passes_liquidity(p)]
    print(f"After C_liquidity:  {len(liquid):>10,} policies")

    if not liquid:
        # Diagnostic: report which (alpha, T) legs are failing
        print("\n  Liquidity diagnostics:")
        for alpha in set(p.alpha_L  for p in struct[:500]):
            for T in T_L_candidates:
                ok = leg_liquid(alpha, T)
                print(f"    alpha_L={alpha:.2f} T_L={T}: {'OK' if ok else 'FAIL'}")
        for alpha in set(p.alpha_S1 for p in struct[:500]):
            for T in T_S_candidates:
                ok = leg_liquid(alpha, T)
                print(f"    alpha_S1={alpha:.2f} T_S1={T}: {'OK' if ok else 'FAIL'}")
        return []

    # ── C_self_fund ─────────────────────────────────────────────────────────
    def inception_net(p: Policy) -> float:
        K_L  = spot * p.alpha_L
        K_S1 = spot * p.alpha_S1
        cost = surface.price(K_L, p.T_L, spot, side="ask") or 0.0
        recv = p.base_q1 * (surface.price(K_S1, p.T_S1, spot, side="bid") or 0.0)
        if p.alpha_S2 > 0:
            K_S2  = spot * p.alpha_S2
            recv += p.base_q2 * (surface.price(K_S2, p.T_S2, spot, side="bid") or 0.0)
        return recv - cost   # positive = net credit

    def passes_self_fund(p: Policy) -> bool:
        return inception_net(p) >= -eta

    feasible = [p for p in liquid if passes_self_fund(p)]
    print(f"After C_self_fund:  {len(feasible):>10,} policies")

    return feasible
