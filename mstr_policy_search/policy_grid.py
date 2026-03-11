"""
policy_grid.py — Policy definition and feasibility checks.

No grid pre-enumeration — the v4 optimizer uses random_policy() + coord ascent.
Search-space lists are populated by main.py via _push_config_to_policy_grid().
"""
import numpy as np
from typing import NamedTuple


class Policy(NamedTuple):
    alpha_L:     float  # long  put moneyness: K_L  = spot × alpha_L  (>1 = ITM)
    alpha_S1:    float  # short put moneyness: K_S1 = spot × alpha_S1 (<1 = OTM)
    T_L:         int    # long  put target expiry (trading days, snapped to chain)
    T_S1:        int    # short put target expiry (trading days, snapped to chain)
    base_q1:     int    # base quantity for short put
    beta:        float  # floor ratchet blend: Floor = DELTA_FLOOR×[(1-β)×S0 + β×H(t)]
                        #   β=0 → fixed capital floor   β=1 → trailing HWM floor
    d_min_short: int    # calendar roll trigger: roll short when T_S1_rem ≤ d_min_short
    d_min_long:  int    # calendar roll trigger: roll long  when T_L_rem  ≤ d_min_long
    eta_pct:     float  # inception constraint: net_premium ≥ eta_pct × spot
                        #   negative = upfront debit allowed  zero = self-funded


# ── Search-space lists (defaults; main.py overwrites via _push_config_to_policy_grid) ──
ALPHA_L_VALUES    = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
ALPHA_S1_VALUES   = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
T_L_VALUES:   list = []   # populated from chain expiries by main.py
T_S1_VALUES:  list = []   # populated from chain expiries by main.py
Q1_VALUES         = [1, 2, 3, 4, 5]
BETA_VALUES       = [0.0, 0.25, 0.50, 0.75, 1.0]
ETA_PCT_VALUES    = [-0.10, -0.05, 0.0, 0.05, 0.10]
DMIN_SHORT_VALUES = [3, 7, 14, 21]
DMIN_LONG_VALUES  = [7, 14, 21]
SPREAD_PCT_MAX    = 25.0

# Param → value-list mapping (used by coord ascent neighbors())
PARAMS = [
    "alpha_L", "alpha_S1", "T_L", "T_S1",
    "base_q1", "beta", "d_min_short", "d_min_long", "eta_pct",
]

_PARAM_LISTS: dict = {
    "alpha_L":     lambda: ALPHA_L_VALUES,
    "alpha_S1":    lambda: ALPHA_S1_VALUES,
    "T_L":         lambda: T_L_VALUES,
    "T_S1":        lambda: T_S1_VALUES,
    "base_q1":     lambda: Q1_VALUES,
    "beta":        lambda: BETA_VALUES,
    "d_min_short": lambda: DMIN_SHORT_VALUES,
    "d_min_long":  lambda: DMIN_LONG_VALUES,
    "eta_pct":     lambda: ETA_PCT_VALUES,
}


# ── Feasibility helpers ────────────────────────────────────────────────────────

def inception_net(policy: Policy, surface, spot: float) -> float:
    """
    Net premium at inception (positive = credit, negative = debit).
    inception_net = q1 × short_put_bid − long_put_ask
    Does not include cost_per_leg (handled in simulator).
    """
    K_L  = spot * policy.alpha_L
    K_S1 = spot * policy.alpha_S1
    cost = surface.price(K_L,  policy.T_L,  spot, side="ask") or 0.0
    recv = policy.base_q1 * (surface.price(K_S1, policy.T_S1, spot, side="bid") or 0.0)
    return recv - cost


def is_feasible(policy: Policy, spot: float, surface) -> bool:
    """
    Three constraints only — no ordering of T_L vs T_S1 (let optimizer discover):
      1. d_min_short < T_S1   (cooldown < expiry, so at least one hold cycle fits)
      2. d_min_long  < T_L
      3. inception_net >= eta_pct × spot
    """
    if policy.d_min_short >= policy.T_S1:
        return False
    if policy.d_min_long >= policy.T_L:
        return False
    if inception_net(policy, surface, spot) < policy.eta_pct * spot:
        return False
    return True


# ── Random sampling and neighbors ─────────────────────────────────────────────

def random_policy(rng: np.random.Generator) -> Policy:
    """Sample a uniformly random policy from the search space."""
    tl_pool  = T_L_VALUES  if T_L_VALUES  else [30, 60, 90, 120, 180, 252]
    ts1_pool = T_S1_VALUES if T_S1_VALUES else [14, 21, 30, 45, 60, 90]
    return Policy(
        alpha_L     = float(rng.choice(ALPHA_L_VALUES)),
        alpha_S1    = float(rng.choice(ALPHA_S1_VALUES)),
        T_L         = int(rng.choice(tl_pool)),
        T_S1        = int(rng.choice(ts1_pool)),
        base_q1     = int(rng.choice(Q1_VALUES)),
        beta        = float(rng.choice(BETA_VALUES)),
        d_min_short = int(rng.choice(DMIN_SHORT_VALUES)),
        d_min_long  = int(rng.choice(DMIN_LONG_VALUES)),
        eta_pct     = float(rng.choice(ETA_PCT_VALUES)),
    )


def neighbors(policy: Policy, param: str) -> list:
    """All Policy instances with `param` cycled through its full value list."""
    values = _PARAM_LISTS[param]()
    if not values:
        return [policy]
    return [policy._replace(**{param: v}) for v in values]
