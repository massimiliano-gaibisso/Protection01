"""
policy_moneyness.py -- MoneyPolicy definition for moneyness-based rolling triggers.

Extends the calendar-based Policy with two new parameters:

  theta_lo_L : roll the LONG  put when K_L/S(t) < theta_lo_L
               (put has drifted too far OTM, refresh protection)

  theta_hi_S : roll the SHORT put when K_S1/S(t) > theta_hi_S
               (put is drifting ITM, close exposure before assignment risk)

All nine base parameters from Policy are kept identical so any calendar-policy
result can be wrapped into a MoneyPolicy by passing **base_policy._asdict()
plus the two new keyword arguments.

Interpretation guide
---------------------
theta_lo_L:
    At each roll K_L = alpha_L * S_roll, so K_L/S = alpha_L right after rolling.
    As S rises (bull run), K_L/S falls.  Roll when K_L/S < theta_lo_L.
    Example: alpha_L=1.15, theta_lo_L=1.00  -> roll when S > K_L  (put goes ATM)
             alpha_L=1.15, theta_lo_L=0.95  -> allow up to 5% OTM before rolling

theta_hi_S:
    At each roll K_S1 = alpha_S1 * S_roll, so K_S1/S = alpha_S1 right after rolling.
    As S falls (crash), K_S1/S rises.  Roll when K_S1/S > theta_hi_S.
    Example: alpha_S1=0.65, theta_hi_S=0.75 -> roll short when it gets 25% OTM from current spot
             alpha_S1=0.65, theta_hi_S=0.90 -> wait until put almost ATM before rolling

Calendar backstops are still applied (T_rem <= 1 forces a roll near expiry).
The d_min_long and d_min_short cooldowns are still enforced between consecutive
rolls on the same leg.
"""
from typing import NamedTuple
import numpy as np


class MoneyPolicy(NamedTuple):
    # ── Base policy parameters (identical to Policy) ──────────────────────────
    alpha_L:     float   # long  put moneyness at roll: K_L  = S × alpha_L  (>1 = ITM)
    alpha_S1:    float   # short put moneyness at roll: K_S1 = S × alpha_S1 (<1 = OTM)
    T_L:         int     # long  put target expiry (trading days)
    T_S1:        int     # short put target expiry (trading days)
    base_q1:     int     # short put quantity
    beta:        float   # floor ratchet: Floor = DELTA × [(1-β)×S0 + β×H(t)]
    d_min_short: int     # minimum days between short rolls (cooldown)
    d_min_long:  int     # minimum days between long  rolls (cooldown)
    eta_pct:     float   # inception net premium constraint (fraction of spot)
    # ── Moneyness trigger parameters (new) ────────────────────────────────────
    theta_lo_L:  float   # roll long  when K_L/S(t)  < theta_lo_L
    theta_hi_S:  float   # roll short when K_S1/S(t) > theta_hi_S


# ── Search grids for the two trigger parameters ───────────────────────────────
# theta_lo_L: threshold for long put becoming too OTM
#   < alpha_L : roll only after put drifts OTM  (less frequent)
#   = alpha_L : roll as soon as put goes OTM    (moderate)
#   > alpha_L : roll while put is still ITM     (more frequent, expensive)
THETA_LO_L_VALUES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

# theta_hi_S: threshold for short put becoming too ITM
#   Values must be > alpha_S1 (otherwise would trigger immediately at inception)
#   Higher = more tolerant before rolling (less frequent, but more ITM exposure)
THETA_HI_S_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
