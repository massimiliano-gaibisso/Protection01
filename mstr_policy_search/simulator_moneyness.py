"""
simulator_moneyness.py -- Vectorized simulator with moneyness-based roll triggers.

Key differences from simulator.py (calendar-based):

SHORT leg (per-path, independent clocks):
    Calendar version: single scalar theta_short ticks uniformly; all paths roll together.
    Moneyness version: each path's short leg rolls independently when K_S1/S > theta_hi_S.
    Triggers:
        - Moneyness: K_S1[i]/S[i,d] > theta_hi_S  AND  days_since_short_roll[i] >= d_min_short
        - Backstop : T_S1_rem[i] <= 1  (near expiry, force roll regardless of cooldown)

LONG leg (per-path, same as calendar version):
    Calendar trigger replaced by moneyness trigger:
        - Moneyness: K_L[i]/S[i,d] < theta_lo_L    AND  days_since_long_roll[i] >= d_min_long
        - Floor    : K_L[i] < Flr[i]  AND  S[i]*alpha_L > K_L[i]  AND  cooldown ok
        - Backstop : T_L_rem[i] <= 1  (near expiry, force roll regardless of cooldown)

All other mechanics (inception cash flow, portfolio valuation, breach detection,
terminal improvement metric, homogeneity-scaled surface pricing) are identical to
simulator.py.
"""
import numpy as np
from policy_moneyness import MoneyPolicy
from surface import FrozenSurface
from bs_pricer import bs_put_vec


# ── Rolling realized-vol helpers (shared with simulator.py) ───────────────────

def _rolling_std_21(log_r: np.ndarray, rvol_base: float) -> np.ndarray:
    """
    Vectorised 21-day rolling std of log_r (n_paths, n_days).

    Uses the cumsum-of-squares identity — no Python loops.
    Days 0..19 lack sufficient history and are filled with rvol_base.
    Returns array of shape (n_paths, n_days).
    """
    n_paths, n_days = log_r.shape
    WINDOW = 21
    rvol   = np.full((n_paths, n_days), rvol_base)
    if n_days < WINDOW:
        return rvol

    cs  = np.cumsum(np.pad(log_r,        ((0, 0), (1, 0))), axis=1)
    cs2 = np.cumsum(np.pad(log_r ** 2,   ((0, 0), (1, 0))), axis=1)

    d_arr = np.arange(WINDOW - 1, n_days)
    s     = cs[:,  d_arr + 1] - cs[:,  d_arr + 1 - WINDOW]
    s2    = cs2[:, d_arr + 1] - cs2[:, d_arr + 1 - WINDOW]
    var   = s2 / WINDOW - (s / WINDOW) ** 2
    rvol[:, WINDOW - 1:] = np.sqrt(np.maximum(var, 1e-12))
    return rvol


def _roll_price_vs(
    surface:   FrozenSurface,
    K_vec:     np.ndarray,
    T_vec:     np.ndarray,
    S_vec:     np.ndarray,
    vol_scale: np.ndarray,
    side:      str,
) -> np.ndarray:
    """
    Vol-scaled option price for a batch of roll-event paths.

    1. Fetch frozen mid IV at (K, T, S) from the IV interpolator.
    2. Scale IV by vol_scale (realized-vol / base-vol ratio, clipped [0.25, 4.0]).
    3. Price via Black-Scholes at the scaled IV (mid price).
    4. Restore frozen bid/ask spread ratio so bid/ask pricing is preserved.

    Falls back to the frozen price surface if IV surface is unavailable.
    """
    if not surface._has_iv_surface:
        return surface.price_vector(K_vec, T_vec, S_vec, side=side)

    iv_scaled  = surface.iv_vector(K_vec, T_vec, S_vec) * vol_scale
    p_mid_vs   = bs_put_vec(S_vec, K_vec, T_vec / 252.0, iv_scaled)

    p_frz_mid  = surface.price_vector(K_vec, T_vec, S_vec, side="mid")
    p_frz_side = surface.price_vector(K_vec, T_vec, S_vec, side=side)
    ratio      = np.where(p_frz_mid > 1e-6, p_frz_side / p_frz_mid, 1.0)
    return p_mid_vs * ratio


def _inception_cash_flow(
    policy:       MoneyPolicy,
    surface:      FrozenSurface,
    spot0:        float,
    cost_per_leg: float,
) -> float:
    """Net cash at inception: q1 * short_bid - long_ask - (1 + q1) * cost_per_leg."""
    K_L  = spot0 * policy.alpha_L
    K_S1 = spot0 * policy.alpha_S1
    long_ask  = surface.price(K_L,  policy.T_L,  spot0, side="ask") or 0.0
    short_bid = surface.price(K_S1, policy.T_S1, spot0, side="bid") or 0.0
    return policy.base_q1 * short_bid - long_ask - (1 + policy.base_q1) * cost_per_leg


def simulate_money_policy(
    policy:          MoneyPolicy,
    paths:           np.ndarray,      # (n_paths, n_days+1) price paths
    surface:         FrozenSurface,
    spot0:           float,
    delta_floor:     float = 0.80,
    cost_per_leg:    float = 1.0,
    verbose:         bool  = False,
    frozen_expiries: list | None = None,
    rvol_base:       float | None = None,  # trailing 21-day realized vol at chain date
    vol_scale_roll:  bool  = False,        # True → vol-scaled BS at roll pricing
) -> dict:
    """
    Simulate a MoneyPolicy across all paths.  Returns the same metrics dict
    as simulate_policy() so the two are directly comparable.
    """
    if frozen_expiries:
        _snap = lambda t: min(frozen_expiries, key=lambda e: abs(e - t))
        policy = policy._replace(T_L=_snap(policy.T_L), T_S1=_snap(policy.T_S1))

    n_paths, n_days_p1 = paths.shape
    n_days = n_days_p1 - 1

    S = paths                           # (n_paths, n_days+1)
    H = np.zeros_like(S)
    H[:, 0] = S[:, 0]

    # ── Initial state ──────────────────────────────────────────────────────────
    W    = np.full(n_paths, _inception_cash_flow(policy, surface, spot0, cost_per_leg))
    K_L  = np.full(n_paths, spot0 * policy.alpha_L)
    K_S1 = np.full(n_paths, spot0 * policy.alpha_S1)
    q1   = float(policy.base_q1)

    # Per-path clocks for BOTH legs (short is now independent per path)
    theta_long            = np.zeros(n_paths, dtype=int)
    theta_short           = np.zeros(n_paths, dtype=int)
    days_since_long_roll  = np.full(n_paths, policy.d_min_long,  dtype=int)
    days_since_short_roll = np.full(n_paths, policy.d_min_short, dtype=int)

    breach           = np.zeros((n_paths, n_days), dtype=np.int8)
    breach_depth_sum = np.zeros(n_paths)
    breach_depth_cnt = np.zeros(n_paths, dtype=int)
    roll_count       = np.zeros(n_paths, dtype=int)

    # ── Pre-compute 21-day rolling realized vol (for vol-scaled roll pricing) ──
    _use_vs = vol_scale_roll and (rvol_base is not None) and (rvol_base > 0)
    if _use_vs:
        _log_r    = np.log(np.maximum(S[:, 1:], 1e-6) / np.maximum(S[:, :-1], 1e-6))
        _rvol_21d = _rolling_std_21(_log_r, rvol_base)   # (n_paths, n_days)

    if verbose:
        ic = _inception_cash_flow(policy, surface, spot0, cost_per_leg)
        print(f"\n=== MONEY POLICY TRACE (first 3 paths, days 1-60) ===")
        print(f"  theta_lo_L={policy.theta_lo_L:.2f}  theta_hi_S={policy.theta_hi_S:.2f}")
        print(f"  alpha_L={policy.alpha_L}  alpha_S1={policy.alpha_S1}  "
              f"T_L={policy.T_L}d  T_S1={policy.T_S1}d  q1={policy.base_q1}  "
              f"beta={policy.beta}  dL={policy.d_min_long}  dS={policy.d_min_short}")
        print(f"  Inception: net_W={ic:.2f}  "
              f"K_L={spot0*policy.alpha_L:.2f}  K_S1={spot0*policy.alpha_S1:.2f}")
        if _use_vs:
            print(f"  Vol-scaled roll pricing ON  rvol_base={rvol_base*100:.3f}%/day")

    # ── Main day loop ──────────────────────────────────────────────────────────
    for d in range(1, n_days + 1):
        H[:, d] = np.maximum(H[:, d - 1], S[:, d])
        theta_long            += 1
        theta_short           += 1
        days_since_long_roll  += 1
        days_since_short_roll += 1

        T_L_rem  = policy.T_L  - theta_long    # (n_paths,) per-path
        T_S1_rem = policy.T_S1 - theta_short   # (n_paths,) per-path (was scalar before)

        # ── Option portfolio value Pi ─────────────────────────────────────────
        T_L_vec  = np.clip(T_L_rem,  1, policy.T_L).astype(float)
        T_S1_vec = np.clip(T_S1_rem, 1, policy.T_S1).astype(float)  # now per-path

        lp_val  = surface.price_vector(K_L,  T_L_vec,  S[:, d], side="mid")
        sp1_val = surface.price_vector(K_S1, T_S1_vec, S[:, d], side="mid")
        lp_val  = np.maximum(lp_val,  np.maximum(K_L  - S[:, d], 0.0))   # intrinsic floor
        sp1_val = np.maximum(sp1_val, np.maximum(K_S1 - S[:, d], 0.0))

        Pi = lp_val - q1 * sp1_val

        # ── Floor (beta-blended) ──────────────────────────────────────────────
        floor_ref = (1.0 - policy.beta) * spot0 + policy.beta * H[:, d]
        Flr       = delta_floor * floor_ref

        # ── Breach detection ──────────────────────────────────────────────────
        V        = S[:, d] + W + Pi
        breached = V < Flr
        breach[:, d - 1]  = breached.astype(np.int8)
        depth             = np.where(breached, Flr - V, 0.0)
        breach_depth_sum += depth
        breach_depth_cnt += breached.astype(int)

        # ── Moneyness ratios ──────────────────────────────────────────────────
        S_safe = np.maximum(S[:, d], 1e-6)     # avoid div-by-zero in default paths
        mn_L   = K_L  / S_safe                 # K_L/S:  falls as S rises (bull)
        mn_S   = K_S1 / S_safe                 # K_S1/S: rises as S falls (crash)

        # ── Roll triggers ─────────────────────────────────────────────────────
        # SHORT: moneyness going high (put going ITM) OR near expiry backstop
        R_short = (
            ((mn_S > policy.theta_hi_S) & (days_since_short_roll >= policy.d_min_short))
            | (T_S1_rem <= 1)
        )

        # LONG: moneyness going low (put going OTM) OR floor trigger OR near expiry
        R_long_mn    = (mn_L < policy.theta_lo_L) & (days_since_long_roll >= policy.d_min_long)
        R_long_floor = (
            (K_L < Flr)
            & (S[:, d] * policy.alpha_L > K_L)
            & (days_since_long_roll >= policy.d_min_long)
        )
        R_long_back  = (T_L_rem <= 1)
        R_long       = R_long_mn | R_long_floor | R_long_back

        if verbose and d <= 60:
            for p in range(min(3, n_paths)):
                tag = ""
                if R_short[p]: tag += " ROLL_S"
                if R_long[p]:  tag += " ROLL_L"
                if breached[p]: tag += " BREACH"
                if not tag:    tag  = " OK"
                print(
                    f"  P{p} d{d:3d}: S={S[p,d]:.2f}  Flr={Flr[p]:.2f}  "
                    f"V={V[p]:.2f}  mnL={mn_L[p]:.3f}  mnS={mn_S[p]:.3f}"
                    f"  tL={theta_long[p]}  tS={theta_short[p]}{tag}"
                )

        # ── Execute short roll (per-path) ─────────────────────────────────────
        if R_short.any():
            idx = np.where(R_short)[0]
            S_r = S[idx, d]
            T_S1_q   = np.clip(T_S1_rem[idx], 1, policy.T_S1).astype(float)
            K_S1_new = S_r * policy.alpha_S1
            T_S1_new = np.full(len(idx), float(policy.T_S1))

            if _use_vs:
                _vs_idx       = np.clip(_rvol_21d[idx, d - 1] / rvol_base, 0.25, 4.0)
                sp1_ask_close = _roll_price_vs(surface, K_S1[idx], T_S1_q,  S_r, _vs_idx, "ask")
                sp1_bid_new   = _roll_price_vs(surface, K_S1_new,  T_S1_new, S_r, _vs_idx, "bid")
            else:
                sp1_ask_close = surface.price_vector(K_S1[idx], T_S1_q,  S_r, side="ask")
                sp1_bid_new   = surface.price_vector(K_S1_new,  T_S1_new, S_r, side="bid")

            sp1_ask_close = np.maximum(sp1_ask_close, np.maximum(K_S1[idx] - S_r, 0.0))
            C_short = q1 * (sp1_bid_new - sp1_ask_close) - 2.0 * q1 * cost_per_leg
            W[idx]                    += C_short
            K_S1[idx]                  = K_S1_new
            theta_short[idx]           = 0
            days_since_short_roll[idx] = 0
            roll_count[idx]           += 1

            if verbose and d <= 60:
                for i, p in enumerate(idx):
                    if p < 3:
                        print(
                            f"    >> SHORT roll P{p}: ask_close={sp1_ask_close[i]:.2f}  "
                            f"bid_new={sp1_bid_new[i]:.2f}  C_S={C_short[i]:+.2f}  "
                            f"K_S1_new={K_S1_new[i]:.2f}  W={W[p]:.2f}"
                        )

        # ── Execute long roll (per-path) ──────────────────────────────────────
        if R_long.any():
            idx = np.where(R_long)[0]
            S_r = S[idx, d]
            T_L_q    = np.clip(T_L_rem[idx], 1, policy.T_L).astype(float)
            K_L_new  = S_r * policy.alpha_L
            T_L_full = np.full(len(idx), float(policy.T_L))

            if _use_vs:
                _vs_idx      = np.clip(_rvol_21d[idx, d - 1] / rvol_base, 0.25, 4.0)
                lp_bid_close = _roll_price_vs(surface, K_L[idx], T_L_q,    S_r, _vs_idx, "bid")
                lp_ask_new   = _roll_price_vs(surface, K_L_new,  T_L_full, S_r, _vs_idx, "ask")
            else:
                lp_bid_close = surface.price_vector(K_L[idx], T_L_q,    S_r, side="bid")
                lp_ask_new   = surface.price_vector(K_L_new,  T_L_full, S_r, side="ask")

            lp_bid_close = np.maximum(lp_bid_close, np.maximum(K_L[idx] - S_r, 0.0))
            lp_ask_new   = np.maximum(lp_ask_new,   np.maximum(K_L_new  - S_r, 0.0))

            C_long = lp_bid_close - lp_ask_new - 2.0 * cost_per_leg
            W[idx]                   += C_long
            K_L[idx]                  = K_L_new
            theta_long[idx]           = 0
            days_since_long_roll[idx] = 0
            roll_count[idx]          += 1

            if verbose and d <= 60:
                for i, p in enumerate(idx):
                    if p < 3:
                        print(
                            f"    >> LONG  roll P{p}: bid_close={lp_bid_close[i]:.2f}  "
                            f"ask_new={lp_ask_new[i]:.2f}  C_L={C_long[i]:+.2f}  "
                            f"K_L_new={K_L_new[i]:.2f}  W={W[p]:.2f}"
                        )

    # ── Aggregate legacy metrics ───────────────────────────────────────────────
    ever_breached = breach.any(axis=1)
    P_success     = float((~ever_breached).mean())
    E_W           = float(W.mean())
    P_W_positive  = float((W > 0).mean())

    cvar_cutoff = int(np.ceil(0.10 * n_paths))
    CVaR_W      = float(np.sort(W)[:cvar_cutoff].mean()) if cvar_cutoff > 0 else float(W.min())

    n_breach_days  = int(breach_depth_cnt.sum())
    E_breach_depth = float(breach_depth_sum.sum() / n_breach_days) if n_breach_days > 0 else 0.0

    # ── Terminal open-option MTM (Pi_T) ───────────────────────────────────────
    T_L_vec_T  = np.clip(policy.T_L  - theta_long,  1, policy.T_L).astype(float)
    T_S1_vec_T = np.clip(policy.T_S1 - theta_short, 1, policy.T_S1).astype(float)

    lp_T  = surface.price_vector(K_L,  T_L_vec_T,  S[:, n_days], side="mid")
    sp1_T = surface.price_vector(K_S1, T_S1_vec_T, S[:, n_days], side="mid")
    lp_T  = np.maximum(lp_T,  np.maximum(K_L  - S[:, n_days], 0.0))
    sp1_T = np.maximum(sp1_T, np.maximum(K_S1 - S[:, n_days], 0.0))
    Pi_T  = lp_T - q1 * sp1_T

    # ── Terminal improvement over buy-and-hold ────────────────────────────────
    improvement = W + Pi_T
    S_T         = S[:, n_days]
    order_by_S  = np.argsort(S_T)
    n_crash     = max(1, int(np.ceil(0.20 * n_paths)))
    crash_idx   = order_by_S[:n_crash]
    bull_idx    = order_by_S[n_crash:]

    CVaR_20_improvement = float(improvement[crash_idx].mean())
    E_drag = float(np.maximum(-improvement[bull_idx], 0).mean()) if len(bull_idx) > 0 else 0.0

    return {
        "P_success":            P_success,
        "E_W":                  E_W,
        "CVaR_W":               CVaR_W,
        "P_W_positive":         P_W_positive,
        "E_breach_depth":       E_breach_depth,
        "n_rolls":              float(roll_count.mean()),
        "W_final":              W,
        "Pi_T":                 Pi_T,
        "improvement":          improvement,
        "CVaR_20_improvement":  CVaR_20_improvement,
        "E_drag":               E_drag,
    }
