"""
simulator.py — Vectorized path simulator (v4).

v4 design vs v1:
  - Independent per-leg roll clocks:
      theta_short  (scalar)   : uniform across paths; only calendar trigger fires all paths
      theta_long   (per-path) : each path's long leg ages independently; floor trigger
                                can roll the long leg for a subset of paths
  - Beta-blended floor:
      Floor(t) = delta_floor × [(1−β)×S0 + β×H(t)]
      β=0 → fixed capital floor (never ratchets)
      β=1 → trailing high-watermark floor
  - No gamma / adaptive quantity: quantity is fixed at base_q1
  - No alpha_S2 / second short leg
  - Roll cash flows:
      Short roll:  receive bid_new_short − pay ask_old_short  (net usually positive)
      Long  roll:  receive bid_old_long  − pay ask_new_long   (net usually negative,
                   covers protection cost)
      Cost: cost_per_leg deducted per contract per side opened/closed

Terminal improvement metrics (new in v5):
  improvement(path) = W_final + Pi_T
      where Pi_T = open option MTM at the terminal day using mid prices.
      This is the net option P&L if everything is liquidated at horizon end.
  Paths are sorted by terminal S_T (ascending = worst stock outcomes first).
  CVaR_20_improvement : mean improvement in the bottom 20% of paths by S_T (crash paths).
  E_drag              : mean of max(-improvement, 0) in the top 80% of paths (bull paths).
  Score = CVaR_20_improvement - lambda * E_drag

Stop-loss benchmark (simulate_stop_loss):
  Zero-cost rule: sell stock when S_t < floor, buy back when S_t >= floor.
  Returns the same terminal improvement metrics for direct comparison.
"""
import numpy as np
from policy_grid import Policy
from surface import FrozenSurface
from bs_pricer import bs_put_vec


# ── Rolling realized-vol helper ────────────────────────────────────────────────

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


def _roll_price_div(
    surface:  FrozenSurface,
    K_vec:    np.ndarray,
    T_vec:    np.ndarray,   # days remaining (not years)
    S_vec:    np.ndarray,
    spot0:    float,        # S0 at simulation start — log(S_t/S0) is the leverage anchor
    iv_beta:  float,        # leverage coefficient; negative = vol rises when S falls
    side:     str,
) -> np.ndarray:
    """
    Dynamic-IV put pricing via the leverage-effect formula:

        IV(K, T, t) = IV_initial(K/S_t, T) + iv_beta * log(S_t / S0)

    where IV_initial is read from the frozen chain IV surface at the current
    moneyness K/S_t and tenor T (same grid as the frozen price surface).

    Economic interpretation:
      iv_beta < 0  ->  IV rises in crashes (S_t < S0), falls in rallies.
                       Puts become cheaper in bull runs and more expensive
                       in crashes — consistent with the leverage / vol-feedback
                       effect observed in equity markets.
      iv_beta = 0  ->  reduces to the frozen surface mid price (via BS).

    Bid/ask spread: the dynamic BS mid is scaled by the frozen surface's
    bid/ask-to-mid ratio, preserving realistic transaction costs.

    Falls back to the frozen price surface if no IV surface is available.
    """
    if not surface._has_iv_surface:
        return surface.price_vector(K_vec, T_vec, S_vec, side=side)

    if np.isscalar(T_vec):
        T_vec = np.full_like(K_vec, float(T_vec), dtype=float)
    T_vec = np.asarray(T_vec, dtype=float)

    # IV_initial at current moneyness K/S_t (same convention as iv_vector)
    iv_init   = surface.iv_vector(K_vec, T_vec, S_vec)

    # Leverage adjustment: log(S_t/S0); clip S_vec to avoid log(0) in crash paths
    log_ratio = np.log(np.maximum(S_vec, 1e-6) / spot0)
    iv_adj    = np.clip(iv_init + iv_beta * log_ratio, 0.05, 5.0)

    # BS mid price at adjusted IV
    p_mid = bs_put_vec(S_vec, K_vec, T_vec / 252.0, iv_adj)

    if side == "mid":
        return p_mid

    # Scale by frozen bid/ask ratio to preserve market spread
    p_frz_mid  = surface.price_vector(K_vec, T_vec, S_vec, side="mid")
    p_frz_side = surface.price_vector(K_vec, T_vec, S_vec, side=side)
    ratio      = np.where(p_frz_mid > 1e-6, p_frz_side / p_frz_mid, 1.0)
    return np.maximum(p_mid * ratio, 0.0)


def _inception_cash_flow(
    policy:       Policy,
    surface:      FrozenSurface,
    spot0:        float,
    cost_per_leg: float,
) -> float:
    """
    Net cash at inception (positive = credit received, negative = debit paid).
    inception_W = q1 × short_bid − long_ask − (1 + q1) × cost_per_leg
    """
    K_L  = spot0 * policy.alpha_L
    K_S1 = spot0 * policy.alpha_S1
    long_ask  = surface.price(K_L,  policy.T_L,  spot0, side="ask") or 0.0
    short_bid = surface.price(K_S1, policy.T_S1, spot0, side="bid") or 0.0
    return policy.base_q1 * short_bid - long_ask - (1 + policy.base_q1) * cost_per_leg


def simulate_policy(
    policy:          Policy,
    paths:           np.ndarray,      # (n_paths, n_days+1)  price paths
    surface:         FrozenSurface,
    spot0:           float,
    delta_floor:     float = 0.80,    # floor fraction
    cost_per_leg:    float = 1.0,     # $ per contract leg per roll side
    verbose:         bool  = False,   # print trace for first 3 paths (days 1-60)
    frozen_expiries: list | None = None,
    rvol_base:       float | None = None,  # trailing 21-day realized vol at chain date
    vol_scale_roll:  bool  = False,        # True → vol-scaled BS at roll pricing
    use_dynamic_iv:  bool  = False,        # True → dynamic-IV BS pricing (leverage effect)
    iv_beta:         float = -1.0,         # leverage coefficient for dynamic IV
) -> dict:
    """
    Simulate policy across all paths.  Returns performance metrics dict.

    Roll mechanics
    ──────────────
    SHORT leg (all paths roll simultaneously — calendar only):
        Trigger : T_S1_rem = T_S1 − theta_short ≤ d_min_short
        Cash    : q1 × (short_bid_new − short_ask_close) − 2×q1×cost_per_leg

    LONG leg (per-path — calendar OR floor trigger):
        Calendar: T_L_rem[i] = T_L − theta_long[i] ≤ d_min_long
        Floor   : K_L_cur[i] < Floor[i]  AND  S[i]×alpha_L > K_L_cur[i]
                                          AND  days_since_long_roll[i] ≥ d_min_long
        Cash    : long_bid_close − long_ask_new − 2×cost_per_leg
    """
    # Snap T values to nearest available chain expiry.
    # Because T_L / T_S1 come from the frozen chain, this is a no-op during
    # optimisation but protects against drift when called from external scripts
    # (e.g. analyze_best_policy.py) where the JSON target T may differ from any
    # exact chain grid point.
    if frozen_expiries:
        _snap = lambda t: min(frozen_expiries, key=lambda e: abs(e - t))
        policy = policy._replace(T_L=_snap(policy.T_L), T_S1=_snap(policy.T_S1))

    n_paths, n_days_p1 = paths.shape
    n_days = n_days_p1 - 1

    S = paths                           # (n_paths, n_days+1)
    H = np.zeros_like(S)
    H[:, 0] = S[:, 0]

    # ── initial state ─────────────────────────────────────────────────────────
    W    = np.full(n_paths, _inception_cash_flow(policy, surface, spot0, cost_per_leg))
    K_L  = np.full(n_paths, spot0 * policy.alpha_L)
    K_S1 = np.full(n_paths, spot0 * policy.alpha_S1)
    q1   = float(policy.base_q1)

    # Independent leg clocks
    theta_short          = 0                                          # scalar int
    theta_long           = np.zeros(n_paths, dtype=int)              # per-path
    days_since_long_roll = np.full(n_paths, policy.d_min_long, dtype=int)

    breach           = np.zeros((n_paths, n_days), dtype=np.int8)
    breach_depth_sum = np.zeros(n_paths)
    breach_depth_cnt = np.zeros(n_paths, dtype=int)
    roll_count       = np.zeros(n_paths, dtype=int)

    # ── Pre-compute 21-day rolling realized vol (for vol-scaled roll pricing) ──
    _use_vs = vol_scale_roll and (rvol_base is not None) and (rvol_base > 0)
    if _use_vs:
        _log_r    = np.log(np.maximum(S[:, 1:], 1e-6) / np.maximum(S[:, :-1], 1e-6))
        _rvol_21d = _rolling_std_21(_log_r, rvol_base)   # (n_paths, n_days)

    # ── Dynamic-IV pricing flag (mutually exclusive with vol-scaled) ──────────
    # _use_div=True → _roll_price_div() at every pricing point (rolls + Pi MTM)
    _use_div = use_dynamic_iv and (not _use_vs) and surface._has_iv_surface

    if verbose:
        print(f"\n=== POLICY TRACE (first 3 paths, days 1-60) ===")
        print(f"  Policy: alpha_L={policy.alpha_L}  alpha_S1={policy.alpha_S1}  "
              f"T_L={policy.T_L}d  T_S1={policy.T_S1}d  q1={policy.base_q1}  "
              f"beta={policy.beta}  d_S={policy.d_min_short}  d_L={policy.d_min_long}  "
              f"eta={policy.eta_pct:+.0%}")
        ic = _inception_cash_flow(policy, surface, spot0, cost_per_leg)
        long_ask  = surface.price(spot0*policy.alpha_L,  policy.T_L,  spot0, side="ask") or 0.0
        short_bid = surface.price(spot0*policy.alpha_S1, policy.T_S1, spot0, side="bid") or 0.0
        print(f"  Inception: long_ask={long_ask:.2f}  short_bid={short_bid:.2f}  "
              f"net_W={ic:.2f}  K_L={spot0*policy.alpha_L:.2f}  K_S1={spot0*policy.alpha_S1:.2f}")
        if _use_vs:
            print(f"  Vol-scaled roll pricing ON  rvol_base={rvol_base*100:.3f}%/day")
        if _use_div:
            print(f"  Dynamic-IV roll pricing ON  iv_beta={iv_beta:.2f}")

    # ── main day loop ─────────────────────────────────────────────────────────
    for d in range(1, n_days + 1):
        H[:, d] = np.maximum(H[:, d - 1], S[:, d])
        theta_short          += 1
        theta_long           += 1
        days_since_long_roll += 1

        T_S1_rem = policy.T_S1 - theta_short       # scalar
        T_L_rem  = policy.T_L  - theta_long        # (n_paths,) array

        # ── option portfolio value Π ──────────────────────────────────────────
        T_L_vec  = np.clip(T_L_rem, 1, policy.T_L)
        T_S1_vec = max(T_S1_rem, 1)                # scalar; shared across paths

        if _use_div:
            lp_val  = _roll_price_div(surface, K_L, T_L_vec,
                                      S[:, d], spot0, iv_beta, "mid")
            sp1_val = _roll_price_div(surface, K_S1,
                                      np.full(n_paths, float(T_S1_vec)),
                                      S[:, d], spot0, iv_beta, "mid")
        else:
            lp_val  = surface.price_vector(K_L,  T_L_vec,           S[:, d], side="mid")
            sp1_val = surface.price_vector(K_S1, np.full(n_paths, float(T_S1_vec)), S[:, d], side="mid")
        # Intrinsic safety-floor: surface already scales with S (homogeneity fix),
        # but grid-edge clipping can still underestimate deep-ITM puts in extreme crashes.
        lp_val  = np.maximum(lp_val,  np.maximum(K_L  - S[:, d], 0.0))
        sp1_val = np.maximum(sp1_val, np.maximum(K_S1 - S[:, d], 0.0))

        Pi = lp_val - q1 * sp1_val

        # ── floor (beta-blended) ──────────────────────────────────────────────
        floor_ref = (1.0 - policy.beta) * spot0 + policy.beta * H[:, d]
        Flr       = delta_floor * floor_ref

        # ── breach detection ──────────────────────────────────────────────────
        V        = S[:, d] + W + Pi
        breached = V < Flr
        breach[:, d - 1]  = breached.astype(np.int8)
        depth             = np.where(breached, Flr - V, 0.0)
        breach_depth_sum += depth
        breach_depth_cnt += breached.astype(int)

        # ── roll triggers ─────────────────────────────────────────────────────
        R_short        = (T_S1_rem <= policy.d_min_short)           # scalar bool
        R_long_cal     = (T_L_rem  <= policy.d_min_long)            # (n_paths,)
        R_long_floor   = (
            (K_L < Flr) &
            (S[:, d] * policy.alpha_L > K_L) &
            (days_since_long_roll >= policy.d_min_long)
        )
        R_long = R_long_cal | R_long_floor                          # (n_paths,)

        if verbose and d <= 60:
            for p in range(min(3, n_paths)):
                tag = ""
                if R_short:              tag += " ROLL_S"
                if R_long[p]:            tag += " ROLL_L"
                if breached[p]:          tag += " BREACH"
                if not tag:              tag  = " OK"
                print(
                    f"  P{p} d{d:3d}: S={S[p,d]:.2f}  H={H[p,d]:.2f}  "
                    f"Flr={Flr[p]:.2f}  V=S{S[p,d]:+.2f}+W{W[p]:+.2f}+Pi{Pi[p]:+.2f}={V[p]:.2f}"
                    f"  tS={theta_short}  tL={theta_long[p]}{tag}"
                )

        # ── execute short roll (all paths) ────────────────────────────────────
        if R_short:
            T_S1_q   = max(T_S1_rem, 1)
            T_S1_arr = np.full(n_paths, float(T_S1_q))
            K_S1_new = S[:, d] * policy.alpha_S1

            if _use_vs:
                _vs = np.clip(_rvol_21d[:, d - 1] / rvol_base, 0.25, 4.0)
                sp1_ask_close = _roll_price_vs(surface, K_S1, T_S1_arr, S[:, d], _vs, "ask")
                T_S1_new_arr  = np.full(n_paths, float(policy.T_S1))
                sp1_bid_new   = _roll_price_vs(surface, K_S1_new, T_S1_new_arr, S[:, d], _vs, "bid")
            elif _use_div:
                sp1_ask_close = _roll_price_div(surface, K_S1, T_S1_arr,
                                                S[:, d], spot0, iv_beta, "ask")
                sp1_bid_new   = _roll_price_div(surface, K_S1_new,
                                                np.full(n_paths, float(policy.T_S1)),
                                                S[:, d], spot0, iv_beta, "bid")
            else:
                sp1_ask_close = surface.price_vector(K_S1, T_S1_arr, S[:, d], side="ask")
                sp1_bid_new   = surface.price_vector(
                    K_S1_new, np.full(n_paths, float(policy.T_S1)), S[:, d], side="bid"
                )

            sp1_ask_close = np.maximum(sp1_ask_close, np.maximum(K_S1     - S[:, d], 0.0))
            sp1_bid_new   = np.maximum(sp1_bid_new,   np.maximum(K_S1_new - S[:, d], 0.0))

            C_short = q1 * (sp1_bid_new - sp1_ask_close) - 2.0 * q1 * cost_per_leg
            W    += C_short
            K_S1  = K_S1_new
            theta_short = 0
            roll_count += 1

            if verbose and d <= 60:
                for p in range(min(3, n_paths)):
                    print(
                        f"    >> SHORT roll P{p}: ask_close={sp1_ask_close[p]:.2f}  "
                        f"bid_new={sp1_bid_new[p]:.2f}  "
                        f"C_S={C_short[p]:+.2f}  K_S1_new={K_S1_new[p]:.2f}  W={W[p]:.2f}"
                    )

        # ── execute long roll (per-path) ──────────────────────────────────────
        if R_long.any():
            idx = np.where(R_long)[0]
            S_r = S[idx, d]
            T_L_q = np.clip(T_L_rem[idx], 1, policy.T_L).astype(float)

            K_L_new   = S_r * policy.alpha_L
            T_L_full  = np.full(len(idx), float(policy.T_L))

            if _use_vs:
                _vs_idx      = np.clip(_rvol_21d[idx, d - 1] / rvol_base, 0.25, 4.0)
                lp_bid_close = _roll_price_vs(surface, K_L[idx], T_L_q,    S_r, _vs_idx, "bid")
                lp_ask_new   = _roll_price_vs(surface, K_L_new,  T_L_full, S_r, _vs_idx, "ask")
            elif _use_div:
                lp_bid_close = _roll_price_div(surface, K_L[idx], T_L_q,
                                               S_r, spot0, iv_beta, "bid")
                lp_ask_new   = _roll_price_div(surface, K_L_new, T_L_full,
                                               S_r, spot0, iv_beta, "ask")
            else:
                lp_bid_close = surface.price_vector(K_L[idx], T_L_q,    S_r, side="bid")
                lp_ask_new   = surface.price_vector(K_L_new,  T_L_full, S_r, side="ask")

            lp_bid_close = np.maximum(lp_bid_close, np.maximum(K_L[idx]  - S_r, 0.0))  # intrinsic safety-floor
            lp_ask_new   = np.maximum(lp_ask_new,   np.maximum(K_L_new   - S_r, 0.0))  # intrinsic safety-floor

            # Net per path: receive old_bid, pay new_ask, deduct 2 legs
            C_long = lp_bid_close - lp_ask_new - 2.0 * cost_per_leg
            W[idx]               += C_long
            K_L[idx]              = K_L_new
            theta_long[idx]       = 0
            days_since_long_roll[idx] = 0
            roll_count[idx]      += 1

            if verbose and d <= 60:
                for i, p in enumerate(idx):
                    if p < 3:
                        print(
                            f"    >> LONG  roll P{p}: bid_close={lp_bid_close[i]:.2f}  "
                            f"ask_new={lp_ask_new[i]:.2f}  "
                            f"C_L={C_long[i]:+.2f}  K_L_new={K_L_new[i]:.2f}  W={W[p]:.2f}"
                        )

    # ── aggregate legacy metrics ──────────────────────────────────────────────
    ever_breached    = breach.any(axis=1)
    P_success        = float((~ever_breached).mean())
    E_W              = float(W.mean())
    P_W_positive     = float((W > 0).mean())

    cvar_cutoff = int(np.ceil(0.10 * n_paths))
    CVaR_W      = float(np.sort(W)[:cvar_cutoff].mean()) if cvar_cutoff > 0 else float(W.min())

    n_breach_days  = int(breach_depth_cnt.sum())
    E_breach_depth = float(breach_depth_sum.sum() / n_breach_days) if n_breach_days > 0 else 0.0

    # ── terminal open-option MTM (Pi_T) ───────────────────────────────────────
    # Recompute Pi using final post-roll state so any roll on the last day is
    # reflected.  This is the liquidation value of the open option positions
    # at the horizon end.
    T_S1_rem_T = policy.T_S1 - theta_short            # remaining days on short leg
    T_L_vec_T  = np.clip(policy.T_L - theta_long, 1, policy.T_L)  # per-path long T_rem
    T_S1_clamp = max(int(T_S1_rem_T), 1)

    # Pi_T: use vol-scaled BS if vol-scaling is active, so Pi_T is consistent
    # with W (which already captured vol spikes at each roll event).
    # In crash paths the elevated terminal rvol_21d boosts lp_T, giving a more
    # accurate CVaR_20_improvement.  intrinsic floor applied after either branch.
    if _use_vs:
        _vs_T = np.clip(_rvol_21d[:, n_days - 1] / rvol_base, 0.25, 4.0)
        lp_T  = _roll_price_vs(surface, K_L,
                               T_L_vec_T.astype(float),
                               S[:, n_days], _vs_T, "mid")
        sp1_T = _roll_price_vs(surface, K_S1,
                               np.full(n_paths, float(T_S1_clamp)),
                               S[:, n_days], _vs_T, "mid")
    elif _use_div:
        lp_T  = _roll_price_div(surface, K_L, T_L_vec_T.astype(float),
                                S[:, n_days], spot0, iv_beta, "mid")
        sp1_T = _roll_price_div(surface, K_S1,
                                np.full(n_paths, float(T_S1_clamp)),
                                S[:, n_days], spot0, iv_beta, "mid")
    else:
        lp_T  = surface.price_vector(K_L,  T_L_vec_T,                           S[:, n_days], side="mid")
        sp1_T = surface.price_vector(K_S1, np.full(n_paths, float(T_S1_clamp)), S[:, n_days], side="mid")
    lp_T  = np.maximum(lp_T,  np.maximum(K_L  - S[:, n_days], 0.0))   # intrinsic floor
    sp1_T = np.maximum(sp1_T, np.maximum(K_S1 - S[:, n_days], 0.0))
    Pi_T  = lp_T - q1 * sp1_T                                           # (n_paths,)

    # ── terminal improvement over buy-and-hold ────────────────────────────────
    # improvement(path) = W_final + Pi_T
    # Paths sorted ascending by S_T: bottom 20% = crash paths, top 80% = bull paths.
    improvement = W + Pi_T
    S_T         = S[:, n_days]
    order_by_S  = np.argsort(S_T)
    n_crash     = max(1, int(np.ceil(0.20 * n_paths)))
    crash_idx   = order_by_S[:n_crash]
    bull_idx    = order_by_S[n_crash:]

    CVaR_20_improvement = float(improvement[crash_idx].mean())
    E_drag = float(np.maximum(-improvement[bull_idx], 0).mean()) if len(bull_idx) > 0 else 0.0

    return {
        # Legacy metrics (kept for reporting / diagnostics)
        "P_success":            P_success,
        "E_W":                  E_W,
        "CVaR_W":               CVaR_W,
        "P_W_positive":         P_W_positive,
        "E_breach_depth":       E_breach_depth,
        "n_rolls":              float(roll_count.mean()),
        "W_final":              W,            # full distribution
        # New terminal-improvement metrics (used in score)
        "Pi_T":                 Pi_T,         # terminal open MTM (n_paths,)
        "improvement":          improvement,  # W + Pi_T  (n_paths,)
        "CVaR_20_improvement":  CVaR_20_improvement,
        "E_drag":               E_drag,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  STOP-LOSS BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def simulate_stop_loss(
    paths:       np.ndarray,   # (n_paths, n_days+1) price paths
    spot0:       float,
    delta_floor: float = 0.80,
    cvar_q:      float = 0.20,
) -> dict:
    """
    Simulate the zero-cost stop-loss / re-entry benchmark.

    Rule (checked every day):
        - While in market: sell entire position when S_t < floor.
        - While in cash:   buy back when S_t >= floor.
        - floor = delta_floor × spot0  (fixed, identical to beta=0 options floor)

    Each path starts holding 1 share worth spot0.  When the position is sold at
    price S_sell, the cash S_sell is held.  When buying back at S_buy > S_sell
    the investor can afford only S_sell / S_buy < 1 shares — the whipsaw penalty.

    Returns dict with:
        V_T              (n_paths,) terminal portfolio value
        improvement      (n_paths,) V_T - S_T  (vs buy-and-hold)
        CVaR_20_imp      mean improvement in bottom cvar_q% of paths by S_T
        E_drag           mean of max(-improvement, 0) in top (1-cvar_q)% by S_T
        E_improvement    unconditional mean improvement
        n_trades         mean number of round-trip trades per path
    """
    n_paths, n_days_p1 = paths.shape
    floor = delta_floor * spot0

    shares    = np.ones(n_paths)              # start: 1 share per path
    cash      = np.zeros(n_paths)
    in_market = np.ones(n_paths, dtype=bool)
    trade_cnt = np.zeros(n_paths, dtype=int)

    for d in range(1, n_days_p1):
        S_d = paths[:, d]

        # Sell: in market AND price falls below floor
        sell = in_market & (S_d < floor)
        if sell.any():
            cash[sell]      = shares[sell] * S_d[sell]
            shares[sell]    = 0.0
            in_market[sell] = False

        # Buy back: in cash AND price recovers to or above floor
        buy = (~in_market) & (S_d >= floor)
        if buy.any():
            shares[buy]    = cash[buy] / S_d[buy]  # fewer shares if bought back higher
            cash[buy]      = 0.0
            in_market[buy] = True
            trade_cnt[buy] += 1                     # count completed round-trips

    # Terminal wealth
    S_T  = paths[:, -1]
    V_T  = np.where(in_market, shares * S_T, cash)
    improvement = V_T - S_T   # positive = better than buy-and-hold

    # Split paths by terminal S_T (same ordering as simulate_policy)
    order_by_S = np.argsort(S_T)
    n_crash    = max(1, int(np.ceil(cvar_q * n_paths)))
    crash_idx  = order_by_S[:n_crash]
    bull_idx   = order_by_S[n_crash:]

    CVaR_20_imp = float(improvement[crash_idx].mean())
    E_drag      = float(np.maximum(-improvement[bull_idx], 0).mean()) if len(bull_idx) > 0 else 0.0

    return {
        "V_T":           V_T,
        "improvement":   improvement,
        "CVaR_20_imp":   CVaR_20_imp,
        "E_drag":        E_drag,
        "E_improvement": float(improvement.mean()),
        "n_trades":      float(trade_cnt.mean()),
    }
