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
    frozen_expiries: list | None = None,  # chain days_out grid; if given, T_L and T_S1
                                          # are snapped to nearest listed expiry so roll
                                          # pricing always hits an exact surface grid point
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
            T_S1_q = max(T_S1_rem, 1)

            sp1_ask_close = surface.price_vector(
                K_S1, np.full(n_paths, float(T_S1_q)), S[:, d], side="ask"
            )
            sp1_ask_close = np.maximum(sp1_ask_close, np.maximum(K_S1 - S[:, d], 0.0))  # intrinsic safety-floor

            K_S1_new  = S[:, d] * policy.alpha_S1
            sp1_bid_new = surface.price_vector(
                K_S1_new, np.full(n_paths, float(policy.T_S1)), S[:, d], side="bid"
            )
            # Net per path: receive new_bid, pay close_ask, deduct 2 legs per contract
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
            T_L_q = np.clip(T_L_rem[idx], 1, policy.T_L)

            lp_bid_close = surface.price_vector(K_L[idx], T_L_q, S_r, side="bid")
            lp_bid_close = np.maximum(lp_bid_close, np.maximum(K_L[idx] - S_r, 0.0))  # intrinsic safety-floor

            K_L_new    = S_r * policy.alpha_L
            lp_ask_new = surface.price_vector(
                K_L_new, np.full(len(idx), float(policy.T_L)), S_r, side="ask"
            )
            lp_ask_new = np.maximum(lp_ask_new, np.maximum(K_L_new - S_r, 0.0))  # intrinsic safety-floor

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

    lp_T  = surface.price_vector(K_L,  T_L_vec_T,                       S[:, n_days], side="mid")
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
