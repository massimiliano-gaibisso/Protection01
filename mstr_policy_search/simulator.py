"""
simulator.py — Vectorized path simulator.

Evaluates one Policy across N pre-generated price paths.
All operations are vectorized across paths — no Python loops over paths.
"""
import numpy as np
from policy_grid import Policy
from surface import FrozenSurface


def _inception_cash_flow(policy: Policy, surface: FrozenSurface, spot0: float) -> float:
    """Net cash collected at inception (positive = credit)."""
    K_L  = spot0 * policy.alpha_L
    K_S1 = spot0 * policy.alpha_S1

    cost = surface.price(K_L,  policy.T_L,  spot0, side="ask") or 0.0
    recv = policy.base_q1 * (surface.price(K_S1, policy.T_S1, spot0, side="bid") or 0.0)

    if policy.alpha_S2 > 0:
        K_S2 = spot0 * policy.alpha_S2
        recv += policy.base_q2 * (surface.price(K_S2, policy.T_S2, spot0, side="bid") or 0.0)

    return recv - cost


def _q_max_c3(K_L: float, K_S: float, S_10pct: float, floor_val: float) -> int:
    """
    Cap on short-put quantity from payoff non-negativity constraint (C3).
    """
    if K_S > S_10pct:
        return max(1, int((K_L - floor_val) / (K_S - S_10pct)))
    return 99  # short put OTM at worst-case: unconstrained


def simulate_policy(
    policy:  Policy,
    paths:   np.ndarray,        # (n_paths, n_days+1)
    surface: FrozenSurface,
    spot0:   float,
    returns: np.ndarray,        # 1-D historical returns pool (for S_10pct)
    delta:   float = 0.30,      # max drawdown tolerance
    epsilon: float = 0.10,      # acceptable breach probability
    eta:     float = 0.0,       # max acceptable cumulative debit
    verbose: bool  = False,     # print trace for first 3 paths
) -> dict:
    """
    Returns a dict of performance metrics.
    """
    n_paths, n_days_p1 = paths.shape
    n_days = n_days_p1 - 1

    # Estimate 10th-percentile spot at 1-year horizon for C3
    horizon_1yr = min(252, n_days)
    log_ret_1yr = np.sum(
        np.random.default_rng(0).choice(returns, size=(5000, horizon_1yr), replace=True),
        axis=1,
    )
    S_10pct = spot0 * np.exp(np.percentile(log_ret_1yr, 10))

    # ── initialize state arrays ──────────────────────────────────────────────
    S   = paths                               # (n_paths, n_days+1)
    H   = np.zeros_like(S)
    H[:, 0] = S[:, 0]

    W   = np.full(n_paths, _inception_cash_flow(policy, surface, spot0))

    # Track current strikes (per-path) — updated at each roll
    K_L_cur  = np.full(n_paths, spot0 * policy.alpha_L)
    K_S1_cur = np.full(n_paths, spot0 * policy.alpha_S1)
    K_S2_cur = np.full(n_paths, spot0 * policy.alpha_S2) if policy.alpha_S2 > 0 else None

    # Track time-since-roll (i.e., age of current short legs, which reset at T_S1)
    theta_age = np.zeros(n_paths, dtype=int)
    # Cooldown: enforce minimum d_min days between rolls to prevent R_floor cascades
    days_since_roll = np.full(n_paths, policy.d_min, dtype=int)  # allow roll on day 1

    # Quantities (per-path, can adapt at roll)
    q1 = np.full(n_paths, float(policy.base_q1))
    q2 = np.full(n_paths, float(policy.base_q2)) if policy.alpha_S2 > 0 else np.zeros(n_paths)

    # Breach tracking
    breach = np.zeros((n_paths, n_days), dtype=np.int8)

    roll_count = np.zeros(n_paths, dtype=int)
    breach_depth_sum = np.zeros(n_paths)
    breach_depth_cnt = np.zeros(n_paths, dtype=int)

    if verbose:
        print("\n=== POLICY SANITY TRACE (first 3 paths) ===")

    # ── main day loop ────────────────────────────────────────────────────────
    for d in range(1, n_days + 1):
        # Update high-water mark (vectorized)
        H[:, d] = np.maximum(H[:, d - 1], S[:, d])
        theta_age += 1
        days_since_roll += 1

        # Days remaining on each leg
        T_L_rem  = policy.T_L  - theta_age   # scalar-like, same for all paths (reset at roll)
        T_S1_rem = policy.T_S1 - theta_age
        T_S2_rem = policy.T_S2 - theta_age if policy.alpha_S2 > 0 else np.zeros(n_paths)

        # ── option portfolio value Π ─────────────────────────────────────────
        # Long put value
        T_L_rem_vec  = np.clip(T_L_rem,  1, policy.T_L)
        T_S1_rem_vec = np.clip(T_S1_rem, 1, policy.T_S1)

        lp_val  = surface.price_vector(K_L_cur,  T_L_rem_vec,  S[:, d], side="mid")
        sp1_val = surface.price_vector(K_S1_cur, T_S1_rem_vec, S[:, d], side="mid")
        # Floor put values at intrinsic: when S crashes far outside the original
        # chain range, the frozen surface's nearest-edge extrapolation underestimates
        # deep-ITM puts. Intrinsic is a no-arbitrage lower bound.
        lp_val  = np.maximum(lp_val,  np.maximum(K_L_cur  - S[:, d], 0.0))
        sp1_val = np.maximum(sp1_val, np.maximum(K_S1_cur - S[:, d], 0.0))

        Pi = lp_val - q1 * sp1_val

        if policy.alpha_S2 > 0:
            T_S2_rem_vec = np.clip(T_S2_rem, 1, policy.T_S2)
            sp2_val = surface.price_vector(K_S2_cur, T_S2_rem_vec, S[:, d], side="mid")
            sp2_val = np.maximum(sp2_val, np.maximum(K_S2_cur - S[:, d], 0.0))
            Pi -= q2 * sp2_val

        # Portfolio value
        V   = S[:, d] + Pi
        Flr = H[:, d] * (1.0 - delta)

        # Breach detection
        breached = (V < Flr)
        breach[:, d - 1] = breached.astype(np.int8)

        depth = np.where(breached, Flr - V, 0.0)
        breach_depth_sum += depth
        breach_depth_cnt += breached.astype(int)

        # ── roll triggers ────────────────────────────────────────────────────
        R_calendar = (T_S1_rem <= policy.d_min)
        # R_floor: long put strike below floor AND rolling would improve K_L.
        # Without the second condition, a crash causes an infinite cascade:
        # K_L_new = S*alpha_L stays below Floor = H*0.70 as long as S/H < 0.7216,
        # so R_floor would fire every day until S recovers.
        # R_floor: fire only if we've had at least d_min days since last roll.
        # This prevents the "ratchet" cascade where K_L sits just below Floor
        # and fires on every small positive day.
        R_floor    = (
            (K_L_cur < Flr) &
            (S[:, d] * policy.alpha_L > K_L_cur) &
            (days_since_roll >= policy.d_min)
        )
        roll_mask  = R_calendar | R_floor       # (n_paths,) bool

        if verbose and d <= 60:
            for p in range(min(3, n_paths)):
                status = "ROLL" if roll_mask[p] else ("BREACH" if breached[p] else "OK")
                print(
                    f"  Path {p}  Day {d:4d}: S={S[p,d]:.2f}  H={H[p,d]:.2f}  "
                    f"Floor={Flr[p]:.2f}  V={S[p,d]:.2f}+{Pi[p]:.2f}={V[p]:.2f}  {status}"
                )

        # ── execute rolls ────────────────────────────────────────────────────
        if roll_mask.any():
            idx = np.where(roll_mask)[0]
            S_r = S[idx, d]          # simulated spots on rolling paths

            # Close all legs at bid
            cash_in = np.zeros(len(idx))

            # Close long put at bid
            T_L_q  = np.clip(T_L_rem,  1, policy.T_L)
            T_S1_q = np.clip(T_S1_rem, 1, policy.T_S1)

            lp_bid  = surface.price_vector(K_L_cur[idx],  T_L_q  if np.isscalar(T_L_q)  else T_L_q[idx],  S_r, side="bid")
            sp1_bid = surface.price_vector(K_S1_cur[idx], T_S1_q if np.isscalar(T_S1_q) else T_S1_q[idx], S_r, side="bid")
            lp_bid  = np.maximum(lp_bid,  np.maximum(K_L_cur[idx]  - S_r, 0.0))
            sp1_bid = np.maximum(sp1_bid, np.maximum(K_S1_cur[idx] - S_r, 0.0))
            cash_in += lp_bid
            cash_in += q1[idx] * sp1_bid
            if policy.alpha_S2 > 0:
                T_S2_q  = np.clip(T_S2_rem, 1, policy.T_S2)
                sp2_bid = surface.price_vector(K_S2_cur[idx], T_S2_q if np.isscalar(T_S2_q) else T_S2_q[idx], S_r, side="bid")
                sp2_bid = np.maximum(sp2_bid, np.maximum(K_S2_cur[idx] - S_r, 0.0))
                cash_in += q2[idx] * sp2_bid

            # New target strikes
            K_L_new  = S_r * policy.alpha_L
            K_S1_new = S_r * policy.alpha_S1
            K_S2_new = S_r * policy.alpha_S2 if policy.alpha_S2 > 0 else None

            # Adaptive quantities via self-funding feedback
            deficit  = np.maximum(0.0, -W[idx])
            P_S1_ask = surface.price_vector(K_S1_new, np.full(len(idx), float(policy.T_S1)), S_r, side="ask")

            floor_r  = H[idx, d] * (1.0 - delta)
            q_max1   = np.array([
                _q_max_c3(K_L_new[i], K_S1_new[i], S_10pct, floor_r[i])
                for i in range(len(idx))
            ], dtype=float)

            q1_new = np.floor(policy.base_q1 + policy.gamma * deficit / (P_S1_ask + 1e-6))
            q1_new = np.clip(q1_new, 1, q_max1)

            q2_new = np.zeros(len(idx))
            if policy.alpha_S2 > 0:
                P_S2_ask = surface.price_vector(K_S2_new, np.full(len(idx), float(policy.T_S2)), S_r, side="ask")
                q_max2   = np.array([
                    _q_max_c3(K_L_new[i], K_S2_new[i], S_10pct, floor_r[i])
                    for i in range(len(idx))
                ], dtype=float)
                q2_new = np.floor(policy.base_q2 + policy.gamma * deficit / (P_S2_ask + 1e-6))
                q2_new = np.clip(q2_new, 0, q_max2)

            # Open new legs: pay ask for long, receive bid for shorts
            cash_out = surface.price_vector(K_L_new, np.full(len(idx), float(policy.T_L)), S_r, side="ask")
            cash_recv = q1_new * surface.price_vector(K_S1_new, np.full(len(idx), float(policy.T_S1)), S_r, side="bid")
            if policy.alpha_S2 > 0:
                cash_recv += q2_new * surface.price_vector(K_S2_new, np.full(len(idx), float(policy.T_S2)), S_r, side="bid")

            C_roll = cash_in + cash_recv - cash_out
            W[idx] += C_roll

            if verbose and d <= 60:
                for i, p in enumerate(idx):
                    if p < 3:
                        print(
                            f"    >> Roll path {p}: cash_in={cash_in[i]:.2f}  "
                            f"cash_out={cash_out[i]:.2f}  C_roll={C_roll[i]:.2f}  "
                            f"W={W[p]:.2f}  K_L_new={K_L_new[i]:.2f}  K_S1_new={K_S1_new[i]:.2f}"
                        )

            # Update per-path state
            K_L_cur[idx]  = K_L_new
            K_S1_cur[idx] = K_S1_new
            if policy.alpha_S2 > 0:
                K_S2_cur[idx] = K_S2_new
            q1[idx] = q1_new
            q2[idx] = q2_new
            theta_age[idx] = 0
            days_since_roll[idx] = 0
            roll_count[idx] += 1

    # ── aggregate metrics ────────────────────────────────────────────────────
    ever_breached = breach.any(axis=1)
    P_success     = float((~ever_breached).mean())
    E_W           = float(W.mean())
    P_W_positive  = float((W > 0).mean())

    # CVaR: mean of worst epsilon-fraction of W
    cutoff        = int(np.ceil(epsilon * n_paths))
    worst_W       = np.sort(W)[:cutoff]
    CVaR_W        = float(worst_W.mean()) if cutoff > 0 else float(W.min())

    n_breaches = breach_depth_cnt.sum()
    E_breach_depth = (
        float(breach_depth_sum.sum() / n_breaches) if n_breaches > 0 else 0.0
    )
    n_rolls_per_path = roll_count.mean()

    return {
        "P_success":       P_success,
        "E_W":             E_W,
        "CVaR_W":          CVaR_W,
        "P_W_positive":    P_W_positive,
        "E_breach_depth":  E_breach_depth,
        "n_rolls":         float(n_rolls_per_path),
        "W_final":         W,           # keep full distribution for best policy
    }
