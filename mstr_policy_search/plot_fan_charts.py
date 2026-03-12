#!/usr/bin/env python3
"""
plot_fan_charts.py — Portfolio MtM fan charts + distributional analytics.

Produces a 2-row x 3-column figure:

  Row 1 — Per-day portfolio MtM percentile fan (P10/P25/P50/P75/P90):
            Buy-and-Hold  |  Stop-Loss  |  PUT Overlay (best policy)

  Row 2 — Supporting analytics:
            Terminal V_T KDE
            | Fraction of paths below fixed floor over time
            | Improvement over buy-and-hold fan (P20/P50/P80)

Usage:
    cd mstr_policy_search
    python plot_fan_charts.py
"""
import sys, io, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_option_chain, load_historical_returns
from surface     import build_surface
from bootstrap   import BootstrapSampler
from policy_grid import Policy

# ── constants — must match main.py ────────────────────────────────────────────
DELTA_FLOOR         = 0.80
BTC_ERA_ONLY        = True
BTC_ERA_CUTOFF      = "2020-08-11"
ANNUAL_DEFAULT_PROB = 0.001
HORIZON_DAYS        = 504
COST_PER_LEG        = 1.0
N_PATHS             = 2_000     # same as N_FINE in main.py
SEED                = 137       # same as Phase-3 seed in main.py
BLOCK_SIZE          = 10        # must match main.py BLOCK_SIZE

# Best policy — block bootstrap optimum (feature/block-bootstrap, block_size=10)
# Key change vs v0.10: base_q1=0 (no short leg); pure rolling long put.
# alpha_L=1.15 and T_L=34 are stable across both bootstrap methods.
BEST = Policy(
    alpha_L     = 1.15,
    alpha_S1    = 0.50,   # irrelevant (q1=0)
    T_L         = 34,
    T_S1        = 69,     # irrelevant (q1=0)
    base_q1     = 0,      # no short put
    beta        = 0.25,
    d_min_short = 60,     # irrelevant (q1=0)
    d_min_long  = 28,
    eta_pct     = -0.20,  # irrelevant (q1=0)
)

# Palette
C_BH      = "#4c72b0"   # blue
C_SL      = "#dd8452"   # orange
C_OVERLAY = "#55a868"   # green


# ═════════════════════════════════════════════════════════════════════════════
#  Per-day simulators — identical roll mechanics to simulator.py,
#  extended to store V(t) = portfolio liquidation value at every day.
# ═════════════════════════════════════════════════════════════════════════════

def sim_bh_series(paths: np.ndarray) -> np.ndarray:
    """Buy-and-hold: V(t) = S(t).  Returns a copy of the paths matrix."""
    return paths.copy()


def sim_stoploss_series(paths: np.ndarray, spot0: float,
                        delta_floor: float = 0.80) -> np.ndarray:
    """
    Stop-loss rule: sell when S(t) < floor, buy back when S(t) >= floor.
    V(t) = shares(t) * S(t) + cash(t).
    Returns array of shape (n_paths, n_days+1).
    """
    n_paths, n_days_p1 = paths.shape
    floor     = delta_floor * spot0
    shares    = np.ones(n_paths)
    cash      = np.zeros(n_paths)
    in_market = np.ones(n_paths, dtype=bool)

    V = np.zeros((n_paths, n_days_p1))
    V[:, 0] = paths[:, 0]   # all paths start at spot0

    for d in range(1, n_days_p1):
        S_d = paths[:, d]
        sell = in_market & (S_d < floor)
        if sell.any():
            cash[sell]      = shares[sell] * S_d[sell]
            shares[sell]    = 0.0
            in_market[sell] = False
        buy = (~in_market) & (S_d >= floor)
        if buy.any():
            shares[buy] = cash[buy] / S_d[buy]   # fewer shares = whipsaw penalty
            cash[buy]   = 0.0
            in_market[buy] = True
        V[:, d] = np.where(in_market, shares * S_d, cash)

    return V


def sim_overlay_series(policy: Policy, paths: np.ndarray, surface,
                       spot0: float, delta_floor: float = 0.80,
                       cost_per_leg: float = 1.0) -> np.ndarray:
    """
    PUT overlay: V(t) = S(t) + W(t) + Pi(t)
        W(t)  = cumulative realised option cash flows up to day t
        Pi(t) = open option mark-to-market at day t (mid prices)

    Roll mechanics are identical to simulate_policy() in simulator.py.
    V is recorded before each roll so it reflects the position going into the day.
    Returns array of shape (n_paths, n_days+1).
    """
    n_paths, n_days_p1 = paths.shape
    n_days = n_days_p1 - 1
    S = paths

    H = np.zeros_like(S)
    H[:, 0] = S[:, 0]

    # Inception: open the initial collar at spot0
    K_L0      = spot0 * policy.alpha_L
    K_S10     = spot0 * policy.alpha_S1
    long_ask  = surface.price(K_L0,  policy.T_L,  spot0, side="ask") or 0.0
    short_bid = surface.price(K_S10, policy.T_S1, spot0, side="bid") or 0.0
    W0        = policy.base_q1 * short_bid - long_ask - (1 + policy.base_q1) * cost_per_leg

    W    = np.full(n_paths, W0)
    K_L  = np.full(n_paths, K_L0)
    K_S1 = np.full(n_paths, K_S10)
    q1   = float(policy.base_q1)

    theta_short          = 0
    theta_long           = np.zeros(n_paths, dtype=int)
    days_since_long_roll = np.full(n_paths, policy.d_min_long, dtype=int)

    V_ov = np.zeros((n_paths, n_days_p1))
    # t=0 snapshot: stock + inception cash + open option MTM at mid
    lp0  = surface.price(K_L0,  policy.T_L,  spot0, side="mid") or 0.0
    sp0  = surface.price(K_S10, policy.T_S1, spot0, side="mid") or 0.0
    V_ov[:, 0] = S[:, 0] + W0 + (lp0 - q1 * sp0)

    for d in range(1, n_days + 1):
        H[:, d] = np.maximum(H[:, d - 1], S[:, d])
        theta_short          += 1
        theta_long           += 1
        days_since_long_roll += 1

        T_S1_rem = policy.T_S1 - theta_short        # scalar
        T_L_rem  = policy.T_L  - theta_long         # (n_paths,)

        T_L_vec  = np.clip(T_L_rem, 1, policy.T_L)
        T_S1_vec = max(T_S1_rem, 1)

        # Open option MTM (mid) with intrinsic safety floor
        lp_val  = surface.price_vector(K_L,  T_L_vec, S[:, d], side="mid")
        sp1_val = surface.price_vector(K_S1, np.full(n_paths, float(T_S1_vec)),
                                       S[:, d], side="mid")
        lp_val  = np.maximum(lp_val,  np.maximum(K_L  - S[:, d], 0.0))
        sp1_val = np.maximum(sp1_val, np.maximum(K_S1 - S[:, d], 0.0))
        Pi      = lp_val - q1 * sp1_val

        # Record V before rolling (liquidation value at this day's prices)
        V_ov[:, d] = S[:, d] + W + Pi

        # Floor (beta-blended high-watermark)
        floor_ref = (1.0 - policy.beta) * spot0 + policy.beta * H[:, d]
        Flr = delta_floor * floor_ref

        # Roll triggers
        R_short      = (T_S1_rem <= policy.d_min_short)
        R_long_cal   = (T_L_rem  <= policy.d_min_long)
        R_long_floor = (
            (K_L < Flr) &
            (S[:, d] * policy.alpha_L > K_L) &
            (days_since_long_roll >= policy.d_min_long)
        )
        R_long = R_long_cal | R_long_floor

        # Execute short roll (all paths simultaneously)
        if R_short:
            T_S1_q = max(T_S1_rem, 1)
            sp1_ask_close = surface.price_vector(
                K_S1, np.full(n_paths, float(T_S1_q)), S[:, d], side="ask")
            sp1_ask_close = np.maximum(sp1_ask_close, np.maximum(K_S1 - S[:, d], 0.0))
            K_S1_new    = S[:, d] * policy.alpha_S1
            sp1_bid_new = surface.price_vector(
                K_S1_new, np.full(n_paths, float(policy.T_S1)), S[:, d], side="bid")
            W    += q1 * (sp1_bid_new - sp1_ask_close) - 2.0 * q1 * cost_per_leg
            K_S1  = K_S1_new
            theta_short = 0

        # Execute long roll (per-path subset)
        if R_long.any():
            idx = np.where(R_long)[0]
            S_r = S[idx, d]
            T_L_q = np.clip(T_L_rem[idx], 1, policy.T_L)
            lp_bid_close = surface.price_vector(K_L[idx], T_L_q, S_r, side="bid")
            lp_bid_close = np.maximum(lp_bid_close, np.maximum(K_L[idx] - S_r, 0.0))
            K_L_new    = S_r * policy.alpha_L
            lp_ask_new = surface.price_vector(
                K_L_new, np.full(len(idx), float(policy.T_L)), S_r, side="ask")
            lp_ask_new = np.maximum(lp_ask_new, np.maximum(K_L_new - S_r, 0.0))
            W[idx]               += lp_bid_close - lp_ask_new - 2.0 * cost_per_leg
            K_L[idx]              = K_L_new
            theta_long[idx]       = 0
            days_since_long_roll[idx] = 0

    return V_ov


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def pct(arr, q):
    """Percentile along axis=0 (across paths), returns (n_days+1,) array."""
    return np.percentile(arr, q, axis=0)


def draw_fan(ax, days, V, color, label):
    """Draw P10/P25/P50/P75/P90 fan for a (n_paths, n_days+1) series."""
    p10 = pct(V, 10);  p25 = pct(V, 25);  p50 = pct(V, 50)
    p75 = pct(V, 75);  p90 = pct(V, 90)
    ax.fill_between(days, p10, p90, alpha=0.18, color=color, label="P10 – P90")
    ax.fill_between(days, p25, p75, alpha=0.35, color=color, label="P25 – P75")
    ax.plot(days, p50, color=color, lw=2.2, label=f"Median  ({label})")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading option chain and returns ...")
    chain, spot0, _ = load_option_chain("MSTR", refresh=False)
    returns = load_historical_returns(
        "MSTR", refresh=False,
        cutoff_date=BTC_ERA_CUTOFF if BTC_ERA_ONLY else None,
    )
    surface = build_surface(chain, spot0)
    print(f"  Spot: ${spot0:.2f}  |  Surface: {len(chain)} contracts")
    print(f"  Bootstrap pool: {len(returns):,} days  "
          f"(mean={returns.mean()*100:+.2f}%/day  std={returns.std()*100:.2f}%/day)")

    # ── generate paths (identical seed to main.py Phase-3 fine paths) ─────────
    print(f"\nGenerating {N_PATHS:,} paths  (seed={SEED}) ...")
    sampler = BootstrapSampler(returns, seed=SEED)
    paths   = sampler.sample_paths(
        N_PATHS, HORIZON_DAYS, spot0, annual_default_prob=ANNUAL_DEFAULT_PROB,
        block_size=BLOCK_SIZE)
    print(f"  Shape: {paths.shape}")
    print(f"  Bootstrap: block_size={BLOCK_SIZE}  "
          f"({'i.i.d.' if BLOCK_SIZE <= 1 else f'circular block, {BLOCK_SIZE}-day blocks'})")

    # ── simulate all three strategies ─────────────────────────────────────────
    print("\nSimulating buy-and-hold ...")
    V_bh = sim_bh_series(paths)

    print("Simulating stop-loss ...")
    V_sl = sim_stoploss_series(paths, spot0, delta_floor=DELTA_FLOOR)

    print("Simulating PUT overlay ...")
    V_ov = sim_overlay_series(BEST, paths, surface, spot0,
                              delta_floor=DELTA_FLOOR, cost_per_leg=COST_PER_LEG)

    days        = np.arange(HORIZON_DAYS + 1)
    floor_fixed = DELTA_FLOOR * spot0   # fixed floor reference = 0.80 * S0

    # ── build figure ──────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                          "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"MSTR Options Protection — Portfolio MtM Distribution along Paths\n"
        f"N={N_PATHS:,} paths  |  BTC-era bootstrap (post 2020-08-11)  |  "
        f"spot0 = ${spot0:.2f}  |  2-year horizon ({HORIZON_DAYS} trading days)",
        fontsize=12, fontweight="bold", y=0.995,
    )

    dollar_fmt = mticker.FuncFormatter(lambda v, _: f"${v:.0f}")
    pct_fmt    = mticker.FuncFormatter(lambda v, _: f"{v:.1f}%")
    delta_fmt  = mticker.FuncFormatter(lambda v, _: f"${v:+.0f}")

    strategies = [
        ("Buy and Hold",              V_bh, C_BH),
        ("Stop-Loss",                 V_sl, C_SL),
        ("PUT Overlay  (best policy)", V_ov, C_OVERLAY),
    ]

    # ── ROW 1: per-day MtM fans ───────────────────────────────────────────────
    for ax, (title, V, c) in zip(axes[0], strategies):
        draw_fan(ax, days, V, c, title)
        ax.axhline(spot0,       color="grey", lw=0.9, ls="--",
                   label=f"S0  ${spot0:.0f}")
        ax.axhline(floor_fixed, color="red",  lw=1.1, ls=":",
                   label=f"Fixed floor  ${floor_fixed:.0f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Trading days")
        ax.set_ylabel("Portfolio value")
        ax.yaxis.set_major_formatter(dollar_fmt)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xlim(0, HORIZON_DAYS)
        ax.grid(True, alpha=0.3)

    # ── ROW 2a: terminal V_T distribution (KDE) ───────────────────────────────
    ax = axes[1, 0]
    all_VT = np.concatenate([V_bh[:, -1], V_sl[:, -1], V_ov[:, -1]])
    x_lo   = 0.0
    x_hi   = float(np.percentile(all_VT, 96))
    x_grid = np.linspace(x_lo, x_hi, 600)
    for title, V, c in strategies:
        vT  = np.clip(V[:, -1], 0, x_hi * 3)
        kde = gaussian_kde(vT, bw_method=0.12)
        ax.plot(x_grid, kde(x_grid), color=c, lw=2.0, label=title)
        ax.axvline(float(np.median(vT)), color=c, lw=0.9, ls="--")
    ax.axvline(floor_fixed, color="red", lw=1.2, ls=":",
               label=f"Floor  ${floor_fixed:.0f}")
    ax.set_title("Terminal V_T — density (dashed = median)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Terminal portfolio value")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(dollar_fmt)
    ax.legend(fontsize=8)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(True, alpha=0.3)

    # ── ROW 2b: breach fraction over time ─────────────────────────────────────
    ax = axes[1, 1]
    for title, V, c in strategies:
        frac = (V < floor_fixed).mean(axis=0) * 100
        ax.plot(days, frac, color=c, lw=2.0, label=title)
    ax.set_title("Paths below fixed floor  (floor = 80% of S0)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("% of paths below floor")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.legend(fontsize=8)
    ax.set_xlim(0, HORIZON_DAYS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ── ROW 2c: improvement over BH fan (stop-loss and overlay only) ──────────
    ax = axes[1, 2]
    for title, V, c in [("Stop-Loss", V_sl, C_SL),
                         ("PUT Overlay", V_ov, C_OVERLAY)]:
        imp = V - V_bh   # (n_paths, n_days+1)
        p20 = pct(imp, 20)
        p50 = pct(imp, 50)
        p80 = pct(imp, 80)
        ax.fill_between(days, p20, p80, alpha=0.28, color=c,
                        label=f"{title}  P20–P80")
        ax.plot(days, p50, color=c, lw=2.0, label=f"{title}  median")
    ax.axhline(0, color="grey", lw=1.0, ls="--", label="BH baseline")
    ax.set_title("Improvement over buy-and-hold  (V_strategy − V_BH)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Improvement ($)")
    ax.yaxis.set_major_formatter(delta_fmt)
    ax.legend(fontsize=8)
    ax.set_xlim(0, HORIZON_DAYS)
    ax.grid(True, alpha=0.3)

    # ── save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    out = pathlib.Path(__file__).parent / "results" / "fan_charts.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
