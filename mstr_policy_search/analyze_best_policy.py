#!/usr/bin/env python3
"""
analyze_best_policy.py  (v4)
----------------------------
Full P&L breakdown for the best v4 policy.

  Full P&L = (S_T - S_0)               [stock leg]
           + W_options                  [cumulative realized option cash flows]
           + Pi_final                   [unrealized MTM of the open position at T]

Usage:
    cd mstr_policy_search
    python analyze_best_policy.py           # full 2000-path analysis
    python analyze_best_policy.py --trace   # + day-by-day trace of path 0
"""
import sys, os, io, json, argparse
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))

import data_loader
from surface import build_surface
from bootstrap import BootstrapSampler
from policy_grid import Policy

# ── analysis parameters ───────────────────────────────────────────────────────
BTC_ERA             = True
BTC_ERA_CUTOFF      = "2020-08-11"  # matches main.py
N_PATHS             = 2_000
HORIZON_DAYS        = 504
DELTA_FLOOR         = 0.80          # floor fraction (v4)
ANNUAL_DEFAULT_PROB = 0.001         # crash overlay (matches main.py)
COST_PER_LEG        = 1.0           # $ per contract leg per roll
SEED                = 42
TRACE_PATH          = 0             # path index for --trace output

# ── load data ─────────────────────────────────────────────────────────────────
chain, spot0, _ = data_loader.load_option_chain("MSTR", refresh=False)
returns = data_loader.load_historical_returns(
    "MSTR", refresh=False,
    cutoff_date=BTC_ERA_CUTOFF if BTC_ERA else None,
)
surface = build_surface(chain, spot0)

_chain_df = pd.DataFrame(chain)
_expiries  = sorted(_chain_df["days_out"].unique())

def snap_expiry(target: int) -> int:
    """Return the chain expiry (days_out) closest to target."""
    return min(_expiries, key=lambda d: abs(d - target))

# ── load best policy ──────────────────────────────────────────────────────────
_json_path = os.path.join(os.path.dirname(__file__), "results", "best_policy.json")
with open(_json_path) as f:
    bp = json.load(f)

p = bp["policy"]
T_L_snap  = snap_expiry(p["T_L"])
T_S1_snap = snap_expiry(p["T_S1"])

if T_L_snap  != p["T_L"]:
    print(f"  NOTE: T_L  snapped {p['T_L']}d -> {T_L_snap}d (nearest liquid expiry)")
if T_S1_snap != p["T_S1"]:
    print(f"  NOTE: T_S1 snapped {p['T_S1']}d -> {T_S1_snap}d (nearest liquid expiry)")

policy = Policy(
    alpha_L     = p["alpha_L"],
    alpha_S1    = p["alpha_S1"],
    T_L         = T_L_snap,
    T_S1        = T_S1_snap,
    base_q1     = p["base_q1"],
    beta        = p.get("beta", 0.0),
    d_min_short = p["d_min_short"],
    d_min_long  = p["d_min_long"],
    eta_pct     = p.get("eta_pct", 0.0),
)

print(f"\nPolicy (v4):")
print(f"  alpha_L={policy.alpha_L}  alpha_S1={policy.alpha_S1}")
print(f"  T_L={policy.T_L}d  T_S1={policy.T_S1}d  q1={policy.base_q1}")
print(f"  beta={policy.beta}  d_min_S={policy.d_min_short}  d_min_L={policy.d_min_long}  eta={policy.eta_pct:+.0%}")
print(f"  At spot={spot0:.2f}: K_L={spot0*policy.alpha_L:.2f}  K_S1={spot0*policy.alpha_S1:.2f}")

# ── parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--trace", action="store_true", help="Print day-by-day trace of path 0")
args, _ = parser.parse_known_args()
trace = args.trace

# ── generate paths ────────────────────────────────────────────────────────────
print(f"\nGenerating {N_PATHS} paths (seed={SEED}, horizon={HORIZON_DAYS}d, "
      f"crash_prob={ANNUAL_DEFAULT_PROB*100:.3f}%/yr) ...")
sampler = BootstrapSampler(returns, seed=SEED)
paths   = sampler.sample_paths(N_PATHS, HORIZON_DAYS, spot0,
                                annual_default_prob=ANNUAL_DEFAULT_PROB)
n_crashed = int((paths[:, -1] < 1.0).sum())
print(f"  Paths shape: {paths.shape}  |  crashed paths (end<$1): {n_crashed}")

# ── v4 simulation ─────────────────────────────────────────────────────────────
print("Running v4 simulation ...")

S  = paths                                              # (N_PATHS, HORIZON_DAYS+1)
n  = N_PATHS
q1 = float(policy.base_q1)

# High-water mark
H = np.zeros_like(S)
H[:, 0] = S[:, 0]

# ── inception cash flow ────────────────────────────────────────────────────────
K_L_inc       = spot0 * policy.alpha_L
K_S1_inc      = spot0 * policy.alpha_S1
long_ask_inc  = surface.price(K_L_inc,  policy.T_L,  spot0, side="ask") or 0.0
short_bid_inc = surface.price(K_S1_inc, policy.T_S1, spot0, side="bid") or 0.0
W_inc         = q1 * short_bid_inc - long_ask_inc - (1.0 + q1) * COST_PER_LEG

W = np.full(n, W_inc)

if trace:
    print(f"\n{'='*72}")
    print(f"PATH {TRACE_PATH} TRACE  (S0={spot0:.4f})")
    print(f"{'='*72}")
    print(f"INCEPTION (day 0)")
    print(f"  Long  put : K={K_L_inc:.4f}  T={policy.T_L}d  ask = surface({K_L_inc:.2f}, {policy.T_L}d, S={spot0:.2f}) = {long_ask_inc:.4f}")
    print(f"  Short put : K={K_S1_inc:.4f}  T={policy.T_S1}d  bid = surface({K_S1_inc:.2f}, {policy.T_S1}d, S={spot0:.2f}) = {short_bid_inc:.4f}  x{q1:.0f}")
    print(f"  Net W = {q1:.0f} x {short_bid_inc:.4f} - {long_ask_inc:.4f} - (1+{q1:.0f}) x {COST_PER_LEG:.2f} = {W_inc:.4f}")
    print(f"{'─'*72}")

# ── leg state ──────────────────────────────────────────────────────────────────
K_L  = np.full(n, K_L_inc)
K_S1 = np.full(n, K_S1_inc)

# Independent leg clocks (v4 design)
theta_short          = 0                                    # scalar: uniform, calendar only
theta_long           = np.zeros(n, dtype=int)              # per-path: floor trigger resets individually
days_since_long_roll = np.full(n, policy.d_min_long, dtype=int)

breach           = np.zeros((n, HORIZON_DAYS), dtype=np.int8)
breach_depth_sum = np.zeros(n)
roll_count       = np.zeros(n, dtype=int)
Pi_final_arr     = np.zeros(n)

# ── main day loop ──────────────────────────────────────────────────────────────
for d in range(1, HORIZON_DAYS + 1):
    H[:, d] = np.maximum(H[:, d - 1], S[:, d])
    theta_short          += 1
    theta_long           += 1
    days_since_long_roll += 1

    T_S1_rem = policy.T_S1 - theta_short          # scalar
    T_L_rem  = policy.T_L  - theta_long           # (n,) array

    # ── portfolio MTM ─────────────────────────────────────────────────────────
    T_L_vec  = np.clip(T_L_rem,  1, policy.T_L)
    T_S1_vec = float(max(T_S1_rem, 1))

    lp_val  = surface.price_vector(K_L,  T_L_vec,                    S[:, d], side="mid")
    sp1_val = surface.price_vector(K_S1, np.full(n, T_S1_vec),       S[:, d], side="mid")
    # Intrinsic floor for deep-ITM puts (surface extrapolation can underestimate)
    lp_val  = np.maximum(lp_val,  np.maximum(K_L  - S[:, d], 0.0))
    sp1_val = np.maximum(sp1_val, np.maximum(K_S1 - S[:, d], 0.0))
    Pi = lp_val - q1 * sp1_val

    if d == HORIZON_DAYS:
        Pi_final_arr = Pi.copy()

    # ── beta-blended floor ────────────────────────────────────────────────────
    floor_ref = (1.0 - policy.beta) * spot0 + policy.beta * H[:, d]
    Flr       = DELTA_FLOOR * floor_ref

    # ── breach detection ──────────────────────────────────────────────────────
    V        = S[:, d] + W + Pi
    breached = V < Flr
    breach[:, d - 1] = breached.astype(np.int8)
    breach_depth_sum += np.where(breached, Flr - V, 0.0)

    # ── roll triggers ─────────────────────────────────────────────────────────
    R_short      = (T_S1_rem <= policy.d_min_short)             # scalar bool
    R_long_cal   = (T_L_rem  <= policy.d_min_long)              # (n,) bool
    R_long_floor = (
        (K_L < Flr) &
        (S[:, d] * policy.alpha_L > K_L) &
        (days_since_long_roll >= policy.d_min_long)
    )
    R_long = R_long_cal | R_long_floor                          # (n,) bool

    # ── trace: per-day compact line ───────────────────────────────────────────
    if trace and TRACE_PATH < n:
        tp  = TRACE_PATH
        tag = ""
        if R_short:        tag += " ROLL_S"
        if R_long[tp]:     tag += " ROLL_L"
        if breached[tp]:   tag += " BREACH"
        if not tag:        tag  = " ok"
        print(
            f"  d{d:3d}: S={S[tp,d]:8.2f}  H={H[tp,d]:8.2f}  Flr={Flr[tp]:8.2f}  "
            f"V={V[tp]:8.2f}(S{S[tp,d]:+.2f}+W{W[tp]:+.2f}+Pi{Pi[tp]:+.2f})"
            f"  tS={theta_short:3d} tL={theta_long[tp]:3d}{tag}"
        )

    # ── execute short roll (all paths simultaneously) ─────────────────────────
    if R_short:
        T_S1_q = max(T_S1_rem, 1)

        sp1_ask_close = surface.price_vector(
            K_S1, np.full(n, float(T_S1_q)), S[:, d], side="ask"
        )
        sp1_ask_close = np.maximum(sp1_ask_close, np.maximum(K_S1 - S[:, d], 0.0))

        K_S1_new    = S[:, d] * policy.alpha_S1
        sp1_bid_new = surface.price_vector(
            K_S1_new, np.full(n, float(policy.T_S1)), S[:, d], side="bid"
        )

        # Short roll: receive new bid, pay close ask, cost = 2 legs per contract
        C_short = q1 * (sp1_bid_new - sp1_ask_close) - 2.0 * q1 * COST_PER_LEG
        W      += C_short
        K_S1    = K_S1_new
        theta_short = 0
        roll_count += 1

        if trace and TRACE_PATH < n:
            tp = TRACE_PATH
            print(
                f"    >> SHORT ROLL  d{d}: K_S1 {K_S1[tp]:.4f} -> {K_S1_new[tp]:.4f}"
                f"  T_rem={T_S1_q}d -> {policy.T_S1}d"
                f"  ask_close=surface({K_S1[tp]:.2f},{T_S1_q}d,{S[tp,d]:.2f})={sp1_ask_close[tp]:.4f}"
                f"  bid_new=surface({K_S1_new[tp]:.2f},{policy.T_S1}d,{S[tp,d]:.2f})={sp1_bid_new[tp]:.4f}"
                f"  C_S={q1:.0f}x({sp1_bid_new[tp]:.4f}-{sp1_ask_close[tp]:.4f})-2x{q1:.0f}x{COST_PER_LEG}={C_short[tp]:+.4f}"
                f"  W={W[tp]:.4f}"
            )

    # ── execute long roll (per-path subset) ───────────────────────────────────
    if R_long.any():
        idx = np.where(R_long)[0]
        S_r   = S[idx, d]
        T_L_q = np.clip(T_L_rem[idx], 1, policy.T_L)

        lp_bid_close = surface.price_vector(K_L[idx], T_L_q, S_r, side="bid")
        lp_bid_close = np.maximum(lp_bid_close, np.maximum(K_L[idx] - S_r, 0.0))

        K_L_new    = S_r * policy.alpha_L
        lp_ask_new = surface.price_vector(
            K_L_new, np.full(len(idx), float(policy.T_L)), S_r, side="ask"
        )
        lp_ask_new = np.maximum(lp_ask_new, np.maximum(K_L_new - S_r, 0.0))

        # Long roll: receive old bid, pay new ask, cost = 2 legs
        C_long = lp_bid_close - lp_ask_new - 2.0 * COST_PER_LEG
        W[idx]               += C_long
        K_L[idx]              = K_L_new
        theta_long[idx]       = 0
        days_since_long_roll[idx] = 0
        roll_count[idx]      += 1

        if trace and TRACE_PATH < n and (TRACE_PATH in idx):
            i       = int(np.where(idx == TRACE_PATH)[0][0])
            trigger = "calendar" if R_long_cal[TRACE_PATH] else "floor"
            tp      = TRACE_PATH
            print(
                f"    >> LONG  ROLL ({trigger})  d{d}: K_L {K_L[tp]:.4f} -> {K_L_new[i]:.4f}"
                f"  T_rem={T_L_q[i]}d -> {policy.T_L}d"
                f"  bid_close=surface({K_L[tp]:.2f},{T_L_q[i]}d,{S[tp,d]:.2f})={lp_bid_close[i]:.4f}"
                f"  ask_new=surface({K_L_new[i]:.2f},{policy.T_L}d,{S[tp,d]:.2f})={lp_ask_new[i]:.4f}"
                f"  C_L={lp_bid_close[i]:.4f}-{lp_ask_new[i]:.4f}-2x{COST_PER_LEG}={C_long[i]:+.4f}"
                f"  W={W[tp]:.4f}"
            )

if trace:
    print(f"{'─'*72}")
    print(f"PATH {TRACE_PATH} SUMMARY: total rolls={roll_count[TRACE_PATH]}  W_final={W[TRACE_PATH]:.4f}")
    print(f"{'='*72}\n")

# ── aggregate ──────────────────────────────────────────────────────────────────
S_T    = S[:, -1]
H_max  = H[:, -1]
W_opts = W.copy()

ever_breached     = breach.any(axis=1)
total_breach_days = int(breach.sum())
n_breach_paths    = int(ever_breached.sum())
avg_breach_depth  = float(breach_depth_sum.sum() / max(total_breach_days, 1))

stock_pnl       = S_T - spot0
full_pnl        = stock_pnl + W_opts + Pi_final_arr

# ideal floor at end of path: depends on beta
floor_ref_final = (1.0 - policy.beta) * spot0 + policy.beta * H_max
ideal_floor_pnl = DELTA_FLOOR * floor_ref_final - spot0
protection_gap  = full_pnl - ideal_floor_pnl

# ── print results ─────────────────────────────────────────────────────────────
def pct(label, arr, q):
    return f"  {label:28s}: {np.percentile(arr, q):+10.2f}"

print()
print("=" * 65)
print("  COMPONENT 1: Stock-only P&L  (S_T - S_0, no options)")
print("=" * 65)
print(f"  S_0 (entry)                 : ${spot0:.2f}")
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, stock_pnl, q))
print(f"  {'Mean':28s}: {stock_pnl.mean():+10.2f}")
print(f"  {'Min':28s}: {stock_pnl.min():+10.2f}")
print(f"  {'Max':28s}: {stock_pnl.max():+10.2f}")

print()
print("=" * 65)
print("  COMPONENT 2: W_options  (cumulative realized option cash)")
print("  = inception net + sum of all short-roll + long-roll flows")
print("=" * 65)
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, W_opts, q))
print(f"  {'Mean':28s}: {W_opts.mean():+10.2f}")
print(f"  {'P(W>0)':28s}: {(W_opts > 0).mean()*100:.1f}%")

print()
print("=" * 65)
print("  COMPONENT 3: Pi_final  (unrealized MTM at day 504)")
print("  = long_put_mid - q1 x short_put_mid")
print("=" * 65)
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, Pi_final_arr, q))
print(f"  {'Mean':28s}: {Pi_final_arr.mean():+10.2f}")

print()
print("=" * 65)
print("  FULL PORTFOLIO P&L  (stock + W_options + Pi_final)")
print("=" * 65)
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, full_pnl, q))
print(f"  {'Mean':28s}: {full_pnl.mean():+10.2f}")
print(f"  {'Min':28s}: {full_pnl.min():+10.2f}")
print(f"  {'Max':28s}: {full_pnl.max():+10.2f}")
print(f"  {'P(full_pnl > 0)':28s}: {(full_pnl > 0).mean()*100:.1f}%")

print()
print("=" * 65)
beta_desc = (
    f"Fixed floor = {DELTA_FLOOR:.2f} x S0 = {DELTA_FLOOR*spot0:.2f}  (beta=0, never ratchets)"
    if policy.beta == 0.0 else
    f"Floor = {DELTA_FLOOR:.2f} x [{1-policy.beta:.2f}xS0 + {policy.beta:.2f}xH(t)]"
)
print(f"  FLOOR GUARANTEE  ({beta_desc})")
print(f"  Ideal floor P&L at end = DELTA_FLOOR x floor_ref - S0")
print("=" * 65)
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, ideal_floor_pnl, q))
print(f"  {'Mean':28s}: {ideal_floor_pnl.mean():+10.2f}")

print()
print("=" * 65)
print("  PROTECTION GAP  (full_pnl - ideal_floor_pnl)")
print("  > 0 : strategy exceeded the floor target")
print("  < 0 : strategy fell short of what the floor promised")
print("=" * 65)
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, protection_gap, q))
print(f"  {'Mean':28s}: {protection_gap.mean():+10.2f}")
print(f"  {'P(gap > 0)':28s}: {(protection_gap > 0).mean()*100:.1f}%")

print()
print("=" * 65)
print(f"  BREACH ANALYSIS  (V = S + W + Pi < {DELTA_FLOOR:.2f} x floor_ref)")
print("=" * 65)
print(f"  Paths with at least 1 breach : {n_breach_paths} / {N_PATHS}  ({ever_breached.mean()*100:.1f}%)")
print(f"  P_success (zero breaches)    : {(~ever_breached).mean()*100:.1f}%")
print(f"  Total breach-days            : {total_breach_days}")
print(f"  Avg breach depth per event   : ${avg_breach_depth:.2f}")
print()

bins   = [0, 133, 200, 300, 500, 10_000]
labels = ["<133 (stock fell)", "133-200", "200-300", "300-500", ">500"]
print("  Breach rate by peak H_max:")
for i, lbl in enumerate(labels):
    mask = (H_max >= bins[i]) & (H_max < bins[i + 1])
    if mask.sum() > 0:
        rate = ever_breached[mask].mean()
        print(f"    H_max {lbl:20s}: {mask.sum():4d} paths  breach rate={rate*100:.1f}%")

print()
print(f"  Mean rolls per path          : {roll_count.mean():.1f}")

# ── save CSV ───────────────────────────────────────────────────────────────────
_results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_results_dir, exist_ok=True)

df = pd.DataFrame({
    "path_idx":        np.arange(N_PATHS),
    "S_T":             S_T.round(2),
    "H_max":           H_max.round(2),
    "stock_pnl":       stock_pnl.round(2),
    "W_options":       W_opts.round(2),
    "Pi_final":        Pi_final_arr.round(2),
    "full_pnl":        full_pnl.round(2),
    "ideal_floor_pnl": ideal_floor_pnl.round(2),
    "protection_gap":  protection_gap.round(2),
    "ever_breached":   ever_breached.astype(int),
    "n_rolls":         roll_count,
})
_csv_path = os.path.join(_results_dir, "full_pnl_analysis.csv")
df.to_csv(_csv_path, index=False)
print(f"Saved {len(df)} rows to {_csv_path}")

# ── recompute metrics and update best_policy.json ─────────────────────────────
cvar_cutoff    = int(np.ceil(0.10 * N_PATHS))
P_success_new  = float((~ever_breached).mean())
E_W_new        = float(W_opts.mean())
CVaR_W_new     = float(np.sort(W_opts)[:cvar_cutoff].mean()) if cvar_cutoff > 0 else float(W_opts.min())
P_W_pos_new    = float((W_opts > 0).mean())
E_breach_new   = avg_breach_depth
n_rolls_new    = float(roll_count.mean())
score_new      = P_success_new - E_breach_new / spot0   # v4 optimizer score formula

bp["metrics"] = {
    "P_success":      round(P_success_new, 4),
    "E_W":            round(E_W_new,       4),
    "CVaR_W":         round(CVaR_W_new,    4),
    "P_W_positive":   round(P_W_pos_new,   4),
    "E_breach_depth": round(E_breach_new,  4),
    "n_rolls":        round(n_rolls_new,   1),
    "score":          round(score_new,     4),
}
bp["analysis_params"] = {
    "N_PATHS":             N_PATHS,
    "HORIZON_DAYS":        HORIZON_DAYS,
    "DELTA_FLOOR":         DELTA_FLOOR,
    "ANNUAL_DEFAULT_PROB": ANNUAL_DEFAULT_PROB,
    "BTC_ERA":             BTC_ERA,
    "SEED":                SEED,
}

with open(_json_path, "w") as f:
    json.dump(bp, f, indent=2)

print()
print("=" * 65)
print("  UPDATED metrics saved to results/best_policy.json")
print("=" * 65)
print(f"  P_success    : {P_success_new*100:.2f}%")
print(f"  E_W          : {E_W_new:+.2f}")
print(f"  CVaR_W (10%) : {CVaR_W_new:+.2f}")
print(f"  P(W>0)       : {P_W_pos_new*100:.1f}%")
print(f"  E_breach_depth: ${E_breach_new:.2f}")
print(f"  n_rolls      : {n_rolls_new:.1f}")
print(f"  score        : {score_new:.4f}  (P_success - E_breach/spot)")
print(f"  [BTC_ERA={BTC_ERA}  DELTA_FLOOR={DELTA_FLOOR}  N={N_PATHS}  H={HORIZON_DAYS}d  seed={SEED}]")
