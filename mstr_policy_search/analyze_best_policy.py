#!/usr/bin/env python3
"""
analyze_best_policy.py
----------------------
Runs the best policy and produces a full portfolio P&L breakdown:

  Full P&L = (S_T - S_0)               [stock leg]
           + W_options                  [cumulative realized option cash flows]
           + Pi_final                   [unrealized MTM of the open position at T]

Compares to:
  Unprotected P&L = S_T - S_0          [just hold MSTR]
  Ideal floor P&L  = H_max * 0.70 - S_0  [the minimum the protection should have kept]

Saves: results/full_pnl_analysis.csv
"""
import sys, os, io, json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))

import data_loader
from surface import build_surface
from bootstrap import BootstrapSampler
from policy_grid import Policy
from surface import FrozenSurface

# ── load cached data ──────────────────────────────────────────────────────────
# Set BTC_ERA=True to restrict bootstrap to post-2020-08-11 (Bitcoin treasury era)
# Set BTC_ERA=False to use full history (includes dot-com -96% crashes)
BTC_ERA = True

chain, spot0, _ = data_loader.load_option_chain("MSTR", refresh=False)
cutoff = data_loader.BTC_ERA_CUTOFF if BTC_ERA else None
returns = data_loader.load_historical_returns("MSTR", refresh=False, cutoff_date=cutoff)
surface          = build_surface(chain, spot0)

# ── best policy ───────────────────────────────────────────────────────────────
with open("results/best_policy.json") as f:
    bp = json.load(f)

p = bp["policy"]
policy = Policy(
    alpha_L  = p["alpha_L"],
    alpha_S1 = p["alpha_S1"],
    alpha_S2 = p["alpha_S2"],
    T_L      = p["T_L"],
    T_S1     = p["T_S1"],
    T_S2     = p["T_S2"],
    base_q1  = p["base_q1"],
    base_q2  = p["base_q2"],
    gamma    = p["gamma"],
    d_min    = p["d_min"],
)

# ── simulation parameters (must match main.py fine stage) ────────────────────
N_PATHS      = 2_000
HORIZON_DAYS = 504
DELTA        = 0.30   # floor = H * (1 - delta)
SEED         = 42

print(f"\nRegenerating {N_PATHS} fine paths (seed={SEED}, horizon={HORIZON_DAYS}d) ...")
sampler = BootstrapSampler(returns, seed=SEED)
paths   = sampler.sample_paths(N_PATHS, HORIZON_DAYS, spot0)  # (N, 505)

print(f"Re-running best policy simulation with full tracking ...")

# ── full simulation with extra tracking ──────────────────────────────────────
# We replicate simulate_policy logic but also record:
#   S_final[i]   = paths[i, -1]
#   H_max[i]     = max(paths[i, :])   [peak high-water mark over path]
#   Pi_final[i]  = option portfolio MTM at day 504 (final open position)

S = paths                                          # (N_PATHS, 505)
n_paths = N_PATHS

# High-water mark
H = np.zeros_like(S)
H[:, 0] = S[:, 0]

# ── Initial inception cash flow
def inception_cf(policy, surface, spot0):
    K_L  = spot0 * policy.alpha_L
    K_S1 = spot0 * policy.alpha_S1
    cost = surface.price(K_L,  policy.T_L,  spot0, "ask") or 0.0
    recv = policy.base_q1 * (surface.price(K_S1, policy.T_S1, spot0, "bid") or 0.0)
    if policy.alpha_S2 > 0:
        K_S2 = spot0 * policy.alpha_S2
        recv += policy.base_q2 * (surface.price(K_S2, policy.T_S2, spot0, "bid") or 0.0)
    return recv - cost

W              = np.full(n_paths, inception_cf(policy, surface, spot0))
K_L_cur        = np.full(n_paths, spot0 * policy.alpha_L)
K_S1_cur       = np.full(n_paths, spot0 * policy.alpha_S1)
K_S2_cur       = np.full(n_paths, spot0 * policy.alpha_S2) if policy.alpha_S2 > 0 else None
theta_age      = np.zeros(n_paths, dtype=int)
days_since_roll= np.full(n_paths, policy.d_min, dtype=int)
q1             = np.full(n_paths, float(policy.base_q1))
q2             = np.full(n_paths, float(policy.base_q2)) if policy.alpha_S2 > 0 else np.zeros(n_paths)
breach         = np.zeros((n_paths, HORIZON_DAYS), dtype=np.int8)
roll_count     = np.zeros(n_paths, dtype=int)
breach_depth_sum = np.zeros(n_paths)

# S_10pct for C3 (same as simulator.py)
rng10 = np.random.default_rng(0)
log_1yr = np.sum(rng10.choice(returns, size=(5000, 252), replace=True), axis=1)
S_10pct = spot0 * np.exp(np.percentile(log_1yr, 10))

def q_max_c3(K_L, K_S, S_10pct, floor_val):
    if K_S > S_10pct:
        return max(1, int((K_L - floor_val) / (K_S - S_10pct)))
    return 99

# store final-day portfolio state to compute Pi_final
Pi_final_arr   = np.zeros(n_paths)

for d in range(1, HORIZON_DAYS + 1):
    H[:, d] = np.maximum(H[:, d - 1], S[:, d])
    theta_age += 1
    days_since_roll += 1

    T_L_rem  = policy.T_L  - theta_age
    T_S1_rem = policy.T_S1 - theta_age
    T_S2_rem = policy.T_S2 - theta_age if policy.alpha_S2 > 0 else 0

    T_L_v  = np.clip(T_L_rem,  1, policy.T_L)
    T_S1_v = np.clip(T_S1_rem, 1, policy.T_S1)

    lp_val  = surface.price_vector(K_L_cur,  T_L_v,  S[:, d], "mid")
    sp1_val = surface.price_vector(K_S1_cur, T_S1_v, S[:, d], "mid")
    lp_val  = np.maximum(lp_val,  np.maximum(K_L_cur  - S[:, d], 0.0))
    sp1_val = np.maximum(sp1_val, np.maximum(K_S1_cur - S[:, d], 0.0))
    Pi      = lp_val - q1 * sp1_val

    if policy.alpha_S2 > 0:
        T_S2_v  = np.clip(T_S2_rem, 1, policy.T_S2)
        sp2_val = surface.price_vector(K_S2_cur, T_S2_v, S[:, d], "mid")
        sp2_val = np.maximum(sp2_val, np.maximum(K_S2_cur - S[:, d], 0.0))
        Pi     -= q2 * sp2_val

    if d == HORIZON_DAYS:
        Pi_final_arr = Pi.copy()

    V   = S[:, d] + W + Pi      # total wealth: stock + realized cash + open MTM
    Flr = H[:, d] * (1.0 - DELTA)
    breached = (V < Flr)
    breach[:, d - 1] = breached.astype(np.int8)
    breach_depth_sum += np.where(breached, Flr - V, 0.0)

    R_calendar = (T_S1_rem <= policy.d_min)
    R_floor    = (
        (K_L_cur < Flr) &
        (S[:, d] * policy.alpha_L > K_L_cur) &
        (days_since_roll >= policy.d_min)
    )
    roll_mask = R_calendar | R_floor

    if roll_mask.any():
        idx = np.where(roll_mask)[0]
        S_r = S[idx, d]
        T_L_q  = np.clip(T_L_rem,  1, policy.T_L)
        T_S1_q = np.clip(T_S1_rem, 1, policy.T_S1)

        lp_bid  = surface.price_vector(K_L_cur[idx],  T_L_q[idx],  S_r, "bid")
        sp1_bid = surface.price_vector(K_S1_cur[idx], T_S1_q[idx], S_r, "bid")
        lp_bid  = np.maximum(lp_bid,  np.maximum(K_L_cur[idx]  - S_r, 0.0))
        sp1_bid = np.maximum(sp1_bid, np.maximum(K_S1_cur[idx] - S_r, 0.0))
        cash_in = lp_bid + q1[idx] * sp1_bid
        if policy.alpha_S2 > 0:
            T_S2_q  = np.clip(T_S2_rem, 1, policy.T_S2)
            sp2_bid = surface.price_vector(K_S2_cur[idx], T_S2_q[idx], S_r, "bid")
            sp2_bid = np.maximum(sp2_bid, np.maximum(K_S2_cur[idx] - S_r, 0.0))
            cash_in += q2[idx] * sp2_bid

        K_L_new  = S_r * policy.alpha_L
        K_S1_new = S_r * policy.alpha_S1
        K_S2_new = S_r * policy.alpha_S2 if policy.alpha_S2 > 0 else None

        deficit  = np.maximum(0.0, -W[idx])
        P_S1_ask = surface.price_vector(K_S1_new, np.full(len(idx), float(policy.T_S1)), S_r, "ask")
        floor_r  = H[idx, d] * (1.0 - DELTA)
        q_max1   = np.array([q_max_c3(K_L_new[i], K_S1_new[i], S_10pct, floor_r[i]) for i in range(len(idx))], dtype=float)
        q1_new   = np.clip(np.floor(policy.base_q1 + policy.gamma * deficit / (P_S1_ask + 1e-6)), 1, q_max1)

        q2_new = np.zeros(len(idx))
        if policy.alpha_S2 > 0:
            P_S2_ask = surface.price_vector(K_S2_new, np.full(len(idx), float(policy.T_S2)), S_r, "ask")
            q_max2   = np.array([q_max_c3(K_L_new[i], K_S2_new[i], S_10pct, floor_r[i]) for i in range(len(idx))], dtype=float)
            q2_new   = np.clip(np.floor(policy.base_q2 + policy.gamma * deficit / (P_S2_ask + 1e-6)), 0, q_max2)

        cash_out  = surface.price_vector(K_L_new, np.full(len(idx), float(policy.T_L)), S_r, "ask")
        cash_recv = q1_new * surface.price_vector(K_S1_new, np.full(len(idx), float(policy.T_S1)), S_r, "bid")
        if policy.alpha_S2 > 0:
            cash_recv += q2_new * surface.price_vector(K_S2_new, np.full(len(idx), float(policy.T_S2)), S_r, "bid")

        C_roll = cash_in + cash_recv - cash_out
        W[idx] += C_roll

        K_L_cur[idx]  = K_L_new
        K_S1_cur[idx] = K_S1_new
        if policy.alpha_S2 > 0:
            K_S2_cur[idx] = K_S2_new
        q1[idx] = q1_new
        q2[idx] = q2_new
        theta_age[idx] = 0
        days_since_roll[idx] = 0
        roll_count[idx] += 1

# ── final values ──────────────────────────────────────────────────────────────
S_T    = S[:, -1]              # final MSTR price at day 504
H_max  = H[:, -1]             # peak high-water mark (H is monotone non-decreasing)
W_opts = W                    # cumulative option cash flow (inception + all rolls)

# ── P&L components ────────────────────────────────────────────────────────────
stock_pnl       = S_T - spot0                   # stock-only P&L (no options)
full_pnl        = stock_pnl + W_opts + Pi_final_arr  # total protected portfolio P&L

# ideal_floor_pnl: the minimum P&L the strategy "should" have delivered
# if the floor V >= H_max * 0.70 was always maintained.
# At the end: V_min_required = H_max * 0.70
# V = S_T + Pi_final, so protected P&L >= H_max * 0.70 - spot0 IF the strategy worked.
# We define this as the "protection target" = what the floor guarantees at end of path.
ideal_floor_pnl = H_max * (1.0 - DELTA) - spot0   # = H_max * 0.70 - 133.53

# protection_gap: how much full_pnl beats (or misses) the ideal floor
protection_gap  = full_pnl - ideal_floor_pnl

ever_breached   = breach.any(axis=1)

# ── build output dataframe ───────────────────────────────────────────────────
df = pd.DataFrame({
    "path_idx":       np.arange(N_PATHS),
    "S_T":            S_T.round(2),
    "H_max":          H_max.round(2),
    "stock_pnl":      stock_pnl.round(2),      # S_T - S_0 (unprotected)
    "W_options":      W_opts.round(2),          # cumulative option cash flow
    "Pi_final":       Pi_final_arr.round(2),    # open option MTM at day 504
    "full_pnl":       full_pnl.round(2),        # total P&L = stock + options realized + unrealized
    "ideal_floor_pnl":ideal_floor_pnl.round(2), # minimum P&L implied by the floor guarantee
    "protection_gap": protection_gap.round(2),  # full_pnl - ideal_floor_pnl (>0 = exceeded floor)
    "ever_breached":  ever_breached.astype(int),
})

# ── summary print ─────────────────────────────────────────────────────────────
def pct(label, arr, q):
    return f"  {label:28s}: {np.percentile(arr, q):+10.2f}"

print()
print("=" * 65)
print("  COMPONENT 1: Stock-only P&L  (S_T - S_0,  no options)")
print("=" * 65)
print(f"  S_0 (entry)                 : ${spot0:.2f}")
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, stock_pnl, q))
print(f"  {'Mean':28s}: {stock_pnl.mean():+10.2f}")
print(f"  {'Min':28s}: {stock_pnl.min():+10.2f}")
print(f"  {'Max':28s}: {stock_pnl.max():+10.2f}")

print()
print("=" * 65)
print("  COMPONENT 2: W_options  (all realized option cash flows)")
print("  = inception debit + sum of all 168 roll cash flows")
print("  NOTE: does NOT include the final open position MTM")
print("=" * 65)
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, W_opts, q))
print(f"  {'Mean':28s}: {W_opts.mean():+10.2f}")

print()
print("=" * 65)
print("  COMPONENT 3: Pi_final  (unrealized option MTM at day 504)")
print("  = long_put_mid - q1*short1_mid - q2*short2_mid")
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
print(f"  P(full_pnl > 0)             : {(full_pnl > 0).mean()*100:.1f}%")

print()
print("=" * 65)
print("  IDEAL FLOOR  (H_max * 0.70 - S_0)")
print("  = the minimum P&L the strategy promised to protect")
print("=" * 65)
print(f"  The floor guarantee means: at all times,")
print(f"  (S + Pi) >= H * 0.70")
print(f"  At end of path, ideal floor P&L = H_max * 0.70 - S_0")
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, ideal_floor_pnl, q))
print(f"  {'Mean':28s}: {ideal_floor_pnl.mean():+10.2f}")

print()
print("=" * 65)
print("  PROTECTION GAP  (full_pnl - ideal_floor_pnl)")
print("  > 0: strategy exceeded the floor target")
print("  < 0: strategy fell short of what the floor promised")
print("=" * 65)
for q, lbl in [(5,'P5'),(10,'P10'),(25,'P25'),(50,'Median'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(pct(lbl, protection_gap, q))
print(f"  {'Mean':28s}: {protection_gap.mean():+10.2f}")
print(f"  P(gap > 0)                  : {(protection_gap > 0).mean()*100:.1f}%")

print()
print("=" * 65)
print("  BREACH ANALYSIS")
print("  A breach = any day where (S + W + Pi) < H * 0.70")
print("  S = stock price,  W = cumulative realized cash,  Pi = open option MTM")
print("=" * 65)
print(f"  Paths with at least 1 breach: {ever_breached.sum()} / {N_PATHS}  ({ever_breached.mean()*100:.1f}%)")
print(f"  Avg breach depth (when breach): ${breach_depth_sum[ever_breached].mean() / np.maximum(breach[:].any(axis=1)[ever_breached].sum(), 1):.2f}")
print()
# Show breach stats by H_max bin
bins = [0, 133, 200, 300, 500, 10000]
labels = ["<133 (stock fell)", "133-200", "200-300", "300-500", ">500"]
print("  Breach rate by peak H_max:")
for i, lbl in enumerate(labels):
    mask = (H_max >= bins[i]) & (H_max < bins[i+1])
    if mask.sum() > 0:
        breach_rate = ever_breached[mask].mean()
        print(f"    H_max {lbl:20s}: {mask.sum():4d} paths  breach rate={breach_rate*100:.1f}%")

# save
os.makedirs("results", exist_ok=True)
df.to_csv("results/full_pnl_analysis.csv", index=False)
print(f"\nSaved {len(df)} rows to results/full_pnl_analysis.csv")
print("Columns: path_idx, S_T, H_max, stock_pnl, W_options, Pi_final, full_pnl, ideal_floor_pnl, protection_gap, ever_breached")
