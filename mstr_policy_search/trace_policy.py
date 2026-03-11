#!/usr/bin/env python3
"""
trace_policy.py
---------------
Single-path execution trace for the best policy loaded from best_policy.json.
Shows surface lookups, cash flows, and running score metrics at every event.

Usage:
    cd mstr_policy_search
    python trace_policy.py              # trace path 0  (seed 42)
    python trace_policy.py --path 7    # trace path 7
    python trace_policy.py --path 7 --seed 99
    python trace_policy.py --all-days  # print every day, not just events

Output:
    Formatted event log on stdout
    results/trace_path_<N>.csv  with full day-by-day state
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

# ── analysis parameters (must match main.py / analyze_best_policy.py) ─────────
BTC_ERA             = True
BTC_ERA_CUTOFF      = "2020-08-11"
N_PATHS             = 2_000          # generate enough paths so --path N works up to N-1
HORIZON_DAYS        = 504
DELTA_FLOOR         = 0.80
ANNUAL_DEFAULT_PROB = 0.001
COST_PER_LEG        = 1.0
SEED                = 42

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--path",     type=int,  default=0,     help="Path index to trace (0-based)")
parser.add_argument("--seed",     type=int,  default=SEED,  help="Bootstrap RNG seed")
parser.add_argument("--all-days", action="store_true",      help="Print every day, not just events")
args = parser.parse_args()

TRACE_IDX = args.path
SEED      = args.seed
ALL_DAYS  = args.all_days

W = 72   # output column width

# ── load data ─────────────────────────────────────────────────────────────────
chain, spot0, _ = data_loader.load_option_chain("MSTR", refresh=False)
returns = data_loader.load_historical_returns(
    "MSTR", refresh=False,
    cutoff_date=BTC_ERA_CUTOFF if BTC_ERA else None,
)
surface = build_surface(chain, spot0)

_chain_df  = pd.DataFrame(chain)
_expiries  = sorted(_chain_df["days_out"].unique())

def snap_expiry(t: int) -> int:
    return min(_expiries, key=lambda e: abs(e - t))

# ── load best policy ──────────────────────────────────────────────────────────
_json_path = os.path.join(os.path.dirname(__file__), "results", "best_policy.json")
with open(_json_path) as f:
    bp = json.load(f)

p = bp["policy"]
T_L  = snap_expiry(p["T_L"])
T_S1 = snap_expiry(p["T_S1"])

policy = Policy(
    alpha_L     = p["alpha_L"],
    alpha_S1    = p["alpha_S1"],
    T_L         = T_L,
    T_S1        = T_S1,
    base_q1     = p["base_q1"],
    beta        = p.get("beta", 0.0),
    d_min_short = p["d_min_short"],
    d_min_long  = p["d_min_long"],
    eta_pct     = p.get("eta_pct", 0.0),
)
q1 = float(policy.base_q1)

# ── generate paths ────────────────────────────────────────────────────────────
sampler = BootstrapSampler(returns, seed=SEED)
paths   = sampler.sample_paths(N_PATHS, HORIZON_DAYS, spot0,
                                annual_default_prob=ANNUAL_DEFAULT_PROB)

if TRACE_IDX >= N_PATHS:
    sys.exit(f"ERROR: --path {TRACE_IDX} out of range (0..{N_PATHS-1})")

S_full = paths[TRACE_IDX]                 # shape (HORIZON_DAYS+1,)

# ── helpers ───────────────────────────────────────────────────────────────────
def surflookup(K, T, S, side):
    """Return surface price + intrinsic-floored value with label."""
    raw = surface.price(float(K), float(T), float(S), side=side) or 0.0
    val = max(raw, max(K - S, 0.0))
    return val

def hdr(title):
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")

def sep():
    print(f"{'─'*W}")

# ── HEADER ────────────────────────────────────────────────────────────────────
hdr(f"POLICY TRACE  |  path={TRACE_IDX}  seed={SEED}  "
    f"BTC-era={BTC_ERA}  {HORIZON_DAYS}d horizon")

print(f"  Policy : aL={policy.alpha_L}  aS={policy.alpha_S1}  "
      f"TL={policy.T_L}d  TS={policy.T_S1}d  q1={policy.base_q1}")
print(f"           beta={policy.beta}  dS={policy.d_min_short}  "
      f"dL={policy.d_min_long}  eta={policy.eta_pct:+.0%}")
print(f"  Snap   : T_L {p['T_L']}d->{T_L}d  T_S1 {p['T_S1']}d->{T_S1}d")
print(f"  S0     : {spot0:.4f}")
floor_ref0 = (1.0 - policy.beta) * spot0 + policy.beta * spot0
print(f"  Floor  : {DELTA_FLOOR:.2f} x [{1-policy.beta:.2f}xS0 + {policy.beta:.2f}xH]  "
      f"= {DELTA_FLOOR * floor_ref0:.4f} at t=0")

# ── INCEPTION ─────────────────────────────────────────────────────────────────
K_L_inc  = spot0 * policy.alpha_L
K_S1_inc = spot0 * policy.alpha_S1

long_ask_inc  = surflookup(K_L_inc,  policy.T_L,  spot0, "ask")
short_bid_inc = surflookup(K_S1_inc, policy.T_S1, spot0, "bid")
W_inc = q1 * short_bid_inc - long_ask_inc - (1.0 + q1) * COST_PER_LEG

sep()
print(f"  INCEPTION (day 0)")
print(f"    Long  put : K={K_L_inc:.4f}  T={policy.T_L}d  S={spot0:.4f}")
print(f"                ask = surface({policy.alpha_L:.2f}x{spot0:.4f}, {policy.T_L}d) = {long_ask_inc:.4f}")
print(f"    Short put : K={K_S1_inc:.4f}  T={policy.T_S1}d  S={spot0:.4f}")
print(f"                bid = surface({policy.alpha_S1:.2f}x{spot0:.4f}, {policy.T_S1}d) = {short_bid_inc:.4f}  x{q1:.0f}")
print(f"    Net W = {q1:.0f}x{short_bid_inc:.4f} - {long_ask_inc:.4f} - "
      f"(1+{q1:.0f})x{COST_PER_LEG:.2f} = {W_inc:+.4f}")

# ── SIMULATION ────────────────────────────────────────────────────────────────
# Day-by-day records
records = []       # list of dicts: one per day
roll_events = []   # list of dicts: one per roll event

# State
W_cur   = W_inc
K_L_cur = K_L_inc
K_S1_cur = K_S1_inc

H_cur   = spot0
theta_short          = 0
theta_long           = 0
days_since_long_roll = policy.d_min_long

breach_days  = 0
depth_sum    = 0.0
roll_count   = 0

for d in range(1, HORIZON_DAYS + 1):
    S_d  = S_full[d]
    H_cur = max(H_cur, S_d)

    theta_short          += 1
    theta_long           += 1
    days_since_long_roll += 1

    T_S1_rem = policy.T_S1 - theta_short
    T_L_rem  = policy.T_L  - theta_long

    # MTM
    T_L_q  = max(T_L_rem,  1)
    T_S1_q = max(T_S1_rem, 1)

    lp_mid  = surflookup(K_L_cur,  T_L_q,  S_d, "mid")
    sp1_mid = surflookup(K_S1_cur, T_S1_q, S_d, "mid")
    Pi      = lp_mid - q1 * sp1_mid

    # Floor
    floor_ref = (1.0 - policy.beta) * spot0 + policy.beta * H_cur
    Flr       = DELTA_FLOOR * floor_ref
    V         = S_d + W_cur + Pi

    breached = V < Flr
    depth    = float(Flr - V) if breached else 0.0
    if breached:
        breach_days += 1
        depth_sum   += depth

    # Roll triggers
    R_short      = (T_S1_rem <= policy.d_min_short)
    R_long_cal   = (T_L_rem  <= policy.d_min_long)
    R_long_floor = (
        K_L_cur < Flr and
        S_d * policy.alpha_L > K_L_cur and
        days_since_long_roll >= policy.d_min_long
    )
    R_long = R_long_cal or R_long_floor

    evt_types = []
    roll_detail = {}

    # ── SHORT ROLL ────────────────────────────────────────────────────────────
    if R_short:
        sp1_ask_close = surflookup(K_S1_cur, T_S1_q, S_d, "ask")
        K_S1_new      = S_d * policy.alpha_S1
        sp1_bid_new   = surflookup(K_S1_new, policy.T_S1, S_d, "bid")
        C_short       = q1 * (sp1_bid_new - sp1_ask_close) - 2.0 * q1 * COST_PER_LEG

        roll_detail["short"] = dict(
            K_old=K_S1_cur, T_rem=T_S1_q, ask_close=sp1_ask_close,
            K_new=K_S1_new, T_new=policy.T_S1, bid_new=sp1_bid_new,
            C=C_short, W_before=W_cur, W_after=W_cur + C_short,
        )
        W_cur    = W_cur + C_short
        K_S1_cur = K_S1_new
        theta_short = 0
        roll_count += 1
        evt_types.append("ROLL_S")

    # ── LONG ROLL ─────────────────────────────────────────────────────────────
    if R_long:
        trigger      = "calendar" if R_long_cal else "floor"
        lp_bid_close = surflookup(K_L_cur, T_L_q, S_d, "bid")
        K_L_new      = S_d * policy.alpha_L
        lp_ask_new   = surflookup(K_L_new, policy.T_L, S_d, "ask")
        C_long       = lp_bid_close - lp_ask_new - 2.0 * COST_PER_LEG

        roll_detail["long"] = dict(
            trigger=trigger,
            K_old=K_L_cur, T_rem=T_L_q, bid_close=lp_bid_close,
            K_new=K_L_new, T_new=policy.T_L, ask_new=lp_ask_new,
            C=C_long, W_before=W_cur, W_after=W_cur + C_long,
        )
        W_cur    = W_cur + C_long
        K_L_cur  = K_L_new
        theta_long           = 0
        days_since_long_roll = 0
        roll_count += 1
        evt_types.append("ROLL_L")

    if breached:
        evt_types.append("BREACH")

    # Recompute Pi/V after rolls (for accurate record)
    T_L_q2  = max(policy.T_L - theta_long,  1)
    T_S1_q2 = max(policy.T_S1 - theta_short, 1)
    lp_mid2  = surflookup(K_L_cur,  T_L_q2,  S_d, "mid")
    sp1_mid2 = surflookup(K_S1_cur, T_S1_q2, S_d, "mid")
    Pi2  = lp_mid2 - q1 * sp1_mid2
    V2   = S_d + W_cur + Pi2

    # Running score at this point (single-path snapshot)
    single_path_breach = breach_days > 0
    sp_score = (0.0 if single_path_breach else 1.0) - (depth_sum / max(breach_days, 1)) / spot0

    records.append(dict(
        day=d, S=S_d, H=H_cur, Flr=Flr, W=W_cur, Pi=Pi2, V=V2,
        breach=int(breached), depth=depth,
        K_L=K_L_cur, K_S1=K_S1_cur, T_L_rem=T_L_q2, T_S1_rem=T_S1_q2,
        tS=theta_short, tL=theta_long,
        roll_count=roll_count, breach_days=breach_days, depth_sum=depth_sum,
        event=",".join(evt_types) if evt_types else "",
    ))

    if roll_detail:
        roll_events.append(dict(day=d, S=S_d, H=H_cur, Flr=Flr, detail=roll_detail,
                                W_after=W_cur, breach_days=breach_days, depth_sum=depth_sum,
                                roll_count=roll_count))

# ── PRINT EVENT LOG ───────────────────────────────────────────────────────────
sep()
print(f"  EVENT LOG  (events only — use --all-days for full table)")
sep()
print(f"  {'d':>4}  {'S':>8}  {'H':>8}  {'Flr':>8}  "
      f"{'W':>8}  {'Pi':>8}  {'V':>8}  {'Bre':>3}  {'Depth':>7}  "
      f"{'Rolls':>5}  Event")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  "
      f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*3}  {'─'*7}  "
      f"{'─'*5}  {'─'*14}")

# Inception row
floor_ref0 = (1.0 - policy.beta) * spot0 + policy.beta * spot0
Flr0 = DELTA_FLOOR * floor_ref0
V0   = spot0 + W_inc
print(f"  {'0':>4}  {spot0:>8.2f}  {spot0:>8.2f}  {Flr0:>8.2f}  "
      f"{W_inc:>+8.2f}  {'---':>8}  {V0:>8.2f}  {'---':>3}  {'---':>7}  "
      f"{'0':>5}  INCEPTION")

df = pd.DataFrame(records)
is_event = df["event"].str.len() > 0

rows_to_print = df[is_event] if not ALL_DAYS else df
for _, r in rows_to_print.iterrows():
    bre_str   = "YES" if r["breach"] else "."
    depth_str = f"${r['depth']:.2f}" if r["breach"] else "."
    print(f"  {int(r['day']):>4}  {r['S']:>8.2f}  {r['H']:>8.2f}  {r['Flr']:>8.2f}  "
          f"{r['W']:>+8.2f}  {r['Pi']:>+8.2f}  {r['V']:>8.2f}  {bre_str:>3}  {depth_str:>7}  "
          f"{int(r['roll_count']):>5}  {r['event']}")

# ── ROLL EVENT DETAIL CARDS ───────────────────────────────────────────────────
if roll_events:
    sep()
    print(f"  ROLL DETAIL  ({len(roll_events)} roll events)")
    for ev in roll_events:
        sep()
        d   = ev["day"]
        S_d = ev["S"]
        H_d = ev["H"]
        fl  = ev["Flr"]
        det = ev["detail"]
        print(f"  DAY {d}  S={S_d:.4f}  H={H_d:.4f}  Floor={fl:.4f}")

        if "short" in det:
            s = det["short"]
            print(f"  SHORT ROLL (calendar: T_S1_rem={s['T_rem']}d <= d_min_short={policy.d_min_short})")
            print(f"    Close: K_S1={s['K_old']:.4f}  T_rem={s['T_rem']}d  S={S_d:.4f}")
            print(f"           ask = surface({policy.alpha_S1:.2f}xS, {s['T_rem']}d) = {s['ask_close']:.4f}")
            print(f"    Open : K_S1={s['K_new']:.4f}  T={s['T_new']}d  S={S_d:.4f}")
            print(f"           bid = surface({policy.alpha_S1:.2f}xS, {s['T_new']}d) = {s['bid_new']:.4f}")
            print(f"    C_short = {q1:.0f}x({s['bid_new']:.4f} - {s['ask_close']:.4f}) "
                  f"- 2x{q1:.0f}x{COST_PER_LEG:.2f} = {s['C']:+.4f}")
            print(f"    W: {s['W_before']:+.4f} -> {s['W_after']:+.4f}")

        if "long" in det:
            ll = det["long"]
            trig = ll["trigger"]
            print(f"  LONG ROLL ({trig}: "
                  + (f"T_L_rem={ll['T_rem']}d <= d_min_long={policy.d_min_long})"
                     if trig == "calendar"
                     else f"K_L={ll['K_old']:.2f} < Floor={fl:.2f})"))
            print(f"    Close: K_L={ll['K_old']:.4f}  T_rem={ll['T_rem']}d  S={S_d:.4f}")
            print(f"           bid = surface({policy.alpha_L:.2f}xS, {ll['T_rem']}d) = {ll['bid_close']:.4f}")
            print(f"    Open : K_L={ll['K_new']:.4f}  T={ll['T_new']}d  S={S_d:.4f}")
            print(f"           ask = surface({policy.alpha_L:.2f}xS, {ll['T_new']}d) = {ll['ask_new']:.4f}")
            print(f"    C_long  = {ll['bid_close']:.4f} - {ll['ask_new']:.4f} "
                  f"- 2x{COST_PER_LEG:.2f} = {ll['C']:+.4f}")
            print(f"    W: {ll['W_before']:+.4f} -> {ll['W_after']:+.4f}")

        print(f"    >> Running: rolls={ev['roll_count']}  "
              f"breach_days={ev['breach_days']}  depth_sum=${ev['depth_sum']:.2f}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
last = df.iloc[-1]
ever_breached  = breach_days > 0
W_final        = last["W"]
Pi_final       = last["Pi"]
S_final        = last["S"]
H_max          = last["H"]
full_pnl       = (S_final - spot0) + W_final + Pi_final
floor_ref_end  = (1.0 - policy.beta) * spot0 + policy.beta * H_max
ideal_floor    = DELTA_FLOOR * floor_ref_end - spot0

# Score decomposition
ps_contrib      = 0.0 if ever_breached else 1.0
breach_depth_pp = (depth_sum / max(breach_days, 1)) if breach_days > 0 else 0.0
path_score      = ps_contrib - breach_depth_pp / spot0

hdr("FINAL SUMMARY")
print(f"  S_final        : {S_final:.4f}  (S0={spot0:.4f}  stock PnL={S_final-spot0:+.2f})")
print(f"  H_max          : {H_max:.4f}")
print(f"  W_final        : {W_final:+.4f}  (option realized cash)")
print(f"  Pi_final       : {Pi_final:+.4f}  (open MTM at day {HORIZON_DAYS})")
print(f"  Full PnL       : {full_pnl:+.4f}  (stock + W + Pi)")
print(f"  Ideal floor PnL: {ideal_floor:+.4f}  ({DELTA_FLOOR:.2f} x floor_ref - S0)")
print(f"  Protection gap : {full_pnl - ideal_floor:+.4f}  (Full PnL - Ideal floor PnL)")
print()
print(f"  n_rolls        : {roll_count}")
print(f"  breach_days    : {breach_days}")
print(f"  depth_sum      : ${depth_sum:.4f}")
print(f"  ever_breached  : {'YES' if ever_breached else 'no'}")

sep()
print(f"  SCORE DECOMPOSITION  (score = P_success - E_breach_depth / spot)")
print()
print(f"  This path contributes to the POPULATION metrics as follows:")
print(f"    P_success_i      = {ps_contrib:.1f}   "
      f"({'NOT breached' if ps_contrib else 'BREACHED -- contributes 0 to P_success'})")
print(f"    breach_depth_i   = depth_sum / breach_days")
print(f"                     = {depth_sum:.4f} / {max(breach_days,1)} = {breach_depth_pp:.4f} per breach-day")
print(f"    Single-path score= {ps_contrib:.1f} - {breach_depth_pp:.4f}/{spot0:.2f}")
print(f"                     = {path_score:.4f}")
print()
print(f"  Population score   = mean(P_success_i) - [sum(depth_sums)/sum(breach_days)] / spot")
print(f"  Optimizer target   : maximise score subject to P_success >= {90:.0f}%")
sep()

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
_results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_results_dir, exist_ok=True)
csv_path = os.path.join(_results_dir, f"trace_path_{TRACE_IDX}.csv")
df.to_csv(csv_path, index=False)
print(f"\nFull day-by-day state saved to {csv_path}")
print(f"Columns: {', '.join(df.columns.tolist())}")
