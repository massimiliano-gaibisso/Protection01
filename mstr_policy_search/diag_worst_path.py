"""
Diagnostic: simulate the worst-W path and show what drives the loss.
Uses same seed/paths as Phase 3 fine verification (seed=137, N=2000).
"""
import sys, json, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── load modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from data_loader  import load_option_chain, load_historical_returns
BTC_ERA_CUTOFF = "2020-08-11"
from surface      import FrozenSurface
from policy_grid  import Policy
from bootstrap    import BootstrapSampler

# ── config ────────────────────────────────────────────────────────────────────
DELTA_FLOOR     = 0.80
COST_PER_LEG    = 1.0
ANNUAL_DEF_PROB = 0.001
N_PATHS         = 2000
HORIZON         = 504
FINE_SEED       = 137
WORST_PATH      = 1202   # from diag_w.py

# ── load data ─────────────────────────────────────────────────────────────────
chain, spot0_raw, fetch_date = load_option_chain()
spot0   = spot0_raw
surface = FrozenSurface(chain, spot0)
rets    = load_historical_returns(cutoff_date=BTC_ERA_CUTOFF)

sampler = BootstrapSampler(rets, seed=FINE_SEED)
paths   = sampler.sample_paths(N_PATHS, HORIZON, s0=spot0,
                               annual_default_prob=ANNUAL_DEF_PROB)

# ── load best policy ──────────────────────────────────────────────────────────
with open("results/best_policy.json") as f:
    bp = json.load(f)
pd = bp["policy"]
policy = Policy(
    alpha_L     = pd["alpha_L"],
    alpha_S1    = pd["alpha_S1"],
    T_L         = pd["T_L"],
    T_S1        = pd["T_S1"],
    base_q1     = pd["base_q1"],
    beta        = pd["beta"],
    d_min_short = pd["d_min_short"],
    d_min_long  = pd["d_min_long"],
    eta_pct     = pd["eta_pct"],
)

# ── snap T ────────────────────────────────────────────────────────────────────
all_T = sorted(set(int(c["days_out"]) for c in chain))
snap  = lambda t: min(all_T, key=lambda e: abs(e - t))
policy = policy._replace(T_L=snap(policy.T_L), T_S1=snap(policy.T_S1))

# ── single-path simulation of worst path ─────────────────────────────────────
PATH_IDX = WORST_PATH
p = paths[PATH_IDX]           # shape (HORIZON+1,)
S = p

print(f"\n{'='*70}")
print(f"WORST-W PATH TRACE  (path {PATH_IDX}, seed={FINE_SEED})")
print(f"{'='*70}")
print(f"Policy: alpha_L={policy.alpha_L}  alpha_S1={policy.alpha_S1}  "
      f"T_L={policy.T_L}d  T_S1={policy.T_S1}d  q1={policy.base_q1}  "
      f"beta={policy.beta}")
print(f"S0={S[0]:.2f}  S_final={S[-1]:.2f}  S_max={S.max():.2f}")
print(f"Total return: {(S[-1]/S[0]-1)*100:.1f}%   Max DD from peak: "
      f"{(S.min()/S.max()-1)*100:.1f}%")

# Build the path S series stats
returns_path = np.diff(np.log(S))
print(f"Daily log-ret: mean={returns_path.mean()*100:.2f}%  "
      f"std={returns_path.std()*100:.2f}%  "
      f"min={returns_path.min()*100:.1f}%  max={returns_path.max()*100:.1f}%")

# ── reproduce the simulation manually, logging roll events ───────────────────
H          = np.zeros(HORIZON + 1); H[0] = S[0]
W          = surface.price(S[0]*policy.alpha_S1, policy.T_S1, S[0], "bid") * policy.base_q1 \
           - surface.price(S[0]*policy.alpha_L,  policy.T_L,  S[0], "ask")              \
           - (1 + policy.base_q1) * COST_PER_LEG
K_L        = S[0] * policy.alpha_L
K_S1       = S[0] * policy.alpha_S1
theta_s    = 0
theta_l    = 0
dslr       = policy.d_min_long   # days since long roll

roll_log   = []   # list of dicts per roll event
W_series   = [W]

for d in range(1, HORIZON + 1):
    H[d]   = max(H[d-1], S[d])
    theta_s += 1
    theta_l += 1
    dslr    += 1

    T_S1_rem = policy.T_S1 - theta_s
    T_L_rem  = policy.T_L  - theta_l

    floor_ref = (1-policy.beta)*S[0] + policy.beta*H[d]
    Flr       = DELTA_FLOOR * floor_ref

    # option MTM
    lp_v  = max(surface.price(K_L,  max(T_L_rem,1), S[d], "mid") or 0,
                max(K_L-S[d], 0))
    sp_v  = max(surface.price(K_S1, max(T_S1_rem,1), S[d], "mid") or 0,
                max(K_S1-S[d], 0))
    Pi    = lp_v - policy.base_q1 * sp_v
    V     = S[d] + W + Pi

    # short roll
    R_short = T_S1_rem <= policy.d_min_short
    if R_short:
        ask_cl  = max(surface.price(K_S1, max(T_S1_rem,1), S[d], "ask") or 0,
                      max(K_S1-S[d], 0))
        K_S1_n  = S[d] * policy.alpha_S1
        bid_new = surface.price(K_S1_n, policy.T_S1, S[d], "bid") or 0
        C_s     = policy.base_q1*(bid_new - ask_cl) - 2*policy.base_q1*COST_PER_LEG
        roll_log.append(dict(d=d, leg="SHORT", S=S[d], H=H[d], Flr=Flr, W=W,
                             K_old=K_S1, K_new=K_S1_n,
                             bid_close=ask_cl, ask_new=bid_new, C=C_s, W_after=W+C_s))
        W    += C_s
        K_S1  = K_S1_n
        theta_s = 0

    # long roll
    R_long_cal   = T_L_rem <= policy.d_min_long
    R_long_floor = (K_L < Flr) and (S[d]*policy.alpha_L > K_L) and (dslr >= policy.d_min_long)
    if R_long_cal or R_long_floor:
        reason  = "FLOOR" if (R_long_floor and not R_long_cal) else "CAL"
        bid_cl  = max(surface.price(K_L, max(T_L_rem,1), S[d], "bid") or 0,
                      max(K_L-S[d], 0))
        K_L_n   = S[d] * policy.alpha_L
        ask_new = max(surface.price(K_L_n, policy.T_L, S[d], "ask") or 0,
                      max(K_L_n-S[d], 0))
        C_l     = bid_cl - ask_new - 2*COST_PER_LEG
        roll_log.append(dict(d=d, leg=f"LONG({reason})", S=S[d], H=H[d], Flr=Flr, W=W,
                             K_old=K_L, K_new=K_L_n,
                             bid_close=bid_cl, ask_new=ask_new, C=C_l, W_after=W+C_l))
        W    += C_l
        K_L   = K_L_n
        theta_l = 0
        dslr    = 0

    W_series.append(W)

W_final = W_series[-1]
print(f"\n=== Final W: {W_final:+.2f} ===")

# ── print roll log ─────────────────────────────────────────────────────────────
print(f"\n=== Roll events ({len(roll_log)} total) ===")
print(f"{'d':>4} {'leg':>12}  {'S':>8}  {'K_old':>8}  {'K_new':>8}  "
      f"{'bid_cl':>8}  {'ask_new':>8}  {'C':>8}  {'W_after':>9}")
print("-"*95)
for r in roll_log:
    print(f"{r['d']:4d} {r['leg']:>12}  {r['S']:8.2f}  {r['K_old']:8.2f}  {r['K_new']:8.2f}  "
          f"{r['bid_close']:8.3f}  {r['ask_new']:8.3f}  {r['C']:+8.2f}  {r['W_after']:+9.2f}")

# ── summary by leg ─────────────────────────────────────────────────────────────
long_rolls  = [r for r in roll_log if r["leg"].startswith("LONG")]
short_rolls = [r for r in roll_log if r["leg"] == "SHORT"]
long_cost   = sum(r["C"] for r in long_rolls)
short_income= sum(r["C"] for r in short_rolls)
inception_W = W_series[0]

print(f"\n=== P&L Attribution ===")
print(f"  Inception W:        {inception_W:+.2f}")
print(f"  Long roll total:    {long_cost:+.2f}  ({len(long_rolls)} rolls)")
print(f"  Short roll total:   {short_income:+.2f}  ({len(short_rolls)} rolls)")
print(f"  Total W:            {inception_W+long_cost+short_income:+.2f}")
print(f"\n  Avg long roll cost: {long_cost/max(len(long_rolls),1):+.2f}")
print(f"  Avg S at long roll: {np.mean([r['S'] for r in long_rolls]):.2f}")
print(f"  Avg S at short roll:{np.mean([r['S'] for r in short_rolls]):.2f}")

# ── W_series percentiles ───────────────────────────────────────────────────────
ws = np.array(W_series)
print(f"\n=== W running series ===")
for q in [0, 25, 50, 75, 100, 252, 504]:
    if q <= HORIZON:
        print(f"  Day {q:3d}: W = {ws[q]:+.2f}  S = {S[q]:.2f}")
