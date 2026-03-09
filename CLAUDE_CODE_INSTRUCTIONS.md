# MSTR Options Protection — Parametric Policy Search
## Instructions for Claude Code

---

## What You Are Building

A Python implementation of a **parametric policy search** for an options protection program on MSTR (MicroStrategy) stock.

The program holds MSTR stock and wraps it with a rolling options structure: one long put (protection) and up to two short puts (premium generation). The goal is to find the best **policy** — a set of fixed rules expressed as moneyness ratios — that:

1. Keeps portfolio value above a dynamic floor (stock high-water mark × 0.70) at least 90% of the time
2. Remains self-funding (net premium collected ≥ 0) on average
3. Maximizes expected cumulative net premium over the program horizon

The optimization searches over a grid of policy parameters and evaluates each one via **bootstrap simulation** using historical MSTR daily returns.

---

## Project Structure

```
mstr_policy_search/
  main.py                  entry point — runs full pipeline
  data_loader.py           fetch option chain + historical returns via yfinance
  surface.py               frozen volatility surface: F(moneyness, days) → price
  bootstrap.py             bootstrap return sampler
  simulator.py             path simulator: evaluates one policy on N paths
  policy_grid.py           generate feasible policy parameter grid
  optimizer.py             coarse + fine grid search
  results.py               ranking, reporting, output
  requirements.txt
```

---

## Dependencies

```
yfinance>=0.2.40
numpy>=1.26
pandas>=2.0
scipy>=1.12
tqdm>=4.66
tabulate>=0.9
```

---

## Data Inputs — Fetched Automatically via yfinance

### 1. MSTR Option Chain

Fetch the full put option chain for MSTR across all available expiries.

For each contract retain:
```python
{
    "expiry":     str,    # "2025-06-20"
    "days_out":   int,    # calendar days from today
    "strike":     float,  # dollar strike
    "bid":        float,
    "ask":        float,
    "mid":        float,  # (bid + ask) / 2
    "spread_pct": float,  # (ask - bid) / mid * 100
    "iv":         float,  # implied volatility (annualized)
    "oi":         int,    # open interest
    "volume":     int,
}
```

Also record:
```python
spot       = current MSTR last price
fetch_date = today's date (string)
```

Discard any contract where `mid == 0` or both bid and ask are zero.

### 2. MSTR Historical Daily Returns

Fetch maximum available history of MSTR daily closing prices via yfinance.
Compute log returns:

```python
returns = np.log(prices / prices.shift(1)).dropna().values
# returns is a 1D numpy array of daily log returns
```

Store the raw returns array — it is the bootstrap pool.

Print summary statistics on load:
```
MSTR historical returns: N=1842 trading days
  Mean daily:  +0.31%
  Std daily:   4.82%
  Min (worst): −46.2%  (date)
  Max (best):  +24.3%  (date)
  10th pct:    −5.8%
  90th pct:    +6.1%
```

---

## Module Specifications

---

### `surface.py` — Frozen Option Surface

The surface is a lookup function built from the fetched option chain snapshot.

```python
def build_surface(chain_rows: list[dict]) -> callable:
    """
    Returns a function: price(strike, days_out, spot, side='bid'|'ask'|'mid')
    
    Implementation:
    - Store all chain rows in a DataFrame indexed by (strike, days_out)
    - For a query (K, T, S):
        moneyness = K / S
        Find nearest available (strike, days_out) pair by:
          1. Filter rows where days_out is within ±7 days of T
          2. Among those, find strike closest to K
          3. Return requested price (bid, ask, or mid)
    - If no row found within tolerance: return None (caller handles)
    """
```

The surface is "frozen" — it does not change during simulation. All option prices at any future simulated state are read from this snapshot, with moneyness rescaled to the simulated spot.

**Moneyness rescaling:**
When the simulated spot S(τ) differs from the current spot S₀, find the option price by looking up the strike that has the same moneyness ratio relative to the current chain:

```
K_effective = K_target / S(τ) * S₀
```

Then look up `K_effective` in the frozen chain at the appropriate days_out.

This is the frozen surface assumption: the shape of the surface in moneyness space does not change, only the spot moves.

---

### `bootstrap.py` — Return Sampler

```python
class BootstrapSampler:
    def __init__(self, returns: np.ndarray, seed: int = 42):
        self.returns = returns
        self.rng = np.random.default_rng(seed)

    def sample_paths(self, n_paths: int, n_days: int, s0: float) -> np.ndarray:
        """
        Returns: price_paths of shape (n_paths, n_days+1)
        price_paths[:, 0] = s0
        Each day: draw with replacement from self.returns
        price_paths[:, d+1] = price_paths[:, d] * exp(r_d)
        All paths computed simultaneously via numpy vectorization.
        No Python loops over paths.
        """
        drawn = self.rng.choice(self.returns, size=(n_paths, n_days), replace=True)
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(drawn, axis=1)], axis=1
        )
        return s0 * np.exp(log_paths)
```

**Critical:** use numpy vectorization across paths. Never loop over paths in Python.

---

### `policy_grid.py` — Policy Parameter Space

A policy is a named tuple:

```python
from typing import NamedTuple

class Policy(NamedTuple):
    alpha_L:    float   # long put moneyness:   K_L = spot × alpha_L
    alpha_S1:   float   # short put 1 moneyness: K_S1 = spot × alpha_S1
    alpha_S2:   float   # short put 2 moneyness: K_S2 = spot × alpha_S2
                        # set to 0.0 to disable second short leg (N=1 mode)
    T_L:        int     # long put target expiry in days
    T_S1:       int     # short put 1 target expiry in days
    T_S2:       int     # short put 2 target expiry in days (ignored if alpha_S2=0)
    base_q1:    int     # base quantity for short put 1
    base_q2:    int     # base quantity for short put 2 (0 if N=1)
    gamma:      float   # self-funding aggressiveness
    d_min:      int     # calendar roll trigger: days to expiry threshold
```

**Parameter ranges:**

```python
PARAM_GRID = {
    "alpha_L":   [0.90, 0.95, 1.00, 1.05],
    "alpha_S1":  [0.65, 0.70, 0.75, 0.80],
    "alpha_S2":  [0.0, 0.50, 0.55, 0.60, 0.65],   # 0.0 = N=1 mode
    "T_L":       [180, 270, 360],
    "T_S1":      [30, 45, 60],
    "T_S2":      [30, 45, 60, 90],                  # ignored when alpha_S2=0
    "base_q1":   [1, 2],
    "base_q2":   [0, 1, 2],                         # 0 when alpha_S2=0
    "gamma":     [0.0, 0.5, 1.0],
    "d_min":     [7, 14, 21],
}
```

**Structural constraints applied during grid generation (eliminate before simulation):**

```
C_structure:  alpha_S1 < alpha_L
              alpha_S2 < alpha_S1  (when alpha_S2 > 0)
              base_q2 == 0  iff  alpha_S2 == 0

C_liquidity:  For each leg, check that a liquid option exists in the chain:
              snap K = spot × alpha to nearest available strike
              check spread_pct of that strike ≤ 15%
              if no liquid strike found within ±5% of spot × alpha: discard policy

C_self_fund:  At inception (t=t₀):
              premium_received = base_q1 × surface.price(K_S1, T_S1, 'bid')
                               + base_q2 × surface.price(K_S2, T_S2, 'bid')
              premium_paid     = surface.price(K_L, T_L, 'ask')
              net = premium_received - premium_paid
              Discard if net < -eta  (eta = 0 for strict self-funding,
                                      eta = 5.0 for relaxed)
              NOTE: do not discard if net < 0 but > -eta — keep as candidates
              (the rolling program may still be self-funding over time)
```

Print grid size before and after each filter:
```
Raw grid:              124,416 policies
After C_structure:      77,760 policies
After C_liquidity:      ~31,000 policies
After C_self_fund:      ~18,000 policies
```

---

### `simulator.py` — Path Simulator

This is the core engine. It evaluates ONE policy across N simulated paths.

```python
def simulate_policy(
    policy:    Policy,
    paths:     np.ndarray,    # shape (n_paths, n_days+1), pre-generated
    surface:   callable,
    spot0:     float,
    delta:     float = 0.30,  # max drawdown tolerance
    epsilon:   float = 0.10,  # acceptable breach probability
    eta:       float = 0.0,   # max acceptable cumulative debit
) -> dict:
```

**Algorithm — vectorized over paths:**

```
INITIALIZE (all paths simultaneously):
  S[:, 0]     = spot0
  H[:, 0]     = spot0
  W[:]        = inception_cash_flow(policy, surface, spot0)
  theta_age[:] = 0   # days since last roll on each path

FOR each day d = 1 ... n_days:

  S[:, d] already set from pre-generated paths

  Update high-water mark:
    H[:, d] = np.maximum(H[:, d-1], S[:, d])

  Update theta age:
    theta_age += 1

  Compute option portfolio value Π for each path:
    For each leg in policy:
      K_leg  = S[:, d] × alpha_leg      (vector over paths)
      T_rem  = T_leg - theta_age        (days remaining)
      price  = surface.price_vector(K_leg, T_rem, S[:, d])
    Π[:] = q_L × long_put_value - q_S1 × short_put1_value - q_S2 × short_put2_value

  Portfolio value:
    V[:, d] = S[:, d] + Π[:]

  Floor:
    Floor[:, d] = H[:, d] × (1 - delta)

  Breach flag:
    breach[:, d] = (V[:, d] < Floor[:, d]).astype(int)

  CHECK ROLL TRIGGERS (vectorized):
    R_calendar = (T_S1_remaining <= d_min) | (T_S2_remaining <= d_min)
    R_floor    = (K_L_current < Floor[:, d])    # K_L needs updating
    R_liq      = spread_pct_any_leg > 0.15       # simplified: check monthly

    roll_mask = R_calendar | R_floor  (per path boolean mask)

  EXECUTE ROLLS on paths where roll_mask == True:
    For rolling paths:
      Close all legs at bid (shorts) and bid (long put)
      cash_in = q_S1 × surface.bid(K_S1, T_S1_rem) + q_S2 × surface.bid(K_S2, T_S2_rem)
      cash_in += q_L × surface.bid(K_L, T_L_rem)   # close long put

      Recompute new strikes from current S on rolling paths:
        K_L_new   = S[roll_mask, d] × alpha_L    → snap to nearest liquid
        K_S1_new  = S[roll_mask, d] × alpha_S1   → snap to nearest liquid
        K_S2_new  = S[roll_mask, d] × alpha_S2   → snap to nearest liquid

      Recompute adaptive quantities (self-funding feedback):
        deficit   = np.maximum(0, -W[roll_mask])
        P_S1      = surface.price(K_S1_new, T_S1)
        q1_adapt  = base_q1 + gamma × deficit / (P_S1 + 1e-6)
        q1_adapt  = np.floor(q1_adapt).clip(1, q_max_C3)
        (same for q2)

      Open new legs at ask (long) and bid (short):
        cash_out = q_L × surface.ask(K_L_new, T_L)
        cash_in += q1_adapt × surface.bid(K_S1_new, T_S1) + ...

      Net cash flow this roll:
        C_roll = cash_in - cash_out
        W[roll_mask] += C_roll

      Reset theta_age on rolling paths to 0
      Update current strikes on rolling paths

RECORD OUTPUT:
  P_success    = fraction of paths with sum(breach, axis=1) == 0
  E_W          = mean(W_final)
  CVaR_W       = mean of worst epsilon-fraction of W_final values
  E_breach_depth = mean breach depth across all breaches
  n_rolls      = mean number of rolls per path
  P_W_positive = fraction of paths ending with W > 0
```

**Important implementation note on the surface under simulation:**

When simulating, S(τ) moves away from spot0. The frozen surface is indexed by the original chain's strikes. To price an option with target strike K at simulated spot S(τ):

```python
# Rescale: find equivalent moneyness in original chain
moneyness   = K / S_simulated          # e.g., 0.95 (5% OTM)
K_equiv     = moneyness * spot0        # equivalent strike in original chain
price       = chain_lookup(K_equiv, days_remaining, side)
```

This keeps the surface shape fixed while allowing spot to move.

---

### `optimizer.py` — Two-Stage Grid Search

```python
def run_optimization(
    feasible_policies: list[Policy],
    paths_coarse:      np.ndarray,   # (1000, n_days+1)
    paths_fine:        np.ndarray,   # (10000, n_days+1)
    surface:           callable,
    spot0:             float,
    constraints: dict = {
        "min_P_success": 0.90,
        "min_P_success_coarse": 0.80,   # relaxed for coarse pass
        "eta": 0.0,
    }
) -> list[dict]:
```

**Stage 1 — Coarse (1,000 paths):**

```
For each policy in feasible_policies:
  Run simulate_policy(policy, paths_coarse, ...)
  Record: P_success, E_W, CVaR_W

Filter: P_success >= 0.80
Sort by: E_W descending
Keep: top 10% of survivors

Print progress bar (tqdm)
Print:
  Stage 1 complete: 18,000 evaluated → 412 survivors
```

**Stage 2 — Fine (10,000 paths):**

```
For each policy in Stage 1 survivors:
  Run simulate_policy(policy, paths_fine, ...)
  Record: full metric set

Filter: P_success >= 0.90
Sort by composite score (see below)

Print:
  Stage 2 complete: 412 evaluated → 87 valid policies
```

**Composite ranking score:**

```python
score = (
    1.0 × E_W
  - 0.5 × abs(min(CVaR_W, 0))    # penalize negative tail
  - 2.0 × max(0, 0.90 - P_success) × 1000  # hard penalty for constraint violation
)
```

---

### `results.py` — Output and Reporting

Print the following to stdout:

**Top 10 policies table:**

```
Rank  alpha_L  alpha_S1  alpha_S2  T_L  T_S1  T_S2  q1  q2  gamma  d_min  P_success  E[W]   CVaR_W  Score
  1     0.95     0.75      0.60   270    45    90   1   2    0.5     14     93.2%    +12.4   -6.1   11.3
  2     0.95     0.70      0.55   270    45    60   1   2    0.5     14     91.8%    +10.1   -4.8    9.5
  ...
```

**Best policy detail:**

```
═══════════════════════════════════════════════════════
  OPTIMAL POLICY π*
═══════════════════════════════════════════════════════
  Structure:
    Long put:    K = spot × 0.95    (e.g., at spot=$133: K=$126)
    Short put 1: K = spot × 0.75    (e.g., at spot=$133: K=$100)
    Short put 2: K = spot × 0.60    (e.g., at spot=$133: K=$80)

    Long put expiry:    270 days
    Short put 1 expiry:  45 days
    Short put 2 expiry:  90 days

    Base quantities:    q1=1, q2=2
    Self-funding γ:     0.5
    Calendar roll at:   14 days to expiry

  Performance (10,000 paths, 2-year horizon):
    Floor protection:   93.2% of paths never breached
    Expected net W:    +$12.40
    P(W > 0):          71.3%
    CVaR W (worst 10%): −$6.10
    Mean rolls/year:    8.3
    Mean breach depth:  $4.20 (when breached)

  Operating rule at any future roll event τ:
    Given S(τ), H(τ), W(τ):

    1. Compute target strikes:
       K_L   = S(τ) × 0.95  → snap to nearest liquid strike ≤ 15% spread
       K_S1  = S(τ) × 0.75  → snap to nearest liquid strike ≤ 15% spread
       K_S2  = S(τ) × 0.60  → snap to nearest liquid strike ≤ 15% spread

    2. Compute adaptive quantities:
       deficit   = max(0, −W(τ))
       P_S1      = current market ask for K_S1
       P_S2      = current market ask for K_S2
       q1 = floor(1 + 0.5 × deficit / P_S1),  capped by C3 limit
       q2 = floor(2 + 0.5 × deficit / P_S2),  capped by C3 limit

    3. Roll triggers to check daily:
       LIQUIDITY:  any current leg spread% > 15%?  → roll immediately
       FLOOR:      K_L < H(τ) × 0.70?             → roll long put
       CALENDAR:   any short leg ≤ 14 days left?   → roll short legs

    4. Transaction:
       Close all expiring/triggered legs at BID
       Open new legs at ASK
       Record net cash flow → update W(τ)

═══════════════════════════════════════════════════════
```

Save outputs:
- `results/top_policies.csv` — full ranked table
- `results/best_policy.json` — best policy params + metrics
- `results/simulation_summary.txt` — the printed report above
- `results/path_distribution.csv` — W_final distribution for best policy (for plotting)

---

### `main.py` — Entry Point

```python
def main():
    # 1. Load data
    chain, spot, fetch_date = data_loader.load_option_chain("MSTR")
    returns = data_loader.load_historical_returns("MSTR")

    # 2. Build frozen surface
    surface = build_surface(chain)

    # 3. Generate policy grid
    feasible = policy_grid.generate(chain, spot, eta=0.0)

    # 4. Pre-generate bootstrap paths (shared across all policy evaluations)
    sampler = BootstrapSampler(returns)
    horizon_days = 504   # 2 years of trading days
    paths_coarse = sampler.sample_paths(1_000,  horizon_days, spot)
    paths_fine   = sampler.sample_paths(10_000, horizon_days, spot)

    # 5. Run optimization
    results = optimizer.run_optimization(
        feasible_policies = feasible,
        paths_coarse      = paths_coarse,
        paths_fine        = paths_fine,
        surface           = surface,
        spot0             = spot,
    )

    # 6. Report
    results_module.report(results, spot)
```

Run with: `python main.py`

Expected wall time:
- Stage 1 (coarse): 2–5 minutes
- Stage 2 (fine): 5–15 minutes
- Total: under 20 minutes on a modern laptop

---

## Key Implementation Rules

### Vectorization (Critical for Performance)

**Never** loop over paths in Python. All path-level operations must use numpy array operations across the entire path batch simultaneously.

```python
# WRONG — will be 1000x too slow:
for p in range(n_paths):
    H[p, d] = max(H[p, d-1], S[p, d])

# CORRECT:
H[:, d] = np.maximum(H[:, d-1], S[:, d])
```

### Surface Lookup During Simulation

The surface lookup is called millions of times. Cache it as a 2D interpolation grid:

```python
from scipy.interpolate import RegularGridInterpolator

# At build time: create interpolator over (moneyness, days_out)
# At query time: vectorized lookup across all paths simultaneously
prices = interpolator(np.stack([moneyness_array, days_array], axis=1))
```

Where `moneyness_array` has shape `(n_paths,)` — all paths priced simultaneously.

### Snapping to Liquid Strikes

When computing K_new = S × alpha at a roll event, snap to the nearest strike in the chain that passes the liquidity filter:

```python
def snap_strike(target_K, spot, chain_df, max_spread_pct=15.0):
    liquid = chain_df[chain_df['spread_pct'] <= max_spread_pct]['strike'].values
    if len(liquid) == 0:
        return None   # liquidity failure — defer roll
    idx = np.argmin(np.abs(liquid - target_K))
    return liquid[idx]
```

### q_max from C3 (Payoff Non-Negativity)

Before accepting an adaptive quantity q, cap it:

```python
# 10th percentile of simulated spot price (empirical from bootstrap)
q_10pct = np.percentile(returns_empirical_1yr, 10)  # log return
S_10pct = spot * np.exp(q_10pct)   # estimated worst-case spot at horizon

# C3 ceiling:
# V_expiry = S + q_L*(K_L-S)+ - q_S*(K_S-S)+ >= Floor
# Binding case: S = S_10pct, both puts ITM
# Solve: S_10pct + K_L - S_10pct - q*(K_S - S_10pct) >= Floor(t)
# q_max = (K_L - Floor) / (K_S - S_10pct)   when K_S > S_10pct

if K_S > S_10pct:
    q_max = int((K_L - floor_value) / (K_S - S_10pct))
else:
    q_max = 99   # unconstrained: short put OTM at worst case

q_final = min(q_adaptive, q_max)
```

---

## What the Output Means and How To Use It

The result is `π*` — a tuple of 10 numbers. It is not a one-time trade. It is a **standing operating procedure** for the life of the position.

**You use it as follows:**

1. **At inception**: compute strikes from current spot × alpha ratios, check liquidity, enter the structure. Record W₀.

2. **Daily monitoring** (5 minutes/day): check the three roll triggers against current market data. If none fire, do nothing.

3. **At each roll event**: recompute strikes from current spot × alpha ratios, recompute adaptive quantities from current W(τ), execute the roll at bid/ask, update W(τ).

4. **Monthly**: verify the frozen surface assumption still holds (ATM IV within ±15% of original level). If not, re-run the optimizer with updated chain data.

5. **Quarterly or after >30% spot move from t₀**: full re-run of optimizer.

The policy does not tell you to buy or sell MSTR shares — it only governs the options overlay on an existing position.

---

## Testing

Before running the full optimization, run a single-policy sanity check:

```python
# Use a simple known policy
test_policy = Policy(
    alpha_L=0.97, alpha_S1=0.75, alpha_S2=0.0,
    T_L=270, T_S1=45, T_S2=0,
    base_q1=1, base_q2=0,
    gamma=0.0, d_min=14
)
# Run on 100 paths, print full trace for first 3 paths
result = simulate_policy(test_policy, paths[:100], surface, spot0, verbose=True)
```

Expected output for the trace:
```
Path 0:
  Day   0: S=133.53  H=133.53  Floor=93.47  V=133.53+0.00=133.53  OK
  Day   1: S=128.10  H=133.53  Floor=93.47  V=128.10+1.20=129.30  OK
  ...
  Day  45: S=110.00  H=133.53  Floor=93.47  V=110.00+18.40=128.40  ROLL
    → Close: long put bid $16.20, short put bid $1.80
    → Open:  K_L=$105 (ask $19.50), K_S1=$82 (bid $2.10)
    → C(τ) = $16.20 + $1.80 - $19.50 + $2.10 = +$0.60
    → W: 0.00 → +$0.60
  ...
```

---

## Error Handling

- If yfinance returns empty chain for any expiry: skip silently, log warning
- If surface lookup returns None (no liquid strike in range): log "LIQUIDITY FAILURE at day d, path p" and hold current structure for that path on that day
- If W drops below −50 on any path: flag as "self-funding failure" but continue simulation (do not terminate path)
- Wrap entire simulation in try/except, save partial results if interrupted
