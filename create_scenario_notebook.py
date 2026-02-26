"""
create_scenario_notebook.py
============================
Generates Scenario_Explorer.ipynb — a step-by-step scenario workbook.

Structure (6 cells):
  Cell 0  : Markdown guide
  Cell 1  : Imports + helper functions
  Cell 2  : PATH  — define the stock/vol trajectory (edit & re-run)
  Cell 3  : QUERY — price any option at any path point  (edit & re-run)
  Cell 4  : TRADES — build a position; auto-fills BS prices  (edit & re-run)
  Cell 5  : P&L — mark-to-market the book at any chosen time  (edit & re-run)

Run once:  python create_scenario_notebook.py
"""

import nbformat as nbf

nb    = nbf.v4.new_notebook()
cells = []

# ---------------------------------------------------------------------------
# Cell 0 — Markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""# Calendar Put Spread — Scenario Explorer

Design a **stock/vol path**, then interactively price options and build trades at each waypoint.

| Cell | What it does | When to re-run |
|------|-------------|----------------|
| **2 — PATH**   | Define time-points with `(t, S, vol)` | Any time you change the path |
| **3 — QUERY**  | Price one or more options at a path point | After editing strikes / expiries |
| **4 — TRADES** | Record entries and exits; auto-fills BS prices | After adding/closing legs |
| **5 — P&L**    | MtM the full book at `EVAL_TIME` | After changing eval time or trades |

**Workflow:**
1. Edit `PATH` in Cell 2, run it.
2. Use Cell 3 to shop for option prices at any point on the path.
3. Enter trades in Cell 4 (leave `fill=None` to auto-fill from BS).
4. Run Cell 5 to see your P&L at any moment.
"""))


# ---------------------------------------------------------------------------
# Cell 1 — Imports + helpers
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"""import sys, os, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore')

import importlib
import BlackScholes
importlib.reload(BlackScholes)
from BlackScholes import BlackScholesCalculator

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.4f}'.format

%matplotlib inline
print("Modules loaded OK")

# -----------------------------------------------------------------------
# Helper : interpolate S and vol at any time t from the PATH list
# -----------------------------------------------------------------------
def path_state(t_query, path):
    '''
    Linear interpolation of (S, vol) at time t_query from a list of
    dicts like [dict(t=0, S=100, vol=0.20), ...].
    Clamps to first/last point outside the defined range.
    '''
    times = [p['t'] for p in path]
    if t_query <= times[0]:
        return float(path[0]['S']), float(path[0]['vol'])
    if t_query >= times[-1]:
        return float(path[-1]['S']), float(path[-1]['vol'])
    for i in range(len(times) - 1):
        if times[i] <= t_query <= times[i + 1]:
            frac = (t_query - times[i]) / (times[i + 1] - times[i])
            S   = path[i]['S']   + frac * (path[i + 1]['S']   - path[i]['S'])
            vol = path[i]['vol'] + frac * (path[i + 1]['vol'] - path[i]['vol'])
            return float(S), float(vol)
    return float(path[-1]['S']), float(path[-1]['vol'])


# -----------------------------------------------------------------------
# Helper : Black-Scholes price + greeks for a single option
# -----------------------------------------------------------------------
def option_info(S, K, T_exp, t_now, vol, opt='put', r=0.05, q=0.00):
    '''
    Returns a dict with Price, Delta, Gamma, Theta_1d, Vega_1pct,
    Rho, PctITM for a European put or call.

    Parameters
    ----------
    S      : spot price at t_now
    K      : strike
    T_exp  : expiry (absolute, years from T0)
    t_now  : current time  (absolute, years from T0)
    vol    : implied vol (annual)
    opt    : 'put' or 'call'
    '''
    T_rem = float(T_exp) - float(t_now)
    if T_rem < 1e-8:          # at or past expiry -> intrinsic
        px = max(K - S, 0.0) if opt == 'put' else max(S - K, 0.0)
        itm = 1.0 if (opt == 'put' and S < K) or (opt == 'call' and S > K) else 0.0
        return dict(Price=px, Delta=itm*(-1 if opt=='put' else 1),
                    Gamma=0.0, Theta_1d=0.0, Vega_1pct=0.0, Rho=0.0, PctITM=itm*100)
    bsc = BlackScholesCalculator(float(S), float(K), T_rem, float(r), float(vol), float(q))
    px  = bsc.put_price() if opt == 'put' else bsc.call_price()
    return dict(
        Price    = px,
        Delta    = bsc.delta(opt),
        Gamma    = bsc.gamma(),
        Theta_1d = bsc.theta(opt) / 365.0,
        Vega_1pct= bsc.vega() / 100.0,
        Rho      = bsc.rho(opt),
        PctITM   = bsc.itm_prob(opt) * 100.0,
    )
"""))


# ---------------------------------------------------------------------------
# Cell 2 — PATH definition
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"""# ==========================================================================
# GLOBAL PARAMETERS
# ==========================================================================
r = 0.05    # risk-free rate (annual, continuous)
q = 0.00    # continuous dividend yield

# ==========================================================================
# PATH DESIGN
# Add, remove or edit waypoints freely.
# t   : time in years from T0 (must be strictly increasing)
# S   : stock price at that waypoint
# vol : implied vol (annualised, e.g. 0.20 = 20%)
# ==========================================================================
PATH = [
    dict(t=0.00, S=100.0, vol=0.20),   # T0      initial conditions
    dict(t=0.50, S=150.0, vol=0.12),   # T_a=0.5 stock +50%, vol -40%
    dict(t=0.75, S=100.0, vol=0.20),   # T_b=0.75 stock back, vol back
    dict(t=1.00, S= 90.0, vol=0.22),   # T1=1.0  long put expiry
    dict(t=2.00, S= 80.0, vol=0.25),   # T2=2.0  short put expiry
]

# --- Display -----------------------------------------------------------------
_df = pd.DataFrame(PATH).set_index('t')
_df.index.name = 'Time (y)'
_s0 = PATH[0]['S'];  _v0 = PATH[0]['vol']
_df['S return %']   = (_df['S']   / _s0 - 1) * 100
_df['vol change %'] = (_df['vol'] / _v0 - 1) * 100
_df.columns = ['Stock S', 'Impl. vol', 'S return %', 'vol chg %']

print("PATH")
display(_df.style
          .format({'Stock S': '{:.1f}', 'Impl. vol': '{:.1%}',
                   'S return %': '{:+.1f}%', 'vol chg %': '{:+.1f}%'})
          .bar(subset=['S return %'], color=['#d65f5f','#5fba7d'])
          .bar(subset=['vol chg %'],  color=['#5fba7d','#d65f5f']))
"""))


# ---------------------------------------------------------------------------
# Cell 3 — QUERY: price inspector
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"""# ==========================================================================
# OPTION PRICE QUERY
# Price any option at any point on the path.
#
# t_now  : current time (years) — looks up S and vol from PATH
# K      : strike
# T_exp  : option expiry (absolute years from T0, must be >= t_now)
# opt    : 'put' or 'call'
# label  : free-text description (for the output table)
# ==========================================================================
QUERY = [
    dict(t_now=0.00, K= 90.0, T_exp=1.00, opt='put', label='Long put  @ T0'),
    dict(t_now=0.00, K= 77.0, T_exp=2.00, opt='put', label='Short put @ T0'),

    dict(t_now=0.50, K= 90.0, T_exp=1.00, opt='put', label='Long put  @ T_a (0.5y)'),
    dict(t_now=0.50, K= 77.0, T_exp=2.00, opt='put', label='Short put @ T_a (0.5y)'),

    dict(t_now=0.75, K= 90.0, T_exp=1.00, opt='put', label='Long put  @ T_b (0.75y)'),
    dict(t_now=0.75, K= 70.0, T_exp=2.00, opt='put', label='New short put K=70 @ T_b'),
]

# --- Compute -----------------------------------------------------------------
_rows = []
for _q in QUERY:
    _S, _vol = path_state(_q['t_now'], PATH)
    _info = option_info(_S, _q['K'], _q['T_exp'], _q['t_now'], _vol, _q['opt'], r, q)
    _rows.append({
        'Label':       _q.get('label', ''),
        't_now':       _q['t_now'],
        'S':           _S,
        'Vol':         _vol,
        'Strike K':    _q['K'],
        'T_exp':       _q['T_exp'],
        'T_rem (y)':   round(_q['T_exp'] - _q['t_now'], 4),
        'Type':        _q['opt'],
        'Price':       _info['Price'],
        'Delta':       _info['Delta'],
        'Gamma':       _info['Gamma'],
        'Theta 1d':    _info['Theta_1d'],
        'Vega 1%':     _info['Vega_1pct'],
        'P(ITM) %':    _info['PctITM'],
    })

df_query = pd.DataFrame(_rows).set_index('Label')
print("OPTION PRICES")
display(df_query)
"""))


# ---------------------------------------------------------------------------
# Cell 4 — TRADES: position builder
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"""# ==========================================================================
# TRADE BOOK
# Record every entry and exit.
#
# t      : time of the trade (years)
# opt    : 'put' or 'call'
# K      : strike
# T_exp  : option expiry (absolute years)
# qty    : +N = long (buy), -N = short (sell)
# fill   : execution price; set to None to auto-fill from BS at t
# label  : description
# ==========================================================================
TRADES = [
    # --- Initial structure at T0 -------------------------------------------
    dict(t=0.00, opt='put', K=90.0, T_exp=1.0, qty=+1, fill=None,
         label='A: Long put K=90 T=1y'),
    dict(t=0.00, opt='put', K=77.0, T_exp=2.0, qty=-2, fill=None,
         label='B: Short 2x put K=77 T=2y'),

    # --- Phase A (T_a=0.5y): stock +50%, vol -40% → buy back short puts ----
    # Uncomment to record the closing trade:
    # dict(t=0.50, opt='put', K=77.0, T_exp=2.0, qty=+2, fill=None,
    #      label='B-close: Buy back 2x short put @ T_a'),

    # --- Phase B (T_b=0.75y): re-sell at lower strike ----------------------
    # Uncomment to record the new short puts:
    # dict(t=0.75, opt='put', K=70.0, T_exp=2.0, qty=-2, fill=None,
    #      label='C: Re-sell 2x put K=70 T=2y @ T_b'),
]

# --- Auto-fill missing prices from BS ------------------------------------
for _t in TRADES:
    if _t['fill'] is None:
        _S, _vol = path_state(_t['t'], PATH)
        _t['fill'] = option_info(_S, _t['K'], _t['T_exp'], _t['t'],
                                  _vol, _t['opt'], r, q)['Price']

# --- Display position ----------------------------------------------------
_book = []
for _t in TRADES:
    _book.append({
        'Label':    _t['label'],
        't_trade':  _t['t'],
        'Type':     _t['opt'],
        'K':        _t['K'],
        'T_exp':    _t['T_exp'],
        'Qty':      _t['qty'],
        'Fill $':   _t['fill'],
        'Cash flow': round(-_t['qty'] * _t['fill'], 6),  # +received / -paid
    })

df_book = pd.DataFrame(_book).set_index('Label')
_net = df_book['Cash flow'].sum()

print("TRADE BOOK")
display(df_book)
print(f"\\n  Net cash flow at inception  (+ received / - paid): {_net:+.4f}")

# --- Active positions summary (net qty per instrument) -------------------
from collections import defaultdict
_pos = defaultdict(float)
for _t in TRADES:
    _key = f"{_t['opt'].upper()} K={_t['K']:.1f} T_exp={_t['T_exp']:.1f}y"
    _pos[_key] += _t['qty']

_net_pos = {k: v for k, v in _pos.items() if abs(v) > 1e-9}
if _net_pos:
    print("\\n  Net open positions:")
    for _k, _v in _net_pos.items():
        print(f"    {_v:+.0f}  {_k}")
"""))


# ---------------------------------------------------------------------------
# Cell 5 — P&L: mark-to-market
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"""# ==========================================================================
# MARK-TO-MARKET P&L
# Set EVAL_TIME to any point along the path.
# The book must have been built in Cell 4 first.
# ==========================================================================
EVAL_TIME = 0.50    # <-- change me

# -------------------------------------------------------------------------
_S_eval, _vol_eval = path_state(EVAL_TIME, PATH)

_pnl_rows = []
for _t in TRADES:
    _info = option_info(_S_eval, _t['K'], _t['T_exp'], EVAL_TIME,
                        _vol_eval, _t['opt'], r, q)
    _mtm    = _info['Price']
    _pnl_u  = _mtm - _t['fill']          # per unit, from buyer's perspective
    _pnl    = _t['qty'] * _pnl_u         # signed by qty
    _pnl_rows.append({
        'Label':         _t['label'],
        'Qty':           _t['qty'],
        'Fill $':        _t['fill'],
        'MtM $':         _mtm,
        'P&L / unit':    _pnl_u,
        'Total P&L':     _pnl,
        'Delta (net)':   _t['qty'] * _info['Delta'],
        'P(ITM) %':      _info['PctITM'],
    })

df_pnl  = pd.DataFrame(_pnl_rows).set_index('Label')
_ttl    = df_pnl['Total P&L'].sum()
_tdelta = df_pnl['Delta (net)'].sum()

print(f"P&L  AT  t = {EVAL_TIME:.3f}y   |   S = {_S_eval:.2f}   |   Vol = {_vol_eval*100:.1f}%")
print('-' * 62)
display(df_pnl.style
          .format({c: '{:+.4f}' for c in ['P&L / unit','Total P&L','Delta (net)']})
          .bar(subset=['Total P&L'], color=['#d65f5f','#5fba7d'], align='zero'))
print(f"\\n  Total P&L   : {_ttl:+.4f}")
print(f"  Net delta   : {_tdelta:+.4f}")
"""))


# ---------------------------------------------------------------------------
# Assemble and write
# ---------------------------------------------------------------------------
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.8.0"
    }
}

out_path = "Scenario_Explorer.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created {out_path}")
