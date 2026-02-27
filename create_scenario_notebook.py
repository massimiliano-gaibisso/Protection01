"""
create_scenario_notebook.py
============================
Generates Scenario_Explorer.ipynb

New structure (4 cells):
  Cell 0  : Markdown guide
  Cell 1  : Imports + math helpers  (path_state, option_info)
  Cell 2  : Display functions       (fill_trades, show_path, show_query,
                                     show_trades, show_positions_evolution)
  Cell 3  : CONFIG ONLY  (last cell — the only cell the user ever edits)
             r, q
             PATH    → show_path(PATH)
             QUERY   → show_query(QUERY, PATH, r, q)
             TRADES  → show_trades(TRADES, PATH, r, q)
                     → show_positions_evolution(TRADES, PATH, r, q)

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

Design a **stock/vol path**, then price options and manage trades at each waypoint.
**You only need to edit the last cell (CONFIG).**

| Cell | Role | Edit? |
|------|------|-------|
| 1 — Imports & math helpers | `path_state`, `option_info` | No |
| 2 — Display functions | `show_path`, `show_query`, `show_trades`, `show_positions_evolution` | No |
| **3 — CONFIG** | `PATH`, `QUERY`, `TRADES` + display calls | **Yes — only this cell** |

**Workflow — inside the CONFIG cell:**
1. Design your `PATH` (time-points with S and vol) → `show_path(PATH)` draws the table.
2. Use `QUERY` to price options at any path point → `show_query(...)` shows price + greeks.
3. Record trades in `TRADES` (leave `fill=None` for auto BS fill) → `show_trades(...)` shows the book.
4. `show_positions_evolution(...)` snapshots the portfolio at **every trade time**: cash, MtM, P&L.
"""))


# ---------------------------------------------------------------------------
# Cell 1 — Imports + math helpers
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

%matplotlib inline
print("Cell 1 — helpers loaded")


# -----------------------------------------------------------------------
# path_state : linear interpolation of (S, vol) along the PATH
# -----------------------------------------------------------------------
def path_state(t_query, path):
    '''Return (S, vol) at t_query by linear interpolation. Clamps at ends.'''
    times = [p['t'] for p in path]
    if t_query <= times[0]:
        return float(path[0]['S']), float(path[0]['vol'])
    if t_query >= times[-1]:
        return float(path[-1]['S']), float(path[-1]['vol'])
    for i in range(len(times) - 1):
        if times[i] <= t_query <= times[i + 1]:
            frac = (t_query - times[i]) / (times[i + 1] - times[i])
            S   = path[i]['S']   + frac * (path[i+1]['S']   - path[i]['S'])
            vol = path[i]['vol'] + frac * (path[i+1]['vol'] - path[i]['vol'])
            return float(S), float(vol)
    return float(path[-1]['S']), float(path[-1]['vol'])


# -----------------------------------------------------------------------
# option_info : BS price + full greeks + P(ITM) for one option
# -----------------------------------------------------------------------
def option_info(S, K, T_exp, t_now, vol, opt='put', r=0.05, q=0.00):
    '''
    Returns dict: Price, Delta, Gamma, Theta_1d, Vega_1pct, Rho, PctITM.
    T_exp and t_now are absolute times (years from T0).
    At/past expiry returns intrinsic value.
    '''
    T_rem = float(T_exp) - float(t_now)
    if T_rem < 1e-8:
        px  = max(K - S, 0.0) if opt == 'put' else max(S - K, 0.0)
        itm = 1.0 if (opt=='put' and S < K) or (opt=='call' and S > K) else 0.0
        return dict(Price=px,
                    Delta=itm * (-1 if opt == 'put' else 1),
                    Gamma=0.0, Theta_1d=0.0, Vega_1pct=0.0, Rho=0.0,
                    PctITM=itm * 100.0)
    bsc = BlackScholesCalculator(float(S), float(K), T_rem,
                                  float(r), float(vol), float(q))
    px  = bsc.put_price() if opt == 'put' else bsc.call_price()
    return dict(
        Price     = px,
        Delta     = bsc.delta(opt),
        Gamma     = bsc.gamma(),
        Theta_1d  = bsc.theta(opt) / 365.0,
        Vega_1pct = bsc.vega()  / 100.0,
        Rho       = bsc.rho(opt),
        PctITM    = bsc.itm_prob(opt) * 100.0,
    )
"""))


# ---------------------------------------------------------------------------
# Cell 2 — Display functions  (no config here)
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"""# =========================================================================
# DISPLAY FUNCTIONS  — do not edit; all configuration is in the last cell
# =========================================================================
from collections import defaultdict as _ddict
pd.options.display.float_format = '{:.4f}'.format


# -----------------------------------------------------------------------
# fill_trades : auto-price any trade whose fill is None  (idempotent)
# -----------------------------------------------------------------------
def fill_trades(trades, path, r=0.05, q=0.00):
    '''Fill None prices in-place from BS at each trade's path point.'''
    for t in trades:
        if t.get('fill') is None:
            S, vol = path_state(t['t'], path)
            t['fill'] = option_info(S, t['K'], t['T_exp'], t['t'],
                                     vol, t['opt'], r, q)['Price']
    return trades


# -----------------------------------------------------------------------
# show_path
# -----------------------------------------------------------------------
def show_path(path):
    '''Styled DataFrame: time-points with S, vol, cumulative return and vol change.'''
    df = pd.DataFrame(path).set_index('t')
    df.index.name = 'Time (y)'
    s0 = path[0]['S']; v0 = path[0]['vol']
    df['S ret %']  = (df['S']   / s0 - 1) * 100
    df['vol chg %']= (df['vol'] / v0 - 1) * 100
    df.columns = ['Stock S', 'Impl. vol', 'S ret %', 'vol chg %']
    print('PATH')
    display(df.style
              .format({'Stock S':   '{:.2f}',
                       'Impl. vol': '{:.1%}',
                       'S ret %':   '{:+.1f}%',
                       'vol chg %': '{:+.1f}%'})
              .bar(subset=['S ret %'],   color=['#d65f5f', '#5fba7d'])
              .bar(subset=['vol chg %'], color=['#5fba7d', '#d65f5f']))
    print()


# -----------------------------------------------------------------------
# show_query
# -----------------------------------------------------------------------
def show_query(query, path, r=0.05, q=0.00):
    '''Price a list of option queries; display price + full greeks table.'''
    rows = []
    for qr in query:
        S, vol = path_state(qr['t_now'], path)
        info   = option_info(S, qr['K'], qr['T_exp'], qr['t_now'],
                              vol, qr['opt'], r, q)
        rows.append({
            'Label':     qr.get('label', ''),
            't_now':     qr['t_now'],
            'S':         round(S, 2),
            'Vol':       vol,
            'K':         qr['K'],
            'T_exp':     qr['T_exp'],
            'T_rem':     round(qr['T_exp'] - qr['t_now'], 3),
            'Type':      qr['opt'],
            'Price':     info['Price'],
            'Delta':     info['Delta'],
            'Gamma':     info['Gamma'],
            'Theta 1d':  info['Theta_1d'],
            'Vega 1%':   info['Vega_1pct'],
            'P(ITM) %':  info['PctITM'],
        })
    df = pd.DataFrame(rows).set_index('Label')
    print('OPTION PRICES')
    display(df)
    print()


# -----------------------------------------------------------------------
# show_trades
# -----------------------------------------------------------------------
def show_trades(trades, path, r=0.05, q=0.00):
    '''Fill missing prices in-place then display the trade book with cash flows.'''
    fill_trades(trades, path, r, q)
    rows = []
    for t in trades:
        rows.append({
            'Label':     t.get('label', ''),
            't_trade':   t['t'],
            'Type':      t['opt'],
            'K':         t['K'],
            'T_exp':     t['T_exp'],
            'Qty':       t['qty'],
            'Fill $':    t['fill'],
            'Cash flow': -t['qty'] * t['fill'],   # +received / -paid
        })
    df  = pd.DataFrame(rows).set_index('Label')
    net = df['Cash flow'].sum()
    print('TRADE BOOK')
    display(df)
    print(f'  Net inception cash  (+ received / - paid) : {net:+.4f}')
    # Net open quantities per instrument
    pos = _ddict(float)
    for t in trades:
        pos[f"{t['opt'].upper()} K={t['K']:.1f} T={t['T_exp']:.1f}y"] += t['qty']
    open_pos = {k: v for k, v in pos.items() if abs(v) > 1e-9}
    if open_pos:
        print('  Net open positions:')
        for k, v in open_pos.items():
            print(f'    {v:+.0f}  {k}')
    print()


# -----------------------------------------------------------------------
# show_positions_evolution
# -----------------------------------------------------------------------
def show_positions_evolution(trades, path, r=0.05, q=0.00):
    '''
    At each distinct trade time in TRADES, print a full portfolio snapshot:
      - which new trades executed at that time
      - every active leg with current MtM, P&L, delta, P(ITM)
      - summary: current cash | open position MtM | total portfolio value | total P&L
    '''
    fill_trades(trades, path, r, q)
    sep = '=' * 74
    mid = '-' * 74

    trade_times = sorted(set(t['t'] for t in trades))
    cum_cash    = 0.0

    for t_now in trade_times:
        S, vol = path_state(t_now, path)

        # ---- cash from trades executed at this time -----------------------
        new_trades   = [t for t in trades if t['t'] == t_now]
        cash_step    = sum(-t['qty'] * t['fill'] for t in new_trades)
        cum_cash    += cash_step

        # ---- all trades entered up to (and including) t_now ---------------
        active = [t for t in trades if t['t'] <= t_now]

        rows       = []
        pos_mtm    = 0.0
        total_pnl  = 0.0

        for t in active:
            is_open  = t['T_exp'] > t_now
            info     = option_info(S, t['K'], t['T_exp'], t_now,
                                    vol, t['opt'], r, q)
            curr     = info['Price']
            pnl_u    = curr - t['fill']           # per unit (buyer's view)
            pnl_pos  = t['qty'] * pnl_u           # signed by quantity

            total_pnl += pnl_pos
            if is_open:
                pos_mtm += t['qty'] * curr

            rows.append({
                'Leg':       t.get('label', ''),
                't_trade':   t['t'],
                'Qty':       t['qty'],
                'Fill $':    round(t['fill'], 4),
                'MtM $':     round(curr, 4)  if is_open else 'SETTLED',
                'P&L/unit':  round(pnl_u, 4),
                'Pos. P&L':  round(pnl_pos, 4),
                'Delta':     round(t['qty'] * info['Delta'], 4) if is_open else 0.0,
                'P(ITM) %':  round(info['PctITM'], 1) if is_open else '--',
                'Status':    'OPEN'    if is_open else 'SETTLED',
            })

        df = pd.DataFrame(rows).set_index('Leg')

        # ---- header -------------------------------------------------------
        print(sep)
        print(f'  SNAPSHOT   t = {t_now:.3f}y  |  S = {S:.2f}  |  Vol = {vol*100:.1f}%')
        new_labels = ',  '.join(t.get('label', '?') for t in new_trades)
        print(f'  New trades : {new_labels}')
        print(mid)

        # ---- position table -----------------------------------------------
        display(df)

        # ---- summary footer -----------------------------------------------
        print()
        print(f'  Current cash  (all executed trades)   : {cum_cash:+.4f}')
        print(f'  Open position MtM                     : {pos_mtm:+.4f}')
        print(f'  Total portfolio value  (cash + MtM)   : {cum_cash + pos_mtm:+.4f}')
        print(f'  Total P&L                             : {total_pnl:+.4f}')
        print()

    print(sep)
    print()


print("Cell 2 — display functions loaded")
"""))


# ---------------------------------------------------------------------------
# Cell 3 — CONFIG (last cell — the only one the user edits)
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"""# =========================================================================
# CONFIG  —  edit only this cell
# =========================================================================

r = 0.05    # risk-free rate (annual, continuous)
q = 0.00    # continuous dividend yield

# =========================================================================
# PATH
# t   : time in years from T0  (strictly increasing)
# S   : stock price at that waypoint
# vol : implied vol annual  (e.g. 0.20 = 20%)
# =========================================================================
PATH = [
    dict(t=0.00, S=100.0, vol=0.20),   # T0      initial conditions
    dict(t=0.50, S=150.0, vol=0.12),   # T_a=0.5 stock +50%, vol -40%
    dict(t=0.75, S=100.0, vol=0.20),   # T_b=0.75 stock back, vol back
    dict(t=1.00, S= 90.0, vol=0.22),   # T1=1.0  long put expiry
    dict(t=2.00, S= 80.0, vol=0.25),   # T2=2.0  short put expiry
]
show_path(PATH)

# =========================================================================
# QUERY
# t_now : current time (years) — S and vol looked up from PATH
# K     : strike
# T_exp : option expiry (absolute years from T0, >= t_now)
# opt   : 'put' or 'call'
# label : free text
# =========================================================================
QUERY = [
    dict(t_now=0.00, K= 90.0, T_exp=1.00, opt='put', label='Long put  @ T0'),
    dict(t_now=0.00, K= 77.0, T_exp=2.00, opt='put', label='Short put @ T0'),

    dict(t_now=0.50, K= 90.0, T_exp=1.00, opt='put', label='Long put  @ T_a (0.5y)'),
    dict(t_now=0.50, K= 77.0, T_exp=2.00, opt='put', label='Short put @ T_a (0.5y)'),

    dict(t_now=0.75, K= 90.0, T_exp=1.00, opt='put', label='Long put  @ T_b (0.75y)'),
    dict(t_now=0.75, K= 70.0, T_exp=2.00, opt='put', label='New short put K=70 @ T_b'),
]
show_query(QUERY, PATH, r, q)

# =========================================================================
# TRADES
# t      : trade time (years)
# opt    : 'put' or 'call'
# K      : strike
# T_exp  : option expiry (absolute years)
# qty    : +N long  /  -N short
# fill   : execution price;  None = auto-fill from BS at trade time
# label  : description shown in output
# =========================================================================
TRADES = [
    # --- Initial structure at T0 -------------------------------------------
    dict(t=0.00, opt='put', K=90.0, T_exp=1.0, qty=+1, fill=None,
         label='A: Long put K=90 T=1y'),
    dict(t=0.00, opt='put', K=77.0, T_exp=2.0, qty=-2, fill=None,
         label='B: Short 2x put K=77 T=2y'),

    # --- Phase A (T_a=0.5y): stock +50%, vol -40% -> buy back short puts ---
    # Uncomment to record the closing trade:
    # dict(t=0.50, opt='put', K=77.0, T_exp=2.0, qty=+2, fill=None,
    #      label='B-close: Buy back 2x short put @ T_a'),

    # --- Phase B (T_b=0.75y): re-sell at lower strike ----------------------
    # Uncomment to record the new short puts:
    # dict(t=0.75, opt='put', K=70.0, T_exp=2.0, qty=-2, fill=None,
    #      label='C: Re-sell 2x put K=70 T=2y @ T_b'),
]
show_trades(TRADES, PATH, r, q)
show_positions_evolution(TRADES, PATH, r, q)
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
