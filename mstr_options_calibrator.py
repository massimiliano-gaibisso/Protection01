#!/usr/bin/env python3
"""
MSTR Diagonal Put Spread — Liquidity Calibrator
================================================
Run:  pip install yfinance && python mstr_options_calibrator.py
Output: mstr_options_calibrator.html  (open in any browser)
"""

import json, math, sys
from datetime import datetime, date

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not found — run: pip install yfinance")

# ─── 1. FETCH DATA ────────────────────────────────────────────────────────────
SYMBOL = "MSTR"
print(f"Fetching {SYMBOL} option chain …")

tk   = yf.Ticker(SYMBOL)
info = tk.fast_info
spot = round(info.last_price, 2)
print(f"  Spot price : ${spot}")

all_exps = tk.options
print(f"  Expiries   : {len(all_exps)} series")

today = date.today()
rows  = []

for exp_str in all_exps:
    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
    days_out = (exp_date - today).days
    if days_out < 5:
        continue                     # skip already-expired / same-week
    chain = tk.option_chain(exp_str)
    puts  = chain.puts

    for _, r in puts.iterrows():
        strike = float(r["strike"])
        bid    = float(r["bid"])   if r["bid"]   == r["bid"] else 0.0
        ask    = float(r["ask"])   if r["ask"]   == r["ask"] else bid * 1.2
        oi     = int(r["openInterest"]) if r["openInterest"] == r["openInterest"] else 0
        vol    = int(r["volume"])       if r["volume"]       == r["volume"]       else 0
        iv     = float(r["impliedVolatility"]) if r["impliedVolatility"] == r["impliedVolatility"] else 0.0

        mid = (bid + ask) / 2 if bid + ask > 0 else 0.0
        if mid == 0:
            continue
        spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999

        rows.append({
            "expiry"    : exp_str,
            "daysOut"   : days_out,
            "strike"    : strike,
            "bid"       : round(bid, 2),
            "ask"       : round(ask, 2),
            "mid"       : round(mid, 2),
            "oi"        : oi,
            "vol"       : vol,
            "iv"        : round(iv, 4),
            "spreadPct" : round(spread_pct, 1),
        })

print(f"  Contracts  : {len(rows)} put options loaded")

# ─── 2. COMPUTE LIQUIDITY SCORES ─────────────────────────────────────────────
max_oi  = max((r["oi"]  for r in rows), default=1) or 1
max_vol = max((r["vol"] for r in rows), default=1) or 1

for r in rows:
    sp  = r["spreadPct"]
    spread_score = max(0, 100 - sp * 1.5)
    oi_score     = min(100, (r["oi"]  / max_oi)  * 100)
    vol_score    = min(100, (r["vol"] / max_vol)  * 100)
    r["liqScore"] = round(spread_score * 0.50 + oi_score * 0.30 + vol_score * 0.20)

# ─── 3. BUILD DIAGONAL SPREAD CANDIDATES ─────────────────────────────────────
# Index rows by (expiry, strike) for fast lookup
idx = {(r["expiry"], r["strike"]): r for r in rows}

candidates = []
expiries = sorted(set(r["expiry"] for r in rows))

for i, lexp in enumerate(expiries):
    for j, sexp in enumerate(expiries):
        if j <= i:
            continue
        l_days = None
        s_days = None
        for r in rows:
            if r["expiry"] == lexp: l_days = r["daysOut"]; break
        for r in rows:
            if r["expiry"] == sexp: s_days = r["daysOut"]; break
        if l_days is None or s_days is None: continue
        time_diff = s_days - l_days
        if time_diff < 30 or time_diff > 180:
            continue

        # All valid strike pairs
        l_strikes = sorted(set(r["strike"] for r in rows if r["expiry"] == lexp))
        s_strikes = sorted(set(r["strike"] for r in rows if r["expiry"] == sexp))

        for lk in l_strikes:
            for sk in s_strikes:
                if sk >= lk: continue
                width = lk - sk
                if width < spot * 0.05 or width > spot * 0.25:
                    continue          # 5%–25% of spot
                lp = idx.get((lexp, lk))
                sp2 = idx.get((sexp, sk))
                if not lp or not sp2: continue
                if lp["liqScore"] < 30 or sp2["liqScore"] < 30: continue

                net_cost   = lp["mid"] - sp2["mid"]
                if net_cost <= 0: continue
                cost_ratio = sp2["mid"] / lp["mid"]

                # Composite score
                score = round(
                    lp["liqScore"]  * 0.30 +
                    sp2["liqScore"] * 0.30 +
                    min(100, cost_ratio * 120) * 0.25 +
                    min(100, (width / spot) * 400) * 0.15
                )

                candidates.append({
                    "longExp"  : lexp,
                    "shortExp" : sexp,
                    "longK"    : lk,
                    "shortK"   : sk,
                    "longMid"  : lp["mid"],
                    "shortMid" : sp2["mid"],
                    "netCost"  : round(net_cost, 2),
                    "costPct"  : round(cost_ratio * 100, 1),
                    "width"    : round(width, 1),
                    "timeDiff" : time_diff,
                    "liqL"     : lp["liqScore"],
                    "liqS"     : sp2["liqScore"],
                    "score"    : score,
                    "longOTM"  : round((spot - lk) / spot * 100, 1),
                    "shortOTM" : round((spot - sk) / spot * 100, 1),
                })

candidates.sort(key=lambda x: -x["score"])
top = candidates[:15]
print(f"  Candidates : {len(candidates)} spreads found, showing top {len(top)}")

# Unique expiries present in data
exp_labels = sorted(set(r["expiry"] for r in rows))
strikes_present = sorted(set(r["strike"] for r in rows))

# ─── 4. SERIALIZE ─────────────────────────────────────────────────────────────
payload = json.dumps({
    "symbol"    : SYMBOL,
    "spot"      : spot,
    "asOf"      : today.isoformat(),
    "rows"      : rows,
    "candidates": top,
    "expiries"  : exp_labels,
    "strikes"   : strikes_present,
}, indent=None)

# ─── 5. HTML TEMPLATE ─────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MSTR · Options Liquidity Calibrator</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#06060f;color:#c8cad8;font-family:'IBM Plex Mono',Courier,monospace;font-size:13px}
  ::-webkit-scrollbar{width:6px;height:6px}
  ::-webkit-scrollbar-track{background:#0d0d1f}
  ::-webkit-scrollbar-thumb{background:#2a2a4a;border-radius:3px}

  /* ── layout ── */
  #header{padding:18px 28px;border-bottom:1px solid #1a1a2e;display:flex;align-items:center;justify-content:space-between;background:linear-gradient(90deg,#06060f 60%,#0d0d1f)}
  #header .title{font-size:19px;color:#e8e8ff;font-weight:700;letter-spacing:1px}
  #header .sub{font-size:10px;color:#4a4a6a;letter-spacing:4px;margin-bottom:4px}
  #header .meta{display:flex;gap:28px;font-size:11px}
  #header .meta .val{color:#00e87a;font-size:17px;font-weight:700}
  #header .meta .lbl{color:#4a4a6a;font-size:9px;letter-spacing:1px}

  /* ── tabs ── */
  #tabs{display:flex;border-bottom:1px solid #1a1a2e;padding:0 28px}
  .tab{background:none;border:none;border-bottom:2px solid transparent;padding:13px 18px;font-family:inherit;font-size:10px;letter-spacing:2px;cursor:pointer;color:#4a4a6a;transition:color .15s}
  .tab.active{color:#00e87a;border-bottom-color:#00e87a}

  /* ── content ── */
  #content{padding:22px 28px}
  .panel{display:none}
  .panel.active{display:block}

  /* ── heatmap ── */
  #hm-wrap{display:flex;gap:20px}
  #hm-table-wrap{flex:1;overflow-x:auto}
  #hm-side{width:210px;flex-shrink:0}
  table{border-collapse:collapse;width:100%}
  th{padding:7px 9px;font-size:9px;color:#4a4a6a;letter-spacing:1px;font-weight:normal;white-space:nowrap}
  td{padding:5px 7px;white-space:nowrap}
  .hm-cell{text-align:center;cursor:pointer;transition:background .1s;min-width:80px}
  .hm-cell:hover{border:1px solid #3a3a5a !important}
  .hm-val{font-size:11px;font-weight:600}
  .hm-sub{font-size:8px;color:#3a3a5a;margin-top:1px}
  .strike-label{font-size:11px;border-right:1px solid #1a1a2e;padding-right:10px}

  /* ── metric buttons ── */
  .metric-bar{display:flex;gap:6px;margin-bottom:16px;align-items:center}
  .metric-btn{background:none;border:1px solid #1a1a2e;border-radius:3px;padding:4px 11px;font-family:inherit;font-size:9px;letter-spacing:1px;cursor:pointer;color:#4a4a6a;transition:all .15s}
  .metric-btn.active{border-color:#00e87a;color:#00e87a;background:#1a1a2e}

  /* ── side cards ── */
  .card{background:#0d0d1f;border:1px solid #1a1a2e;border-radius:6px;padding:13px;margin-bottom:12px}
  .card-title{font-size:9px;color:#4a4a6a;letter-spacing:2px;margin-bottom:9px}
  .kv{display:flex;justify-content:space-between;margin-bottom:4px}
  .kv .k{font-size:9px;color:#4a4a6a}
  .kv .v{font-size:9px;color:#c8cad8}
  .legend-row{display:flex;align-items:center;gap:7px;margin-bottom:6px}
  .dot{width:9px;height:9px;border-radius:2px;flex-shrink:0}

  /* ── chain tab ── */
  .chain-table th,.chain-table td{text-align:right;padding:6px 9px}
  .chain-table th:first-child,.chain-table td:first-child{text-align:left}
  .chain-table tbody tr:nth-child(even){background:#08081a}
  .chain-table tbody tr:hover{background:#0f0f22}
  .badge{font-size:8px;padding:2px 6px;border-radius:2px;letter-spacing:1px}

  /* ── candidates tab ── */
  .top-card{background:linear-gradient(135deg,#0d1f14,#0a0a1f);border:1px solid #00e87a44;border-radius:8px;padding:18px;margin-bottom:20px}
  .top-card .big{font-size:22px;font-weight:700}
  .top-card-grid{display:flex;gap:28px;flex-wrap:wrap}
  .cand-table th,.cand-table td{text-align:right;padding:7px 9px}
  .cand-table tbody tr:nth-child(even){background:#08081a}
  .cand-table tbody tr:first-child{background:rgba(0,232,122,.04)}
  .bar-wrap{display:flex;align-items:center;gap:6px;justify-content:flex-end}
  .bar-bg{width:38px;height:4px;background:#1a1a2e;border-radius:2px}
  .bar-fg{height:100%;border-radius:2px}

  /* ── tooltip ── */
  #tooltip{position:fixed;pointer-events:none;background:#0d0d1f;border:1px solid #3a3a5a;border-radius:5px;padding:10px 13px;font-size:10px;display:none;z-index:999}
  #tooltip.show{display:block}
  #tooltip .tk{color:#4a4a6a;margin-right:8px}
  #tooltip .tv{color:#e8e8ff}

  /* ── risk flags ── */
  .flag-row{display:flex;justify-content:space-between;padding:3px 7px;border-radius:3px;background:rgba(255,59,92,.06);margin-bottom:4px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:20px}
</style>
</head>
<body>

<div id="header">
  <div>
    <div class="sub">DIAGONAL PUT SPREAD</div>
    <div class="title">MSTR · OPTIONS LIQUIDITY CALIBRATOR</div>
  </div>
  <div class="meta" id="meta-bar">
    <div><div class="lbl">SYMBOL</div><div class="val" id="m-sym">—</div></div>
    <div><div class="lbl">SPOT PRICE</div><div class="val" id="m-spot">—</div></div>
    <div><div class="lbl">AS OF</div><div class="val" style="font-size:12px" id="m-date">—</div></div>
    <div><div class="lbl">CONTRACTS</div><div class="val" style="font-size:12px" id="m-count">—</div></div>
    <div><div class="lbl">EXPIRIES</div><div class="val" style="font-size:12px" id="m-exps">—</div></div>
  </div>
</div>

<div id="tabs">
  <button class="tab active" onclick="showTab('heatmap',this)">LIQUIDITY MAP</button>
  <button class="tab" onclick="showTab('chain',this)">CHAIN DETAIL</button>
  <button class="tab" onclick="showTab('candidates',this)">SPREAD CANDIDATES</button>
</div>

<div id="content">

  <!-- ══ HEATMAP ══ -->
  <div id="panel-heatmap" class="panel active">
    <div class="metric-bar">
      <span style="font-size:9px;color:#4a4a6a;letter-spacing:2px;margin-right:6px">METRIC</span>
      <button class="metric-btn active" onclick="setMetric('score',this)">LIQ SCORE</button>
      <button class="metric-btn" onclick="setMetric('spreadPct',this)">SPREAD %</button>
      <button class="metric-btn" onclick="setMetric('oi',this)">OPEN INT</button>
      <button class="metric-btn" onclick="setMetric('vol',this)">VOLUME</button>
    </div>
    <div id="hm-wrap">
      <div id="hm-table-wrap"><div id="hm-grid"></div></div>
      <div id="hm-side">
        <div class="card">
          <div class="card-title">LIQUIDITY LEGEND</div>
          <div class="legend-row"><div class="dot" style="background:#00e87a"></div><div><div style="font-size:10px;color:#00e87a">LIQUID</div><div style="font-size:9px;color:#4a4a6a">Score ≥ 70</div></div></div>
          <div class="legend-row"><div class="dot" style="background:#f0c040"></div><div><div style="font-size:10px;color:#f0c040">TRADEABLE</div><div style="font-size:9px;color:#4a4a6a">Score 45–69</div></div></div>
          <div class="legend-row"><div class="dot" style="background:#ff8c42"></div><div><div style="font-size:10px;color:#ff8c42">THIN</div><div style="font-size:9px;color:#4a4a6a">Score 25–44</div></div></div>
          <div class="legend-row"><div class="dot" style="background:#ff3b5c"></div><div><div style="font-size:10px;color:#ff3b5c">AVOID</div><div style="font-size:9px;color:#4a4a6a">Score &lt; 25</div></div></div>
        </div>
        <div class="card">
          <div class="card-title">SCORE FORMULA</div>
          <div class="kv"><span class="k"><span style="color:#00e87a">50%</span> Spread tightness</span></div>
          <div class="kv"><span class="k"><span style="color:#f0c040">30%</span> Open interest</span></div>
          <div class="kv"><span class="k"><span style="color:#ff8c42">20%</span> Daily volume</span></div>
        </div>
        <div class="card" id="hover-card" style="display:none">
          <div class="card-title">CONTRACT DETAIL</div>
          <div id="hover-body"></div>
        </div>
        <div class="card" id="sel-card">
          <div class="card-title">CLICK TO SELECT</div>
          <div style="font-size:10px;color:#6a6a8a;line-height:1.7">1st click → <span style="color:#00e87a">LONG PUT</span><br>2nd click → <span style="color:#ff3b5c">SHORT PUT</span><br>3rd click → reset</div>
          <div id="sel-body" style="margin-top:8px"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ══ CHAIN ══ -->
  <div id="panel-chain" class="panel">
    <div style="font-size:9px;color:#4a4a6a;letter-spacing:2px;margin-bottom:14px">PUT OPTIONS · ALL EXPIRIES · SORTED BY LIQUIDITY SCORE (OTM STRIKES ONLY)</div>
    <div style="overflow-x:auto">
      <table class="chain-table">
        <thead>
          <tr style="border-bottom:1px solid #1a1a2e">
            <th>EXPIRY</th><th>DAYS</th><th>STRIKE</th><th>OTM%</th>
            <th>BID</th><th>ASK</th><th>MID</th><th>SPREAD%</th>
            <th>OPEN INT</th><th>VOLUME</th><th>IV%</th><th>LIQ SCORE</th><th>RATING</th>
          </tr>
        </thead>
        <tbody id="chain-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ══ CANDIDATES ══ -->
  <div id="panel-candidates" class="panel">
    <div id="top-cand"></div>
    <div style="font-size:9px;color:#4a4a6a;letter-spacing:2px;margin-bottom:10px">TOP DIAGONAL SPREAD CANDIDATES — RANKED BY COMPOSITE SCORE</div>
    <div style="overflow-x:auto">
      <table class="cand-table" style="width:100%">
        <thead>
          <tr style="border-bottom:1px solid #1a1a2e">
            <th style="text-align:left">RANK</th>
            <th>LONG PUT</th><th>LONG EXP</th><th>SHORT PUT</th><th>SHORT EXP</th>
            <th>NET COST</th><th>OFFSET%</th><th>WIDTH</th><th>TIME DIFF</th>
            <th>L-LIQ</th><th>S-LIQ</th><th>SCORE</th>
          </tr>
        </thead>
        <tbody id="cand-tbody"></tbody>
      </table>
    </div>
    <div class="grid2" id="cand-bottom"></div>
  </div>

</div>

<div id="tooltip"></div>

<script>
// ── DATA INJECTION ──────────────────────────────────────────────────────────
const DATA = __PAYLOAD__;

const {symbol, spot, asOf, rows, candidates, expiries, strikes} = DATA;

// ── HELPERS ──────────────────────────────────────────────────────────────────
function liqColor(s){
  if(s>=70) return '#00e87a';
  if(s>=45) return '#f0c040';
  if(s>=25) return '#ff8c42';
  return '#ff3b5c';
}
function liqLabel(s){
  if(s>=70) return 'LIQUID';
  if(s>=45) return 'TRADEABLE';
  if(s>=25) return 'THIN';
  return 'AVOID';
}
function cellBg(s){
  const a = 0.10 + s/500;
  if(s>=70) return `rgba(0,232,122,${a})`;
  if(s>=45) return `rgba(240,192,64,${a})`;
  if(s>=25) return `rgba(255,140,66,${a})`;
  return `rgba(255,59,92,${a})`;
}
function badge(s){
  const c=liqColor(s),l=liqLabel(s);
  return `<span class="badge" style="background:${c}18;color:${c}">${l}</span>`;
}

// ── META BAR ─────────────────────────────────────────────────────────────────
document.getElementById('m-sym').textContent   = symbol;
document.getElementById('m-spot').textContent  = '$'+spot.toFixed(2);
document.getElementById('m-date').textContent  = asOf;
document.getElementById('m-count').textContent = rows.length;
document.getElementById('m-exps').textContent  = expiries.length;

// ── LOOKUP ───────────────────────────────────────────────────────────────────
const lookup = {};
rows.forEach(r => { lookup[r.expiry+'_'+r.strike] = r; });
function get(exp,k){ return lookup[exp+'_'+k]; }

// ── HEATMAP ──────────────────────────────────────────────────────────────────
let metric = 'score';
let selLong = null, selShort = null;

// Build a reduced set of expiries for the heatmap (max 10 for readability)
const hmExps = expiries.length > 10
  ? expiries.filter((_,i)=> i % Math.ceil(expiries.length/10) === 0).slice(0,10)
  : expiries;

// Build a set of strikes that are OTM or near ATM
const hmStrikes = strikes.filter(s => s <= spot * 1.10).sort((a,b)=>b-a);
// Limit to reasonable range around spot
const filtered = hmStrikes.filter(s => s >= spot * 0.50);

function metricVal(r){
  if(!r) return null;
  if(metric==='score')    return r.liqScore;
  if(metric==='spreadPct')return r.spreadPct;
  if(metric==='oi')       return r.oi;
  if(metric==='vol')      return r.vol;
  return r.liqScore;
}
function fmt(r){
  const v = metricVal(r);
  if(v===null) return '—';
  if(metric==='spreadPct') return v.toFixed(1)+'%';
  if(metric==='oi'||metric==='vol') return v>=1000? (v/1000).toFixed(1)+'k': v;
  return v;
}

function buildHeatmap(){
  const wrap = document.getElementById('hm-grid');
  let html = '<table><thead><tr><th style="text-align:left;padding:7px 12px;font-size:9px;color:#4a4a6a">STRIKE ↓ / EXPIRY →</th>';
  hmExps.forEach(e => {
    const d = rows.find(r=>r.expiry===e);
    html += `<th style="font-size:9px;color:#6a6a8a;text-align:center;padding:7px 6px">${e.slice(5)}<br><span style="color:#3a3a5a">${d?d.daysOut+'d':''}</span></th>`;
  });
  html += '</tr></thead><tbody>';

  filtered.forEach(strike => {
    const otmPct = ((spot - strike)/spot*100).toFixed(1);
    const isATM  = Math.abs(strike - spot)/spot < 0.03;
    html += `<tr><td class="strike-label" style="color:${isATM?'#00e87a':'#c8cad8'}">$${strike} <span style="font-size:9px;color:#3a3a5a">${otmPct}%</span></td>`;
    hmExps.forEach(exp => {
      const r = get(exp, strike);
      const sc = r ? r.liqScore : 0;
      const isSelL = selLong  && selLong.exp===exp  && selLong.k===strike;
      const isSelS = selShort && selShort.exp===exp && selShort.k===strike;
      const bg = isSelL ? 'rgba(0,232,122,.25)' : isSelS ? 'rgba(255,59,92,.25)' : r ? cellBg(sc) : '#0a0a0f';
      const border = isSelL ? '1px solid #00e87a' : isSelS ? '1px solid #ff3b5c' : '1px solid transparent';
      const cellId = `c_${exp}_${strike}`;
      html += `<td class="hm-cell" id="${cellId}" style="background:${bg};border:${border}" `+
        `onmouseenter="onHover('${exp}',${strike},this)" onmouseleave="offHover()" onclick="onCellClick('${exp}',${strike})">`;
      if(r){
        html += `<div class="hm-val" style="color:${liqColor(sc)}">${fmt(r)}</div>`;
        if(metric==='score') html += `<div class="hm-sub">${liqLabel(sc)}</div>`;
        else html += `<div class="hm-sub">sc:${sc}</div>`;
      } else {
        html += `<span style="color:#2a2a3a;font-size:10px">—</span>`;
      }
      html += '</td>';
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function onHover(exp,k,el){
  const r = get(exp,k);
  if(!r) return;
  const tt = document.getElementById('tooltip');
  tt.innerHTML = `
    <div style="font-size:9px;color:#4a4a6a;letter-spacing:2px;margin-bottom:6px">${symbol} PUT DETAILS</div>
    ${[['STRIKE','$'+r.strike],['EXPIRY',r.expiry],['DAYS OUT',r.daysOut],
       ['BID','$'+r.bid.toFixed(2)],['ASK','$'+r.ask.toFixed(2)],['MID','$'+r.mid.toFixed(2)],
       ['SPREAD',r.spreadPct.toFixed(1)+'%'],['OPEN INT',r.oi.toLocaleString()],
       ['VOLUME',r.vol.toLocaleString()],['IV',(r.iv*100).toFixed(1)+'%'],
       ['LIQ SCORE',r.liqScore+' / '+liqLabel(r.liqScore)]
      ].map(([k2,v])=>`<div style="display:flex;justify-content:space-between;gap:16px;margin-bottom:3px"><span class="tk">${k2}</span><span class="tv">${v}</span></div>`).join('')}
  `;
  tt.classList.add('show');
  // position near cursor
  document.onmousemove = e => { tt.style.left=(e.clientX+14)+'px'; tt.style.top=(e.clientY-10)+'px'; };

  const card = document.getElementById('hover-card');
  card.style.display='block';
  card.style.borderColor = liqColor(r.liqScore);
  document.getElementById('hover-body').innerHTML =
    [['Strike','$'+r.strike],['Expiry',r.expiry],['Bid','$'+r.bid.toFixed(2)],
     ['Ask','$'+r.ask.toFixed(2)],['Mid','$'+r.mid.toFixed(2)],
     ['Spread',r.spreadPct.toFixed(1)+'%'],['Open Int',r.oi.toLocaleString()],
     ['Volume',r.vol.toLocaleString()],['IV',(r.iv*100).toFixed(1)+'%'],
     ['Score',r.liqScore]]
    .map(([k2,v])=>`<div class="kv"><span class="k">${k2}</span><span class="v">${v}</span></div>`).join('')+
    `<div style="margin-top:8px;padding:3px 8px;background:${liqColor(r.liqScore)}22;border-radius:3px;text-align:center;font-size:9px;color:${liqColor(r.liqScore)};letter-spacing:2px">${liqLabel(r.liqScore)}</div>`;
}
function offHover(){
  document.getElementById('tooltip').classList.remove('show');
  document.onmousemove=null;
}

function onCellClick(exp,k){
  const r = get(exp,k); if(!r) return;
  if(!selLong){ selLong={exp,k,r}; }
  else if(!selShort){ selShort={exp,k,r}; }
  else { selLong={exp,k,r}; selShort=null; }
  buildHeatmap();
  renderSelCard();
}

function renderSelCard(){
  let html='';
  if(selLong){
    const r=selLong.r;
    html+=`<div style="padding:7px;background:rgba(0,232,122,.08);border-radius:4px;margin-bottom:6px">
      <div style="font-size:9px;color:#00e87a;margin-bottom:2px">LONG PUT</div>
      <div style="font-size:11px;color:#c8cad8">$${selLong.k} / ${selLong.exp}</div>
      <div style="font-size:9px;color:#6a6a8a">Mid $${r.mid.toFixed(2)} · Score ${r.liqScore}</div>
    </div>`;
  }
  if(selShort){
    const r=selShort.r;
    html+=`<div style="padding:7px;background:rgba(255,59,92,.08);border-radius:4px;margin-bottom:6px">
      <div style="font-size:9px;color:#ff3b5c;margin-bottom:2px">SHORT PUT</div>
      <div style="font-size:11px;color:#c8cad8">$${selShort.k} / ${selShort.exp}</div>
      <div style="font-size:9px;color:#6a6a8a">Mid $${r.mid.toFixed(2)} · Score ${r.liqScore}</div>
    </div>`;
  }
  if(selLong&&selShort){
    const net=selLong.r.mid-selShort.r.mid;
    const ratio=(selShort.r.mid/selLong.r.mid*100).toFixed(0);
    const wd=selLong.k-selShort.k;
    const td=selShort.r.daysOut-selLong.r.daysOut;
    const valid=wd>0&&td>0;
    html+=`<div style="padding:8px;background:#1a1a2e;border-radius:4px;margin-top:4px">
      <div style="font-size:9px;color:#f0c040;letter-spacing:2px;margin-bottom:6px">SPREAD ANALYSIS</div>
      ${[['Net cost','$'+net.toFixed(2)],['Premium offset',ratio+'%'],
         ['Strike width','$'+wd],['Time diff',td+'d'],
         ['Valid',valid?'✓ YES':'✗ INVALID']]
        .map(([k,v])=>`<div class="kv"><span class="k">${k}</span><span style="font-size:9px;color:${v.includes('✓')?'#00e87a':v.includes('✗')?'#ff3b5c':'#c8cad8'}">${v}</span></div>`).join('')}
    </div>`;
  }
  document.getElementById('sel-body').innerHTML=html;
}

// ── METRIC TOGGLE ─────────────────────────────────────────────────────────────
function setMetric(m, btn){
  metric=m;
  document.querySelectorAll('.metric-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  buildHeatmap();
}

// ── CHAIN TABLE ───────────────────────────────────────────────────────────────
function buildChain(){
  const sorted = rows
    .filter(r => r.strike <= spot * 1.05)
    .sort((a,b)=>b.liqScore-a.liqScore);
  let html='';
  sorted.forEach(r=>{
    const otm=((spot-r.strike)/spot*100).toFixed(1);
    const sc = r.liqScore;
    const spColor = r.spreadPct<15?'#00e87a':r.spreadPct<30?'#f0c040':r.spreadPct<60?'#ff8c42':'#ff3b5c';
    html+=`<tr>
      <td style="text-align:left;color:#6a6a8a">${r.expiry}</td>
      <td style="color:#6a6a8a">${r.daysOut}</td>
      <td style="color:#c8cad8">$${r.strike}</td>
      <td style="color:#6a6a8a">${otm}%</td>
      <td>$${r.bid.toFixed(2)}</td>
      <td>$${r.ask.toFixed(2)}</td>
      <td style="color:#e8e8ff">$${r.mid.toFixed(2)}</td>
      <td style="color:${spColor}">${r.spreadPct.toFixed(1)}%</td>
      <td>${r.oi.toLocaleString()}</td>
      <td>${r.vol.toLocaleString()}</td>
      <td style="color:#6a6a8a">${(r.iv*100).toFixed(1)}%</td>
      <td style="color:${liqColor(sc)}">${sc}</td>
      <td>${badge(sc)}</td>
    </tr>`;
  });
  document.getElementById('chain-tbody').innerHTML=html;
}

// ── CANDIDATES ────────────────────────────────────────────────────────────────
function buildCandidates(){
  const top = candidates[0];
  if(top){
    document.getElementById('top-cand').innerHTML=`
    <div class="top-card">
      <div style="font-size:9px;color:#00e87a;letter-spacing:3px;margin-bottom:12px">★ OPTIMAL STRUCTURE</div>
      <div class="top-card-grid">
        <div>
          <div style="font-size:9px;color:#4a4a6a;margin-bottom:3px">LONG PUT (BUY)</div>
          <div class="big" style="color:#00e87a">K=$${top.longK}</div>
          <div style="font-size:11px;color:#6a6a8a">${top.longExp} · Mid $${top.longMid.toFixed(2)}</div>
          <div style="font-size:9px;color:#4a4a6a;margin-top:4px">OTM: ${top.longOTM}% · Score: ${top.liqL}</div>
        </div>
        <div style="display:flex;align-items:center;color:#3a3a5a;font-size:18px">—</div>
        <div>
          <div style="font-size:9px;color:#4a4a6a;margin-bottom:3px">SHORT PUT (SELL)</div>
          <div class="big" style="color:#ff8c42">K=$${top.shortK}</div>
          <div style="font-size:11px;color:#6a6a8a">${top.shortExp} · Mid $${top.shortMid.toFixed(2)}</div>
          <div style="font-size:9px;color:#4a4a6a;margin-top:4px">OTM: ${top.shortOTM}% · Score: ${top.liqS}</div>
        </div>
        <div style="border-left:1px solid #1a1a2e;padding-left:24px">
          <div style="font-size:9px;color:#4a4a6a;margin-bottom:8px">STRUCTURE</div>
          ${[['Net debit','$'+top.netCost.toFixed(2)],
             ['Premium offset',top.costPct.toFixed(0)+'%'],
             ['Strike width','$'+top.width.toFixed(1)],
             ['Time spread',top.timeDiff+'d'],
             ['Score',top.score+'/100']]
            .map(([k,v])=>`<div class="kv"><span class="k">${k}</span><span style="font-size:9px;font-weight:600;color:#e8e8ff">${v}</span></div>`).join('')}
        </div>
        <div style="border-left:1px solid #1a1a2e;padding-left:24px">
          <div style="font-size:9px;color:#4a4a6a;margin-bottom:8px">PROTECTION</div>
          ${[['Protection band','$'+top.shortK+'–$'+top.longK],
             ['Losses resume','< $'+top.shortK],
             ['Cost as % spot','$'+((top.netCost/spot)*100).toFixed(2)+'%'],
             ['Long put liq',liqLabel(top.liqL)],
             ['Short put liq',liqLabel(top.liqS)]]
            .map(([k,v])=>`<div class="kv"><span class="k">${k}</span><span style="font-size:9px;color:#c8cad8">${v}</span></div>`).join('')}
        </div>
      </div>
    </div>`;
  }

  let html='';
  candidates.forEach((c,i)=>{
    const lc=liqColor(c.liqL), sc=liqColor(c.liqS);
    html+=`<tr>
      <td style="text-align:left;color:${i===0?'#00e87a':'#4a4a6a'};font-weight:${i===0?700:400}">${i===0?'★':'#'+(i+1)}</td>
      <td style="color:#c8cad8">$${c.longK}</td>
      <td style="color:#6a6a8a">${c.longExp}</td>
      <td style="color:#c8cad8">$${c.shortK}</td>
      <td style="color:#6a6a8a">${c.shortExp}</td>
      <td style="color:#e8e8ff">$${c.netCost.toFixed(2)}</td>
      <td style="color:${c.costPct>70?'#00e87a':c.costPct>50?'#f0c040':'#ff8c42'}">${c.costPct.toFixed(0)}%</td>
      <td style="color:#c8cad8">$${c.width.toFixed(0)}</td>
      <td style="color:#6a6a8a">${c.timeDiff}d</td>
      <td><span class="badge" style="background:${lc}18;color:${lc}">${liqLabel(c.liqL)}</span></td>
      <td><span class="badge" style="background:${sc}18;color:${sc}">${liqLabel(c.liqS)}</span></td>
      <td>
        <div class="bar-wrap">
          <div class="bar-bg"><div class="bar-fg" style="width:${c.score}%;background:${c.score>=70?'#00e87a':'#f0c040'}"></div></div>
          <span style="color:${c.score>=70?'#00e87a':'#f0c040'}">${c.score}</span>
        </div>
      </td>
    </tr>`;
  });
  document.getElementById('cand-tbody').innerHTML=html;

  // Bottom cards: rolling thresholds + risk flags
  const riskRows = rows
    .filter(r=>r.strike<=spot&&r.liqScore<35)
    .sort((a,b)=>a.liqScore-b.liqScore).slice(0,6);

  document.getElementById('cand-bottom').innerHTML=`
    <div class="card">
      <div class="card-title" style="color:#f0c040">ROLLING TRIGGER THRESHOLDS</div>
      ${[['Roll when spread% >','20% of mid'],['Roll when OI drops below','200 contracts'],
         ['Roll when score drops to','< 35'],['Preferred long put OTM','5–12% (high IV)'],
         ['Preferred short put OTM','15–25% (high IV)'],['Min time spread','45 days'],
         ['Max time spread','150 days']]
        .map(([k,v])=>`<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('')}
    </div>
    <div class="card">
      <div class="card-title" style="color:#ff8c42">LIQUIDITY RISK FLAGS (SCORE &lt; 35)</div>
      ${riskRows.map(r=>`
        <div class="flag-row">
          <span style="color:#ff3b5c;font-size:9px">K$${r.strike} / ${r.expiry}</span>
          <span style="color:#6a6a8a;font-size:9px">Sp:${r.spreadPct.toFixed(0)}% OI:${r.oi}</span>
        </div>`).join('')}
      <div style="margin-top:8px;font-size:9px;color:#4a4a6a">Note: MSTR has very high IV (~80–120%). Far-OTM strikes in long-dated series carry wide spreads — always check liquidity before entry.</div>
    </div>`;
}

// ── TABS ──────────────────────────────────────────────────────────────────────
function showTab(id, btn){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  btn.classList.add('active');
}

// ── INIT ──────────────────────────────────────────────────────────────────────
buildHeatmap();
buildChain();
buildCandidates();
</script>
</body>
</html>
"""

HTML_OUT = HTML.replace('__PAYLOAD__', payload)

out_file = "mstr_options_calibrator.html"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(HTML_OUT)

print(f"\n✓  Saved → {out_file}")
print(f"   Open this file in any browser to explore the interactive calibrator.")
if top:
    print(f"\n★  Top structure: Long K${top[0]['longK']} ({top[0]['longExp']}) / Short K${top[0]['shortK']} ({top[0]['shortExp']})")
    print(f"   Net cost: ${top[0]['netCost']:.2f}  |  Premium offset: {top[0]['costPct']:.0f}%  |  Score: {top[0]['score']}/100")
