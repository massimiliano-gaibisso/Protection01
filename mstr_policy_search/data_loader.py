"""
data_loader.py — Fetch MSTR option chain and historical returns via yfinance.
Reuses the fetch pattern from mstr_options_calibrator.py.
Both datasets are cached to CSV so yfinance is only hit on the first run
or when --refresh is passed.

Option chain cache supports multiple dated snapshots:
  - Without --refresh : loads the most recent snapshot automatically.
  - With    --refresh : fetches live, APPENDS a new snapshot (history kept).
  - snapshot_date added as first column; all other field names are unchanged.
"""
import os
import csv
import sys
from datetime import datetime, date

import numpy as np
import pandas as pd

SYMBOL        = "MSTR"
CACHE_FILE    = os.path.join(os.path.dirname(__file__), "..", "mstr_options_cache.csv")
RETURNS_CACHE = os.path.join(os.path.dirname(__file__), "..", "mstr_returns_cache.csv")

# Full cache schema.  snapshot_date is new (first column); every other field
# name is kept exactly as it was in the original single-snapshot file.
_CACHE_FIELDS = [
    "spot", "expiry", "daysOut", "strike",
    "bid", "ask", "mid", "oi", "vol", "iv", "spreadPct",
    "snapshot_date",
]


# ─── Cache helpers ─────────────────────────────────────────────────────────────

def _read_cache_raw() -> tuple[list[dict], bool]:
    """
    Read every raw row from the cache CSV.
    Returns (rows, has_snapshot_date).
    has_snapshot_date=False means the file pre-dates the multi-snapshot schema.
    """
    if not os.path.exists(CACHE_FILE):
        return [], False
    with open(CACHE_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
        has_snap = "snapshot_date" in (reader.fieldnames or [])
    return rows, has_snap


def _write_cache(rows: list[dict]) -> None:
    """Overwrite the cache with *rows* using the full _CACHE_FIELDS schema."""
    with open(CACHE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CACHE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _migrate_legacy(rows: list[dict]) -> list[dict]:
    """
    Tag rows that pre-date the multi-snapshot schema with snapshot_date='legacy'.
    Operates in-place and returns the same list.
    """
    for r in rows:
        if "snapshot_date" not in r:
            r["snapshot_date"] = "legacy"
    return rows


def list_snapshots() -> list[str]:
    """
    Return snapshot dates present in the cache, oldest first.
    'legacy' (pre-schema rows) is always treated as the oldest entry.
    """
    rows, has_snap = _read_cache_raw()
    if not rows:
        return []
    if not has_snap:
        return ["legacy"]
    # Sort ISO dates normally; force 'legacy' to the front regardless of lex order
    raw = set(r["snapshot_date"] for r in rows)
    dated  = sorted(s for s in raw if s != "legacy")
    legacy = ["legacy"] if "legacy" in raw else []
    return legacy + dated


# ─── Internal row parser ───────────────────────────────────────────────────────

def _parse_chain_rows(raw_rows: list[dict]) -> tuple[list[dict], float]:
    """
    Convert raw CSV rows (camelCase keys, values as strings) to the internal
    chain format (snake_case keys, typed values).
    Filters out daysOut < 5 and mid == 0.
    Returns (chain_rows, spot).
    """
    chain = []
    spot  = None
    for row in raw_rows:
        if spot is None:
            spot = float(row["spot"])
        days_out = int(float(row["daysOut"]))
        if days_out < 5:
            continue
        mid = float(row["mid"])
        if mid == 0:
            continue
        chain.append({
            "expiry":     row["expiry"],
            "days_out":   days_out,
            "strike":     float(row["strike"]),
            "bid":        float(row["bid"]),
            "ask":        float(row["ask"]),
            "mid":        mid,
            "spread_pct": float(row["spreadPct"]),
            "iv":         float(row["iv"]),
            "oi":         int(row["oi"]),
            "volume":     int(row["vol"]),
        })
    return chain, (spot or 0.0)


# ─── Option chain ─────────────────────────────────────────────────────────────

def load_option_chain(
    symbol:  str  = SYMBOL,
    refresh: bool = False,
) -> tuple[list[dict], float, str]:
    """
    Returns (chain_rows, spot, fetch_date).

    refresh=False (default)
        Loads the most recent snapshot from the multi-snapshot cache.
        If the cache has no snapshot_date column (legacy format), loads all rows
        exactly as the original code did — no behaviour change for existing files.

    refresh=True
        Fetches live from yfinance, appends a new dated snapshot to the cache
        (all previous snapshots are preserved), and returns the fresh chain.
        Migrates a legacy cache to the new schema on the first refresh run.
    """
    today_str = date.today().isoformat()

    # ── no-refresh path: load from cache ──────────────────────────────────────
    if not refresh:
        raw_rows, has_snap = _read_cache_raw()
        if raw_rows:
            if not has_snap:
                # ── legacy file (no snapshot_date): original behaviour preserved ──
                chain, spot = _parse_chain_rows(raw_rows)
                if spot == 0:
                    sys.exit(f"Cache {CACHE_FILE} is empty — delete it and retry.")
                print(f"Loading cached option chain from {CACHE_FILE}"
                      f"  (pass --refresh to fetch live) ...")
                print(f"  Spot  : ${spot}  (cached)")
                print(f"  Chain : {len(chain)} put contracts loaded")
            else:
                # ── multi-snapshot file: load the latest snapshot ──────────────
                # Use list_snapshots() so 'legacy' is always treated as oldest.
                snapshots = list_snapshots()
                latest    = snapshots[-1]
                snap_rows = [r for r in raw_rows if r["snapshot_date"] == latest]
                chain, spot = _parse_chain_rows(snap_rows)
                if spot == 0:
                    sys.exit(f"Cache {CACHE_FILE} is empty — delete it and retry.")
                print(f"Loading cached option chain  [snapshot: {latest}]"
                      f"  (pass --refresh to fetch live) ...")
                print(f"  Spot  : ${spot}  (cached)")
                print(f"  Chain : {len(chain)} put contracts loaded")
                if len(snapshots) > 1:
                    print(f"  Archive: {len(snapshots)} snapshot(s)  "
                          f"({snapshots[0]} … {snapshots[-1]})")
            return chain, spot, today_str
        # No cache at all → fall through to live fetch

    # ── live fetch ─────────────────────────────────────────────────────────────
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not found — run: pip install yfinance")

    print(f"Fetching {symbol} option chain (live) ...")
    tk         = yf.Ticker(symbol)
    spot       = round(float(tk.fast_info.last_price), 2)
    today_date = date.today()
    print(f"  Spot  : ${spot}")

    all_exps = tk.options
    print(f"  Expiries: {len(all_exps)} series")

    fetched: list[dict] = []
    for exp_str in all_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        days_out = (exp_date - today_date).days
        if days_out < 5:
            continue
        try:
            chain_data = tk.option_chain(exp_str)
            puts = chain_data.puts
        except Exception as e:
            print(f"  Warning: could not fetch {exp_str}: {e}")
            continue

        for _, r in puts.iterrows():
            strike = float(r["strike"])
            bid    = float(r["bid"])              if r["bid"]   == r["bid"]   else 0.0
            ask    = float(r["ask"])              if r["ask"]   == r["ask"]   else bid * 1.2
            oi     = int(r["openInterest"])       if r["openInterest"] == r["openInterest"] else 0
            vol    = int(r["volume"])             if r["volume"] == r["volume"]             else 0
            iv     = float(r["impliedVolatility"]) \
                     if r["impliedVolatility"] == r["impliedVolatility"] else 0.0
            mid        = (bid + ask) / 2 if (bid + ask) > 0 else 0.0
            if mid == 0:
                continue
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999.0
            fetched.append({
                "expiry":     exp_str,
                "days_out":   days_out,
                "strike":     round(strike, 2),
                "bid":        round(bid, 2),
                "ask":        round(ask, 2),
                "mid":        round(mid, 2),
                "spread_pct": round(spread_pct, 1),
                "iv":         round(iv, 4),
                "oi":         oi,
                "volume":     vol,
            })

    print(f"  Chain : {len(fetched)} put contracts loaded")

    # ── persist: migrate legacy rows if needed, then append new snapshot ───────
    existing, has_snap = _read_cache_raw()
    if existing and not has_snap:
        _migrate_legacy(existing)          # adds snapshot_date='legacy' in-place

    # Build new snapshot rows (camelCase keys to match cache schema)
    new_rows = [
        {
            "snapshot_date": today_str,
            "spot":          str(spot),
            "expiry":        r["expiry"],
            "daysOut":       str(r["days_out"]),
            "strike":        str(r["strike"]),
            "bid":           str(r["bid"]),
            "ask":           str(r["ask"]),
            "mid":           str(r["mid"]),
            "oi":            str(r["oi"]),
            "vol":           str(r["volume"]),
            "iv":            str(r["iv"]),
            "spreadPct":     str(r["spread_pct"]),
        }
        for r in fetched
    ]

    _write_cache(existing + new_rows)
    n_snaps = len(list_snapshots())
    print(f"  Snapshot {today_str} appended → {CACHE_FILE}  "
          f"(archive: {n_snaps} snapshot(s))")

    return fetched, spot, today_str


# ─── Historical returns ────────────────────────────────────────────────────────

def load_historical_returns(
    symbol:      str       = SYMBOL,
    refresh:     bool      = False,
    cutoff_date: str | None = None,
) -> np.ndarray:
    """
    Fetch maximum available MSTR daily closing prices and return log-return array.
    Results are cached to RETURNS_CACHE so yfinance is only hit once (or on --refresh).

    cutoff_date : if given (e.g. "2020-08-11"), only returns on-or-after that date
                  are included in the pool.  Use BTC_ERA_CUTOFF for the BTC-era filter.
    """
    use_cache = (not refresh) and os.path.exists(RETURNS_CACHE)

    if use_cache:
        print(f"Loading cached returns from {RETURNS_CACHE} ...")
        df = pd.read_csv(RETURNS_CACHE)
        if cutoff_date is not None:
            df = df[df["date"] >= cutoff_date]
            print(f"  BTC-era filter applied (>= {cutoff_date}): {len(df)} days retained")
        returns = df["log_return"].values
        _print_return_stats(returns)
        return returns

    # ── live fetch ──────────────────────────────────────────────────────────
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not found — run: pip install yfinance")

    print(f"Fetching {symbol} historical prices (live) ...")
    tk      = yf.Ticker(symbol)
    hist    = tk.history(period="max")
    prices  = hist["Close"].dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    fetch_date = date.today().isoformat()

    cache_df = pd.DataFrame({
        "date":       log_ret.index.strftime("%Y-%m-%d")
                      if hasattr(log_ret.index, "strftime")
                      else range(len(log_ret)),
        "log_return": log_ret.values,
    })
    cache_df["fetch_date"] = fetch_date
    cache_df.to_csv(RETURNS_CACHE, index=False)
    print(f"  Returns cached → {RETURNS_CACHE}")

    if cutoff_date is not None:
        mask    = cache_df["date"] >= cutoff_date
        returns = cache_df.loc[mask, "log_return"].values
        print(f"  BTC-era filter applied (>= {cutoff_date}): {len(returns)} days retained")
    else:
        returns = log_ret.values

    _print_return_stats(returns)
    return returns


def _print_return_stats(returns: np.ndarray) -> None:
    worst_idx = int(np.argmin(returns))
    best_idx  = int(np.argmax(returns))
    print(f"MSTR historical returns: N={len(returns)} trading days")
    print(f"  Mean daily:  {returns.mean()*100:+.2f}%")
    print(f"  Std daily:   {returns.std()*100:.2f}%")
    print(f"  Min (worst): {returns[worst_idx]*100:.1f}%")
    print(f"  Max (best):  {returns[best_idx]*100:+.1f}%")
    print(f"  10th pct:    {np.percentile(returns,10)*100:.1f}%")
    print(f"  90th pct:    {np.percentile(returns,90)*100:+.1f}%")
