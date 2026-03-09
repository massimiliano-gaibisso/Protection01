"""
data_loader.py — Fetch MSTR option chain and historical returns via yfinance.
Reuses the fetch pattern from mstr_options_calibrator.py.
Both datasets are cached to CSV so yfinance is only hit on the first run
or when --refresh is passed.
"""
import os
import csv
import sys
from datetime import datetime, date

import numpy as np
import pandas as pd

SYMBOL            = "MSTR"
CACHE_FILE        = os.path.join(os.path.dirname(__file__), "..", "mstr_options_cache.csv")
RETURNS_CACHE     = os.path.join(os.path.dirname(__file__), "..", "mstr_returns_cache.csv")


# ─── Option chain ─────────────────────────────────────────────────────────────

def load_option_chain(symbol: str = SYMBOL, refresh: bool = False) -> tuple[list[dict], float, str]:
    """
    Returns (chain_rows, spot, fetch_date).
    Loads from cache if available; fetches live otherwise.
    """
    today = date.today()
    rows  = []
    spot  = None

    use_cache = (not refresh) and os.path.exists(CACHE_FILE)

    if use_cache:
        print(f"Loading cached option chain from {CACHE_FILE}  (pass --refresh to fetch live) ...")
        with open(CACHE_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if spot is None:
                    spot = float(row["spot"])
                exp_date = datetime.strptime(row["expiry"], "%Y-%m-%d").date()
                days_out = (exp_date - today).days
                if days_out < 5:
                    continue
                mid = float(row["mid"])
                if mid == 0:
                    continue
                rows.append({
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
        if spot is None:
            sys.exit(f"Cache file {CACHE_FILE} is empty -- delete it and retry.")
        print(f"  Spot  : ${spot}  (cached)")
        print(f"  Chain : {len(rows)} put contracts loaded")
        return rows, spot, today.isoformat()

    # ── live fetch ──────────────────────────────────────────────────────────
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not found -- run: pip install yfinance")

    print(f"Fetching {symbol} option chain (live) ...")
    tk   = yf.Ticker(symbol)
    spot = round(float(tk.fast_info.last_price), 2)
    print(f"  Spot  : ${spot}")

    all_exps = tk.options
    print(f"  Expiries: {len(all_exps)} series")

    for exp_str in all_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        days_out = (exp_date - today).days
        if days_out < 5:
            continue
        try:
            chain = tk.option_chain(exp_str)
            puts  = chain.puts
        except Exception as e:
            print(f"  Warning: could not fetch {exp_str}: {e}")
            continue

        for _, r in puts.iterrows():
            strike = float(r["strike"])
            bid    = float(r["bid"])            if r["bid"]   == r["bid"]   else 0.0
            ask    = float(r["ask"])            if r["ask"]   == r["ask"]   else bid * 1.2
            oi     = int(r["openInterest"])     if r["openInterest"] == r["openInterest"] else 0
            vol    = int(r["volume"])           if r["volume"] == r["volume"]            else 0
            iv     = float(r["impliedVolatility"]) if r["impliedVolatility"] == r["impliedVolatility"] else 0.0

            mid = (bid + ask) / 2 if (bid + ask) > 0 else 0.0
            if mid == 0:
                continue
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999.0

            rows.append({
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

    print(f"  Chain : {len(rows)} put contracts loaded")
    return rows, spot, today.isoformat()


# ─── Historical returns ────────────────────────────────────────────────────────

def load_historical_returns(symbol: str = SYMBOL, refresh: bool = False) -> np.ndarray:
    """
    Fetch maximum available MSTR daily closing prices and return log-return array.
    Results are cached to RETURNS_CACHE so yfinance is only hit once (or on --refresh).
    """
    use_cache = (not refresh) and os.path.exists(RETURNS_CACHE)

    if use_cache:
        print(f"Loading cached returns from {RETURNS_CACHE} ...")
        df = pd.read_csv(RETURNS_CACHE)
        returns = df["log_return"].values
        fetch_date = df.attrs.get("fetch_date", "unknown") if hasattr(df, "attrs") else "cached"
        _print_return_stats(returns)
        return returns

    # ── live fetch ──────────────────────────────────────────────────────────
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not found -- run: pip install yfinance")

    print(f"Fetching {symbol} historical prices (live) ...")
    tk      = yf.Ticker(symbol)
    hist    = tk.history(period="max")
    prices  = hist["Close"].dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    fetch_date = date.today().isoformat()

    # Save cache
    cache_df = pd.DataFrame({
        "date":       log_ret.index.strftime("%Y-%m-%d") if hasattr(log_ret.index, "strftime") else range(len(log_ret)),
        "log_return": log_ret.values,
    })
    cache_df["fetch_date"] = fetch_date
    cache_df.to_csv(RETURNS_CACHE, index=False)
    print(f"  Returns cached -> {RETURNS_CACHE}")

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
