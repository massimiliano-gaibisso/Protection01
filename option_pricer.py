"""
option_pricer.py
================
Black-Scholes option pricer (vectorised) + zero-cost calendar-put-spread solver.

Wraps BlackScholes.BlackScholesCalculator for Greek calculations;
provides a fully vectorised put_price() for use with Monte Carlo arrays.

Public API
----------
put_price(S, K, T, r, sigma, q=0.0) -> float | ndarray
    Vectorised B-S put price.  S may be scalar or ndarray.

solve_zero_cost_strike(params) -> ZeroCostResult

ZeroCostResult fields
---------------------
K_long          : float   long put strike   = S0 × (1 − loss_pct)
K_short         : float   short put strike  (auto-solved for zero net premium)
long_put_prem   : float   total premium paid for the single long put
short_put_prem  : float   premium of *each* short put  (= long_put_prem / n_puts)
net_premium     : float   verification: long_put_prem − n_puts × short_put_prem  ≈ 0
n_puts          : int     number of short puts (always 2 for this strategy)
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass

from BlackScholes import BlackScholesCalculator


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised B-S put price
# ──────────────────────────────────────────────────────────────────────────────

def put_price(S, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """
    Black-Scholes European put price.

    Parameters
    ----------
    S     : float or ndarray   spot price(s)
    K     : float              strike price
    T     : float              time to maturity in years  (T=0 → intrinsic value)
    r     : float              continuously-compounded risk-free rate
    sigma : float              annualised implied volatility
    q     : float              continuous dividend yield  (default 0)

    Returns
    -------
    float or ndarray  — same shape as S
    """
    S = np.asarray(S, dtype=np.float64)

    # Edge cases
    if T <= 0.0:
        return np.maximum(K * np.exp(-r * T) - S, 0.0)
    if sigma <= 0.0:
        return np.maximum(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZeroCostResult:
    K_long:         float
    K_short:        float
    long_put_prem:  float
    short_put_prem: float   # per short put
    net_premium:    float   # ≈ 0
    n_puts:         int = 2

    def summary(self) -> str:
        lines = [
            f"  K_long        = {self.K_long:.4f}   (long put strike)",
            f"  K_short       = {self.K_short:.4f}  (short put strike, auto-solved)",
            f"  Long  put prem= {self.long_put_prem:.6f}",
            f"  Short put prem= {self.short_put_prem:.6f}  (each, ×{self.n_puts})",
            f"  Net premium   = {self.net_premium:+.2e}  (≈ 0 by construction)",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Greeks via BlackScholesCalculator (scalar, for display / diagnostics)
# ──────────────────────────────────────────────────────────────────────────────

def greeks(S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> dict:
    """
    Return a dict of put Greeks at a single spot price S0, using BlackScholesCalculator.

    Keys: delta, gamma, theta, vega, rho
    """
    calc = BlackScholesCalculator(S0, K, T, r, sigma, q)
    return {
        "delta": calc.delta("put"),
        "gamma": calc.gamma(),
        "theta": calc.theta("put"),
        "vega":  calc.vega(),
        "rho":   calc.rho("put"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Zero-cost strike solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_zero_cost_strike(params: dict) -> ZeroCostResult:
    """
    Find the short-put strike K_short such that the calendar put spread
    costs nothing at inception:

        n_puts × BS_put(S0, K_short, T2, r, σ, q)  =  BS_put(S0, K_long, T1, r, σ, q)

    where  K_long = S0 × (1 − loss_pct).

    Parameters (params dict)
    ------------------------
    Required
    ~~~~~~~~
    S0       : float   initial stock price
    loss_pct : float   fractional loss limit for the long put (e.g. 0.10 → K_long = 90)
    T1       : float   long put expiry in years   (protection horizon)
    T2       : float   short put expiry in years  (must be > T1)
    r        : float   risk-free rate
    sigma    : float   implied volatility (flat)

    Optional
    ~~~~~~~~
    q        : float   dividend yield (default 0)
    n_puts   : int     number of short puts sold  (default 2)

    Returns
    -------
    ZeroCostResult

    Raises
    ------
    ValueError  if no zero-cost strike can be found (e.g. T2 ≤ T1, or vol too low).
    """
    S0       = float(params['S0'])
    loss_pct = float(params['loss_pct'])
    T1       = float(params['T1'])
    T2       = float(params['T2'])
    r        = float(params['r'])
    sigma    = float(params['sigma'])
    q        = float(params.get('q', 0.0))
    n_puts   = int(params.get('n_puts', 2))

    if T2 <= T1:
        raise ValueError(f"T2 ({T2:.2f}y) must be strictly greater than T1 ({T1:.2f}y).")

    K_long         = S0 * (1.0 - loss_pct)
    long_put_prem  = float(put_price(S0, K_long, T1, r, sigma, q))
    target_per_put = long_put_prem / n_puts   + 1.16  # what each short put must fetch

    def _objective(K_s: float) -> float:
        return float(put_price(S0, K_s, T2, r, sigma, q)) - target_per_put

    # Bracket search: K_short in (ε, K_long)
    #   At K_s → 0  :  put premium → 0  →  objective → –target  < 0
    #   At K_s → K_long : premium ≥ target (longer expiry, same or higher strike)
    lo = max(S0 * 0.01, 0.01)
    hi = K_long * (1.0 - 1e-6)

    if _objective(hi) < 0:
        raise ValueError(
            f"Even a short put struck just below K_long ({K_long:.2f}) is not worth "
            f"enough to pay for the long put. "
            f"Try a longer T2, higher σ, or a larger loss limit."
        )
    if _objective(lo) > 0:
        raise ValueError(
            f"Even the lowest valid strike produces a premium that exceeds the target. "
            f"This should not happen — check input parameters."
        )

    K_short        = brentq(_objective, lo, hi, xtol=1e-6, maxiter=200)
    short_put_prem = float(put_price(S0, K_short, T2, r, sigma, q))
    net_premium    = long_put_prem - n_puts * short_put_prem

    return ZeroCostResult(
        K_long        = K_long,
        K_short       = K_short,
        long_put_prem = long_put_prem,
        short_put_prem= short_put_prem,
        net_premium   = net_premium,
        n_puts        = n_puts,
    )
