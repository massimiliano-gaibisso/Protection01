"""
gbm_simulator.py
================
Simple Geometric Brownian Motion (GBM) path simulator.

This is the plain Black-Scholes-compatible simulator (no jumps).
It uses sigma_total directly so simulation results are directly
comparable to BS option prices and deltas.

The log-return over each step dt is:

    ln(S_{t+dt}/S_t)  =  (mu - sigma^2/2) * dt  +  sigma * sqrt(dt) * Z

where Z ~ N(0,1).  No jump decomposition is needed.

Public API
----------
simulate_gbm(params) -> SimResult
    Returns the same SimResult type as stock_simulator.simulate(),
    so pnl_engine.compute_pnl() works unchanged.
"""

from __future__ import annotations
import numpy as np
from stock_simulator import SimResult      # reuse the shared result dataclass


def simulate_gbm(params: dict) -> SimResult:
    """
    Simulate stock-price paths under standard GBM (Black-Scholes world).

    Parameters (params dict)
    ------------------------
    S0               : float   initial stock price
    mu               : float   real-world annual drift  (e.g. 0.08)
    sigma_total      : float   annual volatility used directly for diffusion
    T                : float   simulation horizon in years
    n_steps_per_year : int     time steps per year  (default 52 -- weekly)
    n_sims           : int     number of MC paths    (default 10_000)
    seed             : int     RNG seed              (default 42)

    Note
    ----
    sigma_total is used directly (no variance-decomposition); this matches
    the volatility passed to Black-Scholes for option pricing, making the
    MC-vs-delta sanity check exact (up to sampling noise).

    Returns
    -------
    SimResult  -- identical structure to stock_simulator.SimResult
    """
    S0    = float(params['S0'])
    mu    = float(params['mu'])
    sigma = float(params['sigma_total'])
    T     = float(params['T'])
    n_spy  = int(params.get('n_steps_per_year', 52))
    n_sims = int(params.get('n_sims', 10_000))
    seed   = params.get('seed', 42)

    n_steps = max(int(round(T * n_spy)), 1)
    dt      = T / n_steps

    log_drift     = (mu - 0.5 * sigma ** 2) * dt
    log_diffusion = sigma * np.sqrt(dt)

    rng   = np.random.default_rng(seed)
    log_S = np.full(n_sims, np.log(S0), dtype=np.float64)
    paths = np.empty((n_sims, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    for t in range(n_steps):
        Z      = rng.standard_normal(n_sims)
        log_S += log_drift + log_diffusion * Z
        paths[:, t + 1] = np.exp(log_S)

    time_grid = np.linspace(0.0, T, n_steps + 1)
    return SimResult(paths=paths, time_grid=time_grid, params=dict(params))
