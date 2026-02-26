"""
stock_simulator.py
==================
Merton (1976) jump-diffusion Monte Carlo path simulator.

Public API
----------
simulate(params) -> SimResult

SimResult fields
----------------
paths       : ndarray (n_sims, n_steps+1)   stock price at each time step
time_grid   : ndarray (n_steps+1,)           time in years at each step
params      : dict                            original input params (unchanged)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    paths:      np.ndarray          # shape (n_sims, n_steps+1)
    time_grid:  np.ndarray          # shape (n_steps+1,)  — values in years
    params:     dict = field(repr=False)

    # ── convenience ───────────────────────────────────────────────────────────

    def price_at(self, t: float) -> np.ndarray:
        """Return the (n_sims,) stock-price vector at the time closest to t."""
        idx = int(np.searchsorted(self.time_grid, t))
        idx = min(idx, len(self.time_grid) - 1)
        return self.paths[:, idx]

    @property
    def n_sims(self) -> int:
        return self.paths.shape[0]


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def simulate(params: dict) -> SimResult:
    """
    Simulate stock-price paths under the Merton (1976) jump-diffusion model.

    Under the real-world measure the log-return over each step dt is:

        ln(S_{t+dt}/S_t) = (μ − ½σ² − λk̄) dt  +  σ√dt · Z  +  J

    where
        Z  ~ N(0,1)  (Brownian increment)
        J  = Σᵢ₌₁ᴺ Yᵢ   with  N ~ Poisson(λ dt),  Yᵢ ~ N(μⱼ, σⱼ²)
        k̄  = E[eʸ − 1] = exp(μⱼ + ½σⱼ²) − 1   (mean proportional jump size)

    Parameters (params dict)
    ------------------------
    Required
    ~~~~~~~~
    S0              : float   initial stock price
    mu              : float   real-world annual drift  (e.g. 0.08)
    sigma           : float   annual diffusion volatility (e.g. 0.20)
    T               : float   simulation horizon in years

    Jump model (Merton)
    ~~~~~~~~~~~~~~~~~~~
    lam             : float   jump intensity  λ — expected jumps per year (default 0.5)
    mu_j            : float   mean log-jump size  μⱼ             (default -0.20)
    sigma_j         : float   std-dev of log-jump size  σⱼ       (default 0.15)

    Discretisation
    ~~~~~~~~~~~~~~
    n_steps_per_year: int     time steps per year  (default 52 — weekly)
    n_sims          : int     number of Monte Carlo paths  (default 10_000)
    seed            : int|None  RNG seed for reproducibility  (default 42)

    Returns
    -------
    SimResult  — see class docstring above.

    Examples
    --------
    >>> from stock_simulator import simulate
    >>> res = simulate({'S0': 100, 'mu': 0.08, 'sigma': 0.20, 'T': 2.0,
    ...                 'lam': 0.5, 'mu_j': -0.20, 'sigma_j': 0.15,
    ...                 'n_sims': 5000})
    >>> res.paths.shape
    (5000, 105)
    >>> prices_at_1yr = res.price_at(1.0)   # shape (5000,)
    """
    # ── unpack ────────────────────────────────────────────────────────────────
    S0      = float(params['S0'])
    mu      = float(params['mu'])
    sigma   = float(params['sigma'])
    T       = float(params['T'])
    lam     = float(params.get('lam',    0.5))
    mu_j    = float(params.get('mu_j',  -0.20))
    sigma_j = float(params.get('sigma_j', 0.15))
    n_spy   = int(params.get('n_steps_per_year', 52))
    n_sims  = int(params.get('n_sims', 10_000))
    seed    = params.get('seed', 42)

    n_steps = max(int(round(T * n_spy)), 1)
    dt      = T / n_steps

    # ── Merton drift correction ───────────────────────────────────────────────
    k_bar      = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0   # E[e^Y - 1]
    log_drift  = (mu - 0.5 * sigma ** 2 - lam * k_bar) * dt

    # ── simulation ────────────────────────────────────────────────────────────
    rng    = np.random.default_rng(seed)
    log_S  = np.full(n_sims, np.log(S0), dtype=np.float64)
    paths  = np.empty((n_sims, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    for t in range(n_steps):
        # diffusion
        Z = rng.standard_normal(n_sims)

        # compound-Poisson jumps (vectorised)
        N    = rng.poisson(lam * dt, n_sims)        # number of jumps per path
        mask = N > 0
        jump_log = np.zeros(n_sims)
        if mask.any():
            n_j = N[mask]
            # total log-jump | N ~ N(N·μⱼ, N·σⱼ²)
            jump_log[mask] = (n_j * mu_j
                              + np.sqrt(n_j.astype(float)) * sigma_j
                              * rng.standard_normal(int(mask.sum())))

        log_S          += log_drift + sigma * np.sqrt(dt) * Z + jump_log
        paths[:, t + 1] = np.exp(log_S)

    time_grid = np.linspace(0.0, T, n_steps + 1)
    return SimResult(paths=paths, time_grid=time_grid, params=dict(params))
