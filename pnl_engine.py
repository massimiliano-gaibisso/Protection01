"""
pnl_engine.py
=============
Computes hedged vs unhedged P&L at two horizons for the calendar put spread:

  ─── T1 (protection horizon, MtM unwind) ───────────────────────────────────
  Investor closes everything at T1 mark-to-market:

    Hedged P&L(T1)  =  S(T1)
                      + max(K_long − S(T1), 0)           # long put exercised
                      − n_puts × BS_put(S(T1), K_short,  # short puts bought back MtM
                                        T2−T1, r, σ, q)
                      − S0

  ─── T2 (short-put expiry, hold to expiry) ──────────────────────────────────
  Investor holds the stock through T2; long put already paid off at T1:

    Hedged P&L(T2)  =  S(T2)
                      + max(K_long − S(T1), 0)           # long put cash received at T1
                      − n_puts × max(K_short − S(T2), 0) # short puts settled at expiry
                      − S0

  The key difference: at T1 the short puts still have TIME VALUE (drag);
  at T2 they have zero time value — only intrinsic value matters.

Both expressed as % of S0 (normalised, S0 = 100).

Public API
----------
compute_pnl(sim, zc, params) -> PnLResult

PnLResult fields / methods
--------------------------
unhedged_pct      : ndarray (n_sims,)   unhedged P&L at T1
hedged_pct        : ndarray (n_sims,)   hedged   P&L at T1  (MtM unwind)
unhedged_pct_T2   : ndarray (n_sims,)   unhedged P&L at T2  (stock held to T2)
hedged_pct_T2     : ndarray (n_sims,)   hedged   P&L at T2  (hold to expiry)
S_T1 / S_T2       : ndarray (n_sims,)   stock prices at T1 / T2
prob_protected    : float               % paths  hedged(T1) > loss_threshold
prob_protected_T2 : float               % paths  hedged(T2) > loss_threshold
prob_short_itm    : float               % paths  S(T2) < K_short
loss_threshold    : float               e.g. −10.0
.stats()          : dict                full percentile table for both horizons
.payoff_curve(S_range)    -> dict       T1 deterministic slice (short puts MtM)
.payoff_curve_T2(S_range) -> dict       T2 deterministic slice (short puts at expiry,
                                        assuming S(T1) = S(T2) — "crash-and-hold")
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from stock_simulator import SimResult
from option_pricer   import ZeroCostResult, put_price


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PnLResult:
    # ── T1: MtM unwind ────────────────────────────────────────────────────────
    unhedged_pct      : np.ndarray      # (n_sims,) stock-only P&L at T1
    hedged_pct        : np.ndarray      # (n_sims,) strategy P&L at T1 (MtM)
    # ── T2: hold to expiry ────────────────────────────────────────────────────
    unhedged_pct_T2   : np.ndarray      # (n_sims,) stock-only P&L at T2
    hedged_pct_T2     : np.ndarray      # (n_sims,) strategy P&L at T2 (expiry)
    # ── prices ────────────────────────────────────────────────────────────────
    S_T1              : np.ndarray      # (n_sims,)
    S_T2              : np.ndarray      # (n_sims,)
    # ── scalars ───────────────────────────────────────────────────────────────
    prob_protected    : float           # % paths hedged(T1) > loss_threshold
    prob_protected_T2 : float           # % paths hedged(T2) > loss_threshold
    prob_short_itm    : float           # % paths S(T2) < K_short
    loss_threshold    : float           # e.g. -10.0
    zc                : ZeroCostResult = field(repr=False)
    _params           : dict           = field(repr=False)

    # ── summary statistics ────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return a flat dict of summary statistics for both T1 and T2.

        T1 keys :  unhedged_{mean,std,p1,p5,p25,median,p75,p95,p99}
                   hedged_{...}    prob_protected
        T2 keys :  unhedged_T2_{...}
                   hedged_T2_{...} prob_protected_T2
        Risk    :  prob_short_itm, loss_threshold
        Options :  K_long, K_short, long_put_prem, short_put_prem
        """
        def _pcts(arr, prefix):
            out = {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}
            for q in (1, 5, 25, 50, 75, 95, 99):
                label = 'median' if q == 50 else f'p{q}'
                out[label] = float(np.percentile(arr, q))
            return {f'{prefix}_{k}': v for k, v in out.items()}

        d = {}
        d.update(_pcts(self.unhedged_pct,    'unhedged'))
        d.update(_pcts(self.hedged_pct,      'hedged'))
        d.update(_pcts(self.unhedged_pct_T2, 'unhedged_T2'))
        d.update(_pcts(self.hedged_pct_T2,   'hedged_T2'))
        d.update({
            'prob_protected'    : self.prob_protected,
            'prob_protected_T2' : self.prob_protected_T2,
            'prob_short_itm'    : self.prob_short_itm,
            'loss_threshold'    : self.loss_threshold,
            'K_long'            : self.zc.K_long,
            'K_short'           : self.zc.K_short,
            'long_put_prem'     : self.zc.long_put_prem,
            'short_put_prem'    : self.zc.short_put_prem,
        })
        return d

    # ── T1 deterministic payoff curve (short puts MtM) ────────────────────────

    def payoff_curve(self, S_range: np.ndarray) -> dict:
        """
        Deterministic payoff at T1 vs S(T1), short puts marked-to-market
        (BS with T2−T1 remaining).

        Returns dict: S_range, unhedged, hedged, long_put_payoff, short_put_mtm
        """
        S_range = np.asarray(S_range, dtype=np.float64)
        r, sig, q = (float(self._params[k]) for k in ('r', 'sigma', 'q'))
        T1, T2, S0 = (float(self._params[k]) for k in ('T1', 'T2', 'S0'))
        K_long, K_short, n_puts = self.zc.K_long, self.zc.K_short, self.zc.n_puts

        long_put_payoff = np.maximum(K_long - S_range, 0.0)
        short_put_mtm   = n_puts * put_price(S_range, K_short, T2 - T1, r, sig, q)
        unhedged        = S_range - S0
        hedged          = S_range + long_put_payoff - short_put_mtm - S0

        return dict(S_range=S_range, unhedged=unhedged, hedged=hedged,
                    long_put_payoff=long_put_payoff, short_put_mtm=short_put_mtm)

    # ── T2 deterministic payoff curve (short puts at expiry) ──────────────────

    def payoff_curve_T2(self, S_range: np.ndarray) -> dict:
        """
        Deterministic payoff at T2 vs S(T2), assuming S(T1) = S(T2)
        ("crash-and-hold" scenario: the stock drops once and stays there).

        At T2 the short puts settle at intrinsic value — no time value remains.
        The long put payoff used here is max(K_long − S(T2), 0), consistent
        with the same crash-and-hold assumption.

        Returns dict: S_range, unhedged, hedged,
                      long_put_payoff_T1, short_put_expiry
        """
        S_range = np.asarray(S_range, dtype=np.float64)
        S0 = float(self._params['S0'])
        K_long, K_short, n_puts = self.zc.K_long, self.zc.K_short, self.zc.n_puts

        # S(T1) = S(T2) = S_range  (crash-and-hold assumption)
        long_put_payoff_T1 = np.maximum(K_long  - S_range, 0.0)
        short_put_expiry   = n_puts * np.maximum(K_short - S_range, 0.0)
        unhedged           = S_range - S0
        hedged             = S_range + long_put_payoff_T1 - short_put_expiry - S0

        return dict(S_range=S_range, unhedged=unhedged, hedged=hedged,
                    long_put_payoff_T1=long_put_payoff_T1,
                    short_put_expiry=short_put_expiry)


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def compute_pnl(sim: SimResult, zc: ZeroCostResult, params: dict) -> PnLResult:
    """
    Compute hedged and unhedged P&L at the protection horizon T1.

    Parameters
    ----------
    sim    : SimResult
        Output of ``stock_simulator.simulate()``.
        sim.params['T'] must be >= T2 so that paths cover the full horizon.

    zc     : ZeroCostResult
        Output of ``option_pricer.solve_zero_cost_strike()``.

    params : dict
        Must contain : T1, T2, r, sigma, loss_pct
        Optional     : q  (dividend yield, default 0.0)
        Also used    : S0  (read from sim.params if not present here)

    Returns
    -------
    PnLResult
    """
    T1   = float(params['T1'])
    T2   = float(params['T2'])
    r    = float(params['r'])
    sig  = float(params['sigma'])
    q    = float(params.get('q', 0.0))
    loss = float(params['loss_pct'])
    S0   = float(params.get('S0', sim.params['S0']))

    K_long  = zc.K_long
    K_short = zc.K_short
    n_puts  = zc.n_puts

    # ── extract stock prices at T1 and T2 ─────────────────────────────────────
    S_T1 = sim.price_at(T1)     # ndarray (n_sims,)
    S_T2 = sim.price_at(T2)     # ndarray (n_sims,)

    # ── unhedged ──────────────────────────────────────────────────────────
    unhedged_PL_T1 = S_T1 - S0
    unhedged_PL_T2 = S_T2 - S0


    # ── T1: hedged (MtM unwind — short puts bought back at BS price) ──────────
    long_put_payoff = np.maximum(K_long - S_T1, 0.0)
    short_put_mtm   = n_puts * put_price(S_T1, K_short, T2 - T1, r, sig, q)
    hedged_PL_T1    = S_T1 - S0 +  long_put_payoff - short_put_mtm

    # ── T2: hedged (hold to expiry) ────────────────────────────────────────────
    # Long put cash was received at T1 (already in long_put_payoff above).
    # Short puts settle at intrinsic value at T2 — no time value remains.
    short_put_expiry = n_puts * np.maximum(K_short - S_T2, 0.0)
    hedged_PL_T2    = S_T2 - S0 + long_put_payoff - short_put_expiry

    # ── risk metrics ──────────────────────────────────────────────────────────
    loss_threshold    = -loss * S0 # i.e. - 0.1 * 100.0 = -10 

    prob_loss_unhedged_T1 = float(np.mean(unhedged_PL_T1  < loss_threshold)) 
    prob_loss_unhedged_T2 = float(np.mean(unhedged_PL_T2  < loss_threshold)) 

    prob_loss_hedged_T1   = float(np.mean(hedged_PL_T1    < loss_threshold)) 
    prob_loss_hedged_T2   = float(np.mean(hedged_PL_T2    < loss_threshold))


    prob_short_itm_T2    = float(np.mean(S_T2 < K_short))                 

    return   (
        S_T1, 
        S_T2,
        unhedged_PL_T1,
        unhedged_PL_T2,
        hedged_PL_T1,
        hedged_PL_T2,
        prob_loss_unhedged_T1,
        prob_loss_unhedged_T2,
        prob_loss_hedged_T1, 
        prob_loss_hedged_T2,
        prob_short_itm_T2 
    )
    
