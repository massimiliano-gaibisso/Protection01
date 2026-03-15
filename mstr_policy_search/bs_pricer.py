"""
bs_pricer.py -- Vectorised Black-Scholes put pricer.

Used at roll events to price options at the vol-scaled implied volatility.
The frozen surface provides the IV; this module converts IV + market params
to a dollar price.

All inputs may be scalars or numpy arrays of the same shape.
"""
import numpy as np
from scipy.special import erf as _erf

_SQRT2 = np.sqrt(2.0)
_MIN_T = 1.0 / 252.0       # floor: 1 trading day in years


def bs_put_vec(
    S:       np.ndarray,   # current spot
    K:       np.ndarray,   # strike
    T_years: np.ndarray,   # time-to-expiry in years
    iv:      np.ndarray,   # annualised implied vol (e.g. 1.5 = 150%)
    r:       float = 0.0,  # risk-free rate (continuously compounded)
) -> np.ndarray:
    """
    Vectorised European put price via Black-Scholes.

    Returns max(BS_put, intrinsic) as a safety floor — consistent with the
    intrinsic-floor treatment already applied in simulator.py.

    Parameters
    ----------
    S, K, T_years, iv : broadcastable arrays
    r                  : scalar risk-free rate

    Notes
    -----
    Normal CDF implemented as 0.5*(1 + erf(x/sqrt(2))) via scipy.special.erf.
    T_years is floored at 1/252 to avoid division-by-zero on expiry day.
    iv is expected to be already clipped by surface.iv_vector (0.05, 5.0).
    """
    S  = np.asarray(S,       dtype=float)
    K  = np.asarray(K,       dtype=float)
    iv = np.asarray(iv,      dtype=float)
    T  = np.maximum(np.asarray(T_years, dtype=float), _MIN_T)

    sqT = np.sqrt(T)

    # Protect against S=0 (crash paths) to avoid log(0)
    S_safe = np.maximum(S, 1e-6)

    d1 = (np.log(S_safe / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqT)
    d2 = d1 - iv * sqT

    # N(-x) = 0.5*(1 - erf(x/sqrt(2)))
    Nd2 = 0.5 * (1.0 - _erf( d2 / _SQRT2))   # N(-d2) = P(put expires ITM)
    Nd1 = 0.5 * (1.0 - _erf( d1 / _SQRT2))   # N(-d1)

    disc = np.exp(-r * T)
    p    = K * disc * Nd2 - S * Nd1

    # Intrinsic floor: max(K - S, 0)
    return np.maximum(p, np.maximum(K - S, 0.0))
