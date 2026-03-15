"""
egarch.py -- EGARCH(1,1) fitting and simulation with Student-t(5) innovations.

Model (Nelson 1991):
    log(sigma²_t) = omega + beta_g * log(sigma²_{t-1})
                  + alpha * (|z_{t-1}| - E[|z|])
                  + gamma * z_{t-1}
    r_t = mu + sigma_t * z_t
    z_t ~ t_5 / sqrt(5/3)     (standardised to unit variance)

Leverage effect: gamma < 0 means negative shocks raise variance more than
positive shocks of equal magnitude — the key feature MSTR exhibits.

Student-t(5) innovations: fatter tails than Normal without an extra fitted
parameter.  ν=5 is pre-fixed; E[|z|] is computed numerically at module load.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# ── Student-t(5) constants ────────────────────────────────────────────────────
_NU      = 5.0
_DF_M2   = _NU - 2.0                                    # = 3.0
_SCALE   = float(np.sqrt(_DF_M2 / _NU))                 # = sqrt(3/5) ≈ 0.7746

# Log-density constant: log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(π*(ν-2))
_C_NU = float(
    gammaln((_NU + 1) / 2) - gammaln(_NU / 2) - 0.5 * np.log(np.pi * _DF_M2)
)

# E[|z|] for the standardised t_5 (z = t_5 * scale), computed numerically once
_rng_c  = np.random.default_rng(0)
_z_samp = _rng_c.standard_t(_NU, size=2_000_000) * _SCALE
_E_ABS_Z = float(np.mean(np.abs(_z_samp)))              # ≈ 0.877
del _rng_c, _z_samp


# ── MLE fitting ───────────────────────────────────────────────────────────────

def fit_egarch(returns: np.ndarray, verbose: bool = True) -> dict:
    """
    Fit EGARCH(1,1) with fixed-ν Student-t(5) innovations to a 1-D return array.

    Parameters
    ----------
    returns : 1-D array of daily log-returns (BTC-era pool, length ~1399)
    verbose : print fitted parameters

    Returns
    -------
    dict with keys: omega, alpha, gamma, beta_g, mu, sigma0
        sigma0 = trailing 21-day realised vol (initial state for simulation)
    """
    returns = np.asarray(returns, dtype=float)
    N       = len(returns)
    mu0     = float(returns.mean())
    var0    = float(returns.var())
    log_var0 = np.log(max(var0, 1e-8))

    def neg_loglik(theta: np.ndarray) -> float:
        omega, alpha, gamma, beta_g, mu = theta
        if abs(beta_g) >= 0.9999:
            return 1e10
        lsig2 = log_var0          # initialise at unconditional variance
        z_prev = 0.0
        ll = 0.0
        for t in range(N):
            lsig2  = omega + beta_g * lsig2 + alpha * (abs(z_prev) - _E_ABS_Z) + gamma * z_prev
            sigma  = np.exp(0.5 * lsig2)
            z_t    = (returns[t] - mu) / max(sigma, 1e-8)
            ll    += _C_NU - 0.5 * lsig2 - ((_NU + 1) / 2) * np.log(1.0 + z_t ** 2 / _DF_M2)
            z_prev = z_t
        return -ll

    # Initial guess: stationary target = var0, moderate persistence
    beta0  = 0.95
    omega0 = log_var0 * (1.0 - beta0)    # so that E[log_sigma2] ≈ log_var0
    theta0 = [omega0, 0.10, -0.10, beta0, mu0]

    bounds = [
        (None,  None),        # omega
        (1e-4,  0.50),        # alpha > 0
        (-0.50, 0.50),        # gamma (leverage expected negative)
        (0.001, 0.9999),      # beta_g in (0,1) for stationarity
        (None,  None),        # mu
    ]

    result = minimize(
        neg_loglik, theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-6},
    )

    omega, alpha, gamma, beta_g, mu = result.x
    sigma0 = float(returns[-21:].std())          # trailing 21-day realised vol

    params = {
        "omega":  float(omega),
        "alpha":  float(alpha),
        "gamma":  float(gamma),
        "beta_g": float(beta_g),
        "mu":     float(mu),
        "sigma0": float(sigma0),
    }

    if verbose:
        lr = np.exp(omega / max(1.0 - beta_g, 1e-4))        # long-run variance
        print(f"\n  EGARCH(1,1) fit  [innovations: t_{int(_NU)}, fixed]")
        print(f"    omega ={omega:.6f}   alpha={alpha:.4f}   gamma={gamma:.4f}")
        print(f"    beta_g={beta_g:.4f}  mu   ={mu*100:.4f}%/day")
        print(f"    sigma0 (trailing 21d): {sigma0*100:.3f}%/day")
        print(f"    Long-run vol (annualised): {np.sqrt(lr)*np.sqrt(252)*100:.1f}%/yr")
        print(f"    Persistence alpha+beta_g = {alpha+beta_g:.4f}  (1 = unit root)")
        print(f"    E[|z|] for t_{int(_NU)} standardised: {_E_ABS_Z:.4f}")
        print(f"    MLE converged: {result.success}  nit={result.nit}  nll={result.fun:.1f}")

    return params


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_paths(
    params:              dict,
    rng:                 np.random.Generator,
    n_paths:             int,
    n_days:              int,
    spot:                float,
    drift_adj:           float = 0.0,
    annual_default_prob: float = 0.0,
    vol_cap_ann:         float | None = None,
) -> np.ndarray:
    """
    Simulate n_paths price paths of length n_days+1 using the fitted EGARCH model.

    The conditional variance follows EGARCH(1,1); innovations are t_5 standardised
    to unit variance.  All n_paths evolve simultaneously (vectorised across paths);
    the time loop is sequential (inherent EGARCH recursion).

    Parameters
    ----------
    params      : dict from fit_egarch()
    rng         : seeded numpy Generator (for reproducibility)
    drift_adj   : constant subtracted from every return (shifts mean, leaves vol intact)
    annual_default_prob : Poisson crash-to-$0.01 overlay (same as BootstrapSampler)
    vol_cap_ann : if not None, caps the EGARCH conditional sigma_t at this annualised
                  volatility (% per year, e.g. 400.0).  Applied each step by clamping
                  log_sigma2 before drawing the return.  Prevents super-unit-root
                  variance explosions beyond the worst level ever historically observed.

    Returns
    -------
    paths : (n_paths, n_days+1) price array, paths[:, 0] = spot
    """
    omega  = params["omega"]
    alpha  = params["alpha"]
    gamma  = params["gamma"]
    beta_g = params["beta_g"]
    mu     = params["mu"]
    sigma0 = params["sigma0"]

    # Pre-compute vol cap in daily sigma units (log_sigma2 ceiling) AND return clip
    # By Popoviciu's inequality: if |r_t| <= r_clip for all t, then
    #   std(r_{1..21}) <= r_clip  =>  rolling-21d-vol <= r_clip * sqrt(252) = vol_cap_ann
    # So capping both sigma_t AND clipping returns guarantees the rolling vol bound.
    _log_sig2_cap = None
    _ret_clip     = None
    if vol_cap_ann is not None:
        _daily_sigma_cap = (vol_cap_ann / 100.0) / np.sqrt(252)
        _log_sig2_cap    = 2.0 * np.log(_daily_sigma_cap)
        _ret_clip        = _daily_sigma_cap   # |r_t| <= r_clip  =>  rolling vol <= vol_cap_ann

    # Pre-draw all innovations — vectorised, shape (n_paths, n_days)
    u = rng.standard_t(_NU, size=(n_paths, n_days))
    z = u * _SCALE    # standardise to unit variance

    # Initialise state
    log_sigma2 = np.full(n_paths, np.log(max(sigma0 ** 2, 1e-10)))
    z_prev     = np.zeros(n_paths)

    log_returns = np.empty((n_paths, n_days))

    for d in range(n_days):
        log_sigma2 = (
            omega
            + beta_g * log_sigma2
            + alpha  * (np.abs(z_prev) - _E_ABS_Z)
            + gamma  * z_prev
        )
        # ── Vol cap: clamp log_sigma2 so sigma_t never exceeds vol_cap_ann ──
        if _log_sig2_cap is not None:
            np.minimum(log_sigma2, _log_sig2_cap, out=log_sigma2)
        sigma_t = np.exp(0.5 * log_sigma2)
        r_t     = mu + sigma_t * z[:, d] - drift_adj
        # ── Return clip: hard bound on each daily return (guarantees rolling-vol cap) ──
        if _ret_clip is not None:
            np.clip(r_t, -_ret_clip, _ret_clip, out=r_t)
        log_returns[:, d] = r_t
        z_prev            = z[:, d]

    # Build price paths
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    paths = spot * np.exp(log_paths)

    # Crash overlay (identical logic to BootstrapSampler)
    if annual_default_prob > 0.0:
        daily_prob = annual_default_prob / 252.0
        crash_mask = rng.random(size=(n_paths, n_days)) < daily_prob
        has_crash  = crash_mask.any(axis=1)
        if has_crash.any():
            first_crash_day = np.where(
                has_crash, crash_mask.argmax(axis=1), n_days
            ).astype(int)
            day_idx      = np.arange(n_days + 1)[np.newaxis, :]
            crash_active = day_idx > first_crash_day[:, np.newaxis]
            paths[crash_active] = 0.01

    return paths
