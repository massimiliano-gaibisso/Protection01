"""
bootstrap.py — Vectorized bootstrap return sampler with optional crash overlay.

Supports two sampling modes controlled by the block_size parameter:

  block_size = 1  (default)
      I.I.D. bootstrap: each return drawn independently with replacement.
      Preserves marginal distribution; destroys all serial dependence.

  block_size > 1
      Circular block bootstrap: contiguous blocks of length block_size are
      sampled with replacement from the return pool.  The pool is treated as
      circular so blocks wrap around the end, avoiding edge bias.
      Preserves short-term serial autocorrelation and GARCH vol clustering up
      to approximately block_size lags.  Marginal distribution is unchanged.

      Recommended block sizes for daily MSTR returns:
          5  — 1 trading week  (minimal clustering capture)
         10  — 2 trading weeks (default on feature/block-bootstrap branch)
         21  — 1 trading month (captures most GARCH persistence)
"""
import numpy as np


class BootstrapSampler:
    def __init__(self, returns: np.ndarray, seed: int = 42):
        self.returns      = returns
        self.rng          = np.random.default_rng(seed)
        self.egarch_params: dict | None = None

    def set_egarch_params(self, params: dict) -> None:
        """Store fitted EGARCH params for use by sample_paths(use_egarch=True)."""
        self.egarch_params = params

    def sample_paths(
        self,
        n_paths:             int,
        n_days:              int,
        s0:                  float,
        annual_default_prob: float = 0.0,
        block_size:          int   = 1,
        drift_adj:           float = 0.0,
        use_egarch:          bool  = False,
        vol_cap_ann:         float | None = None,
    ) -> np.ndarray:
        """
        Returns price paths of shape (n_paths, n_days+1).
        paths[:, 0] = s0.

        Parameters
        ----------
        n_paths             : number of Monte Carlo paths
        n_days              : length of each path in trading days
        s0                  : initial stock price
        annual_default_prob : if > 0, inject Poisson crash events;
                              daily prob = annual_default_prob / 252;
                              on crash day price drops to $0.01 permanently.
        block_size          : 1  → i.i.d. bootstrap (original behaviour)
                              >1 → circular block bootstrap
        drift_adj           : constant subtracted from every bootstrapped
                              log-return before path construction (daily units).
                              0.0      → raw historical mean preserved
                              -0.001   → subtract 0.10%/day (moderate headwind)
                              -0.002   → subtract 0.20%/day (severe stress)
                              vol structure is unchanged; only the mean shifts.
        use_egarch          : if True AND egarch_params have been set via
                              set_egarch_params(), delegate to the EGARCH
                              simulator instead of block bootstrap.
                              drift_adj and annual_default_prob are forwarded;
                              block_size is ignored.
        vol_cap_ann         : passed to EGARCH simulate_paths; caps conditional
                              sigma_t at this annualised vol (%/yr).  None = no cap.
                              Ignored when use_egarch=False.
        """
        # ── EGARCH delegation ─────────────────────────────────────────────────
        if use_egarch and self.egarch_params is not None:
            from egarch import simulate_paths as _eg_sim
            return _eg_sim(
                params              = self.egarch_params,
                rng                 = self.rng,
                n_paths             = n_paths,
                n_days              = n_days,
                spot                = s0,
                drift_adj           = drift_adj,
                annual_default_prob = annual_default_prob,
                vol_cap_ann         = vol_cap_ann,
            )

        N = len(self.returns)

        if block_size <= 1:
            # ── I.I.D. bootstrap (original) ───────────────────────────────────
            drawn = self.rng.choice(self.returns, size=(n_paths, n_days),
                                    replace=True)
        else:
            # ── Circular block bootstrap ──────────────────────────────────────
            # Number of blocks needed to cover n_days (last block may overshoot)
            n_blocks = int(np.ceil(n_days / block_size))

            # Random block start positions: (n_paths, n_blocks), uniform in [0, N)
            starts = self.rng.integers(0, N, size=(n_paths, n_blocks))

            # Block offsets: (block_size,) → broadcast to (n_paths, n_blocks, block_size)
            offsets = np.arange(block_size)
            indices = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]) % N

            # Flatten to (n_paths, n_blocks * block_size) and trim to n_days
            drawn = self.returns[indices.reshape(n_paths, n_blocks * block_size)
                                 ][:, :n_days]

        # ── Parametric drift adjustment ───────────────────────────────────────
        # Subtract a constant from every return to shift the simulated mean
        # without altering the vol structure or serial dependence pattern.
        if drift_adj != 0.0:
            drawn = drawn - drift_adj

        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(drawn, axis=1)], axis=1
        )
        paths = s0 * np.exp(log_paths)

        if annual_default_prob > 0.0:
            daily_prob = annual_default_prob / 252.0
            crash_mask = self.rng.random(size=(n_paths, n_days)) < daily_prob
            has_crash  = crash_mask.any(axis=1)

            if has_crash.any():
                first_crash_day = np.where(
                    has_crash,
                    crash_mask.argmax(axis=1),
                    n_days,
                ).astype(int)

                day_idx      = np.arange(n_days + 1)[np.newaxis, :]
                crash_active = day_idx > first_crash_day[:, np.newaxis]
                paths[crash_active] = 0.01

        return paths
