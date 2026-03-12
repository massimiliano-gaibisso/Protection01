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
        self.returns = returns
        self.rng     = np.random.default_rng(seed)

    def sample_paths(
        self,
        n_paths:             int,
        n_days:              int,
        s0:                  float,
        annual_default_prob: float = 0.0,
        block_size:          int   = 1,
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
        """
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
