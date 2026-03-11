"""
bootstrap.py — Vectorized bootstrap return sampler with optional crash overlay.
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
    ) -> np.ndarray:
        """
        Returns price paths of shape (n_paths, n_days+1).
        paths[:, 0] = s0.

        annual_default_prob : if > 0, inject Poisson crash events.
            Daily crash probability = annual_default_prob / 252.
            On crash day, price drops to $0.01 and stays there permanently.
        """
        drawn     = self.rng.choice(self.returns, size=(n_paths, n_days), replace=True)
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(drawn, axis=1)], axis=1
        )
        paths = s0 * np.exp(log_paths)

        if annual_default_prob > 0.0:
            daily_prob   = annual_default_prob / 252.0
            crash_mask   = self.rng.random(size=(n_paths, n_days)) < daily_prob
            has_crash    = crash_mask.any(axis=1)                       # (n_paths,)

            if has_crash.any():
                # First crash day index (0-based in the n_days dimension)
                first_crash_day = np.where(
                    has_crash,
                    crash_mask.argmax(axis=1),   # first True column
                    n_days,                       # sentinel: no crash
                ).astype(int)

                # Apply: paths[i, d] = 0.01 for d > first_crash_day[i]
                day_idx = np.arange(n_days + 1)[np.newaxis, :]         # (1, n_days+1)
                crash_active = day_idx > first_crash_day[:, np.newaxis] # (n_paths, n_days+1)
                paths[crash_active] = 0.01

        return paths
