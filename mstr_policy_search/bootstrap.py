"""
bootstrap.py — Vectorized bootstrap return sampler.
All paths computed simultaneously via numpy — no Python loops over paths.
"""
import numpy as np


class BootstrapSampler:
    def __init__(self, returns: np.ndarray, seed: int = 42):
        self.returns = returns
        self.rng     = np.random.default_rng(seed)

    def sample_paths(self, n_paths: int, n_days: int, s0: float) -> np.ndarray:
        """
        Returns price_paths of shape (n_paths, n_days+1).
        price_paths[:, 0] = s0.
        Each day: draw with replacement from self.returns.
        price_paths[:, d+1] = price_paths[:, d] * exp(r_d).
        """
        drawn    = self.rng.choice(self.returns, size=(n_paths, n_days), replace=True)
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(drawn, axis=1)], axis=1
        )
        return s0 * np.exp(log_paths)
