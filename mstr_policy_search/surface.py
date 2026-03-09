"""
surface.py — Frozen option surface built from the live chain snapshot.

The surface is indexed in moneyness space (K/S0) × days_out.
All simulation queries rescale the target strike to moneyness and look up
the equivalent strike in the original chain (frozen surface assumption).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def build_surface(chain_rows: list[dict], spot0: float) -> "FrozenSurface":
    """
    Build a FrozenSurface from the chain snapshot.
    chain_rows: list of dicts as returned by data_loader.load_option_chain()
    spot0: current spot price used to compute moneyness grid
    """
    return FrozenSurface(chain_rows, spot0)


class FrozenSurface:
    """
    Vectorized lookup over a frozen (moneyness, days_out) grid.

    At query time:
        moneyness  = K_target / S_simulated
        K_equiv    = moneyness * spot0          (rescale to original chain)
        price      = interpolate(moneyness, days_out, side)
    """

    def __init__(self, chain_rows: list[dict], spot0: float):
        self.spot0 = spot0
        df = pd.DataFrame(chain_rows)

        # Compute moneyness for each row
        df["moneyness"] = df["strike"] / spot0

        # Build separate grids for bid, ask, mid
        # We create a 2-D interpolator over (moneyness, days_out)
        # Aggregate duplicate (moneyness, days_out) pairs by mean
        for side in ("bid", "ask", "mid"):
            pivot = (
                df.groupby(["moneyness", "days_out"])[side]
                .mean()
                .unstack("days_out")      # rows=moneyness, cols=days_out
            )
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            mn_grid   = pivot.index.values.astype(float)
            days_grid = pivot.columns.values.astype(float)
            values    = pivot.values  # shape (n_mn, n_days)

            # Fill NaN by forward-fill then backward-fill along moneyness axis
            df_fill = pd.DataFrame(values, index=mn_grid, columns=days_grid)
            df_fill = df_fill.ffill(axis=0).bfill(axis=0)
            df_fill = df_fill.ffill(axis=1).bfill(axis=1)
            values  = df_fill.values.clip(0)

            interp = RegularGridInterpolator(
                (mn_grid, days_grid),
                values,
                method="linear",
                bounds_error=False,
                fill_value=None,   # nearest-edge extrapolation
            )
            setattr(self, f"_interp_{side}", interp)
            setattr(self, f"_mn_grid_{side}", mn_grid)
            setattr(self, f"_days_grid_{side}", days_grid)

        # Store raw df for snap_strike
        self._df = df

    # ── scalar lookup (used during grid filtering) ──────────────────────────
    def price(self, K: float, days_out: float, S_sim: float, side: str = "mid") -> float | None:
        """
        Price one option.  K is the target strike at simulated spot S_sim.
        Rescales to moneyness in the original chain.
        Returns None if out of range.
        """
        moneyness = K / S_sim
        interp = getattr(self, f"_interp_{side}")
        mn_min = getattr(self, f"_mn_grid_{side}").min()
        mn_max = getattr(self, f"_mn_grid_{side}").max()
        days_min = getattr(self, f"_days_grid_{side}").min()
        days_max = getattr(self, f"_days_grid_{side}").max()

        # Clamp to grid edges (nearest-edge extrapolation)
        mn_q   = float(np.clip(moneyness, mn_min, mn_max))
        days_q = float(np.clip(days_out, days_min, days_max))

        val = float(interp([[mn_q, days_q]])[0])
        return max(val, 0.0)

    # ── vectorized lookup (called inside simulator for all paths at once) ──
    def price_vector(
        self,
        K_vec:    np.ndarray,   # shape (n_paths,)  target strikes
        days_vec: np.ndarray,   # shape (n_paths,) or scalar  days remaining
        S_vec:    np.ndarray,   # shape (n_paths,)  simulated spot
        side:     str = "mid",
    ) -> np.ndarray:
        """
        Vectorized pricing.  Returns shape (n_paths,).
        """
        moneyness = K_vec / S_vec                  # (n_paths,)
        interp    = getattr(self, f"_interp_{side}")
        mn_grid   = getattr(self, f"_mn_grid_{side}")
        days_grid = getattr(self, f"_days_grid_{side}")

        if np.isscalar(days_vec):
            days_vec = np.full_like(moneyness, days_vec)

        mn_q   = np.clip(moneyness, mn_grid.min(), mn_grid.max())
        days_q = np.clip(days_vec,  days_grid.min(), days_grid.max())

        pts    = np.stack([mn_q, days_q], axis=1)   # (n_paths, 2)
        prices = interp(pts)
        return np.maximum(prices, 0.0)

    # ── snap to nearest liquid strike in original chain ───────────────────
    def snap_strike(
        self,
        target_K:      float,
        target_days:   float,
        max_spread_pct: float = 15.0,
        days_tolerance: int   = 7,
    ) -> float | None:
        """
        Find the nearest liquid strike in the chain within ±days_tolerance of target_days.
        Returns None if nothing found.
        """
        df = self._df
        mask = (
            (df["days_out"] >= target_days - days_tolerance) &
            (df["days_out"] <= target_days + days_tolerance) &
            (df["spread_pct"] <= max_spread_pct)
        )
        candidates = df[mask]
        if candidates.empty:
            return None
        idx = (candidates["strike"] - target_K).abs().idxmin()
        return float(candidates.loc[idx, "strike"])
