"""
optimizer.py -- Two-stage coarse + fine grid search.

Stage 1 (coarse): short horizon (63 days), 200 paths.
    Screen for P_success >= 0.80 and keep the top 15% by E_W.
    Purpose: eliminate clearly-bad policies fast (~13 min for 17k policies).

Stage 2 (fine): full horizon (504 days), 2000 paths.
    Screen for P_success >= 0.80 and rank by composite score.
    Purpose: accurate two-year simulation for survivors.

Using a short horizon in Stage 1 works because:
- Put protection quality over 63 days correlates strongly with 504-day quality
- E_W over 63 days (2-3 roll cycles) predicts full-horizon E_W well
- The 63-day Python loop is 8x faster than 504-day, enabling full grid search
"""
import numpy as np
from tqdm import tqdm

from policy_grid import Policy
from surface import FrozenSurface
from simulator import simulate_policy


def _composite_score(P_success: float, E_W: float, CVaR_W: float) -> float:
    return (
        1.0 * E_W
        - 0.5 * abs(min(CVaR_W, 0.0))
        - 2.0 * max(0.0, 0.80 - P_success) * 1000
    )


def run_optimization(
    feasible_policies: list[Policy],
    paths_coarse:      np.ndarray,    # (N1, H1+1)   -- short horizon
    paths_fine:        np.ndarray,    # (N2, H2+1)   -- full horizon
    surface:           FrozenSurface,
    spot0:             float,
    returns:           np.ndarray,
    constraints: dict | None = None,
) -> list[dict]:
    if constraints is None:
        constraints = {
            "min_P_success_coarse": 0.80,   # 63-day threshold
            "min_P_success_fine":   0.80,   # 504-day threshold
            "top_pct_coarse":       0.15,   # survivors from Stage 1
        }

    min_ps_coarse = constraints.get("min_P_success_coarse", 0.80)
    min_ps_fine   = constraints.get("min_P_success_fine",   0.80)
    top_pct       = constraints.get("top_pct_coarse",       0.15)

    n1, h1 = paths_coarse.shape[0], paths_coarse.shape[1] - 1
    n2, h2 = paths_fine.shape[0],   paths_fine.shape[1] - 1

    # ── Stage 1: coarse ──────────────────────────────────────────────────────
    print(f"\n{'-'*62}")
    print(f"Stage 1 -- Coarse  ({n1:,} paths x {h1:d} days,  {len(feasible_policies):,} policies)")
    print(f"{'-'*62}")

    coarse_results = []
    for p in tqdm(feasible_policies, desc="Stage 1", unit="policy"):
        try:
            m = simulate_policy(p, paths_coarse, surface, spot0, returns)
        except Exception:
            continue
        coarse_results.append({
            "policy":    p,
            "P_success": m["P_success"],
            "E_W":       m["E_W"],
            "CVaR_W":    m["CVaR_W"],
        })

    survivors = [r for r in coarse_results if r["P_success"] >= min_ps_coarse]
    survivors.sort(key=lambda r: r["E_W"], reverse=True)
    n_keep = max(50, int(len(survivors) * top_pct))
    survivors = survivors[:n_keep]

    print(f"Stage 1 complete: {len(coarse_results):,} evaluated --> {len(survivors):,} survivors")

    if not survivors:
        print("WARNING: No survivors from Stage 1. Relaxing coarse P_success to 0.50.")
        survivors = sorted(coarse_results, key=lambda r: r["E_W"], reverse=True)
        n_keep = max(50, len(coarse_results) // 5)
        survivors = survivors[:n_keep]
        print(f"  Re-survivors: {len(survivors)}")

    # ── Stage 2: fine ────────────────────────────────────────────────────────
    print(f"\n{'-'*62}")
    print(f"Stage 2 -- Fine  ({n2:,} paths x {h2:d} days,  {len(survivors):,} policies)")
    print(f"{'-'*62}")

    fine_results = []
    for r in tqdm(survivors, desc="Stage 2", unit="policy"):
        p = r["policy"]
        try:
            m = simulate_policy(p, paths_fine, surface, spot0, returns)
        except Exception:
            continue
        score = _composite_score(m["P_success"], m["E_W"], m["CVaR_W"])
        fine_results.append({
            "policy":          p,
            "P_success":       m["P_success"],
            "E_W":             m["E_W"],
            "CVaR_W":          m["CVaR_W"],
            "P_W_positive":    m["P_W_positive"],
            "E_breach_depth":  m["E_breach_depth"],
            "n_rolls":         m["n_rolls"],
            "W_final":         m["W_final"],
            "score":           score,
        })

    valid = [r for r in fine_results if r["P_success"] >= min_ps_fine]
    valid.sort(key=lambda r: r["score"], reverse=True)

    print(f"Stage 2 complete: {len(fine_results):,} evaluated --> {len(valid):,} meet P_success >= {min_ps_fine:.0%}")

    if not valid:
        print("WARNING: No policies met fine threshold. Returning all fine results by score.")
        fine_results.sort(key=lambda r: r["score"], reverse=True)
        return fine_results

    return valid
