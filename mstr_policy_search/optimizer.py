"""
optimizer.py — Three-phase coordinate ascent with Common Random Numbers (CRN).

Phase 1 — Random init   : evaluate N_INIT random feasible policies on search paths
Phase 2 — Coord ascent  : from top K_STARTS seeds, run coordinate ascent on each param
Phase 3 — Fine verify   : re-evaluate the top local optima on 2,000 fine paths

CRN: Phases 1 and 2 use the SAME pre-generated search paths so that path noise
cancels in pairwise score comparisons (variance reduction).

Score: P_success − E_breach_depth / spot
  → maximising this prioritises breach protection above all else.
  → no income bias: E_W is not in the objective.
"""
import numpy as np
from tqdm import tqdm

import policy_grid as pg
from surface import FrozenSurface
from simulator import simulate_policy


def _score(m: dict, spot: float) -> float:
    return m["P_success"] - m["E_breach_depth"] / spot


def _evaluate(
    policy:          pg.Policy,
    paths:           np.ndarray,
    surface:         FrozenSurface,
    spot0:           float,
    delta_floor:     float,
    cost_per_leg:    float,
    frozen_expiries: list | None = None,
) -> float:
    try:
        m = simulate_policy(policy, paths, surface, spot0,
                            delta_floor=delta_floor, cost_per_leg=cost_per_leg,
                            frozen_expiries=frozen_expiries)
        return _score(m, spot0)
    except Exception:
        return -1e9


def _coord_ascent(
    seed:            pg.Policy,
    paths:           np.ndarray,
    surface:         FrozenSurface,
    spot0:           float,
    delta_floor:     float,
    cost_per_leg:    float,
    frozen_expiries: list | None = None,
    rng:             np.random.Generator | None = None,
) -> tuple:
    """
    Single-start coordinate ascent.  Cycles through all params until no improvement.
    Coordinate order is shuffled on every outer pass (randomised coordinate descent)
    so the local optimum found does not depend on the fixed alphabetical scan order.
    Returns (best_policy, best_score).
    """
    current       = seed
    current_score = _evaluate(current, paths, surface, spot0, delta_floor, cost_per_leg,
                              frozen_expiries=frozen_expiries)

    improved = True
    while improved:
        improved = False
        params_order = list(pg.PARAMS)          # fresh copy each pass
        if rng is not None:
            rng.shuffle(params_order)           # randomise scan order
        for param in params_order:
            best_score    = current_score
            best_neighbor = current
            for neighbor in pg.neighbors(current, param):
                if not pg.is_feasible(neighbor, spot0, surface):
                    continue
                s = _evaluate(neighbor, paths, surface, spot0, delta_floor, cost_per_leg,
                              frozen_expiries=frozen_expiries)
                if s > best_score + 1e-6:
                    best_score    = s
                    best_neighbor = neighbor
            if best_neighbor is not current:
                current       = best_neighbor
                current_score = best_score
                improved      = True

    return current, current_score


def run_optimization(
    surface:         FrozenSurface,
    spot0:           float,
    paths_search:    np.ndarray,    # (N_SEARCH, H+1)  CRN paths for phases 1+2
    paths_fine:      np.ndarray,    # (N_FINE,   H+1)  fresh paths for phase 3
    delta_floor:     float = 0.80,
    cost_per_leg:    float = 1.0,
    n_init:          int   = 50,    # random policies in Phase 1
    k_starts:        int   = 15,    # best seeds for Phase 2 coord ascent
    min_p_success:   float = 0.90,  # Phase 3 filter threshold
    rng_seed:        int   = 0,
    frozen_expiries: list | None = None,
) -> list[dict]:
    """
    Returns list of result dicts sorted by score (best first).
    Each dict: policy, P_success, E_W, CVaR_W, P_W_positive,
               E_breach_depth, n_rolls, W_final, score.
    """
    rng = np.random.default_rng(rng_seed)

    # ── Phase 1: Random init ──────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Phase 1 -- Random init  ({n_init} random policies, CRN search paths)")
    print(f"{'─'*62}")

    phase1 = []
    for _ in tqdm(range(n_init), desc="Phase 1", unit="policy"):
        for _attempt in range(30):        # retry until feasible
            p = pg.random_policy(rng)
            if pg.is_feasible(p, spot0, surface):
                break
        else:
            continue
        s = _evaluate(p, paths_search, surface, spot0, delta_floor, cost_per_leg,
                      frozen_expiries=frozen_expiries)
        phase1.append((p, s))

    phase1.sort(key=lambda x: x[1], reverse=True)
    best_p1_score, best_p1_ps = phase1[0][1], 0.0
    if phase1:
        m0 = simulate_policy(phase1[0][0], paths_search, surface, spot0, delta_floor, cost_per_leg,
                             frozen_expiries=frozen_expiries)
        best_p1_ps = m0["P_success"]
        best_p1_breach = m0["E_breach_depth"]
    print(f"Phase 1 complete: {len(phase1)} evaluated  -> top {k_starts} seeds for Phase 2")
    print(f"  Best Phase-1 score: {phase1[0][1]:+.4f}  "
          f"(P_success={best_p1_ps*100:.1f}%  E_breach=${best_p1_breach:.2f})")

    seeds = [p for p, _ in phase1[:k_starts]]

    # ── Phase 2: Coordinate ascent ────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Phase 2 -- Coordinate ascent  ({len(seeds)} starts, CRN)")
    print(f"{'─'*62}")

    phase2 = []
    for seed in tqdm(seeds, desc="Phase 2", unit="start"):
        local_opt, local_score = _coord_ascent(
            seed, paths_search, surface, spot0, delta_floor, cost_per_leg,
            frozen_expiries=frozen_expiries,
            rng=rng,                            # shared rng → shuffled coord order each pass
        )
        phase2.append((local_opt, local_score))

    # Deduplicate by policy identity
    seen:   set  = set()
    unique: list = []
    for p, s in sorted(phase2, key=lambda x: x[1], reverse=True):
        if p not in seen:
            seen.add(p)
            unique.append((p, s))

    phase2_top = unique[:k_starts]
    m2 = simulate_policy(phase2_top[0][0], paths_search, surface, spot0, delta_floor, cost_per_leg,
                         frozen_expiries=frozen_expiries)
    print(f"Phase 2 complete: {len(unique)} unique local optima  -> top {len(phase2_top)} for Phase 3")
    print(f"  Best Phase-2 score: {phase2_top[0][1]:+.4f}  "
          f"(P_success={m2['P_success']*100:.1f}%  E_breach=${m2['E_breach_depth']:.2f})")

    # ── Phase 3: Fine verification ────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Phase 3 -- Fine verification  "
          f"({paths_fine.shape[0]:,} paths x {paths_fine.shape[1]-1:d}d, "
          f"{len(phase2_top)} candidates)")
    print(f"{'─'*62}")

    fine_results: list[dict] = []
    for p, _ in tqdm(phase2_top, desc="Phase 3", unit="policy"):
        m = simulate_policy(p, paths_fine, surface, spot0, delta_floor, cost_per_leg,
                            frozen_expiries=frozen_expiries)
        fine_results.append({
            "policy":          p,
            "P_success":       m["P_success"],
            "E_W":             m["E_W"],
            "CVaR_W":          m["CVaR_W"],
            "P_W_positive":    m["P_W_positive"],
            "E_breach_depth":  m["E_breach_depth"],
            "n_rolls":         m["n_rolls"],
            "W_final":         m["W_final"],
            "score":           _score(m, spot0),
        })

    valid = [r for r in fine_results if r["P_success"] >= min_p_success]
    valid.sort(key=lambda r: r["score"], reverse=True)

    n_valid = len(valid)
    print(f"Phase 3 complete: {len(fine_results)} evaluated  -> "
          f"{n_valid} meet P_success >= {min_p_success:.0%}")

    if not valid:
        print("WARNING: No policies met fine threshold. Returning all by score.")
        fine_results.sort(key=lambda r: r["score"], reverse=True)
        return fine_results

    return valid
