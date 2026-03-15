#!/usr/bin/env python3
"""
main_moneyness.py -- Optimize moneyness-based rolling triggers.

Loads the best calendar policy from the specified drift scenario, fixes all
9 base parameters, then searches for the optimal pair:

  theta_lo_L : roll long  put when K_L/S(t)  < theta_lo_L  (refreshes OTM protection)
  theta_hi_S : roll short put when K_S1/S(t) > theta_hi_S  (closes ITM exposure early)

Procedure
---------
1. Load saved best policy for the target drift scenario.
2. Re-simulate the calendar baseline on fresh fine paths (benchmark).
3. Grid search over THETA_LO_L_VALUES x THETA_HI_S_VALUES on CRN search paths.
4. Fine-verify the top N_FINE_TOP grid combos on fresh fine paths.
5. Print ranked table and delta-vs-baseline summary.
6. Save best moneyness policy to results/best_money_policy{tag}.json.

Usage
-----
    cd mstr_policy_search
    python main_moneyness.py                  # drift=-0.002 (default)
    python main_moneyness.py --drift -0.002   # explicit drift
    python main_moneyness.py --drift 0.0      # raw BTC-era scenario
    python main_moneyness.py --sanity         # verbose 10-path trace, no full run
    python main_moneyness.py --refresh        # force live yfinance fetch
"""
import sys
import os
import json

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import data_loader
from surface import build_surface
from bootstrap import BootstrapSampler
from simulator import simulate_policy
from simulator_moneyness import simulate_money_policy
from policy_grid import Policy
from policy_moneyness import MoneyPolicy, THETA_LO_L_VALUES, THETA_HI_S_VALUES

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration (must match the main.py settings used to generate saved policies)
# ══════════════════════════════════════════════════════════════════════════════
HORIZON_DAYS        = 504
BTC_ERA_ONLY        = True
BTC_ERA_CUTOFF      = "2020-08-11"
DELTA_FLOOR         = 0.80
ANNUAL_DEFAULT_PROB = 0.001
COST_PER_LEG        = 1.0
LAMBDA              = 1.0
BLOCK_SIZE          = 10

# Drift scenario to use (can be overridden by --drift CLI arg)
DEFAULT_DRIFT       = -0.002

# Path counts
N_SEARCH     = 500        # CRN paths for grid search (fast, biased but consistent)
N_FINE       = 2_000      # fresh paths for fine verification
N_FINE_TOP   = 10         # how many top grid combos to re-evaluate on fine paths
# ══════════════════════════════════════════════════════════════════════════════


def _drift_tag(drift: float) -> str:
    return f"_drift_{drift*100:+.2f}pct"


def _load_best_policy(drift: float) -> Policy:
    """Load the saved best calendar policy for a given drift scenario."""
    tag         = _drift_tag(drift)
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    json_path   = os.path.join(results_dir, f"best_policy{tag}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No saved policy found at {json_path}\n"
            f"Run main.py with DRIFT_ADJ_VALUES=[{drift}] first."
        )
    with open(json_path) as f:
        data = json.load(f)
    p = data["policy"]
    return Policy(
        alpha_L     = float(p["alpha_L"]),
        alpha_S1    = float(p["alpha_S1"]),
        T_L         = int(p["T_L"]),
        T_S1        = int(p["T_S1"]),
        base_q1     = int(p["base_q1"]),
        beta        = float(p["beta"]),
        d_min_short = int(p["d_min_short"]),
        d_min_long  = int(p["d_min_long"]),
        eta_pct     = float(p["eta_pct"]),
    )


def _score(m: dict, lambda_: float = 1.0) -> float:
    return m["CVaR_20_improvement"] - lambda_ * m["E_drag"]


def _tabulate(rows: list, headers: list, col_width: int = 12) -> str:
    """Minimal tabulate implementation without the tabulate package."""
    lines = []
    fmts  = [f"{{:<{col_width}}}"] * len(headers)
    sep   = "  ".join(fmts)
    lines.append(sep.format(*headers))
    lines.append("-" * (col_width * len(headers) + 2 * (len(headers) - 1)))
    for row in rows:
        lines.append(sep.format(*[str(c) for c in row]))
    return "\n".join(lines)


def main() -> None:
    sanity_only = "--sanity"  in sys.argv
    refresh     = "--refresh" in sys.argv

    # Parse --drift
    drift = DEFAULT_DRIFT
    if "--drift" in sys.argv:
        idx   = sys.argv.index("--drift")
        drift = float(sys.argv[idx + 1])

    tag = _drift_tag(drift)

    print(f"Moneyness trigger search -- drift={drift*100:+.4f}%/day ({drift*252*100:+.1f}%/yr)")
    print(f"  theta_lo_L grid ({len(THETA_LO_L_VALUES)}): {THETA_LO_L_VALUES}")
    print(f"  theta_hi_S grid ({len(THETA_HI_S_VALUES)}): {THETA_HI_S_VALUES}")
    print(f"  N_search={N_SEARCH}  N_fine={N_FINE}  N_fine_top={N_FINE_TOP}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    chain, spot, _ = data_loader.load_option_chain("MSTR", refresh=refresh)
    returns = data_loader.load_historical_returns(
        "MSTR", refresh=refresh,
        cutoff_date=BTC_ERA_CUTOFF if BTC_ERA_ONLY else None,
    )
    surface = build_surface(chain, spot)
    print(f"\n  spot=${spot}  return_pool={len(returns)}d")

    frozen_expiries = sorted(
        set(int(c["days_out"]) for c in chain if c.get("spread_pct", 100) <= 25.0)
    )

    # ── 2. Load best calendar policy ─────────────────────────────────────────
    base_policy = _load_best_policy(drift)
    print(f"\nCalendar baseline policy (from best_policy{tag}.json):")
    print(f"  alpha_L={base_policy.alpha_L}  alpha_S1={base_policy.alpha_S1}  "
          f"T_L={base_policy.T_L}  T_S1={base_policy.T_S1}  q1={base_policy.base_q1}")
    print(f"  beta={base_policy.beta}  d_min_short={base_policy.d_min_short}  "
          f"d_min_long={base_policy.d_min_long}  eta_pct={base_policy.eta_pct}")

    # ── 3. Generate paths (once, shared by all evaluations) ──────────────────
    print(f"\nGenerating paths ...")
    sampler_search = BootstrapSampler(returns, seed=42)
    sampler_fine   = BootstrapSampler(returns, seed=137)
    paths_search = sampler_search.sample_paths(
        N_SEARCH, HORIZON_DAYS, spot,
        annual_default_prob=ANNUAL_DEFAULT_PROB,
        block_size=BLOCK_SIZE, drift_adj=drift,
    )
    paths_fine = sampler_fine.sample_paths(
        N_FINE, HORIZON_DAYS, spot,
        annual_default_prob=ANNUAL_DEFAULT_PROB,
        block_size=BLOCK_SIZE, drift_adj=drift,
    )
    print(f"  Search: {paths_search.shape}  Fine: {paths_fine.shape}")

    # ── 4. Calendar baseline (fine paths) ─────────────────────────────────────
    m_cal     = simulate_policy(base_policy, paths_fine, surface, spot,
                                delta_floor=DELTA_FLOOR, cost_per_leg=COST_PER_LEG,
                                frozen_expiries=frozen_expiries)
    score_cal = _score(m_cal, LAMBDA)
    print(f"\nCalendar baseline (fine paths, N={N_FINE:,}):")
    print(f"  score={score_cal:+.4f}  CVaR_20={m_cal['CVaR_20_improvement']:+.2f}  "
          f"E_drag={m_cal['E_drag']:.2f}  P_success={m_cal['P_success']*100:.1f}%  "
          f"n_rolls={m_cal['n_rolls']:.1f}")

    # ── 5. Sanity check (optional) ────────────────────────────────────────────
    if sanity_only:
        test_tlo, test_thi = 0.95, 0.80
        mp = MoneyPolicy(**base_policy._asdict(), theta_lo_L=test_tlo, theta_hi_S=test_thi)
        print(f"\nSanity trace: theta_lo_L={test_tlo}  theta_hi_S={test_thi}  (10 paths)")
        simulate_money_policy(
            mp, paths_search[:10], surface, spot,
            delta_floor=DELTA_FLOOR, cost_per_leg=COST_PER_LEG,
            verbose=True, frozen_expiries=frozen_expiries,
        )
        return

    # ── 6. Grid search (CRN search paths) ────────────────────────────────────
    n_combos = len(THETA_LO_L_VALUES) * len(THETA_HI_S_VALUES)
    print(f"\nGrid search: {len(THETA_LO_L_VALUES)} x {len(THETA_HI_S_VALUES)} "
          f"= {n_combos} combos on {N_SEARCH} CRN paths ...")
    print(f"  {'theta_lo_L':>10}  {'theta_hi_S':>10}  {'score':>8}  "
          f"{'CVaR_20':>8}  {'E_drag':>7}  {'P_succ':>7}  {'n_rolls':>7}")
    print(f"  {'-'*72}")

    grid_results = []
    for tlo in THETA_LO_L_VALUES:
        for thi in THETA_HI_S_VALUES:
            mp = MoneyPolicy(**base_policy._asdict(), theta_lo_L=tlo, theta_hi_S=thi)
            m  = simulate_money_policy(
                mp, paths_search, surface, spot,
                delta_floor=DELTA_FLOOR, cost_per_leg=COST_PER_LEG,
                frozen_expiries=frozen_expiries,
            )
            s = _score(m, LAMBDA)
            grid_results.append({
                "theta_lo_L": tlo,
                "theta_hi_S": thi,
                "score":      s,
                "CVaR_20":    m["CVaR_20_improvement"],
                "E_drag":     m["E_drag"],
                "P_success":  m["P_success"],
                "n_rolls":    m["n_rolls"],
            })
            print(f"  {tlo:>10.2f}  {thi:>10.2f}  {s:>+8.2f}  "
                  f"{m['CVaR_20_improvement']:>+8.2f}  {m['E_drag']:>7.2f}  "
                  f"{m['P_success']*100:>6.1f}%  {m['n_rolls']:>7.1f}")

    grid_results.sort(key=lambda r: r["score"], reverse=True)
    best_g = grid_results[0]
    print(f"\nBest grid combo: theta_lo_L={best_g['theta_lo_L']:.2f}  "
          f"theta_hi_S={best_g['theta_hi_S']:.2f}  "
          f"search_score={best_g['score']:+.4f}")

    # ── 7. Fine verification (top N_FINE_TOP combos) ──────────────────────────
    top_combos = grid_results[:N_FINE_TOP]
    print(f"\nFine verification: top {len(top_combos)} combos on {N_FINE:,} fresh paths ...")
    fine_results = []
    for gr in top_combos:
        mp = MoneyPolicy(**base_policy._asdict(),
                         theta_lo_L=gr["theta_lo_L"], theta_hi_S=gr["theta_hi_S"])
        m  = simulate_money_policy(
            mp, paths_fine, surface, spot,
            delta_floor=DELTA_FLOOR, cost_per_leg=COST_PER_LEG,
            frozen_expiries=frozen_expiries,
        )
        s = _score(m, LAMBDA)
        fine_results.append({
            "theta_lo_L": gr["theta_lo_L"],
            "theta_hi_S": gr["theta_hi_S"],
            "score":      s,
            "CVaR_20":    m["CVaR_20_improvement"],
            "E_drag":     m["E_drag"],
            "P_success":  m["P_success"],
            "n_rolls":    m["n_rolls"],
        })

    fine_results.sort(key=lambda r: r["score"], reverse=True)
    best_f = fine_results[0]

    # ── 8. Print results table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"MONEYNESS TRIGGER RESULTS  (drift={drift*100:+.2f}%/day, N_fine={N_FINE:,})")
    print(f"{'='*80}")

    rows = []
    for r in fine_results:
        delta = r["score"] - score_cal
        rows.append([
            f"{r['theta_lo_L']:.2f}",
            f"{r['theta_hi_S']:.2f}",
            f"{r['score']:+.2f}",
            f"{r['CVaR_20']:+.2f}",
            f"{r['E_drag']:.2f}",
            f"{r['P_success']*100:.1f}%",
            f"{r['n_rolls']:.1f}",
            f"{delta:+.2f}",
        ])
    # Calendar baseline row (last, for comparison)
    rows.append([
        "calendar", "calendar",
        f"{score_cal:+.2f}",
        f"{m_cal['CVaR_20_improvement']:+.2f}",
        f"{m_cal['E_drag']:.2f}",
        f"{m_cal['P_success']*100:.1f}%",
        f"{m_cal['n_rolls']:.1f}",
        "0.00",
    ])
    headers = ["theta_lo_L", "theta_hi_S", "Score", "CVaR_20_imp",
               "E_drag", "P_success", "n_rolls", "vs_cal"]
    print(_tabulate(rows, headers, col_width=11))

    delta_best = best_f["score"] - score_cal
    sign = "+" if delta_best >= 0 else ""
    print(f"\nBest moneyness vs calendar baseline: {sign}{delta_best:+.4f}")
    print(f"  theta_lo_L={best_f['theta_lo_L']:.2f}  theta_hi_S={best_f['theta_hi_S']:.2f}")
    print(f"  score={best_f['score']:+.4f}  CVaR_20={best_f['CVaR_20']:+.2f}  "
          f"E_drag={best_f['E_drag']:.2f}  n_rolls={best_f['n_rolls']:.1f}")
    print(f"\nInterpretation:")
    print(f"  Roll long  when K_L/S < {best_f['theta_lo_L']:.2f}  "
          f"(i.e. S rises >{(best_f['theta_lo_L']/base_policy.alpha_L - 1)*100:+.1f}% from last roll)")
    print(f"  Roll short when K_S1/S > {best_f['theta_hi_S']:.2f}  "
          f"(i.e. S drops >{(1 - base_policy.alpha_S1/best_f['theta_hi_S'])*100:+.1f}% from last roll)")

    # ── 9. Save best moneyness policy ─────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"best_money_policy{tag}.json")

    out = {
        "policy": {
            **base_policy._asdict(),
            "theta_lo_L": best_f["theta_lo_L"],
            "theta_hi_S": best_f["theta_hi_S"],
        },
        "metrics": {
            "score":               best_f["score"],
            "CVaR_20_improvement": best_f["CVaR_20"],
            "E_drag":              best_f["E_drag"],
            "P_success":           best_f["P_success"],
            "n_rolls":             best_f["n_rolls"],
        },
        "calendar_baseline": {
            "score":               score_cal,
            "CVaR_20_improvement": m_cal["CVaR_20_improvement"],
            "E_drag":              m_cal["E_drag"],
            "P_success":           m_cal["P_success"],
            "n_rolls":             m_cal["n_rolls"],
        },
        "grid_results": [
            {"theta_lo_L": r["theta_lo_L"], "theta_hi_S": r["theta_hi_S"],
             "score": r["score"], "CVaR_20": r["CVaR_20"], "E_drag": r["E_drag"]}
            for r in fine_results
        ],
    }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
