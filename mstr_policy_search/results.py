"""
results.py — Ranking, reporting, and file output.
"""
import os
import json
import csv

import numpy as np
from tabulate import tabulate


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def report(ranked: list[dict], spot: float) -> None:
    """Print top-10 table and best policy detail; save output files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not ranked:
        print("No results to report.")
        return

    # ── Top 10 table ─────────────────────────────────────────────────────────
    top10 = ranked[:10]
    table_rows = []
    for i, r in enumerate(top10):
        p = r["policy"]
        table_rows.append([
            i + 1,
            f"{p.alpha_L:.2f}",
            f"{p.alpha_S1:.2f}",
            f"{p.alpha_S2:.2f}" if p.alpha_S2 > 0 else "—",
            p.T_L, p.T_S1,
            p.T_S2 if p.alpha_S2 > 0 else "—",
            p.base_q1, p.base_q2 if p.alpha_S2 > 0 else "—",
            f"{p.gamma:.1f}", p.d_min,
            f"{r['P_success']*100:.1f}%",
            f"{r['E_W']:+.2f}",
            f"{r['CVaR_W']:+.2f}",
            f"{r['score']:+.2f}",
        ])

    headers = [
        "Rank", "α_L", "α_S1", "α_S2", "T_L", "T_S1", "T_S2",
        "q1", "q2", "γ", "d_min",
        "P_succ", "E[W]", "CVaR_W", "Score",
    ]
    print("\n" + "=" * 100)
    print("TOP 10 POLICIES")
    print("=" * 100)
    print(tabulate(table_rows, headers=headers, tablefmt="simple"))

    # ── Best policy detail ───────────────────────────────────────────────────
    best = ranked[0]
    p    = best["policy"]
    print("\n" + "═" * 55)
    print("  OPTIMAL POLICY π*")
    print("═" * 55)
    print("  Structure:")
    print(f"    Long put:    K = spot × {p.alpha_L:.2f}  (e.g., at spot=${spot:.0f}: K=${spot*p.alpha_L:.0f})")
    print(f"    Short put 1: K = spot × {p.alpha_S1:.2f}  (e.g., at spot=${spot:.0f}: K=${spot*p.alpha_S1:.0f})")
    if p.alpha_S2 > 0:
        print(f"    Short put 2: K = spot × {p.alpha_S2:.2f}  (e.g., at spot=${spot:.0f}: K=${spot*p.alpha_S2:.0f})")
    print()
    print(f"    Long put expiry:    {p.T_L} days")
    print(f"    Short put 1 expiry: {p.T_S1} days")
    if p.alpha_S2 > 0:
        print(f"    Short put 2 expiry: {p.T_S2} days")
    print()
    print(f"    Base quantities:    q1={p.base_q1}" + (f", q2={p.base_q2}" if p.alpha_S2 > 0 else ""))
    print(f"    Self-funding γ:     {p.gamma}")
    print(f"    Calendar roll at:   {p.d_min} days to expiry")
    print()
    print(f"  Performance (paths, 2-year horizon):")
    print(f"    Floor protection:   {best['P_success']*100:.1f}% of paths never breached")
    print(f"    Expected net W:    {best['E_W']:+.2f}")
    print(f"    P(W > 0):          {best['P_W_positive']*100:.1f}%")
    print(f"    CVaR W (worst 10%): {best['CVaR_W']:+.2f}")
    print(f"    Mean rolls/path:    {best['n_rolls']:.1f}")
    print(f"    Mean breach depth:  ${best['E_breach_depth']:.2f} (when breached)")
    print()
    print("  Operating rule at any future roll event τ:")
    print("    Given S(τ), H(τ), W(τ):")
    print()
    print(f"    1. Target strikes:")
    print(f"       K_L   = S(τ) × {p.alpha_L:.2f}  → snap to nearest liquid ≤ 15% spread")
    print(f"       K_S1  = S(τ) × {p.alpha_S1:.2f}  → snap to nearest liquid ≤ 15% spread")
    if p.alpha_S2 > 0:
        print(f"       K_S2  = S(τ) × {p.alpha_S2:.2f}  → snap to nearest liquid ≤ 15% spread")
    print()
    print(f"    2. Adaptive quantities:")
    print(f"       deficit = max(0, −W(τ))")
    print(f"       q1 = floor({p.base_q1} + {p.gamma} × deficit / P_S1),  capped by C3 limit")
    if p.alpha_S2 > 0:
        print(f"       q2 = floor({p.base_q2} + {p.gamma} × deficit / P_S2),  capped by C3 limit")
    print()
    print("    3. Roll triggers (check daily):")
    print(f"       CALENDAR: any short leg ≤ {p.d_min} days left?  → roll short legs")
    print(f"       FLOOR:    K_L < H(τ) × {1-0.30:.2f}?              → roll long put")
    print(f"       LIQUIDITY: any leg spread% > 15%?           → roll immediately")
    print()
    print("    4. Transaction:")
    print("       Close all triggered legs at BID")
    print("       Open new legs at ASK")
    print("       Record net cash flow → update W(τ)")
    print("═" * 55)

    # ── Save files ────────────────────────────────────────────────────────────
    _save_top_policies_csv(ranked, spot)
    _save_best_policy_json(best, spot)
    _save_w_final_distribution(best)
    print(f"\nResults saved to {RESULTS_DIR}/")


def _save_top_policies_csv(ranked: list[dict], spot: float) -> None:
    path = os.path.join(RESULTS_DIR, "top_policies.csv")
    fieldnames = [
        "rank", "alpha_L", "alpha_S1", "alpha_S2",
        "T_L", "T_S1", "T_S2", "base_q1", "base_q2",
        "gamma", "d_min",
        "P_success", "E_W", "CVaR_W", "P_W_positive",
        "E_breach_depth", "n_rolls", "score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(ranked):
            p = r["policy"]
            w.writerow({
                "rank":          i + 1,
                "alpha_L":       p.alpha_L,
                "alpha_S1":      p.alpha_S1,
                "alpha_S2":      p.alpha_S2,
                "T_L":           p.T_L,
                "T_S1":          p.T_S1,
                "T_S2":          p.T_S2,
                "base_q1":       p.base_q1,
                "base_q2":       p.base_q2,
                "gamma":         p.gamma,
                "d_min":         p.d_min,
                "P_success":     round(r["P_success"], 4),
                "E_W":           round(r["E_W"], 4),
                "CVaR_W":        round(r["CVaR_W"], 4),
                "P_W_positive":  round(r["P_W_positive"], 4),
                "E_breach_depth":round(r["E_breach_depth"], 4),
                "n_rolls":       round(r["n_rolls"], 2),
                "score":         round(r["score"], 4),
            })


def _save_best_policy_json(best: dict, spot: float) -> None:
    path = os.path.join(RESULTS_DIR, "best_policy.json")
    p = best["policy"]
    data = {
        "policy": {
            "alpha_L":  p.alpha_L,
            "alpha_S1": p.alpha_S1,
            "alpha_S2": p.alpha_S2,
            "T_L":      p.T_L,
            "T_S1":     p.T_S1,
            "T_S2":     p.T_S2,
            "base_q1":  p.base_q1,
            "base_q2":  p.base_q2,
            "gamma":    p.gamma,
            "d_min":    p.d_min,
        },
        "strikes_at_current_spot": {
            "K_L":  round(spot * p.alpha_L,  2),
            "K_S1": round(spot * p.alpha_S1, 2),
            "K_S2": round(spot * p.alpha_S2, 2) if p.alpha_S2 > 0 else None,
        },
        "metrics": {
            "P_success":      round(best["P_success"],      4),
            "E_W":            round(best["E_W"],            4),
            "CVaR_W":         round(best["CVaR_W"],         4),
            "P_W_positive":   round(best["P_W_positive"],   4),
            "E_breach_depth": round(best["E_breach_depth"], 4),
            "n_rolls":        round(best["n_rolls"],        2),
            "score":          round(best["score"],          4),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_w_final_distribution(best: dict) -> None:
    """Save W_final array for the best policy (for RL baseline comparison)."""
    path = os.path.join(RESULTS_DIR, "path_distribution.csv")
    W = best["W_final"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path_idx", "W_final"])
        for i, v in enumerate(W):
            w.writerow([i, round(float(v), 4)])

    # Also save as numpy for easy loading
    np.save(os.path.join(RESULTS_DIR, "W_final_best.npy"), W)
