"""
results.py — Ranking, reporting, and file output (v4 Policy).
"""
import os
import json
import csv

import numpy as np
from tabulate import tabulate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def report(ranked: list[dict], spot: float, delta_floor: float = 0.80) -> None:
    """Print top-10 table and best policy detail; save output files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not ranked:
        print("No results to report.")
        return

    # ── Top 10 table ──────────────────────────────────────────────────────────
    top10 = ranked[:10]
    table_rows = []
    for i, r in enumerate(top10):
        p = r["policy"]
        net_pct = f"{p.eta_pct:+.0%}"
        table_rows.append([
            i + 1,
            f"{p.alpha_L:.2f}",
            f"{p.alpha_S1:.2f}",
            p.T_L, p.T_S1,
            p.base_q1,
            f"{p.beta:.2f}",
            p.d_min_short, p.d_min_long,
            net_pct,
            f"{r['P_success']*100:.1f}%",
            f"${r['E_breach_depth']:.2f}",
            f"{r['score']:+.4f}",
        ])

    headers = [
        "Rank", "α_L", "α_S1", "T_L", "T_S1", "q1",
        "beta", "d_S", "d_L", "cost%",
        "P_succ", "E[breach]", "Score",
    ]
    print("\n" + "=" * 110)
    print(f"TOP 10 POLICIES  (Score = P_success - E_breach_depth / spot)")
    print("=" * 110)
    print(tabulate(table_rows, headers=headers, tablefmt="simple"))

    # ── Best policy detail ────────────────────────────────────────────────────
    best = ranked[0]
    p    = best["policy"]
    floor_desc = (
        f"Fixed capital floor   Floor = {delta_floor} x S0  (no ratchet)"
        if p.beta == 0.0 else
        f"beta={p.beta:.2f}: Floor = {delta_floor} x [{1-p.beta:.2f}xS0 + {p.beta:.2f}xH(t)]"
    )

    print("\n" + "=" * 60)
    print("  OPTIMAL POLICY pi*")
    print("=" * 60)
    print("  Structure:")
    print(f"    Long put:    K = spot x {p.alpha_L:.2f}  "
          f"(at spot=${spot:.0f}: K=${spot*p.alpha_L:.0f})")
    print(f"    Short put:   K = spot x {p.alpha_S1:.2f}  "
          f"(at spot=${spot:.0f}: K=${spot*p.alpha_S1:.0f})")
    print(f"    Quantity:    q1 = {p.base_q1}")
    print(f"    Long expiry:  {p.T_L} trading days")
    print(f"    Short expiry: {p.T_S1} trading days")
    print()
    print(f"    Floor ({floor_desc})")
    print(f"    Cost constraint:      {'self-funded at inception (zero-cost)' if p.eta_pct == 0 else f'net >= {p.eta_pct:+.0%} x spot'}")
    print()
    print("  Performance (Phase-3 fine paths):")
    print(f"    P_success   (floor never breached):  {best['P_success']*100:.1f}%")
    print(f"    E[breach]   (mean depth if breached): ${best['E_breach_depth']:.2f}")
    print(f"    Score       (P_succ - breach/spot):  {best['score']:+.4f}")
    print(f"    E[W]        (net option cash):        {best['E_W']:+.2f}")
    print(f"    CVaR[W]     (worst 10% W):            {best['CVaR_W']:+.2f}")
    print(f"    P(W>0):                               {best['P_W_positive']*100:.1f}%")
    print(f"    Mean rolls / path:                    {best['n_rolls']:.1f}")
    print()
    print("  Operating rule at any future roll event t:")
    print("    Given S(t), H(t), W(t):")
    print()
    print("    1. Target strikes:")
    print(f"       K_L  = S(t) x {p.alpha_L:.2f}  -> snap to nearest liquid (spread/mid <= {int(25)}%)")
    print(f"       K_S1 = S(t) x {p.alpha_S1:.2f}  -> snap to nearest liquid (spread/mid <= {int(25)}%)")
    print()
    print(f"    2. Quantity (fixed):")
    print(f"       q1 = {p.base_q1}")
    print()
    print(f"    3. Floor reference:")
    print(f"       {floor_desc}")
    print()
    print(f"    4. Roll triggers (per-leg, checked daily):")
    print(f"       SHORT: T_S1_rem <= {p.d_min_short}d  -> roll short only")
    print(f"       LONG : T_L_rem  <= {p.d_min_long}d  -> roll long  only")
    print(f"       FLOOR: K_L < Floor AND S x {p.alpha_L:.2f} > K_L AND cooldown >= {p.d_min_long}d"
          f"  -> roll long only")
    print()
    print("    5. Transaction:")
    print("       Close triggered legs at BID (long) / ASK (short buy-back)")
    print("       Open  new legs at ASK (long) / BID (short sell)")
    print("       Deduct $1 transaction cost per contract leg")
    print("=" * 60)

    # ── Save files ────────────────────────────────────────────────────────────
    _save_top_policies_csv(ranked, spot)
    _save_best_policy_json(best, spot, delta_floor)
    _save_w_final_distribution(best)
    print(f"\nResults saved to {RESULTS_DIR}/")


def _save_top_policies_csv(ranked: list[dict], spot: float) -> None:
    path = os.path.join(RESULTS_DIR, "top_policies.csv")
    fieldnames = [
        "rank", "alpha_L", "alpha_S1", "T_L", "T_S1", "base_q1",
        "beta", "d_min_short", "d_min_long", "eta_pct",
        "P_success", "E_W", "CVaR_W", "P_W_positive",
        "E_breach_depth", "n_rolls", "score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(ranked):
            p = r["policy"]
            w.writerow({
                "rank":           i + 1,
                "alpha_L":        p.alpha_L,
                "alpha_S1":       p.alpha_S1,
                "T_L":            p.T_L,
                "T_S1":           p.T_S1,
                "base_q1":        p.base_q1,
                "beta":           p.beta,
                "d_min_short":    p.d_min_short,
                "d_min_long":     p.d_min_long,
                "eta_pct":        p.eta_pct,
                "P_success":      round(r["P_success"],      4),
                "E_W":            round(r["E_W"],            4),
                "CVaR_W":         round(r["CVaR_W"],         4),
                "P_W_positive":   round(r["P_W_positive"],   4),
                "E_breach_depth": round(r["E_breach_depth"], 4),
                "n_rolls":        round(r["n_rolls"],        2),
                "score":          round(r["score"],          4),
            })


def _save_best_policy_json(best: dict, spot: float, delta_floor: float) -> None:
    path = os.path.join(RESULTS_DIR, "best_policy.json")
    p = best["policy"]
    data = {
        "policy": {
            "alpha_L":     p.alpha_L,
            "alpha_S1":    p.alpha_S1,
            "T_L":         p.T_L,
            "T_S1":        p.T_S1,
            "base_q1":     p.base_q1,
            "beta":        p.beta,
            "d_min_short": p.d_min_short,
            "d_min_long":  p.d_min_long,
            "eta_pct":     p.eta_pct,
        },
        "strikes_at_current_spot": {
            "K_L":  round(spot * p.alpha_L,  2),
            "K_S1": round(spot * p.alpha_S1, 2),
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
        "optimizer_params": {
            "delta_floor": delta_floor,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_w_final_distribution(best: dict) -> None:
    path = os.path.join(RESULTS_DIR, "path_distribution.csv")
    W = best["W_final"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path_idx", "W_final"])
        for i, v in enumerate(W):
            w.writerow([i, round(float(v), 4)])
    np.save(os.path.join(RESULTS_DIR, "W_final_best.npy"), W)
