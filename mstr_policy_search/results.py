"""
results.py — Ranking, reporting, and file output (v4 Policy).
"""
import os
import json
import csv

import numpy as np
from tabulate import tabulate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def report(
    ranked:     list[dict],
    spot:       float,
    delta_floor: float = 0.80,
    lambda_:    float  = 1.0,
    benchmarks: dict   = None,   # optional dict with stop-loss and BH metrics
) -> None:
    """Print top-10 table, best policy detail, benchmark comparison; save files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not ranked:
        print("No results to report.")
        return

    # ── Top 10 table ──────────────────────────────────────────────────────────
    top10 = ranked[:10]
    table_rows = []
    for i, r in enumerate(top10):
        p = r["policy"]
        table_rows.append([
            i + 1,
            f"{p.alpha_L:.2f}",
            f"{p.alpha_S1:.2f}",
            p.T_L, p.T_S1,
            p.base_q1,
            f"{p.beta:.2f}",
            p.d_min_short,
            f"{r['P_success']*100:.1f}%",
            f"{r['CVaR_20_improvement']:+.2f}",
            f"{r['E_drag']:.2f}",
            f"{r['score']:+.4f}",
        ])

    headers = [
        "Rank", "aL", "aS1", "T_L", "T_S1", "q1",
        "beta", "d_S",
        "P_succ", "CVaR20_imp", "E_drag", "Score",
    ]
    print("\n" + "=" * 110)
    print(f"TOP 10 POLICIES  (Score = CVaR_20_improvement - {lambda_:.1f} x E_drag)")
    print("=" * 110)
    print(tabulate(table_rows, headers=headers, tablefmt="simple"))

    # ── Benchmark comparison ──────────────────────────────────────────────────
    if benchmarks:
        sl  = benchmarks.get("stop_loss", {})
        print("\n" + "=" * 70)
        print("  BENCHMARK COMPARISON  (same 2000 fine paths, terminal wealth at 504d)")
        print("=" * 70)
        print(f"  {'Strategy':<28}  {'CVaR_20_imp':>12}  {'E_drag':>10}  {'E_imp':>10}  {'n_trades':>9}")
        print(f"  {'-'*28}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*9}")
        # Buy-and-hold (improvement = 0 by definition)
        print(f"  {'Buy-and-hold':<28}  {'0.00':>12}  {'0.00':>10}  {'0.00':>10}  {'--':>9}")
        # Stop-loss
        print(f"  {'Stop-loss (sell<floor, buy>floor)':<28}  "
              f"{sl.get('CVaR_20_imp', 0):>+12.2f}  "
              f"{sl.get('E_drag', 0):>10.2f}  "
              f"{sl.get('E_improvement', 0):>+10.2f}  "
              f"{sl.get('n_trades', 0):>9.1f}")
        # Best options policy
        best = ranked[0]
        print(f"  {'Options overlay (best policy)':<28}  "
              f"{best['CVaR_20_improvement']:>+12.2f}  "
              f"{best['E_drag']:>10.2f}  "
              f"{float(best['improvement'].mean()):>+10.2f}  "
              f"{best['n_rolls']:>9.1f}")
        print()
        print("  CVaR_20_imp : mean improvement over BH in the 20% of paths with lowest S_T (crash paths)")
        print("  E_drag      : mean loss vs BH in the top 80% of paths (bull paths)")
        print("  E_imp       : unconditional mean improvement over BH across all paths")
        print("  n_trades    : mean round-trip trades (options = roll events)")
        print("=" * 70)

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
          f"(at spot={spot:.0f}: K={spot*p.alpha_L:.0f})")
    print(f"    Short put:   K = spot x {p.alpha_S1:.2f}  "
          f"(at spot={spot:.0f}: K={spot*p.alpha_S1:.0f})")
    print(f"    Quantity:    q1 = {p.base_q1}")
    print(f"    Long expiry:  {p.T_L} trading days")
    print(f"    Short expiry: {p.T_S1} trading days")
    print()
    print(f"    Floor: {floor_desc}")
    print(f"    Cost:  {'self-funded at inception' if p.eta_pct == 0 else f'net >= {p.eta_pct:+.0%} x spot'}")
    print()
    print("  Terminal improvement over buy-and-hold (Phase-3 fine paths):")
    print(f"    Score       CVaR_20 - {lambda_:.0f}xE_drag:   {best['score']:+.4f}")
    print(f"    CVaR_20_imp (crash paths, bottom 20%):  {best['CVaR_20_improvement']:+.2f}")
    print(f"    E_drag      (bull paths, top 80%):      {best['E_drag']:.2f}")
    print(f"    E_imp       (unconditional mean):       {float(best['improvement'].mean()):+.2f}")
    print()
    print("  Option cash-flow metrics (W only, stock excluded):")
    print(f"    E[W]        (net option cash):          {best['E_W']:+.2f}")
    print(f"    CVaR[W]     (worst 10% W):              {best['CVaR_W']:+.2f}")
    print(f"    P(W>0):                                 {best['P_W_positive']*100:.1f}%")
    print(f"    Mean rolls / path:                      {best['n_rolls']:.1f}")
    print()
    print("  Daily floor-breach metrics (reference only, not in score):")
    print(f"    P_success   (floor never breached):     {best['P_success']*100:.1f}%")
    print(f"    E[breach]   (mean depth when breached): {best['E_breach_depth']:.2f}")
    print()
    print("  Operating rule at any future roll event t:")
    print("    Given S(t), H(t), W(t):")
    print()
    print("    1. Target strikes:")
    print(f"       K_L  = S(t) x {p.alpha_L:.2f}  -> snap to nearest liquid (spread/mid <= 25%)")
    print(f"       K_S1 = S(t) x {p.alpha_S1:.2f}  -> snap to nearest liquid (spread/mid <= 25%)")
    print()
    print(f"    2. Quantity (fixed):  q1 = {p.base_q1}")
    print()
    print(f"    3. Floor reference:  {floor_desc}")
    print()
    print(f"    4. Roll triggers (per-leg, checked daily):")
    print(f"       SHORT: T_S1_rem <= {p.d_min_short}d  -> roll short only")
    print(f"       LONG : T_L_rem  <= {p.d_min_long}d  -> roll long  only")
    print(f"       FLOOR: K_L < Floor AND S x {p.alpha_L:.2f} > K_L AND cooldown >= {p.d_min_long}d"
          f"  -> roll long only")
    print()
    print("    5. Transaction:")
    print("       Close at BID (long) / ASK (short buy-back)")
    print("       Open  at ASK (long) / BID (short sell)")
    print("       Deduct 1 USD transaction cost per contract leg")
    print("=" * 60)

    # ── Save files ────────────────────────────────────────────────────────────
    _save_top_policies_csv(ranked, spot)
    _save_best_policy_json(best, spot, delta_floor, lambda_)
    _save_terminal_distributions(best, benchmarks)
    print(f"\nResults saved to {RESULTS_DIR}/")


def _save_top_policies_csv(ranked: list[dict], spot: float) -> None:
    path = os.path.join(RESULTS_DIR, "top_policies.csv")
    fieldnames = [
        "rank", "alpha_L", "alpha_S1", "T_L", "T_S1", "base_q1",
        "beta", "d_min_short", "d_min_long", "eta_pct",
        "P_success", "E_W", "CVaR_W", "P_W_positive",
        "E_breach_depth", "n_rolls",
        "CVaR_20_improvement", "E_drag", "score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(ranked):
            p = r["policy"]
            w.writerow({
                "rank":                i + 1,
                "alpha_L":             p.alpha_L,
                "alpha_S1":            p.alpha_S1,
                "T_L":                 p.T_L,
                "T_S1":                p.T_S1,
                "base_q1":             p.base_q1,
                "beta":                p.beta,
                "d_min_short":         p.d_min_short,
                "d_min_long":          p.d_min_long,
                "eta_pct":             p.eta_pct,
                "P_success":           round(r["P_success"],           4),
                "E_W":                 round(r["E_W"],                 4),
                "CVaR_W":              round(r["CVaR_W"],              4),
                "P_W_positive":        round(r["P_W_positive"],        4),
                "E_breach_depth":      round(r["E_breach_depth"],      4),
                "n_rolls":             round(r["n_rolls"],             2),
                "CVaR_20_improvement": round(r["CVaR_20_improvement"], 4),
                "E_drag":              round(r["E_drag"],              4),
                "score":               round(r["score"],               4),
            })


def _save_best_policy_json(best: dict, spot: float, delta_floor: float,
                           lambda_: float = 1.0) -> None:
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
            "P_success":           round(best["P_success"],           4),
            "E_W":                 round(best["E_W"],                 4),
            "CVaR_W":              round(best["CVaR_W"],              4),
            "P_W_positive":        round(best["P_W_positive"],        4),
            "E_breach_depth":      round(best["E_breach_depth"],      4),
            "n_rolls":             round(best["n_rolls"],             2),
            "CVaR_20_improvement": round(best["CVaR_20_improvement"], 4),
            "E_drag":              round(best["E_drag"],              4),
            "score":               round(best["score"],              4),
        },
        "optimizer_params": {
            "delta_floor": delta_floor,
            "lambda":      lambda_,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_terminal_distributions(best: dict, benchmarks: dict = None) -> None:
    """Save per-path terminal improvement for options and benchmarks."""
    # Options improvement (W + Pi_T)
    imp = best["improvement"]
    W   = best["W_final"]
    path_csv = os.path.join(RESULTS_DIR, "path_distribution.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["path_idx", "W_final", "improvement"]
        if benchmarks and "stop_loss" in benchmarks:
            header.append("sl_improvement")
        w.writerow(header)
        sl_imp = benchmarks["stop_loss"]["improvement"] if (benchmarks and "stop_loss" in benchmarks) else None
        for i in range(len(imp)):
            row = [i, round(float(W[i]), 4), round(float(imp[i]), 4)]
            if sl_imp is not None:
                row.append(round(float(sl_imp[i]), 4))
            w.writerow(row)

    np.save(os.path.join(RESULTS_DIR, "W_final_best.npy"), W)
    np.save(os.path.join(RESULTS_DIR, "improvement_best.npy"), imp)
    if benchmarks and "stop_loss" in benchmarks:
        np.save(os.path.join(RESULTS_DIR, "improvement_stoploss.npy"),
                benchmarks["stop_loss"]["improvement"])
