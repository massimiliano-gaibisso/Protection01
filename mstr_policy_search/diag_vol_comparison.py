"""
diag_vol_comparison.py — Volatility distribution: historical BTC-era vs EGARCH simulation.

Four panels in one figure:
  [0,0] Histogram + KDE of rolling 21d realised vol — historical vs simulated (pooled)
  [0,1] ECDF of rolling 21d vol — historical vs simulated
  [1,0] Time-series: historical rolling 21d vol (blue) overlaid with EGARCH
        in-sample conditional sigma_t (orange) — tracks vol regimes
  [1,1] QQ-plot: historical rolling 21d vol quantiles (x) vs
        simulated rolling 21d vol quantiles (y) — diagonal = perfect fit

Run from mstr_policy_search/:
    python diag_vol_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import data_loader
from bootstrap import BootstrapSampler
from egarch import fit_egarch, _E_ABS_Z

# ── Config ────────────────────────────────────────────────────────────────────
N_PATHS    = 2_000
HORIZON    = 504        # trading days (2 years)
SEED       = 137
BTC_CUTOFF = "2020-08-11"
ROLL_WIN   = 21         # rolling vol window (trading days)
ACF_LAGS   = 40
VOL_CAP_ANN = 400.0     # %/yr — matches main.py; set None to reproduce uncapped behaviour

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
chain, spot, _ = data_loader.load_option_chain("MSTR")
returns = data_loader.load_historical_returns("MSTR", cutoff_date=BTC_CUTOFF)
N_hist = len(returns)
print(f"  BTC-era returns: N={N_hist}  mean={returns.mean()*100:+.4f}%/d  "
      f"std={returns.std()*np.sqrt(252)*100:.1f}%/yr")

# ── Fit EGARCH ────────────────────────────────────────────────────────────────
print("\nFitting EGARCH(1,1)+t5 ...")
ep = fit_egarch(returns, verbose=True)
lr_vol_ann = float(np.sqrt(np.exp(ep["omega"] / max(1.0 - ep["beta_g"], 1e-6)))
                   * np.sqrt(252) * 100)

# ── EGARCH in-sample filter: conditional sigma for historical period ───────────
def egarch_filter(r, params):
    """Return conditional daily sigma_t for each historical day (filtering step)."""
    omega  = params["omega"]
    alpha  = params["alpha"]
    gamma  = params["gamma"]
    beta_g = params["beta_g"]
    mu     = params["mu"]
    log_var0 = np.log(max(np.var(r), 1e-10))
    lsig2    = log_var0
    z_prev   = 0.0
    out = np.empty(len(r))
    for t in range(len(r)):
        lsig2 = (omega + beta_g * lsig2
                 + alpha * (abs(z_prev) - _E_ABS_Z)
                 + gamma * z_prev)
        sigma    = np.exp(0.5 * lsig2)
        out[t]   = sigma
        z_prev   = (r[t] - mu) / max(sigma, 1e-8)
    return out

print("\nRunning EGARCH filter on historical returns ...")
cond_sigma = egarch_filter(returns, ep)          # daily sigma, length N_hist
cond_vol   = cond_sigma * np.sqrt(252) * 100     # annualised %

# ── Historical rolling 21d realised vol ───────────────────────────────────────
def rolling_vol_1d(r, w):
    n  = len(r)
    rv = np.full(n, np.nan)
    for d in range(w - 1, n):
        rv[d] = r[d - w + 1: d + 1].std()
    return rv * np.sqrt(252) * 100   # annualised %

print("Computing historical rolling vol ...")
rvol_hist     = rolling_vol_1d(returns, ROLL_WIN)
rvol_hist_c   = rvol_hist[~np.isnan(rvol_hist)]   # drop NaN startup

# ── Generate simulated paths ───────────────────────────────────────────────────
print(f"\nGenerating {N_PATHS} EGARCH paths — uncapped (seed={SEED}) ...")
sampler_unc = BootstrapSampler(returns, seed=SEED)
sampler_unc.set_egarch_params(ep)
paths_unc = sampler_unc.sample_paths(
    N_PATHS, HORIZON, spot,
    annual_default_prob=0.001,
    block_size=10,
    use_egarch=True,
    vol_cap_ann=None,          # no cap — shows the super-unit-root explosions
)

print(f"Generating {N_PATHS} EGARCH paths — capped at {VOL_CAP_ANN:.0f}%/yr (seed={SEED}) ...")
sampler_cap = BootstrapSampler(returns, seed=SEED)
sampler_cap.set_egarch_params(ep)
paths_cap = sampler_cap.sample_paths(
    N_PATHS, HORIZON, spot,
    annual_default_prob=0.001,
    block_size=10,
    use_egarch=True,
    vol_cap_ann=VOL_CAP_ANN,
)

# Use capped paths as the primary simulation; keep uncapped for tail comparison
paths  = paths_cap
lr_sim = np.diff(np.log(paths), axis=1)   # (N_PATHS, HORIZON)

# ── Rolling 21d vol for capped and uncapped paths ─────────────────────────────
def all_rolling_vol(lr, w):
    """Returns (N_PATHS, HORIZON) array of annualised rolling vol, NaN for startup."""
    rv = np.full_like(lr, np.nan)
    for d in range(w - 1, lr.shape[1]):
        rv[:, d] = lr[:, d - w + 1: d + 1].std(axis=1)
    return rv * np.sqrt(252) * 100

print("Computing rolling vol for capped paths ...")
# Exclude crash paths (company-default Poisson overlay): they produce single-day
# log-returns of -900%+ (log(0.01/S_prev)) which dominate 21-day rolling vol and
# represent default events, not volatility regimes.  Report excluded count separately.
crashed_cap = (paths_cap.min(axis=1) <= 0.01)
crashed_unc = (paths_unc.min(axis=1) <= 0.01)
print(f"  Crash paths (excluded from vol stats): "
      f"capped={crashed_cap.sum()}, uncapped={crashed_unc.sum()}")

rvol_sim_ann  = all_rolling_vol(lr_sim[~crashed_cap], ROLL_WIN)
rvol_sim_flat = rvol_sim_ann[~np.isnan(rvol_sim_ann)].flatten()

print("Computing rolling vol for uncapped paths ...")
lr_unc        = np.diff(np.log(paths_unc), axis=1)
rvol_unc_ann  = all_rolling_vol(lr_unc[~crashed_unc], ROLL_WIN)
rvol_unc_flat = rvol_unc_ann[~np.isnan(rvol_unc_ann)].flatten()

# ── Summary stats ─────────────────────────────────────────────────────────────
pcts = [5, 25, 50, 75, 95, 99]
print("\nRolling 21d vol distribution:")
print(f"{'Stat':>10}  {'Historical':>12}  {'Sim uncapped':>14}  {'Sim capped':>12}")
print("-" * 56)
for p in pcts:
    h  = np.percentile(rvol_hist_c,   p)
    u  = np.percentile(rvol_unc_flat, p)
    s  = np.percentile(rvol_sim_flat, p)
    print(f"  P{p:<5d}   {h:>10.1f}%   {u:>12.1f}%   {s:>10.1f}%")
print(f"  {'Mean':>7}   {rvol_hist_c.mean():>10.1f}%   {rvol_unc_flat.mean():>12.1f}%   {rvol_sim_flat.mean():>10.1f}%")
print(f"  {'Std':>7}   {rvol_hist_c.std():>10.1f}%   {rvol_unc_flat.std():>12.1f}%   {rvol_sim_flat.std():>10.1f}%")
print(f"  {'Max':>7}   {rvol_hist_c.max():>10.1f}%   {rvol_unc_flat.max():>12.1f}%   {rvol_sim_flat.max():>10.1f}%")

# ── KDE helpers ───────────────────────────────────────────────────────────────
def kde_curve(data, x_grid):
    kde = gaussian_kde(data, bw_method="scott")
    return kde(x_grid)

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "MSTR Volatility Distribution: Historical BTC-era vs EGARCH (uncapped vs capped)  |  "
    f"N={N_PATHS} paths × {HORIZON}d  |  Rolling window={ROLL_WIN}d  |  Cap={VOL_CAP_ANN:.0f}%/yr",
    fontsize=10, fontweight="bold"
)

COL_H = "#e6550d"   # orange — historical
COL_S = "#2171b5"   # blue   — simulated (capped)
COL_U = "#9e9ac8"   # purple — simulated (uncapped)
COL_C = "#31a354"   # green  — EGARCH conditional

# ── [0,0] Histogram + KDE ─────────────────────────────────────────────────────
ax = axes[0, 0]
x_max = min(max(np.percentile(rvol_hist_c, 99.5),
                np.percentile(rvol_sim_flat, 99.5)), 600)
bins  = np.linspace(0, x_max, 70)
x_kde = np.linspace(0, x_max, 400)

ax.hist(rvol_hist_c,   bins=bins, density=True, alpha=0.40, color=COL_H,
        label=f"Historical (N={len(rvol_hist_c)})")
ax.hist(rvol_unc_flat[rvol_unc_flat <= x_max],
        bins=bins, density=True, alpha=0.25, color=COL_U,
        label=f"Sim uncapped (tail truncated for display)")
ax.hist(rvol_sim_flat, bins=bins, density=True, alpha=0.30, color=COL_S,
        label=f"Sim capped {VOL_CAP_ANN:.0f}%/yr")
ax.plot(x_kde, kde_curve(rvol_hist_c[rvol_hist_c < x_max],   x_kde), color=COL_H, lw=2)
ax.plot(x_kde, kde_curve(rvol_sim_flat[rvol_sim_flat < x_max], x_kde), color=COL_S, lw=2)
for pct, ls in [(50, "-"), (95, "--")]:
    ax.axvline(np.percentile(rvol_hist_c,   pct), color=COL_H, lw=0.9, ls=ls, alpha=0.8)
    ax.axvline(np.percentile(rvol_sim_flat, pct), color=COL_S, lw=0.9, ls=ls, alpha=0.8)
ax.axvline(VOL_CAP_ANN,  color="red",  lw=1.4, ls="--", label=f"Vol cap = {VOL_CAP_ANN:.0f}%/yr")
ax.axvline(lr_vol_ann,   color="gray", lw=1.0, ls=":",  label=f"EGARCH LR = {lr_vol_ann:.0f}%/yr")
ax.set_xlabel("Annualised rolling 21d vol (%/yr)"); ax.set_ylabel("Density")
ax.set_title("Distribution of rolling 21d vol  (histogram + KDE)\n"
             "vertical lines: median (solid) P95 (dashed)")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# ── [0,1] ECDF ────────────────────────────────────────────────────────────────
ax = axes[0, 1]
def ecdf(data):
    s = np.sort(data)
    p = np.arange(1, len(s) + 1) / len(s)
    return s, p

xh, yh  = ecdf(rvol_hist_c)
xs, ys  = ecdf(rvol_sim_flat)
xu, yu  = ecdf(rvol_unc_flat)

ax.plot(xh, yh, color=COL_H, lw=2,   label=f"Historical  (mean={rvol_hist_c.mean():.0f}%)")
ax.plot(xu, yu, color=COL_U, lw=1.2, ls="--", alpha=0.8,
        label=f"Sim uncapped (mean={rvol_unc_flat.mean():.0f}%)")
ax.plot(xs, ys, color=COL_S, lw=1.8, label=f"Sim capped  (mean={rvol_sim_flat.mean():.0f}%)")
ax.axvline(VOL_CAP_ANN,  color="red",  lw=1.2, ls="--", label=f"Cap={VOL_CAP_ANN:.0f}%")
ax.axvline(lr_vol_ann,   color="gray", lw=1.0, ls=":",  label=f"EGARCH LR={lr_vol_ann:.0f}%")
for pct in [25, 50, 75, 95]:
    ax.axhline(pct / 100, color="lightgray", lw=0.6, ls="--")
    ax.text(1, pct / 100 + 0.01, f"P{pct}", fontsize=7, color="gray")
ax.set_xlim(0, min(x_max, 500)); ax.set_ylim(0, 1)
ax.set_xlabel("Annualised rolling 21d vol (%/yr)"); ax.set_ylabel("Cumulative probability")
ax.set_title("ECDF of rolling 21d vol\nCapped ECDF aligns with historical beyond P95")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# ── [1,0] Historical rolling vol vs EGARCH conditional sigma_t ────────────────
ax = axes[1, 0]
days_hist = np.arange(N_hist)
ax.fill_between(days_hist, rvol_hist, alpha=0.25, color=COL_S)
ax.plot(days_hist, rvol_hist, color=COL_S, lw=0.8, alpha=0.8,
        label=f"Rolling {ROLL_WIN}d realised vol (historical)")
ax.plot(days_hist, cond_vol,  color=COL_C, lw=1.3,
        label="EGARCH conditional sigma_t (in-sample)")
ax.axhline(rvol_hist_c.mean(), color=COL_H, lw=1, ls="--",
           label=f"Hist mean={rvol_hist_c.mean():.0f}%")
ax.axhline(lr_vol_ann, color="gray", lw=1, ls=":",
           label=f"EGARCH LR={lr_vol_ann:.0f}%")
ax.set_xlabel("Historical trading day (BTC-era: Aug 2020 → present)")
ax.set_ylabel("Annualised vol (%/yr)")
ax.set_title("Historical vol vs EGARCH in-sample conditional sigma_t\n"
             "Green tracks blue = model captures vol regimes")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── [1,1] QQ-plot: historical rolling vol vs simulated rolling vol ─────────────
ax = axes[1, 1]
q_levels = np.linspace(1, 99, 200)
q_hist = np.percentile(rvol_hist_c,   q_levels)
q_sim  = np.percentile(rvol_sim_flat, q_levels)

ax.plot(q_hist, q_sim, "o", color=COL_S, ms=3, alpha=0.7, label="Quantile pairs")
q_min  = min(q_hist[0],  q_sim[0])
q_max  = max(q_hist[-1], q_sim[-1])
q_max  = min(q_max, 600)
ax.plot([q_min, q_max], [q_min, q_max], "k--", lw=1.2, label="y = x (perfect fit)")
# Mark key percentiles
for pct in [50, 75, 90, 95, 99]:
    qh = np.percentile(rvol_hist_c,   pct)
    qs = np.percentile(rvol_sim_flat, pct)
    ax.annotate(f"P{pct}", (qh, qs), textcoords="offset points",
                xytext=(5, 3), fontsize=7, color="gray")
    ax.plot(qh, qs, "x", color=COL_H, ms=6, zorder=5)

ax.set_xlim(q_min - 5, q_max + 5); ax.set_ylim(q_min - 5, q_max + 5)
ax.axvline(VOL_CAP_ANN, color="red", lw=1, ls="--", alpha=0.6, label=f"Cap={VOL_CAP_ANN:.0f}%")
ax.set_xlabel("Historical rolling 21d vol quantiles (%/yr)")
ax.set_ylabel("Simulated (capped) rolling 21d vol quantiles (%/yr)")
ax.set_title("QQ-plot: historical vs simulated (capped) rolling vol\n"
             "Points on diagonal = perfect distributional match")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, "diag_vol_comparison.png")
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close("all")
print("Done.")
