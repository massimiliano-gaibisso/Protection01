"""
diag_paths.py — Stock path diagnostics: drift, moments, rolling vol, clustering.

Generates two figures:
  Figure 1 — Path & return diagnostics (4 panels)
    [0,0] Sample paths (20 paths, S_t/S0)
    [0,1] Log-return histogram vs Normal + t5 fit
    [1,0] Rolling 21d realised vol for 5 sample paths
    [1,1] ACF of |returns| (vol clustering proxy)

  Figure 2 — Distribution across all paths (3 panels)
    [0]   Box plot of terminal S_T / S0 at day 504
    [1]   Annualised drift = mean(log(S_T/S0)) / T  across all paths histogram
    [2]   Cross-path std(log-return) histogram (realised vol per path)

Run from mstr_policy_search/:
    python diag_paths.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t, skew, kurtosis

import data_loader
from bootstrap import BootstrapSampler
from egarch import fit_egarch

# ── Config ────────────────────────────────────────────────────────────────────
N_PATHS      = 2_000
HORIZON      = 504      # trading days (2 years)
SEED         = 137
BTC_CUTOFF   = "2020-08-11"
N_SAMPLE_PATHS = 20     # paths to draw on fan chart
ROLL_WIN     = 21       # rolling vol window (trading days)
ACF_LAGS     = 40       # autocorrelation lags

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
chain, spot, _ = data_loader.load_option_chain("MSTR")
returns = data_loader.load_historical_returns("MSTR", cutoff_date=BTC_CUTOFF)

raw_mean = float(returns.mean())
raw_std  = float(returns.std())
raw_skew = float(skew(returns))
raw_kurt = float(kurtosis(returns))   # excess kurtosis

print(f"\nHistorical BTC-era returns (N={len(returns)}):")
print(f"  Mean   : {raw_mean*100:+.4f}%/day  ({raw_mean*252*100:+.1f}%/yr)")
print(f"  Std    : {raw_std*100:.4f}%/day   ({raw_std*np.sqrt(252)*100:.1f}%/yr)")
print(f"  Skew   : {raw_skew:+.3f}")
print(f"  Ex.Kurt: {raw_kurt:+.3f}  (Normal=0)")
print(f"  Min    : {returns.min()*100:.1f}%")
print(f"  Max    : {returns.max()*100:+.1f}%")

# ── Fit EGARCH ────────────────────────────────────────────────────────────────
print("\nFitting EGARCH(1,1) ...")
egarch_params = fit_egarch(returns, verbose=True)

# ── Generate paths ────────────────────────────────────────────────────────────
print(f"\nGenerating {N_PATHS} EGARCH paths (seed={SEED}) ...")
sampler = BootstrapSampler(returns, seed=SEED)
sampler.set_egarch_params(egarch_params)
paths = sampler.sample_paths(
    N_PATHS, HORIZON, spot,
    annual_default_prob=0.001,
    block_size=10,
    use_egarch=True,
)
print(f"  Shape: {paths.shape}   (paths x days+1)")

# paths[:,0] = S0 = spot; paths[:,d] = S_d
log_returns_sim = np.diff(np.log(paths), axis=1)   # shape (N_PATHS, HORIZON)

# ── Path-level statistics ─────────────────────────────────────────────────────
terminal_ratio = paths[:, -1] / spot          # S_T / S_0
log_terminal   = np.log(terminal_ratio)
ann_drift_sim  = log_terminal / (HORIZON / 252.0)  # annualised log-drift per path
path_vol_sim   = log_returns_sim.std(axis=1) * np.sqrt(252)  # realised ann vol per path

print(f"\nSimulated path statistics ({N_PATHS} paths, {HORIZON}d horizon):")
print(f"  Median S_T/S0        : {np.median(terminal_ratio):.3f}x")
print(f"  Mean  S_T/S0         : {np.mean(terminal_ratio):.3f}x")
print(f"  P5 / P95 S_T/S0      : {np.percentile(terminal_ratio,5):.3f}x / {np.percentile(terminal_ratio,95):.3f}x")
print(f"  Mean ann drift       : {np.mean(ann_drift_sim)*100:+.1f}%/yr")
print(f"  Median ann drift     : {np.median(ann_drift_sim)*100:+.1f}%/yr")
print(f"  Mean realised vol    : {np.mean(path_vol_sim)*100:.1f}%/yr")
print(f"  Median realised vol  : {np.median(path_vol_sim)*100:.1f}%/yr")
print(f"  P5 realised vol      : {np.percentile(path_vol_sim,5)*100:.1f}%/yr")
print(f"  P95 realised vol     : {np.percentile(path_vol_sim,95)*100:.1f}%/yr")

# ── Rolling vol for sample paths ──────────────────────────────────────────────
sample_idx = np.random.default_rng(0).choice(N_PATHS, N_SAMPLE_PATHS, replace=False)
sample_paths = paths[sample_idx]         # (N_SAMPLE_PATHS, HORIZON+1)
sample_lr    = log_returns_sim[sample_idx]  # (N_SAMPLE_PATHS, HORIZON)

# rolling std over ROLL_WIN days, annualised
def rolling_vol(lr, w):
    n = lr.shape[1]
    rv = np.full_like(lr, np.nan)
    for d in range(w - 1, n):
        rv[:, d] = lr[:, d - w + 1: d + 1].std(axis=1)
    return rv * np.sqrt(252)

rvol = rolling_vol(sample_lr, ROLL_WIN)
days_axis = np.arange(HORIZON)

# ── ACF of absolute returns (vol clustering) ──────────────────────────────────
# Use all simulated returns pooled
all_lr_flat = log_returns_sim.flatten()
abs_lr = np.abs(all_lr_flat - all_lr_flat.mean())

def acf(x, n_lags):
    """Sample ACF at lags 1..n_lags"""
    x = x - x.mean()
    var = np.dot(x, x)
    acf_vals = np.array([np.dot(x[:-k], x[k:]) / var if k > 0 else 1.0
                         for k in range(n_lags + 1)])
    return acf_vals

acf_abs = acf(abs_lr, ACF_LAGS)
conf_band = 1.96 / np.sqrt(len(abs_lr))

# Also compute ACF for actual historical returns
hist_abs_lr = np.abs(returns - returns.mean())
acf_hist = acf(hist_abs_lr, ACF_LAGS)
conf_hist = 1.96 / np.sqrt(len(hist_abs_lr))

# ── Figure 1: path & return diagnostics ──────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    f"MSTR Path & Return Diagnostics  |  EGARCH(1,1)+t5  |  BTC-era  |  N={N_PATHS} paths x {HORIZON}d",
    fontsize=11, fontweight="bold"
)

# [0,0] Sample paths
ax = axes[0, 0]
for i in range(N_SAMPLE_PATHS):
    alpha_p = 0.35
    ax.plot(sample_paths[i] / spot, lw=0.7, alpha=alpha_p, color="#2171b5")
# percentiles across all paths
pct5  = np.percentile(paths / spot, 5,  axis=0)
pct25 = np.percentile(paths / spot, 25, axis=0)
pct50 = np.percentile(paths / spot, 50, axis=0)
pct75 = np.percentile(paths / spot, 75, axis=0)
pct95 = np.percentile(paths / spot, 95, axis=0)
t_ax  = np.arange(HORIZON + 1)
ax.fill_between(t_ax, pct5,  pct95, alpha=0.08, color="#2171b5", label="P5-P95")
ax.fill_between(t_ax, pct25, pct75, alpha=0.15, color="#2171b5", label="P25-P75")
ax.plot(t_ax, pct50, lw=1.8, color="#08306b", label=f"Median")
ax.axhline(1.0, ls="--", lw=0.8, color="gray")
ax.set_xlabel("Trading days"); ax.set_ylabel("S_t / S_0")
ax.set_title(f"Sample paths (N={N_SAMPLE_PATHS} shown) + percentile fan")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# [0,1] Log-return distribution
ax = axes[0, 1]
all_lr_sample = log_returns_sim[:500].flatten()  # use 500 paths to keep histogram manageable
bins = np.linspace(np.percentile(all_lr_sample, 0.5),
                   np.percentile(all_lr_sample, 99.5), 80)
ax.hist(all_lr_sample, bins=bins, density=True, alpha=0.5, color="#2171b5",
        label=f"Simulated (N=500 paths)")
ax.hist(returns, bins=bins, density=True, alpha=0.45, color="#e6550d",
        label="Historical BTC-era")
# Normal fit to historical
x_fit = np.linspace(bins[0], bins[-1], 300)
ax.plot(x_fit, norm.pdf(x_fit, raw_mean, raw_std), "k--", lw=1.5, label="Normal fit (hist)")
# t5 fit
from scipy.stats import t as tdist
df_fit, loc_fit, scale_fit = tdist.fit(returns, floc=raw_mean)
ax.plot(x_fit, tdist.pdf(x_fit, df_fit, loc_fit, scale_fit), "g-.", lw=1.5,
        label=f"t({df_fit:.1f}) fit (hist)")
ax.set_xlabel("Daily log-return"); ax.set_ylabel("Density")
ax.set_title(f"Return distribution  |  hist skew={raw_skew:+.2f}  kurt={raw_kurt:+.2f}")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# [1,0] Rolling vol
ax = axes[1, 0]
for i in range(N_SAMPLE_PATHS):
    ax.plot(days_axis, rvol[i] * 100, lw=0.8, alpha=0.5, color="#2171b5")
med_rvol = np.nanmedian(rolling_vol(log_returns_sim, ROLL_WIN), axis=0) * 100
ax.plot(days_axis, med_rvol, lw=2, color="#08306b", label="Median across all paths")
ax.axhline(raw_std * np.sqrt(252) * 100, ls="--", lw=1, color="#e6550d",
           label=f"Hist. vol = {raw_std*np.sqrt(252)*100:.0f}%/yr")
# EGARCH long-run vol
lr_vol_pct = egarch_params.get("long_run_vol_annual", raw_std * np.sqrt(252)) * 100
ax.axhline(lr_vol_pct, ls=":", lw=1, color="gray",
           label=f"EGARCH long-run = {lr_vol_pct:.0f}%/yr")
ax.set_xlabel("Trading days"); ax.set_ylabel("Annualised vol (%)")
ax.set_title(f"Rolling {ROLL_WIN}d realised vol  (N={N_SAMPLE_PATHS} paths shown)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# [1,1] ACF of |returns|
ax = axes[1, 1]
lag_arr = np.arange(ACF_LAGS + 1)
ax.bar(lag_arr[1:], acf_abs[1:], width=0.4, color="#2171b5", alpha=0.7,
       label="Sim |returns| ACF")
ax.bar(lag_arr[1:] + 0.4, acf_hist[1:], width=0.4, color="#e6550d", alpha=0.7,
       label="Hist |returns| ACF")
ax.axhline(conf_band,  ls="--", lw=0.8, color="#2171b5", alpha=0.6)
ax.axhline(-conf_band, ls="--", lw=0.8, color="#2171b5", alpha=0.6)
ax.axhline(conf_hist,  ls="--", lw=0.8, color="#e6550d", alpha=0.6)
ax.axhline(-conf_hist, ls="--", lw=0.8, color="#e6550d", alpha=0.6)
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
ax.set_title("ACF of |daily returns|  (vol clustering)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig1.tight_layout()
path1 = os.path.join(OUT_DIR, "diag_paths_fig1.png")
fig1.savefig(path1, dpi=140, bbox_inches="tight")
print(f"\nSaved Fig 1: {path1}")

# ── Figure 2: cross-path distribution statistics ──────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    f"MSTR Cross-Path Distribution  |  EGARCH(1,1)+t5  |  N={N_PATHS} paths x {HORIZON}d",
    fontsize=11, fontweight="bold"
)

# [0] Terminal S_T/S0 distribution
ax = axes2[0]
ax.hist(terminal_ratio, bins=60, density=True, color="#2171b5", alpha=0.7, edgecolor="none")
ax.axvline(np.median(terminal_ratio), color="#08306b", lw=2, label=f"Median={np.median(terminal_ratio):.2f}x")
ax.axvline(np.mean(terminal_ratio),   color="#e6550d", lw=1.5, ls="--",
           label=f"Mean={np.mean(terminal_ratio):.2f}x")
ax.axvline(np.percentile(terminal_ratio,5), color="gray", lw=1, ls=":",
           label=f"P5={np.percentile(terminal_ratio,5):.2f}x")
ax.set_xlabel("S_T / S_0  (terminal price ratio)"); ax.set_ylabel("Density")
ax.set_title(f"Terminal price  (day {HORIZON})")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
# cap x-axis to ignore extreme outliers for readability
ax.set_xlim(0, np.percentile(terminal_ratio, 98))

# [1] Per-path annualised drift
ax = axes2[1]
ax.hist(ann_drift_sim * 100, bins=60, density=True, color="#2ca25f", alpha=0.7, edgecolor="none")
ax.axvline(np.mean(ann_drift_sim)*100, color="#00441b", lw=2,
           label=f"Mean={np.mean(ann_drift_sim)*100:+.1f}%/yr")
ax.axvline(np.median(ann_drift_sim)*100, color="#006d2c", lw=1.5, ls="--",
           label=f"Median={np.median(ann_drift_sim)*100:+.1f}%/yr")
ax.axvline(raw_mean*252*100, color="#e6550d", lw=1.5, ls=":",
           label=f"Hist mean={raw_mean*252*100:+.1f}%/yr")
ax.set_xlabel("Annualised log-drift (%/yr)"); ax.set_ylabel("Density")
ax.set_title("Per-path annualised drift")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# [2] Per-path realised vol
ax = axes2[2]
ax.hist(path_vol_sim * 100, bins=60, density=True, color="#feb24c", alpha=0.7, edgecolor="none")
ax.axvline(np.mean(path_vol_sim)*100, color="#a50f15", lw=2,
           label=f"Mean={np.mean(path_vol_sim)*100:.0f}%/yr")
ax.axvline(np.median(path_vol_sim)*100, color="#de2d26", lw=1.5, ls="--",
           label=f"Median={np.median(path_vol_sim)*100:.0f}%/yr")
ax.axvline(raw_std*np.sqrt(252)*100, color="#252525", lw=1.5, ls=":",
           label=f"Hist vol={raw_std*np.sqrt(252)*100:.0f}%/yr")
ax.axvline(lr_vol_pct, color="blue", lw=1, ls="-.",
           label=f"EGARCH LR vol={lr_vol_pct:.0f}%/yr")
ax.set_xlabel("Realised annualised vol (%/yr)"); ax.set_ylabel("Density")
ax.set_title("Per-path realised volatility")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig2.tight_layout()
path2 = os.path.join(OUT_DIR, "diag_paths_fig2.png")
fig2.savefig(path2, dpi=140, bbox_inches="tight")
print(f"Saved Fig 2: {path2}")

plt.close("all")
print("\nDone.")
