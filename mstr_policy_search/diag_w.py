"""
Diagnostic: analyse the W_final distribution to understand CVaR_W drivers.
"""
import sys, json
import numpy as np

W = np.load("results/W_final_best.npy")
N = len(W)

sw   = np.sort(W)
idx  = np.argsort(W)

print(f"\n=== W_final distribution ({N} paths) ===")
print(f"  Mean:         {W.mean():+.2f}")
print(f"  Median:       {np.median(W):+.2f}")
print(f"  Std:          {W.std():.2f}")
print(f"  Min:          {W.min():+.2f}")
print(f"  Max:          {W.max():+.2f}")
print(f"  CVaR 10%:     {sw[:int(0.10*N)].mean():+.2f}")
print(f"  CVaR 25%:     {sw[:int(0.25*N)].mean():+.2f}")
print(f"  P(W > 0):     {(W>0).mean()*100:.1f}%")
print(f"  P(W < -500):  {(W<-500).mean()*100:.1f}%")
print(f"  P(W < -1000): {(W<-1000).mean()*100:.1f}%")
print(f"  P(W < -1500): {(W<-1500).mean()*100:.1f}%")

print("\n=== Worst 10 paths by W ===")
for rank, pi in enumerate(idx[:10]):
    print(f"  Rank {rank+1:2d}: path {pi:4d}  W = {W[pi]:+.2f}")

print("\n=== Best 10 paths by W ===")
for rank, pi in enumerate(idx[-10:][::-1]):
    print(f"  Rank {rank+1:2d}: path {pi:4d}  W = {W[pi]:+.2f}")

worst_path = int(idx[0])
print(f"\nWorst W path index: {worst_path}  (W={W[worst_path]:+.2f})")
print("Run:  python trace_policy.py --path", worst_path, "--all-days")
print("to see the day-by-day execution.")
