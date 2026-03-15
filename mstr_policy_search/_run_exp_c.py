"""
Experiment C — Disable vol-scaled roll pricing: VOL_SCALE_ROLL = False
EGARCH paths still used; roll prices revert to frozen surface (no BS vol-scaling).
Diagnostic: isolates how much of the E_drag increase is from vol-scaling vs EGARCH paths.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("main.py", encoding="utf-8") as fh:
    src = fh.read()

# Patch: disable vol-scaled roll pricing
src = src.replace(
    "VOL_SCALE_ROLL      = True            # True  = vol-scaled Black-Scholes at roll pricing events",
    "VOL_SCALE_ROLL      = False           # EXP-C: diagnostic — frozen surface roll pricing (no vol-scale)"
)
assert "VOL_SCALE_ROLL      = False" in src, "Patch failed — check exact line in main.py"

exec(compile(src, "main.py", "exec"), {"__name__": "__main__", "__file__": "main.py"})
