"""
Experiment B — Force ITM long puts: ALPHA_L_VALUES = [1.10, 1.15, 1.20]
Everything else identical to the baseline EGARCH run.
Results tag: _drift_-0.20pct  (same tag, will overwrite if run after baseline)
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("main.py", encoding="utf-8") as fh:
    src = fh.read()

# Patch: remove ATM (1.00) and near-ATM (1.05) from long-put search space
src = src.replace(
    "ALPHA_L_VALUES = [1.00, 1.05, 1.10, 1.15, 1.20]   # 0.95 is so OTM it provides little protection",
    "ALPHA_L_VALUES = [1.10, 1.15, 1.20]  # EXP-B: ITM only — remove ATM (1.00/1.05) to fix drag"
)
assert "ALPHA_L_VALUES = [1.10, 1.15, 1.20]" in src, "Patch failed — check exact line in main.py"

exec(compile(src, "main.py", "exec"), {"__name__": "__main__", "__file__": "main.py"})
