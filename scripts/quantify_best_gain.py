import pathlib as pl
import pandas as pd
import numpy as np
import math
import scipy.stats as st
import sys
import operator

# ----------------------------------------------------------------------
# Figure out where the latest runs/<timestamp>/all_metrics.csv lives
# ----------------------------------------------------------------------
repo_root = pl.Path(__file__).resolve().parents[1]          # project root
runs_dir  = repo_root / "runs"
if not runs_dir.exists():
    sys.exit("ERROR: runs/ directory not found – did you execute any experiments?")

# pick the most recently modified all_metrics.csv
all_csvs = sorted(
    runs_dir.rglob("all_metrics.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not all_csvs:
    sys.exit("ERROR: no all_metrics.csv found under runs/")
csv_path = all_csvs[0]
print(f"Using results at {csv_path.relative_to(repo_root)}")

df = pd.read_csv(csv_path)

# ----- summarise hard xi improvement -----
base = df[df.use_xi == 0].sort_values('seed')
xi   = df[(df.use_xi == 1)].sort_values(['lambda_coef','tau','seed'])

# pivot mean hard xi by (lambda, tau)
pivot = xi.pivot_table(index='lambda_coef', columns='tau',
                       values='test_hard_xi', aggfunc='mean')
print('\nMEAN hard xi\n', pivot.round(4))

# compute per-seed delta to baseline for each (lambda,tau)
best = None
for lam, tau in pivot.stack().index:
    cur = xi[(xi.lambda_coef == lam) & (xi.tau == tau)].sort_values('seed')
    delta = cur.test_hard_xi.values - base.test_hard_xi.values
    gain = delta.mean()
    if best is None or gain > best[0]:
        best = (gain, lam, tau, delta)
print(f'\nBest gain {best[0]:.4f} at lambda={best[1]} tau={best[2]}')

# paired t-test vs baseline
import scipy.stats as st
t, p = st.ttest_rel(xi[(xi.lambda_coef==best[1]) & (xi.tau==best[2])].sort_values("seed").test_hard_xi,
                    base.test_hard_xi)
print(f'paired t {t:.3f},  p = {p:.4g}')

df  = pd.read_csv(csv_path)
base = df[df.use_xi==0].sort_values('seed')
best = df[(df.use_xi==1)&(df.lambda_coef==0.5)&(df.tau==0.2)].sort_values('seed')
t,p = st.ttest_rel(best.test_mse, base.test_mse)
print(f'MSE delta {best.test_mse.mean()-base.test_mse.mean():.4f}, t={t:.2f}, p={p:.3f}')