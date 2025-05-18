import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel

rows = []
for csv_path in Path("runs").rglob("metrics_summary.csv"):
    df = pd.read_csv(csv_path)
    rows.append(df)
full = pd.concat(rows, ignore_index=True)
full.to_csv("runs/all_metrics.csv", index=False)
print(full.groupby(["use_xi", "lambda_coef", "tau"]).agg(
    test_mse_mean=("test_mse", "mean"),
    xi_mean=("test_hard_xi", "mean"),
    n=("test_mse", "size"),
))

##t-test

base = full[full.use_xi == 0].sort_values("seed")
xi   = full[full.use_xi == 1].sort_values("seed")
stat, p = ttest_rel(base["test_hard_xi"].values, xi["test_hard_xi"].values)
print(f"Paired t-test on hard ξ: t={stat:.3f}, p={p:.4f}")
