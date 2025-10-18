# execute_diabetes.py
"""
End-to-end executor for diabetes experiments using the optional Xi regularizer.

This single pass produces ALL artifacts for both RQ2 and RQ3 with selection/evaluation
properly aligned, and uses the Nadeau–Bengio corrected resampled t-test.

Outputs are written to
    runs/YYYYMMDD_HHMMSS_diabetes_exec/
and include:
    all_metrics.csv        - 1 row per trained model (baseline + xi grid), expanded schema
    best_models_rq2.csv    - per seed, baseline vs best-by-valMSE xi (report test_mse_mseSel)
    best_models_rq3.csv    - per seed, baseline vs best-by-valHardXi xi (report test_hard_xi_xiSel)
    stats_summary_rq2.txt  - corrected resampled t-test on MSE improvements (baseline - xi)
    stats_summary_rq3.txt  - corrected resampled t-test on xi improvements (xi - baseline)
    synth_metrics.csv      - synthetic noise sweep
    timing_bench.csv       - overhead numbers
    figures/heatmap_xi_xiSel.png             - mean test hard xi (xi-selected), λ×τ (xi only)
    figures/heatmap_test_mse_mseSel.png      - mean test MSE (mse-selected), includes baseline at λ=0,τ=0
    figures/heatmap_test_r2_mseSel.png       - mean test R² (mse-selected), includes baseline at λ=0,τ=0
    figures/synth_scatter.png
    figures/timing_overhead.png
    session.log            - full console log
"""
from __future__ import annotations

import logging
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from scipy import stats as sp_stats
except ImportError:
    print("SciPy not found - install with  pip install scipy")
    sys.exit(1)

RED   = "\033[91m"
RESET = "\033[0m"

# --------------------------------------------------------------------------- #
#  Local imports (assumes run_diabetes.py sits next to this script)           #
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).parent))
from run_diabetes import main as run_diabetes_main  # type: ignore
import run_diabetes as rd  # noqa: F401  (kept for completeness)

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
SEEDS = list(range(10))  # 0 … 9 inclusive
LAMBDA_SET = [5, 15, 30, 45, 60]
TAU_SET = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]

EPOCHS = 60
WARMUP = 5
BATCH = 64

RUNS_ROOT = Path("runs")

# Synthetic correlation sanity study
SYN_FUNCS = {
    "linear": lambda x: x,
    "quadratic": lambda x: x * x,
    "sine": lambda x: np.sin(x),
}
SYN_SIGMA = [0.01, 0.1, 1.0]
SYN_N = 1000

# Timing benchmark
TIMING_STEPS = 100

# Corrected resampled t-test parameters
R = len(SEEDS)
TRAIN_RATIO = 0.80
TEST_RATIO = 0.10
NB_C = (1.0 / R) + (TEST_RATIO / TRAIN_RATIO)  # = 0.1 + 0.125 = 0.225

# --------------------------------------------------------------------------- #
#  Session directories & logging                                              #
# --------------------------------------------------------------------------- #
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = RUNS_ROOT / f"{STAMP}_diabetes_exec"
FIG_DIR = SESSION_DIR / "figures"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = SESSION_DIR / "session.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("exec")

log.info("===== Diabetes execution started =====")
log.info("Outputs will be written to %s", SESSION_DIR.resolve())

# Basic disk-space sanity
free_mb = shutil.disk_usage(SESSION_DIR).free / 1_048_576
if free_mb < 50:
    log.error("Less than 50 MB free disk space - aborting.")
    sys.exit(1)

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def run_one_diabetes(seed: int, use_xi: bool, lam: float, tau: float) -> Dict[str, float]:
    """
    Train one model (or reuse cached results) and return its metrics row.
    Hard-fail if performance is clearly wrong, preventing wasted grid time.
    """
    outdir = SESSION_DIR / f"s{seed}_xi{int(use_xi)}_l{lam}_t{tau}"
    summary_csv = outdir / "metrics_summary.csv"

    # Fast path: reuse existing, but only if all numbers are finite
    if summary_csv.exists():
        row = pd.read_csv(summary_csv).iloc[0].to_dict()
        if np.isfinite(list(row.values())).all():
            return row
        log.warning("Non-finite values in %s - re-running model.", summary_csv)

    # Build argparse-like namespace expected by run_diabetes.main()
    args = SimpleNamespace(
        outdir=str(outdir),
        seed=seed,
        use_xi=use_xi,
        lambda_coef=lam,
        tau=tau,
        epochs=EPOCHS,
        warmup_epochs=WARMUP,
        batch_size=BATCH,
        lr=1e-3,
        grad_clip=1.0,
        cpu=True,
    )

    run_diabetes_main(args)

    if not summary_csv.exists():
        raise RuntimeError(f"Missing metrics_summary.csv in {outdir}")

    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    if not np.isfinite(list(row.values())).all():
        raise RuntimeError(f"Non-finite metric in {outdir}")

    # Quick sanity threshold: R² should not be terrible (on MSE-selected model)
    if row["test_r2_mseSel"] < 0.40:
        log.warning(
            "%sLow performance: seed=%d xi=%s  R2(mseSel)=%.3f%s",
            RED,
            seed,
            use_xi,
            row["test_r2_mseSel"],
            RESET,
        )

    return row


def paired_t_naive(diff: np.ndarray) -> Dict[str, float]:
    n = diff.shape[0]
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    sem = sd / math.sqrt(n) if n > 0 else float("nan")
    t_val = mean_diff / sem if sem > 0 else float("inf")
    p_two = 2 * (1 - sp_stats.t.cdf(abs(t_val), df=n - 1)) if sem > 0 else 0.0
    return dict(n=n, mean=mean_diff, sd=sd, sem=sem, t=t_val, p=p_two)


def nadeau_bengio_corrected_t(diff: np.ndarray, c: float) -> Dict[str, float]:
    """
    Nadeau–Bengio corrected resampled t-test.
    se = sqrt( c * s^2 ), where c = 1/r + n_test/n_train.
    """
    n = diff.shape[0]
    mean_diff = float(diff.mean())
    s2 = float(diff.var(ddof=1))
    se = math.sqrt(c * s2) if n > 1 and s2 >= 0 else float("nan")
    t_val = mean_diff / se if se > 0 else float("inf")
    p_two = 2 * (1 - sp_stats.t.cdf(abs(t_val), df=n - 1)) if se > 0 else 0.0
    return dict(n=n, mean=mean_diff, var=s2, se=se, t=t_val, p=p_two, c=c)


# --------------------------------------------------------------------------- #
#  1. Experiment grid                                                         #
# --------------------------------------------------------------------------- #
rows: List[Dict[str, float]] = []

total_runs_expected = len(SEEDS) * (1 + len(LAMBDA_SET) * len(TAU_SET))
total_runs_done = 0

for seed in SEEDS:
    # Baseline (Xi off) first
    rows.append(run_one_diabetes(seed, False, 0.0, 0.0))
    total_runs_done += 1
    remaining = total_runs_expected - total_runs_done
    log.info(
        "run %d completed, %d remaining. est minutes left: %.1f",
        total_runs_done,
        remaining,
        (remaining * 0.5) / 60,
    )

    # Xi variants
    for lam in LAMBDA_SET:
        for tau in TAU_SET:
            rows.append(run_one_diabetes(seed, True, lam, tau))
            total_runs_done += 1
            remaining = total_runs_expected - total_runs_done
            log.info(
                "run %d completed, %d remaining. est minutes left: %.1f",
                total_runs_done,
                remaining,
                (remaining * 0.5) / 60,
            )

log.info("Diabetes runs completed - total %d models", total_runs_done)
all_df = pd.DataFrame(rows)
all_df.to_csv(SESSION_DIR / "all_metrics.csv", index=False)

# --------------------------------------------------------------------------- #
#  2. Winner selection per seed & stats                                       #
# --------------------------------------------------------------------------- #
# Build per-seed comparison tables (one row per seed) for RQ2 and RQ3
rq2_rows = []
rq3_rows = []

for seed in SEEDS:
    base = all_df[(all_df.seed == seed) & (all_df.use_xi == 0)].iloc[0]

    # RQ2: choose xi config with MINIMUM best_val_mse
    cand2 = all_df[(all_df.seed == seed) & (all_df.use_xi == 1)]
    win2 = cand2.loc[cand2.best_val_mse.idxmin()]

    rq2_rows.append(
        dict(
            seed=seed,
            base_test_mse_mseSel=base.test_mse_mseSel,
            xi_lambda=win2.lambda_coef,
            xi_tau=win2.tau,
            xi_test_mse_mseSel=win2.test_mse_mseSel,
        )
    )

    # RQ3: choose xi config with MAXIMUM best_val_hard_xi
    cand3 = all_df[(all_df.seed == seed) & (all_df.use_xi == 1)]
    win3 = cand3.loc[cand3.best_val_hard_xi.idxmax()]

    rq3_rows.append(
        dict(
            seed=seed,
            base_test_hard_xi_xiSel=base.test_hard_xi_xiSel,
            xi_lambda=win3.lambda_coef,
            xi_tau=win3.tau,
            xi_test_hard_xi_xiSel=win3.test_hard_xi_xiSel,
        )
    )

best_rq2_df = pd.DataFrame(rq2_rows)
best_rq3_df = pd.DataFrame(rq3_rows)

best_rq2_df.to_csv(SESSION_DIR / "best_models_rq2.csv", index=False)
best_rq3_df.to_csv(SESSION_DIR / "best_models_rq3.csv", index=False)

# ---- Stats: Nadeau–Bengio corrected resampled t-tests ----
# RQ2 (lower is better): diff_i = baseline_i - xi_i (positive => xi improves MSE)
rq2_diff = best_rq2_df.base_test_mse_mseSel.values - best_rq2_df.xi_test_mse_mseSel.values
rq2_nb = nadeau_bengio_corrected_t(rq2_diff, NB_C)
rq2_naive = paired_t_naive(rq2_diff)

with (SESSION_DIR / "stats_summary_rq2.txt").open("w") as fh:
    fh.write("Corrected resampled t-test (Nadeau–Bengio) on Test MSE improvements\n")
    fh.write(f"r={R}, train={TRAIN_RATIO:.2f}, test={TEST_RATIO:.2f}, c={NB_C:.3f}\n")
    fh.write(f"mean diff (base - xi) = {rq2_nb['mean']:.6f}\n")
    fh.write(f"var={rq2_nb['var']:.6f}, se={rq2_nb['se']:.6f}\n")
    fh.write(f"t = {rq2_nb['t']:.6f},  p(two-sided) = {rq2_nb['p']:.6f}\n")
    fh.write("\nNaïve paired t-test (UNCORRECTED, anti-conservative)\n")
    fh.write(f"n={rq2_naive['n']}, mean={rq2_naive['mean']:.6f}, sd={rq2_naive['sd']:.6f}, "
             f"sem={rq2_naive['sem']:.6f}, t={rq2_naive['t']:.6f}, p={rq2_naive['p']:.6f}\n")

# RQ3 (higher is better): diff_i = xi_i - baseline_i (positive => xi improves hard-ξ)
rq3_diff = best_rq3_df.xi_test_hard_xi_xiSel.values - best_rq3_df.base_test_hard_xi_xiSel.values
rq3_nb = nadeau_bengio_corrected_t(rq3_diff, NB_C)
rq3_naive = paired_t_naive(rq3_diff)

with (SESSION_DIR / "stats_summary_rq3.txt").open("w") as fh:
    fh.write("Corrected resampled t-test (Nadeau–Bengio) on Test hard-ξ improvements\n")
    fh.write(f"r={R}, train={TRAIN_RATIO:.2f}, test={TEST_RATIO:.2f}, c={NB_C:.3f}\n")
    fh.write(f"mean diff (xi - base) = {rq3_nb['mean']:.6f}\n")
    fh.write(f"var={rq3_nb['var']:.6f}, se={rq3_nb['se']:.6f}\n")
    fh.write(f"t = {rq3_nb['t']:.6f},  p(two-sided) = {rq3_nb['p']:.6f}\n")
    fh.write("\nNaïve paired t-test (UNCORRECTED, anti-conservative)\n")
    fh.write(f"n={rq3_naive['n']}, mean={rq3_naive['mean']:.6f}, sd={rq3_naive['sd']:.6f}, "
             f"sem={rq3_naive['sem']:.6f}, t={rq3_naive['t']:.6f}, p={rq3_naive['p']:.6f}\n")

log.info("Stats written: stats_summary_rq2.txt, stats_summary_rq3.txt")

# --------------------------------------------------------------------------- #
#  3. Synthetic study                                                         #
# --------------------------------------------------------------------------- #
from losses.xi_loss import xi_hard  # local import late to avoid heavy deps

synthetic_rows = []
rng = np.random.default_rng(0)
for func_name, func in SYN_FUNCS.items():
    for sigma in SYN_SIGMA:
        x = rng.uniform(-3, 3, SYN_N)
        y_true = func(x)
        y = y_true + rng.normal(0, sigma, SYN_N)
        xi_val = xi_hard(torch.from_numpy(x), torch.from_numpy(y)).item()
        rho, _ = sp_stats.spearmanr(x, y)
        pear, _ = sp_stats.pearsonr(x, y)
        synthetic_rows.append(
            dict(func=func_name, sigma=sigma, xi=xi_val, spearman=rho, pearson=pear)
        )
syn_df = pd.DataFrame(synthetic_rows)
syn_df.to_csv(SESSION_DIR / "synth_metrics.csv", index=False)

# Monotonic sanity
for func_name in SYN_FUNCS:
    xi_vals = syn_df[syn_df.func == func_name].sort_values("sigma").xi.values
    if not np.all(np.diff(xi_vals) <= 1e-6):
        log.warning("Non-monotone xi in synthetic %s - investigate later", func_name)

# --------------------------------------------------------------------------- #
#  4. Timing benchmark                                                        #
# --------------------------------------------------------------------------- #
def timing_variant(use_xi: bool) -> float:
    torch.manual_seed(0)
    n_feat = 16
    x = torch.randn(BATCH, n_feat)
    y = torch.randn(BATCH, 1)
    model = torch.nn.Linear(n_feat, 1)

    if use_xi:
        from losses.xi_loss import XiLoss

        crit_obj = XiLoss(tau=0.1, lambda_=1.0)
    else:

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.task_loss = torch.nn.MSELoss()

            def forward(self, yp, yt):
                return self.task_loss(yp, yt), torch.tensor(0.0)

        crit_obj = Wrap()

    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    t0 = time.perf_counter()
    for _ in range(TIMING_STEPS):
        opt.zero_grad()
        out = model(x)
        loss, _ = crit_obj(out, y)
        loss.backward()
        opt.step()
    return time.perf_counter() - t0


t_base = timing_variant(False)
t_xi = timing_variant(True)
over_pct = 100.0 * (t_xi - t_base) / t_base

tim_df = pd.DataFrame(
    [{"baseline_s": t_base, "xi_s": t_xi, "overhead_pct": over_pct}]
)
tim_df.to_csv(SESSION_DIR / "timing_bench.csv", index=False)

# --------------------------------------------------------------------------- #
#  5. Figure generation                                                       #
# --------------------------------------------------------------------------- #
# Heat map mean hard xi (xi-selected) by lambda and tau (xi models only)
pivot_xi = (
    all_df[all_df.use_xi == 1]
    .groupby(["lambda_coef", "tau"])
    .test_hard_xi_xiSel.mean()
    .unstack()
    .sort_index()
    .reindex(sorted(all_df[all_df.use_xi == 1].tau.unique(), key=float), axis=1)
)
plt.figure(figsize=(6, 4))
im = plt.imshow(pivot_xi, aspect="auto", origin="lower", interpolation="nearest")
plt.xticks(range(len(pivot_xi.columns)), pivot_xi.columns)
plt.yticks(range(len(pivot_xi.index)), pivot_xi.index)
plt.colorbar(im, label="mean test hard xi (xi-selected)")
plt.xlabel("tau")
plt.ylabel("lambda")
plt.title("Xi dependency by lambda and tau (xi-selected)")
plt.tight_layout()
plt.savefig(FIG_DIR / "heatmap_xi_xiSel.png", dpi=120)
plt.close()
log.info("heatmap_xi_xiSel.png saved")

# Predictive heatmaps: include baseline cell at λ=0, τ=0
for metric, fname, title in [
    ("test_mse_mseSel", "heatmap_test_mse_mseSel.png", "Mean Test MSE (mse-selected, lower is better)"),
    ("test_r2_mseSel",  "heatmap_test_r2_mseSel.png",  "Mean Test R² (mse-selected, higher is better)"),
]:
    pivot = (
        all_df
        .groupby(["lambda_coef", "tau"])[metric]
        .mean()
        .unstack()
        .sort_index()
        .reindex(sorted(all_df.tau.unique(), key=float), axis=1)
    )

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot, aspect="auto", origin="lower", interpolation="nearest")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(im, label=f"mean {metric}")
    plt.xlabel("tau")
    plt.ylabel("lambda")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=120)
    plt.close()
    log.info("%s saved", fname)

# Synthetic scatter plot
plt.figure(figsize=(5, 4))
for label, col in [("Xi", "xi"), ("Spearman", "spearman"), ("Pearson", "pearson")]:
    vals, lbls = [], []
    for func_name in SYN_FUNCS:
        sub = syn_df[syn_df.func == func_name]
        vals.extend(sub[col].values.tolist())
        lbls.extend([func_name] * len(sub))
    plt.scatter(lbls, vals, label=label)
plt.ylabel("Correlation value")
plt.title("Noise sweep: three measures")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "synth_scatter.png", dpi=120)
plt.close()

# Timing bar
plt.figure(figsize=(4, 4))
plt.bar(["Baseline", "Xi model"], [t_base, t_xi])
plt.ylabel("Time (s) for 100 steps")
plt.title(f"Overhead {over_pct:.1f} percent")
plt.tight_layout()
plt.savefig(FIG_DIR / "timing_overhead.png", dpi=120)
plt.close()

log.info("Figures saved to %s", FIG_DIR.name)

# --------------------------------------------------------------------------- #
#  6. Completion check                                                        #
# --------------------------------------------------------------------------- #
required = [
    SESSION_DIR / "all_metrics.csv",
    SESSION_DIR / "best_models_rq2.csv",
    SESSION_DIR / "best_models_rq3.csv",
    SESSION_DIR / "stats_summary_rq2.txt",
    SESSION_DIR / "stats_summary_rq3.txt",
    SESSION_DIR / "synth_metrics.csv",
    SESSION_DIR / "timing_bench.csv",
    FIG_DIR / "heatmap_xi_xiSel.png",
    FIG_DIR / "heatmap_test_mse_mseSel.png",
    FIG_DIR / "heatmap_test_r2_mseSel.png",
]
missing = [p.name for p in required if not p.exists()]
if missing:
    log.error("Missing artefacts: %s", ", ".join(missing))
    sys.exit(1)

log.info("EXECUTION COMPLETE - ALL ARTEFACTS READY")