#execute_parkinsons.py
"""
End‑to‑end executor for all Parkinson ξₙ experiments, synthetic study,
timing benchmark, aggregation, statistics, and figure generation.

Outputs go to   runs/YYYYMMDD_HHMMSS_parkinsons_exec/
and include:
    all_metrics.csv      – 1 row per trained model
    synth_metrics.csv    – synthetic noise sweep
    timing_bench.csv     – overhead numbers
    stats_summary.txt    – paired‑t result
    figures/ *.png       – heat‑map, barplots, learning curves, timing
    session.log          – full console log
"""
from __future__ import annotations

import csv
import logging
import math
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------- optional/slow imports guarded ----------
try:
    from scipy import stats as sp_stats  # for t‑test
except ImportError:
    print("SciPy not found – install with  pip install scipy  and re‑run.")
    sys.exit(1)

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import torch

# local code
sys.path.insert(0, str(Path(__file__).parent))  # enable relative import
from run_parkinsons import main as run_parkinsons_main  # type: ignore
import run_parkinsons as rp


# ------------------------------- configuration -------------------------------

SEEDS = list(range(10))  # 0 … 9 inclusive
LAMBDA_SET = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
TAU_SET = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
EPOCHS = 60
WARMUP = 5
BATCH = 256

DATA_DIR = Path("data")
RUNS_ROOT = Path("runs")
DATA_CSV = DATA_DIR / "parkinsons_tele.csv"

# synthetic study
SYN_FUNCS = {
    "linear": lambda x: x,
    "quadratic": lambda x: x**2,
    "sine": lambda x: np.sin(x),
}
SYN_SIGMA = [0.01, 0.1, 1.0]
SYN_N = 1_000

# timing
TIMING_STEPS = 100

# --------------------------- session directories -----------------------------

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = RUNS_ROOT / f"{STAMP}_parkinsons_exec"
FIG_DIR = SESSION_DIR / "figures"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------- logging setup -----------------------------------

LOG_FILE = SESSION_DIR / "session.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("exec")

log.info("===== Parkinson ξₙ execution started =====")
log.info("Outputs will be written to %s", SESSION_DIR.resolve())

# basic sanity
if not DATA_CSV.exists():
    log.error("Dataset not found at %s", DATA_CSV)
    sys.exit(1)

free_mb = shutil.disk_usage(SESSION_DIR).free / 1_048_576
if free_mb < 50:
    log.error("Less than 50 MB free disk space – aborting.")
    sys.exit(1)

# --------------------------- helper utilities --------------------------------


def run_one_parkinsons(seed: int, use_xi: bool, lam: float, tau: float) -> Dict[str, float]:
    """
    Call run_parkinsons.main() directly with an argparse.Namespace.
    Returns the metrics row as a dict.
    """
    outdir = SESSION_DIR / f"s{seed}_xi{int(use_xi)}_l{lam}_t{tau}"
    if outdir.exists() and (outdir / "metrics_summary.csv").exists():
        log.info("Skip existing %s", outdir.name)
        # read existing row
        row = pd.read_csv(outdir / "metrics_summary.csv").iloc[0].to_dict()
        return row

    args_ns = type("Args", (), {})()
    args_ns.data_dir = str(DATA_DIR)
    args_ns.outdir = str(outdir)
    args_ns.seed = seed
    args_ns.target = "total_UPDRS"
    args_ns.use_xi = use_xi
    args_ns.lambda_coef = lam
    args_ns.tau = tau
    args_ns.epochs = EPOCHS
    args_ns.warmup_epochs = WARMUP
    args_ns.batch_size = BATCH
    args_ns.lr = 1e-3
    args_ns.grad_clip = 1.0
    args_ns.cpu = True  # enforce CPU
    rp.args = args_ns
    run_parkinsons_main(args_ns)  # runs training, writes files

    # validate outputs
    summary_csv = outdir / "metrics_summary.csv"
    hist_csv = outdir / "history.csv"
    if not summary_csv.exists() or not hist_csv.exists():
        raise RuntimeError(f"Missing expected outputs in {outdir}")
    hist_rows = pd.read_csv(hist_csv).shape[0]
    if hist_rows != EPOCHS:
        raise RuntimeError(f"history.csv row count mismatch in {outdir}")

    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    if not np.isfinite(list(row.values())).all():
        raise RuntimeError(f"Non‑finite metric in {outdir}")

    return row


def paired_t(baseline: pd.Series, xi: pd.Series) -> Dict[str, float]:
    diff = xi.values - baseline.values
    n = diff.shape[0]
    mean_diff = diff.mean()
    sd = diff.std(ddof=1)
    sem = sd / math.sqrt(n)
    t_val = mean_diff / sem if sem > 0 else np.inf
    p_two = 2 * (1 - sp_stats.t.cdf(abs(t_val), df=n - 1))
    return dict(n=n, mean=mean_diff, sd=sd, sem=sem, t=t_val, p=p_two)


# --------------------------- 2. experiment grid ------------------------------

rows: List[Dict[str, float]] = []

# --- progress bookkeeping ----------------------------------------------------
total_runs_expected = len(SEEDS) * (1 + len(LAMBDA_SET) * len(TAU_SET))
total_runs_done = 0
# -----------------------------------------------------------------------------

for seed in SEEDS:
    # baseline first
    rows.append(run_one_parkinsons(seed, False, 0.0, 0.0))
    total_runs_done += 1
    print(f"run {total_runs_done} completed, {total_runs_expected - total_runs_done} remaining. Estimated Time Remaining in Minutes: {((total_runs_expected - total_runs_done) * 3 ) / 60}")

    # xi variants
    for lam in LAMBDA_SET:
        for tau in TAU_SET:
            rows.append(run_one_parkinsons(seed, True, lam, tau))
            total_runs_done += 1
            print(f"run {total_runs_done} completed, {total_runs_expected - total_runs_done} remaining. Estimated Time Remaining in Minutes: {((total_runs_expected - total_runs_done) * 3 ) / 60}")


log.info("Parkinson runs completed – total %d models", total_runs_done)
all_df = pd.DataFrame(rows)
all_df.to_csv(SESSION_DIR / "all_metrics.csv", index=False)

# --------------------------- 3. statistics -----------------------------------

best_xi_mask = (
    (all_df.use_xi == 1)
    & (all_df.lambda_coef == 1.0)
    & (all_df.tau == 0.1)
)
baseline_df = all_df[all_df.use_xi == 0].sort_values("seed")
xi_df = all_df[best_xi_mask].sort_values("seed")

if not (baseline_df.seed.values == xi_df.seed.values).all():
    raise RuntimeError("Seed mismatch between baseline and xi models")

stats_res = paired_t(
    baseline_df.test_hard_xi,
    xi_df.test_hard_xi,
)
stats_path = SESSION_DIR / "stats_summary.txt"
with stats_path.open("w") as fh:
    fh.write(f"Paired t‑test hard ξ  (n={stats_res['n']} seeds)\n")
    fh.write(f"mean diff  {stats_res['mean']:.4f}\n")
    fh.write(f"t = {stats_res['t']:.4f},  p = {stats_res['p']:.5f}\n")
log.info("Stats written to %s", stats_path.name)

# --------------------------- 4. synthetic study ------------------------------

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

# monotonic check
for func_name in SYN_FUNCS:
    xi_vals = syn_df[syn_df.func == func_name].sort_values("sigma").xi.values
    if not np.all(np.diff(xi_vals) <= 1e-6):
        log.warning("non‑monotone ξ in synthetic %s – investigate later", func_name)

# --------------------------- 5. timing benchmark -----------------------------

def timing_variant(use_xi: bool) -> float:
    torch.manual_seed(0)
    n_feat = 16
    x = torch.randn(BATCH, n_feat)
    y = torch.randn(BATCH, 1)
    model = torch.nn.Linear(n_feat, 1)
    crit = xi_hard  # placeholder for type
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

# --------------------------- 6. figure generation ----------------------------

# Heat‑map mean ξ by λ‑τ
pivot = (
    all_df[all_df.use_xi == 1]
    .groupby(["lambda_coef", "tau"])
    .test_hard_xi.mean()
    .unstack()
)
plt.figure(figsize=(6, 4))
plt.imshow(pivot, aspect="auto", origin="lower")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.colorbar(label="mean hard ξ")
plt.xlabel("τ")
plt.ylabel("λ")
plt.title("ξₙ dependency by λ and τ")
plt.tight_layout()
plt.savefig(FIG_DIR / "heatmap_xi.png", dpi=120)
plt.close()

# Synthetic bar plot
plt.figure(figsize=(5, 4))
for m, color, col in [("xi", None, "xi"), ("Spearman", None, "spearman"), ("Pearson", None, "pearson")]:
    vals = []
    lbls = []
    for func_name in SYN_FUNCS:
        sub = syn_df[syn_df.func == func_name]
        vals.extend(sub[col].values.tolist())
        lbls.extend([func_name] * len(sub))
    plt.scatter(lbls, vals, label=m)
plt.ylabel("Correlation value")
plt.title("Noise sweep – 3 measures")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "synth_bar.png", dpi=120)
plt.close()

# Timing bar
plt.figure(figsize=(4, 4))
plt.bar(["Baseline", "ξ‑model"], [t_base, t_xi])
plt.ylabel("Time (s) for 100 steps")
plt.title(f"Overhead {over_pct:.1f} %")
plt.tight_layout()
plt.savefig(FIG_DIR / "timing_overhead.png", dpi=120)
plt.close()

log.info("Figures saved to %s", FIG_DIR.name)

# --------------------------- 7. completion -----------------------------------

required = [
    SESSION_DIR / "all_metrics.csv",
    SESSION_DIR / "synth_metrics.csv",
    SESSION_DIR / "timing_bench.csv",
    FIG_DIR / "heatmap_xi.png",
]
missing = [p.name for p in required if not p.exists()]
if missing:
    log.error("Missing artefacts: %s", ", ".join(missing))
    sys.exit(1)

log.info("EXECUTION COMPLETE — ALL ARTEFACTS READY")