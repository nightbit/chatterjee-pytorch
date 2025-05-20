#execute_friedman.py
"""
End-to-end executor for Friedman #1 regression experiments (optionally with Xi regularizer).

Outputs are written to
    runs/YYYYMMDD_HHMMSS_friedman_exec/
and include:
    all_metrics.csv      - one row per trained model
    synth_metrics.csv    - synthetic noise sweep
    timing_bench.csv     - Xi overhead numbers
    stats_summary.txt    - paired t-test result
    figures/*.png        - heat map, scatter plots, learning curves, timing
    session.log          - full console log
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

matplotlib.use("Agg")  # headless back-end
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
#  Local imports (assumes run_friedman.py sits next to this script)           #
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).parent))
from run_friedman import main as run_friedman_main  # type: ignore

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
SEEDS       = list(range(10))        # 0 … 9 inclusive
LAMBDA_SET  = [5, 15, 30, 45, 60]
TAU_SET     = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]

EPOCHS      = 60
WARMUP      = 5
BATCH       = 64

# Friedman-specific parameters
N_SAMPLES   = 1000                  # total dataset size
NOISE_STD   = 0.0                   # Gaussian noise sigma

RUNS_ROOT   = Path("runs")

# Synthetic correlation sanity study
SYN_FUNCS = {
    "linear":    lambda x: x,
    "quadratic": lambda x: x * x,
    "sine":      lambda x: np.sin(x),
}
SYN_SIGMA = [0.01, 0.1, 1.0]
SYN_N     = 1000

# Timing benchmark
TIMING_STEPS = 100

# --------------------------------------------------------------------------- #
#  Session directories & logging                                              #
# --------------------------------------------------------------------------- #
STAMP        = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR  = RUNS_ROOT / f"{STAMP}_friedman_exec"
FIG_DIR      = SESSION_DIR / "figures"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = SESSION_DIR / "session.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("exec")

log.info("===== Friedman execution started =====")
log.info("Outputs will be written to %s", SESSION_DIR.resolve())

# Basic disk-space sanity
free_mb = shutil.disk_usage(SESSION_DIR).free / 1_048_576
if free_mb < 50:
    log.error("Less than 50 MB free disk space - aborting.")
    sys.exit(1)

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def run_one_friedman(seed: int, use_xi: bool, lam: float, tau: float) -> Dict[str, float]:
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

    # Build argparse-like namespace expected by run_friedman.main()
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
        n_samples=N_SAMPLES,   # NEW arguments for Friedman script
        noise=NOISE_STD,
    )

    run_friedman_main(args)

    if not summary_csv.exists():
        raise RuntimeError(f"Missing metrics_summary.csv in {outdir}")

    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    if not np.isfinite(list(row.values())).all():
        raise RuntimeError(f"Non-finite metric in {outdir}")

    # Quick sanity threshold: R² should not be terrible
    if row["test_r2"] < 0.40:
        log.warning(
            "%sLow performance: seed=%d xi=%s  R2=%.3f%s",
            RED,
            seed,
            use_xi,
            row["test_r2"],
            RESET,
        )

    return row


def paired_t(baseline: pd.Series, xi: pd.Series) -> Dict[str, float]:
    diff = xi.values - baseline.values
    n = diff.shape[0]
    mean_diff = diff.mean()
    sd = diff.std(ddof=1)
    sem = sd / math.sqrt(n) if n > 0 else float("nan")
    t_val = mean_diff / sem if sem > 0 else float("inf")
    p_two = 2 * (1 - sp_stats.t.cdf(abs(t_val), df=n - 1)) if sem > 0 else 0.0
    return dict(n=n, mean=mean_diff, sd=sd, sem=sem, t=t_val, p=p_two)


# --------------------------------------------------------------------------- #
#  1. Experiment grid                                                         #
# --------------------------------------------------------------------------- #
rows: List[Dict[str, float]] = []

total_runs_expected = len(SEEDS) * (1 + len(LAMBDA_SET) * len(TAU_SET))
total_runs_done = 0

for seed in SEEDS:
    # Baseline (Xi off) first
    rows.append(run_one_friedman(seed, False, 0.0, 0.0))
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
            rows.append(run_one_friedman(seed, True, lam, tau))
            total_runs_done += 1
            remaining = total_runs_expected - total_runs_done
            log.info(
                "run %d completed, %d remaining. est minutes left: %.1f",
                total_runs_done,
                remaining,
                (remaining * 0.5) / 60,
            )

log.info("Friedman runs completed - total %d models", total_runs_done)
all_df = pd.DataFrame(rows)
all_df.to_csv(SESSION_DIR / "all_metrics.csv", index=False)

# --------------------------------------------------------------------------- #
#  2. Statistics: paired t-test                                               #
# --------------------------------------------------------------------------- #
best_rows = []
for seed in SEEDS:
    # -------- baseline (Xi off) --------
    base = all_df[(all_df.seed == seed) & (all_df.use_xi == 0)].iloc[0]
    best_rows.append(
        {"seed": seed, "variant": "baseline", "test_hard_xi": base.test_hard_xi}
    )

    # -------- Xi models for this seed --------
    sub = all_df[(all_df.seed == seed) & (all_df.use_xi == 1)]

    # choose the λ,τ with the highest *validation* xi
    winner = sub.loc[sub.val_xi.idxmax()]
    best_rows.append(
        {"seed": seed, "variant": "xi", "test_hard_xi": winner.test_hard_xi}
    )

best_df = pd.DataFrame(best_rows)

# paired t-test: Xi vs baseline, one row per seed
xi_scores   = best_df[best_df.variant == "xi"].sort_values("seed").test_hard_xi
base_scores = best_df[best_df.variant == "baseline"].sort_values("seed").test_hard_xi
stats_res   = paired_t(base_scores, xi_scores)

stats_path = SESSION_DIR / "stats_summary.txt"
with stats_path.open("w") as fh:
    fh.write(f"Paired t-test on hard xi (n={stats_res['n']})\n")
    fh.write(f"mean diff  {stats_res['mean']:.4f}\n")
    fh.write(f"t = {stats_res['t']:.4f},  p = {stats_res['p']:.5f}\n")

log.info("Stats written to %s", stats_path.name)

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
t_xi   = timing_variant(True)
over_pct = 100.0 * (t_xi - t_base) / t_base

tim_df = pd.DataFrame(
    [{"baseline_s": t_base, "xi_s": t_xi, "overhead_pct": over_pct}]
)
tim_df.to_csv(SESSION_DIR / "timing_bench.csv", index=False)

# --------------------------------------------------------------------------- #
#  5. Figure generation                                                       #
# --------------------------------------------------------------------------- #
# Heat map mean xi by lambda and tau
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
plt.colorbar(label="mean hard xi")
plt.xlabel("tau")
plt.ylabel("lambda")
plt.title("Xi dependency by lambda and tau")
plt.tight_layout()
plt.savefig(FIG_DIR / "heatmap_xi.png", dpi=120)
plt.close()

# === EXTRA HEATMAPS: MSE and R2
metrics = {
    "test_mse": dict(
        title="Mean Test MSE (lower is better)",
        cmap ="viridis_r"),   # reversed so low = bright
    "test_r2":  dict(
        title="Mean Test R2 (higher is better)",
        cmap ="viridis"),
}

for metric, cfg in metrics.items():
    pivot = (
        all_df
        .groupby(["lambda_coef", "tau"])[metric]
        .mean()
        .unstack()
        .sort_index()
        .reindex(sorted(all_df.tau.unique(), key=float), axis=1)
    )

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot, aspect="auto", origin="lower",
                    cmap=cfg["cmap"], interpolation="nearest")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(im, label=f"mean {metric}")
    plt.xlabel("tau")
    plt.ylabel("lambda")
    plt.title(cfg["title"])
    plt.tight_layout()
    fname = f"heatmap_{metric}.png"
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
    SESSION_DIR / "synth_metrics.csv",
    SESSION_DIR / "timing_bench.csv",
    FIG_DIR / "heatmap_xi.png",
]
missing = [p.name for p in required if not p.exists()]
if missing:
    log.error("Missing artefacts: %s", ", ".join(missing))
    sys.exit(1)

log.info("EXECUTION COMPLETE - ALL ARTEFACTS READY")