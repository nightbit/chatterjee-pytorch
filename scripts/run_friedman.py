# run_friedman.py
import argparse
import csv
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------- Local import ----------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from losses.xi_loss import XiLoss, xi_hard  # noqa: E402


# ------------------------------ Utils ------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU


def make_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------ Data ------------------------------


def load_friedman1_dataset(
    n_samples: int,
    n_features: int,
    noise: float,
    random_state: int | None,
) -> pd.DataFrame:
    """Return the Friedman #1 synthetic data set as DataFrame (features + target)."""
    X, y = make_friedman1(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    col_names = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=col_names)
    df["target"] = y.astype(np.float32)
    return df


def random_split_df(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way random split with reproducibility."""
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1, "Ratios must be between 0 and 1"
    assert train_ratio + val_ratio < 1, "Ratios must sum to < 1"

    train_val_df, test_df = train_test_split(
        df,
        test_size=1.0 - (train_ratio + val_ratio),
        random_state=seed,
        shuffle=True,
    )

    rel_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=rel_val_ratio,
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    feature_cols = [c for c in train_df.columns if c != "target"]
    scaler = StandardScaler().fit(train_df[feature_cols])

    def _df_to_tensor(df: pd.DataFrame) -> TensorDataset:
        X = scaler.transform(df[feature_cols]).astype(np.float32)
        y = df["target"].values.astype(np.float32).reshape(-1, 1)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    train_ds = _df_to_tensor(train_df)
    val_ds = _df_to_tensor(val_df)
    test_ds = _df_to_tensor(test_df)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        scaler,
    )


# ------------------------------ Model ------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------ Train / Eval ------------------------------


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> Tuple[float, float, float]:
    """Run one epoch of train or eval and return (total, mse, xi_soft)."""
    running_total = running_mse = running_xi = 0.0
    count = 0
    mode = "Train" if optimizer else "Eval"

    for X, y in tqdm(dataloader, desc=mode, leave=False):
        X, y = X.to(device), y.to(device)

        if optimizer:
            optimizer.zero_grad()

        out = model(X)
        total, xi_soft = criterion(out, y)
        mse = criterion.task_loss(out, y)

        if optimizer:
            total.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch = y.size(0)
        running_total += total.item() * batch
        running_mse += mse.item() * batch
        running_xi += xi_soft.item() * batch
        count += batch

    return running_total / count, running_mse / count, running_xi / count


@torch.no_grad()
def evaluate_hard_xi(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    preds, truths = [], []
    for X, y in loader:
        X = X.to(device)
        preds.append(model(X).cpu().numpy())
        truths.append(y.numpy())
    preds = np.concatenate(preds).flatten()
    truths = np.concatenate(truths).flatten()
    xi = xi_hard(torch.from_numpy(preds), torch.from_numpy(truths)).item()
    return preds, truths, xi


def compute_test_metrics(preds: np.ndarray, truths: np.ndarray) -> Tuple[float, float, float]:
    mse = float(np.mean((preds - truths) ** 2))
    mae = float(np.mean(np.abs(preds - truths)))
    r2 = float(1.0 - mse / np.var(truths, ddof=0))
    return mse, mae, r2


# ------------------------------ Plotting ------------------------------


def plot_learning_curves(history: dict, out_png: Path) -> None:
    epochs = np.arange(1, len(history["val_mse"]) + 1)
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    l1, = ax1.plot(epochs, history["val_mse"], label="Val MSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MSE")
    ax2 = ax1.twinx()
    l2, = ax2.plot(epochs, history["val_hard_xi"], "g--", label="Val xi_hard")
    ax2.set_ylabel("Val xi_hard")
    ax1.legend(handles=[l1, l2], loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=96)
    plt.close()


def plot_scatter(truth: np.ndarray, pred: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.scatter(truth, pred, s=8, alpha=0.6)
    lims = [float(min(truth.min(), pred.min())), float(max(truth.max(), pred.max()))]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True target")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=96)
    plt.close()


# ------------------------------ Main ------------------------------


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Using device: {device}")

    # ---------- Data ----------
    # IMPORTANT: fix the dataset across seeds for valid NB correction
    df = load_friedman1_dataset(
        n_samples=args.n_samples,
        n_features=10,
        noise=args.noise,
        random_state=0,  # fixed dataset; seed only affects splits & init
    )
    train_df, val_df, test_df = random_split_df(df, 0.80, 0.10, seed=args.seed)

    train_loader, val_loader, test_loader, _ = prepare_tensors(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    in_dim = train_loader.dataset.tensors[0].shape[1]
    model = MLP(in_dim).to(device)

    # ---------- Criterion ----------
    if args.use_xi:
        criterion = XiLoss(tau=args.tau, lambda_=0.0)  # start with warmup lambda=0
    else:
        mse_loss = nn.MSELoss()

        class _Wrap(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.task_loss = mse_loss

            def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
                return self.task_loss(y_pred, y_true), torch.tensor(0.0, device=y_pred.device)

        criterion = _Wrap()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_mse": [], "train_xi": [], "val_mse": [], "val_xi": [], "val_hard_xi": []}

    checkpoints_dir = Path(args.outdir) / "checkpoints"
    make_outdir(checkpoints_dir)

    # ---------- Dual-selection trackers ----------
    best_val_mse = float("inf")
    best_val_hard_xi = float("-inf")
    best_epoch_mse = -1
    best_epoch_xi_post = -1

    # Track "any-epoch" xi best to warn if it came from warm-up (when xi is enabled)
    best_val_hard_xi_any = float("-inf")
    best_epoch_xi_any = -1

    best_mse_path = checkpoints_dir / "best_mse.pt"
    best_xi_path = checkpoints_dir / "best_xi.pt"
    alias_best_path = checkpoints_dir / "best.pt"  # alias to best_mse for backward-compat

    # ---------- Training ----------
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        # Warmup: zero-out lambda for the first warmup epochs (only if xi enabled)
        if hasattr(criterion, "lambda_") and args.use_xi:
            criterion.lambda_ = 0.0 if epoch <= args.warmup_epochs else args.lambda_coef

        model.train()
        tr_total, tr_mse, tr_xi_soft = run_epoch(
            model, train_loader, criterion, device, optimizer, grad_clip=args.grad_clip
        )

        model.eval()
        with torch.no_grad():
            val_total, val_mse, val_xi_soft = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )

        # Hard xi on validation for selection-by-dependence
        _, _, val_hard_xi = evaluate_hard_xi(model, val_loader, device)

        history["train_mse"].append(tr_mse)
        history["train_xi"].append(tr_xi_soft)
        history["val_mse"].append(val_mse)
        history["val_xi"].append(val_xi_soft)
        history["val_hard_xi"].append(val_hard_xi)

        # ---- Selection: best by validation MSE ----
        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            best_epoch_mse = epoch
            torch.save(model.state_dict(), best_mse_path)
            # keep alias aligned
            shutil.copyfile(best_mse_path, alias_best_path)

        # ---- Selection: best by validation hard xi ----
        # - If xi regularization is enabled, exclude warm-up epochs (epoch <= warmup)
        #   from the xi-based selection to avoid picking a "baseline" snapshot.
        # - Always track the "any-epoch" xi best for diagnostic warning.
        if val_hard_xi > best_val_hard_xi_any:
            best_val_hard_xi_any = float(val_hard_xi)
            best_epoch_xi_any = epoch

        xi_selection_allowed = (not args.use_xi) or (epoch > args.warmup_epochs and getattr(criterion, "lambda_", 0) > 0)

        if xi_selection_allowed and val_hard_xi > best_val_hard_xi:
            best_val_hard_xi = float(val_hard_xi)
            best_epoch_xi_post = epoch
            torch.save(model.state_dict(), best_xi_path)

        if epoch % 10 == 0 or epoch == args.epochs:
            tqdm.write(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Val MSE {val_mse:.4f} | "
                f"Val xi_soft {val_xi_soft:.4f} | "
                f"Val xi_hard {val_hard_xi:.4f}"
            )

    # If xi was enabled and the best xi overall came from warm-up, warn that we excluded it.
    if args.use_xi and best_epoch_xi_any != -1 and best_epoch_xi_any <= args.warmup_epochs:
        print(
            f"[WARN] Highest validation xi_hard occurs during warm-up epoch {best_epoch_xi_any} "
            f"(lambda=0). Xi-selected checkpoint uses best post-warm-up epoch {best_epoch_xi_post}."
        )

    # Safety: ensure xi checkpoint exists (in degenerate cases pick mse checkpoint)
    if not best_xi_path.exists():
        shutil.copyfile(best_mse_path, best_xi_path)
        best_val_hard_xi = history["val_hard_xi"][best_epoch_mse - 1] if best_epoch_mse > 0 else float("nan")
        best_epoch_xi_post = best_epoch_mse
        print("[INFO] Using MSE-selected checkpoint as xi-selected fallback.")

    # ---------- Test: evaluate BOTH checkpoints ----------
    def _eval_checkpoint(ckpt_path: Path) -> Tuple[float, float, float, float]:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        preds, truths, hard_xi = evaluate_hard_xi(model, test_loader, device)
        mse, mae, r2 = compute_test_metrics(preds, truths)
        return mse, mae, r2, hard_xi

    # RQ2: predictive accuracy — model selected by best validation MSE
    test_mse_mseSel, test_mae_mseSel, test_r2_mseSel, test_hard_xi_mseSel = _eval_checkpoint(best_mse_path)
    # RQ3: dependence fidelity — model selected by best validation hard xi
    test_mse_xiSel, test_mae_xiSel, test_r2_xiSel, test_hard_xi_xiSel = _eval_checkpoint(best_xi_path)

    # Simple constant baseline for context (not used downstream)
    model.load_state_dict(torch.load(best_mse_path, map_location=device))
    preds_mseSel, truths_mseSel, _ = evaluate_hard_xi(model, test_loader, device)
    baseline = np.full_like(truths_mseSel, truths_mseSel.mean())
    mse_baseline = float(np.mean((baseline - truths_mseSel) ** 2))
    r2_baseline = float(1.0 - mse_baseline / np.var(truths_mseSel, ddof=0))

    print(f"[BASELINE (const mean)]  MSE {mse_baseline:.4f} | R2 {r2_baseline:.4f}")
    print(f"[MSE-selected  ]  MSE {test_mse_mseSel:.4f} | R2 {test_r2_mseSel:.4f} | xi_hard {test_hard_xi_mseSel:.4f}")
    print(f"[XI-selected   ]  MSE {test_mse_xiSel:.4f} | R2 {test_r2_xiSel:.4f} | xi_hard {test_hard_xi_xiSel:.4f}")

    # ---------- Save raw arrays (only for MSE-selected to limit churn) ----------
    make_outdir(Path(args.outdir))
    np.save(Path(args.outdir) / "preds.npy", preds_mseSel)
    np.save(Path(args.outdir) / "truths.npy", truths_mseSel)

    # ---------- Logging ----------
    summary_csv = Path(args.outdir) / "metrics_summary.csv"

    # Keep legacy columns AND add new ones (order chosen for readability).
    header = [
        # Identity / config
        "seed",
        "use_xi",
        "lambda_coef",
        "tau",
        # Legacy epoch-end validation (kept for compatibility)
        "val_mse",
        "val_xi",
        # Selection summaries (NEW)
        "best_val_mse",
        "best_val_hard_xi",
        # Test metrics for MSE-selected checkpoint (NEW)
        "test_mse_mseSel",
        "test_mae_mseSel",
        "test_r2_mseSel",
        "test_hard_xi_mseSel",
        # Test metrics for xi-selected checkpoint (NEW)
        "test_mse_xiSel",
        "test_mae_xiSel",
        "test_r2_xiSel",
        "test_hard_xi_xiSel",
    ]

    # Last-epoch validation (legacy)
    last_val_mse = history["val_mse"][-1] if history["val_mse"] else float("nan")
    last_val_xi_soft = history["val_xi"][-1] if history["val_xi"] else float("nan")

    row = [
        args.seed,
        int(args.use_xi),
        args.lambda_coef if args.use_xi else 0.0,
        args.tau if args.use_xi else 0.0,
        last_val_mse,
        last_val_xi_soft,
        best_val_mse,
        best_val_hard_xi,
        test_mse_mseSel,
        test_mae_mseSel,
        test_r2_mseSel,
        test_hard_xi_mseSel,
        test_mse_xiSel,
        test_mae_xiSel,
        test_r2_xiSel,
        test_hard_xi_xiSel,
    ]

    write_header = not summary_csv.exists()
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, args.epochs + 1),
            "train_mse": history["train_mse"],
            "train_xi": history["train_xi"],
            "val_mse": history["val_mse"],
            "val_xi": history["val_xi"],
            "val_hard_xi": history["val_hard_xi"],
        }
    )
    hist_df.to_csv(Path(args.outdir) / "history.csv", index=False)

    plot_learning_curves(history, Path(args.outdir) / "learning_curves.png")
    # Scatter only for MSE-selected
    plot_scatter(truths_mseSel, preds_mseSel, Path(args.outdir) / "scatter_pred_vs_true.png", "MSE-selected")

    print(f"[DONE] MSE-selected Test MSE {test_mse_mseSel:.4f} | xi_hard {test_hard_xi_mseSel:.4f}")
    print(f"[DONE] Xi-selected  Test MSE {test_mse_xiSel:.4f} | xi_hard {test_hard_xi_xiSel:.4f}")
    print(f"Outputs saved to {args.outdir}")


# ------------------------------ CLI ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Friedman #1 regression experiment")
    p.add_argument("--outdir", type=str, required=True, help="Directory to write outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_xi", action="store_true", help="Enable Xi regularizer")
    p.add_argument("--lambda_coef", type=float, default=1.0, help="Lambda for Xi")
    p.add_argument("--tau", type=float, default=0.1, help="Soft-rank tau")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--n_samples", type=int, default=100, help="Total samples for Friedman #1")
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian noise STD in Friedman #1")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_outdir(Path(args.outdir))
    main(args)