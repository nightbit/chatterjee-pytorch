# run_diabetes.py
import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
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


def load_diabetes_dataset() -> pd.DataFrame:
    """Return the diabetes dataset as a DataFrame (features + target)."""
    # The frame already includes the 'target' column; copy to break linkage.
    return load_diabetes(as_frame=True).frame.copy()


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
    """Run one epoch of train or eval and return (total, mse, xi)."""
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


# ------------------------------ Plotting ------------------------------


def plot_learning_curves(history: dict, out_png: Path) -> None:
    epochs = np.arange(1, len(history["val_mse"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["val_mse"], label="Val MSE")
    plt.twinx()
    plt.plot(epochs, history["val_hard_xi"], "g--", label="Val xi_hard")
    plt.ylabel("Val xi_hard")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=96)
    plt.close()


def plot_scatter(truth: np.ndarray, pred: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.scatter(truth, pred, s=8, alpha=0.6)
    lims = [truth.min(), truth.max()]
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
    df = load_diabetes_dataset()
    train_df, val_df, test_df = random_split_df(df, 0.80, 0.10, seed=args.seed)

    train_loader, val_loader, test_loader, _ = prepare_tensors(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    in_dim = train_loader.dataset.tensors[0].shape[1]
    model = MLP(in_dim).to(device)

    # ---------- Criterion ----------
    if args.use_xi:
        criterion = XiLoss(tau=args.tau, lambda_=args.lambda_coef)
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

    history = {"train_mse": [], "train_xi": [], "val_mse": [], "val_xi": []}
    best_val_mse = float("inf")
    checkpoints_dir = Path(args.outdir) / "checkpoints"
    make_outdir(checkpoints_dir)

    # ---------- Training ----------
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        if args.use_xi and epoch <= args.warmup_epochs:
            criterion.lambda_ = 0.0
        elif args.use_xi:
            criterion.lambda_ = args.lambda_coef

        model.train()
        tr_total, tr_mse, tr_xi = run_epoch(
            model, train_loader, criterion, device, optimizer, grad_clip=args.grad_clip
        )

        model.eval()
        with torch.no_grad():
            val_total, val_mse, val_xi = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )

        _, _, val_hard_xi = evaluate_hard_xi(model, val_loader, device)
        history["train_mse"].append(tr_mse)
        history["train_xi"].append(tr_xi)
        history["val_mse"].append(val_mse)
        history["val_xi"].append(val_xi)
        history.setdefault("val_hard_xi", []).append(val_hard_xi)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), checkpoints_dir / "best.pt")

        if epoch % 10 == 0 or epoch == args.epochs:
            tqdm.write(f"Epoch {epoch:03d}/{args.epochs} | Val MSE {val_mse:.4f} | Val xi {val_xi:.4f}")

    # ---------- Test ----------
    model.load_state_dict(torch.load(checkpoints_dir / "best.pt"))
    preds, truths, hard_xi = evaluate_hard_xi(model, test_loader, device)

    mse_test = np.mean((preds - truths) ** 2)
    mae_test = np.mean(np.abs(preds - truths))
    r2_test = 1.0 - mse_test / np.var(truths, ddof=0)

    baseline = np.full_like(truths, truths.mean())
    mse_baseline = np.mean((baseline - truths) ** 2)
    r2_baseline = 1.0 - mse_baseline / np.var(truths, ddof=0)

    print(f"[BASELINE]  MSE {mse_baseline:.4f} | R2 {r2_baseline:.4f}")
    print(f"[MODEL   ]  MSE {mse_test:.4f} | R2 {r2_test:.4f}")

    # ---------- Save raw arrays ----------
    make_outdir(Path(args.outdir))
    np.save(Path(args.outdir) / "preds.npy", preds)
    np.save(Path(args.outdir) / "truths.npy", truths)

    # ---------- Logging ----------
    summary_csv = Path(args.outdir) / "metrics_summary.csv"
    header = [
        "seed",
        "use_xi",
        "lambda_coef",
        "tau",
        "test_mse",
        "test_mae",
        "test_r2",
        "test_hard_xi",
    ]
    row = [
        args.seed,
        int(args.use_xi),
        args.lambda_coef if args.use_xi else 0.0,
        args.tau if args.use_xi else 0.0,
        mse_test,
        mae_test,
        r2_test,
        hard_xi,
    ]
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, args.epochs + 1),
            "train_mse": history["train_mse"],
            "train_xi": history["train_xi"],
            "val_mse": history["val_mse"],
            "val_xi": history["val_xi"],
        }
    )
    hist_df.to_csv(Path(args.outdir) / "history.csv", index=False)

    plot_learning_curves(history, Path(args.outdir) / "learning_curves.png")
    title = "Xi Model" if args.use_xi else "Baseline"
    plot_scatter(truths, preds, Path(args.outdir) / "scatter_pred_vs_true.png", title)

    print(f"[DONE] Test MSE {mse_test:.4f} | Hard xi {hard_xi:.4f}")
    print(f"Outputs saved to {args.outdir}")


# ------------------------------ CLI ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Diabetes regression experiment")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)