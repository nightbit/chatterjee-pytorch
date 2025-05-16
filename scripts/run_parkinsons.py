# run_parkinsons.py
import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ---------- Local import ----------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from losses.xi_loss import XiLoss, xi_hard


# ------------------------------ Utils ------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU


def make_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------ Data ------------------------------


def load_parkinsons(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def split_by_subject(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = df["subject#"].unique()
    rng.shuffle(subjects)

    n = subjects.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_subj = subjects[:n_train]
    val_subj = subjects[n_train : n_train + n_val]
    test_subj = subjects[n_train + n_val :]

    train_df = df[df["subject#"].isin(train_subj)].reset_index(drop=True)
    val_df = df[df["subject#"].isin(val_subj)].reset_index(drop=True)
    test_df = df[df["subject#"].isin(test_subj)].reset_index(drop=True)

    # Guard: no leakage
    dup = set(train_subj) & set(val_subj) | set(train_subj) & set(test_subj) | set(val_subj) & set(test_subj)
    assert not dup, "Subject leakage across splits"

    return train_df, val_df, test_df


def prepare_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    feature_cols = [c for c in train_df.columns if c not in ("subject#", "motor_UPDRS", "total_UPDRS")]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    def _df_to_tensor(df: pd.DataFrame) -> TensorDataset:
        X = scaler.transform(df[feature_cols]).astype(np.float32)
        y = df[target_col].values.astype(np.float32).reshape(-1, 1)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    train_ds = _df_to_tensor(train_df)
    val_ds = _df_to_tensor(val_df)
    test_ds = _df_to_tensor(test_df)

    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
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
) -> tuple[float, float, float]:
    """
    Returns per-sample averages:
        total_loss  – (MSE − λ·ξ_soft)
        mse         – pure MSE
        xi_soft     – soft ξₙ
    """
    running_total = 0.0
    running_mse = 0.0
    running_xi = 0.0
    count = 0

    mode = "Train" if optimizer is not None else "Eval"
    for X, y in tqdm(dataloader, desc=mode, leave=False):
        X = X.to(device)
        y = y.to(device)

        if optimizer:
            optimizer.zero_grad()

        out = model(X)
        total, xi_soft = criterion(out, y)
        mse = criterion.task_loss(out, y)

        if optimizer:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        batch = y.size(0)
        running_total += total.item() * batch
        running_mse += mse.item() * batch
        running_xi += xi_soft.item() * batch
        count += batch

    return running_total / count, running_mse / count, running_xi / count



@torch.no_grad()
def evaluate_hard_xi(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    preds, truths = [], []
    for X, y in loader:
        X = X.to(device)
        out = model(X)
        preds.append(out.cpu().numpy())
        truths.append(y.numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    truths = np.concatenate(truths, axis=0).flatten()
    xi = xi_hard(torch.from_numpy(preds), torch.from_numpy(truths)).item()
    return preds, truths, xi


# ------------------------------ Plotting ------------------------------


def plot_learning_curves(history: dict[str, list[float]], out_png: Path) -> None:
    epochs = np.arange(1, len(history["val_mse"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["val_mse"], label="Val MSE")
    plt.plot(epochs, history["val_xi"], label="Val xi_soft")
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
    plt.xlabel("True total_UPDRS")
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

    data_csv = Path(args.data_dir) / "parkinsons_tele.csv"
    df = load_parkinsons(data_csv)

    train_df, val_df, test_df = split_by_subject(df, 0.8, 0.1, args.seed)

    train_loader, val_loader, test_loader, scaler = prepare_tensors(train_df, val_df, test_df, args.target)

    in_dim = train_loader.dataset.tensors[0].shape[1]
    model = MLP(in_dim).to(device)

    # Criterion
    if args.use_xi:
        criterion = XiLoss(tau=args.tau, lambda_=args.lambda_coef)
    else:
        # ------- Baseline (no ξₙ) -------
        mse_loss = nn.MSELoss()

        class _Wrap(nn.Module):
            """MSE-only loss wrapper that mimics XiLoss’ interface."""
            def __init__(self) -> None:
                super().__init__()
                self.task_loss = mse_loss  # <-- expose for run_epoch

            def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
                # return (total_loss, xi_soft) exactly like XiLoss
                return self.task_loss(y_pred, y_true), torch.tensor(0.0, device=y_pred.device)

        criterion = _Wrap()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logs
    history = {"train_mse": [], "train_xi": [], "val_mse": [], "val_xi": []}

    best_val_mse = float("inf")
    checkpoints_dir = Path(args.outdir) / "checkpoints"
    make_outdir(checkpoints_dir)

    # ---------- Training loop with progress ----------
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        # λ schedule
        if args.use_xi and epoch <= args.warmup_epochs:
            criterion.lambda_ = 0.0
        elif args.use_xi:
            criterion.lambda_ = args.lambda_coef

        # Train
        model.train()
        tr_total, tr_mse, tr_xi = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        # Validate
        model.eval()
        with torch.no_grad():
            val_total, val_mse, val_xi = run_epoch(
                model, val_loader, criterion, device
            )

        # Logging
        history["train_mse"].append(tr_mse)
        history["train_xi"].append(tr_xi)
        history["val_mse"].append(val_mse)
        history["val_xi"].append(val_xi)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), checkpoints_dir / "best.pt")

        # print every 10 epochs (tqdm bar stays visible)
        if epoch % 10 == 0 or epoch == args.epochs:
            tqdm.write(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Val MSE {val_mse:.4f} | Val xi {val_xi:.4f}"
            )



    # Reload best
    model.load_state_dict(torch.load(checkpoints_dir / "best.pt"))

    # ---------- Test ----------
    preds, truths, hard_xi = evaluate_hard_xi(model, test_loader, device)
    mse_test = np.mean((preds - truths) ** 2)
    mae_test = np.mean(np.abs(preds - truths))
    r2_test = 1.0 - mse_test / np.var(truths, ddof=0)

    # ---------- Output ----------
    make_outdir(Path(args.outdir))
    # Metrics CSV (append or create)
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

    # Save learning curves
    plot_learning_curves(history, Path(args.outdir) / "learning_curves.png")
    # Scatter
    title = "Xi Model" if args.use_xi else "Baseline"
    plot_scatter(truths, preds, Path(args.outdir) / "scatter_pred_vs_true.png", title)

    print(f"[DONE] Test MSE {mse_test:.4f} | Hard xi {hard_xi:.4f}")
    print(f"Outputs saved to {args.outdir}")


# ------------------------------ CLI ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Parkinsons Tele-monitoring ξ_n experiment")
    p.add_argument("--data_dir", type=str, default="data", help="Directory containing parkinsons_tele.csv")
    p.add_argument("--outdir", type=str, required=True, help="Directory to write outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=str, default="total_UPDRS", choices=["total_UPDRS", "motor_UPDRS"])
    p.add_argument("--use_xi", action="store_true", help="Enable ξ_n regularizer")
    p.add_argument("--lambda_coef", type=float, default=1.0, help="λ for ξ_n")
    p.add_argument("--tau", type=float, default=0.1, help="Soft-rank τ")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
