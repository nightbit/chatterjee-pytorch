# run_lynxhare.py
import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


DEFAULT_PELT_CSV = (
    # Annual Hudson's Bay Company pelt records (Year,Hare,Lynx), 1845–1935.
    # Values are in thousands; source is the tidyverts/tsibbledata repository.
    "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/pelt/lynxhare.csv"
)


def load_lynxhare_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Lynx–Hare pelt dataset (Year, Hare, Lynx) as a DataFrame.

    If `csv_path` is None, a vetted CSV from the tsibbledata repo is used.
    The returned frame contains the raw columns and is sorted by Year ascending.
    """
    src = csv_path if csv_path else DEFAULT_PELT_CSV
    df = pd.read_csv(src)
    # Normalize column names just in case; keep canonical 'Year','Hare','Lynx'
    df.columns = [c.strip().title() for c in df.columns]
    assert {"Year", "Hare", "Lynx"}.issubset(
        set(df.columns)
    ), "CSV must contain Year, Hare, Lynx columns"
    df = df.sort_values("Year").reset_index(drop=True)
    return df


def make_supervised_frame_b2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a tabular supervised dataset for Task B2:
      Predict Lynx_{t+1} from [Hare_t, Lynx_t, Hare_{t-1}, Lynx_{t-1}].

    Returns a DataFrame with numeric feature columns and a 'target' column.
    Keeps the 'Year' column for potential inspection but excludes it from features downstream.
    """
    df = df.copy()
    # Current-year features
    df["hare_t"] = df["Hare"]
    df["lynx_t"] = df["Lynx"]
    # 1-year lag features (t-1)
    df["hare_t_1"] = df["Hare"].shift(1)
    df["lynx_t_1"] = df["Lynx"].shift(1)
    # Target is next year's Lynx (t+1)
    df["target"] = df["Lynx"].shift(-1)

    # Drop rows where lagged/lead values are undefined (first and last year)
    sup = df.dropna(subset=["hare_t", "lynx_t", "hare_t_1", "lynx_t_1", "target"]).reset_index(drop=True)
    # Enforce numeric dtypes
    feat_cols = ["hare_t", "lynx_t", "hare_t_1", "lynx_t_1"]
    sup[feat_cols + ["target"]] = sup[feat_cols + ["target"]].astype(float)
    return sup[["Year"] + feat_cols + ["target"]]


def chronological_split_df(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic chronological split (80/10/10-style) preserving order.
    Assumes `df` is already sorted by time ascending.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1, "Ratios must be between 0 and 1"
    assert train_ratio + val_ratio < 1, "Ratios must sum to < 1"

    n = len(df)
    n_train = int(np.floor(train_ratio * n))
    n_val = int(np.floor(val_ratio * n))
    n_test = n - n_train - n_val

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    assert len(train_df) + len(val_df) + len(test_df) == n
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0, "Empty split partition detected"
    return train_df, val_df, test_df


def prepare_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Prepare TensorDatasets and DataLoaders.
    Excludes 'Year' from features; expects a 'target' column.
    """
    feature_cols = [c for c in train_df.columns if c not in ("target", "Year")]
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
    epochs = np.arange(1, len(history.get("val_mse", [])) + 1)
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
    lims = [min(truth.min(), pred.min()), max(truth.max(), pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True target (Lynx_{t+1})")
    plt.ylabel("Predicted (Lynx_{t+1})")
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
    raw_df = load_lynxhare_dataset(args.data_csv)
    sup_df = make_supervised_frame_b2(raw_df)
    # Chronological 80/10/10 split (no shuffling)
    train_df, val_df, test_df = chronological_split_df(sup_df, 0.80, 0.10)

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

    history = {"train_mse": [], "train_xi": [], "val_mse": [], "val_xi": [], "val_hard_xi": []}
    best_val_mse = float("inf")
    checkpoints_dir = Path(args.outdir) / "checkpoints"
    make_outdir(checkpoints_dir)
    make_outdir(Path(args.outdir))

    # ---------- Training ----------
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        # warmup: disable xi until warmup_epochs
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
        history["val_hard_xi"].append(val_hard_xi)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), checkpoints_dir / "best.pt")

        if epoch % 10 == 0 or epoch == args.epochs:
            tqdm.write(f"Epoch {epoch:03d}/{args.epochs} | Val MSE {val_mse:.4f} | Val xi {val_xi:.4f}")

    # ---------- Test ----------
    model.load_state_dict(torch.load(checkpoints_dir / "best.pt", map_location=device))
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
    np.save(Path(args.outdir) / "preds.npy", preds)
    np.save(Path(args.outdir) / "truths.npy", truths)

    # ---------- Logging ----------
    summary_csv = Path(args.outdir) / "metrics_summary.csv"
    header = [
        "seed",
        "use_xi",
        "lambda_coef",
        "tau",
        "val_mse",
        "val_xi",
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
        history["val_mse"][-1],
        history["val_xi"][-1],
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
            "val_hard_xi": history["val_hard_xi"],
        }
    )
    hist_df.to_csv(Path(args.outdir) / "history.csv", index=False)

    plot_learning_curves(history, Path(args.outdir) / "learning_curves.png")
    title = "Xi Model (Lynx_{t+1})" if args.use_xi else "Baseline (Lynx_{t+1})"
    plot_scatter(truths, preds, Path(args.outdir) / "scatter_pred_vs_true.png", title)

    print(f"[DONE] Test MSE {mse_test:.4f} | Hard xi {hard_xi:.4f}")
    print(f"Samples: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Outputs saved to {args.outdir}")


# ------------------------------ CLI ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Lynx–Hare (B2) regression experiment")
    p.add_argument("--outdir", type=str, required=True, help="Directory to write outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_xi", action="store_true", help="Enable Xi regularizer")
    p.add_argument("--lambda_coef", type=float, default=1.0, help="Lambda for Xi")
    p.add_argument("--tau", type=float, default=0.1, help="Soft-rank tau")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument(
        "--data_csv",
        type=str,
        default=None,
        help="Optional path/URL to Year,Hare,Lynx CSV. Defaults to tsibbledata pelt CSV if omitted.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)