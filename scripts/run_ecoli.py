# run_ecoli.py
"""
Regression experiment on E. coli growth (A2 task):
Predict 24h OD from Temperature (°C), Drug ∈ {ERY, TMP}, and Concentration.

This runner mirrors run_diabetes.py as closely as possible. Only the data
loading / preprocessing and minor labeling differ. Everything else (model,
loss integration, training loop, logging, plots, outputs) stays the same.

Key architectural decisions (implemented here):
- Aggregate technical/biological replicates BEFORE any train/val/test split.
- Keep purely numeric features; one-hot encode the drug factor (both ERY,TMP).
- Use random splits (i.i.d. tabular) after replicate aggregation.
- Use the same MLP, Xi loss integration (lambda/tau), and evaluation metrics.

Expected CSV schema (flexible; auto-detection + CLI overrides provided):
    - Temperature column: one of ["temperature", "temp", "temp_c", "t"]
    - Concentration column: one of ["concentration", "conc", "dose"]
    - Drug column: one of ["drug", "antibiotic"]
    - OD/response column: one of ["od_24h", "od", "response", "growth"]

Drugs must map to {ERY, TMP}. We normalize e.g. "erythromycin" -> "ERY",
"trimethoprim" -> "TMP". If other levels are present, we error out to avoid
silently training on the wrong task.

Outputs (same as run_diabetes.py):
    - preds.npy, truths.npy
    - metrics_summary.csv
    - history.csv
    - learning_curves.png
    - scatter_pred_vs_true.png
    - checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def _autodetect_column(
    cols_lower: List[str],
    candidates: List[str],
    provided: Optional[str],
    name_for_error: str,
) -> str:
    """
    Return the actual column name in the original DataFrame (not lowercased),
    using a provided override if given; else first matching candidate by
    case-insensitive comparison. Raise ValueError if not found.
    """
    if provided is not None:
        if provided.lower() in cols_lower:
            # Return the original-cased column name
            idx = cols_lower.index(provided.lower())
            return original_cols[idx]
        raise ValueError(
            f"Column override '{provided}' for {name_for_error} not found. "
            f"Available columns: {original_cols}"
        )

    for cand in candidates:
        if cand in cols_lower:
            idx = cols_lower.index(cand)
            return original_cols[idx]

    raise ValueError(
        f"Could not find a column for {name_for_error}. "
        f"Looked for {candidates}. Available columns: {original_cols}"
    )


def _normalize_drug(val: str) -> str:
    """
    Normalize drug labels to {'ERY','TMP'}.
    Accepts e.g. 'ery', 'erythromycin' -> ERY; 'tmp', 'trimethoprim' -> TMP.
    """
    s = str(val).strip().lower()
    if s in {"ery", "erythro", "erythromycin"} or "erythro" in s:
        return "ERY"
    if s in {"tmp", "trimethoprim"} or "trimeth" in s:
        return "TMP"
    # Occasionally datasets use uppercase already
    if s.upper() in {"ERY", "TMP"}:
        return s.upper()
    return s.upper()  # fallthrough; will be validated later


def load_ecoli_conc_dataset(
    csv_path: Path,
    col_temp: Optional[str] = None,
    col_conc: Optional[str] = None,
    col_drug: Optional[str] = None,
    col_od: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the E. coli (A2) concentration sweep dataset and return a DataFrame with:
        numeric features + 'target' column.
    Steps:
        1) Read CSV
        2) Detect/normalize columns
        3) Keep only ERY and TMP; normalize labels
        4) Convert to numeric, drop rows with NaNs after parsing
        5) Aggregate replicates by (drug, temperature, concentration)
        6) One-hot encode 'drug' to numeric features with both columns present
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    if df_raw.empty:
        raise ValueError("Input CSV is empty.")

    # Case-insensitive column detection
    global original_cols  # used by _autodetect_column
    original_cols = list(df_raw.columns)
    cols_lower = [c.lower().strip() for c in original_cols]

    temp_col = _autodetect_column(
        cols_lower, ["temperature", "temp", "temp_c", "t"], col_temp, "temperature"
    )
    conc_col = _autodetect_column(
        cols_lower, ["concentration", "conc", "dose"], col_conc, "concentration"
    )
    drug_col = _autodetect_column(
        cols_lower, ["drug", "antibiotic"], col_drug, "drug/antibiotic"
    )
    od_col = _autodetect_column(
        cols_lower, ["od_24h", "od", "response", "growth"], col_od, "OD/response"
    )

    # Normalize & coerce types
    df = df_raw[[temp_col, conc_col, drug_col, od_col]].copy()
    df.rename(
        columns={
            temp_col: "temp_c",
            conc_col: "concentration",
            drug_col: "drug",
            od_col: "od",
        },
        inplace=True,
    )

    # Drug normalization and filtering
    df["drug"] = df["drug"].apply(_normalize_drug)
    valid = df["drug"].isin({"ERY", "TMP"})
    if not valid.any():
        raise ValueError(
            "No rows found for drugs ERY/TMP after normalization. "
            "Check your column or file selection."
        )
    if (~valid).any():
        # Be strict: the A2 task must only contain ERY/TMP
        unknown = sorted(df.loc[~valid, "drug"].unique().tolist())
        raise ValueError(
            f"Found unexpected drug labels {unknown}. "
            "A2 task requires only ERY and TMP."
        )
    # Ensure categorical with fixed levels so dummies include both columns
    df["drug"] = pd.Categorical(df["drug"], categories=["ERY", "TMP"], ordered=False)

    # Numeric parsing
    for c in ["temp_c", "concentration", "od"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing numeric data
    before = len(df)
    df = df.dropna(subset=["temp_c", "concentration", "od"])
    if len(df) < before:
        print(f"[WARN] Dropped {before - len(df)} rows with non-numeric entries.")

    # Aggregate replicates (mean response per condition)
    agg = (
        df.groupby(["drug", "temp_c", "concentration"], as_index=False)
        .agg(target=("od", "mean"), n_rep=("od", "size"))
        .sort_values(["drug", "temp_c", "concentration"])
        .reset_index(drop=True)
    )

    # One-hot encode drug with both columns present
    dummies = pd.get_dummies(agg["drug"], prefix="drug", drop_first=False)
    # Guarantee both columns exist (even if one level is absent due to filtering)
    for level in ["ERY", "TMP"]:
        col = f"drug_{level}"
        if col not in dummies.columns:
            dummies[col] = 0

    # Final numeric frame
    out = pd.concat(
        [agg[["temp_c", "concentration"]].reset_index(drop=True), dummies, agg[["target", "n_rep"]]],
        axis=1,
    )

    # Sanity checks
    if not np.isfinite(out[["temp_c", "concentration", "target"]].values).all():
        raise ValueError("Non-finite values found after aggregation.")
    if out.shape[0] < 50:
        print(
            f"[WARN] Very small number of unique conditions ({out.shape[0]}). "
            "Check the input file/filters."
        )

    # Reorder columns to be deterministic
    feature_cols = ["temp_c", "concentration", "drug_ERY", "drug_TMP"]
    out = out[feature_cols + ["target", "n_rep"]]
    return out


def random_split_df(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way random split with reproducibility (post-aggregation)."""
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
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, List[str]]:
    """
    Convert aggregated & encoded DataFrames to DataLoaders.
    All features are numeric; target is OD mean per condition.
    """
    feature_cols = [c for c in train_df.columns if c not in {"target", "n_rep"}]
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
        feature_cols,
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
    lims = [min(truth.min(), pred.min()), max(truth.max(), pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True OD (24h)")
    plt.ylabel("Predicted OD")
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
    data_csv = Path(args.data_csv)
    df_all = load_ecoli_conc_dataset(
        data_csv,
        col_temp=args.col_temp,
        col_conc=args.col_conc,
        col_drug=args.col_drug,
        col_od=args.col_od,
    )

    # Basic dataset report
    n_rows = len(df_all)
    n_rep_mean = float(df_all["n_rep"].mean())
    print(
        f"[DATA] Conditions: {n_rows} | mean replicates per condition: {n_rep_mean:.2f} "
        f"| min/max OD: {df_all['target'].min():.3f} / {df_all['target'].max():.3f}"
    )

    # Split AFTER aggregation
    train_df, val_df, test_df = random_split_df(df_all, 0.80, 0.10, seed=args.seed)

    # Prepare tensors (numeric features only)
    train_loader, val_loader, test_loader, scaler, feature_cols = prepare_tensors(
        train_df, val_df, test_df, batch_size=args.batch_size
    )
    in_dim = len(feature_cols)

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
        history["val_hard_xi"].append(val_hard_xi)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), checkpoints_dir / "best.pt")

        if epoch % 10 == 0 or epoch == args.epochs:
            tqdm.write(f"Epoch {epoch:03d}/{args.epochs} | Val MSE {val_mse:.4f} | Val xi {val_xi:.4f}")

    # ---------- Test ----------
    model.load_state_dict(torch.load(checkpoints_dir / "best.pt", map_location=device))
    preds, truths, hard_xi = evaluate_hard_xi(model, test_loader, device)

    mse_test = float(np.mean((preds - truths) ** 2))
    mae_test = float(np.mean(np.abs(preds - truths)))
    r2_test = float(1.0 - mse_test / np.var(truths, ddof=0))

    baseline = np.full_like(truths, truths.mean())
    mse_baseline = float(np.mean((baseline - truths) ** 2))
    r2_baseline = float(1.0 - mse_baseline / np.var(truths, ddof=0))

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
        "val_mse",
        "val_xi",
        "test_mse",
        "test_mae",
        "test_r2",
        "test_hard_xi",
        "n_conditions",
        "mean_reps",
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
        n_rows,
        n_rep_mean,
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
    title = "E. coli A2 • Xi Model" if args.use_xi else "E. coli A2 • Baseline"
    plot_scatter(truths, preds, Path(args.outdir) / "scatter_pred_vs_true.png", title)

    print(f"[DONE] Test MSE {mse_test:.4f} | Hard xi {hard_xi:.4f}")
    print(f"Outputs saved to {args.outdir}")


# ------------------------------ CLI ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E. coli (A2) regression experiment")
    p.add_argument("--data_csv", type=str, required=True, help="Path to concentration-sweep CSV (ERY & TMP only)")
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

    # Optional column overrides for robustness
    p.add_argument("--col_temp", type=str, default=None, help="Override temperature column name")
    p.add_argument("--col_conc", type=str, default=None, help="Override concentration column name")
    p.add_argument("--col_drug", type=str, default=None, help="Override drug/antibiotic column name")
    p.add_argument("--col_od", type=str, default=None, help="Override OD/response column name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)