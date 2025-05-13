#!/usr/bin/env python3
"""
Synthetic dependency experiments for Chatterjee xi_n.
Runs one complete training+evaluation job in a single file.

Usage:
    python run_synthetic.py --func linear   --sigma 0.1
    python run_synthetic.py --func quad     --sigma 0.01
    python run_synthetic.py --func sine     --sigma 1.0
"""

import argparse
from pathlib import Path
import time
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  # headless back-end
import matplotlib.pyplot as plt
import pandas as pd

import sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from xi_loss import XiLoss, xi_hard  # local import


# --------------------------- Config constants --------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 10000            # total points per experiment
TRAIN_FRAC = 0.8             # remainder goes to validation
BATCH_SIZE = 256             # drops to 128 automatically if OOM
EPOCHS_WARMUP = 5
EPOCHS_MAIN = 45
LR = 1e-3
TAU = 0.1                    # soft-rank reg strength
LAMBDA = 1.0                 # xi weight during main phase
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
# ----------------------------------------------------------------------- #


def generate_dataset(func: str, sigma: float):
    """Return tensors x, y of shape [N_SAMPLES]."""
    x = torch.empty(N_SAMPLES).uniform_(-2.0, 2.0)
    noise = torch.randn(N_SAMPLES) * sigma
    if func == "linear":
        y = 2.0 * x + 0.5 + noise
    elif func == "quad":
        y = x.pow(2) + noise
    elif func == "sine":
        y = torch.sin(3.0 * x) + noise
    else:
        raise ValueError(f"Unknown func '{func}'")
    return x, y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    total_xi_soft = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb).squeeze(1)
        loss, xi_soft = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_xi_soft += xi_soft.item() * xb.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_xi_soft / n


@torch.no_grad()
def evaluate(model, x_val, y_val):
    model.eval()
    preds = model(x_val.to(DEVICE)).squeeze(1).cpu()
    mse = torch.nn.functional.mse_loss(preds, y_val).item()
    xi_h = xi_hard(preds, y_val).item()
    return mse, xi_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func",  choices=["linear", "quad", "sine"],
                        required=True, help="True dependency type")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Gaussian noise std dev")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.func}_sigma{args.sigma}".replace(".", "")
    csv_path = out_dir / f"log_{tag}.csv"
    fig_path = out_dir / f"plot_{tag}.png"
    # ------------------------------------------------------------------ #

    # 1. data
    x, y = generate_dataset(args.func, args.sigma)
    idx = torch.randperm(N_SAMPLES)
    n_train = int(TRAIN_FRAC * N_SAMPLES)
    train_x, train_y = x[idx[:n_train]], y[idx[:n_train]]
    val_x, val_y = x[idx[n_train:]], y[idx[n_train:]]
    train_ds = TensorDataset(train_x.unsqueeze(1), train_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. model + loss
    model = MLP().to(DEVICE)
    mse_loss = nn.MSELoss()
    xi_loss = XiLoss(tau=TAU, lambda_=0.0, task_loss_fn=mse_loss)  # warm-up
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    logs = []
    start_time = time.time()

    # 3a. warm-up epochs (lambda=0)
    for epoch in range(1, EPOCHS_WARMUP + 1):
        train_loss, train_xi = train_one_epoch(model, train_loader, xi_loss, optimizer)
        val_mse, val_xi_h = evaluate(model, val_x, val_y)
        logs.append((epoch, 0.0, train_loss, train_xi, val_mse, val_xi_h))
        print(f"Epoch {epoch:02d} warmup  train_loss={train_loss:.4f}  "
              f"val_mse={val_mse:.4f}")

    # 3b. main epochs (lambda=1)
    xi_loss.lambda_ = LAMBDA
    for epoch in range(EPOCHS_WARMUP + 1, EPOCHS_WARMUP + EPOCHS_MAIN + 1):
        train_loss, train_xi = train_one_epoch(model, train_loader, xi_loss, optimizer)
        val_mse, val_xi_h = evaluate(model, val_x, val_y)
        logs.append((epoch, LAMBDA, train_loss, train_xi, val_mse, val_xi_h))
        print(f"Epoch {epoch:02d} main    train_xi={train_xi:.4f}  "
              f"val_xi_h={val_xi_h:.4f}  val_mse={val_mse:.4f}")

    total_time = time.time() - start_time
    print(f"Finished in {total_time/60:.1f} min on {DEVICE.type}")

    # 4. save CSV log
    df = pd.DataFrame(logs,
        columns=["epoch","lambda","train_loss","train_xi_soft","val_mse","val_xi_hard"])
    df.to_csv(csv_path, index=False)
    print(f"Log saved to {csv_path}")

    # 5. save figure
    plt.figure()
    plt.title(f"{args.func}, sigma={args.sigma}")
    plt.plot(df["epoch"], df["train_xi_soft"], label="train xi soft")
    plt.plot(df["epoch"], df["val_xi_hard"], label="val xi hard")
    plt.ylabel("xi_n")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # handle out-of-memory gracefully
        if "out of memory" in str(e).lower():
            print("OOM on GPU – retrying on CPU with batch size 128")
            BATCH_SIZE = 128
            DEVICE = torch.device("cpu")
            main()
        else:
            raise