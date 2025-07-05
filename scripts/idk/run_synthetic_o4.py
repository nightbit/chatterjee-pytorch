#!/usr/bin/env python3
"""
Day 3 smoke-test: train an MLP on a simple synthetic sin(X)+noise task,
using XiLoss(ξₙ) as a regularizer. Prints epoch-by-epoch MSE & ξₙ
to verify everything wires up and gradients flow.
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# import your XiLoss from day 2
from losses.xi_loss import XiLoss

# ── 1) Config & seeds ────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description="Day 3: Synthetic XiLoss smoke-test")
parser.add_argument("--function", choices=["linear","quadratic","sin"], default="sin")
parser.add_argument("--sigma", type=float, default=0.1,
                    help="Gaussian noise stddev")
parser.add_argument("--n-samples", type=int, default=1000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=10,
                    help="Total epochs (incl. warm-up)")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=0.1,
                    help="soft_rank regularization strength")
parser.add_argument("--lambda_", type=float, default=1.0,
                    help="coefficient on ξₙ term")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 2) Synthetic data generator ──────────────────────────────────────────────

def make_synthetic(func: str, n: int, sigma: float):
    X = np.random.uniform(-2, 2, size=(n,1)).astype(np.float32)
    if func == "linear":
        Y = 3.0*X + 5.0
    elif func == "quadratic":
        Y = X**2
    elif func == "sin":
        Y = np.sin(X)
    else:
        raise ValueError(func)
    Y += np.random.normal(0, sigma, size=Y.shape).astype(np.float32)
    return X, Y

# generate & split 80/20
X, Y = make_synthetic(args.function, args.n_samples, args.sigma)
idx = np.arange(args.n_samples)
np.random.shuffle(idx)
split = int(0.8 * args.n_samples)
train_idx, val_idx = idx[:split], idx[split:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]

train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                          torch.from_numpy(Y_train))
val_ds   = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                          torch.from_numpy(Y_val))
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,
                                           batch_size=args.batch_size,
                                           shuffle=False)

# ── 3) Model & loss ─────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
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

model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# XiLoss returns (total_loss, xi_soft)
loss_fn = XiLoss(tau=args.tau, lambda_=args.lambda_,
                 task_loss_fn=nn.MSELoss()).to(device)

# ── 4) Training loop ─────────────────────────────────────────────────────────

def evaluate(loader):
    model.eval()
    mse_tot = 0.0
    xi_tot = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            pred, yb = model(xb), yb
            total, xi_val = loss_fn(pred, yb)
            mse = nn.functional.mse_loss(pred, yb)
            # note: xi_val comes from forward call
            batch = xb.size(0)
            mse_tot += mse.item() * batch
            xi_tot  += xi_val.item() * batch
            count  += batch
    return mse_tot / count, xi_tot / count

for epoch in range(1, args.epochs+1):
    model.train()
    running_mse = 0.0
    running_xi  = 0.0
    batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        pred  = model(xb)
        total, xi_val = loss_fn(pred, yb)
        total.backward()
        optimizer.step()

        # track
        running_mse += nn.functional.mse_loss(pred, yb).item()
        running_xi  += xi_val.item()
        batches += 1

    train_mse = running_mse / batches
    train_xi  = running_xi  / batches
    val_mse, val_xi = evaluate(val_loader)

    print(f"Epoch {epoch:2d} | "
          f"Train MSE: {train_mse:.4f}, Train ξₙ: {train_xi:.4f} | "
          f" Val MSE: {val_mse:.4f},  Val ξₙ: {val_xi:.4f}")

print("✅ Day 3 smoke-test complete.")
