#!/usr/bin/env python3
"""
End-to-end synthetic experiments for Chatterjee xi_n regularizer.
Run with:  python run_synthetic.py
Creates results/synthetic_summary.csv + two PNG plots.
"""

import os, math, json, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from xi_loss import XiLoss, xi_hard   # local module

# ---------- constants ----------
SEED = 42
BATCH = 256 if torch.cuda.is_available() else 128
EPOCHS_WARMUP = 5
EPOCHS_MAIN   = 50
SAMPLES = 4096
SIGMAS = [0.01, 0.10, 1.00]

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)

# ---------- helper: synthetic generators ----------
def gen_linear(n, sigma):
    x = np.random.uniform(-2.0, 2.0, size=n)
    y = 2.0 * x + 1.0 + np.random.normal(0.0, sigma, size=n)
    return x, y

def gen_quadratic(n, sigma):
    x = np.random.uniform(-2.0, 2.0, size=n)
    y = x * x + np.random.normal(0.0, sigma, size=n)
    return x, y

def gen_sine(n, sigma):
    x = np.random.uniform(0.0, 2.0 * math.pi, size=n)
    y = np.sin(x) + np.random.normal(0.0, sigma, size=n)
    return x, y

GENS = {
    "linear": gen_linear,
    "quadratic": gen_quadratic,
    "sine": gen_sine,
}

# ---------- model ----------
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

# ---------- train loop ----------
def run_one(function_name, sigma):
    x_np, y_np = GENS[function_name](SAMPLES, sigma)
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x_np, y_np, train_size=0.70, random_state=SEED)
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, train_size=0.50, random_state=SEED)

    def to_tensor(a):
        return torch.as_tensor(a, dtype=torch.float32).view(-1, 1).to(DEVICE)

    x_train, y_train = to_tensor(x_train), to_tensor(y_train)
    x_val,   y_val   = to_tensor(x_val),   to_tensor(y_val)
    x_test,  y_test  = to_tensor(x_test),  to_tensor(y_test)

    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xi_loss_fn = XiLoss(tau=0.1, lambda_=1.0).to(DEVICE)

    def batch_indices(n):
        idx = torch.randperm(n)
        for i in range(0, n, BATCH):
            yield idx[i : i + BATCH]

    # warm-up (lambda = 0)
    xi_loss_fn.lambda_ = 0.0
    for epoch in range(EPOCHS_WARMUP):
        for idx in batch_indices(len(x_train)):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            loss.backward()
            optimizer.step()

    # main (lambda = 1)
    xi_loss_fn.lambda_ = 1.0
    for epoch in range(EPOCHS_MAIN):
        for idx in batch_indices(len(x_train)):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train[idx])
            total_loss, _ = xi_loss_fn(pred, y_train[idx])
            total_loss.backward()
            optimizer.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        test_mse = F.mse_loss(test_pred, y_test).item()
        test_xi = xi_hard(test_pred.squeeze(), y_test.squeeze()).item()

        val_pred = model(x_val)
        val_mse = F.mse_loss(val_pred, y_val).item()
        val_xi = xi_hard(val_pred.squeeze(), y_val.squeeze()).item()

    return {
        "function": function_name,
        "sigma": sigma,
        "test_mse": test_mse,
        "test_xi_hard": test_xi,
        "val_mse": val_mse,
        "val_xi_hard": val_xi,
    }

# ---------- main ----------
def main():
    all_results = []
    t0 = time.time()
    for f in GENS:
        for s in SIGMAS:
            print(f"[*] {f}  sigma={s}")
            res = run_one(f, s)
            all_results.append(res)
    dt = time.time() - t0
    print(f"Finished in {dt/60:.1f} minutes on {DEVICE.type.upper()}")

    df = pd.DataFrame(all_results)
    csv_path = OUTDIR / "synthetic_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # --------- plots ----------
    for metric, fname in [("test_xi_hard", "fig_xi_vs_noise.png"),
                          ("test_mse", "fig_mse_vs_noise.png")]:
        plt.figure()
        for f in GENS:
            subset = df[df["function"] == f]
            xs = subset["sigma"]
            ys = subset[metric]
            plt.plot(xs, ys, marker="o", label=f)
        plt.xlabel("Noise sigma")
        plt.ylabel(metric)
        plt.title(metric + " vs noise")
        plt.legend()
        out_path = OUTDIR / fname
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()