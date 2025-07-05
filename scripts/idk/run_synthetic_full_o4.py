#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from losses.xi_loss import XiLoss, xi_hard

#  ─── 2.1  Config ────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results/synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Synthetic functions
FUNCS = {
    "linear": lambda x: 2.0*x + 1.0,
    "quadratic": lambda x: x**2,
    "sine": lambda x: torch.sin(x),
}
NOISE_LEVELS = [0.01, 0.1, 1.0]
N_SAMPLES = 10_000
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
WARMUP_EPOCHS = 5
MODEL_HIDDEN = 64
SEED = 42

#  ─── 2.2  Helpers ──────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, MODEL_HIDDEN),
            nn.ReLU(),
            nn.Linear(MODEL_HIDDEN, MODEL_HIDDEN),
            nn.ReLU(),
            nn.Linear(MODEL_HIDDEN, 1),
        )
    def forward(self, x):
        return self.net(x)

def make_dataset(func, σ, n):
    x = torch.rand(n,1)*4 - 2    # uniform in [-2,2]
    y = func(x)
    y = y + torch.randn_like(y)*σ
    return x, y

def train_and_evaluate(func_name, σ):
    #  ─ Setup
    set_seed(SEED)
    func = FUNCS[func_name]
    x, y = make_dataset(func, σ, N_SAMPLES)
    split = int(0.8 * N_SAMPLES)
    x_train, y_train = x[:split].to(DEVICE), y[:split].to(DEVICE)
    x_val,   y_val   = x[split:].to(DEVICE), y[split:].to(DEVICE)

    model = MLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = XiLoss(tau=0.1, lambda_=1.0, task_loss_fn=nn.MSELoss())

    #  ─ Training loop
    history = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        idx = torch.randperm(split)
        batches = split // BATCH_SIZE
        epoch_losses = []
        for i in range(batches):
            batch_idx = idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            xb, yb = x_train[batch_idx], y_train[batch_idx]
            optimizer.zero_grad()
            total, xi_val = loss_fn(model(xb).squeeze(), yb.squeeze())
            total.backward()
            optimizer.step()
            epoch_losses.append((total.item(), xi_val.item()))
        #  ─ Warm-up toggle
        if epoch == WARMUP_EPOCHS+1:
            loss_fn.lambda_ = 1.0

        #  ─ Validation metrics
        model.eval()
        with torch.no_grad():
            y_pred = model(x_val).squeeze()
            mse = nn.MSELoss()(y_pred, y_val.squeeze()).item()
            xi_soft = loss_fn(model(x_val).squeeze(), y_val.squeeze())[1].item()
        history.append({
            "func": func_name, "noise": σ, "epoch": epoch,
            "train_loss": np.mean([t for t,x in epoch_losses]),
            "train_xi":   np.mean([x for t,x in epoch_losses]),
            "val_mse":    mse, "val_xi_soft": xi_soft
        })

    #  ─ Final Hard ξₙ on val set
    with torch.no_grad():
        xi_h = xi_hard(model(x_val).squeeze().cpu(), y_val.squeeze().cpu())
    return history, xi_h.item()

#  ─── 2.3  Main Loop ────────────────────────────────────────────
if __name__ == "__main__":
    all_hist = []
    hard_results = []
    for fname in FUNCS:
        for σ in NOISE_LEVELS:
            hist, xi_h_final = train_and_evaluate(fname, σ)
            all_hist.extend(hist)
            hard_results.append({"func": fname, "noise": σ, "xi_hard": xi_h_final})

    #  ─── 2.4  Save & Plot ────────────────────────────────────────
    df_hist = pd.DataFrame(all_hist)
    df_hard = pd.DataFrame(hard_results)
    df_hist.to_csv(f"{OUTPUT_DIR}/history.csv", index=False)
    df_hard.to_csv(f"{OUTPUT_DIR}/hard_xi.csv", index=False)

    # Plotting
    for metric, ylab in [("val_mse","Validation MSE"), ("val_xi_soft","Validation ξₙ (soft)")]:
        plt.figure()
        for fname in FUNCS:
            subset = df_hist[df_hist.func==fname].groupby("noise")[metric].nth(-1)
            plt.plot(NOISE_LEVELS, subset.values, marker='o', label=fname)
        plt.xlabel("Noise σ")
        plt.ylabel(ylab)
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/{metric}_by_noise.png")
        plt.close()

    # Hard-ξ plot
    plt.figure()
    for fname in FUNCS:
        subset = df_hard[df_hard.func==fname]
        plt.plot(NOISE_LEVELS, subset.xi_hard, marker='s', label=fname)
    plt.xlabel("Noise σ")
    plt.ylabel("Final ξₙ (hard)")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/hard_xi_by_noise.png")
    plt.close()

    print("✅ All synthetic experiments complete. Outputs in", OUTPUT_DIR)