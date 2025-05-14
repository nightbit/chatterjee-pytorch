import torch
import torch.nn as nn
import torch.optim as optim
import torchsort
import random

import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ---------------------------------------------------------------------------
# Import XiLoss from your existing implementation
# ---------------------------------------------------------------------------
from losses.xi_loss import XiLoss

# ---------------------------------------------------------------------------
# Synthetic Data Generators
# ---------------------------------------------------------------------------
def generate_linear(n, noise_std):
    """Y = 3X + 5 + noise."""
    x = torch.linspace(-2, 2, n)
    y = 3.0 * x + 5.0 + noise_std * torch.randn(n)
    return x.unsqueeze(-1), y.unsqueeze(-1)

def generate_quadratic(n, noise_std):
    """Y = X^2 + noise."""
    x = torch.linspace(-2, 2, n)
    y = x.pow(2) + noise_std * torch.randn(n)
    return x.unsqueeze(-1), y.unsqueeze(-1)

def generate_sinusoidal(n, noise_std):
    """Y = sin(X) + noise."""
    x = torch.linspace(-3.14, 3.14, n)
    y = torch.sin(x) + noise_std * torch.randn(n)
    return x.unsqueeze(-1), y.unsqueeze(-1)

# ---------------------------------------------------------------------------
# Simple MLP
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Training / Testing Flow
# ---------------------------------------------------------------------------
def train_synthetic(
    function_name: str = "linear",
    n_samples: int = 256,
    noise_std: float = 0.1,
    device: torch.device | str = "cpu",
    warmup_epochs: int = 5,
    total_epochs: int = 30,
    lr: float = 1e-3,
    tau: float = 0.1,
    lambda_: float = 1.0,
):
    """
    Train an MLP on synthetic data with XiLoss as a regularizer after warm-up.
    Returns final MSE and Xi.
    """
    # 1) Data generation
    if function_name.lower() == "linear":
        x, y = generate_linear(n_samples, noise_std)
    elif function_name.lower() == "quadratic":
        x, y = generate_quadratic(n_samples, noise_std)
    elif function_name.lower() == "sinusoidal":
        x, y = generate_sinusoidal(n_samples, noise_std)
    else:
        raise ValueError(f"Unknown function {function_name}")

    x, y = x.to(device), y.to(device)

    # 2) Model & Loss
    model = SimpleMLP(input_dim=1, hidden_dim=64).to(device)
    xi_loss = XiLoss(tau=tau, lambda_=lambda_, task_loss_fn=nn.MSELoss()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3) Train
    for epoch in range(1, total_epochs + 1):
        # forward
        y_pred = model(x)
        if epoch <= warmup_epochs:
            # Xi turned off during warm-up
            total_loss = nn.MSELoss()(y_pred, y)
            xi_val = torch.tensor(0.0, device=device)
        else:
            total_loss, xi_val = xi_loss(y_pred, y)

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # log
        mse_val = nn.MSELoss()(y_pred, y).item()
        print(f"Epoch {epoch:02d}/{total_epochs} | MSE={mse_val:.6f} | Xi={xi_val.item():.6f} | TotalLoss={total_loss.item():.6f}")

    # 4) Final values
    final_mse = mse_val
    final_xi = xi_val.item()
    return final_mse, final_xi

# ---------------------------------------------------------------------------
# MAIN (Optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # fix seeds for reproducibility
    random.seed(0)
    torch.manual_seed(0)

    # Example usage
    final_mse, final_xi = train_synthetic(
        function_name="sinusoidal",
        n_samples=512,
        noise_std=0.1,
        device="cpu",
        warmup_epochs=5,
        total_epochs=30,
        lr=1e-3,
        tau=0.1,
        lambda_=1.0
    )
    print(f"\nDone! Final MSE={final_mse}, Xi={final_xi}")
"""
Epoch 01/30 | MSE=0.433492 | Xi=0.000000 | TotalLoss=0.433492
Epoch 02/30 | MSE=0.366550 | Xi=0.000000 | TotalLoss=0.366550
Epoch 03/30 | MSE=0.310941 | Xi=0.000000 | TotalLoss=0.310941
Epoch 04/30 | MSE=0.267309 | Xi=0.000000 | TotalLoss=0.267309
Epoch 05/30 | MSE=0.234159 | Xi=0.000000 | TotalLoss=0.234159
Epoch 06/30 | MSE=0.210137 | Xi=0.554777 | TotalLoss=-0.344640
Epoch 07/30 | MSE=0.193889 | Xi=0.554777 | TotalLoss=-0.360889
Epoch 08/30 | MSE=0.184950 | Xi=0.554777 | TotalLoss=-0.369827
Epoch 09/30 | MSE=0.181214 | Xi=0.554777 | TotalLoss=-0.373563
Epoch 10/30 | MSE=0.180693 | Xi=0.554777 | TotalLoss=-0.374084
Epoch 11/30 | MSE=0.181863 | Xi=0.554777 | TotalLoss=-0.372914
Epoch 12/30 | MSE=0.183065 | Xi=0.554777 | TotalLoss=-0.371712
Epoch 13/30 | MSE=0.183119 | Xi=0.554777 | TotalLoss=-0.371658
Epoch 14/30 | MSE=0.181597 | Xi=0.554777 | TotalLoss=-0.373180
Epoch 15/30 | MSE=0.178449 | Xi=0.554777 | TotalLoss=-0.376329
Epoch 16/30 | MSE=0.173840 | Xi=0.554777 | TotalLoss=-0.380938
Epoch 17/30 | MSE=0.168126 | Xi=0.554777 | TotalLoss=-0.386651
Epoch 18/30 | MSE=0.161881 | Xi=0.554777 | TotalLoss=-0.392896
Epoch 19/30 | MSE=0.155619 | Xi=0.554777 | TotalLoss=-0.399158
Epoch 20/30 | MSE=0.149745 | Xi=0.554777 | TotalLoss=-0.405032
Epoch 21/30 | MSE=0.144657 | Xi=0.554777 | TotalLoss=-0.410121
Epoch 22/30 | MSE=0.140571 | Xi=0.554777 | TotalLoss=-0.414206
Epoch 23/30 | MSE=0.137185 | Xi=0.554777 | TotalLoss=-0.417593
Epoch 24/30 | MSE=0.134246 | Xi=0.554777 | TotalLoss=-0.420531
Epoch 25/30 | MSE=0.131601 | Xi=0.554777 | TotalLoss=-0.423177
Epoch 26/30 | MSE=0.129089 | Xi=0.554777 | TotalLoss=-0.425688
Epoch 27/30 | MSE=0.126563 | Xi=0.554777 | TotalLoss=-0.428215
Epoch 28/30 | MSE=0.123911 | Xi=0.554777 | TotalLoss=-0.430866
Epoch 29/30 | MSE=0.121123 | Xi=0.555670 | TotalLoss=-0.434547
Epoch 30/30 | MSE=0.118346 | Xi=0.551733 | TotalLoss=-0.433387

Done! Final MSE=0.11834610998630524, Xi=0.5517332553863525
"""