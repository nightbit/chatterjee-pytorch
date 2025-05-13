import torch
import torch.nn as nn
import torch.optim as optim
import torchsort
import random

# ---------------------------------------------------------------------------
# XiLoss Implementation (straight-through for ranking) to keep everything local
# ---------------------------------------------------------------------------
_TIE_EPS = 1e-6

class XiLoss(nn.Module):
    def __init__(self, tau: float = 0.1, lambda_: float = 1.0,
                 task_loss_fn: nn.Module | None = None,
                 epsilon: float = _TIE_EPS):
        super().__init__()
        self.tau = float(tau)
        self.lambda_ = float(lambda_)
        self.epsilon = float(epsilon)
        self.task_loss = task_loss_fn or nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shape.")
        if y_true.numel() < 2:
            raise ValueError("Xi requires at least 2 samples.")
        if torch.allclose(y_true, y_true[0]):
            raise ValueError("y_true is constant — xi is undefined.")

        # flatten to 1-D
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        n = y_true.numel()

        # minimal tie-breaking for y_true
        if self.epsilon > 0.0:
            with torch.no_grad():
                y_true = y_true + (torch.rand_like(y_true) - 0.5) * self.epsilon

        # reorder predictions by sorted true y
        perm = torch.argsort(y_true, dim=0)
        y_pred_ord = y_pred[perm]

        # minimal tie-breaking for y_pred_ord
        if self.epsilon > 0.0:
            with torch.no_grad():
                y_pred_ord = y_pred_ord + (torch.rand_like(y_pred_ord) - 0.5) * self.epsilon

        # hard ranks for forward pass
        hard_ranks = torch.argsort(torch.argsort(y_pred_ord, dim=0), dim=0).float() + 1.0

        # soft ranks for backward pass
        soft_ranks = torchsort.soft_rank(
            y_pred_ord.unsqueeze(0),
            regularization="l2",
            regularization_strength=self.tau
        ).squeeze(0)

        # straight-through
        ranks = hard_ranks + (soft_ranks - soft_ranks.detach())

        # compute xi
        diffs = torch.abs(ranks[1:] - ranks[:-1]).sum()
        xi_soft = 1.0 - 3.0 * diffs / (n**2 - 1)

        # total loss
        task = self.task_loss(y_pred, y_true)
        total = task - self.lambda_ * xi_soft
        return total, xi_soft

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