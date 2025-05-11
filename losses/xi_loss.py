#xi_loss.py
import torch
import torch.nn as nn
import torchsort

class XiLoss(nn.Module):
    def __init__(self, tau: float = 0.1, lambda_: float = 1.0,
                 task_loss_fn: nn.Module | None = None):
        super().__init__()
        self.tau = float(tau)
        self.lambda_ = float(lambda_)
        # default regression task loss: MSE
        self.task_loss = task_loss_fn or nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shape.")
        if y_true.numel() < 2:
            raise ValueError("Xi requires at least 2 samples.")
        if torch.allclose(y_true, y_true[0]):
            raise ValueError("y_true is constant — ξₙ undefined.")

        # flatten to 1-D
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        n = y_true.numel()

        # argsort by y_true (no grad needed)
        perm = torch.argsort(y_true, dim=0)
        y_pred_ord = y_pred[perm]

        # differentiable ranks of predictions
        ranks = torchsort.soft_rank(
            y_pred_ord.unsqueeze(0),
            regularization="l2",
            regularization_strength=self.tau,
        ).squeeze(0)

        # ξₙ
        diffs = torch.abs(ranks[1:] - ranks[:-1]).sum()
        xi_soft = 1.0 - 3.0 * diffs / (n ** 2 - 1)

        # total loss (maximize ξₙ ⇒ subtract)
        task = self.task_loss(y_pred, y_true)
        total = task - self.lambda_ * xi_soft
        return total, xi_soft


def xi_hard(x: torch.Tensor, y: torch.Tensor):
    if x.shape != y.shape:
        raise ValueError("Shapes differ.")
    n = x.numel()
    if n < 2:
        raise ValueError("Need at least 2")
    if torch.allclose(y, y[0]):
        raise ValueError("y is constant.")

    # reorder by x
    idx = torch.argsort(x, dim=0)
    y_ord = y[idx]
    # ranks of y_ord (ties: average rank)
    ranks = torch.argsort(torch.argsort(y_ord))
    # +1 to make it 1-based like soft_rank
    ranks = ranks.to(dtype=torch.float64) + 1
    xi = 1.0 - 3.0 * torch.abs(ranks[1:] - ranks[:-1]).sum() / (n**2 - 1)
    return xi.to(dtype=x.dtype)