import torch
import torch.nn as nn
import torchsort
_TIE_EPS = 1e-6  # magnitude of the random jitter used to break exact ties

class XiLoss(nn.Module):
    def __init__(self, tau: float = 0.1, lambda_: float = 1.0,
                 task_loss_fn: nn.Module | None = None,
                 epsilon: float = _TIE_EPS):          # NEW
        super().__init__()
        self.tau = float(tau)
        self.lambda_ = float(lambda_)
        self.epsilon = float(epsilon)                # NEW
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

        # ----- minimal tie-breaking for y_true -----
        if self.epsilon > 0.0:
            with torch.no_grad():  # keep jitter out of autograd graph
                y_true = y_true + (torch.rand_like(y_true) - 0.5) * self.epsilon

        # argsort by y_true (no grad needed)
        perm = torch.argsort(y_true, dim=0)
        y_pred_ord = y_pred[perm]

        # ----- minimal tie-breaking for y_pred_ord -----
        if self.epsilon > 0.0:
            noise = ((torch.rand_like(y_pred_ord) - 0.5) * self.epsilon).detach()
            y_pred_ord = y_pred_ord + noise   # KEEP AUTOGRAD PATH
            
# -----------------------------------------------


        # exact ranks (hard) for forward pass
        hard_ranks = torch.argsort(torch.argsort(y_pred_ord, dim=0), dim=0)
        hard_ranks = hard_ranks.to(y_pred_ord.dtype) + 1.0

        # soft ranks (differentiable proxy) for backward pass
        soft_ranks = torchsort.soft_rank(
            y_pred_ord.unsqueeze(0),
            regularization="l2",
            regularization_strength=self.tau,
        ).squeeze(0)

        # straight-through combination:
        #   forward uses hard_ranks, backward uses soft_ranks
        ranks = hard_ranks + (soft_ranks - soft_ranks.detach())

        # compute ξₙ
        diffs = torch.abs(ranks[1:] - ranks[:-1]).sum()
        xi_soft = 1.0 - 3.0 * diffs / (n ** 2 - 1)

        # total loss (maximize ξₙ ⇒ subtract)
        task = self.task_loss(y_pred, y_true)
        total = task - self.lambda_ * xi_soft
        return total, xi_soft


def xi_hard(x: torch.Tensor, y: torch.Tensor, epsilon: float = _TIE_EPS):
    if x.shape != y.shape:
        raise ValueError("Shapes differ.")
    n = x.numel()
    if epsilon > 0.0:
        with torch.no_grad():
            y = y + (torch.rand_like(y) - 0.5) * epsilon
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    if torch.allclose(y, y[0]):
        raise ValueError("y is constant.")

    # reorder by x
    idx = torch.argsort(x, dim=0)
    y_ord = y[idx]
    if epsilon > 0.0:
        with torch.no_grad():
            y_ord = y_ord + (torch.rand_like(y_ord) - 0.5) * epsilon

    # hard ranks of y_ord (0-based → +1 for 1-based)
    ranks = torch.argsort(torch.argsort(y_ord, dim=0), dim=0)
    ranks = ranks.to(dtype=torch.float64) + 1.0

    xi = 1.0 - 3.0 * torch.abs(ranks[1:] - ranks[:-1]).sum() / (n ** 2 - 1)
    return xi.to(dtype=x.dtype)