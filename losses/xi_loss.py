#xi_loss.py
import torch
import torch.nn as nn
import torchsort  # only used for soft_rank

__all__ = ["XiLoss", "xi_hard"]

def _soft_perm_1(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """
    NeuralSort soft permutation (Grover et al., ICLR 2019).
    Returns a unimodal row-stochastic matrix P̂ ∈ ℝ^{n×n} that
    in the τ→0 limit becomes the ascending-sort permutation by `scores`.
    """
    n = scores.size(0)
    device = scores.device

    # To sort ascending, flip sign (NeuralSort sorts descending by default)
    s = -scores

    # pairwise absolute differences A_ij = |s_i - s_j|
    diff = s.unsqueeze(1) - s.unsqueeze(0)       # (n,n)
    A_sum = diff.abs().sum(dim=1)               # (n,)

    # row-scaling weights: (n+1−2i) for i=1…n
    idx = torch.arange(1, n+1, device=device)   # (n,)
    coeff = (n + 1 - 2*idx).unsqueeze(1)        # (n,1)

    # build the score matrix for softmax
    # v[i,j] = coeff[i] * s[j]  -  A_sum[j]
    v = coeff * s.unsqueeze(0) - A_sum.unsqueeze(0)  # (n,n)

    # row-wise softmax
    P_hat = torch.softmax(v / tau, dim=1)       # (n,n)
    return P_hat

def _soft_perm_2(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Gaussian-kernel soft permutation based on soft ranks.
    As tau -> 0, P_hat -> exact permutation matrix sorting ascending `scores`.
    Otherwise, row i puts high weight on items whose soft‐rank is near i+1.

    Args:
        scores: 1-D tensor of shape (n,)
        tau   : temperature (> 0)

    Returns:
        P_hat: 2-D tensor of shape (n, n), row-stochastic
    """
    n = scores.size(0)
    # 1) compute soft‐ranks (shape 1 x n -> squeeze to n)
    r = torchsort.soft_rank(
        scores.unsqueeze(0),
        regularization_strength=tau
    ).squeeze(0)  # shape (n,)

    # 2) squared distance between each soft‐rank and each integer position 1..n
    pos = torch.arange(1, n + 1,
                       device=scores.device,
                       dtype=scores.dtype)  # shape (n,)
    dist2 = (r.unsqueeze(0) - pos.unsqueeze(1)) ** 2  # shape (n, n)

    # 3) row-wise softmax of −dist2/tau
    P_hat = torch.softmax(-dist2 / tau, dim=1)  # shape (n, n)
    return P_hat


def _soft_perm_3(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """
    NeuralSort soft permutation (Grover et al., ICLR 2019, Eq. 5).
    As tau -> 0, P_hat -> exact ascending‐sort permutation of `scores`.

    Args:
        scores: 1-D tensor of shape (n,)
        tau   : temperature (> 0)

    Returns:
        P_hat: 2-D tensor of shape (n, n), row-stochastic
    """
    n = scores.size(0)
    device = scores.device
    dtype = scores.dtype

    # 1) negate to sort ascending
    s = -scores  # shape (n,)

    # 2) sum of absolute differences for each j: sum_k |s_j - s_k|
    abs_diff = torch.abs(s.unsqueeze(1) - s.unsqueeze(0))  # shape (n, n)
    sum_abs = abs_diff.sum(dim=1)                         # shape (n,)

    # 3) compute coefficients (n + 1 - 2*i) for i = 1..n
    idx = torch.arange(1, n + 1, device=device, dtype=dtype)  # shape (n,)
    coeff = (n + 1 - 2 * idx).unsqueeze(1)                   # shape (n, 1)

    # 4) build score matrix v_{i,j} = coeff[i] * s[j] - sum_abs[j]
    v = coeff * s.unsqueeze(0) - sum_abs.unsqueeze(0)       # shape (n, n)

    # 5) row-wise softmax
    P_hat = torch.softmax(v / tau, dim=1)                   # shape (n, n)
    return P_hat


class XiLoss(nn.Module):
    """
    Differentiable Chatterjee ξₙ loss for mini‑batch training.

    L_total = L_task  –  λ · ξ̃ₙ(pred, target)

    Call returns (loss, xi_soft) so callers can both
    • back‑prop through 'loss'
    • log the detached ξ̃ₙ
    """
    def __init__(self, tau: float = 0.0000001, weight: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.weight = float(weight)

    @torch.no_grad()
    def set_tau(self, new_tau: float):
        self.tau = float(new_tau)

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim != 1 or target.ndim != 1:
            raise ValueError("XiLoss expects 1-D tensors (batch,)")
        if preds.size(0) != target.size(0):
            raise ValueError("preds and target must share batch size")

        n = preds.size(0)
        if n < 3:
            # ξₙ undefined / uninformative; skip
            dummy = torch.zeros((), dtype=preds.dtype, device=preds.device)
            return dummy, dummy

        # --- differentiable ξ̃ₙ (minimal fix) -------------------------
        # 1) soft-sort X to get soft permutation P
        P = _soft_perm(preds, self.tau)                  # (n, n)

        # 2) apply P to the raw targets -> soft-sorted Y
        y_sorted = (P @ target.unsqueeze(-1)).squeeze(-1)  # (n,)

        # 3) compute soft-ranks of the permuted Y
        r_sorted = torchsort.soft_rank(
            y_sorted.unsqueeze(0),
            regularization_strength=self.tau
        ).squeeze(0)                                       # (n,)

        # 4) sum of adjacent absolute differences
        diff_sum = torch.abs(r_sorted[1:] - r_sorted[:-1]).sum()

        # 5) final soft ξₙ
        xi_soft = 1.0 - 3.0 * diff_sum / (n * n - 1)

        # 6) loss to minimize
        loss = -self.weight * xi_soft

        return loss, xi_soft.detach()


# -------------------------------------------------------------------
#  Hard (non‑differentiable) ξₙ — O(n log n) for evaluation & tests
# -------------------------------------------------------------------
@torch.no_grad()
def xi_hard(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.ndim != 1 or y.ndim != 1 or x.size(0) != y.size(0):
        raise ValueError("xi_hard expects two 1‑D tensors of equal length")
    n = x.size(0)
    if n < 3:
        return 0.0
    # sort by x (ascending)
    idx = torch.argsort(x, stable=True)
    y_sorted = y[idx]
    # ranks of y
    r = torch.argsort(torch.argsort(y_sorted))
    diff = torch.abs(r[1:] - r[:-1]).sum().item()
    xi = 1.0 - 3.0 * diff / (n * n - 1)
    return xi

def xi_hard_general(x, y):
    idx   = torch.argsort(x, stable=True)
    y_ord = y[idx]
    r     = torch.argsort(torch.argsort(y_ord))
    # l_i = n - r_i   for zero‑based ranks with no ties in Y
    denom = n*(n*n - 1)/3
    diff  = torch.abs(r[1:] - r[:-1]).sum()
    return 1.0 - 3.0*diff / (n*n - 1)       # falls back to no‑ties formula