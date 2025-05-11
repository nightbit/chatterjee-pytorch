import torch
import torch.nn as nn
import torchsort  # only used for soft_rank

__all__ = ["XiLoss", "xi_hard"]

def _soft_perm(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """
    NeuralSort soft permutation matrix (Grover et al., 2019)

    scores: (n,) 1‑D tensor
    tau   : temperature (>0)
    returns P̂ ∈ ℝ^{n×n} row‑stochastic
    """
    n = scores.size(0)
    scores_row = scores.unsqueeze(0)         # (1,n)
    scores_col = scores.unsqueeze(1)         # (n,1)
    pairwise_diff = scores_col - scores_row  # (n,n)
    P_hat = torch.softmax(-pairwise_diff / tau, dim=1)  # row‑wise
    return P_hat


class XiLoss(nn.Module):
    """
    Differentiable Chatterjee ξₙ loss for mini‑batch training.

    L_total = L_task  –  λ · ξ̃ₙ(pred, target)

    Call returns (loss, xi_soft) so callers can both
    • back‑prop through 'loss'
    • log the detached ξ̃ₙ
    """
    def __init__(self, tau: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.weight = float(weight)

    @torch.no_grad()
    def set_tau(self, new_tau: float):
        self.tau = float(new_tau)

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim != 1 or target.ndim != 1:
            raise ValueError("XiLoss expects 1‑D tensors (batch,)")
        if preds.size(0) != target.size(0):
            raise ValueError("preds and target must share batch size")

        n = preds.size(0)
        if n < 3:
            # ξₙ undefined / uninformative; skip
            dummy = torch.zeros((), dtype=preds.dtype, device=preds.device)
            return dummy, dummy

        # --- differentiable ξ̃ₙ ------------------------------------------
        P = _soft_perm(preds, self.tau)               # (n,n)
        r_y = torchsort.soft_rank(
            target.unsqueeze(0), 
            regularization_strength=self.tau
        ).squeeze(0)                                             # (n,)
        r_sorted = P @ r_y.unsqueeze(-1)              # (n,1)
        r_sorted = r_sorted.squeeze(-1)               # (n,)

        diff_sum = torch.abs(r_sorted[1:] - r_sorted[:-1]).sum()
        xi_soft = 1.0 - 3.0 * diff_sum / (n * n - 1)

        # optimiser minimises, hence negative sign
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
