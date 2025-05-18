#xi_loss.py
import torch
import torch.nn as nn
import torchsort

_TIE_EPS = 1e-6  # magnitude of random jitter for tie-breaking


def _softsort_matrix(x: torch.Tensor, tau: float, descending: bool = False) -> torch.Tensor:
    """Return the SoftSort permutation matrix P_tau(x).

    Args
    ----
    x : 1-D tensor (length n) – the vector to be “sorted”.
    tau : float > 0 – temperature of the relaxation.
    descending : if True, largest values go to position 0.

    Returns
    -------
    P : (n, n) row-stochastic matrix; row i gives the soft
        assignment of element x[i] to sorted positions.
    """
    sorted_x, _ = torch.sort(x, descending=descending)
    pairwise = -torch.abs(x.unsqueeze(-1) - sorted_x.unsqueeze(0))
    return torch.softmax(pairwise / tau, dim=-1)


class XiLoss(nn.Module):
    """Differentiable approximation of Chatterjee’s xi_n(pred, true).

    We compute xi_n(X = y_pred, Y = y_true).  Both the permutation that
    sorts X and the ranks of Y after that permutation are relaxed with
    SoftSort / SoftRank, yielding an end-to-end differentiable loss.
    """

    def __init__(
        self,
        tau: float = 0.1,
        lambda_: float = 1.0,
        task_loss_fn: nn.Module | None = None,
        epsilon: float = _TIE_EPS,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        self.lambda_ = float(lambda_)
        self.epsilon = float(epsilon)
        self.task_loss = task_loss_fn or nn.MSELoss()

    # --------------------------------------------------------------------- #
    # forward                                                               #
    # --------------------------------------------------------------------- #
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shape.")
        if y_true.numel() < 2:
            raise ValueError("xi requires at least two samples.")
        if torch.allclose(y_true, y_true[0]):
            raise ValueError("y_true is constant; xi_n undefined.")

        # flatten to 1-D
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        n = y_true.numel()

        # ---- tie-breaking jitter ----------------------------------------
        if self.epsilon > 0.0:
            with torch.no_grad():
                y_true = y_true + (torch.rand_like(y_true) - 0.5) * self.epsilon
            noise_pred = ((torch.rand_like(y_pred) - 0.5) * self.epsilon).detach()
            y_pred = y_pred + noise_pred

        # ---- soft permutation induced by y_pred -------------------------
        P = _softsort_matrix(y_pred, tau=self.tau, descending=False)  # (n, n)

        # ---- soft ranks of y_true ---------------------------------------
        soft_ranks_true = torchsort.soft_rank(
            y_true.unsqueeze(0),
            regularization="l2",
            regularization_strength=self.tau,
        ).squeeze(0)                                                   # (n,)

        # reorder ranks using transpose(P)
        ranks_ord = P.t().matmul(soft_ranks_true)                      # (n,)

        # ---- xi_n (soft) -------------------------------------------------
        diffs = torch.abs(ranks_ord[1:] - ranks_ord[:-1]).sum()
        xi_soft = 1.0 - 3.0 * diffs / (n ** 2 - 1)

        # ---- overall objective ------------------------------------------
        task = self.task_loss(y_pred, y_true)
        total = task - self.lambda_ * xi_soft
        return total, xi_soft


# ------------------------------------------------------------------------- #
# Reference (hard) implementation for evaluation / debugging               #
# ------------------------------------------------------------------------- #
def xi_hard(x: torch.Tensor, y: torch.Tensor, epsilon: float = _TIE_EPS) -> torch.Tensor:
    """Non-differentiable xi_n(x, y) that orders by x and ranks y."""
    if x.shape != y.shape:
        raise ValueError("Shapes differ.")
    n = x.numel()
    if n < 2:
        raise ValueError("Need at least two samples.")
    if torch.allclose(y, y[0]):
        raise ValueError("y is constant.")

    if epsilon > 0.0:
        with torch.no_grad():
            y = y + (torch.rand_like(y) - 0.5) * epsilon

    idx = torch.argsort(x, dim=0)          # hard permutation by x
    y_ord = y[idx]

    if epsilon > 0.0:
        with torch.no_grad():
            y_ord = y_ord + (torch.rand_like(y_ord) - 0.5) * epsilon

    ranks = torch.argsort(torch.argsort(y_ord, dim=0), dim=0).to(torch.float64) + 1.0
    xi = 1.0 - 3.0 * torch.abs(ranks[1:] - ranks[:-1]).sum() / (n ** 2 - 1)
    return xi.to(dtype=x.dtype)