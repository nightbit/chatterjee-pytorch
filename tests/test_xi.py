import pytest
import torch

from losses.xi_loss import XiLoss, xi_hard

torch.manual_seed(0)

def test_hard_perfect_monotonic():
    """
    A. Perfect monotonicity ⇒ ξₙ = (n−2)/(n+1)
    """
    n = 128
    x = torch.linspace(-2.0, 2.0, n)
    y = 3.0 * x + 5.0  # strictly increasing, no ties

    xi = xi_hard(x, y)
    expected = (n - 2) / (n + 1)

    assert torch.isclose(xi, torch.tensor(expected), atol=1e-8), (
        f"Expected ξ_hard={expected}, got {xi.item()}"
    )

def test_constant_y_error_for_hard_and_soft():
    """
    B. Constant-Y must raise ValueError in both hard and soft implementations.
    """
    n = 64
    x = torch.randn(n)
    y_const = torch.ones(n)

    # Hard ξₙ
    with pytest.raises(ValueError) as exc_h:
        _ = xi_hard(x, y_const)
    assert "constant" in str(exc_h.value).lower()

    # Soft ξₙ via XiLoss
    loss_fn = XiLoss(tau=0.1, lambda_=1.0)
    with pytest.raises(ValueError) as exc_s:
        _ = loss_fn(x.requires_grad_(), y_const)
    assert "constant" in str(exc_s.value).lower()

def test_small_n_edge_cases():
    """
    C. Small-n edge cases:
       - n=2 ⇒ ξₙ = 0
       - n<2 ⇒ ValueError
    """
    # n = 2
    x2 = torch.tensor([1.0, 2.0])
    y2 = torch.tensor([3.0, 4.0])
    xi2 = xi_hard(x2, y2)
    assert torch.isclose(xi2, torch.tensor(0.0), atol=1e-8)

    loss_fn = XiLoss(tau=0.1, lambda_=1.0)
    loss2, xi_soft2 = loss_fn(x2.requires_grad_(), y2)
    assert torch.isclose(xi_soft2, torch.tensor(0.0), atol=1e-5)

    # n < 2
    x1 = torch.tensor([1.0])
    y1 = torch.tensor([2.0])
    with pytest.raises(ValueError) as exc_h1:
        _ = xi_hard(x1, y1)
    assert "at least 2" in str(exc_h1.value).lower()

    with pytest.raises(ValueError) as exc_s1:
        _ = loss_fn(x1.requires_grad_(), y1)
    assert "at least 2" in str(exc_s1.value).lower()

def test_monotone_invariance_hard_and_soft():
    """
    D. Invariance under strictly increasing transforms (hard & soft).
    """
    n = 128
    x = torch.linspace(0.1, 3.14, n)
    y = 2.0 * x + 1.0

    # Apply transforms
    x_t = 5.0 * x - 3.0
    y_t = y.pow(3)

    # Hard ξₙ invariance
    xi_orig = xi_hard(x, y)
    xi_trans = xi_hard(x_t, y_t)
    assert torch.isclose(xi_orig, xi_trans, atol=1e-12), (
        f"Hard ξ changed: {xi_orig.item()} vs {xi_trans.item()}"
    )

    # Soft ξₙ invariance
    loss_fn = XiLoss(tau=0.1, lambda_=1.0)
    _, xi_soft_orig = loss_fn(x.requires_grad_(), y)
    _, xi_soft_trans = loss_fn(x_t.requires_grad_(), y_t)
    assert torch.isclose(xi_soft_orig, xi_soft_trans, atol=1e-5, rtol=1e-4), (
        f"Soft ξ changed: {xi_soft_orig.item()} vs {xi_soft_trans.item()}"
    )

def test_gradient_flow_soft_xi():
    """
    E. Ensure soft ξₙ contributes non-zero gradients.
    """
    n = 64
    x = torch.randn(n, requires_grad=True)
    y = 2.0 * x + 1.0

    loss_fn = XiLoss(tau=0.5, lambda_=1.0)
    loss, _ = loss_fn(x, y)
    loss.backward()

    grad_norm = x.grad.norm().item()
    assert grad_norm > 1e-4, f"Gradient norm too small: {grad_norm:.2e}"