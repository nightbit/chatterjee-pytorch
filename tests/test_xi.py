import torch, pytest
from losses.xi_loss import XiLoss, xi_hard

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_monotone_high_xi(device):
    x = torch.linspace(-2, 2, 128, device=device)
    y = x ** 2                  # deterministic monotone (on x≥0)
    loss_fn = XiLoss(tau=1.0).to(device)
    _, xi_soft = loss_fn(x, y)
    assert xi_soft >= 0.95

def test_independence_low_xi():
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(128, generator=rng)
    y = torch.randn(128, generator=rng)
    loss_fn = XiLoss()
    _, xi_soft = loss_fn(x, y)
    assert xi_soft <= 0.10

def test_gradient_nonzero():
    x = torch.randn(64, requires_grad=True)
    y = x ** 2
    loss_fn = XiLoss()
    loss, _ = loss_fn(x, y)
    loss.backward()
    grad_norm = x.grad.norm().item()
    assert grad_norm > 0.0