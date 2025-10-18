import pytest
import torch
import math
from torch.autograd import gradcheck
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

# new tests v2

def test_hard_and_soft_decreasing_monotonic():
    """
    Test that xi is invariant to monotonic reversal: xi(x, y_inc) == xi(x, y_dec)
    for both hard and soft implementations.
    """
    n = 128
    x = torch.linspace(0,1,n)
    y_inc = 5*x + 2
    y_dec = -1 * y_inc + 10
    xi_inc = xi_hard(x, y_inc)
    xi_dec = xi_hard(x, y_dec)
    assert torch.isclose(xi_inc, xi_dec, atol=1e-8) ##

    loss_fn = XiLoss(tau=0.1, lambda_=1.0)
    _, xi_soft_inc = loss_fn(x.requires_grad_(), y_inc)
    _, xi_soft_dec = loss_fn(x.requires_grad_(), y_dec)
    assert torch.isclose(xi_soft_inc, xi_soft_dec, atol=1e-2) ##

def test_shape_mismatch_raises():
    """
    Test that shape mismatch between y_pred and y_true raises ValueError.
    """
    loss_fn = XiLoss()
    y_pred = torch.randn(10,1)
    y_true = torch.randn(10)
    with pytest.raises(ValueError):
        _ = loss_fn(y_pred, y_true)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_gpu_forward_and_backward():
    """
    Test that XiLoss forward and backward work on CUDA tensors.
    """
    device = torch.device("cuda")
    loss_fn = XiLoss().to(device)
    x = torch.randn(64, device=device, requires_grad=True)
    y = 2*x + 1
    total, xi = loss_fn(x, y)
    total.backward()
    assert x.grad is not None and x.grad.norm() > 0

def test_soft_matches_hard_on_perfect():
    """
    Test that soft xi converges to hard xi on perfectly monotonic input.
    """
    n = 100
    x = torch.linspace(-1,1,n)
    y = x**3  # perfectly monotonic
    hard = xi_hard(x, y)
    _, soft = XiLoss(tau=0.01, lambda_=1.0)(x, y)
    assert torch.isclose(soft, hard, atol=1e-3), f"{soft} vs {hard}"

def test_xi_gradients_without_task_loss():
    """
    Test that xi regularization alone produces gradients (task_loss_fn returns 0).
    """
    n = 64
    y_pred = torch.randn(n, requires_grad=True)
    y_true = torch.sin(torch.linspace(-3, 3, n))  # any non-constant target

    loss_fn = XiLoss(tau=0.2, lambda_=1.0, task_loss_fn=lambda a, b: 0.0)
    total, xi_soft = loss_fn(y_pred, y_true)
    total.backward()
    assert y_pred.grad.norm() > 1e-4, "ξₙ contributes no gradient!"

def test_independence_null_xi_near_zero():
    """
    Test that xi is near zero for independent x and y (null case).
    """
    n = 4000
    x = torch.randn(n)
    y = torch.randn(n)  # independent
    xi = xi_hard(x, y).item()
    assert abs(xi) < 0.03, f"independence should give ~0, got {xi:.4f}"

def test_monotone_transform_invariance():
    """
    Test that monotonic transforms of x or y (or sign flip of both) leave xi unchanged.
    """
    n = 512
    x = torch.linspace(-2, 3, n)
    y = torch.sin(x) + 0.1*torch.randn(n)
    # strictly increasing transforms
    fx = torch.exp(x/5)         # increasing in x
    gy = torch.log(y - y.min() + 1.0)  # increasing in y's order
    base = xi_hard(x, y)
    x_inc = xi_hard(fx, y)
    y_inc = xi_hard(x, gy)
    # joint sign flip should preserve ξ
    both_flip = xi_hard(-x, -y)
    for got in [x_inc, y_inc, both_flip]:
        assert torch.isclose(got, base, atol=1e-3), f"{got} vs {base}"

def test_noise_monotonicity():
    """
    Test that adding increasing noise to a monotonic relationship reduces xi.
    """
    torch.manual_seed(0)
    n = 2000
    x = torch.rand(n)*4 - 2
    f = torch.tanh(x)  # monotone-ish
    sigmas = [0.01, 0.1, 0.5]
    xis = []
    for s in sigmas:
        y = f + s*torch.randn(n)
        xis.append(xi_hard(x, y).item())
    assert xis[0] > xis[1] > xis[2], f"ξ should drop with noise: {xis}"

def test_pair_order_permutation_invariance():
    """
    Test that xi is invariant to joint permutation of x and y.
    """
    n = 333
    x = torch.randn(n)
    y = x**3 + 0.1*torch.randn(n)
    base = xi_hard(x, y)
    perm = torch.randperm(n)
    shuf = xi_hard(x[perm], y[perm])
    assert torch.isclose(base, shuf, atol=1e-8)

def test_soft_bounds_and_finiteness_across_tau():
    """
    Test that soft xi stays finite and within [0,1] for a range of tau values,
    and that gradients remain finite.
    """
    n = 128
    yp = torch.randn(n, requires_grad=True)
    yt = torch.randn(n)
    for tau in [0.005, 0.01, 0.1, 0.5, 1.0, 2.0]:
        yp.grad = None
        loss, xi_soft = XiLoss(tau=tau, lambda_=1.0)(yp, yt)
        assert torch.isfinite(loss) and torch.isfinite(xi_soft)
        assert -1e-3 <= xi_soft.item() <= 1+1e-3  # numeric wiggle, should be [0,1]
        loss.backward()
        assert torch.isfinite(yp.grad).all()

def test_near_ties_do_not_explode_soft():
    """
    Test that soft xi remains finite (and gradients finite) when values are nearly tied.
    """
    n = 256
    base = torch.linspace(-1, 1, n)
    y = base + 1e-6*torch.randn(n)  # near ties
    yp = base + 1e-6*torch.randn(n)
    loss, xi_soft = XiLoss(tau=0.1, lambda_=1.0)(yp.requires_grad_(), y)
    loss.backward()
    assert torch.isfinite(xi_soft) and torch.isfinite(yp.grad).all()

def test_shape_equivalence_n_vs_n1():
    """
    Test that xi_hard gives the same result for shape (n,) vs (n,1) squeezed.
    """
    n = 200
    x = torch.randn(n)
    y = torch.randn(n)
    hard_a = xi_hard(x, y)
    hard_b = xi_hard(x.view(-1,1).squeeze(), y.view(-1,1).squeeze())
    assert torch.isclose(hard_a, hard_b, atol=1e-12)

def test_gradcheck_soft_xi_scalar_output():
    """
    Test that the soft xi loss passes gradcheck (is numerically differentiable).
    ONLY PASSES IF JITTER IS DISABLED!
    """
    torch.manual_seed(0)
    n = 16
    yp = (torch.randn(n, dtype=torch.float64, requires_grad=True))
    yt = torch.randn(n, dtype=torch.float64)  # constant-free
    loss_fn = XiLoss(tau=0.2, lambda_=1.0)
    def f(v):
        total, _ = loss_fn(v, yt)
        return total
    assert gradcheck(f, (yp,), eps=1e-6, atol=1e-4, rtol=1e-4)

def test_concatenation_invariance():
    """
    Test that xi_hard is invariant to concatenation order of data.
    """
    torch.manual_seed(0)
    n = 300
    x = torch.randn(n)
    y = torch.sin(x) + 0.1*torch.randn(n)
    whole = xi_hard(x, y)
    idx = torch.randperm(n)
    a, b = idx[:150], idx[150:]
    recon = xi_hard(torch.cat([x[a], x[b]]), torch.cat([y[a], y[b]]))
    assert torch.isclose(whole, recon, atol=1e-12)

def test_dtype_consistency():
    """
    Test that xi_hard returns the same value for float32 vs float64 (within tolerance).
    """
    torch.manual_seed(0)
    n = 512
    x32 = torch.randn(n, dtype=torch.float32)
    y32 = torch.tanh(x32) + 0.05*torch.randn(n, dtype=torch.float32)
    x64, y64 = x32.double(), y32.double()
    h32 = xi_hard(x32, y32).double()
    h64 = xi_hard(x64, y64)
    assert torch.isclose(h32, h64, atol=1e-6)

def test_numerical_stability_extremes():
    """
    Test that XiLoss and its gradients remain finite for extremely large input magnitudes.
    """
    torch.manual_seed(0)
    n = 256
    x = (1e6*torch.randn(n)).clamp(-1e9, 1e9)
    y = x + 1e-3*torch.randn(n)
    loss, xi = XiLoss(tau=0.1, lambda_=1.0)(x.requires_grad_(), y)
    loss.backward()
    assert torch.isfinite(xi) and torch.isfinite(x.grad).all()

def test_index_order_irrelevant():
    """
    Test that xi_hard is invariant to data index ordering.
    """
    n = 401
    x = torch.randn(n); y = torch.randn(n)
    base = xi_hard(x, y)
    perm = torch.randperm(n)
    assert torch.isclose(base, xi_hard(x[perm], y[perm]), atol=1e-12)

def test_shape_ducktyping_both_axes():
    """
    Test that xi_hard produces identical result with duck-typed shapes.
    """
    n = 200
    x = torch.randn(n); y = torch.randn(n)
    a = xi_hard(x, y)
    b = xi_hard(x.view(-1,1).squeeze(1), y.view(-1,1).squeeze(1))
    assert torch.isclose(a, b, atol=1e-12)

def test_independence_null_xi_near_zero():
    """
    Test that xi is near zero for independent x and y (null case).
    """
    n = 4000
    x = torch.randn(n)
    y = torch.randn(n)  # independent
    xi = xi_hard(x, y).item()
    assert abs(xi) < 0.03, f"independence should give ~0, got {xi:.4f}"

def test_monotone_transform_invariance():
    """
    Test that monotonic transforms of x or y (or sign flip of both) leave xi unchanged.
    """
    n = 512
    x = torch.linspace(-2, 3, n)
    y = torch.sin(x) + 0.1*torch.randn(n)
    # strictly increasing transforms
    fx = torch.exp(x/5)         # increasing in x
    gy = torch.log(y - y.min() + 1.0)  # increasing in y's order
    base = xi_hard(x, y)
    x_inc = xi_hard(fx, y)
    y_inc = xi_hard(x, gy)
    # joint sign flip should preserve ξ
    both_flip = xi_hard(-x, -y)
    for got in [x_inc, y_inc, both_flip]:
        assert torch.isclose(got, base, atol=1e-3), f"{got} vs {base}"

def test_noise_monotonicity():
    """
    Test that adding increasing noise to a monotonic relationship reduces xi.
    """
    torch.manual_seed(0)
    n = 2000
    x = torch.rand(n)*4 - 2
    f = torch.tanh(x)  # monotone-ish
    sigmas = [0.01, 0.1, 0.5]
    xis = []
    for s in sigmas:
        y = f + s*torch.randn(n)
        xis.append(xi_hard(x, y).item())
    assert xis[0] > xis[1] > xis[2], f"ξ should drop with noise: {xis}"

def test_pair_order_permutation_invariance():
    """
    Test that xi is invariant to joint permutation of x and y.
    """
    n = 333
    x = torch.randn(n)
    y = x**3 + 0.1*torch.randn(n)
    base = xi_hard(x, y)
    perm = torch.randperm(n)
    shuf = xi_hard(x[perm], y[perm])
    assert torch.isclose(base, shuf, atol=1e-8)

def test_soft_bounds_and_finiteness_across_tau():
    """
    Test that soft xi stays finite and within [0,1] for a range of tau values,
    and that gradients remain finite.
    """
    n = 128
    yp = torch.randn(n, requires_grad=True)
    yt = torch.randn(n)
    for tau in [0.005, 0.01, 0.1, 0.5, 1.0, 2.0]:
        yp.grad = None
        loss, xi_soft = XiLoss(tau=tau, lambda_=1.0)(yp, yt)
        assert torch.isfinite(loss) and torch.isfinite(xi_soft)
        assert -1e-3 <= xi_soft.item() <= 1+1e-3  # numeric wiggle, should be [0,1]
        loss.backward()
        assert torch.isfinite(yp.grad).all()

def test_near_ties_do_not_explode_soft():
    """
    Test that soft xi remains finite (and gradients finite) when values are nearly tied.
    """
    n = 256
    base = torch.linspace(-1, 1, n)
    y = base + 1e-6*torch.randn(n)  # near ties
    yp = base + 1e-6*torch.randn(n)
    loss, xi_soft = XiLoss(tau=0.1, lambda_=1.0)(yp.requires_grad_(), y)
    loss.backward()
    assert torch.isfinite(xi_soft) and torch.isfinite(yp.grad).all()