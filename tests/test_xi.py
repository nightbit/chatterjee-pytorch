#tests/test_xi.py`

import time
import pytest
import torch

from losses.xi_loss import XiLoss
from utils.hard_xi import xi_hard

# Devices to test on
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

def monotonic_data(n=128, device="cpu"):
    x = torch.linspace(-2, 2, n, device=device)
    return x, x

@pytest.mark.parametrize("device", DEVICES)
def test_hard_xi_monotonic(device):
    x, y = monotonic_data(device=device)
    xi = xi_hard(x, y)
    n = x.numel()
    xi_max = (n - 2) / (n + 1)
    assert torch.isclose(xi, torch.tensor(xi_max, device=device), atol=1e-6)

@pytest.mark.parametrize("device", DEVICES)
def test_soft_xi_monotonic(device):
    x, y = monotonic_data(device=device)
    n = x.numel()
    xi_max = (n - 2) / (n + 1)
    loss_fn = XiLoss(tau=0.1, lambda_=1.0).to(device)
    loss, xi_soft = loss_fn(x, y)
    # xi_soft should be very close to xi_max
    assert xi_soft >= xi_max - 1e-2
    assert xi_soft <= 1.0

@pytest.mark.parametrize("device", DEVICES)
def test_gradient_flows(device):
    x = torch.randn(64, device=device, requires_grad=True)
    y = 2 * x + 1
    loss_fn = XiLoss(tau=0.5, lambda_=1.0).to(device)
    loss, xi_soft = loss_fn(x, y)
    loss.backward()
    # Ensure gradient is substantial
    grad_norm = x.grad.norm().item()
    assert grad_norm > 1e-4

@pytest.mark.parametrize("device", DEVICES)
def test_soft_xi_independence(device):
    rng1 = torch.Generator().manual_seed(0)
    rng2 = torch.Generator().manual_seed(1)
    x = torch.randn(128, generator=rng1, device=device)
    y = torch.randn(128, generator=rng2, device=device)
    loss_fn = XiLoss(tau=1.0, lambda_=1.0).to(device)
    loss, xi_soft = loss_fn(x, y)
    # Independent data should yield xi_soft near zero
    assert abs(xi_soft) < 0.1

@pytest.mark.parametrize("device", DEVICES)
def test_soft_xi_invariance(device):
    x, y = monotonic_data(device=device)
    x2 = x.exp()
    y2 = y.log()
    loss_fn = XiLoss(tau=0.1, lambda_=1.0).to(device)
    _, xi1 = loss_fn(x, y)
    _, xi2 = loss_fn(x2, y2)
    assert torch.isclose(xi1, xi2, atol=1e-5)

@pytest.mark.parametrize("device", DEVICES)
def test_directional_asymmetry(device):
    x = torch.linspace(-2, 2, 128, device=device)
    y = x ** 2
    # Hard xi
    xi_xy_h = xi_hard(x, y)
    xi_yx_h = xi_hard(y, x)
    assert xi_xy_h > 0.9
    assert xi_yx_h < 0.2
    # Soft xi
    loss_fn = XiLoss(tau=0.1, lambda_=1.0).to(device)
    _, xi_xy_s = loss_fn(x, y)
    _, xi_yx_s = loss_fn(y, x)
    assert xi_xy_s > 0.9
    assert xi_yx_s < 0.2

@pytest.mark.parametrize("device", DEVICES)
def test_tie_handling(device):
    # Create ties in X
    x = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
    y = torch.tensor([1.0, 2.0, 3.0, 1.5, 2.5, 3.5], device=device)
    # Hard xi should not error and lie in [-0.5, 1]
    xi_h = xi_hard(x, y)
    assert -0.5 <= xi_h <= 1.0
    # Soft xi likewise
    loss_fn = XiLoss(tau=0.1, lambda_=1.0).to(device)
    _, xi_s = loss_fn(x, y)
    assert -0.5 <= xi_s <= 1.0

@pytest.mark.parametrize("device", DEVICES)
def test_soft_vs_hard_difference(device):
    # Compare soft and hard xi on random monotonic data
    x, y = monotonic_data(device=device)
    xi_h = xi_hard(x, y)
    loss_fn = XiLoss(tau=0.1, lambda_=1.0).to(device)
    _, xi_s = loss_fn(x, y)
    assert abs(xi_s - xi_h) < 0.02

@pytest.mark.parametrize("device", DEVICES)
def test_gpu_cpu_parity(device):
    if device != "cuda":
        pytest.skip("GPU/CPU parity check requires CUDA")
    # Generate random data
    rng1 = torch.Generator().manual_seed(42)
    rng2 = torch.Generator().manual_seed(43)
    x_cpu = torch.randn(64, generator=rng1, device="cpu")
    y_cpu = torch.randn(64, generator=rng2, device="cpu")
    # CPU
    loss_fn_cpu = XiLoss(tau=0.5, lambda_=1.0).to("cpu")
    _, xi_cpu = loss_fn_cpu(x_cpu, y_cpu)
    # Move to GPU
    x_gpu = x_cpu.to("cuda")
    y_gpu = y_cpu.to("cuda")
    loss_fn_gpu = XiLoss(tau=0.5, lambda_=1.0).to("cuda")
    _, xi_gpu = loss_fn_gpu(x_gpu, y_gpu)
    # Parity
    assert torch.isclose(xi_cpu, xi_gpu.cpu(), atol=1e-6)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.slow
def test_runtime_bound(device):
    if device != "cuda":
        pytest.skip("Runtime bound only tested on GPU")
    # Create random data
    x = torch.randn(256, device=device, requires_grad=True)
    y = torch.randn(256, device=device)
    loss_fn = XiLoss(tau=0.1, lambda_=1.0).to(device)
    # Measure forward + backward time
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, xi = loss_fn(x, y)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # Should be under 25 ms
    assert elapsed < 0.025, f"Elapsed time {elapsed:.3f}s exceeds 0.025s"