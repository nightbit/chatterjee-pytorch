#!/usr/bin/env python
import torch, time
import torchsort
import os
import sys

# ─── Make sure the project root (one level up) is on Python’s module path ───
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from losses.xi_loss import _soft_perm_gaussian, _soft_perm_neuralsort


def soft_xi(preds, target, perm_fn, tau=1.0):
    n = preds.size(0)
    # build P̂
    P = perm_fn(preds, tau)                   # (n,n)
    # soft ranks of targets (2‑D in -> 1‑D out)
    r_y = torchsort.soft_rank(
        target.unsqueeze(0), regularization_strength=tau
    ).squeeze(0)                             # (n,)
    r_sorted = (P @ r_y.unsqueeze(-1)).squeeze(-1)
    diff_sum  = torch.abs(r_sorted[1:] - r_sorted[:-1]).sum()
    xi_soft = 1.0 - 3.0 * diff_sum / (n * n - 1)
    return xi_soft.item()

def run_case(batch=128, tau=1.0):
    rng = torch.Generator().manual_seed(42)
    # M1 data
    x_mono = torch.linspace(-2, 2, batch)
    y_mono = x_mono ** 2
    # I1 / I2 data
    x_rand  = torch.randn(batch, generator=rng)
    y_rand  = torch.randn(batch, generator=rng)
    for name, fn in [("Gaussian", _soft_perm_gaussian),
                     ("NeuralSort", _soft_perm_neuralsort)]:
        xi_m  = soft_xi(x_mono, y_mono, fn, tau)
        xi_i  = soft_xi(x_rand, y_rand, fn, tau)
        print(f"{name:<10}  tau={tau:<5}  monotone={xi_m:.3f}  independent={xi_i:.3f}")

if __name__ == "__main__":
    t0 = time.time()
    print("=== τ = 1.0 ===")
    run_case(tau=1.0)
    print("\n=== τ = 0.05 ===")
    run_case(tau=0.05)
    print(f"\nDone in {time.time()-t0:.2f}s")
