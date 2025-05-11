#!/usr/bin/env python3
import os
import torch
import sys

# ─── Make sure the project root (one level up) is on Python’s module path ───
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# adjust import path if needed
from losses.xi_loss import _soft_perm_gaussian, _soft_perm_neuralsort

def test_perm(scores: torch.Tensor, tau: float, func, name: str):
    """
    Compute P_hat = func(scores, tau),
    check row sums ≈ 1, then compute how many rows
    argmax(P_hat[i]) == sorted_index[i].
    """
    P = func(scores, tau)  # (n,n)
    # 1) Row-stochasticity
    row_sums = P.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        print(f"[{name}] τ={tau}: row sums != 1: {row_sums.cpu().numpy()}", file=sys.stderr)

    # 2) True ascending‐sort indices
    sorted_idx = torch.argsort(scores, stable=True)
    # 3) Predicted positions via argmax
    pred_idx = P.argmax(dim=1)
    acc = (pred_idx == sorted_idx).float().mean().item()
    print(f"[{name}] τ={tau:<5}  accuracy: {acc*100:5.1f}%")

def main():
    torch.manual_seed(0)
    tests = {
        "ascending": torch.arange(10, dtype=torch.float32),
        "descending": torch.arange(10, 0, -1, dtype=torch.float32),
        "random": torch.randn(10),
    }
    taus = [1.0, 0.1, 0.01]
    funcs = [
        (_soft_perm_gaussian,  "Gaussian"),
        (_soft_perm_neuralsort,"NeuralSort"),
    ]

    for test_name, scores in tests.items():
        print(f"\n=== Test case: {test_name} ===")
        for tau in taus:
            for func, name in funcs:
                test_perm(scores, tau, func, name)

if __name__ == "__main__":
    main()