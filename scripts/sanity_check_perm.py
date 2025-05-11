# sanity_check_perm.py
import torch

# pick a small vector of “scores”
scores = torch.randn(5)

for tau in [1.0, 0.1, 0.01]:
    # this matches your old implementation:
    diff = scores.unsqueeze(0) - scores.unsqueeze(1)    # (5,5)
    P = torch.softmax(diff / tau, dim=1)                # row-wise softmax

    # compare row 0 vs row 1
    same = torch.allclose(P[0], P[1], atol=1e-6)
    print(f"tau={tau:4.2f}  rows identical? {same}")