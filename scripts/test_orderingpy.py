"""
xi_orientation_test.py
--------------------------------
Detects whether the current implementation follows the correct
(X = y_pred, Y = y_true) ordering for Chatterjee’s ξₙ inside a
learning-time loss.

HOW IT WORKS
============
1.  Build a *coarse* prediction vector that is a many-to-one
    function of the true labels:

        y_true  = [-1.0, -0.996, …, 0.996, 1.0]      (strictly increasing)
        y_pred  = sign(y_true)  →  {-1, +1} only      (two-level step)

    •  In theory,  ξₙ(X, Y) == 1  **iff**  Y is a measurable function of X.
       Here, *predictions* are a function of *truth* (two levels),
       but the converse is false.

    •  Therefore:
          ξₙ( y_true, y_pred )  →  1      (because y_pred = f(y_true))
          ξₙ( y_pred, y_true )  →  << 1   (because y_true ≠ g(y_pred))

2.  Let **ξ_keep**  be the value returned by *your* current ordering
    (whatever `xi_hard(x, y)` uses internally).

    Let **ξ_swap** be the value when the two arguments are reversed.

3.  If  ξ_keep  is the *small* one (<< 1) and  ξ_swap  is ≈1,
    your code already uses the correct `(y_pred, y_true)` order → **keep**.
    Otherwise → **swap**.

Run with:
    python xi_orientation_test.py
--------------------------------
"""
import torch
import sys
import os

# ─── Make sure the project root (one level up) is on Python’s module path ───
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# import your own implementation exactly as used in training
from losses.xi_loss import xi_hard

def directionality_test(n=1000, seed=0):
    torch.manual_seed(seed)
    # 1) X uniform in [-1,1]
    x = torch.rand(n) * 2 - 1
    # 2) True relationship: Y = X^2 (non-injective ⇒
    #    Xi_n(X, Y)=1, Xi_n(Y, X)<1)
    y_true = x.pow(2)
    # 3) “Prediction” is just identity X (i.e. our y_pred)
    y_pred = x.clone()

    # Compute both directional Xi
    xi_pred_true = xi_hard(y_pred, y_true)   # sort by y_pred, rank y_true
    xi_true_pred = xi_hard(y_true, y_pred)   # sort by y_true, rank y_pred

    margin = (xi_pred_true - xi_true_pred).item()
    decision = "keep" if margin >= 0 else "swap"

    print(f"Xi(pred→true) = {xi_pred_true:.6f}")
    print(f"Xi(true→pred) = {xi_true_pred:.6f}")
    print(f"Margin        = {margin:.6f}")
    print(f"Decision      = {decision}")

if __name__=="__main__":
    directionality_test()