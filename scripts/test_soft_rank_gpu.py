import torch
import torchsort

# 1) Use GPU on cloud
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 2) Create a 1×8 tensor with requires_grad
x = torch.randn(1, 8, requires_grad=True, device=device)

# 3) Compute soft ranks across the 8 points
r = torchsort.soft_rank(
    x,
    regularization="l2",
    regularization_strength=1.0
)

# 4) Define a non-constant loss: sum of elementwise product of ranks and values
loss = (r * x).sum()

# 5) Backpropagate
loss.backward()

# 6) Print shapes and gradient norm to verify
print(f"x.shape = {x.shape}, r.shape = {r.shape}")
print(f"x.grad norm = {x.grad.norm().item():.4f}")