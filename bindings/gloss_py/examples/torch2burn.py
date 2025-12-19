#!/usr/bin/env python3

import torch
from gloss import Torch2BurnTest
import random

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Initialize the torch2burn interface
torch2burn = Torch2BurnTest()

# Create input tensor that requires gradients for backprop
tensor_input = torch.randn((5, 3)).cuda()  # 5 points in 3D space
tensor_input.requires_grad_(True)

# Create some weights/parameters that will be optimized
weights = torch.randn((3, 3)).cuda()  # 3x3 transformation matrix
weights.requires_grad_(True)

# Setup optimizer
optimizer_parameters = [
    {"params": tensor_input},
    {"params": weights},
]
optimizer = torch.optim.AdamW(
    optimizer_parameters, lr=1e-2, weight_decay=0.0, amsgrad=False
)

# Target values we want to optimize towards
target_output = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [2.0, 1.0, 0.0],
        [1.5, 0.5, 1.0],
        [0.0, 3.0, 1.5],
    ]
).cuda()

print("Starting optimization...")
print(f"Initial tensor_input shape: {tensor_input.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Target shape: {target_output.shape}")

for iteration in range(1000):
    # Change target occasionally to make optimization more interesting
    if iteration % 200 == 0 and iteration > 0:
        print(f"  Input mean: {tensor_input.mean().item():.4f}")
        print(f"  Weights mean: {weights.mean().item():.4f}")
        target_output = torch.randn_like(target_output) * 2.0
        print(f"Iteration {iteration}: Changed target")

    # Pass tensors to Rust for processing
    # Rust should:
    # 1. Take the input points (tensor_input) and transformation matrix (weights)
    # 2. Apply matrix multiplication: output = tensor_input @ weights
    # 3. Add some non-linear operation like: output = output + sin(output) * 0.1
    # 4. Return the transformed points back to PyTorch

    out = torch2burn.forward(tensor_input, weights)
    # This simulates what the Rust forward function should do:
    # out = tensor_input @ weights + torch.sin(tensor_input @ weights) * 0.1

    # Compute loss - mean squared error between output and target
    loss = torch.nn.functional.mse_loss(out, target_output)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Loss = {loss.item():.6f}")

    # Stop if loss is very small
    if loss.item() < 1e-6:
        print(f"Converged at iteration {iteration} with loss {loss.item():.8f}")
        break

print("Optimization complete!")
