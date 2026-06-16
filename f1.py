import torch

# 1. Automatically detect the best available hardware accelerator
if torch.cuda.is_available():
    device = torch.device("cuda")      # For NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = torch.device("mps")       # For Apple Silicon Macs
else:
    device = torch.device("cpu")       # Fallback to CPU

print(f"Using device: {device}")

# 2. Create your model or data tensor
my_tensor = torch.rand(3, 3)

# 3. Move the object over to your GPU accelerator
my_tensor = my_tensor.to(device)

# 4. Verify the item successfully moved
print("Tensor location:", my_tensor.device)
