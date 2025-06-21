import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Simple CUDA test
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("CUDA test passed: matrix multiplication successful")
else:
    print("CUDA not available") 