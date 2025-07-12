#!/usr/bin/env python3
"""
Isolated MemFlow inference to avoid import conflicts with VideoFlow
"""

import os
import sys
import torch
import subprocess
import pickle
import tempfile
import numpy as np

def compute_memflow_isolated(frames_tensor, model_path, stage, device):
    """
    Compute MemFlow optical flow in isolated process
    
    Args:
        frames_tensor: Input frames tensor [B, T, C, H, W]
        model_path: Path to MemFlow model weights
        stage: Training stage (sintel, things, kitti)
        device: Target device (cpu, cuda)
        
    Returns:
        Optical flow tensor [2, H, W]
    """
    
    # Get absolute paths
    memflow_dir = os.path.abspath(os.path.join(os.getcwd(), "MemFlow"))
    abs_model_path = os.path.abspath(model_path)
    temp_dir = tempfile.gettempdir()
    
    # Save input tensor to temporary file (move to CPU first to avoid device conflicts)
    input_file = os.path.join(temp_dir, 'memflow_input.pth')
    print(f"[MemFlow] Saving input tensor: {frames_tensor.shape} from device {frames_tensor.device}")
    torch.save(frames_tensor.cpu(), input_file)
    
    # Create inference script
    script_content = f'''
import os
import sys
import torch
import pickle
import numpy as np

# Change to MemFlow directory
os.chdir(r"{memflow_dir}")

# Add MemFlow paths
sys.path.insert(0, '.')
sys.path.insert(0, 'core')
sys.path.insert(0, 'inference')

# Import MemFlow modules
from core.Networks import build_network
from inference_core_skflow import InferenceCore
from utils.utils import InputPadder, forward_interpolate

# Load configuration
if "{stage}" == "sintel":
    from configs.sintel_memflownet import get_cfg
elif "{stage}" == "things":
    from configs.things_memflownet import get_cfg  
elif "{stage}" == "kitti":
    from configs.kitti_memflownet import get_cfg
else:
    raise ValueError(f"Unsupported stage: {stage}")

cfg = get_cfg()
cfg.restore_ckpt = r"{abs_model_path}"

# Build and load model
model = build_network(cfg).to("{device}")
print(f"Model loaded on device: {device}")

# Load input tensor (ensure it's loaded on CPU first, then move to target device)
frames_tensor = torch.load(r"{input_file}", map_location='cpu')
frames_tensor = frames_tensor.to("{device}")
print(f"Input tensor moved to device: {{frames_tensor.device}}")

print(f"Input tensor shape: {{frames_tensor.shape}}")

# Normalize if needed
max_val = frames_tensor.max().item()
if max_val > 2.0:  # Assume [0, 255] range
    frames_tensor = 2 * (frames_tensor / 255.0) - 1.0
elif max_val > 1.0:  # Assume [0, 1] range
    frames_tensor = 2 * frames_tensor - 1.0

# Create padder
input_padder = InputPadder(frames_tensor.shape)
frames_tensor = input_padder.pad(frames_tensor)

# Create inference processor
processor = InferenceCore(model, config=cfg)

with torch.no_grad():
    # Use last two frames
    B, T, C, H, W = frames_tensor.shape
    frame_pair = frames_tensor[:, -2:]  # Take last 2 frames
    
    print(f"Processing frame pair shape: {{frame_pair.shape}}")
    
    # Compute flow
    flow_low, flow_pred = processor.step(
        frame_pair, 
        end=True,
        add_pe=('rope' in cfg and cfg.rope),
        flow_init=None
    )
    
    # Unpad result
    flow_result = input_padder.unpad(flow_pred[0]).cpu()
    
    print(f"Flow output shape: {{flow_result.shape}}")

# Save result
torch.save(flow_result, r"{temp_dir}\\memflow_output.pth")
print("MemFlow inference completed successfully")
'''
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the script in isolated process
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        print("MemFlow inference output:")
        print(result.stdout)
        
        # Load the result
        output_file = os.path.join(temp_dir, 'memflow_output.pth')
        flow_result = torch.load(output_file, map_location='cpu')
        print(f"[MemFlow] Loaded result tensor: {flow_result.shape} on device {flow_result.device}")
        
        # Move result to target device if needed
        if device != 'cpu':
            flow_result = flow_result.to(device)
            print(f"[MemFlow] Moved result to device: {flow_result.device}")
        
        # Clean up
        os.unlink(output_file)
        os.unlink(input_file)
        
        return flow_result
        
    except subprocess.CalledProcessError as e:
        print(f"MemFlow inference failed: {e.stderr}")
        raise RuntimeError(f"Failed to compute MemFlow flow: {e.stderr}")
    finally:
        # Clean up script
        try:
            os.unlink(script_path)
        except:
            pass
        try:
            os.unlink(input_file)
        except:
            pass 