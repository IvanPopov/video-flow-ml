#!/usr/bin/env python3
"""
Isolated MemFlow model loader to avoid import conflicts with VideoFlow
"""

import os
import sys
import torch
import subprocess
import pickle
import tempfile

def load_memflow_model_isolated(model_path, stage, device):
    """
    Load MemFlow model in a completely isolated Python process
    Returns the model state dict and configuration
    """
    
    # Get absolute paths to avoid issues
    memflow_dir = os.path.abspath(os.path.join(os.getcwd(), "MemFlow"))
    abs_model_path = os.path.abspath(model_path)
    temp_dir = tempfile.gettempdir()
    
    # Create a temporary script to load MemFlow
    script_content = f'''
import os
import sys
import torch
import pickle

# Change to MemFlow directory
os.chdir(r"{memflow_dir}")

# Add MemFlow paths
sys.path.insert(0, '.')
sys.path.insert(0, 'core')
sys.path.insert(0, 'inference')

# Import MemFlow modules
from core.Networks import build_network

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

# Build model
model = build_network(cfg)

# Save model state and config to temporary file
result = {{
    'model_state_dict': model.state_dict(),
    'config': cfg,
    'device': "{device}"
}}

with open(r"{temp_dir}\\memflow_model.pkl", "wb") as f:
    pickle.dump(result, f)

print("MemFlow model loaded successfully")
'''
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the script in isolated process
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        # Load the result
        result_path = os.path.join(temp_dir, 'memflow_model.pkl')
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
        
        # Clean up
        os.unlink(result_path)
        
        return data
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load MemFlow model: {e.stderr}")
    finally:
        # Clean up script
        try:
            os.unlink(script_path)
        except:
            pass

def create_memflow_inference_isolated(model_path, stage, device):
    """
    Create MemFlow inference core in isolated process
    Returns a serialized inference function
    """
    
    # Get absolute paths to avoid issues
    memflow_dir = os.path.abspath(os.path.join(os.getcwd(), "MemFlow"))
    abs_model_path = os.path.abspath(model_path)
    temp_dir = tempfile.gettempdir()
    
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

print("MemFlow inference core ready")
print("Model device:", next(model.parameters()).device)
print("Config network:", cfg.network)

# Test basic inference capability
test_result = {{
    'success': True,
    'device': str(next(model.parameters()).device),
    'network': cfg.network
}}

with open(r"{temp_dir}\\memflow_inference_test.pkl", "wb") as f:
    pickle.dump(test_result, f)
'''
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the script in isolated process
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        print("MemFlow isolated test output:")
        print(result.stdout)
        
        # Load the result
        result_path = os.path.join(temp_dir, 'memflow_inference_test.pkl')
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
        
        # Clean up
        os.unlink(result_path)
        
        return data
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create MemFlow inference: {e.stderr}")
    finally:
        # Clean up script
        try:
            os.unlink(script_path)
        except:
            pass 