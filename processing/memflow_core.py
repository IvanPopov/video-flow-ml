#!/usr/bin/env python3
"""
MemFlow Core Module

Low-level MemFlow optical flow computation module. 
Provides minimal interface for pure tensor-to-tensor flow computation.
Based on original MemFlow implementation by Qiaole Dong, Yanwei Fu (CVPR 2024).
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add MemFlow paths to sys.path at the beginning to prioritize MemFlow imports
memflow_root = os.path.join(os.getcwd(), 'MemFlow')
memflow_core = os.path.join(memflow_root, 'core')
memflow_inference = os.path.join(memflow_root, 'inference')

# Insert at the beginning to prioritize MemFlow imports over VideoFlow
sys.path.insert(0, memflow_inference)
sys.path.insert(0, memflow_core)
sys.path.insert(0, memflow_root)


class MemFlowCore:
    """
    Low-level MemFlow core engine for direct tensor-to-tensor optical flow computation.
    
    This class provides the minimal interface for MemFlow processing:
    - Loads MemFlow model with specified configuration
    - Accepts preprocessed tensors in format [B, T, C, H, W]
    - Returns optical flow tensors in format [2, H, W]
    - Optimized for memory usage and performance
    """
    
    def __init__(self, device='cuda', model_path='MemFlow_ckpt/MemFlowNet_sintel.pth', stage='sintel'):
        """
        Initialize MemFlow core engine.
        
        Args:
            device: Processing device ('cuda' or 'cpu')
            model_path: Path to MemFlow model weights
            stage: Training stage configuration ('sintel', 'things', 'kitti')
        """
        self.device = self._validate_device(device)
        self.model_path = model_path
        self.stage = stage
        self.model = None
        self.cfg = None
        self.input_padder = None
        
        print(f"[MemFlow] Core engine initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model: {model_path}")
        print(f"  Stage: {stage}")
    
    def _validate_device(self, device):
        """Validate and normalize device specification"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Normalize device string
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                # Handle 'cuda' vs 'cuda:0' equivalence
                if device == 'cuda':
                    device = 'cuda:0'
                return device
            else:
                print("Warning: CUDA requested but not available, falling back to CPU")
                return 'cpu'
        
        return device
    
    def load_model(self):
        """Load MemFlow model with specified configuration"""
        if self.model is not None:
            print("MemFlow model already loaded")
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MemFlow model not found: {self.model_path}")
        
        print(f"[Model] Loading MemFlow model from: {self.model_path}")
        
        # Use isolated loading to avoid import conflicts
        from .memflow_loader import load_memflow_model_isolated
        
        try:
            # Load model in isolated process
            model_data = load_memflow_model_isolated(self.model_path, self.stage, self.device)
            
            # Extract data
            self.cfg = model_data['config']
            
            # Create simple proxy that will delegate inference to isolated process
            import torch.nn as nn
            
            class MemFlowModelProxy(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    # Add a dummy parameter to make it a valid PyTorch module
                    self.dummy = nn.Parameter(torch.tensor(0.0))
                    
                def forward(self, x):
                    # This will be implemented with proper inference calls
                    raise NotImplementedError("Use compute_flow_from_tensor instead")
            
            self.model = MemFlowModelProxy(self.cfg).to(self.device)
            
            print(f"[Model] MemFlow model loaded successfully:")
            print(f"  Path: {self.model_path}")
            print(f"  Stage: {self.stage}")
            print(f"  Device: {self.device}")
            print(f"  Inference: Isolated process")
            print(f"  Memory Management: {self.cfg.get('memory_optimization', 'standard') if self.cfg else 'standard'}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MemFlow model: {e}")
    
    def validate_input_tensor(self, tensor):
        """Validate input tensor format and device compatibility"""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if len(tensor.shape) != 5:
            raise ValueError(f"Input tensor must be 5D [B, T, C, H, W], got shape: {tensor.shape}")
        
        B, T, C, H, W = tensor.shape
        
        if B != 1:
            raise ValueError(f"Batch size must be 1, got: {B}")
        
        if T < 2:
            raise ValueError(f"Temporal dimension must be at least 2, got: {T}")
        
        if C != 3:
            raise ValueError(f"Channel dimension must be 3 (RGB), got: {C}")
        
        if H < 64 or W < 64:
            raise ValueError(f"Spatial dimensions must be at least 64x64, got: {H}x{W}")
        
        # Check device compatibility
        tensor_device = str(tensor.device)
        target_device = self.device
        
        # Handle device equivalence (cuda vs cuda:0)
        if tensor_device.startswith('cuda') and target_device.startswith('cuda'):
            if tensor_device == 'cuda' and target_device == 'cuda:0':
                pass  # Compatible
            elif tensor_device == 'cuda:0' and target_device == 'cuda':
                pass  # Compatible
            elif tensor_device != target_device:
                print(f"Warning: Tensor device ({tensor_device}) differs from model device ({target_device})")
        elif tensor_device != target_device:
            print(f"Warning: Tensor device ({tensor_device}) differs from model device ({target_device})")
    
    def compute_flow_from_tensor(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow from preprocessed tensor using isolated MemFlow process.
        
        Args:
            frames_tensor: Input frames tensor [B, T, C, H, W] with values in range [0, 255] or [-1, 1]
            
        Returns:
            Optical flow tensor [2, H, W]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.validate_input_tensor(frames_tensor)
        
        # Use isolated inference to avoid import conflicts
        from .memflow_inference_isolated import compute_memflow_isolated
        
        # Compute flow in isolated process
        flow_result = compute_memflow_isolated(
            frames_tensor, 
            self.model_path, 
            self.stage, 
            self.device
        )
        
        return flow_result
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if self.device.startswith('cuda'):
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'device': self.device
            }
        else:
            return {'device': 'cpu', 'note': 'CPU memory tracking not available'}
    
    def cleanup(self):
        """Clean up resources"""
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        self.input_padder = None 