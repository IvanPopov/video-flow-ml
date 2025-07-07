"""
VideoFlow Core - Low-level optical flow computation module

This module provides the core VideoFlow model operations:
- Model loading and configuration
- Direct optical flow computation from prepared tensor batches
- Minimal dependencies and clean interface

This is a low-level module that expects pre-processed tensor inputs
and returns raw optical flow tensors. It does NOT handle:
- Frame sequence preparation
- Tile-based processing
- Progress tracking
- Input validation beyond basic tensor checks

For high-level operations, use VideoFlowProcessor instead.
"""

import os
import sys
import torch

# Add VideoFlow core to path
sys.path.insert(0, os.path.join(os.getcwd(), 'VideoFlow'))
sys.path.insert(0, os.path.join(os.getcwd(), 'VideoFlow', 'core'))

# VideoFlow imports
from core.Networks import build_network
from utils.utils import InputPadder
from configs.multiframes_sintel_submission import get_cfg


class VideoFlowCore:
    """
    Core VideoFlow inference engine for optical flow computation
    
    This class provides only the essential VideoFlow operations:
    - Model loading with optional fast mode optimizations
    - Direct tensor-to-tensor optical flow computation
    - Minimal overhead and dependencies
    
    Input requirements:
    - Pre-processed tensor batches in VideoFlow format: [B, T, C, H, W]
    - Normalized to [0,1] float32 values
    - RGB channel order
    
    Output:
    - Raw optical flow tensor in format [C, H, W] (2 channels: x, y flow)
    - No padding, no format conversion
    """
    
    def __init__(self, device, fast_mode=False):
        """
        Initialize VideoFlow core engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
        """
        self.device = device
        self.fast_mode = fast_mode
        
        # Model components
        self.model = None
        self.cfg = None
        
    def load_model(self):
        """Load VideoFlow MOF model with optional fast mode optimizations"""
        # Get VideoFlow configuration
        self.cfg = get_cfg()
        
        # Apply fast mode optimizations
        if self.fast_mode:
            self.cfg.decoder_depth = 6  # Reduce from default 12
            self.cfg.corr_levels = 3    # Reduce correlation levels
            self.cfg.corr_radius = 3    # Reduce correlation radius
        
        # Check if model weights exist
        model_path = self.cfg.model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VideoFlow model weights not found: {model_path}")
        
        # Build network
        self.model = build_network(self.cfg)
        
        # Load pre-trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Remove 'module.' prefix from keys if present (for models trained with DataParallel)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
        
        self.model.load_state_dict(checkpoint)
        
        # Move to device and set evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        return model_path
    
    def compute_flow_from_tensor(self, frame_batch_tensor):
        """
        Compute optical flow from pre-processed tensor batch
        
        Args:
            frame_batch_tensor: Pre-processed tensor [B, T, C, H, W]
                              - B=1 (batch size)
                              - T=sequence_length (typically 5)
                              - C=3 (RGB channels)
                              - H, W: frame dimensions
                              - Values normalized to [0,1]
                              - Device: must match self.device
        
        Returns:
            flow_tensor: Raw optical flow tensor [2, H, W]
                        - 2 channels: [flow_x, flow_y]
                        - H, W: same as input frame dimensions
                        - No padding, no format conversion
        
        Raises:
            RuntimeError: If model not loaded
            ValueError: If tensor format is incorrect
        """
        if self.model is None:
            raise RuntimeError("VideoFlow model not loaded. Call load_model() first.")
        
        # Validate input tensor
        if not isinstance(frame_batch_tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        # Check device compatibility (handle 'cuda' vs 'cuda:0' equivalence)
        tensor_device_str = str(frame_batch_tensor.device)
        model_device_str = str(self.device)
        
        # Normalize device strings for comparison
        if tensor_device_str.startswith('cuda') and model_device_str == 'cuda':
            pass  # Compatible
        elif model_device_str.startswith('cuda') and tensor_device_str == 'cuda':
            pass  # Compatible
        elif tensor_device_str != model_device_str:
            raise ValueError(f"Input tensor device ({frame_batch_tensor.device}) doesn't match model device ({self.device})")
        
        if len(frame_batch_tensor.shape) != 5:
            raise ValueError(f"Input tensor must have 5 dimensions [B,T,C,H,W], got {len(frame_batch_tensor.shape)}")
        
        if frame_batch_tensor.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {frame_batch_tensor.shape[0]}")
        
        if frame_batch_tensor.shape[2] != 3:
            raise ValueError(f"Must have 3 color channels, got {frame_batch_tensor.shape[2]}")
        
        # Create input padder for the spatial dimensions
        padder = InputPadder(frame_batch_tensor.shape[-2:])
        frame_batch_padded = padder.pad(frame_batch_tensor)
        
        # Run VideoFlow inference
        with torch.no_grad():
            # VideoFlow forward pass
            flow_predictions, _ = self.model(frame_batch_padded, {})
            
            # Unpad results
            flow_tensor = padder.unpad(flow_predictions)
            
            # Get the middle flow (center frame of the sequence)
            middle_idx = flow_tensor.shape[1] // 2
            flow_tensor = flow_tensor[0, middle_idx]  # Remove batch dim, get middle flow
            
            # Return raw tensor [2, H, W]
            return flow_tensor
    
    def is_model_loaded(self):
        """Check if VideoFlow model is loaded"""
        return self.model is not None
    
    def get_model_info(self):
        """Get information about loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": getattr(self.cfg, 'model', 'unknown') if self.cfg else 'unknown',
            "config": {
                "decoder_depth": getattr(self.cfg, 'decoder_depth', 'default'),
                "corr_levels": getattr(self.cfg, 'corr_levels', 'default'),
                "corr_radius": getattr(self.cfg, 'corr_radius', 'default'),
            },
            "fast_mode": self.fast_mode,
            "device": str(self.device)
        }
    
    def get_device(self):
        """Get the device used by the model"""
        return self.device
    
    def set_eval_mode(self):
        """Ensure model is in evaluation mode"""
        if self.model is not None:
            self.model.eval()
    
    def get_memory_usage(self):
        """Get current GPU memory usage if using CUDA"""
        if self.device.type == 'cuda':
            return {
                "allocated": torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
                "cached": torch.cuda.memory_reserved(self.device) / 1024**2,  # MB
                "max_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
            }
        else:
            return {"message": "Memory tracking only available for CUDA devices"} 