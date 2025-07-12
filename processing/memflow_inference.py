"""
MemFlow Optical Flow Inference Module - Compatibility Layer

This module maintains backward compatibility with existing code while using
the new modular architecture:
- MemFlowCore: Low-level model operations
- MemFlowProcessor: High-level processing pipeline

The MemFlowInference class acts as a compatibility wrapper that delegates
to the appropriate new modules while preserving the original API.

WARNING: This module requires CUDA/GPU support for optimal performance.
The model loading and inference operations cannot be easily parallelized
across multiple processes due to CUDA context limitations.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Import the new modular components
from .memflow_processor import MemFlowProcessor
from .base_flow_processor import BaseFlowInference


class MemFlowInference(BaseFlowInference):
    """
    MemFlow inference engine for optical flow computation (Compatibility Layer)
    
    This class provides backward compatibility with existing code while using
    the new modular architecture internally. It delegates all operations to
    MemFlowProcessor, which in turn uses MemFlowCore for model operations.
    
    For new code, consider using MemFlowProcessor or MemFlowCore directly.
    """
    
    def __init__(self, device: str = 'cuda', fast_mode: bool = False, tile_mode: bool = False, 
                 sequence_length: int = 3, stage: str = 'sintel', model_path: str = None):
        """
        Initialize MemFlow inference engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode (currently not implemented for MemFlow)
            tile_mode: Enable tile-based processing (currently not implemented for MemFlow)
            sequence_length: Number of frames to use in sequence for inference
            stage: Training stage/dataset ('sintel', 'things', 'kitti')
            model_path: Custom path to model weights
        """
        super().__init__(device, fast_mode, tile_mode, sequence_length, 
                         stage=stage, model_path=model_path)
        
        # Delegate to MemFlowProcessor
        self._processor = MemFlowProcessor(device, fast_mode, tile_mode, sequence_length, 
                                          stage, model_path)
        
        # Store parameters for compatibility
        self.stage = stage
        self.model_path = model_path
        
        # Show tile mode warning if enabled
        if tile_mode:
            print("Warning: Tile mode is not implemented for MemFlow. Using full-frame processing.")
    
    def load_model(self):
        """Load MemFlow model"""
        self._processor.load_model()
        
        # Set legacy attributes for backward compatibility
        self.model = self._processor.core.model
        self.cfg = self._processor.core.cfg
    
    def is_model_loaded(self) -> bool:
        """Check if MemFlow model is loaded"""
        return self._processor.is_model_loaded()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        info = self._processor.get_model_info()
        
        # Add compatibility layer information
        if info["status"] == "loaded":
            info["compatibility_layer"] = "MemFlowInference"
        
        return info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        return self._processor.get_memory_usage()
    
    # Additional methods for direct access to new architecture
    def get_core_engine(self):
        """Get direct access to MemFlowCore engine"""
        return self._processor.core
    
    def get_processor(self):
        """Get direct access to MemFlowProcessor"""
        return self._processor
    
    def validate_frames(self, frames: List[np.ndarray], frame_idx: int) -> bool:
        """Validate frame input format and parameters"""
        return self._processor.validate_frames(frames, frame_idx)
    
    def set_tile_mode(self, enabled: bool):
        """Enable or disable tile-based processing (not implemented for MemFlow)"""
        if enabled:
            print("Warning: Tile mode is not implemented for MemFlow. Using full-frame processing.")
        self.tile_mode = False
        self._processor.set_tile_mode(False)
    
    def set_sequence_length(self, length: int):
        """Set the sequence length for multi-frame processing"""
        self.sequence_length = length
        self._processor.set_sequence_length(length)
    
    def cleanup(self):
        """Clean up resources"""
        if self._processor is not None:
            self._processor.cleanup()
    
    # Additional MemFlow-specific methods
    def get_stage(self) -> str:
        """Get current training stage"""
        return self.stage
    
    def get_model_path(self) -> str:
        """Get path to model weights"""
        return self.model_path or f'MemFlow_ckpt/MemFlowNet_{self.stage}.pth'
    
    def supports_tile_mode(self) -> bool:
        """Check if tile mode is supported (always False for MemFlow)"""
        return False
    
    def get_recommended_sequence_length(self) -> int:
        """Get recommended sequence length for MemFlow (typically 2-3)"""
        return 3
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get information about the MemFlow framework"""
        return {
            "framework": "MemFlow",
            "architecture": "MemFlowNet",
            "stage": self.stage,
            "supports_tile_mode": False,
            "recommended_sequence_length": self.get_recommended_sequence_length(),
            "fast_mode_available": False,
            "compatibility_layer": "MemFlowInference"
        } 