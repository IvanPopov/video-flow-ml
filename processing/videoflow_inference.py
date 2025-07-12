"""
VideoFlow Optical Flow Inference Module - Compatibility Layer

This module maintains backward compatibility with existing code while using
the new modular architecture:
- VideoFlowCore: Low-level model operations
- VideoFlowProcessor: High-level processing pipeline

The VideoFlowInference class acts as a compatibility wrapper that delegates
to the appropriate new modules while preserving the original API.

WARNING: This module requires CUDA/GPU support for optimal performance.
The model loading and inference operations cannot be easily parallelized
across multiple processes due to CUDA context limitations.
"""

import os
import sys

# Import the new modular components
from .videoflow_processor import VideoFlowProcessor
from .base_flow_processor import BaseFlowInference


class VideoFlowInference(BaseFlowInference):
    """
    VideoFlow inference engine for optical flow computation (Compatibility Layer)
    
    This class provides backward compatibility with existing code while using
    the new modular architecture internally. It delegates all operations to
    VideoFlowProcessor, which in turn uses VideoFlowCore for model operations.
    
    For new code, consider using VideoFlowProcessor or VideoFlowCore directly.
    """
    
    def __init__(self, device, fast_mode=False, tile_mode=False, sequence_length=5, 
                 dataset='sintel', architecture='mof', variant='standard'):
        """
        Initialize VideoFlow inference engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            tile_mode: Enable tile-based processing for large frames
            sequence_length: Number of frames to use in sequence for inference
            dataset: Training dataset ('sintel', 'things', 'kitti')
            architecture: Model architecture ('mof' for MOFNet, 'bof' for BOFNet)
            variant: Model variant ('standard' or 'noise' for things_288960noise)
        """
        super().__init__(device, fast_mode, tile_mode, sequence_length, 
                         dataset=dataset, architecture=architecture, variant=variant)
        
        # Delegate to VideoFlowProcessor
        self._processor = VideoFlowProcessor(device, fast_mode, tile_mode, sequence_length, 
                                           dataset, architecture, variant)
        
        # Store parameters for compatibility
        self.dataset = dataset
        self.architecture = architecture
        self.variant = variant
    
    def load_model(self):
        """Load VideoFlow MOF model"""
        self._processor.load_model()
        
        # Set legacy attributes for backward compatibility
        self.model = self._processor.core.model
        self.cfg = self._processor.core.cfg
    
    # Additional VideoFlow-specific methods
    def get_dataset(self):
        """Get current dataset"""
        return self.dataset
    
    def get_architecture(self):
        """Get model architecture"""
        return self.architecture
    
    def get_variant(self):
        """Get model variant"""
        return self.variant
    
    def get_framework_info(self):
        """Get information about the VideoFlow framework"""
        return {
            "framework": "VideoFlow",
            "architecture": self.architecture.upper(),
            "dataset": self.dataset,
            "variant": self.variant,
            "supports_tile_mode": True,
            "recommended_sequence_length": 5,
            "fast_mode_available": True,
            "compatibility_layer": "VideoFlowInference"
        } 