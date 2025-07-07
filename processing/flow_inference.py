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


class VideoFlowInference:
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
        # Delegate to VideoFlowProcessor
        self._processor = VideoFlowProcessor(device, fast_mode, tile_mode, sequence_length, 
                                           dataset, architecture, variant)
        
        # Store parameters for compatibility
        self.device = device
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self.dataset = dataset
        self.architecture = architecture
        self.variant = variant
        
        # Legacy attributes for backward compatibility
        self.model = None  # Will be set when model is loaded
        self.cfg = None    # Will be set when model is loaded
    
    def load_model(self):
        """Load VideoFlow MOF model"""
        self._processor.load_model()
        
        # Set legacy attributes for backward compatibility
        self.model = self._processor.core.model
        self.cfg = self._processor.core.cfg
    
    def calculate_tile_grid(self, width, height, tile_size=1280):
        """
        Calculate tile grid for fixed square tiles (optimized for VideoFlow MOF model)
        
        Args:
            width, height: Original frame dimensions
            tile_size: Fixed tile size (default: 1280x1280, optimal for MOF model)
            
        Returns:
            (tile_width, tile_height, cols, rows, tiles_info)
        """
        return self._processor.calculate_tile_grid(width, height, tile_size)
    
    def extract_tile(self, frame, tile_info):
        """Extract a tile from the frame without padding"""
        return self._processor.extract_tile(frame, tile_info)
    
    def prepare_frame_sequence(self, frames, frame_idx):
        """Prepare frame sequence for VideoFlow MOF model"""
        return self._processor.prepare_frame_sequence(frames, frame_idx)
    
    def compute_optical_flow(self, frames, frame_idx):
        """Compute optical flow using VideoFlow model"""
        return self._processor.compute_optical_flow(frames, frame_idx)
    
    def compute_optical_flow_with_progress(self, frames, frame_idx, tile_pbar=None):
        """Compute optical flow with progress updates for tile processing"""
        return self._processor.compute_optical_flow_with_progress(frames, frame_idx, tile_pbar)
    
    def compute_optical_flow_tiled(self, frames, frame_idx, tile_pbar=None, overall_pbar=None):
        """
        Compute optical flow using tile-based processing with 1280x1280 square tiles
        
        Args:
            frames: List of frames
            frame_idx: Current frame index
            tile_pbar: Progress bar for current tile processing
            overall_pbar: Progress bar for overall tiles progress
            
        Returns:
            Full-resolution optical flow
        """
        return self._processor.compute_optical_flow_tiled(frames, frame_idx, tile_pbar, overall_pbar)
    
    def is_model_loaded(self):
        """Check if VideoFlow model is loaded"""
        return self._processor.is_model_loaded()
    
    def get_model_info(self):
        """Get information about loaded model"""
        info = self._processor.get_model_info()
        
        # Add compatibility layer information
        if info["status"] == "loaded":
            info["compatibility_layer"] = "VideoFlowInference"
        
        return info
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        return self._processor.get_memory_usage()
    
    # Additional methods for direct access to new architecture
    def get_core_engine(self):
        """Get direct access to VideoFlowCore engine"""
        return self._processor.core
    
    def get_processor(self):
        """Get direct access to VideoFlowProcessor"""
        return self._processor
    
    def validate_frames(self, frames, frame_idx):
        """Validate frame input format and parameters"""
        return self._processor.validate_frames(frames, frame_idx)
    
    def set_tile_mode(self, enabled):
        """Enable or disable tile-based processing"""
        self.tile_mode = enabled
        self._processor.set_tile_mode(enabled)
    
    def set_sequence_length(self, length):
        """Set the sequence length for multi-frame processing"""
        self.sequence_length = length
        self._processor.set_sequence_length(length) 