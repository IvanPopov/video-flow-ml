#!/usr/bin/env python3
"""
MemFlow Processor Module

High-level MemFlow processor for data preparation and tile-based processing.
Handles frame sequence preparation, input validation, and format conversions.
Uses MemFlowCore internally for actual model inference.
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm

from .memflow_core import MemFlowCore


class MemFlowProcessor:
    """
    High-level MemFlow processor that handles data preparation and tile processing.
    
    This class provides:
    - Frame sequence preparation from numpy arrays
    - Input validation and error handling
    - Format conversions between numpy and tensors
    - Uses MemFlowCore internally for model inference
    """
    
    def __init__(self, device='cuda', model_path='MemFlow_ckpt/MemFlowNet_sintel.pth', 
                 stage='sintel', sequence_length=3):
        """
        Initialize MemFlow processor.
        
        Args:
            device: Processing device ('cuda' or 'cpu')
            model_path: Path to MemFlow model weights
            stage: Training stage configuration ('sintel', 'things', 'kitti')
            sequence_length: Number of frames to use (MemFlow typically uses 3)
        """
        self.device = device
        self.model_path = model_path
        self.stage = stage
        self.sequence_length = max(2, sequence_length)  # MemFlow needs at least 2 frames
        
        # Initialize core engine
        self.core_engine = MemFlowCore(
            device=device,
            model_path=model_path,
            stage=stage
        )
        
        print(f"MemFlow Processor initialized:")
        print(f"  Sequence length: {self.sequence_length} frames")
        print(f"  Note: MemFlow processes frame pairs (last 2 frames of sequence)")
    
    def load_model(self):
        """Load MemFlow model through core engine"""
        self.core_engine.load_model()
    
    def validate_frame_sequence(self, frames: List[np.ndarray]) -> bool:
        """
        Validate frame sequence format and consistency.
        
        Args:
            frames: List of numpy arrays representing video frames
            
        Returns:
            True if valid, raises exception otherwise
        """
        if not frames:
            raise ValueError("Frame sequence is empty")
        
        if len(frames) < 2:
            raise ValueError(f"Need at least 2 frames for optical flow, got {len(frames)}")
        
        # Check frame format consistency
        first_frame = frames[0]
        if len(first_frame.shape) != 3:
            raise ValueError(f"Frames must be 3D (H, W, C), got shape: {first_frame.shape}")
        
        height, width, channels = first_frame.shape
        if channels != 3:
            raise ValueError(f"Frames must have 3 channels (RGB), got: {channels}")
        
        if height < 64 or width < 64:
            raise ValueError(f"Frame dimensions must be at least 64x64, got: {height}x{width}")
        
        # Check all frames have same dimensions
        for i, frame in enumerate(frames[1:], 1):
            if frame.shape != first_frame.shape:
                raise ValueError(f"Frame {i} shape {frame.shape} differs from first frame {first_frame.shape}")
        
        return True
    
    def prepare_frame_sequence(self, frames: List[np.ndarray], frame_idx: int) -> torch.Tensor:
        """
        Prepare frame sequence for MemFlow processing.
        
        Args:
            frames: List of video frames as numpy arrays [H, W, C]
            frame_idx: Target frame index for flow computation
            
        Returns:
            Preprocessed tensor [1, T, C, H, W] ready for MemFlow
        """
        self.validate_frame_sequence(frames)
        
        # Determine frame indices for sequence
        total_frames = len(frames)
        
        # For MemFlow, we use the last sequence_length frames up to frame_idx
        end_idx = frame_idx + 1
        start_idx = max(0, end_idx - self.sequence_length)
        
        # Get frame sequence
        frame_sequence = frames[start_idx:end_idx]
        
        # If we don't have enough frames, pad by repeating first frame
        while len(frame_sequence) < self.sequence_length:
            frame_sequence.insert(0, frame_sequence[0])
        
        # Convert to tensors and normalize
        tensor_frames = []
        for frame in frame_sequence:
            # Ensure frame is RGB uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Convert to tensor [C, H, W] and normalize to [0, 1]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            tensor_frames.append(frame_tensor)
        
        # Stack into [T, C, H, W] then add batch dimension
        sequence_tensor = torch.stack(tensor_frames, dim=0)  # [T, C, H, W]
        sequence_tensor = sequence_tensor.unsqueeze(0)       # [1, T, C, H, W]
        
        return sequence_tensor
    
    def compute_optical_flow(self, frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        """
        Compute optical flow for a specific frame using MemFlow.
        
        Args:
            frames: List of video frames as numpy arrays [H, W, C]
            frame_idx: Target frame index for flow computation
            
        Returns:
            Optical flow as numpy array [H, W, 2]
        """
        # Prepare input tensor
        input_tensor = self.prepare_frame_sequence(frames, frame_idx)
        
        # Compute flow using core engine
        flow_tensor = self.core_engine.compute_flow_from_tensor(input_tensor)
        
        # Convert to numpy [H, W, 2] (move to CPU first)
        flow_numpy = flow_tensor.permute(1, 2, 0).cpu().numpy()
        
        return flow_numpy
    
    def compute_optical_flow_with_progress(self, frames: List[np.ndarray], frame_idx: int, 
                                         tile_pbar: Optional[tqdm] = None) -> np.ndarray:
        """
        Compute optical flow with progress tracking (for tile mode compatibility).
        
        Args:
            frames: List of video frames as numpy arrays [H, W, C]
            frame_idx: Target frame index for flow computation
            tile_pbar: Progress bar for tile processing (unused for MemFlow, but kept for compatibility)
            
        Returns:
            Optical flow as numpy array [H, W, 2]
        """
        # Note: MemFlow doesn't use tile processing, but we keep this interface for compatibility
        if tile_pbar is not None:
            tile_pbar.set_description("MemFlow processing")
            tile_pbar.update(1)
        
        return self.compute_optical_flow(frames, frame_idx)
    
    @staticmethod
    def calculate_tile_grid(width: int, height: int, tile_size: int = 1280) -> Tuple:

        """
        Calculate tile grid (compatibility method - MemFlow doesn't use tiles).
        
        Args:
            width: Frame width
            height: Frame height  
            tile_size: Tile size (ignored)
            
        Returns:
            Tuple compatible with VideoFlow tile interface (no actual tiling)
        """
        # MemFlow processes full frames, not tiles
        # Return single "tile" covering the entire frame for compatibility
        tiles_info = [{
            'x': 0, 'y': 0,
            'width': width, 'height': height,
            'tile_idx': 0
        }]
        
        return width, height, 1, 1, tiles_info
    
    def extract_tile(self, frame: np.ndarray, tile_info: Dict[str, int]) -> np.ndarray:
        """
        Extract tile from frame (compatibility method - returns full frame).
        
        Args:
            frame: Input frame [H, W, C]
            tile_info: Tile information (ignored)
            
        Returns:
            Full frame (MemFlow doesn't use tiling)
        """
        return frame
    
    def compute_optical_flow_tiled(self, frames: List[np.ndarray], frame_idx: int,
                                 tile_pbar: Optional[tqdm] = None, 
                                 overall_pbar: Optional[tqdm] = None) -> np.ndarray:
        """
        Compute optical flow using "tiled" processing (actually full frame for MemFlow).
        
        Args:
            frames: List of video frames as numpy arrays [H, W, C]
            frame_idx: Target frame index for flow computation
            tile_pbar: Progress bar for individual tiles (minimal update for MemFlow)
            overall_pbar: Progress bar for overall tile progress (minimal update for MemFlow)
            
        Returns:
            Optical flow as numpy array [H, W, 2]
        """
        # Update progress bars to show activity
        if tile_pbar is not None:
            tile_pbar.set_description("MemFlow full-frame")
            tile_pbar.reset(total=1)
            tile_pbar.update(1)
        
        if overall_pbar is not None:
            overall_pbar.set_description("MemFlow processing")
            overall_pbar.reset(total=1)
            overall_pbar.update(1)
        
        # Compute flow on full frame
        return self.compute_optical_flow(frames, frame_idx)
    
    def get_core_engine(self) -> MemFlowCore:
        """Get access to underlying MemFlowCore engine"""
        return self.core_engine
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        return self.core_engine.get_memory_usage()
    
    def cleanup(self):
        """Clean up resources"""
        self.core_engine.cleanup() 