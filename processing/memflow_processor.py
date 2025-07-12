"""
MemFlow Processor - High-level optical flow processing module

This module provides high-level MemFlow operations:
- Frame sequence preparation and management
- Compatibility with VideoFlow interface
- Progress tracking and coordination
- Format conversions (numpy <-> tensor)
- Input validation and error handling

This module uses MemFlowCore for actual model inference and provides
a complete processing pipeline for practical applications.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .base_flow_processor import BaseFlowProcessor
from .memflow_core import MemFlowCore


class MemFlowProcessor(BaseFlowProcessor):
    """
    High-level MemFlow processor for optical flow computation
    
    This class provides a complete processing pipeline:
    - Frame sequence preparation from numpy arrays
    - Compatibility with VideoFlow interface
    - Progress tracking integration
    - Format conversions and validation
    - Error handling and recovery
    
    Uses MemFlowCore internally for actual model inference.
    """
    
    def __init__(self, device: str = 'cuda', fast_mode: bool = False, tile_mode: bool = False,
                 sequence_length: int = 3, stage: str = 'sintel', model_path: str = None, **kwargs):
        """
        Initialize MemFlow processor
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode (currently not implemented for MemFlow)
            tile_mode: Enable tile-based processing (currently not implemented for MemFlow)
            sequence_length: Number of frames to use in sequence for inference
            stage: Training stage/dataset ('sintel', 'things', 'kitti')
            model_path: Custom path to model weights
            **kwargs: Additional configuration parameters
        """
        super().__init__(device, fast_mode, tile_mode, sequence_length, **kwargs)
        
        self.stage = stage
        self.model_path = model_path
        
        # Initialize core inference engine with model configuration
        self.core = MemFlowCore(device, fast_mode, stage, model_path)
        
        print(f"MemFlow Processor initialized:")
        print(f"  Device: {device}")
        print(f"  Fast mode: {fast_mode}")
        print(f"  Tile mode: {tile_mode} (note: not implemented for MemFlow)")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Stage: {stage}")
        print(f"  Model path: {model_path or f'MemFlow_ckpt/MemFlowNet_{stage}.pth'}")
    
    def load_model(self):
        """Load MemFlow model using core engine"""
        model_path = self.core.load_model()
        print(f"MemFlow model loaded successfully from: {model_path}")
    
    def prepare_frame_sequence(self, frames: List[np.ndarray], frame_idx: int) -> torch.Tensor:
        """
        Prepare frame sequence for MemFlow inference
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            
        Returns:
            frame_batch: Tensor in MemFlow format [1, T, 3, H, W], values 0.0-1.0
        """
        # MemFlow requires at least 2 frames
        sequence_length = max(2, self.sequence_length)
        
        # Determine frame indices for sequence
        total_frames = len(frames)
        end_idx = frame_idx + 1
        start_idx = max(0, end_idx - sequence_length)
        
        # Get frame sequence
        frame_sequence = frames[start_idx:end_idx]
        
        # Pad with first frame if needed
        while len(frame_sequence) < sequence_length:
            frame_sequence.insert(0, frame_sequence[0])
        
        # Convert to tensors
        tensor_frames = []
        for frame in frame_sequence:
            # Ensure frame is in correct format
            if frame.dtype == np.uint8:
                # Convert uint8 to float32 and normalize to [0,1]
                tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
            else:
                # Already float, assume it's in correct range
                tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
            tensor_frames.append(tensor)
        
        # Stack frames and add batch dimension
        batch = torch.stack(tensor_frames).unsqueeze(0).to(self.device)
        return batch
    
    def compute_optical_flow_tiled(self, frames: List[np.ndarray], frame_idx: int,
                                  tile_pbar: Optional[tqdm] = None, 
                                  overall_pbar: Optional[tqdm] = None) -> np.ndarray:
        """
        Compute optical flow using MemFlow (no actual tiling - compatibility method)
        
        Args:
            frames: List of frames
            frame_idx: Current frame index
            tile_pbar: Progress bar for current tile processing (updated for compatibility)
            overall_pbar: Progress bar for overall tiles progress (updated for compatibility)
            
        Returns:
            Full-resolution optical flow
        """
        # Update progress bars for compatibility
        if tile_pbar is not None:
            tile_pbar.set_description("MemFlow processing")
            tile_pbar.reset(total=1)
        
        if overall_pbar is not None:
            overall_pbar.set_description("MemFlow full-frame")
            overall_pbar.reset(total=1)
        
        # Compute flow using standard method
        flow = self.compute_optical_flow(frames, frame_idx)
        
        # Update progress bars
        if tile_pbar is not None:
            tile_pbar.update(1)
        if overall_pbar is not None:
            overall_pbar.update(1)
        
        return flow
    
    def compute_optical_flow_with_progress(self, frames: List[np.ndarray], frame_idx: int, 
                                         tile_pbar: Optional[tqdm] = None) -> np.ndarray:
        """
        Compute optical flow with progress updates
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            tile_pbar: Optional progress bar for processing updates
            
        Returns:
            flow_np: Optical flow as numpy array [H, W, 2], values in pixels
        """
        # Update progress bar
        if tile_pbar is not None:
            tile_pbar.set_description("MemFlow processing")
            tile_pbar.reset(total=1)
        
        # Compute flow
        flow = self.compute_optical_flow(frames, frame_idx)
        
        # Update progress bar
        if tile_pbar is not None:
            tile_pbar.update(1)
        
        return flow
    
    def set_tile_mode(self, enabled: bool):
        """Enable or disable tile-based processing (not implemented for MemFlow)"""
        if enabled:
            print("Warning: Tile mode is not implemented for MemFlow. Using full-frame processing.")
        self.tile_mode = False  # Always keep disabled for MemFlow
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        info = super().get_model_info()
        if self.core is not None:
            info.update({
                "stage": self.stage,
                "model_path": self.model_path,
                "note": "Tile mode not supported for MemFlow"
            })
        return info
    
    def cleanup(self):
        """Clean up resources"""
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        if self.core is not None:
            self.core.model = None
            self.core.processor = None
     