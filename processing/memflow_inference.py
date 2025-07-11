#!/usr/bin/env python3
"""
MemFlow Inference Module

Compatibility layer that maintains backward compatibility with existing code
while providing access to the new modular MemFlow architecture.
Delegates all operations to MemFlowProcessor.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm

from .memflow_processor import MemFlowProcessor


class MemFlowInference:
    """
    MemFlow inference compatibility layer.
    
    This class maintains backward compatibility with existing code patterns
    while providing access to the new modular architecture through delegation.
    """
    
    def __init__(self, device='cuda', model_path='MemFlow_ckpt/MemFlowNet_sintel.pth', 
                 stage='sintel', sequence_length=3):
        """
        Initialize MemFlow inference engine.
        
        Args:
            device: Processing device ('cuda' or 'cpu')
            model_path: Path to MemFlow model weights
            stage: Training stage configuration ('sintel', 'things', 'kitti')
            sequence_length: Number of frames to use for MemFlow processing
        """
        self.device = device
        self.model_path = model_path
        self.stage = stage
        self.sequence_length = sequence_length
        
        # Initialize processor
        self.processor = MemFlowProcessor(
            device=device,
            model_path=model_path,
            stage=stage,
            sequence_length=sequence_length
        )
        
        # Legacy attributes for compatibility
        self.model = None  # Will be set after loading
        self.cfg = None    # Will be set after loading
        
        print(f"MemFlow Inference initialized - compatibility layer active")
    
    def load_model(self):
        """Load MemFlow model through processor"""
        self.processor.load_model()
        
        # Set legacy attributes for compatibility
        core_engine = self.processor.get_core_engine()
        self.model = core_engine.model
        self.cfg = core_engine.cfg
        
        print("MemFlow model loaded through compatibility layer")
    
    def prepare_frame_sequence(self, frames: List[np.ndarray], frame_idx: int) -> torch.Tensor:
        """Prepare frame sequence for MemFlow processing"""
        return self.processor.prepare_frame_sequence(frames, frame_idx)
    
    def compute_optical_flow(self, frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        """Compute optical flow using MemFlow model"""
        return self.processor.compute_optical_flow(frames, frame_idx)
    
    def compute_optical_flow_with_progress(self, frames: List[np.ndarray], frame_idx: int,
                                         tile_pbar: Optional[tqdm] = None) -> np.ndarray:
        """Compute optical flow with progress updates"""
        return self.processor.compute_optical_flow_with_progress(frames, frame_idx, tile_pbar)
    
    def calculate_tile_grid(self, width: int, height: int, tile_size: int = 1280) -> Tuple:
        """Calculate tile grid (delegates to MemFlowProcessor)"""
        return MemFlowProcessor.calculate_tile_grid(width, height, tile_size)
    
    def extract_tile(self, frame: np.ndarray, tile_info: Dict[str, int]) -> np.ndarray:
        """Extract tile from frame (compatibility - returns full frame)"""
        return self.processor.extract_tile(frame, tile_info)
    
    def compute_optical_flow_tiled(self, frames: List[np.ndarray], frame_idx: int,
                                 tile_pbar: Optional[tqdm] = None,
                                 overall_pbar: Optional[tqdm] = None) -> np.ndarray:
        """Compute optical flow using tiled processing (actually full frame for MemFlow)"""
        return self.processor.compute_optical_flow_tiled(frames, frame_idx, tile_pbar, overall_pbar)
    
    def get_processor(self) -> MemFlowProcessor:
        """Get access to underlying MemFlowProcessor"""
        return self.processor
    
    def get_core_engine(self):
        """Get access to underlying MemFlowCore engine"""
        return self.processor.get_core_engine()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        return self.processor.get_memory_usage()
    
    def cleanup(self):
        """Clean up resources"""
        self.processor.cleanup() 