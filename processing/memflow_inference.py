#!/usr/bin/env python3
"""
MemFlow Inference Module

Simplified inference layer using direct MemFlow integration.
Based on original inference.py logic for maximum quality and simplicity.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm

from .memflow_simple import MemFlowSimple


class MemFlowInference:
    """
    Simplified MemFlow inference using direct integration.
    
    Based on original inference.py logic for maximum quality and simplicity.
    No isolated processes, no complex architecture.
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
        
        # Initialize simplified processor
        self.processor = MemFlowSimple(
            device=device,
            model_path=model_path,
            stage=stage,
            sequence_length=sequence_length
        )
        
        # Legacy attributes for compatibility
        self.model = None  # Will be set after loading
        self.cfg = None    # Will be set after loading
        
        print(f"MemFlow Inference initialized - simplified direct integration")
    
    def load_model(self):
        """Load MemFlow model through simplified processor"""
        self.processor.load_model()
        
        # Set legacy attributes for compatibility
        self.model = self.processor.model
        self.cfg = self.processor.cfg
        
        print("MemFlow model loaded through simplified processor")
    
    def prepare_frame_sequence(self, frames: List[np.ndarray], frame_idx: int) -> torch.Tensor:
        """Prepare frame sequence for MemFlow processing"""
        return self.processor._prepare_frames(frames, frame_idx)
    
    def compute_optical_flow(self, frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        """Compute optical flow using MemFlow model"""
        return self.processor.compute_optical_flow(frames, frame_idx)
    
    def compute_optical_flow_with_progress(self, frames: List[np.ndarray], frame_idx: int,
                                         tile_pbar: Optional[tqdm] = None) -> np.ndarray:
        """Compute optical flow with progress updates"""
        if tile_pbar is not None:
            tile_pbar.set_description("MemFlow processing")
            tile_pbar.update(1)
        return self.processor.compute_optical_flow(frames, frame_idx)
    
    def calculate_tile_grid(self, width: int, height: int, tile_size: int = 1280) -> Tuple:
        """Calculate tile grid (delegates to MemFlowSimple)"""
        return MemFlowSimple.calculate_tile_grid(width, height, tile_size)
    
    def extract_tile(self, frame: np.ndarray, tile_info: Dict[str, int]) -> np.ndarray:
        """Extract tile from frame (compatibility - returns full frame)"""
        return frame  # MemFlow doesn't use tiles, return full frame
    
    def compute_optical_flow_tiled(self, frames: List[np.ndarray], frame_idx: int,
                                 tile_pbar: Optional[tqdm] = None,
                                 overall_pbar: Optional[tqdm] = None) -> np.ndarray:
        """Compute optical flow using tiled processing (actually full frame for MemFlow)"""
        return self.processor.compute_optical_flow_tiled(frames, frame_idx, tile_pbar, overall_pbar)
    
    def get_processor(self) -> MemFlowSimple:
        """Get access to underlying MemFlowSimple processor"""
        return self.processor
    
    def get_core_engine(self):
        """Get access to underlying processor (compatibility method)"""
        return self.processor
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        return self.processor.get_memory_usage()
    
    def cleanup(self):
        """Clean up resources"""
        self.processor.cleanup() 