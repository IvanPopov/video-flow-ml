"""
VideoFlow Optical Flow Inference Module

This module contains the VideoFlowInference class responsible for:
- Loading VideoFlow models
- Preparing frame sequences for inference
- Computing optical flow using VideoFlow models
- Tile-based processing for large frames

WARNING: This module requires CUDA/GPU support for optimal performance.
The model loading and inference operations cannot be easily parallelized
across multiple processes due to CUDA context limitations.
"""

import os
import sys
import torch
import numpy as np

# Add VideoFlow core to path
sys.path.insert(0, os.path.join(os.getcwd(), 'VideoFlow'))
sys.path.insert(0, os.path.join(os.getcwd(), 'VideoFlow', 'core'))

# VideoFlow imports
from core.Networks import build_network
from utils.utils import InputPadder
from configs.multiframes_sintel_submission import get_cfg


class VideoFlowInference:
    """
    VideoFlow inference engine for optical flow computation
    
    This class encapsulates all VideoFlow model operations including loading,
    frame preparation, and optical flow computation with optional tiling support.
    """
    
    def __init__(self, device, fast_mode=False, tile_mode=False, sequence_length=5):
        """
        Initialize VideoFlow inference engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            tile_mode: Enable tile-based processing for large frames
            sequence_length: Number of frames to use in sequence for inference
        """
        self.device = device
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        
        # Model components
        self.model = None
        self.cfg = None
        
        print(f"VideoFlow Inference Engine initialized:")
        print(f"  Device: {device}")
        print(f"  Fast mode: {fast_mode}")
        print(f"  Tile mode: {tile_mode}")
        print(f"  Sequence length: {sequence_length}")
    
    def load_model(self):
        """Load VideoFlow MOF model"""
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
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
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
        
        print(f"VideoFlow model loaded successfully from: {model_path}")
    
    def calculate_tile_grid(self, width, height, tile_size=1280):
        """
        Calculate tile grid for fixed square tiles (optimized for VideoFlow MOF model)
        
        Args:
            width, height: Original frame dimensions
            tile_size: Fixed tile size (default: 1280x1280, optimal for MOF model)
            
        Returns:
            (tile_width, tile_height, cols, rows, tiles_info)
        """
        # Use fixed square tiles
        tile_width = tile_size
        tile_height = tile_size
        
        # Calculate number of tiles needed
        cols = int(np.ceil(width / tile_width))
        rows = int(np.ceil(height / tile_height))
        
        # Calculate actual tile positions
        tiles_info = []
        for row in range(rows):
            for col in range(cols):
                # Calculate tile position
                x = col * tile_width
                y = row * tile_height
                
                # Calculate actual tile size (edge tiles might be smaller)
                actual_width = min(tile_width, width - x)
                actual_height = min(tile_height, height - y)
                
                tiles_info.append({
                    'x': x, 'y': y,
                    'width': actual_width, 'height': actual_height,
                    'col': col, 'row': row
                })
        
        return tile_width, tile_height, cols, rows, tiles_info
    
    def extract_tile(self, frame, tile_info):
        """Extract a tile from the frame without padding"""
        x, y = tile_info['x'], tile_info['y']
        w, h = tile_info['width'], tile_info['height']
        
        # Extract the tile from frame
        tile = frame[y:y+h, x:x+w]
        
        return tile
    
    def prepare_frame_sequence(self, frames, frame_idx):
        """Prepare frame sequence for VideoFlow MOF model"""
        # Multi-frame: use consecutive frames centered around current frame
        half_seq = self.sequence_length // 2
        start_idx = max(0, frame_idx - half_seq)
        end_idx = min(len(frames), frame_idx + half_seq + 1)
        sequence = frames[start_idx:end_idx]
        
        # Pad to exactly sequence_length frames
        while len(sequence) < self.sequence_length:
            if start_idx == 0:
                sequence.insert(0, sequence[0])
            else:
                sequence.append(sequence[-1])
        
        # Ensure exactly sequence_length frames
        sequence = sequence[:self.sequence_length]

        # Convert to tensors (same format as VideoFlow inference.py)
        tensors = []
        for frame in sequence:
            # Convert to tensor and normalize to [0,1], then change HWC to CHW
            tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
            tensors.append(tensor)
        
        # Stack frames and add batch dimension
        batch = torch.stack(tensors).unsqueeze(0).to(self.device)
        return batch
    
    def compute_optical_flow(self, frames, frame_idx):
        """Compute optical flow using VideoFlow model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare frame sequence
        frame_batch = self.prepare_frame_sequence(frames, frame_idx)
        
        # Create input padder
        padder = InputPadder(frame_batch.shape[-2:])
        frame_batch_padded = padder.pad(frame_batch)
        
        # Run VideoFlow inference (following their inference structure)
        with torch.no_grad():
            # VideoFlow forward pass
            flow_predictions, _ = self.model(frame_batch_padded, {})
            
            # Unpad results
            flow_tensor = padder.unpad(flow_predictions)
            
            # Get the middle flow (index 2 out of 0-4 for 5 frames)
            # Since we want flow for the center frame
            middle_idx = flow_tensor.shape[1] // 2
            flow_tensor = flow_tensor[0, middle_idx]  # Remove batch dim and get middle flow
            
            # Convert to numpy: CHW -> HWC  
            flow_np = flow_tensor.permute(1, 2, 0).cpu().numpy()
            
        return flow_np
    
    def compute_optical_flow_with_progress(self, frames, frame_idx, tile_pbar=None):
        """Compute optical flow with progress updates for tile processing"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if tile_pbar is not None:
            tile_pbar.set_description("Preparing frames")
            tile_pbar.reset()
        
        # Prepare frame sequence
        frame_batch = self.prepare_frame_sequence(frames, frame_idx)
        
        if tile_pbar is not None:
            tile_pbar.set_description("Creating padder")
            tile_pbar.update(1)
        
        # Create input padder
        padder = InputPadder(frame_batch.shape[-2:])
        frame_batch_padded = padder.pad(frame_batch)
        
        if tile_pbar is not None:
            tile_pbar.set_description("Running VideoFlow")
            tile_pbar.update(1)
        
        # Run VideoFlow inference
        with torch.no_grad():
            # VideoFlow forward pass
            flow_predictions, _ = self.model(frame_batch_padded, {})
            
            if tile_pbar is not None:
                tile_pbar.set_description("Processing output")
                tile_pbar.update(1)
            
            # Unpad results
            flow_tensor = padder.unpad(flow_predictions)
            
            # Get the middle flow
            middle_idx = flow_tensor.shape[1] // 2
            flow_tensor = flow_tensor[0, middle_idx]
            
            # Convert to numpy: CHW -> HWC  
            flow_np = flow_tensor.permute(1, 2, 0).cpu().numpy()
            
            if tile_pbar is not None:
                tile_pbar.set_description("Completed")
                tile_pbar.update(1)
        
        return flow_np
    
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
        if not self.tile_mode:
            # Use standard processing if tile mode is disabled
            return self.compute_optical_flow(frames, frame_idx)
        
        current_frame = frames[frame_idx]
        height, width = current_frame.shape[:2]
        
        # Calculate tile grid
        tile_width, tile_height, cols, rows, tiles_info = self.calculate_tile_grid(width, height)
        
        # Initialize full flow map
        full_flow = np.zeros((height, width, 2), dtype=np.float32)
        
        # Process each tile
        for i, tile_info in enumerate(tiles_info):
            # Update overall progress bar description
            if overall_pbar is not None:
                overall_pbar.set_description(f"Tile {i+1}/{len(tiles_info)} ({tile_info['width']}x{tile_info['height']})")
            
            # Extract tile from all frames in sequence
            tile_frames = []
            for frame in frames:
                tile = self.extract_tile(frame, tile_info)
                tile_frames.append(tile)
            
            # Compute flow for this tile with tile progress bar
            tile_flow = self.compute_optical_flow_with_progress(tile_frames, frame_idx, tile_pbar)
            
            # Place tile flow back into full flow map
            x, y = tile_info['x'], tile_info['y']
            w, h = tile_info['width'], tile_info['height']
            
            # Place flow back into full flow map
            full_flow[y:y+h, x:x+w] = tile_flow
            
            # Update overall progress
            if overall_pbar is not None:
                overall_pbar.update(1)
        
        return full_flow
    
    def is_model_loaded(self):
        """Check if VideoFlow model is loaded"""
        return self.model is not None
    
    def get_model_info(self):
        """Get information about loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "config": {
                "decoder_depth": getattr(self.cfg, 'decoder_depth', 'default'),
                "corr_levels": getattr(self.cfg, 'corr_levels', 'default'),
                "corr_radius": getattr(self.cfg, 'corr_radius', 'default'),
            },
            "fast_mode": self.fast_mode,
            "sequence_length": self.sequence_length,
            "device": str(self.device)
        } 