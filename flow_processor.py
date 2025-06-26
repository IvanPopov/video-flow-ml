#!/usr/bin/env python3
"""
VideoFlow Optical Flow Processor

Pure VideoFlow implementation for optical flow generation with gamedev encoding.
Processes only first 1000 frames of the video.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add VideoFlow core to path
sys.path.insert(0, os.path.join(os.getcwd(), 'VideoFlow'))
sys.path.insert(0, os.path.join(os.getcwd(), 'VideoFlow', 'core'))

# Import VideoFlow modules
from core.Networks import build_network
from utils.utils import InputPadder
from configs.multiframes_sintel_submission import get_cfg

class VideoFlowProcessor:
    def __init__(self, device='auto', fast_mode=False, tile_mode=False, sequence_length=5, flow_smoothing=0.0):
        """Initialize VideoFlow processor with pure VideoFlow implementation"""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = device
            
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self.flow_smoothing = flow_smoothing
        self.model = None
        self.input_padder = None
        self.cfg = None
        self.previous_smoothed_flow = None  # For temporal flow smoothing
        
        print(f"VideoFlow Processor initialized - Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Fast mode: {fast_mode}")
        print(f"Tile mode: {tile_mode}")
        print(f"Sequence length: {sequence_length} frames")
        if flow_smoothing > 0:
            print(f"Flow smoothing: {flow_smoothing:.2f} (color-consistency stabilization enabled)")
        
    def load_videoflow_model(self):
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
        
    def get_video_fps(self, video_path):
        """Get video FPS for time calculations"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    
    def time_to_frame(self, time_seconds, fps):
        """Convert time in seconds to frame number"""
        return int(time_seconds * fps)
    
    def extract_frames(self, video_path, max_frames=1000, start_frame=0):
        """Extract frames from video starting at start_frame"""
        end_frame = start_frame + max_frames
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check bounds
        if start_frame >= total_frames:
            raise ValueError(f"Start frame {start_frame} exceeds total frames {total_frames}")
        
        actual_end = min(end_frame, total_frames)
        frames_to_extract = actual_end - start_frame
        
        # Apply fast mode resolution reduction
        if self.fast_mode:
            # More aggressive resolution reduction for fast mode
            # Target maximum 256x256, but maintain aspect ratio
            max_dimension = 256
            scale_factor = min(max_dimension / orig_width, max_dimension / orig_height)
            
            # Don't upscale if already small
            if scale_factor > 1.0:
                scale_factor = 1.0
            
            # Apply additional reduction for large videos
            if max(orig_width, orig_height) > 512:
                scale_factor = min(scale_factor, 0.25)  # Quarter size for very large videos
            elif max(orig_width, orig_height) > 256:
                scale_factor = min(scale_factor, 0.5)   # Half size for medium videos
            
            width = int(orig_width * scale_factor)
            height = int(orig_height * scale_factor)
            
            # Ensure dimensions are even (required for some codecs) and minimum 64x64
            width = max(64, width - (width % 2))
            height = max(64, height - (height % 2))
            
            print(f"Fast mode: aggressive resolution reduction from {orig_width}x{orig_height} to {width}x{height} (scale: {scale_factor:.2f})")
        else:
            width = orig_width
            height = orig_height

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frames = []
        pbar = tqdm(total=frames_to_extract, desc="Extracting frames")
        
        for i in range(frames_to_extract):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if in fast mode
            if self.fast_mode:
                frame_rgb = cv2.resize(frame_rgb, (width, height))
            
            frames.append(frame_rgb)
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        return frames, fps, width, height, start_frame
    
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
        
    def compute_optical_flow_with_progress(self, frames, frame_idx, tile_pbar=None):
        """Compute optical flow with progress updates for tile processing"""
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
        
    def compute_optical_flow(self, frames, frame_idx):
        """Compute optical flow using VideoFlow model"""
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
    
    def stabilize_optical_flow(self, current_flow, current_frame, previous_frame):
        """
        Apply color-consistency based stabilization to optical flow
        
        This method stabilizes flow vectors by testing multiple candidates and selecting
        the one that best preserves color when used for reprojection (TAA-style validation).
        
        Args:
            current_flow: Current frame's optical flow (HWC numpy array)
            current_frame: Current frame (HWC RGB, 0-255)
            previous_frame: Previous frame (HWC RGB, 0-255)
            
        Returns:
            Stabilized optical flow
        """
        if self.flow_smoothing <= 0.0:
            # No stabilization - return original flow
            return current_flow
            
        if self.previous_smoothed_flow is None:
            # First frame - initialize with current flow
            self.previous_smoothed_flow = current_flow.copy()
            return current_flow
        
        h, w = current_flow.shape[:2]
        stabilized_flow = current_flow.copy()
        
        # Convert frames to float for processing
        current_float = current_frame.astype(np.float32)
        previous_float = previous_frame.astype(np.float32)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Generate flow candidates for testing
        candidates = []
        weights = []
        
        # Candidate 1: Current raw flow
        candidates.append(current_flow)
        weights.append(0.4)
        
        # Candidate 2: Previous smoothed flow (temporal consistency)
        candidates.append(self.previous_smoothed_flow)
        weights.append(0.3)
        
        # Candidate 3: Exponentially smoothed flow
        alpha = 1.0 - self.flow_smoothing
        smooth_flow = alpha * current_flow + self.flow_smoothing * self.previous_smoothed_flow
        candidates.append(smooth_flow)
        weights.append(0.3)
        
        # Test each candidate and compute color consistency score
        best_flow = current_flow.copy()
        
        # Process in blocks to avoid memory issues
        block_size = 64
        for by in range(0, h, block_size):
            for bx in range(0, w, block_size):
                # Define block boundaries
                y_end = min(by + block_size, h)
                x_end = min(bx + block_size, w)
                
                block_h = y_end - by
                block_w = x_end - bx
                
                # Extract block data
                block_current = current_float[by:y_end, bx:x_end]
                block_y_coords = y_coords[by:y_end, bx:x_end]
                block_x_coords = x_coords[by:y_end, bx:x_end]
                
                best_score = np.full((block_h, block_w), float('inf'))
                best_block_flow = current_flow[by:y_end, bx:x_end].copy()
                
                # Test each candidate flow
                for candidate_flow, weight in zip(candidates, weights):
                    block_flow = candidate_flow[by:y_end, bx:x_end]
                    
                    # Calculate reprojection coordinates using inverted flow (TAA-style)
                    # Invert flow to get "where did this pixel come from"
                    inv_flow = -block_flow
                    
                    prev_x = block_x_coords + inv_flow[:, :, 0]
                    prev_y = block_y_coords + inv_flow[:, :, 1]
                    
                    # Clamp coordinates to valid range
                    prev_x = np.clip(prev_x, 0, w - 1)
                    prev_y = np.clip(prev_y, 0, h - 1)
                    
                    # Bilinear interpolation from previous frame
                    x0 = np.floor(prev_x).astype(int)
                    x1 = np.minimum(x0 + 1, w - 1)
                    y0 = np.floor(prev_y).astype(int)
                    y1 = np.minimum(y0 + 1, h - 1)
                    
                    # Interpolation weights
                    wx = prev_x - x0
                    wy = prev_y - y0
                    
                    # Sample previous frame with bilinear interpolation
                    reprojected = np.zeros_like(block_current)
                    
                    for c in range(3):  # RGB channels
                        reprojected[:, :, c] = (
                            previous_float[y0, x0, c] * (1 - wx) * (1 - wy) +
                            previous_float[y0, x1, c] * wx * (1 - wy) +
                            previous_float[y1, x0, c] * (1 - wx) * wy +
                            previous_float[y1, x1, c] * wx * wy
                        )
                    
                    # Calculate color consistency score (lower is better)
                    color_diff = np.mean((block_current - reprojected) ** 2, axis=2)
                    
                    # Apply weight to the score
                    weighted_score = color_diff / weight
                    
                    # Update best flow where this candidate is better
                    better_mask = weighted_score < best_score
                    best_score[better_mask] = weighted_score[better_mask]
                    best_block_flow[better_mask] = block_flow[better_mask]
                
                # Store the best flow for this block
                best_flow[by:y_end, bx:x_end] = best_block_flow
        
        # Apply additional smoothing to reduce remaining jitter
        # Use bilateral filter to preserve edges while smoothing
        best_flow_u = cv2.bilateralFilter(best_flow[:,:,0].astype(np.float32), 5, 10.0, 10.0)
        best_flow_v = cv2.bilateralFilter(best_flow[:,:,1].astype(np.float32), 5, 10.0, 10.0)
        best_flow = np.stack([best_flow_u, best_flow_v], axis=2)
        
        # Update previous smoothed flow for next iteration
        self.previous_smoothed_flow = best_flow.copy()
        
        return best_flow
        
    def save_flow_flo(self, flow, filename):
        """
        Save optical flow in Middlebury .flo format (lossless)
        
        Args:
            flow: Raw optical flow data [H, W, 2]
            filename: Output filename with .flo extension
        """
        # Convert to numpy if tensor
        if torch.is_tensor(flow):
            flow = flow.cpu().numpy()
        
        height, width = flow.shape[:2]
        
        with open(filename, 'wb') as f:
            # Write magic number
            f.write(b'PIEH')
            
            # Write dimensions
            import struct
            f.write(struct.pack('<I', width))
            f.write(struct.pack('<I', height))
            
            # Write flow data as float32
            flow_data = flow.astype(np.float32)
            f.write(flow_data.tobytes())
    
    def save_flow_npz(self, flow, filename, frame_idx=None, metadata=None):
        """
        Save optical flow in NumPy .npz format (lossless, compressed)
        
        Args:
            flow: Raw optical flow data [H, W, 2]
            filename: Output filename with .npz extension
            frame_idx: Frame index for metadata
            metadata: Additional metadata dictionary
        """
        # Convert to numpy if tensor
        if torch.is_tensor(flow):
            flow = flow.cpu().numpy()
        
        # Prepare save data
        save_data = {'flow': flow.astype(np.float32)}
        
        if frame_idx is not None:
            save_data['frame_idx'] = frame_idx
            
        if metadata is not None:
            save_data.update(metadata)
            
        # Save with compression
        np.savez_compressed(filename, **save_data)
    
    def save_optical_flow_files(self, flow, base_filename, frame_idx, save_format):
        """
        Save optical flow in specified format(s)
        
        Args:
            flow: Raw optical flow data [H, W, 2]
            base_filename: Base filename without extension
            frame_idx: Current frame index
            save_format: Format to save ('flo', 'npz', or 'both')
        """
        if save_format in ['flo', 'both']:
            flo_filename = f"{base_filename}_frame_{frame_idx:06d}.flo"
            self.save_flow_flo(flow, flo_filename)
            
        if save_format in ['npz', 'both']:
            npz_filename = f"{base_filename}_frame_{frame_idx:06d}.npz"
            metadata = {
                'frame_idx': frame_idx,
                'shape': flow.shape,
                'dtype': str(flow.dtype)
            }
            self.save_flow_npz(flow, npz_filename, frame_idx, metadata)
    
    def load_flow_flo(self, filename):
        """
        Load optical flow from Middlebury .flo format
        
        Args:
            filename: Input .flo filename
            
        Returns:
            Optical flow data [H, W, 2] as numpy array
        """
        with open(filename, 'rb') as f:
            # Read magic number
            magic = f.read(4)
            if magic != b'PIEH':
                raise ValueError(f"Invalid .flo file magic number: {magic}")
            
            # Read dimensions
            import struct
            width = struct.unpack('<I', f.read(4))[0]
            height = struct.unpack('<I', f.read(4))[0]
            
            # Read flow data
            flow_data = f.read(width * height * 2 * 4)  # 2 channels, 4 bytes per float32
            flow = np.frombuffer(flow_data, dtype=np.float32)
            flow = flow.reshape(height, width, 2)
            
        return flow
    
    def load_flow_npz(self, filename):
        """
        Load optical flow from NumPy .npz format
        
        Args:
            filename: Input .npz filename
            
        Returns:
            Dictionary with 'flow' and metadata
        """
        data = np.load(filename)
        return dict(data)
    
    def generate_flow_cache_path(self, input_path, start_frame, max_frames, sequence_length, fast_mode, tile_mode):
        """
        Generate cache directory path based on video processing parameters that affect raw optical flow computation
        
        Args:
            input_path: Input video path
            start_frame: Starting frame
            max_frames: Number of frames to process
            sequence_length: VideoFlow sequence length
            fast_mode: Fast mode flag
            tile_mode: Tile mode flag
            
        Returns:
            Cache directory path
        """
        # Create cache identifier based on processing parameters that affect raw flow computation
        # NOTE: flow_smoothing is NOT included because cache stores raw flow before stabilization
        video_name = Path(input_path).stem
        cache_params = [
            f"seq{sequence_length}",
            f"start{start_frame}",
            f"frames{max_frames}"
        ]
        
        if fast_mode:
            cache_params.append("fast")
        if tile_mode:
            cache_params.append("tile")
            
        cache_id = "_".join(cache_params)
        cache_dir_name = f"{video_name}_flow_cache_{cache_id}"
        
        # Place cache next to input video
        cache_path = Path(input_path).parent / cache_dir_name
        return str(cache_path)
    
    def check_flow_cache_exists(self, cache_dir, max_frames):
        """
        Check if complete flow cache exists for the requested number of frames
        
        Args:
            cache_dir: Cache directory path
            max_frames: Expected number of frames
            
        Returns:
            (exists, format) where format is 'flo', 'npz', or None
        """
        if not os.path.exists(cache_dir):
            return False, None
            
        # Check for .flo files
        flo_files = []
        npz_files = []
        
        for i in range(max_frames):
            flo_file = os.path.join(cache_dir, f"flow_frame_{i:06d}.flo")
            npz_file = os.path.join(cache_dir, f"flow_frame_{i:06d}.npz")
            
            if os.path.exists(flo_file):
                flo_files.append(flo_file)
            if os.path.exists(npz_file):
                npz_files.append(npz_file)
        
        # Determine which format is complete
        if len(flo_files) == max_frames:
            return True, 'flo'
        elif len(npz_files) == max_frames:
            return True, 'npz'
        else:
            return False, None
    
    def load_cached_flow(self, cache_dir, frame_idx, format_type='auto'):
        """
        Load cached optical flow for specific frame
        
        Args:
            cache_dir: Cache directory path
            frame_idx: Frame index to load
            format_type: 'flo', 'npz', or 'auto'
            
        Returns:
            Optical flow data [H, W, 2]
        """
        if format_type == 'auto':
            # Try .flo first, then .npz
            flo_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.flo")
            npz_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.npz")
            
            if os.path.exists(flo_file):
                return self.load_flow_flo(flo_file)
            elif os.path.exists(npz_file):
                npz_data = self.load_flow_npz(npz_file)
                return npz_data['flow']
            else:
                raise FileNotFoundError(f"No cached flow found for frame {frame_idx}")
        
        elif format_type == 'flo':
            flo_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.flo")
            return self.load_flow_flo(flo_file)
            
        elif format_type == 'npz':
            npz_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.npz")
            npz_data = self.load_flow_npz(npz_file)
            return npz_data['flow']
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def generate_flow_lods(self, flow_data, num_lods=5):
        """
        Generate Level-of-Detail (LOD) pyramid for flow data using arithmetic averaging
        
        Args:
            flow_data: Original flow data [H, W, 2]
            num_lods: Number of LOD levels to generate (default: 5)
            
        Returns:
            List of flow data at different LOD levels [original, lod1, lod2, ...]
        """
        lods = [flow_data]  # LOD 0 is original
        
        current_flow = flow_data.copy()
        
        for lod_level in range(1, num_lods):
            h, w = current_flow.shape[:2]
            
            # Check if we can create another LOD level
            if h < 4 or w < 4:
                # Need padding
                target_h = max(4, h)
                target_w = max(4, w)
                
                # Calculate padding
                pad_h = target_h - h
                pad_w = target_w - w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                # Create weight mask (1 for original data, 0 for padding)
                weight_mask = np.ones((h, w), dtype=np.float32)
                padded_weight = np.pad(weight_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                     mode='constant', constant_values=0)
                
                # Pad flow data
                padded_flow = np.pad(current_flow, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                                   mode='constant', constant_values=0)
                
                current_flow = padded_flow
                h, w = current_flow.shape[:2]
            else:
                # Create uniform weight mask
                padded_weight = np.ones((h, w), dtype=np.float32)
            
            # Downsample by factor of 2 using weighted averaging
            new_h = h // 2
            new_w = w // 2
            
            downsampled_flow = np.zeros((new_h, new_w, 2), dtype=np.float32)
            
            for y in range(new_h):
                for x in range(new_w):
                    # Get 2x2 block
                    y_start, y_end = y * 2, min((y + 1) * 2, h)
                    x_start, x_end = x * 2, min((x + 1) * 2, w)
                    
                    flow_block = current_flow[y_start:y_end, x_start:x_end]
                    weight_block = padded_weight[y_start:y_end, x_start:x_end]
                    
                    # Calculate weighted average
                    total_weight = np.sum(weight_block)
                    if total_weight > 0:
                        # Weighted average for each channel
                        weighted_flow_u = np.sum(flow_block[:, :, 0] * weight_block) / total_weight
                        weighted_flow_v = np.sum(flow_block[:, :, 1] * weight_block) / total_weight
                        
                        # Scale flow vectors by 0.5 (since we're downsampling by 2)
                        downsampled_flow[y, x, 0] = weighted_flow_u * 0.5
                        downsampled_flow[y, x, 1] = weighted_flow_v * 0.5
                    else:
                        downsampled_flow[y, x] = 0
            
            lods.append(downsampled_flow)
            current_flow = downsampled_flow
            
            # Update weight mask for next iteration
            padded_weight = np.ones((new_h, new_w), dtype=np.float32)
        
        return lods
    
    def save_flow_lods(self, lods, cache_dir, frame_idx):
        """
        Save LOD pyramid for a frame
        
        Args:
            lods: List of LOD flow data
            cache_dir: Cache directory
            frame_idx: Frame index
        """
        lod_dir = os.path.join(cache_dir, 'lods')
        os.makedirs(lod_dir, exist_ok=True)
        
        for lod_level, lod_data in enumerate(lods):
            filename = os.path.join(lod_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
            metadata = {
                'frame_idx': frame_idx,
                'lod_level': lod_level,
                'shape': lod_data.shape,
                'dtype': str(lod_data.dtype)
            }
            self.save_flow_npz(lod_data, filename, frame_idx, metadata)
    
    def load_flow_lod(self, cache_dir, frame_idx, lod_level=0):
        """
        Load specific LOD level for a frame
        
        Args:
            cache_dir: Cache directory
            frame_idx: Frame index
            lod_level: LOD level to load (0 = original)
            
        Returns:
            Flow data for specified LOD level
        """
        lod_dir = os.path.join(cache_dir, 'lods')
        filename = os.path.join(lod_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
        
        if os.path.exists(filename):
            npz_data = self.load_flow_npz(filename)
            return npz_data['flow']
        else:
            raise FileNotFoundError(f"LOD {lod_level} not found for frame {frame_idx}")
    
    def check_flow_lods_exist(self, cache_dir, max_frames, num_lods=5):
        """
        Check if LOD pyramid exists for all frames
        
        Args:
            cache_dir: Cache directory path
            max_frames: Expected number of frames
            num_lods: Number of LOD levels expected
            
        Returns:
            True if all LODs exist for all frames
        """
        lod_dir = os.path.join(cache_dir, 'lods')
        if not os.path.exists(lod_dir):
            return False
        
        for frame_idx in range(max_frames):
            for lod_level in range(num_lods):
                filename = os.path.join(lod_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
                if not os.path.exists(filename):
                    return False
        
        return True
    
    def generate_lods_for_cache(self, cache_dir, max_frames, num_lods=5):
        """
        Generate LOD pyramids for all frames in cache
        
        Args:
            cache_dir: Cache directory path
            max_frames: Number of frames to process
            num_lods: Number of LOD levels to generate
        """
        print(f"Generating {num_lods} LOD levels for {max_frames} frames...")
        
        with tqdm(total=max_frames, desc="Generating LODs") as pbar:
            for frame_idx in range(max_frames):
                try:
                    # Load original flow data
                    flow_data = self.load_cached_flow(cache_dir, frame_idx)
                    
                    # Generate LOD pyramid
                    lods = self.generate_flow_lods(flow_data, num_lods)
                    
                    # Save LOD pyramid
                    self.save_flow_lods(lods, cache_dir, frame_idx)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error generating LODs for frame {frame_idx}: {e}")
                    pbar.update(1)
                    continue

    def encode_hsv_format(self, flow, width, height):
        """
        Encode optical flow in HSV format (standard visualization):
        - Hue: Flow direction (angle)
        - Saturation: Flow magnitude (normalized)
        - Value: Constant brightness
        """
        # Handle NaN and inf values first
        flow = np.nan_to_num(flow, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        
        # Normalize angle to [0, 2π] and convert to hue [0, 180] for OpenCV
        hue = (angle + np.pi) / (2 * np.pi) * 180
        hue = np.clip(hue, 0, 180).astype(np.uint8)
        
        # Normalize magnitude for saturation
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            saturation = (magnitude / max_magnitude * 255).astype(np.uint8)
            # print(f"HSV Flow - max magnitude: {max_magnitude:.4f}")
        else:
            saturation = np.zeros_like(magnitude, dtype=np.uint8)
            # print("HSV Flow - no motion detected")
        
        # Set constant value (brightness)
        value = np.full_like(magnitude, 255, dtype=np.uint8)
        
        # Create HSV image
        hsv = np.stack([hue, saturation, value], axis=2)
        
        # Convert HSV to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb
    
    def encode_gamedev_format(self, flow, width, height):
        """
        Encode optical flow in gamedev format:
        - Normalize flow by image dimensions
        - Scale and clamp to [-20, +20] range  
        - Map to [0, 1] where 0 = -20, 1 = +20
        - Store in RG channels (R=horizontal, G=vertical)
        """
        # Normalize flow by image dimensions
        norm_flow = flow.copy()
        norm_flow[:, :, 0] /= width    # Horizontal flow
        norm_flow[:, :, 1] /= height   # Vertical flow
        
        # Scale to make motion visible
        norm_flow *= 200
        
        # Clamp to [-20, +20] range
        clamped = np.clip(norm_flow, -20, 20)
        
        # Map [-20, +20] to [0, 1]: 0 = -20, 1 = +20
        encoded = (clamped + 20) / 40
        encoded = np.clip(encoded, 0, 1)
        
        # Create RGB image
        h, w = flow.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, 0] = encoded[:, :, 0]  # R channel: horizontal flow
        rgb[:, :, 1] = encoded[:, :, 1]  # G channel: vertical flow
        rgb[:, :, 2] = 0.0               # B channel: unused
        
        # Convert to 8-bit, handle NaN and inf values
        rgb_8bit = rgb * 255
        rgb_8bit = np.nan_to_num(rgb_8bit, nan=0.0, posinf=255.0, neginf=0.0)
        return rgb_8bit.astype(np.uint8)
    
    def encode_torchvision_format(self, flow, width, height):
        """
        Encode optical flow using torchvision.utils.flow_to_image format:
        - Uses the standard torchvision visualization which creates a color wheel
        - More accurate color mapping compared to custom HSV implementations
        - Consistent with PyTorch/torchvision ecosystem
        """
        try:
            from torchvision.utils import flow_to_image
        except ImportError:
            print("Warning: torchvision not available, falling back to HSV format")
            return self.encode_hsv_format(flow, width, height)
        
        # Convert numpy flow to torch tensor
        # torchvision expects flow in CHW format (channels first)
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # HWC -> CHW
        
        # Add batch dimension if needed
        if flow_tensor.dim() == 3:
            flow_tensor = flow_tensor.unsqueeze(0)  # Add batch dimension: CHW -> BCHW
        
        # Use torchvision's flow_to_image function
        # This creates a color wheel visualization similar to Middlebury flow dataset
        with torch.no_grad():
            flow_image_tensor = flow_to_image(flow_tensor)
        
        # Remove batch dimension and convert back to numpy
        if flow_image_tensor.dim() == 4:
            flow_image_tensor = flow_image_tensor.squeeze(0)  # Remove batch: BCHW -> CHW
        
        # Convert from CHW to HWC and to numpy
        flow_image_np = flow_image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # torchvision returns values in [0, 1] range, convert to [0, 255]
        flow_image_np = (flow_image_np * 255).astype(np.uint8)
        
        return flow_image_np
    
    def apply_taa_effect(self, current_frame, flow_pixels=None, previous_taa_frame=None, alpha=0.1, use_flow=True):
        """
        Apply TAA (Temporal Anti-Aliasing) effect with or without optical flow
        
        Args:
            current_frame: Current frame (RGB, 0-255)
            flow: Inverted optical flow from previous frame (HWC, normalized) - only used if use_flow=True
            previous_taa_frame: Previous TAA result frame
            alpha: Blending weight (0.0 = full history, 1.0 = no history)
            use_flow: Whether to use optical flow for reprojection
        
        Returns:
            TAA processed frame
        """
        if previous_taa_frame is None:
            # First frame, no history
            return current_frame.astype(np.float32)
        
        current_float = current_frame.astype(np.float32)
        
        if not use_flow or flow_pixels is None:
            # Simple temporal blending without flow (basic TAA)
            taa_result = alpha * current_float + (1 - alpha) * previous_taa_frame
            return taa_result
        
        # Flow-based TAA (motion-compensated)
        h, w = current_frame.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Calculate previous pixel positions using flow
        prev_x = x_coords + flow_pixels[:, :, 0]
        prev_y = y_coords + flow_pixels[:, :, 1]
        
        # Handle NaN and inf values
        prev_x = np.nan_to_num(prev_x, nan=0.0, posinf=w-1, neginf=0.0)
        prev_y = np.nan_to_num(prev_y, nan=0.0, posinf=h-1, neginf=0.0)
        
        # Clamp coordinates to valid range
        prev_x = np.clip(prev_x, 0, w - 1)
        prev_y = np.clip(prev_y, 0, h - 1)
        
        # Bilinear interpolation from previous frame
        x0 = np.floor(prev_x).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(prev_y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        
        # Ensure indices are valid
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        
        # Interpolation weights
        wx = prev_x - x0
        wy = prev_y - y0
        
        # Sample previous TAA frame with bilinear interpolation
        reprojected = np.zeros_like(current_frame, dtype=np.float32)
        
        for c in range(3):  # RGB channels
            reprojected[:, :, c] = (
                previous_taa_frame[y0, x0, c] * (1 - wx) * (1 - wy) +
                previous_taa_frame[y0, x1, c] * wx * (1 - wy) +
                previous_taa_frame[y1, x0, c] * (1 - wx) * wy +
                previous_taa_frame[y1, x1, c] * wx * wy
            )
        
        # Exponential moving average (TAA blending)
        taa_result = alpha * current_float + (1 - alpha) * reprojected
        
        return taa_result
    
    def add_text_overlay(self, frame, text, position='top-left', font_scale=0.4, color=(255, 255, 255), thickness=1):
        """
        Add text overlay to frame
        
        Args:
            frame: Input frame (BGR format for OpenCV)
            text: Text to add
            position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
            font_scale: Size of the font
            color: Text color (BGR)
            thickness: Text thickness
        
        Returns:
            Frame with text overlay
        """
        frame_with_text = frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Calculate position
        margin = 5
        if position == 'top-left':
            pos = (margin, text_size[1] + margin)
        elif position == 'top-right':
            pos = (w - text_size[0] - margin, text_size[1] + margin)
        elif position == 'bottom-left':
            pos = (margin, h - margin)
        elif position == 'bottom-right':
            pos = (w - text_size[0] - margin, h - margin)
        else:
            pos = (margin, text_size[1] + margin)  # Default to top-left
        
        # Add black outline for better visibility
        cv2.putText(frame_with_text, text, pos, font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Add white text
        cv2.putText(frame_with_text, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
        
        return frame_with_text
        
    def create_side_by_side(self, original, flow_viz, vertical=False, flow_only=False, 
                           taa_frame=None, taa_simple_frame=None, model_name="VideoFlow", fast_mode=False, flow_format="gamedev"):
        """Create side-by-side, top-bottom, flow-only, or TAA visualization with text overlays"""
        # Ensure same dimensions
        h, w = original.shape[:2]
        if flow_viz.shape[:2] != (h, w):
            flow_viz = cv2.resize(flow_viz, (w, h))
        
        # Convert to BGR for video writing and add text overlays
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        flow_bgr = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)
        
        # Add text overlays
        mode_text = " (Fast)" if fast_mode else ""
        
        orig_bgr = self.add_text_overlay(orig_bgr, f"Original{mode_text}", 'top-left')
        flow_bgr = self.add_text_overlay(flow_bgr, f"Optical Flow{mode_text}", 'top-left')
        flow_bgr = self.add_text_overlay(flow_bgr, f"{model_name} ({flow_format.upper()})", 'bottom-left')
        
        if flow_only:
            # Return only optical flow
            return flow_bgr
        
        if taa_frame is not None and taa_simple_frame is not None:
            # Both TAA modes: flow-based and simple
            taa_bgr = cv2.cvtColor(np.clip(taa_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            taa_simple_bgr = cv2.cvtColor(np.clip(taa_simple_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Add TAA text overlays
            taa_bgr = self.add_text_overlay(taa_bgr, "TAA + Inv.Flow", 'top-left')
            taa_bgr = self.add_text_overlay(taa_bgr, "Alpha: 0.1", 'bottom-left')
            
            taa_simple_bgr = self.add_text_overlay(taa_simple_bgr, "TAA Simple", 'top-left')
            taa_simple_bgr = self.add_text_overlay(taa_simple_bgr, "Alpha: 0.1", 'bottom-left')
            
            if vertical:
                # Stack vertically: Original | Flow | TAA+Flow | TAA Simple
                return np.concatenate([orig_bgr, flow_bgr, taa_bgr, taa_simple_bgr], axis=0)
            else:
                # Create 2x2 grid layout
                # Top row: Original | Flow
                top_row = np.concatenate([orig_bgr, flow_bgr], axis=1)
                # Bottom row: TAA+Flow | TAA Simple
                bottom_row = np.concatenate([taa_bgr, taa_simple_bgr], axis=1)
                # Stack rows vertically
                return np.concatenate([top_row, bottom_row], axis=0)
        elif taa_frame is not None:
            # Single TAA mode (backward compatibility)
            taa_bgr = cv2.cvtColor(np.clip(taa_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            taa_bgr = self.add_text_overlay(taa_bgr, "TAA + Inv.Flow", 'top-left')
            taa_bgr = self.add_text_overlay(taa_bgr, "Alpha: 0.1", 'bottom-left')
            
            if vertical:
                # Stack vertically: Original | Flow | TAA
                return np.concatenate([orig_bgr, flow_bgr, taa_bgr], axis=0)
            else:
                # Stack horizontally: Original | Flow | TAA
                return np.concatenate([orig_bgr, flow_bgr, taa_bgr], axis=1)
        else:
            if vertical:
                # Concatenate vertically (top-bottom)
                return np.concatenate([orig_bgr, flow_bgr], axis=0)
            else:
                # Concatenate horizontally (side-by-side)
                return np.concatenate([orig_bgr, flow_bgr], axis=1)
        
    def generate_output_filename(self, input_path, output_dir, start_time=None, duration=None, 
                                start_frame=0, max_frames=1000, vertical=False, flow_only=False, taa=False):
        """Generate automatic output filename based on parameters"""
        import os
        
        # Always use results directory
        results_dir = os.path.join(output_dir, "results") if output_dir != "results" else "results"
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Build filename parts
        parts = [base_name]
        
        # Add time/frame info
        if start_time is not None:
            parts.append(f"{start_time}s")
        elif start_frame > 0:
            parts.append(f"f{start_frame}")
            
        if duration is not None:
            parts.append(f"{duration}s")
        elif max_frames != 1000:
            parts.append(f"{max_frames}f")
        
        # Add mode info
        if self.fast_mode:
            parts.append("fast")
        
        if self.tile_mode:
            parts.append("tile")
        
        if flow_only:
            parts.append("flow")
        elif taa:
            parts.append("taa")
        elif vertical:
            parts.append("vert")
        
        # Join parts and add extension
        filename = "_".join(parts) + ".mp4"
        return os.path.join(results_dir, filename)
    
    def process_video(self, input_path, output_path, max_frames=1000, start_frame=0, 
                     start_time=None, duration=None, vertical=False, flow_only=False, taa=False, flow_format='gamedev', 
                     save_flow=None, force_recompute=False, use_flow_cache=None):
        """Main processing function"""
        import os
        
        # Handle time-based parameters
        if start_time is not None or duration is not None:
            fps = self.get_video_fps(input_path)
            print(f"Video FPS: {fps:.2f}")
            
            if start_time is not None:
                start_frame = self.time_to_frame(start_time, fps)
                print(f"Start time: {start_time}s -> frame {start_frame}")
            
            if duration is not None:
                max_frames = self.time_to_frame(duration, fps)
                print(f"Duration: {duration}s -> {max_frames} frames")
        
        # Check if output_path is a directory and generate filename if needed
        if os.path.isdir(output_path):
            output_path = self.generate_output_filename(
                input_path, output_path, start_time, duration, 
                start_frame, max_frames, vertical, flow_only, taa
            )
            print(f"Auto-generated output filename: {os.path.basename(output_path)}")
        
        print(f"Processing: {input_path} -> {output_path}")
        print(f"Frame range: {start_frame} to {start_frame + max_frames - 1}")
        
        # Extract frames first
        frames, fps, width, height, actual_start = self.extract_frames(input_path, max_frames=max_frames, start_frame=start_frame)
        
        # Setup flow caching
        flow_cache_dir = None
        use_cached_flow = False
        cached_flow_format = None
        cache_save_format = 'npz'  # Default cache format
        
        if use_flow_cache is not None:
            # Use specific cache directory
            flow_cache_dir = use_flow_cache
            if os.path.exists(flow_cache_dir):
                cache_exists, cached_flow_format = self.check_flow_cache_exists(flow_cache_dir, len(frames))
                if cache_exists:
                    use_cached_flow = True
                    print(f"Using optical flow cache from: {flow_cache_dir} (format: {cached_flow_format})")
                else:
                    print(f"Warning: Specified cache directory incomplete or missing: {flow_cache_dir}")
            else:
                print(f"Warning: Specified cache directory not found: {flow_cache_dir}")
        else:
            # Generate automatic cache directory
            flow_cache_dir = self.generate_flow_cache_path(
                input_path, start_frame, len(frames), self.sequence_length, 
                self.fast_mode, self.tile_mode
            )
            
            if not force_recompute:
                cache_exists, cached_flow_format = self.check_flow_cache_exists(flow_cache_dir, len(frames))
                if cache_exists:
                    use_cached_flow = True
                    print(f"Found existing optical flow cache: {flow_cache_dir} (format: {cached_flow_format})")
                else:
                    print(f"No existing cache found, will compute and save to: {flow_cache_dir}")
            else:
                print(f"Force recompute enabled, will overwrite cache: {flow_cache_dir}")
        
        # Check and generate LOD pyramids if needed
        if use_cached_flow:
            # Check if LOD pyramids exist
            lods_exist = self.check_flow_lods_exist(flow_cache_dir, len(frames))
            if not lods_exist:
                print("LOD pyramids not found, generating...")
                self.generate_lods_for_cache(flow_cache_dir, len(frames))
                print("LOD pyramids generated successfully!")
            else:
                print("LOD pyramids found in cache")
        
        # Load VideoFlow model only if we need to compute flow
        if not use_cached_flow:
            self.load_videoflow_model()
        
        # Setup flow saving directory if requested or if we need to cache
        flow_base_filename = None
        if save_flow is not None or (not use_cached_flow):
            # Determine where to save flow data
            if save_flow is not None:
                # Create flow directory next to output video
                output_dir = os.path.dirname(output_path)
                output_name = os.path.splitext(os.path.basename(output_path))[0]
                flow_dir = os.path.join(output_dir, f"{output_name}_flow")
                os.makedirs(flow_dir, exist_ok=True)
                flow_base_filename = os.path.join(flow_dir, "flow")
                print(f"Saving optical flow to: {flow_dir}")
            
            # Also save to cache if we're computing
            if not use_cached_flow:
                os.makedirs(flow_cache_dir, exist_ok=True)
                # If no explicit save format, default to npz for cache
                if save_flow is not None:
                    cache_save_format = save_flow
                else:
                    cache_save_format = 'npz'
        
        # Setup output video using processed frame dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if flow_only:
            output_size = (width, height)  # Flow only: processed dimensions
        elif taa:
            if vertical:
                output_size = (width, height * 4)  # Vertical: same width, quad height (Original + Flow + TAA+Flow + TAA Simple)
            else:
                output_size = (width * 2, height * 2)  # 2x2 grid: double width, double height
        elif vertical:
            output_size = (width, height * 2)  # Vertical: same width, double height
        else:
            output_size = (width * 2, height)  # Horizontal: double width, same height
        out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        
        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")
        
        # Process each frame
        import time
        frame_times = []
        
        # TAA state
        previous_taa_frame = None
        previous_taa_simple_frame = None
        previous_flow = None  # Store previous frame's optical flow for TAA
        
        # Create progress bars for tile mode or single progress bar for normal mode
        if self.tile_mode:
            # Calculate tile count for first frame to setup progress bars
            current_frame = frames[0]
            height_temp, width_temp = current_frame.shape[:2]
            _, _, _, _, tiles_info_temp = self.calculate_tile_grid(width_temp, height_temp)
            total_tiles = len(tiles_info_temp)
            
            # Create two progress bars for tile mode
            main_pbar = tqdm(total=len(frames), desc="Frame processing", 
                           position=0, leave=True, ncols=100,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            tile_pbar = tqdm(total=4, desc="Tile processing", 
                           position=1, leave=False, ncols=100,
                           bar_format='{desc}: {bar}| {n_fmt}/{total_fmt}')
            
            overall_tile_pbar = tqdm(total=total_tiles, desc="Overall tiles", 
                                   position=2, leave=False, ncols=100,
                                   bar_format='{desc}: {bar}| {n_fmt}/{total_fmt}')
        else:
            # Single progress bar for normal mode
            main_pbar = tqdm(total=len(frames), desc="VideoFlow processing", 
                           unit="frame", ncols=100, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            tile_pbar = None
            overall_tile_pbar = None
        
        for i in range(len(frames)):
            start_time = time.time()
            
            # Get optical flow (from cache or compute)
            if use_cached_flow:
                # Load from cache (raw flow before stabilization)
                raw_flow = self.load_cached_flow(flow_cache_dir, i, cached_flow_format)
            else:
                # Compute optical flow using VideoFlow (with tiling if enabled)
                if self.tile_mode:
                    # Reset overall tile progress for each frame
                    overall_tile_pbar.reset()
                    raw_flow = self.compute_optical_flow_tiled(frames, i, tile_pbar, overall_tile_pbar)
                else:
                    raw_flow = self.compute_optical_flow_tiled(frames, i)
                
                # Save raw flow to cache BEFORE stabilization
                cache_base_filename = os.path.join(flow_cache_dir, "flow")
                self.save_optical_flow_files(raw_flow, cache_base_filename, i, cache_save_format)
            
            # Save raw optical flow if explicitly requested (before stabilization)
            if save_flow is not None and flow_base_filename is not None:
                self.save_optical_flow_files(raw_flow, flow_base_filename, i, save_flow)
            
            # Apply color-consistency based stabilization if enabled
            if self.flow_smoothing > 0.0:
                if i > 0:  # Need previous frame for stabilization
                    flow = self.stabilize_optical_flow(raw_flow, frames[i], frames[i-1])
                else:
                    # First frame - just initialize and use raw flow
                    self.previous_smoothed_flow = raw_flow.copy()
                    flow = raw_flow
            else:
                # No stabilization - use raw flow
                flow = raw_flow
            
            # Encode optical flow based on selected format
            if flow_format == 'hsv':
                flow_viz = self.encode_hsv_format(flow, width, height)
            elif flow_format == 'torchvision':
                flow_viz = self.encode_torchvision_format(flow, width, height)
            else:
                flow_viz = self.encode_gamedev_format(flow, width, height)
            
            # Apply TAA effects if requested
            taa_frame = None
            taa_simple_frame = None

            if taa:
                # Use previous frame's flow with inverted direction for TAA
                if previous_flow is not None:
                    # Apply flow-based TAA with alpha=0.1 (90% history, 10% current)
                    taa_result = self.apply_taa_effect(frames[i], previous_flow, previous_taa_frame, alpha=0.1, use_flow=True)
                    previous_taa_frame = taa_result.copy()
                    taa_frame = taa_result
                else:
                    # First frame or no previous flow - just copy current frame
                    taa_result = frames[i].astype(np.float32)
                    previous_taa_frame = taa_result.copy()
                    taa_frame = taa_result
                
                # Apply simple TAA (no flow) with alpha=0.1
                taa_simple_result = self.apply_taa_effect(frames[i], None, previous_taa_simple_frame, alpha=0.1, use_flow=False)
                previous_taa_simple_frame = taa_simple_result.copy()
                taa_simple_frame = taa_simple_result
                
            # Store current flow for next frame's TAA
            previous_flow = flow.copy()
            
            # Create combined frame (side-by-side, top-bottom, flow-only, or with TAA)
            model_name = "MOF_sintel" if hasattr(self, 'model') else "VideoFlow"
            combined = self.create_side_by_side(frames[i], flow_viz, vertical=vertical, flow_only=flow_only, 
                                              taa_frame=taa_frame, taa_simple_frame=taa_simple_frame,
                                              model_name=model_name, fast_mode=self.fast_mode, flow_format=flow_format)
            
            # Write frame
            out.write(combined)
            
            # Update timing and progress
            total_time = time.time() - start_time
            frame_times.append(total_time)
            
            # Calculate ETA based on recent frames (more accurate)
            if len(frame_times) > 5:
                avg_time = sum(frame_times[-5:]) / 5  # Average of last 5 frames
            else:
                avg_time = sum(frame_times) / len(frame_times)
            
            remaining_frames = len(frames) - i - 1
            eta_seconds = remaining_frames * avg_time
            
            # Update progress bar description
            if self.tile_mode:
                main_pbar.set_description(f"Frame {i+1}/{len(frames)} (ETA: {eta_seconds:.0f}s)")
            else:
                main_pbar.set_description(f"VideoFlow processing (ETA: {eta_seconds:.0f}s)")
            main_pbar.update(1)
        
        # Close progress bars
        main_pbar.close()
        if self.tile_mode:
            tile_pbar.close()
            overall_tile_pbar.close()
            print()  # Add spacing after tile progress bars
        out.release()
        
        # Generate LOD pyramids for cached flow if we computed new flow
        if not use_cached_flow and flow_cache_dir:
            print("Generating LOD pyramids for computed flow...")
            self.generate_lods_for_cache(flow_cache_dir, len(frames))
            print("LOD pyramids generated!")

def main():
    parser = argparse.ArgumentParser(description='VideoFlow Optical Flow Processor')
    parser.add_argument('--input', default='big_buck_bunny_720p_h264.mov',
                       help='Input video file')
    parser.add_argument('--output', default='results',
                       help='Output video file or directory (default: results)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Processing device')
    parser.add_argument('--frames', type=int, default=1000,
                       help='Maximum number of frames to process (default: 1000)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number (0-based, default: 0)')
    parser.add_argument('--start-time', type=float, default=None,
                       help='Starting time in seconds (overrides --start-frame)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration in seconds (overrides --frames)')
    parser.add_argument('--fast', action='store_true',
                       help='Enable fast mode (lower resolution, fewer iterations for faster processing)')
    parser.add_argument('--vertical', action='store_true',
                       help='Stack videos vertically (top-bottom) instead of horizontally (side-by-side)')
    parser.add_argument('--flow-only', action='store_true',
                       help='Output only optical flow visualization (no original video)')
    parser.add_argument('--taa', action='store_true',
                       help='Add TAA (Temporal Anti-Aliasing) effect visualization using inverted optical flow from previous frame')
    parser.add_argument('--flow-format', choices=['gamedev', 'hsv', 'torchvision'], default='gamedev',
                       help='Optical flow encoding format: gamedev (RG channels), hsv (standard visualization), or torchvision (color wheel)')
    parser.add_argument('--tile', action='store_true',
                       help='Enable tile-based processing: split frames into 1280x1280 square tiles (optimal for VideoFlow MOF model)')
    parser.add_argument('--sequence-length', type=int, default=5,
                       help='Number of frames to use in sequence for VideoFlow processing (default: 5, recommended: 5-9)')
    parser.add_argument('--flow-smoothing', type=float, default=0.0,
                       help='Color-consistency flow stabilization (0.0=disabled, 0.1-0.3=light, 0.4-0.7=medium, 0.8+=strong)')
    parser.add_argument('--save-flow', choices=['flo', 'npz', 'both'], default=None,
                       help='Save raw optical flow data: flo (Middlebury format), npz (NumPy format), both')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation of optical flow even if cached data exists')
    parser.add_argument('--use-flow-cache', type=str, default=None,
                       help='Use optical flow from specific cache directory instead of computing')
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive flow visualizer instead of creating video output')
    parser.add_argument('--show-tiles', action='store_true',
                       help='Only show tile grid calculation without processing video')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    if not os.path.exists('VideoFlow'):
        print("Error: VideoFlow repository not found. Please run:")
        print("git clone https://github.com/XiaoyuShi97/VideoFlow.git")
        return
        
    if not os.path.exists('VideoFlow_ckpt/MOF_sintel.pth'):
        print("Error: VideoFlow model weights not found.")
        print("Please download MOF_sintel.pth from:")
        print("https://github.com/XiaoyuShi97/VideoFlow")
        print("and place it in VideoFlow_ckpt/")
        return
    
    # Special mode: interactive flow visualizer
    if args.interactive:
        print(f"Interactive mode: computing/loading optical flow for {args.input}")
        
        # Create processor to compute or find flow cache
        processor = VideoFlowProcessor(device=args.device, fast_mode=args.fast, tile_mode=args.tile, 
                                      sequence_length=args.sequence_length, flow_smoothing=0.0)
        
        # Handle time-based parameters for frame extraction
        if args.start_time is not None or args.duration is not None:
            fps = processor.get_video_fps(args.input)
            print(f"Video FPS: {fps:.2f}")
            
            if args.start_time is not None:
                start_frame = processor.time_to_frame(args.start_time, fps)
                print(f"Start time: {args.start_time}s -> frame {start_frame}")
            else:
                start_frame = args.start_frame
            
            if args.duration is not None:
                max_frames = processor.time_to_frame(args.duration, fps)
                print(f"Duration: {args.duration}s -> {max_frames} frames")
            else:
                max_frames = args.frames
        else:
            start_frame = args.start_frame
            max_frames = args.frames
        
        # Extract frames to determine actual count
        frames, fps, width, height, actual_start = processor.extract_frames(args.input, max_frames=max_frames, start_frame=start_frame)
        
        # Determine flow cache directory
        if args.use_flow_cache is not None:
            flow_cache_dir = args.use_flow_cache
        else:
            flow_cache_dir = processor.generate_flow_cache_path(
                args.input, start_frame, len(frames), args.sequence_length, 
                args.fast, args.tile
            )
        
        # Check if cache exists, if not compute it
        cache_exists, cached_flow_format = processor.check_flow_cache_exists(flow_cache_dir, len(frames))
        
        if not cache_exists or args.force_recompute:
            print(f"Computing optical flow and saving to cache: {flow_cache_dir}")
            
            # Load model and compute flow
            processor.load_videoflow_model()
            os.makedirs(flow_cache_dir, exist_ok=True)
            
            # Compute flow for all frames
            from tqdm import tqdm
            pbar = tqdm(total=len(frames), desc="Computing optical flow", unit="frame")
            
            for i in range(len(frames)):
                # Compute flow
                if processor.tile_mode:
                    flow = processor.compute_optical_flow_tiled(frames, i)
                else:
                    flow = processor.compute_optical_flow(frames, i)
                
                # Save to cache
                cache_base_filename = os.path.join(flow_cache_dir, "flow")
                processor.save_optical_flow_files(flow, cache_base_filename, i, 'npz')
                
                pbar.update(1)
            
            pbar.close()
            print("Flow computation completed!")
            
            # Generate LOD pyramids for the computed flow
            print("Generating LOD pyramids...")
            processor.generate_lods_for_cache(flow_cache_dir, len(frames))
            print("LOD pyramids generated!")
        else:
            print(f"Using existing flow cache: {flow_cache_dir}")
            
            # Check and generate LOD pyramids if needed
            lods_exist = processor.check_flow_lods_exist(flow_cache_dir, len(frames))
            if not lods_exist:
                print("LOD pyramids not found, generating...")
                processor.generate_lods_for_cache(flow_cache_dir, len(frames))
                print("LOD pyramids generated successfully!")
            else:
                print("LOD pyramids found in cache")
        
        # Launch interactive visualizer
        print("Launching interactive flow visualizer...")
        import subprocess
        import sys
        
        visualizer_cmd = [
            sys.executable, "flow_visualizer.py",
            "--video", args.input,
            "--flow-dir", flow_cache_dir,
            "--start-frame", str(start_frame),
            "--max-frames", str(len(frames))
        ]
        
        try:
            subprocess.run(visualizer_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error launching visualizer: {e}")
        except FileNotFoundError:
            print("Error: flow_visualizer.py not found. Make sure it's in the same directory.")
        
        return
    
    # Special mode: only show tile grid calculation
    if args.show_tiles:
        print(f"Analyzing tile grid for: {args.input}")
        
        # Get video properties
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {args.input}")
            return
            
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"Video properties:")
        print(f"  Resolution: {orig_width}x{orig_height}")
        print(f"  Aspect ratio: {orig_width/orig_height:.3f}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s")
        
        # Apply fast mode scaling if enabled
        if args.fast:
            max_dimension = 256
            scale_factor = min(max_dimension / orig_width, max_dimension / orig_height)
            if scale_factor > 1.0:
                scale_factor = 1.0
            if max(orig_width, orig_height) > 512:
                scale_factor = min(scale_factor, 0.25)
            elif max(orig_width, orig_height) > 256:
                scale_factor = min(scale_factor, 0.5)
            
            width = max(64, int(orig_width * scale_factor) - (int(orig_width * scale_factor) % 2))
            height = max(64, int(orig_height * scale_factor) - (int(orig_height * scale_factor) % 2))
            
            print(f"\nFast mode scaling:")
            print(f"  Scale factor: {scale_factor:.3f}")
            print(f"  Processed resolution: {width}x{height}")
        else:
            width = orig_width
            height = orig_height
            print(f"\nProcessed resolution: {width}x{height} (no scaling)")
        
        # Create temporary processor just for tile calculation
        temp_processor = VideoFlowProcessor(device='cpu', fast_mode=False, tile_mode=True, 
                                          sequence_length=args.sequence_length, flow_smoothing=0.0)
        
        print(f"\nTile grid analysis:")
        tile_width, tile_height, cols, rows, tiles_info = temp_processor.calculate_tile_grid(width, height)
        
        print(f"\nDetailed tile information:")
        for i, tile_info in enumerate(tiles_info):
            print(f"  Tile {i+1}: position ({tile_info['x']}, {tile_info['y']}), "
                  f"size {tile_info['width']}x{tile_info['height']}")
        
        print(f"\nSummary:")
        print(f"  Grid: {cols}x{rows} tiles")
        print(f"  Tile aspect ratio: {tile_width/tile_height:.3f} (target: 1.000 - square)")
        print(f"  Total tiles: {len(tiles_info)}")
        return
    
    processor = VideoFlowProcessor(device=args.device, fast_mode=args.fast, tile_mode=args.tile, 
                                  sequence_length=args.sequence_length, flow_smoothing=args.flow_smoothing)
    
    try:
        # Create output filename with frame/time range if not specified
        if args.output == 'videoflow_result.mp4':
            # Default output name, add range info
            mode = ""
            if args.fast:
                mode += "_fast"
            if args.tile:
                mode += "_tile"
            if args.vertical:
                mode += "_vertical"
            if args.flow_only:
                mode += "_flow_only"
            if args.taa:
                mode += "_taa"
            
            if args.start_time is not None or args.duration is not None:
                # Use time-based naming
                fps = processor.get_video_fps(args.input)
                start_frame = processor.time_to_frame(args.start_time, fps) if args.start_time is not None else args.start_frame
                max_frames = processor.time_to_frame(args.duration, fps) if args.duration is not None else args.frames
                end_frame = start_frame + max_frames - 1
                
                start_time_str = f"{args.start_time:.1f}s" if args.start_time is not None else f"{start_frame}f"
                duration_str = f"{args.duration:.1f}s" if args.duration is not None else f"{max_frames}f"
                args.output = f"videoflow_{start_time_str}_{duration_str}{mode}.mp4"
            else:
                # Use frame-based naming
                end_frame = args.start_frame + args.frames - 1
                args.output = f"videoflow_{args.start_frame:06d}_{end_frame:06d}{mode}.mp4"
        
        processor.process_video(args.input, args.output, max_frames=args.frames, start_frame=args.start_frame,
                              start_time=args.start_time, duration=args.duration, vertical=args.vertical, 
                              flow_only=args.flow_only, taa=args.taa, flow_format=args.flow_format, save_flow=args.save_flow,
                              force_recompute=args.force_recompute, use_flow_cache=args.use_flow_cache)
        print("\n✓ VideoFlow processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 