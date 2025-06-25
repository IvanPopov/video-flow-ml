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
    def __init__(self, device='auto', fast_mode=False, tile_mode=False):
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
        self.model = None
        self.input_padder = None
        self.cfg = None
        
        print(f"VideoFlow Processor initialized - Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Fast mode: {fast_mode}")
        print(f"Tile mode: {tile_mode}")
        
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
        """Prepare 5-frame sequence for VideoFlow MOF model"""
        # Multi-frame: use 5 consecutive frames centered around current frame
        start_idx = max(0, frame_idx - 2)
        end_idx = min(len(frames), frame_idx + 3)
        sequence = frames[start_idx:end_idx]
        
        # Pad to exactly 5 frames
        while len(sequence) < 5:
            if start_idx == 0:
                sequence.insert(0, sequence[0])
            else:
                sequence.append(sequence[-1])
        
        # Ensure exactly 5 frames
        sequence = sequence[:5]
        
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
    
    def apply_taa_effect(self, current_frame, flow=None, previous_taa_frame=None, alpha=0.1, use_flow=True):
        """
        Apply TAA (Temporal Anti-Aliasing) effect with or without optical flow
        
        Args:
            current_frame: Current frame (RGB, 0-255)
            flow: Optical flow (HWC, normalized) - only used if use_flow=True
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
        
        if not use_flow or flow is None:
            # Simple temporal blending without flow (basic TAA)
            taa_result = alpha * current_float + (1 - alpha) * previous_taa_frame
            return taa_result
        
        # Flow-based TAA (motion-compensated)
        h, w = current_frame.shape[:2]
        
        # Convert flow to pixel coordinates
        flow_pixels = flow.copy()
        flow_pixels[:, :, 0] *= w  # Horizontal flow in pixels
        flow_pixels[:, :, 1] *= h  # Vertical flow in pixels
        
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
            taa_bgr = self.add_text_overlay(taa_bgr, "TAA + Flow", 'top-left')
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
            taa_bgr = self.add_text_overlay(taa_bgr, "TAA + Flow", 'top-left')
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
                     start_time=None, duration=None, vertical=False, flow_only=False, taa=False, flow_format='gamedev'):
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
        
        # Load VideoFlow model
        self.load_videoflow_model()
        
        # Extract frames
        frames, fps, width, height, actual_start = self.extract_frames(input_path, max_frames=max_frames, start_frame=start_frame)
        
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
            
            # Compute optical flow using VideoFlow (with tiling if enabled)
            if self.tile_mode:
                # Reset overall tile progress for each frame
                overall_tile_pbar.reset()
                flow = self.compute_optical_flow_tiled(frames, i, tile_pbar, overall_tile_pbar)
            else:
                flow = self.compute_optical_flow_tiled(frames, i)
            
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
                # Convert flow back to normalized format for TAA
                flow_normalized = flow.copy()
                flow_normalized[:, :, 0] /= width
                flow_normalized[:, :, 1] /= height
                
                # Apply flow-based TAA with alpha=0.1 (90% history, 10% current)
                taa_result = self.apply_taa_effect(frames[i], flow_normalized, previous_taa_frame, alpha=0.1, use_flow=True)
                previous_taa_frame = taa_result.copy()
                taa_frame = taa_result
                
                # Apply simple TAA (no flow) with alpha=0.1
                taa_simple_result = self.apply_taa_effect(frames[i], None, previous_taa_simple_frame, alpha=0.1, use_flow=False)
                previous_taa_simple_frame = taa_simple_result.copy()
                taa_simple_frame = taa_simple_result
            
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
                       help='Add TAA (Temporal Anti-Aliasing) effect visualization using optical flow')
    parser.add_argument('--flow-format', choices=['gamedev', 'hsv', 'torchvision'], default='gamedev',
                       help='Optical flow encoding format: gamedev (RG channels), hsv (standard visualization), or torchvision (color wheel)')
    parser.add_argument('--tile', action='store_true',
                       help='Enable tile-based processing: split frames into 1280x1280 square tiles (optimal for VideoFlow MOF model)')
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
        temp_processor = VideoFlowProcessor(device='cpu', fast_mode=False, tile_mode=True)
        
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
    
    processor = VideoFlowProcessor(device=args.device, fast_mode=args.fast, tile_mode=args.tile)
    
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
                              flow_only=args.flow_only, taa=args.taa, flow_format=args.flow_format)
        print("\n✓ VideoFlow processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 