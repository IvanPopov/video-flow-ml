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

# Import our modules
from config import DeviceManager
from video import VideoInfo, FrameExtractor
from encoding import FlowEncoderFactory, encode_flow
from effects import TAAProcessor, apply_taa_effect

class AdaptiveOpticalFlowKalmanFilter:
    """
    Adaptive Kalman Filter for optical flow smoothing with support for sudden motion changes
    """
    def __init__(self, process_noise=0.01, measurement_noise=0.1, prediction_confidence=0.7, 
                 motion_model='constant_acceleration', outlier_threshold=3.0, min_track_length=3):
        self.base_process_noise = process_noise
        self.base_measurement_noise = measurement_noise
        self.prediction_confidence = prediction_confidence
        self.motion_model = motion_model
        self.outlier_threshold = outlier_threshold
        self.min_track_length = min_track_length
        
        # Adaptive parameters
        self.adaptation_factor = 2.0  # How much to increase noise during sudden changes
        self.motion_threshold = 5.0   # Threshold for detecting sudden motion changes
        self.adaptation_decay = 0.9   # How quickly to return to normal after adaptation
        
        # State tracking
        self.kalman_filters = {}      # Dict of Kalman filters per pixel
        self.motion_history = {}      # Motion history for adaptation
        self.frame_count = 0
        self.is_initialized = False
        
        # Statistics for monitoring
        self.adaptation_map = None    # Map of current adaptation levels
        self.outlier_count = 0
        self.total_pixels = 0
        
    def _create_kalman_filter(self, initial_position, initial_velocity):
        """Create a new Kalman filter for a pixel"""
        if self.motion_model == 'constant_velocity':
            # State: [x, y, vx, vy]
            state_size = 4
            measurement_size = 2
            
            # State transition matrix (constant velocity model)
            dt = 1.0
            transition_matrix = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
        else:  # constant_acceleration
            # State: [x, y, vx, vy, ax, ay]
            state_size = 6
            measurement_size = 2
            
            # State transition matrix (constant acceleration model)
            dt = 1.0
            dt2 = dt * dt / 2
            transition_matrix = np.array([
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
        
        # Create Kalman filter
        kf = cv2.KalmanFilter(state_size, measurement_size)
        
        # Set matrices
        kf.transitionMatrix = transition_matrix
        
        # Measurement matrix (we observe position)
        kf.measurementMatrix = np.zeros((measurement_size, state_size), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1  # x
        kf.measurementMatrix[1, 1] = 1  # y
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(state_size, dtype=np.float32) * self.base_process_noise
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(measurement_size, dtype=np.float32) * self.base_measurement_noise
        
        # Error covariance
        kf.errorCovPost = np.eye(state_size, dtype=np.float32)
        
        # Initialize state
        if self.motion_model == 'constant_velocity':
            kf.statePre = np.array([initial_position[0], initial_position[1], 
                                   initial_velocity[0], initial_velocity[1]], dtype=np.float32)
        else:  # constant_acceleration
            kf.statePre = np.array([initial_position[0], initial_position[1], 
                                   initial_velocity[0], initial_velocity[1], 0, 0], dtype=np.float32)
        
        kf.statePost = kf.statePre.copy()
        
        return kf
    
    def _detect_sudden_motion(self, pixel_key, current_velocity, predicted_velocity):
        """Detect sudden changes in motion for adaptive filtering"""
        if pixel_key not in self.motion_history:
            self.motion_history[pixel_key] = {'velocities': [], 'adaptations': []}
            return False, 1.0
        
        history = self.motion_history[pixel_key]
        
        # Calculate velocity change
        velocity_change = np.linalg.norm(current_velocity - predicted_velocity)
        
        # Calculate recent average velocity change
        if len(history['velocities']) > 0:
            recent_changes = history['velocities'][-3:]  # Last 3 frames
            avg_change = np.mean(recent_changes)
            std_change = np.std(recent_changes) if len(recent_changes) > 1 else 1.0
            
            # Detect sudden motion if change is significantly larger than recent average
            threshold = avg_change + self.motion_threshold * max(std_change, 0.1)
            sudden_motion = velocity_change > threshold
            
            # Calculate adaptation factor
            if sudden_motion:
                adaptation = min(self.adaptation_factor, 1.0 + velocity_change / max(avg_change, 0.1))
            else:
                # Gradually return to normal
                last_adaptation = history['adaptations'][-1] if history['adaptations'] else 1.0
                adaptation = max(1.0, last_adaptation * self.adaptation_decay)
        else:
            sudden_motion = False
            adaptation = 1.0
        
        # Update history
        history['velocities'].append(velocity_change)
        history['adaptations'].append(adaptation)
        
        # Keep only recent history
        if len(history['velocities']) > 10:
            history['velocities'] = history['velocities'][-10:]
            history['adaptations'] = history['adaptations'][-10:]
        
        return sudden_motion, adaptation
    
    def _is_outlier(self, measurement, prediction, covariance):
        """Detect outlier measurements using Mahalanobis distance"""
        if covariance is None:
            return False
        
        diff = measurement - prediction
        
        # Calculate Mahalanobis distance
        try:
            inv_cov = np.linalg.inv(covariance + np.eye(2) * 1e-6)  # Add small regularization
            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
            return mahal_dist > self.outlier_threshold
        except:
            # Fallback to Euclidean distance
            euclidean_dist = np.linalg.norm(diff)
            return euclidean_dist > self.outlier_threshold * 2
    
    def update(self, flow_field, frame_idx):
        """Update Kalman filters for entire flow field"""
        if flow_field is None:
            return flow_field
        
        h, w = flow_field.shape[:2]
        smoothed_flow = np.zeros_like(flow_field)
        
        # Initialize adaptation map
        if self.adaptation_map is None or self.adaptation_map.shape[:2] != (h, w):
            self.adaptation_map = np.ones((h, w), dtype=np.float32)
        
        self.frame_count += 1
        self.outlier_count = 0
        self.total_pixels = 0
        
        # Process pixels with subsampling for performance
        step = max(1, min(8, max(h, w) // 100))  # Adaptive step size
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                pixel_key = (y, x)
                current_flow = flow_field[y, x]
                current_position = np.array([x, y], dtype=np.float32)
                current_velocity = current_flow.astype(np.float32)
                
                self.total_pixels += 1
                
                # Skip zero flow during initialization
                if self.frame_count <= self.min_track_length and np.linalg.norm(current_velocity) < 0.1:
                    smoothed_flow[y, x] = current_flow
                    continue
                
                # Create new filter if needed
                if pixel_key not in self.kalman_filters:
                    self.kalman_filters[pixel_key] = self._create_kalman_filter(current_position, current_velocity)
                    smoothed_flow[y, x] = current_flow
                    continue
                
                kf = self.kalman_filters[pixel_key]
                
                # Predict
                predicted_state = kf.predict()
                predicted_position = predicted_state[:2]
                predicted_velocity = predicted_state[2:4] if self.motion_model == 'constant_velocity' else predicted_state[2:4]
                
                # Detect sudden motion and adapt
                sudden_motion, adaptation = self._detect_sudden_motion(pixel_key, current_velocity, predicted_velocity)
                self.adaptation_map[y, x] = adaptation
                
                # Adjust noise based on adaptation
                if adaptation > 1.1:  # Significant adaptation needed
                    # Temporarily increase process noise to allow faster adaptation
                    kf.processNoiseCov *= adaptation
                    kf.measurementNoiseCov /= np.sqrt(adaptation)  # Trust measurements more during sudden changes
                
                # Check for outliers
                measurement_position = current_position + current_velocity  # Where pixel moved to
                prediction_position = predicted_position + predicted_velocity
                
                if self._is_outlier(measurement_position, prediction_position, kf.errorCovPost[:2, :2]):
                    # Outlier detected - skip this measurement
                    self.outlier_count += 1
                    if self.motion_model == 'constant_velocity':
                        smoothed_velocity = predicted_state[2:4]
                    else:
                        smoothed_velocity = predicted_state[2:4]
                    smoothed_flow[y, x] = smoothed_velocity
                else:
                    # Normal measurement - update filter
                    measurement = current_position + current_velocity  # End position
                    kf.correct(measurement)
                    
                    # Get smoothed velocity from updated state
                    if self.motion_model == 'constant_velocity':
                        smoothed_velocity = kf.statePost[2:4]
                    else:
                        smoothed_velocity = kf.statePost[2:4]
                    
                    # Blend with original based on confidence
                    confidence = self.prediction_confidence * (2.0 - adaptation)  # Lower confidence during adaptation
                    confidence = np.clip(confidence, 0.1, 0.9)
                    
                    smoothed_flow[y, x] = (confidence * smoothed_velocity + 
                                         (1 - confidence) * current_velocity)
                
                # Restore normal noise levels
                if adaptation > 1.1:
                    kf.processNoiseCov /= adaptation
                    kf.measurementNoiseCov *= np.sqrt(adaptation)
        
        # Interpolate smoothed flow to full resolution
        if step > 1:
            smoothed_flow = self._interpolate_flow(smoothed_flow, flow_field, step)
        
        # Print statistics
        if self.frame_count % 30 == 0:  # Every 30 frames
            outlier_rate = self.outlier_count / max(self.total_pixels, 1) * 100
            avg_adaptation = np.mean(self.adaptation_map)
            print(f"Kalman Filter Stats - Frame {self.frame_count}: "
                  f"Outliers: {outlier_rate:.1f}%, Avg Adaptation: {avg_adaptation:.2f}")
        
        self.is_initialized = True
        return smoothed_flow
    
    def _interpolate_flow(self, sparse_flow, full_flow, step):
        """Interpolate sparse smoothed flow to full resolution"""
        h, w = full_flow.shape[:2]
        
        # Create coordinate grids
        y_sparse, x_sparse = np.mgrid[0:h:step, 0:w:step]
        y_full, x_full = np.mgrid[0:h, 0:w]
        
        # Interpolate each flow component
        from scipy.interpolate import griddata
        
        points = np.column_stack([y_sparse.ravel(), x_sparse.ravel()])
        
        # Interpolate flow_x
        values_x = sparse_flow[::step, ::step, 0].ravel()
        flow_x_interp = griddata(points, values_x, (y_full, x_full), method='linear', fill_value=0)
        
        # Interpolate flow_y  
        values_y = sparse_flow[::step, ::step, 1].ravel()
        flow_y_interp = griddata(points, values_y, (y_full, x_full), method='linear', fill_value=0)
        
        # Combine and blend with original
        interpolated = np.stack([flow_x_interp, flow_y_interp], axis=2)
        
        # Blend with original flow in areas without sparse coverage
        mask = np.zeros((h, w), dtype=np.float32)
        mask[::step, ::step] = 1
        mask = cv2.GaussianBlur(mask, (step*2+1, step*2+1), step/2)
        
        result = full_flow.copy()
        for c in range(2):
            result[:, :, c] = mask * interpolated[:, :, c] + (1 - mask) * full_flow[:, :, c]
        
        return result

class VideoFlowProcessor:
    def __init__(self, device='auto', fast_mode=False, tile_mode=False, sequence_length=5, flow_smoothing=0.0,
                 kalman_process_noise=0.01, kalman_measurement_noise=0.1, kalman_prediction_confidence=0.7,
                 kalman_motion_model='constant_acceleration', kalman_outlier_threshold=3.0, kalman_min_track_length=3):
        """Initialize VideoFlow processor with pure VideoFlow implementation"""
        # Initialize device manager
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device(device)
            
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self.flow_smoothing = flow_smoothing
        self.model = None
        self.input_padder = None
        self.cfg = None
        self.previous_smoothed_flow = None  # For temporal flow smoothing
        
        # Kalman filter parameters
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_prediction_confidence = kalman_prediction_confidence
        self.kalman_motion_model = kalman_motion_model
        self.kalman_outlier_threshold = kalman_outlier_threshold
        self.kalman_min_track_length = kalman_min_track_length
        self.kalman_filter = None
        
        # TAA compare mode state
        self.taa_compare_kalman_filters = {}
        
        # TAA processors for consistent state management
        self.taa_flow_processor = TAAProcessor(alpha=0.1)
        self.taa_simple_processor = TAAProcessor(alpha=0.1)
        
        print(f"VideoFlow Processor initialized - Device: {self.device}")
        self.device_manager.print_device_info()
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
        video_info = VideoInfo(video_path)
        return video_info.get_fps()
    
    def time_to_frame(self, time_seconds, fps):
        """Convert time in seconds to frame number"""
        return int(time_seconds * fps)
    
    def extract_frames(self, video_path, max_frames=1000, start_frame=0):
        """Extract frames from video starting at start_frame"""
        frame_extractor = FrameExtractor(video_path, fast_mode=self.fast_mode)
        return frame_extractor.extract_frames(max_frames=max_frames, start_frame=start_frame)
    
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
        Apply adaptive Kalman filter based stabilization to optical flow
        
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
            
        # Initialize Kalman filter if needed
        if self.kalman_filter is None:
            self.kalman_filter = AdaptiveOpticalFlowKalmanFilter(
                process_noise=self.kalman_process_noise,
                measurement_noise=self.kalman_measurement_noise,
                prediction_confidence=self.kalman_prediction_confidence,
                motion_model=self.kalman_motion_model,
                outlier_threshold=self.kalman_outlier_threshold,
                min_track_length=self.kalman_min_track_length
            )
            print(f"Initialized adaptive Kalman filter:")
            print(f"  Model: {self.kalman_motion_model}")
            print(f"  Process noise: {self.kalman_process_noise}")
            print(f"  Measurement noise: {self.kalman_measurement_noise}")
            print(f"  Prediction confidence: {self.kalman_prediction_confidence}")
            print(f"  Outlier threshold: {self.kalman_outlier_threshold}")
        
        # Apply Kalman filtering
        return self.kalman_filter.update(current_flow, self.kalman_filter.frame_count)
        
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
        """Encode optical flow in HSV format using encoding module"""
        return encode_flow(flow, width, height, 'hsv')
    
    def encode_gamedev_format(self, flow, width, height):
        """Encode optical flow in gamedev format using encoding module"""
        return encode_flow(flow, width, height, 'gamedev')
    
    def encode_torchvision_format(self, flow, width, height):
        """Encode optical flow using torchvision format using encoding module"""
        return encode_flow(flow, width, height, 'torchvision')
    
    def apply_taa_effect(self, current_frame, flow_pixels=None, previous_taa_frame=None, alpha=0.1, use_flow=True):
        """Apply TAA effect using effects module with persistent processors"""
        if use_flow:
            return self.taa_flow_processor.apply_taa(
                current_frame=current_frame,
                flow_pixels=flow_pixels,
                previous_taa_frame=previous_taa_frame,
                alpha=alpha,
                use_flow=True,
                sequence_id='flow_taa'
            )
        else:
            return self.taa_simple_processor.apply_taa(
                current_frame=current_frame,
                flow_pixels=None,
                previous_taa_frame=previous_taa_frame,
                alpha=alpha,
                use_flow=False,
                sequence_id='simple_taa'
            )
    
    def add_text_overlay(self, frame, text, position='top-left', font_scale=0.4, color=(255, 255, 255), thickness=1):
        """
        Add text overlay to frame
        
        Args:
            frame: Input frame (BGR format for OpenCV)
            text: Text to add
            position: 'top-left', 'top-right', 'bottom-left', 'bottom-right' or tuple (x, y)
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
        if isinstance(position, tuple):
            # Direct coordinates provided
            pos = position
        elif position == 'top-left':
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
                     save_flow=None, force_recompute=False, use_flow_cache=None, auto_play=True,
                     taa_compare=False):
        """Main processing function"""
        import os
        
        # TAA Compare Mode Setup
        if taa_compare:
            if not taa or self.flow_smoothing <= 0:
                print("Warning: --taa-compare requires --taa and --flow-smoothing to be enabled. Disabling compare mode.")
                taa_compare = False
            else:
                print("TAA Compare Mode Enabled: Generating multiple stabilization strengths.")
                self.taa_compare_kalman_filters = {
                    'Original': None,  # No stabilization (original flow)
                    'Weak': AdaptiveOpticalFlowKalmanFilter(process_noise=0.05, measurement_noise=0.2, prediction_confidence=0.5),
                    'Medium': AdaptiveOpticalFlowKalmanFilter(process_noise=0.01, measurement_noise=0.1, prediction_confidence=0.7),
                    'Strong': AdaptiveOpticalFlowKalmanFilter(process_noise=0.001, measurement_noise=0.05, prediction_confidence=0.9)
                }
        
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
        frame_extractor = FrameExtractor(input_path, fast_mode=self.fast_mode)
        frames, fps, width, height, actual_start = frame_extractor.extract_frames(
            max_frames=max_frames, 
            start_frame=start_frame,
            start_time=start_time,
            duration=duration
        )
        
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
        
        if taa_compare:
            # For 6 videos (orig, flow, 4x TAA), use 2x3 grid (2 cols, 3 rows) for more square aspect
            grid_cols = 2
            grid_rows = 3
            canvas_w = grid_cols * width
            canvas_h = int(canvas_w / (4/3))  # Target 4:3 aspect ratio instead of 16:9
            output_size = (canvas_w, canvas_h)
        elif flow_only:
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
        # Reset TAA processors for new video
        self.taa_flow_processor.reset_history()
        self.taa_simple_processor.reset_history()
        previous_flow = None  # Store previous frame's optical flow for TAA
        
        # TAA Compare state
        taa_compare_frames = {name: None for name in self.taa_compare_kalman_filters.keys()}

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
                    # TAA compare mode generates multiple versions
                    if taa_compare:
                        for name, kalman_filter in self.taa_compare_kalman_filters.items():
                            if name == 'Original':
                                # Use original unstabilized flow
                                stabilized_flow = raw_flow
                            else:
                                # Stabilize flow with this filter's strength
                                stabilized_flow = kalman_filter.update(raw_flow, kalman_filter.frame_count)
                            
                            # Apply TAA
                            prev_taa = taa_compare_frames[name]
                            taa_result = self.apply_taa_effect(frames[i], stabilized_flow, prev_taa, alpha=0.1, use_flow=True)
                            taa_compare_frames[name] = taa_result.copy()
                    
                    # Normal TAA processing (for backward compatibility)
                    taa_result = self.apply_taa_effect(frames[i], flow, None, alpha=0.1, use_flow=True)
                    taa_frame = taa_result
                else:
                    # First frame or no previous flow - apply TAA without flow
                    taa_result = self.apply_taa_effect(frames[i], None, None, alpha=0.1, use_flow=True)
                    taa_frame = taa_result
                    if taa_compare:
                        for name in self.taa_compare_kalman_filters.keys():
                            taa_compare_frames[name] = taa_result.copy()

                # Apply simple TAA (no flow) with alpha=0.1
                taa_simple_result = self.apply_taa_effect(frames[i], None, None, alpha=0.1, use_flow=False)
                taa_simple_frame = taa_simple_result
                
            # Store current flow for next frame's TAA
            previous_flow = flow.copy()
            
            # Create combined frame
            if taa_compare:
                frames_to_display = {
                    'Original': frames[i],
                    'Flow Viz': flow_viz
                }
                
                # Add TAA versions with parameter labels
                for name, kalman_filter in self.taa_compare_kalman_filters.items():
                    if name == 'Original':
                        label = f'TAA-{name}\n(No Stabilization)'
                    else:
                        pn = kalman_filter.base_process_noise
                        mn = kalman_filter.base_measurement_noise  
                        pc = kalman_filter.prediction_confidence
                        label = f'TAA-{name}\n(PN:{pn} MN:{mn} PC:{pc})'
                    frames_to_display[label] = taa_compare_frames[name].astype(np.uint8)
                
                combined = self.create_video_grid(frames_to_display, grid_shape=(3, 2), target_aspect=4/3)
            else:
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
        
        # Auto-play the resulting video if enabled
        if auto_play:
            self.auto_play_video(output_path)
    
    def auto_play_video(self, video_path):
        """Automatically play the resulting video using system default player"""
        import subprocess
        import platform
        import os
        
        if not os.path.exists(video_path):
            print(f"Video file not found for auto-play: {video_path}")
            return
        
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Windows: use start command
                subprocess.run(['cmd', '/c', 'start', '', video_path], check=False)
                print(f"Launching video with default Windows player: {os.path.basename(video_path)}")
            
            elif system == "darwin":  # macOS
                # macOS: use open command
                subprocess.run(['open', video_path], check=False)
                print(f"Launching video with default macOS player: {os.path.basename(video_path)}")
            
            elif system == "linux":
                # Linux: use xdg-open
                subprocess.run(['xdg-open', video_path], check=False)
                print(f"Launching video with default Linux player: {os.path.basename(video_path)}")
            
            else:
                print(f"Unknown operating system '{system}' - cannot auto-play video")
                print(f"Please manually open: {video_path}")
        
        except subprocess.SubprocessError as e:
            print(f"Error launching video player: {e}")
            print(f"Please manually open: {video_path}")
        except Exception as e:
            print(f"Unexpected error launching video: {e}")
            print(f"Please manually open: {video_path}")

    def create_video_grid(self, frames_dict, grid_shape, target_aspect=16/9):
        """
        Arrange multiple video frames into a grid with a target aspect ratio.
        
        Args:
            frames_dict: Dictionary {'label': frame} of frames to arrange.
            grid_shape: (rows, cols) for the grid layout.
            target_aspect: Target aspect ratio for the final output.
            
        Returns:
            A single frame containing the grid.
        """
        if not frames_dict:
            return None
            
        rows, cols = grid_shape
        
        # Get dimensions from the first frame
        first_frame = next(iter(frames_dict.values()))
        h, w = first_frame.shape[:2]
        
        # Calculate canvas size to match target aspect ratio
        canvas_w = cols * w
        canvas_h = int(canvas_w / target_aspect)
        
        # Create black canvas
        grid_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Calculate cell size and position
        cell_h = h
        cell_w = w
        
        y_offset = (canvas_h - rows * cell_h) // 2
        x_offset = (canvas_w - cols * cell_w) // 2
        
        frames = list(frames_dict.items())
        
        for i in range(rows * cols):
            if i >= len(frames):
                break
                
            label, frame = frames[i]
            
            row = i // cols
            col = i % cols
            
            y_start = y_offset + row * cell_h
            x_start = x_offset + col * cell_w
            
            # Convert frame to BGR format for video output (handle different input formats)
            if label == 'Flow Viz':
                # Flow visualization is in RGB format, convert to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif 'TAA-' in label:
                # TAA frames might be in float format, ensure uint8 and convert RGB to BGR
                frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
                if len(frame_uint8.shape) == 3 and frame_uint8.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_uint8
            else:
                # Original frames are typically in RGB format, convert to BGR
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
            
            # Add text label to frame with multi-line support
            labeled_frame = frame_bgr.copy()
            lines = label.split('\n')
            font_scale = 0.7  # Увеличиваем размер шрифта
            thickness = 2
            line_height = 30  # Увеличиваем высоту строки
            start_y = 25
            
            # Добавляем темную подложку для лучшей читаемости
            max_text_width = 0
            for line in lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                max_text_width = max(max_text_width, text_size[0])
            
            # Рисуем полупрозрачную темную подложку
            overlay = labeled_frame.copy()
            cv2.rectangle(overlay, (0, 0), (max_text_width + 15, len(lines) * line_height + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, labeled_frame, 0.3, 0, labeled_frame)
            
            for line_idx, line in enumerate(lines):
                y_pos = start_y + line_idx * line_height
                # Добавляем черную обводку для контраста
                cv2.putText(labeled_frame, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # Добавляем белый текст
                cv2.putText(labeled_frame, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Place frame on canvas
            if y_start + cell_h <= canvas_h and x_start + cell_w <= canvas_w:
                grid_canvas[y_start:y_start+cell_h, x_start:x_start+cell_w] = labeled_frame
                
        return grid_canvas

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
                       help='Enable Kalman filter flow stabilization (0.0=disabled, 0.1-0.9=enabled with increasing strength)')
    parser.add_argument('--kalman-process-noise', type=float, default=0.01,
                       help='Kalman filter process noise (0.001-0.1, default: 0.01)')
    parser.add_argument('--kalman-measurement-noise', type=float, default=0.1,
                       help='Kalman filter measurement noise (0.01-1.0, default: 0.1)')
    parser.add_argument('--kalman-prediction-confidence', type=float, default=0.7,
                       help='Kalman filter prediction confidence (0.1-0.9, default: 0.7)')
    parser.add_argument('--kalman-motion-model', choices=['constant_velocity', 'constant_acceleration'], default='constant_acceleration',
                       help='Kalman filter motion model (default: constant_acceleration)')
    parser.add_argument('--kalman-outlier-threshold', type=float, default=3.0,
                       help='Kalman filter outlier detection threshold in std deviations (1.0-5.0, default: 3.0)')
    parser.add_argument('--kalman-min-track-length', type=int, default=3,
                       help='Minimum track length before applying Kalman filtering (2-10, default: 3)')
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
    parser.add_argument('--no-autoplay', action='store_true',
                       help='Disable automatic video playback after processing')
    parser.add_argument('--taa-compare', action='store_true',
                       help='Enable TAA comparison mode with multiple stabilization strengths')
    
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
                                      sequence_length=args.sequence_length, flow_smoothing=0.0,
                                      kalman_process_noise=args.kalman_process_noise,
                                      kalman_measurement_noise=args.kalman_measurement_noise,
                                      kalman_prediction_confidence=args.kalman_prediction_confidence,
                                      kalman_motion_model=args.kalman_motion_model,
                                      kalman_outlier_threshold=args.kalman_outlier_threshold,
                                      kalman_min_track_length=args.kalman_min_track_length)
        
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
                                  sequence_length=args.sequence_length, flow_smoothing=args.flow_smoothing,
                                  kalman_process_noise=args.kalman_process_noise,
                                  kalman_measurement_noise=args.kalman_measurement_noise,
                                  kalman_prediction_confidence=args.kalman_prediction_confidence,
                                  kalman_motion_model=args.kalman_motion_model,
                                  kalman_outlier_threshold=args.kalman_outlier_threshold,
                                  kalman_min_track_length=args.kalman_min_track_length)
    
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
                              force_recompute=args.force_recompute, use_flow_cache=args.use_flow_cache, 
                              auto_play=not args.no_autoplay,
                              taa_compare=args.taa_compare)
        
        if not args.no_autoplay and not args.taa_compare:
            print("\n✓ VideoFlow processing completed successfully! Video should open automatically.")
        else:
            print("\n✓ VideoFlow processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 