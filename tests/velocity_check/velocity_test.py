#!/usr/bin/env python3
"""
Velocity Check Test Script

This script generates synthetic video with a moving ball and tests optical flow accuracy
for both VideoFlow and MemFlow models by comparing calculated flow vectors against
ground truth velocities.

The ball moves in a sinusoidal pattern with acceleration and deceleration phases,
allowing us to test how well each model captures motion at different speeds.

Usage:
    python velocity_test.py --speed slow|medium|fast [--frames 60] [--fps 30]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math
import time

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

# Add tests common directory
tests_common_dir = os.path.join(script_dir, '..', 'common')
sys.path.insert(0, tests_common_dir)

from processing import FlowProcessorFactory
from synthetic_video import SyntheticVideoGenerator
from flow_analyzer import OpticalFlowAnalyzer
from test_runner import BaseTestRunner


class SyntheticVideoGenerator:
    """Generate synthetic video with moving ball for optical flow testing"""
    
    def __init__(self, width: int = 504, height: int = 216, fps: int = 30):
        """
        Initialize video generator
        
        Args:
            width: Video width in pixels (default: 504 for 21:9 aspect ratio)
            height: Video height in pixels (default: 216 for 21:9 aspect ratio)
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.ball_radius = 20
        
        # Use colors that are more compatible with natural scene training
        # Sintel: natural outdoor scenes (earthy tones work well)
        # Things: diverse objects (neutral colors work best)
        self.bg_color = (85, 95, 105)          # Neutral gray-blue background
        self.ball_color = (180, 150, 120)      # Natural brown/tan ball
        
        # Motion parameters (will be set based on speed)
        self.max_velocity = 0.0  # pixels per frame
        self.motion_period = 0.0  # frames for one complete cycle
        
    def set_motion_parameters(self, speed: str):
        """Set motion parameters based on speed setting"""
        if speed == 'slow':
            self.max_velocity = 2.0    # 2 pixels per frame
            self.motion_period = 120   # 4 seconds at 30fps
        elif speed == 'medium':
            self.max_velocity = 5.0    # 5 pixels per frame
            self.motion_period = 90    # 3 seconds at 30fps
        elif speed == 'fast':
            self.max_velocity = 10.0   # 10 pixels per frame
            self.motion_period = 60    # 2 seconds at 30fps
        elif speed == 'constant':
            # Constant high speed for temporal alignment testing
            self.max_velocity = 16.0   # 16 pixels per frame (very fast but constant)
            self.motion_period = float('inf')  # No period (constant motion)
        else:
            raise ValueError(f"Unknown speed: {speed}")
    
    def calculate_position_and_velocity(self, frame_idx: int, total_frames: int) -> Tuple[float, float, float, float]:
        """
        Calculate ball position and velocity for given frame
        
        Args:
            frame_idx: Current frame index (0-based)
            total_frames: Total number of frames
            
        Returns:
            Tuple of (x_position, y_position, x_velocity, y_velocity)
        """
        if self.max_velocity == 16.0 and self.motion_period == float('inf'):
            # Constant velocity motion for temporal alignment testing
            center_y = self.height // 2
            
            # Start from left edge, move right at constant velocity
            start_x = self.ball_radius + 10
            x_pos = start_x + self.max_velocity * frame_idx
            y_pos = center_y
            
            # Constant velocity
            x_velocity = self.max_velocity
            y_velocity = 0.0
            
            # Wrap around if ball goes off screen
            if x_pos > self.width - self.ball_radius - 10:
                cycles_completed = int((x_pos - start_x) / (self.width - 2 * self.ball_radius - 20))
                x_pos = start_x + ((x_pos - start_x) % (self.width - 2 * self.ball_radius - 20))
            
            return x_pos, y_pos, x_velocity, y_velocity
        else:
            # Original sinusoidal motion
            t = frame_idx / (total_frames - 1)
            
            # Create sinusoidal motion with multiple cycles
            cycles = total_frames / self.motion_period
            phase = 2 * math.pi * cycles * t
            
            # Position: sinusoidal motion in X, centered in Y
            center_x = self.width // 2
            center_y = self.height // 2
            
            # Motion range (keep ball within frame bounds)
            motion_range = self.width // 2 - self.ball_radius - 20
            
            x_pos = center_x + motion_range * math.sin(phase)
            y_pos = center_y  # Keep Y position fixed for simpler analysis
            
            # Velocity: derivative of position
            x_velocity = motion_range * math.cos(phase) * (2 * math.pi * cycles / (total_frames - 1))
            y_velocity = 0.0  # No Y motion
            
            return x_pos, y_pos, x_velocity, y_velocity
    
    def generate_video(self, output_path: str, total_frames: int, speed: str) -> Dict[str, Any]:
        """
        Generate synthetic video with moving ball
        
        Args:
            output_path: Path to output video file
            total_frames: Number of frames to generate
            speed: Motion speed ('slow', 'medium', 'fast')
            
        Returns:
            Dictionary with ground truth data
        """
        self.set_motion_parameters(speed)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")
        
        # Generate frames and collect ground truth data
        ground_truth = {
            'video_info': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': total_frames,
                'speed': speed,
                'max_velocity': self.max_velocity,
                'motion_period': self.motion_period
            },
            'frames': []
        }
        
        print(f"Generating {total_frames} frames with {speed} motion...")
        
        for frame_idx in range(total_frames):
            # Calculate ball position and velocity
            x_pos, y_pos, x_vel, y_vel = self.calculate_position_and_velocity(frame_idx, total_frames)
            
            # Create simple but well-colored frame
            frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
            
            # Draw ball with simple anti-aliasing
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius, self.ball_color, -1)
            
            # Light anti-aliasing border
            border_color = tuple((self.ball_color[i] + self.bg_color[i]) // 2 for i in range(3))
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius + 1, border_color, 1)
            
            # Write frame
            writer.write(frame)
            
            # Store ground truth
            ground_truth['frames'].append({
                'frame_idx': frame_idx,
                'ball_center': (x_pos, y_pos),
                'ball_velocity': (x_vel, y_vel),
                'ball_radius': self.ball_radius
            })
        
        writer.release()
        
        print(f"Video generated: {output_path}")
        print(f"Motion parameters: max_velocity={self.max_velocity:.1f} px/frame, period={self.motion_period} frames")
        
        return ground_truth


class OpticalFlowAnalyzer:
    """Analyze optical flow accuracy against ground truth"""
    
    def __init__(self, ground_truth: Dict[str, Any]):
        """
        Initialize analyzer
        
        Args:
            ground_truth: Ground truth data from video generation
        """
        self.ground_truth = ground_truth
        self.width = ground_truth['video_info']['width']
        self.height = ground_truth['video_info']['height']
        
    def create_ball_mask(self, frame_idx: int, dilation: int = 2) -> np.ndarray:
        """
        Create mask for ball pixels in given frame
        
        Args:
            frame_idx: Frame index
            dilation: Additional pixels around ball to include
            
        Returns:
            Binary mask where ball pixels are True
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        frame_data = self.ground_truth['frames'][frame_idx]
        center_x, center_y = frame_data['ball_center']
        radius = frame_data['ball_radius'] + dilation
        
        # Create circular mask
        y, x = np.ogrid[:self.height, :self.width]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        return mask_circle
    
    def load_flow_data(self, cache_dir: str, frame_idx: int) -> np.ndarray:
        """
        Load optical flow data for given frame
        
        Args:
            cache_dir: Directory containing flow cache
            frame_idx: Frame index
            
        Returns:
            Flow array [H, W, 2] with [dx, dy] per pixel
        """
        flow_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.npz")
        
        if not os.path.exists(flow_file):
            raise FileNotFoundError(f"Flow file not found: {flow_file}")
        
        data = np.load(flow_file)
        flow = data['flow']
        
        return flow
    
    def analyze_flow_accuracy(self, cache_dir: str, model_name: str) -> Dict[str, Any]:
        """
        Analyze flow accuracy for given model
        
        Args:
            cache_dir: Directory containing flow cache
            model_name: Name of the model (for reporting)
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing flow accuracy for {model_name}...")
        
        total_frames = len(self.ground_truth['frames'])
        
        # Statistics accumulators
        velocity_errors = []
        direction_errors = []
        magnitude_errors = []
        ball_pixel_counts = []
        
        # Frame-by-frame analysis
        frame_analyses = []
        
        # We can only analyze frames 0 to total_frames-2 (flow is computed between consecutive frames)
        for frame_idx in range(total_frames - 1):
            try:
                # Load flow data
                flow = self.load_flow_data(cache_dir, frame_idx)
                
                # Get ground truth for this frame
                frame_data = self.ground_truth['frames'][frame_idx]
                gt_velocity = frame_data['ball_velocity']
                
                # Create ball mask
                ball_mask = self.create_ball_mask(frame_idx)
                
                if not np.any(ball_mask):
                    print(f"Warning: No ball pixels found in frame {frame_idx}")
                    continue
                
                # Extract flow vectors from ball region
                ball_flow = flow[ball_mask]  # [N, 2] where N is number of ball pixels
                
                # Calculate statistics
                mean_flow = np.mean(ball_flow, axis=0)
                std_flow = np.std(ball_flow, axis=0)
                
                # Compare with ground truth
                velocity_error = np.linalg.norm(mean_flow - gt_velocity)
                direction_error = self.calculate_direction_error(mean_flow, gt_velocity)
                magnitude_error = abs(np.linalg.norm(mean_flow) - np.linalg.norm(gt_velocity))
                
                velocity_errors.append(velocity_error)
                direction_errors.append(direction_error)
                magnitude_errors.append(magnitude_error)
                ball_pixel_counts.append(np.sum(ball_mask))
                
                # Store frame analysis
                frame_analyses.append({
                    'frame_idx': int(frame_idx),
                    'ground_truth_velocity': [float(gt_velocity[0]), float(gt_velocity[1])],
                    'predicted_velocity': [float(mean_flow[0]), float(mean_flow[1])],
                    'velocity_error': float(velocity_error),
                    'direction_error': float(direction_error),
                    'magnitude_error': float(magnitude_error),
                    'ball_pixel_count': int(np.sum(ball_mask)),
                    'flow_std': [float(std_flow[0]), float(std_flow[1])]
                })
                
            except Exception as e:
                print(f"Error analyzing frame {frame_idx}: {e}")
                continue
        
        # Calculate overall statistics
        if velocity_errors:
            results = {
                'model_name': model_name,
                'total_frames_analyzed': int(len(velocity_errors)),
                'mean_velocity_error': float(np.mean(velocity_errors)),
                'std_velocity_error': float(np.std(velocity_errors)),
                'mean_direction_error': float(np.mean(direction_errors)),
                'std_direction_error': float(np.std(direction_errors)),
                'mean_magnitude_error': float(np.mean(magnitude_errors)),
                'std_magnitude_error': float(np.std(magnitude_errors)),
                'mean_ball_pixels': float(np.mean(ball_pixel_counts)),
                'accuracy_threshold_1px': float(np.mean(np.array(velocity_errors) < 1.0) * 100),
                'accuracy_threshold_2px': float(np.mean(np.array(velocity_errors) < 2.0) * 100),
                'accuracy_threshold_5px': float(np.mean(np.array(velocity_errors) < 5.0) * 100),
                'frame_analyses': frame_analyses
            }
        else:
            results = {
                'model_name': model_name,
                'error': 'No frames could be analyzed',
                'total_frames_analyzed': 0
            }
        
        return results
    
    def calculate_direction_error(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate direction error between predicted and ground truth velocity vectors
        
        Args:
            predicted: Predicted velocity vector [dx, dy]
            ground_truth: Ground truth velocity vector [dx, dy]
            
        Returns:
            Direction error in degrees
        """
        # Handle zero vectors
        pred_magnitude = np.linalg.norm(predicted)
        gt_magnitude = np.linalg.norm(ground_truth)
        
        if pred_magnitude < 1e-6 or gt_magnitude < 1e-6:
            return 0.0  # No meaningful direction for very small vectors
        
        # Calculate angle between vectors
        dot_product = np.dot(predicted, ground_truth)
        cos_angle = dot_product / (pred_magnitude * gt_magnitude)
        
        # Clamp to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def analyze_temporal_alignment(self, cache_dir: str, model_name: str) -> Dict[str, Any]:
        """
        Analyze temporal alignment by checking if flow vectors correspond to wrong frame
        
        This method specifically tests the hypothesis that flow vectors are offset by one frame,
        by comparing the percentage of pixels with incorrect velocities against the analytical
        percentage of pixels that should have background velocity due to object movement.
        
        Args:
            cache_dir: Directory containing flow cache
            model_name: Name of the model (for reporting)
            
        Returns:
            Dictionary with temporal alignment analysis results
        """
        print(f"\nAnalyzing temporal alignment for {model_name}...")
        
        total_frames = len(self.ground_truth['frames'])
        
        # Only analyze if we have constant motion (easier to detect temporal offset)
        speed = self.ground_truth['video_info']['speed']
        if speed != 'constant':
            return {'error': 'Temporal alignment analysis only works with constant motion'}
        
        velocity = self.ground_truth['video_info']['max_velocity']
        radius = self.ground_truth['frames'][0]['ball_radius']
        
        # Statistics for temporal alignment analysis
        frame_analyses = []
        
        # We can analyze frames 1 to total_frames-2 (need previous and current frame data)
        for frame_idx in range(1, total_frames - 1):
            try:
                # Load flow data for this frame
                flow = self.load_flow_data(cache_dir, frame_idx)
                
                # Get ball positions and velocities
                current_frame = self.ground_truth['frames'][frame_idx]
                prev_frame = self.ground_truth['frames'][frame_idx - 1]
                
                current_center = current_frame['ball_center']
                prev_center = prev_frame['ball_center']
                gt_velocity = current_frame['ball_velocity']
                
                # Create masks for current and previous ball positions
                current_mask = self.create_ball_mask(frame_idx, dilation=0)
                prev_mask = self.create_ball_mask(frame_idx - 1, dilation=0)
                
                # Calculate analytical percentage of pixels that should show background velocity
                # (pixels that were ball in previous frame but background in current frame)
                
                # Area that was ball in previous frame but not in current frame
                exposed_bg_mask = prev_mask & ~current_mask
                exposed_bg_pixels = np.sum(exposed_bg_mask)
                
                # Area that was background in previous frame but ball in current frame
                covered_bg_mask = current_mask & ~prev_mask
                covered_bg_pixels = np.sum(covered_bg_mask)
                
                # Total ball pixels in current frame
                total_ball_pixels = np.sum(current_mask)
                
                # Analytical percentage of ball pixels that should have ~zero velocity
                # (were background in previous frame, now covered by ball)
                analytical_bg_percent = (covered_bg_pixels / total_ball_pixels) * 100 if total_ball_pixels > 0 else 0
                
                # Analyze flow vectors in current ball position
                ball_flow = flow[current_mask]
                
                # Find pixels with velocities significantly different from ground truth
                velocity_magnitudes = np.linalg.norm(ball_flow, axis=1)
                gt_magnitude = np.linalg.norm(gt_velocity)
                
                # Define thresholds for "incorrect" velocities
                low_velocity_threshold = gt_magnitude * 0.3  # Less than 30% of expected velocity
                high_velocity_threshold = gt_magnitude * 1.7  # More than 170% of expected velocity
                
                # Count pixels with incorrect velocities
                low_velocity_pixels = np.sum(velocity_magnitudes < low_velocity_threshold)
                high_velocity_pixels = np.sum(velocity_magnitudes > high_velocity_threshold)
                incorrect_velocity_pixels = low_velocity_pixels + high_velocity_pixels
                
                # Calculate percentage of pixels with incorrect velocities
                measured_incorrect_percent = (incorrect_velocity_pixels / len(ball_flow)) * 100 if len(ball_flow) > 0 else 0
                
                # Store frame analysis
                frame_analysis = {
                    'frame_idx': int(frame_idx),
                    'analytical_bg_percent': float(analytical_bg_percent),
                    'measured_incorrect_percent': float(measured_incorrect_percent),
                    'total_ball_pixels': int(total_ball_pixels),
                    'exposed_bg_pixels': int(exposed_bg_pixels),
                    'covered_bg_pixels': int(covered_bg_pixels),
                    'low_velocity_pixels': int(low_velocity_pixels),
                    'high_velocity_pixels': int(high_velocity_pixels),
                    'gt_velocity_magnitude': float(gt_magnitude),
                    'mean_flow_magnitude': float(np.mean(velocity_magnitudes)),
                    'std_flow_magnitude': float(np.std(velocity_magnitudes))
                }
                
                frame_analyses.append(frame_analysis)
                
            except Exception as e:
                print(f"Error analyzing temporal alignment for frame {frame_idx}: {e}")
                continue
        
        if not frame_analyses:
            return {'error': 'No frames could be analyzed for temporal alignment'}
        
        # Calculate overall statistics
        analytical_percentages = [f['analytical_bg_percent'] for f in frame_analyses]
        measured_percentages = [f['measured_incorrect_percent'] for f in frame_analyses]
        
        # Calculate correlation between analytical and measured percentages
        correlation = 0.0
        if len(analytical_percentages) > 1:
            analytical_array = np.array(analytical_percentages)
            measured_array = np.array(measured_percentages)
            correlation_matrix = np.corrcoef(analytical_array, measured_array)
            correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        results = {
            'model_name': model_name,
            'total_frames_analyzed': len(frame_analyses),
            'mean_analytical_bg_percent': float(np.mean(analytical_percentages)),
            'mean_measured_incorrect_percent': float(np.mean(measured_percentages)),
            'correlation_coefficient': float(correlation),
            'temporal_alignment_hypothesis': bool(correlation > 0.7),  # Strong correlation suggests temporal offset
            'frame_analyses': frame_analyses
        }
        
        return results


class VelocityTestRunner(BaseTestRunner):
    """Velocity test runner"""
    
    def __init__(self, test_dir: str, project_root: str):
        """Initialize velocity test runner"""
        super().__init__(test_dir, project_root)
    
    def run_flow_processor(self, input_video: str, output_dir: str, model: str, frames: int, sequence_length: int = 5, dataset: str = 'sintel') -> str:
        """
        Run optical flow processor
        
        Args:
            input_video: Path to input video
            output_dir: Directory for output
            model: Model name ('videoflow' or 'memflow')
            frames: Number of frames to process
            sequence_length: Sequence length for processing
            dataset: Dataset for model configuration ('sintel', 'things', 'things_noise' for VideoFlow; 'sintel', 'things', 'kitti' for MemFlow)
            
        Returns:
            Path to flow cache directory
        """
        # Construct command
        flow_processor_script = os.path.join(self.project_root, 'flow_processor.py')
        
        cmd = [
            sys.executable, flow_processor_script,
            '--input', input_video,
            '--output', output_dir,
            '--flow-only',
            '--skip-lods',
            '--start-frame', '0',
            '--frames', str(frames),
            '--model', model,
            '--flow-format', 'motion-vectors-rg8',
            '--save-flow', 'npz',
            '--force-recompute',
            '--sequence-length', str(sequence_length)
        ]
        
        # Add model-specific parameters
        if model == 'videoflow':
            # Handle things_noise variant
            if dataset == 'things_noise':
                cmd.extend(['--vf-dataset', 'things'])
                cmd.extend(['--vf-variant', 'noise'])
            else:
                cmd.extend(['--vf-dataset', dataset])
        elif model == 'memflow':
            # MemFlow uses 'stage' parameter instead of 'dataset'
            # Map dataset names to stage names
            stage_map = {
                'sintel': 'sintel',
                'things': 'things', 
                'things_noise': 'things',  # Use things stage for noise variant
                'kitti': 'kitti'
            }
            stage = stage_map.get(dataset, 'sintel')
            cmd.extend(['--stage', stage])
        
        print(f"Running optical flow processor with {model}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        # Note: flow_processor may fail with Unicode error but still create cache
        if result.returncode != 0:
            print(f"Flow processor finished with non-zero return code: {result.returncode}")
            print(f"This may be due to Unicode issues in Windows console.")
            print(f"Checking if flow cache was created anyway...")
            # Don't raise error immediately, try to find cache first
        
        # Find the generated cache directory
        # The cache directory name depends on the model and video parameters
        cache_base = os.path.splitext(os.path.basename(input_video))[0]
        
        # Look for cache directory in the video's directory and temp directory
        search_dirs = [os.path.dirname(input_video), self.temp_dir]
        
        # Search for cache directory
        cache_dir = None
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for item in os.listdir(search_dir):
                item_path = os.path.join(search_dir, item)
                if os.path.isdir(item_path) and 'flow_cache' in item and model in item:
                    # Check if cache matches the input video name
                    if cache_base not in item:
                        continue
                    # For VideoFlow, check dataset match (including noise variant)
                    if model == 'videoflow':
                        if dataset == 'things_noise' and 'things' not in item:
                            continue
                        elif dataset != 'things_noise' and dataset not in item:
                            continue
                    # For MemFlow, check stage match
                    elif model == 'memflow':
                        stage_map = {
                            'sintel': 'sintel',
                            'things': 'things', 
                            'things_noise': 'things',
                            'kitti': 'kitti'
                        }
                        expected_stage = stage_map.get(dataset, 'sintel')
                        if expected_stage not in item:
                            continue
                    # Check if frames parameter matches (for more precise cache matching)
                    if f'frames{frames}' in item:
                        cache_dir = item_path
                        break
                    # Fallback: if no frames parameter found, use any matching cache
                    elif cache_dir is None:
                        cache_dir = item_path
            if cache_dir:
                break
        
        if cache_dir is None:
            print(f"Could not find flow cache directory for {model}")
            print(f"  Searched directories: {search_dirs}")
            if os.path.exists(self.temp_dir):
                print(f"  Items in temp directory:")
                for item in os.listdir(self.temp_dir):
                    print(f"    - {item}")
            
            # Show flow processor output for debugging
            if result.returncode != 0:
                print(f"Flow processor STDOUT:")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                print(f"Flow processor STDERR:")
                print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            
            raise RuntimeError(f"Could not find flow cache directory for {model}")
        
        print(f"Flow cache directory: {cache_dir}")
        return cache_dir
    
    def run_flow_processor_with_long_term(self, input_video: str, output_dir: str, model: str, frames: int, sequence_length: int = 5, dataset: str = 'sintel') -> str:
        """
        Run optical flow processor with long-term memory enabled (for MemFlow)
        
        Args:
            input_video: Path to input video
            output_dir: Directory for output  
            model: Model name ('memflow')
            frames: Number of frames to process
            sequence_length: Sequence length for processing
            dataset: Dataset for model configuration
            
        Returns:
            Path to flow cache directory
        """
        # Construct command
        flow_processor_script = os.path.join(self.project_root, 'flow_processor.py')
        
        cmd = [
            sys.executable, flow_processor_script,
            '--input', input_video,
            '--output', output_dir,
            '--flow-only',
            '--skip-lods',
            '--start-frame', '0',
            '--frames', str(frames),
            '--model', model,
            '--flow-format', 'motion-vectors-rg8',
            '--save-flow', 'npz',
            '--force-recompute',
            '--sequence-length', str(sequence_length),
            '--enable-long-term'  # Enable long-term memory for better performance
        ]
        
        # Add MemFlow stage parameter
        stage_map = {
            'sintel': 'sintel',
            'things': 'things', 
            'things_noise': 'things',
            'kitti': 'kitti'
        }
        stage = stage_map.get(dataset, 'sintel')
        cmd.extend(['--stage', stage])
        
        print(f"Running optical flow processor with {model} (long-term enabled)...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode != 0:
            print(f"Flow processor finished with non-zero return code: {result.returncode}")
            print(f"This may be due to Unicode issues in Windows console.")
            print(f"Checking if flow cache was created anyway...")
        
        # Find the generated cache directory
        cache_base = os.path.splitext(os.path.basename(input_video))[0]
        search_dirs = [os.path.dirname(input_video), self.temp_dir]
        
        # Search for cache directory
        cache_dir = None
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for item in os.listdir(search_dir):
                item_path = os.path.join(search_dir, item)
                if os.path.isdir(item_path) and 'flow_cache' in item and model in item:
                    # Check if cache matches the input video name
                    if cache_base not in item:
                        continue
                    # For MemFlow, check stage match and long-term indicator
                    expected_stage = stage_map.get(dataset, 'sintel')
                    if expected_stage not in item:
                        continue
                    # Check if frames parameter matches
                    if f'frames{frames}' in item:
                        cache_dir = item_path
                        break
                    elif cache_dir is None:
                        cache_dir = item_path
            if cache_dir:
                break
        
        if cache_dir is None:
            print(f"Could not find flow cache directory for {model}")
            print(f"  Searched directories: {search_dirs}")
            if os.path.exists(self.temp_dir):
                print(f"  Items in temp directory:")
                for item in os.listdir(self.temp_dir):
                    print(f"    - {item}")
            
            if result.returncode != 0:
                print(f"Flow processor STDOUT:")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                print(f"Flow processor STDERR:")
                print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            
            raise RuntimeError(f"Could not find flow cache directory for {model}")
        
        print(f"Flow cache directory: {cache_dir}")
        return cache_dir
    
    def run_test(self, speed: str, frames: int = 60, fps: int = 30, memflow_only: bool = False, videoflow_only: bool = False) -> Dict[str, Any]:
        """
        Run complete velocity test
        
        Args:
            speed: Motion speed ('slow', 'medium', 'fast', 'constant')
            frames: Number of frames to generate
            fps: Frames per second
            memflow_only: Test only MemFlow model
            videoflow_only: Test only VideoFlow model
            
        Returns:
            Dictionary with test results
        """
        test_start_time = time.time()
        
        print(f"=== Velocity Test: {speed.upper()} ===")
        print(f"Frames: {frames}, FPS: {fps}")
        
        # File paths
        video_file = os.path.join(self.temp_dir, f'test_video_{speed}.mp4')
        ground_truth_file = os.path.join(self.temp_dir, f'ground_truth_{speed}.json')
        results_file = os.path.join(self.temp_dir, f'results_{speed}.json')
        
        # Step 1: Generate synthetic video
        print("\n--- Step 1: Generating synthetic video ---")
        generator = SyntheticVideoGenerator(fps=fps)
        ground_truth = generator.generate_video(video_file, frames, speed)
        
        # Save ground truth
        with open(ground_truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Step 2: Run optical flow processors
        print("\n--- Step 2: Running optical flow processors ---")
        
        # Initialize cache variables
        videoflow_things_cache = None
        videoflow_things_noise_cache = None
        memflow_cache = None
        
        if not memflow_only:
            # Run VideoFlow with Things model (standard)
            try:
                videoflow_things_cache = self.run_flow_processor(video_file, self.temp_dir, 'videoflow', frames, 5, 'things')
            except Exception as e:
                print(f"VideoFlow (Things) processing failed: {e}")
                videoflow_things_cache = None
            
            # Run VideoFlow with Things + Noise model (more robust)
            try:
                videoflow_things_noise_cache = self.run_flow_processor(video_file, self.temp_dir, 'videoflow', frames, 5, 'things_noise')
            except Exception as e:
                print(f"VideoFlow (Things + Noise) processing failed: {e}")
                videoflow_things_noise_cache = None
        else:
            print("Skipping VideoFlow tests (--memflow-only flag enabled)")
        
        if not videoflow_only:
            # Run MemFlow with Things dataset configuration and long-term memory
            try:
                memflow_cache = self.run_flow_processor_with_long_term(video_file, self.temp_dir, 'memflow', frames, 5, 'things')
            except Exception as e:
                print(f"MemFlow processing failed: {e}")
                memflow_cache = None
        else:
            print("Skipping MemFlow tests (--videoflow-only flag enabled)")
        
        # Step 3: Analyze results
        print("\n--- Step 3: Analyzing results ---")
        
        analyzer = OpticalFlowAnalyzer(ground_truth)
        
        results = {
            'test_info': {
                'speed': speed,
                'frames': frames,
                'fps': fps,
                'video_file': video_file,
                'ground_truth_file': ground_truth_file,
                'test_duration': float(time.time() - test_start_time)
            },
            'models': {}
        }
        
        # Analyze VideoFlow Things (standard)
        if videoflow_things_cache:
            try:
                vf_things_results = analyzer.analyze_flow_accuracy(videoflow_things_cache, 'VideoFlow (Things)')
                results['models']['videoflow_things'] = vf_things_results
                
                # Add temporal alignment analysis for constant motion
                if speed == 'constant':
                    temporal_results = analyzer.analyze_temporal_alignment(videoflow_things_cache, 'VideoFlow (Things)')
                    results['models']['videoflow_things']['temporal_alignment'] = temporal_results
                    
            except Exception as e:
                print(f"VideoFlow (Things) analysis failed: {e}")
                results['models']['videoflow_things'] = {'error': str(e)}
        
        # Analyze VideoFlow Things + Noise
        if videoflow_things_noise_cache:
            try:
                vf_things_noise_results = analyzer.analyze_flow_accuracy(videoflow_things_noise_cache, 'VideoFlow (Things + Noise)')
                results['models']['videoflow_things_noise'] = vf_things_noise_results
                
                # Add temporal alignment analysis for constant motion
                if speed == 'constant':
                    temporal_results = analyzer.analyze_temporal_alignment(videoflow_things_noise_cache, 'VideoFlow (Things + Noise)')
                    results['models']['videoflow_things_noise']['temporal_alignment'] = temporal_results
                    
            except Exception as e:
                print(f"VideoFlow (Things + Noise) analysis failed: {e}")
                results['models']['videoflow_things_noise'] = {'error': str(e)}
        
        # Analyze MemFlow with Things dataset
        if memflow_cache:
            try:
                mf_results = analyzer.analyze_flow_accuracy(memflow_cache, 'MemFlow (Things)')
                results['models']['memflow'] = mf_results
                
                # Add temporal alignment analysis for constant motion
                if speed == 'constant':
                    temporal_results = analyzer.analyze_temporal_alignment(memflow_cache, 'MemFlow (Things)')
                    results['models']['memflow']['temporal_alignment'] = temporal_results
                    
            except Exception as e:
                print(f"MemFlow analysis failed: {e}")
                results['models']['memflow'] = {'error': str(e)}
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Step 4: Print summary
        print("\n--- Step 4: Results Summary ---")
        self.print_results_summary(results)
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print formatted results summary"""
        print(f"\n{'='*80}")
        print(f"VELOCITY TEST RESULTS - {results['test_info']['speed'].upper()}")
        print(f"{'='*80}")
        
        # Define model display order and names
        model_order = ['videoflow_things', 'videoflow_things_noise', 'memflow']
        model_display_names = {
            'videoflow_things': 'VIDEOFLOW (THINGS)',
            'videoflow_things_noise': 'VIDEOFLOW (THINGS + NOISE)',
            'memflow': 'MEMFLOW (THINGS)'
        }
        
        for model_key in model_order:
            if model_key not in results['models']:
                continue
                
            model_results = results['models'][model_key]
            display_name = model_display_names[model_key]
            
            print(f"\n{display_name} Results:")
            print(f"{'-'*50}")
            
            if 'error' in model_results:
                print(f"ERROR: {model_results['error']}")
                continue
            
            print(f"Frames analyzed: {model_results['total_frames_analyzed']}")
            print(f"Mean velocity error: {model_results['mean_velocity_error']:.2f} px/frame")
            print(f"Mean direction error: {model_results['mean_direction_error']:.1f}°")
            print(f"Mean magnitude error: {model_results['mean_magnitude_error']:.2f} px/frame")
            print(f"Accuracy (< 1px): {model_results['accuracy_threshold_1px']:.1f}%")
            print(f"Accuracy (< 2px): {model_results['accuracy_threshold_2px']:.1f}%")
            print(f"Accuracy (< 5px): {model_results['accuracy_threshold_5px']:.1f}%")
            
            # Print temporal alignment analysis if available
            if 'temporal_alignment' in model_results and 'error' not in model_results['temporal_alignment']:
                temporal = model_results['temporal_alignment']
                print(f"\nTemporal Alignment Analysis:")
                print(f"  Analytical BG%: {temporal['mean_analytical_bg_percent']:.1f}%")
                print(f"  Measured incorrect%: {temporal['mean_measured_incorrect_percent']:.1f}%")
                print(f"  Correlation: {temporal['correlation_coefficient']:.3f}")
                if temporal['temporal_alignment_hypothesis']:
                    print(f"  *** TEMPORAL OFFSET DETECTED *** (correlation > 0.7)")
                else:
                    print(f"  Temporal alignment appears correct")
        
        print(f"\n{'='*80}")
        print(f"Test completed in {results['test_info']['test_duration']:.1f} seconds")
        print(f"{'='*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Velocity Check Test for Optical Flow Models')
    parser.add_argument('--speed', choices=['slow', 'medium', 'fast', 'constant'], required=True,
                       help='Motion speed for test (constant = 8px/frame for temporal alignment testing)')
    parser.add_argument('--frames', type=int, default=60,
                       help='Number of frames to generate (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--memflow-only', action='store_true',
                       help='Test only MemFlow model (skip VideoFlow for faster testing)')
    parser.add_argument('--videoflow-only', action='store_true',
                       help='Test only VideoFlow model (skip MemFlow for faster testing)')
    
    args = parser.parse_args()
    
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(test_dir, '..', '..')
    
    # Run test
    runner = VelocityTestRunner(test_dir, project_root)
    results = runner.run_test(args.speed, args.frames, args.fps, args.memflow_only, args.videoflow_only)
    
    return results


if __name__ == '__main__':
    main() 