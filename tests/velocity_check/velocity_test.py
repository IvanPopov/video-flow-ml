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

from processing import FlowProcessorFactory


class SyntheticVideoGenerator:
    """Generate synthetic video with moving ball for optical flow testing"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize video generator
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.ball_radius = 20
        self.ball_color = (255, 255, 255)  # White ball
        self.bg_color = (64, 64, 64)       # Dark gray background
        
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
        # Normalize time to [0, 1] over the video duration
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
            
            # Create frame
            frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
            
            # Draw ball
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius, self.ball_color, -1)
            
            # Add slight anti-aliasing for smoother motion
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius + 1, 
                      (self.ball_color[0]//2, self.ball_color[1]//2, self.ball_color[2]//2), 1)
            
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


class VelocityTestRunner:
    """Main test runner"""
    
    def __init__(self, test_dir: str, project_root: str):
        """
        Initialize test runner
        
        Args:
            test_dir: Directory containing test files
            project_root: Root directory of the project
        """
        self.test_dir = test_dir
        self.project_root = project_root
        self.temp_dir = os.path.join(test_dir, 'temp')
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def run_flow_processor(self, input_video: str, output_dir: str, model: str, frames: int) -> str:
        """
        Run optical flow processor
        
        Args:
            input_video: Path to input video
            output_dir: Directory for output
            model: Model name ('videoflow' or 'memflow')
            frames: Number of frames to process
            
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
            '--save-flow', 'npz'
        ]
        
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
                    cache_dir = item_path
                    break
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
    
    def run_test(self, speed: str, frames: int = 60, fps: int = 30) -> Dict[str, Any]:
        """
        Run complete velocity test
        
        Args:
            speed: Motion speed ('slow', 'medium', 'fast')
            frames: Number of frames to generate
            fps: Frames per second
            
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
        
        # Run VideoFlow
        try:
            videoflow_cache = self.run_flow_processor(video_file, self.temp_dir, 'videoflow', frames)
        except Exception as e:
            print(f"VideoFlow processing failed: {e}")
            videoflow_cache = None
        
        # Run MemFlow
        try:
            memflow_cache = self.run_flow_processor(video_file, self.temp_dir, 'memflow', frames)
        except Exception as e:
            print(f"MemFlow processing failed: {e}")
            memflow_cache = None
        
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
        
        # Analyze VideoFlow
        if videoflow_cache:
            try:
                vf_results = analyzer.analyze_flow_accuracy(videoflow_cache, 'VideoFlow')
                results['models']['videoflow'] = vf_results
            except Exception as e:
                print(f"VideoFlow analysis failed: {e}")
                results['models']['videoflow'] = {'error': str(e)}
        
        # Analyze MemFlow
        if memflow_cache:
            try:
                mf_results = analyzer.analyze_flow_accuracy(memflow_cache, 'MemFlow')
                results['models']['memflow'] = mf_results
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
        print(f"\n{'='*60}")
        print(f"VELOCITY TEST RESULTS - {results['test_info']['speed'].upper()}")
        print(f"{'='*60}")
        
        for model_name, model_results in results['models'].items():
            print(f"\n{model_name.upper()} Results:")
            print(f"{'-'*40}")
            
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
        
        print(f"\n{'='*60}")
        print(f"Test completed in {results['test_info']['test_duration']:.1f} seconds")
        print(f"{'='*60}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Velocity Check Test for Optical Flow Models')
    parser.add_argument('--speed', choices=['slow', 'medium', 'fast'], required=True,
                       help='Motion speed for test')
    parser.add_argument('--frames', type=int, default=60,
                       help='Number of frames to generate (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    
    args = parser.parse_args()
    
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(test_dir, '..', '..')
    
    # Run test
    runner = VelocityTestRunner(test_dir, project_root)
    results = runner.run_test(args.speed, args.frames, args.fps)
    
    return results


if __name__ == '__main__':
    main() 