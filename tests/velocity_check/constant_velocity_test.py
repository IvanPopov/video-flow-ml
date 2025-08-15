#!/usr/bin/env python3
"""
Constant Velocity Test Script

This script generates synthetic video with a ball moving at constant velocity
to test temporal alignment between optical flow and ground truth.

The ball moves linearly at constant speed, making it easier to detect
temporal offsets in optical flow analysis.

Usage:
    python constant_velocity_test.py [--frames 90] [--fps 30]
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
from test_runner import BaseTestRunner


class ConstantVelocityVideoGenerator:
    """Generate synthetic video with constant velocity motion for temporal alignment testing"""
    
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
        self.ball_radius = 15
        
        # Colors compatible with Things dataset
        self.bg_color = (85, 95, 105)          # Neutral gray-blue background
        self.ball_color = (180, 150, 120)      # Natural brown/tan ball
        
        # Constant velocity parameters (similar to middle of fast test)
        self.velocity_x = 6.0    # pixels per frame (constant horizontal motion)
        self.velocity_y = 0.0    # no vertical motion for simplicity
        
    def calculate_position(self, frame_idx: int) -> Tuple[float, float]:
        """
        Calculate ball position for given frame with constant velocity
        
        Args:
            frame_idx: Current frame index (0-based)
            
        Returns:
            Tuple of (x_position, y_position)
        """
        # Start position (left side, centered vertically)
        start_x = self.ball_radius + 20
        center_y = self.height // 2
        
        # Constant linear motion
        x_pos = start_x + self.velocity_x * frame_idx
        y_pos = center_y + self.velocity_y * frame_idx
        
        return x_pos, y_pos
    
    def generate_video(self, output_path: str, total_frames: int) -> Dict[str, Any]:
        """
        Generate synthetic video with constant velocity motion
        
        Args:
            output_path: Path to output video file
            total_frames: Number of frames to generate
            
        Returns:
            Dictionary with ground truth data
        """
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
                'motion_type': 'constant_velocity',
                'velocity_x': self.velocity_x,
                'velocity_y': self.velocity_y
            },
            'frames': []
        }
        
        print(f"Generating {total_frames} frames with constant velocity motion...")
        print(f"Velocity: ({self.velocity_x:.1f}, {self.velocity_y:.1f}) px/frame")
        
        for frame_idx in range(total_frames):
            # Calculate ball position
            x_pos, y_pos = self.calculate_position(frame_idx)
            
            # Skip frame if ball moves out of bounds
            if (x_pos - self.ball_radius < 0 or 
                x_pos + self.ball_radius >= self.width or
                y_pos - self.ball_radius < 0 or 
                y_pos + self.ball_radius >= self.height):
                print(f"Ball out of bounds at frame {frame_idx}, stopping generation")
                break
            
            # Create frame
            frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
            
            # Draw ball with anti-aliasing
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius, self.ball_color, -1)
            
            # Light anti-aliasing border
            border_color = tuple((self.ball_color[i] + self.bg_color[i]) // 2 for i in range(3))
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius + 1, border_color, 1)
            
            # Write frame
            writer.write(frame)
            
            # Store ground truth (velocity is constant for all frames)
            ground_truth['frames'].append({
                'frame_idx': frame_idx,
                'ball_center': (x_pos, y_pos),
                'ball_velocity': (self.velocity_x, self.velocity_y),
                'ball_radius': self.ball_radius
            })
        
        writer.release()
        
        actual_frames = len(ground_truth['frames'])
        ground_truth['video_info']['actual_frames'] = actual_frames
        
        print(f"Video generated: {output_path}")
        print(f"Generated {actual_frames} frames with constant motion")
        print(f"Constant velocity: ({self.velocity_x:.1f}, {self.velocity_y:.1f}) px/frame")
        
        return ground_truth


class ConstantVelocityTestRunner(BaseTestRunner):
    """Constant velocity test runner for temporal alignment testing"""
    
    def __init__(self, test_dir: str, project_root: str):
        """Initialize constant velocity test runner"""
        super().__init__(test_dir, project_root)
    
    def run_flow_processor_videoflow_only(self, input_video: str, output_dir: str, frames: int) -> str:
        """
        Run VideoFlow flow processor only (for initial testing)
        
        Args:
            input_video: Path to input video
            output_dir: Directory for output
            frames: Number of frames to process
            
        Returns:
            Path to flow cache directory
        """
        # Construct command for VideoFlow with Things dataset
        flow_processor_script = os.path.join(self.project_root, 'flow_processor.py')
        
        cmd = [
            sys.executable, flow_processor_script,
            '--input', input_video,
            '--output', output_dir,
            '--flow-only',
            '--skip-lods',
            '--start-frame', '0',
            '--frames', str(frames),
            '--model', 'videoflow',
            '--vf-dataset', 'things',
            '--flow-format', 'motion-vectors-rg8',
            '--save-flow', 'npz',
            '--force-recompute',
            '--sequence-length', '5'
        ]
        
        print(f"Running VideoFlow optical flow processor...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode != 0:
            print(f"Flow processor finished with return code: {result.returncode}")
            print(f"STDOUT:")
            print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            print(f"STDERR:")
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        
        # Find the generated cache directory
        cache_base = os.path.splitext(os.path.basename(input_video))[0]
        search_dirs = [os.path.dirname(input_video), output_dir]
        
        # Search for VideoFlow cache directory
        cache_dir = None
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for item in os.listdir(search_dir):
                item_path = os.path.join(search_dir, item)
                if (os.path.isdir(item_path) and 
                    'flow_cache' in item and 
                    'videoflow' in item and 
                    'things' in item and 
                    cache_base in item):
                    cache_dir = item_path
                    break
            if cache_dir:
                break
        
        if cache_dir is None:
            print(f"Could not find VideoFlow cache directory")
            print(f"Searched directories: {search_dirs}")
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    print(f"Items in {search_dir}:")
                    for item in os.listdir(search_dir):
                        print(f"  - {item}")
            raise RuntimeError(f"Could not find VideoFlow cache directory")
        
        print(f"VideoFlow cache directory: {cache_dir}")
        return cache_dir
    
    def analyze_temporal_alignment(self, cache_dir: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal alignment between flow vectors and ground truth
        
        Args:
            cache_dir: Directory containing flow cache
            ground_truth: Ground truth data
            
        Returns:
            Analysis results with frame-by-frame comparison
        """
        print(f"\nAnalyzing temporal alignment...")
        
        total_frames = len(ground_truth['frames'])
        expected_velocity = (ground_truth['video_info']['velocity_x'], 
                           ground_truth['video_info']['velocity_y'])
        
        frame_analyses = []
        
        # Analyze each flow frame (frames 0 to total_frames-2)
        for flow_idx in range(total_frames - 1):
            flow_file = os.path.join(cache_dir, f"flow_frame_{flow_idx:06d}.npz")
            
            if not os.path.exists(flow_file):
                print(f"Warning: Flow file not found: {flow_file}")
                continue
            
            try:
                # Load flow data
                data = np.load(flow_file)
                flow = data['flow']  # [H, W, 2]
                
                # For each ground truth frame, calculate what flow should correspond to
                for gt_frame_idx in [flow_idx, flow_idx + 1]:
                    if gt_frame_idx >= total_frames:
                        continue
                    
                    gt_frame = ground_truth['frames'][gt_frame_idx]
                    ball_center = gt_frame['ball_center']
                    
                    # Create mask for ball region
                    y, x = np.ogrid[:ground_truth['video_info']['height'], 
                                  :ground_truth['video_info']['width']]
                    mask = ((x - ball_center[0])**2 + (y - ball_center[1])**2 <= 
                           (gt_frame['ball_radius'] + 2)**2)
                    
                    if not np.any(mask):
                        continue
                    
                    # Extract flow from ball region
                    ball_flow = flow[mask]
                    mean_flow = np.mean(ball_flow, axis=0)
                    
                    # Calculate alignment score
                    velocity_error = np.linalg.norm(mean_flow - expected_velocity)
                    
                    frame_analyses.append({
                        'flow_frame_idx': int(flow_idx),
                        'gt_frame_idx': int(gt_frame_idx),
                        'ball_center': [float(ball_center[0]), float(ball_center[1])],
                        'expected_velocity': [float(expected_velocity[0]), float(expected_velocity[1])],
                        'measured_velocity': [float(mean_flow[0]), float(mean_flow[1])],
                        'velocity_error': float(velocity_error),
                        'ball_pixels': int(np.sum(mask))
                    })
            
            except Exception as e:
                print(f"Error analyzing flow frame {flow_idx}: {e}")
                continue
        
        # Find best alignment
        best_alignment = None
        best_error = float('inf')
        
        # Group by alignment hypothesis (flow_idx vs gt_frame_idx relationship)
        alignment_hypotheses = {}
        for analysis in frame_analyses:
            flow_idx = analysis['flow_frame_idx']
            gt_idx = analysis['gt_frame_idx']
            offset = gt_idx - flow_idx  # 0 means flow_N corresponds to gt_frame_N, 1 means flow_N -> gt_frame_N+1
            
            if offset not in alignment_hypotheses:
                alignment_hypotheses[offset] = []
            alignment_hypotheses[offset].append(analysis)
        
        # Calculate average error for each hypothesis
        hypothesis_results = {}
        for offset, analyses in alignment_hypotheses.items():
            errors = [a['velocity_error'] for a in analyses]
            avg_error = np.mean(errors)
            hypothesis_results[offset] = {
                'offset': offset,
                'average_error': float(avg_error),
                'sample_count': len(analyses),
                'description': f'flow_frame_N -> gt_frame_{offset}N' if offset == 0 else f'flow_frame_N -> gt_frame_N{offset:+d}'
            }
            
            if avg_error < best_error:
                best_error = avg_error
                best_alignment = offset
        
        results = {
            'expected_velocity': expected_velocity,
            'total_flow_frames': total_frames - 1,
            'alignment_hypotheses': hypothesis_results,
            'best_alignment_offset': best_alignment,
            'best_alignment_error': float(best_error),
            'frame_analyses': frame_analyses
        }
        
        print(f"Temporal alignment analysis:")
        print(f"  Expected velocity: ({expected_velocity[0]:.1f}, {expected_velocity[1]:.1f}) px/frame")
        print(f"  Best alignment: offset = {best_alignment} (error = {best_error:.2f} px/frame)")
        
        for offset, result in hypothesis_results.items():
            print(f"  Hypothesis {result['description']}: {result['average_error']:.2f} px/frame ({result['sample_count']} samples)")
        
        return results
    
    def run_test(self, frames: int = 90, fps: int = 30) -> Dict[str, Any]:
        """
        Run complete constant velocity test
        
        Args:
            frames: Number of frames to generate
            fps: Frames per second
            
        Returns:
            Dictionary with test results
        """
        test_start_time = time.time()
        
        print(f"=== CONSTANT VELOCITY TEST ===")
        print(f"Frames: {frames}, FPS: {fps}")
        
        # File paths
        video_file = os.path.join(self.temp_dir, f'test_constant_velocity.mp4')
        ground_truth_file = os.path.join(self.temp_dir, f'ground_truth_constant.json')
        results_file = os.path.join(self.temp_dir, f'results_constant.json')
        
        # Step 1: Generate synthetic video with constant velocity
        print("\n--- Step 1: Generating constant velocity video ---")
        generator = ConstantVelocityVideoGenerator(fps=fps)
        ground_truth = generator.generate_video(video_file, frames)
        
        # Save ground truth
        with open(ground_truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        actual_frames = ground_truth['video_info']['actual_frames']
        print(f"Generated video with {actual_frames} frames")
        
        # Step 2: Run VideoFlow flow processor
        print("\n--- Step 2: Running VideoFlow optical flow processor ---")
        try:
            videoflow_cache = self.run_flow_processor_videoflow_only(video_file, self.temp_dir, actual_frames)
        except Exception as e:
            print(f"VideoFlow processing failed: {e}")
            return {'error': f'VideoFlow processing failed: {e}'}
        
        # Step 3: Analyze temporal alignment
        print("\n--- Step 3: Analyzing temporal alignment ---")
        try:
            alignment_results = self.analyze_temporal_alignment(videoflow_cache, ground_truth)
        except Exception as e:
            print(f"Temporal alignment analysis failed: {e}")
            return {'error': f'Analysis failed: {e}'}
        
        # Step 4: Compile results
        results = {
            'test_info': {
                'test_type': 'constant_velocity',
                'frames': actual_frames,
                'fps': fps,
                'video_file': video_file,
                'ground_truth_file': ground_truth_file,
                'videoflow_cache': videoflow_cache,
                'test_duration': float(time.time() - test_start_time)
            },
            'ground_truth': ground_truth['video_info'],
            'temporal_alignment': alignment_results
        }
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Step 5: Print summary
        print("\n--- Step 4: Results Summary ---")
        self.print_results_summary(results)
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print formatted results summary"""
        print(f"\n{'='*80}")
        print(f"CONSTANT VELOCITY TEST RESULTS")
        print(f"{'='*80}")
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            return
        
        gt = results['ground_truth']
        alignment = results['temporal_alignment']
        
        print(f"Video info:")
        print(f"  Frames: {gt['actual_frames']}")
        print(f"  Constant velocity: ({gt['velocity_x']:.1f}, {gt['velocity_y']:.1f}) px/frame")
        print(f"  Motion type: {gt['motion_type']}")
        
        print(f"\nTemporal alignment analysis:")
        print(f"  Best alignment offset: {alignment['best_alignment_offset']}")
        print(f"  Best alignment error: {alignment['best_alignment_error']:.2f} px/frame")
        
        print(f"\nAlignment hypotheses:")
        for offset, hyp in alignment['alignment_hypotheses'].items():
            marker = " <-- BEST" if offset == alignment['best_alignment_offset'] else ""
            print(f"  {hyp['description']}: {hyp['average_error']:.2f} px/frame ({hyp['sample_count']} samples){marker}")
        
        print(f"\n{'='*80}")
        print(f"Test completed in {results['test_info']['test_duration']:.1f} seconds")
        
        # Provide interpretation
        best_offset = alignment['best_alignment_offset']
        if best_offset == 0:
            print(f"✓ GOOD: Flow vectors appear aligned with current frame")
        elif best_offset == 1:
            print(f"⚠ ISSUE: Flow vectors appear to correspond to NEXT frame (temporal offset +1)")
        elif best_offset == -1:
            print(f"⚠ ISSUE: Flow vectors appear to correspond to PREVIOUS frame (temporal offset -1)")
        else:
            print(f"⚠ UNUSUAL: Flow vectors have offset of {best_offset} frames")
        
        print(f"{'='*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Constant Velocity Test for Temporal Alignment')
    parser.add_argument('--frames', type=int, default=90,
                       help='Number of frames to generate (default: 90)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    
    args = parser.parse_args()
    
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(test_dir, '..', '..')
    
    # Run test
    runner = ConstantVelocityTestRunner(test_dir, project_root)
    results = runner.run_test(args.frames, args.fps)
    
    return results


if __name__ == '__main__':
    main()
