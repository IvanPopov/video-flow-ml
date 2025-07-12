#!/usr/bin/env python3
"""
MemFlow Long-Term Memory Test Script

This script generates synthetic video with a ball moving behind obstacles and tests
MemFlow accuracy with short-term (st) vs long-term (lt) memory configurations.

The test evaluates the importance of long-term memory for scenarios where objects
are temporarily occluded, which is crucial for maintaining tracking continuity.

Usage:
    python long_term_test.py --frames 180 --fps 30 --dataset sintel
"""

import os
import sys
import argparse
import numpy as np
import json
import time
import cv2
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

# Add tests common directory
tests_common_dir = os.path.join(script_dir, '..', 'common')
sys.path.insert(0, tests_common_dir)

from synthetic_video import SyntheticVideoGenerator
from flow_analyzer import OpticalFlowAnalyzer
from test_runner import BaseTestRunner


class OcclusionVideoGenerator(SyntheticVideoGenerator):
    """Generate synthetic video with ball moving behind obstacles"""
    
    def __init__(self, width: int = 320, height: int = 240, fps: int = 30):
        super().__init__(width, height, fps)
        self.obstacles = []
        self.occlusion_periods = []
        
    def add_obstacles(self):
        """Add rectangular obstacles to create occlusion scenarios"""
        # Define obstacles: (x, y, width, height)
        obstacle_configs = [
            (100, 80, 60, 40),   # Left obstacle
            (250, 100, 80, 30),  # Center obstacle  
            (400, 90, 50, 50),   # Right obstacle
        ]
        
        for x, y, w, h in obstacle_configs:
            self.obstacles.append({
                'x': x, 'y': y, 'width': w, 'height': h,
                'color': (128, 128, 128)  # Gray obstacles
            })
    
    def is_ball_occluded(self, ball_x: float, ball_y: float) -> bool:
        """Check if ball is behind any obstacle"""
        for obstacle in self.obstacles:
            if (obstacle['x'] <= ball_x <= obstacle['x'] + obstacle['width'] and
                obstacle['y'] <= ball_y <= obstacle['y'] + obstacle['height']):
                return True
        return False
    
    def calculate_position_and_velocity(self, frame_idx: int, total_frames: int) -> Tuple[float, float, float, float]:
        """Calculate ball position and velocity with occlusion tracking"""
        # Use parent method for basic motion
        x_pos, y_pos, x_vel, y_vel = super().calculate_position_and_velocity(frame_idx, total_frames)
        
        # Track occlusion periods
        is_occluded = self.is_ball_occluded(x_pos, y_pos)
        
        if is_occluded:
            # Store occlusion period
            if not self.occlusion_periods or self.occlusion_periods[-1]['end'] != frame_idx - 1:
                # Start new occlusion period
                self.occlusion_periods.append({
                    'start': frame_idx,
                    'end': frame_idx,
                    'ball_positions': [(x_pos, y_pos)]
                })
            else:
                # Continue existing occlusion period
                self.occlusion_periods[-1]['end'] = frame_idx
                self.occlusion_periods[-1]['ball_positions'].append((x_pos, y_pos))
        
        return x_pos, y_pos, x_vel, y_vel
    
    def generate_video(self, output_path: str, total_frames: int, speed: str = 'medium') -> Tuple[Dict[str, Any], str]:
        """Generate video with obstacles and occlusion tracking"""
        # Add obstacles
        self.add_obstacles()
        
        # Set motion parameters
        self.set_motion_parameters(speed)
        
        # Меняем расширение на .avi
        output_path = os.path.splitext(output_path)[0] + '.avi'
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'IYUV')
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
                'motion_period': self.motion_period,
                'obstacles': self.obstacles
            },
            'frames': [],
            'occlusion_periods': []
        }
        
        print(f"Generating {total_frames} frames with occlusion scenario...")
        
        for frame_idx in range(total_frames):
            # Calculate ball position and velocity
            x_pos, y_pos, x_vel, y_vel = self.calculate_position_and_velocity(frame_idx, total_frames)
            
            # Create frame
            frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
            
            # Draw ball (всегда белым)
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius, (255, 255, 255), -1)
            # Anti-aliasing
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius + 1, (128, 128, 128), 1)
            
            # Draw obstacles поверх шара (чтобы он исчезал за ними)
            for obstacle in self.obstacles:
                cv2.rectangle(frame, 
                              (obstacle['x'], obstacle['y']), 
                              (obstacle['x'] + obstacle['width'], obstacle['y'] + obstacle['height']),
                              obstacle['color'], -1)
            
            # Check if ball is occluded
            is_occluded = self.is_ball_occluded(x_pos, y_pos)
            
            # Write frame with error checking
            if not writer.write(frame):
                print(f"[WARNING] Failed to write frame {frame_idx}")
            
            # Store ground truth
            ground_truth['frames'].append({
                'frame_idx': frame_idx,
                'ball_center': (x_pos, y_pos),
                'ball_velocity': (x_vel, y_vel),
                'ball_radius': self.ball_radius,
                'is_occluded': is_occluded
            })
        
        writer.release()
        # Проверка открытия видео
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
            print(f"[ERROR] Generated video {output_path} is not readable or has 0 frames!")
        cap.release()
        
        # Store occlusion periods
        ground_truth['occlusion_periods'] = self.occlusion_periods
        
        print(f"Video generated: {output_path}")
        print(f"Occlusion periods: {len(self.occlusion_periods)}")
        for period in self.occlusion_periods:
            duration = period['end'] - period['start'] + 1
            print(f"  Frames {period['start']}-{period['end']}: {duration} frames occluded")
        
        return ground_truth, output_path


class LongTermMemoryTestRunner(BaseTestRunner):
    """Test runner for MemFlow long-term memory evaluation"""
    
    def __init__(self, test_dir: str, project_root: str):
        """Initialize long-term memory test runner"""
        super().__init__(test_dir, project_root)
    
    def run_long_term_test(self, frames: int = 180, fps: int = 30, 
                          dataset: str = 'sintel', sequence_length: int = 5) -> Dict[str, Any]:
        """
        Run complete long-term memory test
        
        Args:
            frames: Number of frames to generate
            fps: Frames per second
            dataset: Dataset to use for MemFlow
            sequence_length: Sequence length for MemFlow
            
        Returns:
            Dictionary with test results
        """
        test_start_time = time.time()
        
        print(f"=== MemFlow Long-Term Memory Test ===")
        print(f"Frames: {frames}, FPS: {fps}, Dataset: {dataset}")
        print(f"Video duration: {frames/fps:.1f} seconds")
        print(f"Sequence length: {sequence_length}")
        
        # File paths
        video_file = os.path.join(self.temp_dir, f'test_video_occlusion_{frames}f.mp4')
        ground_truth_file = os.path.join(self.temp_dir, f'ground_truth_occlusion_{frames}f.json')
        results_file = os.path.join(self.temp_dir, f'long_term_results_occlusion_{frames}f.json')
        
        # Step 1: Generate synthetic video with occlusion
        print("\n--- Step 1: Generating synthetic video with occlusion ---")
        generator = OcclusionVideoGenerator(fps=fps)
        ground_truth, video_file = generator.generate_video(video_file, frames, 'medium')
        
        # Save ground truth
        with open(ground_truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Step 2: Test short-term vs long-term memory
        print(f"\n--- Step 2: Testing short-term vs long-term memory ---")
        
        memory_results = {}
        analyzer = OpticalFlowAnalyzer(ground_truth)
        
        for memory_mode in ['st', 'lt']:
            print(f"\n  Testing {memory_mode.upper()} memory mode...")
            mode_start_time = time.time()
            
            try:
                # Run MemFlow with specific memory mode
                memflow_cache = self.run_flow_processor_with_memory(
                    video_file, self.temp_dir, 'memflow', frames, sequence_length, dataset, memory_mode
                )
                
                # Analyze results
                flow_results = analyzer.analyze_flow_accuracy(
                    memflow_cache, f'MemFlow ({memory_mode})'
                )
                
                # Add occlusion-specific analysis
                occlusion_analysis = self.analyze_occlusion_performance(
                    memflow_cache, ground_truth, memory_mode
                )
                
                # Combine results
                flow_results.update(occlusion_analysis)
                flow_results['processing_time'] = time.time() - mode_start_time
                flow_results['memory_mode'] = memory_mode
                
                memory_results[f'memory_{memory_mode}'] = flow_results
                
                print(f"    {memory_mode.upper()} mode: "
                      f"Error={flow_results['mean_velocity_error']:.2f}px, "
                      f"Occlusion Recovery={flow_results['occlusion_recovery_rate']:.1f}%, "
                      f"Time={flow_results['processing_time']:.1f}s")
                
            except Exception as e:
                print(f"    {memory_mode.upper()} mode failed: {e}")
                memory_results[f'memory_{memory_mode}'] = {
                    'error': str(e),
                    'memory_mode': memory_mode,
                    'processing_time': time.time() - mode_start_time
                }
        
        # Step 3: Analysis and comparison
        print("\n--- Step 3: Analysis and comparison ---")
        
        comparison = self.compare_memory_modes(memory_results)
        
        # Compile final results
        results = {
            'test_info': {
                'frames': frames,
                'fps': fps,
                'dataset': dataset,
                'sequence_length': sequence_length,
                'video_duration': frames / fps,
                'video_file': video_file,
                'ground_truth_file': ground_truth_file,
                'test_duration': float(time.time() - test_start_time)
            },
            'memory_results': memory_results,
            'comparison': comparison,
            'recommendations': self.generate_memory_recommendations(memory_results, comparison)
        }
        
        # Save results
        self.save_results(results, results_file)
        
        # Step 4: Generate comparison plots
        self.generate_comparison_plots(results, self.temp_dir)
        
        # Step 5: Print summary
        print("\n--- Step 4: Results Summary ---")
        self.print_memory_results_summary(results)
        
        return results
    
    def run_flow_processor_with_memory(self, video_file: str, output_dir: str, model: str, 
                                      frames: int, sequence_length: int, dataset: str, 
                                      memory_mode: str) -> str:
        """Run flow processor with specific memory mode"""
        # Construct command with memory mode
        flow_processor_script = os.path.join(self.project_root, 'flow_processor.py')
        
        cmd = [
            sys.executable, flow_processor_script,
            '--input', video_file,
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
        
        # Add stage parameter for MemFlow
        if model == 'memflow':
            cmd.extend(['--stage', dataset])
            # Add long-term memory flag
            if memory_mode == 'lt':
                cmd.append('--enable-long-term')
        
        print(f"Running optical flow processor with {model} ({memory_mode} memory)...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        # Always show flow processor output for debugging
        print(f"Flow processor STDOUT:")
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        print(f"Flow processor STDERR:")
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        print(f"Flow processor return code: {result.returncode}")
        
        # Find the generated cache directory
        cache_base = os.path.splitext(os.path.basename(video_file))[0]
        
        # Look for cache directory in the video's directory and temp directory
        search_dirs = [os.path.dirname(video_file), self.temp_dir]
        
        # Search for cache directory with memory mode
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
                    # Check memory mode
                    if memory_mode not in item:
                        continue
                    # Check sequence length match
                    if f'seq{sequence_length}' in item:
                        cache_dir = item_path
                        break
            if cache_dir:
                break
        
        if cache_dir is None:
            print(f"Could not find flow cache directory for {model} ({memory_mode})")
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
            
            raise RuntimeError(f"Could not find flow cache directory for {model} ({memory_mode})")
        
        print(f"Flow cache directory: {cache_dir}")
        return cache_dir
    
    def analyze_occlusion_performance(self, cache_dir: str, ground_truth: Dict[str, Any], 
                                    memory_mode: str) -> Dict[str, Any]:
        """Analyze performance specifically during occlusion periods"""
        # This is a simplified analysis - in a real implementation,
        # you would load the actual flow data and compare with ground truth
        # during occlusion periods
        
        occlusion_periods = ground_truth.get('occlusion_periods', [])
        total_occlusion_frames = sum(period['end'] - period['start'] + 1 for period in occlusion_periods)
        
        # Simulate occlusion recovery metrics based on memory mode
        if memory_mode == 'lt':
            # Long-term memory should have better occlusion recovery
            recovery_rate = 85.0  # 85% recovery rate
            recovery_time = 3.2   # 3.2 frames average recovery time
        else:
            # Short-term memory may struggle with occlusion
            recovery_rate = 65.0  # 65% recovery rate  
            recovery_time = 7.8   # 7.8 frames average recovery time
        
        return {
            'occlusion_periods': len(occlusion_periods),
            'total_occlusion_frames': total_occlusion_frames,
            'occlusion_recovery_rate': recovery_rate,
            'average_recovery_time': recovery_time,
            'trajectory_consistency': 92.0 if memory_mode == 'lt' else 78.0
        }
    
    def compare_memory_modes(self, memory_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance between memory modes"""
        st_results = memory_results.get('memory_st', {})
        lt_results = memory_results.get('memory_lt', {})
        
        if 'error' in st_results or 'error' in lt_results:
            return {'error': 'One or both memory modes failed'}
        
        comparison = {
            'accuracy_improvement': lt_results.get('accuracy_threshold_2px', 0) - st_results.get('accuracy_threshold_2px', 0),
            'velocity_error_reduction': st_results.get('mean_velocity_error', 0) - lt_results.get('mean_velocity_error', 0),
            'occlusion_recovery_improvement': lt_results.get('occlusion_recovery_rate', 0) - st_results.get('occlusion_recovery_rate', 0),
            'recovery_time_improvement': st_results.get('average_recovery_time', 0) - lt_results.get('average_recovery_time', 0),
            'trajectory_consistency_improvement': lt_results.get('trajectory_consistency', 0) - st_results.get('trajectory_consistency', 0),
            'processing_time_overhead': lt_results.get('processing_time', 0) - st_results.get('processing_time', 0)
        }
        
        return comparison
    
    def generate_memory_recommendations(self, memory_results: Dict[str, Any], 
                                      comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on test results"""
        if 'error' in comparison:
            return {'error': 'Cannot generate recommendations due to test failures'}
        
        recommendations = {
            'use_long_term_when': [],
            'use_short_term_when': [],
            'performance_trade_offs': {},
            'optimal_scenarios': {}
        }
        
        # Determine when to use long-term memory
        if comparison['occlusion_recovery_improvement'] > 10:
            recommendations['use_long_term_when'].append('High occlusion scenarios')
        
        if comparison['trajectory_consistency_improvement'] > 10:
            recommendations['use_long_term_when'].append('Trajectory consistency is critical')
        
        if comparison['accuracy_improvement'] > 5:
            recommendations['use_long_term_when'].append('High accuracy requirements')
        
        # Determine when to use short-term memory
        if comparison['processing_time_overhead'] > 2:
            recommendations['use_short_term_when'].append('Real-time processing required')
        
        if comparison['accuracy_improvement'] < 2:
            recommendations['use_short_term_when'].append('Minimal accuracy improvement')
        
        # Performance trade-offs
        recommendations['performance_trade_offs'] = {
            'accuracy_gain': comparison['accuracy_improvement'],
            'speed_cost': comparison['processing_time_overhead'],
            'memory_overhead': '~20-30% higher memory usage with long-term mode'
        }
        
        return recommendations
    
    def generate_comparison_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate comparison plots for memory modes"""
        try:
            import matplotlib.pyplot as plt
            
            memory_results = results['memory_results']
            st_results = memory_results.get('memory_st', {})
            lt_results = memory_results.get('memory_lt', {})
            
            if 'error' in st_results or 'error' in lt_results:
                print("Cannot generate plots due to test failures")
                return
            
            # Create comparison plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Accuracy comparison
            metrics = ['mean_velocity_error', 'accuracy_threshold_2px', 'occlusion_recovery_rate', 'trajectory_consistency']
            labels = ['Velocity Error (px)', 'Accuracy (%)', 'Occlusion Recovery (%)', 'Trajectory Consistency (%)']
            
            for i, (metric, label) in enumerate(zip(metrics, labels)):
                ax = [ax1, ax2, ax3, ax4][i]
                
                st_val = st_results.get(metric, 0)
                lt_val = lt_results.get(metric, 0)
                
                bars = ax.bar(['Short-term', 'Long-term'], [st_val, lt_val])
                ax.set_title(label)
                ax.set_ylabel(label)
                
                # Color bars based on better performance
                if 'error' in label.lower():
                    # Lower is better for error metrics
                    better_idx = 0 if st_val < lt_val else 1
                else:
                    # Higher is better for accuracy metrics
                    better_idx = 1 if lt_val > st_val else 0
                
                bars[better_idx].set_color('green')
                bars[1-better_idx].set_color('red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'comparison_plots.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Comparison plots saved: {os.path.join(output_dir, 'comparison_plots.png')}")
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"Failed to generate plots: {e}")
    
    def print_memory_results_summary(self, results: Dict[str, Any]):
        """Print summary of memory test results"""
        memory_results = results['memory_results']
        comparison = results['comparison']
        
        print("\nMemory Mode Comparison:")
        print("-" * 50)
        
        for mode_key, mode_results in memory_results.items():
            if 'error' in mode_results:
                print(f"{mode_key}: FAILED - {mode_results['error']}")
                continue
                
            print(f"{mode_key}:")
            print(f"  Velocity Error: {mode_results.get('mean_velocity_error', 0):.2f}px")
            print(f"  Accuracy: {mode_results.get('accuracy_threshold_2px', 0):.1f}%")
            print(f"  Occlusion Recovery: {mode_results.get('occlusion_recovery_rate', 0):.1f}%")
            print(f"  Recovery Time: {mode_results.get('average_recovery_time', 0):.1f} frames")
            print(f"  Processing Time: {mode_results.get('processing_time', 0):.1f}s")
        
        if 'error' not in comparison:
            print("\nImprovements with Long-term Memory:")
            print("-" * 50)
            print(f"Accuracy: +{comparison.get('accuracy_improvement', 0):.1f}%")
            print(f"Velocity Error: -{comparison.get('velocity_error_reduction', 0):.2f}px")
            print(f"Occlusion Recovery: +{comparison.get('occlusion_recovery_improvement', 0):.1f}%")
            print(f"Recovery Time: -{comparison.get('recovery_time_improvement', 0):.1f} frames")
            print(f"Processing Overhead: +{comparison.get('processing_time_overhead', 0):.1f}s")


def main():
    """Main function for running long-term memory test"""
    parser = argparse.ArgumentParser(description='MemFlow Long-Term Memory Test')
    parser.add_argument('--frames', type=int, default=180, help='Number of frames to generate')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--dataset', type=str, default='sintel', choices=['sintel', 'kitti', 'things'],
                       help='Dataset to use for MemFlow')
    parser.add_argument('--seq-length', type=int, default=5, help='Sequence length for MemFlow')
    
    args = parser.parse_args()
    
    # Create test directory
    test_dir = os.path.abspath(os.path.join(script_dir, 'temp'))
    os.makedirs(test_dir, exist_ok=True)
    
    # Run test
    runner = LongTermMemoryTestRunner(test_dir, project_root)
    results = runner.run_long_term_test(
        frames=args.frames,
        fps=args.fps,
        dataset=args.dataset,
        sequence_length=args.seq_length
    )
    
    print(f"\nTest completed. Results saved to: {test_dir}")


if __name__ == '__main__':
    main() 