#!/usr/bin/env python3
"""
MemFlow Sequence Length Test Script

This script generates synthetic video with a moving ball and tests MemFlow accuracy
with different sequence lengths to find the optimal configuration.

The test evaluates specific sequence lengths: 3, 5, 10, 15, 25, 50 frames and compares:
- Accuracy metrics (velocity error, direction error)
- Processing speed
- Memory usage patterns

Usage:
    python seq_length_test.py --motion medium --frames 120 [--seq-lengths 3,5,10,15,25,50]
"""

import os
import sys
import argparse
import numpy as np
import json
import time
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


class SequenceLengthTestRunner(BaseTestRunner):
    """Test runner for MemFlow sequence length optimization"""
    
    def __init__(self, test_dir: str, project_root: str):
        """Initialize sequence length test runner"""
        super().__init__(test_dir, project_root)
    
    def run_sequence_length_test(self, motion: str, frames: int = 120, fps: int = 30,
                                seq_lengths: List[int] = None) -> Dict[str, Any]:
        """
        Run complete sequence length test
        
        Args:
            motion: Motion type ('slow', 'medium', 'fast')
            frames: Number of frames to generate  
            fps: Frames per second
            seq_lengths: List of sequence lengths to test
            
        Returns:
            Dictionary with test results
        """
        if seq_lengths is None:
            seq_lengths = [3, 5, 10, 15, 25, 50]
            
        test_start_time = time.time()
        
        print(f"=== MemFlow Sequence Length Test ===")
        print(f"Motion: {motion}, Frames: {frames}, FPS: {fps}")
        print(f"Video duration: {frames/fps:.1f} seconds")
        print(f"Testing sequence lengths: {seq_lengths}")
        
        # File paths
        video_file = os.path.join(self.temp_dir, f'test_video_{motion}_{frames}f.mp4')
        ground_truth_file = os.path.join(self.temp_dir, f'ground_truth_{motion}_{frames}f.json')
        results_file = os.path.join(self.temp_dir, f'seq_length_results_{motion}_{frames}f.json')
        
        # Step 1: Generate synthetic video
        print("\n--- Step 1: Generating synthetic video ---")
        generator = SyntheticVideoGenerator(fps=fps)
        ground_truth = generator.generate_video(video_file, frames, motion)
        
        # Save ground truth
        with open(ground_truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Step 2: Test different sequence lengths
        print(f"\n--- Step 2: Testing sequence lengths {seq_lengths} ---")
        
        sequence_results = {}
        analyzer = OpticalFlowAnalyzer(ground_truth)
        
        for seq_len in seq_lengths:
            print(f"\n  Testing sequence length {seq_len}...")
            seq_start_time = time.time()
            
            # Validate sequence length
            if seq_len >= frames:
                print(f"    Sequence length {seq_len} too large for {frames} frames, skipping")
                sequence_results[f'seq_len_{seq_len}'] = {
                    'error': f'Sequence length {seq_len} >= total frames {frames}',
                    'sequence_length': seq_len,
                    'processing_time': 0
                }
                continue
            
            try:
                # Run MemFlow with specific sequence length
                memflow_cache = self.run_flow_processor(
                    video_file, self.temp_dir, 'memflow', frames, seq_len, 'sintel'
                )
                
                # Analyze results
                flow_results = analyzer.analyze_flow_accuracy(
                    memflow_cache, f'MemFlow (seq_len={seq_len})'
                )
                
                # Add timing information
                flow_results['processing_time'] = time.time() - seq_start_time
                flow_results['sequence_length'] = seq_len
                
                sequence_results[f'seq_len_{seq_len}'] = flow_results
                
                print(f"    Sequence length {seq_len}: "
                      f"Error={flow_results['mean_velocity_error']:.2f}px, "
                      f"Direction={flow_results['mean_direction_error']:.1f}°, "
                      f"Accuracy={flow_results['accuracy_threshold_2px']:.1f}%, "
                      f"Time={flow_results['processing_time']:.1f}s")
                
            except Exception as e:
                print(f"    Sequence length {seq_len} failed: {e}")
                sequence_results[f'seq_len_{seq_len}'] = {
                    'error': str(e),
                    'sequence_length': seq_len,
                    'processing_time': time.time() - seq_start_time
                }
        
        # Step 3: Analysis and recommendations
        print("\n--- Step 3: Analysis and recommendations ---")
        
        # Find best performing configurations
        best_accuracy = self.find_best_configuration(sequence_results, 'accuracy_threshold_2px')
        best_speed = self.find_best_configuration(sequence_results, 'processing_time', minimize=True)
        best_overall = self.find_best_overall_configuration(sequence_results)
        
        # Compile final results
        results = {
            'test_info': {
                'motion': motion,
                'frames': frames,
                'fps': fps,
                'video_duration': frames / fps,
                'sequence_lengths': seq_lengths,
                'video_file': video_file,
                'ground_truth_file': ground_truth_file,
                'test_duration': float(time.time() - test_start_time)
            },
            'sequence_results': sequence_results,
            'recommendations': {
                'best_accuracy': best_accuracy,
                'best_speed': best_speed,
                'best_overall': best_overall,
                'summary': self.generate_recommendations_summary(sequence_results)
            }
        }
        
        # Save results
        self.save_results(results, results_file)
        
        # Step 4: Print summary
        print("\n--- Step 4: Results Summary ---")
        self.print_sequence_results_summary(results)
        
        return results
    
    def find_best_configuration(self, sequence_results: Dict[str, Any], metric: str, 
                               minimize: bool = False) -> Dict[str, Any]:
        """Find best performing configuration for a specific metric"""
        best_seq = None
        best_value = float('inf') if minimize else float('-inf')
        
        for seq_key, results in sequence_results.items():
            if 'error' in results:
                continue
            
            if metric not in results:
                continue
            
            value = results[metric]
            if minimize:
                if value < best_value:
                    best_value = value
                    best_seq = seq_key
            else:
                if value > best_value:
                    best_value = value
                    best_seq = seq_key
        
        if best_seq is None:
            return {'error': f'No valid results found for metric {metric}'}
        
        return {
            'sequence_length': sequence_results[best_seq]['sequence_length'],
            'metric': metric,
            'value': best_value,
            'results': sequence_results[best_seq]
        }
    
    def find_best_overall_configuration(self, sequence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find best overall configuration balancing accuracy and speed"""
        best_seq = None
        best_score = float('-inf')
        
        for seq_key, results in sequence_results.items():
            if 'error' in results:
                continue
            
            # Calculate composite score: accuracy (70%) + speed bonus (30%)
            if all(key in results for key in ['accuracy_threshold_2px', 'processing_time']):
                accuracy_score = results['accuracy_threshold_2px'] / 100.0  # Normalize to 0-1
                speed_score = max(0, 1.0 - (results['processing_time'] / 60.0))  # Penalty for >60s
                
                composite_score = 0.7 * accuracy_score + 0.3 * speed_score
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_seq = seq_key
        
        if best_seq is None:
            return {'error': 'No valid results found for overall scoring'}
        
        return {
            'sequence_length': sequence_results[best_seq]['sequence_length'],
            'composite_score': best_score,
            'results': sequence_results[best_seq]
        }
    
    def generate_recommendations_summary(self, sequence_results: Dict[str, Any]) -> str:
        """Generate human-readable recommendations summary"""
        valid_results = [(k, v) for k, v in sequence_results.items() if 'error' not in v]
        
        if not valid_results:
            return "No valid results to analyze"
        
        # Sort by accuracy
        by_accuracy = sorted(valid_results, 
                           key=lambda x: x[1].get('accuracy_threshold_2px', 0), 
                           reverse=True)
        
        # Sort by speed  
        by_speed = sorted(valid_results,
                         key=lambda x: x[1].get('processing_time', float('inf')))
        
        summary = []
        summary.append(f"Tested {len(valid_results)} sequence lengths successfully.")
        
        if by_accuracy:
            best_acc = by_accuracy[0]
            summary.append(f"Best accuracy: seq_len={best_acc[1]['sequence_length']} "
                          f"({best_acc[1]['accuracy_threshold_2px']:.1f}% accuracy)")
        
        if by_speed:
            best_speed = by_speed[0]
            summary.append(f"Fastest processing: seq_len={best_speed[1]['sequence_length']} "
                          f"({best_speed[1]['processing_time']:.1f}s)")
        
        # Look for patterns
        if len(valid_results) >= 3:
            seq_lengths = [v['sequence_length'] for k, v in valid_results]
            accuracies = [v.get('accuracy_threshold_2px', 0) for k, v in valid_results]
            
            if len(set(seq_lengths)) == len(seq_lengths):  # No duplicates
                # Check for trends
                if accuracies == sorted(accuracies):
                    summary.append("Accuracy improves with longer sequences.")
                elif accuracies == sorted(accuracies, reverse=True):
                    summary.append("Accuracy decreases with longer sequences.")
                else:
                    summary.append("Accuracy pattern is not monotonic.")
        
        return " ".join(summary)
    
    def print_sequence_results_summary(self, results: Dict[str, Any]):
        """Print formatted sequence length results summary"""
        print(f"\n{'='*80}")
        print(f"MEMFLOW SEQUENCE LENGTH TEST RESULTS")
        print(f"{'='*80}")
        
        test_info = results['test_info']
        print(f"Motion: {test_info['motion']}")
        print(f"Frames: {test_info['frames']}")
        print(f"Video duration: {test_info['video_duration']:.1f} seconds")
        print(f"Test duration: {test_info['test_duration']:.1f} seconds")
        
        # Print sequence results table
        print(f"\n{'Seq Len':<8} {'Accuracy':<10} {'Direction':<10} {'Speed':<8} {'Status'}")
        print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
        
        sequence_results = results['sequence_results']
        for seq_key in sorted(sequence_results.keys(), key=lambda x: int(x.split('_')[-1])):
            result = sequence_results[seq_key]
            seq_len = result['sequence_length']
            
            if 'error' in result:
                print(f"{seq_len:<8} {'ERROR':<10} {'-':<10} {'-':<8} Failed")
            else:
                accuracy = result.get('accuracy_threshold_2px', 0)
                direction = result.get('mean_direction_error', 0)
                speed = result.get('processing_time', 0)
                
                print(f"{seq_len:<8} {accuracy:<10.1f} {direction:<10.1f} {speed:<8.1f} OK")
        
        # Print recommendations
        recommendations = results['recommendations']
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        if 'error' not in recommendations['best_accuracy']:
            best_acc = recommendations['best_accuracy']
            print(f"Best Accuracy: Sequence length {best_acc['sequence_length']} "
                  f"({best_acc['value']:.1f}% within 2px)")
        
        if 'error' not in recommendations['best_speed']:
            best_speed = recommendations['best_speed']
            print(f"Best Speed: Sequence length {best_speed['sequence_length']} "
                  f"({best_speed['value']:.1f}s processing time)")
        
        if 'error' not in recommendations['best_overall']:
            best_overall = recommendations['best_overall']
            print(f"Best Overall: Sequence length {best_overall['sequence_length']} "
                  f"(composite score: {best_overall['composite_score']:.3f})")
        
        print(f"\nSummary: {recommendations['summary']}")
        print(f"\n{'='*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MemFlow Sequence Length Test')
    parser.add_argument('--motion', choices=['slow', 'medium', 'fast'], required=True,
                       help='Motion speed for test')
    parser.add_argument('--frames', type=int, default=120,
                       help='Number of frames to generate (default: 120 - 4 seconds at 30fps)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--seq-lengths', type=str, default='3,5,10,15,25,50',
                       help='Comma-separated sequence lengths to test (default: 3,5,10,15,25,50)')
    
    args = parser.parse_args()
    
    # Parse sequence lengths
    try:
        seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(',')]
    except ValueError:
        print("Error: Invalid sequence lengths format. Use comma-separated integers")
        return 1
    
    # Validate arguments
    if any(seq_len < 2 for seq_len in seq_lengths):
        print("Error: All sequence lengths must be at least 2")
        return 1
    
    if any(seq_len >= args.frames for seq_len in seq_lengths):
        print(f"Warning: Some sequence lengths >= total frames ({args.frames})")
    
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(test_dir, '..', '..')
    
    # Run test
    runner = SequenceLengthTestRunner(test_dir, project_root)
    results = runner.run_sequence_length_test(
        args.motion, args.frames, args.fps, seq_lengths
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 