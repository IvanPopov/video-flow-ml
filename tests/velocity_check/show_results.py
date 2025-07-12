#!/usr/bin/env python3
"""
Show Velocity Test Results

This script displays and compares results from all speed tests
(slow, medium, fast) in a formatted table.
"""

import os
import json
import sys
from typing import Dict, Any, List


def load_results(speed: str) -> Dict[str, Any]:
    """Load results for given speed"""
    # Try multiple possible paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        f'temp/results_{speed}.json',
        os.path.join(script_dir, f'temp/results_{speed}.json'),
        os.path.join(script_dir, '..', '..', 'tests', 'velocity_check', 'temp', f'results_{speed}.json')
    ]
    
    for results_file in possible_paths:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
    
    return None


def format_results_table(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """Format results into a comparison table"""
    
    # Table header
    table = "="*80 + "\n"
    table += "VELOCITY TEST RESULTS COMPARISON\n"
    table += "="*80 + "\n\n"
    
    # Column headers
    table += f"{'Speed':<8} {'Model':<12} {'Frames':<7} {'Mean Error':<12} {'Direction':<12} {'Accuracy':<12}\n"
    table += f"{'Test':<8} {'Name':<12} {'Analyzed':<7} {'(px/frame)':<12} {'Error (°)':<12} {'(< 2px)':<12}\n"
    table += "-"*80 + "\n"
    
    # Data rows
    for speed in ['slow', 'medium', 'fast']:
        if speed not in results_dict:
            continue
            
        results = results_dict[speed]
        first_row = True
        
        for model_name in ['videoflow', 'memflow']:
            if model_name not in results['models']:
                continue
                
            model_results = results['models'][model_name]
            
            if 'error' in model_results:
                error_info = model_results['error'][:30] + "..." if len(model_results['error']) > 30 else model_results['error']
                table += f"{'':8} {model_name:<12} {'ERROR':<7} {error_info:<12} {'':12} {'':12}\n"
                continue
            
            speed_col = speed.upper() if first_row else ""
            frames = model_results['total_frames_analyzed']
            mean_error = f"{model_results['mean_velocity_error']:.2f}"
            direction_error = f"{model_results['mean_direction_error']:.1f}"
            accuracy = f"{model_results['accuracy_threshold_2px']:.1f}%"
            
            table += f"{speed_col:<8} {model_name:<12} {frames:<7} {mean_error:<12} {direction_error:<12} {accuracy:<12}\n"
            first_row = False
    
    table += "-"*80 + "\n"
    
    return table


def show_detailed_analysis(results_dict: Dict[str, Dict[str, Any]]):
    """Show detailed analysis of the results"""
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    for speed in ['slow', 'medium', 'fast']:
        if speed not in results_dict:
            continue
            
        results = results_dict[speed]
        print(f"\n{speed.upper()} MOTION TEST:")
        print("-" * 40)
        
        # Show ground truth info
        video_info = results.get('test_info', {})
        print(f"Ground Truth: {video_info.get('frames', 'N/A')} frames at {video_info.get('fps', 'N/A')} fps")
        
        # Compare models
        vf_results = results['models'].get('videoflow', {})
        mf_results = results['models'].get('memflow', {})
        
        if 'error' not in vf_results and 'error' not in mf_results:
            print(f"VideoFlow - Mean Error: {vf_results.get('mean_velocity_error', 0):.2f} px/frame")
            print(f"MemFlow   - Mean Error: {mf_results.get('mean_velocity_error', 0):.2f} px/frame")
            
            # Determine winner
            vf_error = vf_results.get('mean_velocity_error', float('inf'))
            mf_error = mf_results.get('mean_velocity_error', float('inf'))
            
            if vf_error < mf_error:
                print(f"→ VideoFlow is more accurate by {mf_error - vf_error:.2f} px/frame")
            elif mf_error < vf_error:
                print(f"→ MemFlow is more accurate by {vf_error - mf_error:.2f} px/frame")
            else:
                print(f"→ Both models have similar accuracy")
        else:
            if 'error' in vf_results:
                print(f"VideoFlow - ERROR: {vf_results['error']}")
            if 'error' in mf_results:
                print(f"MemFlow   - ERROR: {mf_results['error']}")


def show_timing_analysis(results_dict: Dict[str, Dict[str, Any]]):
    """Show timing analysis of the results"""
    
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS - CHECKING FOR TIMING ISSUES")
    print("="*80)
    
    for speed in ['slow', 'medium', 'fast']:
        if speed not in results_dict:
            continue
            
        results = results_dict[speed]
        print(f"\n{speed.upper()} MOTION TEST:")
        print("-" * 40)
        
        for model_name in ['videoflow', 'memflow']:
            model_results = results['models'].get(model_name, {})
            if 'error' in model_results or 'frame_analyses' not in model_results:
                continue
                
            print(f"\n{model_name.upper()} Frame-by-frame Analysis:")
            
            frame_analyses = model_results['frame_analyses']
            
            # Check for consistent direction errors
            direction_errors = [frame['direction_error'] for frame in frame_analyses]
            consistent_opposite = sum(1 for err in direction_errors if err > 160) / len(direction_errors)
            
            if consistent_opposite > 0.8:
                print(f"  → ISSUE: {consistent_opposite*100:.0f}% of frames show opposite direction (>160° error)")
                print(f"  → This suggests the model is predicting flow in the wrong direction")
            
            # Check for temporal offset patterns
            velocity_errors = [frame['velocity_error'] for frame in frame_analyses]
            if len(velocity_errors) >= 5:
                # Check if later frames have better accuracy
                early_error = sum(velocity_errors[:len(velocity_errors)//2]) / (len(velocity_errors)//2)
                late_error = sum(velocity_errors[len(velocity_errors)//2:]) / (len(velocity_errors) - len(velocity_errors)//2)
                
                if early_error > late_error * 1.5:
                    print(f"  → POSSIBLE TIMING ISSUE: Early frames much worse ({early_error:.1f} vs {late_error:.1f})")
                    print(f"  → This may indicate temporal offset or model warmup issues")
                elif late_error > early_error * 1.5:
                    print(f"  → POSSIBLE DRIFT: Later frames much worse ({late_error:.1f} vs {early_error:.1f})")
                    print(f"  → This may indicate accumulated error over time")
                else:
                    print(f"  → Temporal consistency: Early={early_error:.1f}, Late={late_error:.1f} px/frame")
            
            # Show first few frames for detailed inspection
            print(f"  → First 3 frames detailed:")
            for i, frame in enumerate(frame_analyses[:3]):
                gt_vel = frame['ground_truth_velocity']
                pred_vel = frame['predicted_velocity']
                print(f"    Frame {i}: GT=({gt_vel[0]:.1f}, {gt_vel[1]:.1f}) → Pred=({pred_vel[0]:.1f}, {pred_vel[1]:.1f})")


def main():
    """Main function"""
    
    # Load all available results
    results_dict = {}
    for speed in ['slow', 'medium', 'fast']:
        results = load_results(speed)
        if results:
            results_dict[speed] = results
    
    if not results_dict:
        print("No test results found. Run some tests first!")
        return
    
    # Show comparison table
    table = format_results_table(results_dict)
    print(table)
    
    # Show detailed analysis
    show_detailed_analysis(results_dict)
    
    # Show timing analysis
    show_timing_analysis(results_dict)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. VideoFlow showing opposite direction suggests coordinate system issues")
    print("2. MemFlow showing better accuracy but still high errors")
    print("3. Consider checking frame indexing and temporal sequence handling")
    print("4. May need to investigate model-specific coordinate conventions")
    print("5. Run tests with more frames to confirm temporal patterns")


if __name__ == '__main__':
    main() 