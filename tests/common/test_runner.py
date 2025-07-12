"""
Base Test Runner for Optical Flow Testing

Provides common functionality for running optical flow tests.
"""

import os
import sys
import subprocess
import time
import json
from typing import Dict, Any, Optional
from tqdm import tqdm

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from processing import FlowProcessorFactory


class BaseTestRunner:
    """Base class for optical flow test runners"""
    
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
    
    def run_flow_processor(self, input_video: str, output_dir: str, model: str, frames: int,
                          sequence_length: int = 5, dataset: str = 'sintel') -> str:
        """
        Run optical flow processor
        
        Args:
            input_video: Path to input video
            output_dir: Directory for output
            model: Model name ('videoflow' or 'memflow')
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
            '--sequence-length', str(sequence_length)
        ]
        
        # Add dataset parameter for VideoFlow
        if model == 'videoflow':
            cmd.extend(['--vf-dataset', dataset])
        
        print(f"Running optical flow processor with {model}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
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
                    # For VideoFlow, also check dataset match
                    if model == 'videoflow' and dataset not in item:
                        continue
                    # Check sequence length match first (most specific)
                    if f'seq{sequence_length}' in item:
                        # Also check frames parameter for additional precision
                        if f'frames{frames}' in item:
                            cache_dir = item_path
                            break
                        # If frames not in name but sequence matches, use it
                        elif cache_dir is None:
                            cache_dir = item_path
                    # Fallback: check frames parameter only
                    elif f'frames{frames}' in item and cache_dir is None:
                        cache_dir = item_path
                    # Final fallback: if no specific parameters found, use any matching cache
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
    
    def save_results(self, results: Dict[str, Any], file_path: str):
        """Save results to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def print_basic_results(self, results: Dict[str, Any], test_name: str):
        """Print basic results summary"""
        print(f"\n{'='*80}")
        print(f"{test_name.upper()} RESULTS")
        print(f"{'='*80}")
        
        test_info = results.get('test_info', {})
        print(f"Test duration: {test_info.get('test_duration', 0):.1f} seconds")
        
        models = results.get('models', {})
        if not models:
            print("No model results found")
            return
        
        for model_key, model_results in models.items():
            print(f"\n{model_key.upper()} Results:")
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
        
        print(f"\n{'='*80}")
    
    def cleanup_temp_files(self, keep_results: bool = True):
        """Clean up temporary files"""
        if not keep_results:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            os.makedirs(self.temp_dir, exist_ok=True) 