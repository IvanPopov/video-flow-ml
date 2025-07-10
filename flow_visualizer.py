#!/usr/bin/env python3
"""
Interactive Optical Flow Visualizer

This script provides an interactive GUI for visualizing optical flow data:
- Shows two consecutive video frames vertically
- Slider to navigate through frame pairs
- Mouse hover shows flow vector as arrow pointing to next frame pixel
- Supports both .flo and .npz flow formats
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import glob
from pathlib import Path
import threading
import queue
import time
from scipy import signal
from scipy.ndimage import rotate
import torch
from tqdm import tqdm
from VideoFlow.core.utils.frame_utils import writeFlow
import subprocess
import concurrent.futures
import math

# Add flow_processor to path for loading flow data
sys.path.insert(0, os.getcwd())
from flow_processor import VideoFlowProcessor
import correction_worker

class FlowVisualizer:
    def __init__(self, video_path, flow_dir, start_frame=0, max_frames=None, 
                 flow_model='videoflow', model_path=None, stage='sintel',
                 vf_dataset='sintel', vf_architecture='mof', vf_variant='standard'):
        self.video_path = video_path
        self.flow_dir = flow_dir
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.processor = VideoFlowProcessor(flow_model=flow_model, model_path=model_path, stage=stage,
                                          vf_dataset=vf_dataset, vf_architecture=vf_architecture, 
                                          vf_variant=vf_variant)
        
        # Load video frames
        self.frames = self.load_video_frames()
        self.flow_files = self.find_flow_files()
        
        if len(self.frames) <= 1:
            raise ValueError("Not enough frames loaded from video")
        if len(self.flow_files) == 0:
            raise ValueError("No flow files found in directory")
            
        print(f"Loaded {len(self.frames) - 1} original frames ({len(self.frames)} with duplicate) and {len(self.flow_files)} flow files")
        
        # Current state
        self.current_pair = 0
        self.max_pairs = len(self.frames) - 1
        
        # LOD support - moved up to be available for preloading
        self.current_lod_level = 0  # Current LOD level
        self.use_lod_for_vectors = False  # Whether to use LOD for flow vectors
        self.max_lod_levels = 5  # Maximum number of LOD levels
        
        # --- Data Caching ---
        # Cache all flow and LOD data in memory for the session
        self.flow_data_cache = {}
        self.lod_data_cache = {}
        self._preload_all_data()
        self._log_cache_statistics()
        
        # Initialize quality maps storage
        self.quality_maps = {}
        self.quality_map_queue = queue.Queue()
        self.computing_quality = set()  # Track which frames are being computed
        self.computation_threads = {}  # Track active threads
        self.stop_computation = threading.Event()  # Signal to stop computations
        self.slider_dragging = False  # Track if slider is being dragged
        
        # Initialize turbulence maps storage
        self.turbulence_maps = {}
        
        # UI state
        self.detail_analysis_mode = False
        self.detail_analysis_data = None
        self.detail_analysis_region_size = 25
        self.template_radius = 5.5  # For 11x11 template
        self.search_radius = self.detail_analysis_region_size # Search in the whole 50x50 region
        
        # --- Quality Thresholds ---
        self.GOOD_QUALITY_THRESHOLD = 0.8  # Defines a "good" vs "bad" pixel similarity.
        self.FINE_CORRECTION_THRESHOLD = 0.9 # If coarse similarity is below this, attempt fine correction.
        
        # Check for LOD data availability
        self.check_lod_availability()
        
        # A tag for the circle marking the selected pixel on the quality map
        self.quality_map_marker_tag = "quality_map_marker"
        
        # Pre-compute only the first quality map
        print("Computing initial quality map...")
        if self.max_pairs > 0:
            frame1 = self.frames[0]
            frame2 = self.frames[1]
            flow = self.load_flow_data(0)
            if 'cuda' in str(self.processor.device):
                self.quality_maps[0] = correction_worker.generate_quality_frame_gpu(frame1, frame2, flow, self.processor.device, self.GOOD_QUALITY_THRESHOLD)
            else:
                self.quality_maps[0] = correction_worker.generate_quality_frame_fast(frame1, frame2, flow, self.GOOD_QUALITY_THRESHOLD)
            print("Initial quality map ready!")
        
        # Zoom state
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_step = 0.1
        
        # Pan state for dragging
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        self.last_pan_x = 0
        self.last_pan_y = 0
        
        # Track pixels that failed correction attempts
        self.failed_correction_pixels = {}
        self.improved_correction_pixels = {}
        
        # UI setup
        self.setup_ui()
        
        # Track quality map marker coordinates
        self.quality_map_marker_coords = None
        
        # Start background quality computation checker
        self.check_quality_queue()
        
        self.update_display()
        
    def _preload_all_data(self):
        """Load all flow and LOD data into memory at startup."""
        print("Preloading all flow and LOD data into memory. This may take a moment...")
        num_flows = len(self.flow_files)
        
        for i in range(num_flows):
            # Print progress
            progress_percent = (i + 1) / (num_flows + 1) * 100
            sys.stdout.write(f"\rLoading data: {i+1}/{num_flows + 1} ({progress_percent:.1f}%)")
            sys.stdout.flush()

            # Load main flow
            try:
                flow_file = self.flow_files[i]
                if flow_file.endswith('.flo'):
                    flow = self.processor.load_flow_flo(flow_file)
                elif flow_file.endswith('.npz'):
                    npz_data = self.processor.load_flow_npz(flow_file)
                    flow = npz_data['flow']
                self.flow_data_cache[i] = flow
            except Exception as e:
                print(f"\nWarning: Could not load flow file {self.flow_files[i]}: {e}")
                self.flow_data_cache[i] = None

            # Load all available LODs for this flow
            for lod_level in range(self.max_lod_levels):
                lod_file = os.path.join(self.flow_dir, f"flow_frame_{i:06d}_lod{lod_level}.npz")
                if os.path.exists(lod_file):
                    try:
                        npz_data = self.processor.load_flow_npz(lod_file)
                        self.lod_data_cache[(i, lod_level)] = npz_data['flow']
                    except Exception as e:
                        print(f"\nWarning: Could not load LOD file {lod_file}: {e}")
        
        # Add flow for the last duplicated frame
        if num_flows > 0:
            last_flow = self.flow_data_cache.get(num_flows - 1)
            if last_flow is not None:
                progress_percent = (num_flows + 1) / (num_flows + 1) * 100
                sys.stdout.write(f"\rLoading data: {num_flows + 1}/{num_flows + 1} ({progress_percent:.1f}%)")
                sys.stdout.flush()

                self.flow_data_cache[num_flows] = last_flow.copy()

                # Add LODs for duplicated frame by copying from the previous one
                for lod_level in range(self.max_lod_levels):
                    last_lod = self.lod_data_cache.get((num_flows - 1, lod_level))
                    if last_lod is not None:
                        self.lod_data_cache[(num_flows, lod_level)] = last_lod.copy()
        
        print("\nAll data preloaded successfully.")
    
    def _generate_and_save_lods(self, frame_indices):
        """Generate, save, and cache LODs for the given frame indices."""
        if not frame_indices:
            return
        
        os.makedirs(self.flow_dir, exist_ok=True)
        
        print(f"Generating LODs for {len(frame_indices)} frames...")
        
        for i in tqdm(frame_indices, desc="Generating LODs"):
            flow_data = self.flow_data_cache.get(i)
            if flow_data is None:
                print(f"\nWarning: Cannot generate LODs for frame {i}, base flow data not found.")
                continue
            
            try:
                # Generate LOD pyramid
                lods = self.generate_flow_lods(flow_data, num_lods=self.max_lod_levels)
                
                # Save LODs to disk using the processor's method to ensure consistency
                self.save_flow_lods(lods, self.flow_dir, i)
                
                # Update in-memory cache
                for lod_level, lod_data in enumerate(lods):
                    # LOD0 is the same as the main flow, so we don't cache it separately
                    if lod_level > 0:
                        self.lod_data_cache[(i, lod_level)] = lod_data
            except Exception as e:
                print(f"\nError generating/saving LODs for frame {i}: {e}")
    
    def _log_cache_statistics(self, is_recursive_call=False):
        if not is_recursive_call:
            print("\n--- Cache Statistics ---")

        if not self.flow_data_cache:
            if not is_recursive_call:
                print("Cache is empty.")
                print("------------------------\n")
            return

        num_flow_frames = len(self.flow_data_cache)
        num_expected_frames = self.max_pairs

        if not is_recursive_call:
            print(f"Flow data loaded for {num_flow_frames} frames in memory.")
        
        total_lods = 0
        frames_with_lods = set()
        total_lod_size_bytes = 0
        lod_details = {}

        # Check LODs by iterating through cached flow data
        for i in self.flow_data_cache.keys():
            if i >= num_expected_frames: continue 

            lod_details[i] = {'count': 0, 'size': 0}
            has_lods_for_frame = False
            for lod_level in range(self.max_lod_levels):
                cache_key = (i, lod_level)
                if cache_key in self.lod_data_cache:
                    lod_data = self.lod_data_cache.get(cache_key)
                    if lod_data is not None:
                        lod_size = lod_data.nbytes
                        total_lods += 1
                        lod_details[i]['count'] += 1
                        lod_details[i]['size'] += lod_size
                        total_lod_size_bytes += lod_size
                        has_lods_for_frame = True
            
            if has_lods_for_frame:
                frames_with_lods.add(i)

        # On first pass, check for and generate missing LODs
        if not is_recursive_call:
            frames_to_generate_lods = []
            for i in range(num_expected_frames):
                # Generate if no LODs exist, or if they are only partial.
                count = lod_details.get(i, {}).get('count', 0)
                if count < self.max_lod_levels:
                    frames_to_generate_lods.append(i)
            
            if frames_to_generate_lods:
                print(f"\nFound {len(frames_to_generate_lods)} frames with missing or partial LODs.")
                print("Generating on-the-fly...")
                self._generate_and_save_lods(frames_to_generate_lods)
                
                # Re-run statistics to print updated info
                print("\n--- Regenerated Cache Statistics ---")
                self._log_cache_statistics(is_recursive_call=True)
                
                # Show detailed LOD statistics after generation
                self.analyze_lod_cache_statistics(self.flow_dir, num_expected_frames)
                return # Exit after recursive call

        # --- Logging logic (only runs on first pass or if no generation needed) ---
        print(f"Found LODs for {len(frames_with_lods)} / {num_expected_frames} frames.")

        if total_lods > 0:
            avg_lod_size_kb = (total_lod_size_bytes / total_lods) / 1024 if total_lods > 0 else 0
            print(f"Total LOD files in memory: {total_lods}")
            print(f"Total size of LODs: {total_lod_size_bytes / (1024*1024):.2f} MB")
            print(f"Average LOD file size: {avg_lod_size_kb:.2f} KB")

        frames_missing_lods = []
        for i in range(num_expected_frames):
            if i not in frames_with_lods:
                frames_missing_lods.append(i)
        
        if frames_missing_lods:
            print("\nFrames missing all LODs:")
            if len(frames_missing_lods) > 15:
                print(f"  {len(frames_missing_lods)} frames are missing LODs.")
            else:
                print(f"  Frame numbers: {', '.join(map(str, frames_missing_lods))}")

        frames_partial_lods = []
        for i in range(num_expected_frames):
            count = lod_details.get(i, {}).get('count', 0)
            if i in frames_with_lods and count < self.max_lod_levels:
                 frames_partial_lods.append(f"{i} ({count}/{self.max_lod_levels})")

        if frames_partial_lods:
            print("\nFrames with partial LODs (found/expected):")
            if len(frames_partial_lods) > 15:
                 print(f"  {len(frames_partial_lods)} frames have partial LODs.")
            else:
                print(f"  Frame (LODs): {', '.join(frames_partial_lods)}")

        if not frames_missing_lods and not frames_partial_lods:
            print("\nAll frames have complete LODs.")
            
        print("------------------------\n")
        
        # Show detailed LOD statistics if this is not a recursive call and LODs exist
        if not is_recursive_call and (frames_with_lods or total_lods > 0):
            self.analyze_lod_cache_statistics(self.flow_dir, num_expected_frames)
    
    def load_video_frames(self):
        """Load frames from video starting at start_frame"""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if we've reached max frames
            if self.max_frames is not None and frame_count >= self.max_frames:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
        cap.release()
        if frames:
            frames.append(frames[-1])
        return frames
    
    def find_flow_files(self):
        """Find all flow files in directory, excluding LOD files."""
        flow_files = []
        
        # Look for .flo files
        flo_pattern = os.path.join(self.flow_dir, "*.flo")
        flo_files = sorted(glob.glob(flo_pattern))
        
        # Look for .npz files if no .flo found
        if not flo_files:
            npz_pattern = os.path.join(self.flow_dir, "*.npz")
            npz_files = sorted(glob.glob(npz_pattern))
            # Filter out LOD files, which have '_lod' in their names
            flow_files = [f for f in npz_files if '_lod' not in os.path.basename(f)]
        else:
            flow_files = flo_files
            
        return flow_files
    
    def check_lod_availability(self):
        """Check if LOD data is available in the flow directory"""
        # With pre-caching, we can just check if the cache has any LOD data
        return any(key[1] > 0 for key in self.lod_data_cache.keys()) if hasattr(self, 'lod_data_cache') else False
    
    def load_lod_data(self, frame_idx, lod_level):
        """Load LOD data for a specific frame and level."""
        # LOD0 is always the original full-resolution flow data.
        if lod_level == 0:
            return self.load_flow_data(frame_idx)
            
        # For other levels, retrieve from the dedicated LOD cache.
        cache_key = (frame_idx, lod_level)
        return self.lod_data_cache.get(cache_key)
    
    def load_flow_data(self, frame_idx):
        """Load flow data for specific frame from the in-memory cache."""
        return self.flow_data_cache.get(frame_idx)
    
    def generate_flow_lods(self, flow_data, num_lods=5):
        """Generate Level-of-Detail (LOD) pyramid for flow data"""
        return self.processor.generate_flow_lods(flow_data, num_lods)
    
    def save_flow_lods(self, lods, cache_dir, frame_idx):
        """Save LOD pyramid for a frame"""
        return self.processor.save_flow_lods(lods, cache_dir, frame_idx)
    
    def load_flow_lod(self, cache_dir, frame_idx, lod_level=0):
        """Load specific LOD level for a frame"""
        return self.processor.load_flow_lod(cache_dir, frame_idx, lod_level)
    
    def check_flow_lods_exist(self, cache_dir, max_frames, num_lods=5):
        """Check if LOD pyramid exists for all frames"""
        return self.processor.check_flow_lods_exist(cache_dir, max_frames, num_lods)
    
    def generate_lods_for_cache(self, cache_dir, max_frames, num_lods=5):
        """Generate LOD pyramids for all frames in cache"""
        return self.processor.generate_lods_for_cache(cache_dir, max_frames, num_lods)
    
    def analyze_lod_cache_statistics(self, cache_dir, max_frames, num_lods=5):
        """
        Analyze and report detailed LOD cache statistics
        
        Args:
            cache_dir: Cache directory path
            max_frames: Number of frames to analyze
            num_lods: Expected number of LOD levels
        """
        print("\n--- LOD Cache Statistics (FlowVisualizer) ---")
        
        if not os.path.exists(cache_dir):
            print("Cache directory not found - no LOD data available.")
            print("--------------------------------------------\n")
            return
            
        # Statistics tracking
        total_lod_files = 0
        total_lod_size_bytes = 0
        frames_with_complete_lods = 0
        frames_with_partial_lods = 0
        frames_missing_lods = 0
        
        # Per-level statistics
        lod_level_stats = {}
        for level in range(num_lods):
            lod_level_stats[level] = {
                'count': 0,
                'total_size': 0,
                'missing_frames': [],
                'dimensions': set()  # Store unique dimensions for this LOD level
            }
        
        # Per-frame analysis
        frame_lod_details = {}
        
        print(f"Analyzing LOD data for {max_frames} frames with {num_lods} expected levels...")
        
        for frame_idx in range(max_frames):
            frame_lods = {}
            frame_total_size = 0
            frame_lod_count = 0
            
            for lod_level in range(num_lods):
                lod_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
                
                if os.path.exists(lod_file):
                    try:
                        # Get file size
                        file_size = os.path.getsize(lod_file)
                        
                        # Load LOD data to get dimensions
                        lod_data = self.load_flow_lod(cache_dir, frame_idx, lod_level)
                        if lod_data is not None:
                            height, width = lod_data.shape[:2]
                            dimensions = (width, height)
                            lod_level_stats[lod_level]['dimensions'].add(dimensions)
                        else:
                            dimensions = None
                        
                        frame_lods[lod_level] = {
                            'size': file_size,
                            'dimensions': dimensions
                        }
                        frame_total_size += file_size
                        frame_lod_count += 1
                        
                        # Update level statistics
                        lod_level_stats[lod_level]['count'] += 1
                        lod_level_stats[lod_level]['total_size'] += file_size
                        
                        total_lod_files += 1
                        total_lod_size_bytes += file_size
                        
                    except Exception as e:
                        print(f"Warning: Could not read LOD file {lod_file}: {e}")
                        lod_level_stats[lod_level]['missing_frames'].append(frame_idx)
                else:
                    lod_level_stats[lod_level]['missing_frames'].append(frame_idx)
            
            # Categorize frame
            if frame_lod_count == num_lods:
                frames_with_complete_lods += 1
            elif frame_lod_count > 0:
                frames_with_partial_lods += 1
            else:
                frames_missing_lods += 1
            
            frame_lod_details[frame_idx] = {
                'count': frame_lod_count,
                'total_size': frame_total_size,
                'lods': frame_lods
            }
        
        # Print summary statistics
        print(f"\nOverall Summary:")
        print(f"  Total LOD files found: {total_lod_files}")
        print(f"  Total LOD data size: {total_lod_size_bytes / (1024*1024):.2f} MB")
        print(f"  Average LOD file size: {(total_lod_size_bytes / total_lod_files / 1024):.1f} KB" if total_lod_files > 0 else "  Average LOD file size: N/A")
        
        print(f"\nFrame Coverage:")
        print(f"  Frames with complete LODs ({num_lods}/{num_lods}): {frames_with_complete_lods}")
        print(f"  Frames with partial LODs: {frames_with_partial_lods}")
        print(f"  Frames missing all LODs: {frames_missing_lods}")
        
        completion_rate = (frames_with_complete_lods / max_frames) * 100 if max_frames > 0 else 0
        print(f"  Completion rate: {completion_rate:.1f}%")
        
        # Per-level statistics
        print(f"\nPer-Level Statistics:")
        for level in range(num_lods):
            stats = lod_level_stats[level]
            coverage = (stats['count'] / max_frames) * 100 if max_frames > 0 else 0
            avg_size = (stats['total_size'] / stats['count'] / 1024) if stats['count'] > 0 else 0
            
            print(f"  LOD Level {level}:")
            print(f"    Files found: {stats['count']}/{max_frames} ({coverage:.1f}%)")
            print(f"    Total size: {stats['total_size'] / (1024*1024):.2f} MB")
            print(f"    Average size: {avg_size:.1f} KB")
            
            # Show dimensions information
            if stats['dimensions']:
                if len(stats['dimensions']) == 1:
                    width, height = list(stats['dimensions'])[0]
                    print(f"    Dimensions: {width}x{height} pixels")
                else:
                    print(f"    Dimensions: {len(stats['dimensions'])} different sizes found:")
                    for width, height in sorted(stats['dimensions']):
                        print(f"      {width}x{height} pixels")
            else:
                print(f"    Dimensions: No valid data")
            
            if len(stats['missing_frames']) > 0:
                if len(stats['missing_frames']) <= 10:
                    missing_str = ', '.join(map(str, stats['missing_frames']))
                    print(f"    Missing frames: {missing_str}")
                else:
                    print(f"    Missing frames: {len(stats['missing_frames'])} frames (showing first 10)")
                    missing_str = ', '.join(map(str, stats['missing_frames'][:10]))
                    print(f"      {missing_str}...")
        
        # Identify problematic frames
        problematic_frames = []
        for frame_idx in range(max_frames):
            details = frame_lod_details[frame_idx]
            if details['count'] < num_lods:
                problematic_frames.append(frame_idx)
        
        if problematic_frames:
            print(f"\nProblematic Frames (missing some/all LODs):")
            if len(problematic_frames) <= 20:
                for frame_idx in problematic_frames:
                    details = frame_lod_details[frame_idx]
                    print(f"  Frame {frame_idx}: {details['count']}/{num_lods} LODs, {details['total_size']/1024:.1f} KB")
            else:
                print(f"  {len(problematic_frames)} frames have missing LODs")
                print(f"  First 10: {', '.join(map(str, problematic_frames[:10]))}")
                print(f"  Last 10: {', '.join(map(str, problematic_frames[-10:]))}")
        
        # Size distribution analysis
        if total_lod_files > 0:
            all_sizes = []
            for frame_idx in range(max_frames):
                for lod_level, lod_info in frame_lod_details[frame_idx]['lods'].items():
                    if isinstance(lod_info, dict) and 'size' in lod_info:
                        all_sizes.append(lod_info['size'])
            
            if all_sizes:
                all_sizes.sort()
                min_size = min(all_sizes) / 1024
                max_size = max(all_sizes) / 1024
                median_size = all_sizes[len(all_sizes)//2] / 1024
                
                print(f"\nSize Distribution:")
                print(f"  Minimum LOD file: {min_size:.1f} KB")
                print(f"  Maximum LOD file: {max_size:.1f} KB")
                print(f"  Median LOD file: {median_size:.1f} KB")
        
        print("--------------------------------------------\n")
    
    def compute_quality_map_background(self, frame_idx):
        """Compute quality map for a specific frame in background thread"""
        def worker():
            try:
                # Check if computation should be stopped
                if self.stop_computation.is_set():
                    return
                
                # Load frames and flow
                frame1 = self.frames[frame_idx]
                frame2 = self.frames[frame_idx + 1]
                flow = self.load_flow_data(frame_idx)
                
                # Check again before heavy computation
                if self.stop_computation.is_set():
                    return
                
                # Generate quality map
                if 'cuda' in str(self.processor.device):
                    quality_map = correction_worker.generate_quality_frame_gpu(frame1, frame2, flow, self.processor.device, self.GOOD_QUALITY_THRESHOLD)
                else:
                    quality_map = correction_worker.generate_quality_frame_fast(frame1, frame2, flow, self.GOOD_QUALITY_THRESHOLD)
                
                # Check if result is still needed
                if not self.stop_computation.is_set():
                    # Put result in queue
                    self.quality_map_queue.put((frame_idx, quality_map))
                
            except Exception as e:
                if not self.stop_computation.is_set():
                    print(f"Error computing quality map for frame {frame_idx}: {e}")
                    # Put error result
                    self.quality_map_queue.put((frame_idx, None))
            finally:
                # Remove from computing set and thread tracking
                self.computing_quality.discard(frame_idx)
                self.computation_threads.pop(frame_idx, None)
        
        # Start background thread
        thread = threading.Thread(target=worker, daemon=True)
        self.computation_threads[frame_idx] = thread
        thread.start()
    
    def check_quality_queue(self):
        """Check for completed quality map computations"""
        try:
            while True:
                frame_idx, quality_map = self.quality_map_queue.get_nowait()
                if quality_map is not None:
                    self.quality_maps[frame_idx] = quality_map
                    # Update display if this is the current frame
                    if frame_idx == self.current_pair:
                        self.update_display()
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_quality_queue)
    
    def stop_all_computations(self):
        """Stop all background computations"""
        self.stop_computation.set()
        # Clear computing set
        self.computing_quality.clear()
        # Clear thread tracking
        self.computation_threads.clear()
        # Clear queue
        while not self.quality_map_queue.empty():
            try:
                self.quality_map_queue.get_nowait()
            except queue.Empty:
                break
    
    def resume_computations(self):
        """Resume background computations"""
        self.stop_computation.clear()
    
    def compute_current_quality_map(self):
        """
        Computes the quality map for the currently selected frame pair synchronously.
        The UI may be unresponsive during this time.
        """
        frame_idx = self.current_pair
        
        try:
            frame1 = self.frames[frame_idx]
            frame2 = self.frames[frame_idx + 1]
            flow = self.load_flow_data(frame_idx)
            
            if flow is None:
                messagebox.showerror("Error", f"No flow data available for frame {frame_idx}.")
                return

            print(f"Generating quality map for frame {frame_idx}...")
            start_time = time.time()
            
            # Choose computation method based on device availability
            if 'cuda' in str(self.processor.device):
                quality_map = correction_worker.generate_quality_frame_gpu(frame1, frame2, flow, self.processor.device, self.GOOD_QUALITY_THRESHOLD)
            else:
                quality_map = correction_worker.generate_quality_frame_fast(frame1, frame2, flow, self.GOOD_QUALITY_THRESHOLD)

            self.quality_maps[frame_idx] = quality_map
            duration = time.time() - start_time
            print(f"Quality map generation finished in {duration:.4f}s")

            self.update_display() # Refresh UI with the new map

        except Exception as e:
            print(f"Error during quality map generation: {e}")
            messagebox.showerror("Error", f"An error occurred during quality map generation:\n{e}")
    
    def request_quality_map(self, frame_idx):
        """Request quality map for a frame (compute if not available)"""
        if (frame_idx not in self.quality_maps and 
            frame_idx not in self.computing_quality and 
            not self.stop_computation.is_set() and
            not self.slider_dragging):
            self.computing_quality.add(frame_idx)
            self.compute_quality_map_background(frame_idx)
    
    def preload_adjacent_frames(self):
        """Preload quality maps for adjacent frames"""
        # Preload previous and next frames
        for offset in [-1, 1]:
            frame_idx = self.current_pair + offset
            if 0 <= frame_idx < self.max_pairs:
                self.request_quality_map(frame_idx)
    
    def phase_correlation_with_rotation(self, img1, img2):
        """
        Compute phase correlation between two images to find translation.
        Rotation is not calculated and is returned as 0.
        Returns: (dx, dy, angle, confidence)
        """
        return correction_worker.phase_correlation_with_rotation(img1, img2)
    
    def _perform_coarse_correction(self, frame1, frame2, source_pixel, lod_flow_vector):
        """Performs coarse correction using phase correlation."""
        return correction_worker.perform_coarse_correction(
            frame1, frame2, source_pixel, lod_flow_vector,
            self.detail_analysis_region_size
        )

    def _perform_fine_correction(self, frame1, frame2, source_pixel, coarse_target_pixel):
        """
        Performs fine-tuning in two stages:
        1. Template Matching (NCC) to find the best structural patch.
        2. Conditional Spiral Search if the patch center's color is a poor match.
        """
        return correction_worker.perform_fine_correction(
            frame1, frame2, source_pixel, coarse_target_pixel,
            self.template_radius, self.search_radius, self.GOOD_QUALITY_THRESHOLD
        )

    def get_highest_available_lod(self, frame_idx):
        """Get the highest available LOD level for a frame"""
        for lod_level in range(self.max_lod_levels - 1, -1, -1):
            lod_data = self.load_lod_data(frame_idx, lod_level)
            if lod_data is not None:
                return lod_level, lod_data
        return None, None
    
    def extract_region(self, image, center_x, center_y, radius):
        """Extract a square region around a center point"""
        return correction_worker.extract_region(image, center_x, center_y, radius)
    
    def on_left_click(self, event):
        """Handle left mouse click for detail analysis or marking the quality map."""
        # First, clear any existing marker on the quality map
        self.canvas.delete(self.quality_map_marker_tag)
        self.quality_map_marker_coords = None
        
        if self.current_flow is None:
            return
            
        # Check if click is on the quality map (frame 3)
        if (self.frame3_x <= event.x <= self.frame3_x + self.display_width and
            self.frame3_y <= event.y <= self.frame3_y + self.display_height):
            
            # Draw a circle on the quality map where the user clicked
            radius = 5
            self.canvas.create_oval(
                event.x - radius, event.y - radius,
                event.x + radius, event.y + radius,
                outline="yellow", width=2, tags=self.quality_map_marker_tag
            )
            self.quality_map_marker_coords = (event.x, event.y)
            return # Stop further processing if click was on quality map

        # Check if click is on first frame for detail analysis
        if (self.frame1_x <= event.x <= self.frame1_x + self.display_width and
            self.frame1_y <= event.y <= self.frame1_y + self.display_height):
            
            # Draw a corresponding marker on the quality map (frame 3)
            marker_x = event.x
            # Y position on frame3 is frame3's y-start + offset of click from frame1's y-start
            marker_y = self.frame3_y + (event.y - self.frame1_y)

            radius = 5
            self.canvas.create_oval(
                marker_x - radius, marker_y - radius,
                marker_x + radius, marker_y + radius,
                outline="yellow", width=2, tags=self.quality_map_marker_tag
            )
            self.quality_map_marker_coords = (marker_x, marker_y)

            # Convert display coordinates to original frame coordinates
            display_x = event.x - self.frame1_x
            display_y = event.y - self.frame1_y
            
            orig_x = int(display_x / self.display_scale)
            orig_y = int(display_y / self.display_scale)
            
            # Clamp to frame bounds
            orig_x = max(0, min(orig_x, self.orig_width - 1))
            orig_y = max(0, min(orig_y, self.orig_height - 1))
            
            # Check if this is a different pixel while in detail mode
            if (self.detail_analysis_mode and 
                self.detail_analysis_data and
                (orig_x, orig_y) != self.detail_analysis_data['source_pixel']):
                
                if not self.check_exit_detail_mode("Clicking on another pixel"):
                    return True
            
            # Check if this pixel has poor flow quality
            if self.current_flow is not None:
                # Get target position using current flow
                fh, fw = self.current_flow.shape[:2]
                scale_x = fw / self.orig_width
                scale_y = fh / self.orig_height
                
                flow_x_coord = int(orig_x * scale_x)
                flow_y_coord = int(orig_y * scale_y)
                flow_x_coord = max(0, min(flow_x_coord, fw - 1))
                flow_y_coord = max(0, min(flow_y_coord, fh - 1))
                
                flow_x = self.current_flow[flow_y_coord, flow_x_coord, 0] / scale_x
                flow_y = self.current_flow[flow_y_coord, flow_x_coord, 1] / scale_y
                
                target_orig_x = orig_x - flow_x
                target_orig_y = orig_y - flow_y
                
                # Check color quality
                arrow_color = self.get_arrow_color(orig_x, orig_y, target_orig_x, target_orig_y)
                shift_pressed = (event.state & 0x0001)
                
                if arrow_color == "red" or shift_pressed:
                    # Poor quality or Shift pressed - start detail analysis, pass original flow vector
                    self.perform_detail_analysis(orig_x, orig_y, (flow_x, flow_y))
                else:
                    messagebox.showinfo("Information", 
                        f"Pixel ({orig_x}, {orig_y}) has good flow quality. "
                        f"Detail analysis is only performed for pixels with poor quality (red arrows).\n\n"
                        f"To analyze any pixel, hold Shift while clicking.")
    
    def _is_good_quality(self, similarity):
        """Check if a similarity score is above the defined quality threshold."""
        return similarity > self.GOOD_QUALITY_THRESHOLD
    
    def perform_detail_analysis(self, orig_x, orig_y, original_flow):
        """Perform detailed analysis on a pixel with poor flow quality"""
        print(f"Starting detail analysis at pixel ({orig_x}, {orig_y})")
        
        # Get highest available LOD
        lod_level, lod_flow = self.get_highest_available_lod(self.current_pair)
        if lod_flow is None:
            messagebox.showerror("Error", "No LOD data available for detail analysis")
            return
        
        print(f"Using LOD level {lod_level}")
        
        # Get LOD flow vector
        lod_h, lod_w = lod_flow.shape[:2]
        lod_scale_x = lod_w / self.orig_width
        lod_scale_y = lod_h / self.orig_height
        
        lod_x = int(orig_x * lod_scale_x)
        lod_y = int(orig_y * lod_scale_y)
        lod_x = max(0, min(lod_x, lod_w - 1))
        lod_y = max(0, min(lod_y, lod_h - 1))
        
        # Get LOD flow vector and scale back to frame coordinates
        lod_flow_x = lod_flow[lod_y, lod_x, 0] / lod_scale_x
        lod_flow_y = lod_flow[lod_y, lod_x, 1] / lod_scale_y
        
        print(f"LOD flow vector: ({lod_flow_x:.2f}, {lod_flow_y:.2f})")
        
        # --- Stage 0: Check for Inconsistency ---
        # Check if this pixel was previously marked as 'failed' to correct
        is_previously_failed = False
        failed_pixels = self.failed_correction_pixels.get(self.current_pair)
        if failed_pixels and (orig_x, orig_y) in failed_pixels:
            is_previously_failed = True
        
        # --- Stage 1: Coarse Correction ---
        coarse_result = self._perform_coarse_correction(self.frame1, self.frame2, (orig_x, orig_y), (lod_flow_x, lod_flow_y))
        print(f"Coarse correction result: flow=({coarse_result['flow'][0]:.2f}, {coarse_result['flow'][1]:.2f}), similarity={coarse_result['similarity']:.3f}")

        # --- Stage 2: Fine Correction (if needed) ---
        fine_result = None
        # If coarse correction is not good enough, try to refine it.
        if coarse_result['similarity'] < self.FINE_CORRECTION_THRESHOLD: 
            print("Coarse result is not good enough, performing fine correction...")
            fine_result = self._perform_fine_correction(self.frame1, self.frame2, (orig_x, orig_y), coarse_result['target'])
            if fine_result:
                print(f"Fine correction result: flow=({fine_result['flow'][0]:.2f}, {fine_result['flow'][1]:.2f}), similarity={fine_result['similarity']:.3f}")
            else:
                print("Fine correction failed (likely out of bounds).")

        # Determine final corrected flow
        if fine_result and fine_result['similarity'] > coarse_result['similarity']:
            final_flow = fine_result['flow']
            final_target = fine_result['target']
            final_similarity = fine_result['similarity']
            confidence = fine_result['confidence']
            print("Using fine correction result.")
        else:
            final_flow = coarse_result['flow']
            final_target = coarse_result['target']
            final_similarity = coarse_result['similarity']
            confidence = coarse_result['confidence']
            print("Using coarse correction result.")
            
        # Store analysis data
        self.detail_analysis_data = {
            'source_pixel': (orig_x, orig_y),
            'original_flow': original_flow,
            'lod_flow': (lod_flow_x, lod_flow_y),
            'lod_target': (orig_x - lod_flow_x, orig_y - lod_flow_y),
            'coarse_result': coarse_result,
            'fine_result': fine_result,
            'final_flow': final_flow,
            'final_target': final_target,
            'confidence': confidence,
            'lod_level': lod_level
        }
        
        # --- Final Check for Inconsistency ---
        if is_previously_failed and self._is_good_quality(final_similarity):
            error_title = "Logic Inconsistency Detected"
            error_message = (
                "Detail analysis found a 'good' correction (similarity > "
                f"{self.GOOD_QUALITY_THRESHOLD}) for a pixel that was previously "
                "marked as 'failed to correct' by the batch process.\n\n"
                "This may indicate a discrepancy between the single-pixel analysis "
                "and the batch correction logic. Please check the console for debug information."
            )
            print(f"--- LOGIC INCONSISTENCY ---")
            print(f"Pixel: ({orig_x}, {orig_y})")
            print(f"Previously marked as: FAILED")
            print(f"Detail analysis found similarity: {final_similarity:.4f} (Good)")
            print(f"Original Flow: {self.detail_analysis_data['original_flow']}")
            print(f"LOD Flow: {self.detail_analysis_data['lod_flow']}")
            print(f"Coarse Result: {coarse_result}")
            print(f"Fine Result: {fine_result}")
            print(f"--------------------------")
            messagebox.showerror(error_title, error_message)

        # Enter detail analysis mode
        self.detail_analysis_mode = True
        
        # Show exit button
        self.exit_detail_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Update display
        self.update_display()
    
    def check_exit_detail_mode(self, action_description):
        """Check if user wants to exit detail analysis mode"""
        if self.detail_analysis_mode:
            result = messagebox.askyesno(
                "Exit Analysis Mode",
                f"{action_description} will reset all additional visualizations. Continue?"
            )
            if result:
                self.exit_detail_analysis_mode()
                return True
            else:
                return False
        return True
    
    def exit_detail_analysis_mode(self):
        """Exit detail analysis mode and return to normal operation"""
        self.detail_analysis_mode = False
        self.detail_analysis_data = None
        # Hide exit button
        self.exit_detail_btn.pack_forget()
        # Clear detail analysis graphics
        self.canvas.delete("detail_analysis")
        # Do not clear the quality map marker when exiting detail mode
        # Update display
        self.update_display()
    
    def setup_ui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("Interactive Optical Flow Visualizer")
        self.root.geometry("1200x1700")
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info label
        video_name = os.path.basename(self.video_path)
        cache_name = os.path.basename(self.flow_dir)
        frame_range = f"Frames {self.start_frame}-{self.start_frame + len(self.frames) - 1}"
        info_text = f"Video: {video_name} | Cache: {cache_name} | {frame_range}"
        self.info_label = ttk.Label(main_frame, text=info_text, font=("Arial", 9))
        self.info_label.pack(pady=(0, 5))
        
        # Statistics label
        stats_text = f"Total frames: {len(self.frames)} | Flow files: {len(self.flow_files)} | Start frame: {self.start_frame}"
        self.stats_label = ttk.Label(main_frame, text=stats_text, font=("Arial", 8), foreground="gray")
        self.stats_label.pack(pady=(0, 10))
        
        # Frame display area
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for frames
        self.canvas = tk.Canvas(self.canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Leave>', self.on_mouse_leave)
        self.canvas.bind('<Button-1>', self.on_left_click)  # Left click for detail analysis
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind('<Double-Button-1>', self.on_double_click)  # Double-click to reset zoom
        self.canvas.bind('<Button-2>', self.on_middle_click)  # Middle mouse button press
        self.canvas.bind('<ButtonRelease-2>', self.on_middle_release)  # Middle mouse button release
        self.canvas.bind('<B2-Motion>', self.on_middle_drag)  # Middle mouse drag
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Frame pair slider
        slider_frame = ttk.Frame(control_frame)
        slider_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(slider_frame, text="Frame Pair:").pack(side=tk.LEFT)
        
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(
            slider_frame, 
            from_=0, 
            to=self.max_pairs-1,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=self.on_slider_change
        )
        
        # Bind slider drag events
        self.frame_slider.bind('<Button-1>', self.on_slider_press)
        self.frame_slider.bind('<ButtonRelease-1>', self.on_slider_release)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        self.frame_label = ttk.Label(slider_frame, text="0/0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Zoom controls
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        
        zoom_out_btn = ttk.Button(zoom_frame, text="-", width=3, command=self.zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=(10, 5))
        
        zoom_reset_btn = ttk.Button(zoom_frame, text="100%", width=6, command=self.zoom_reset)
        zoom_reset_btn.pack(side=tk.LEFT, padx=5)
        
        zoom_in_btn = ttk.Button(zoom_frame, text="+", width=3, command=self.zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=(5, 10))
        
        reset_pos_btn = ttk.Button(zoom_frame, text="Center", width=8, command=self.reset_position)
        reset_pos_btn.pack(side=tk.LEFT, padx=(10, 10))
        
        ttk.Label(zoom_frame, text="(Mouse wheel: zoom, Middle button: drag, Double-click: reset, Left click on red arrow or Shift+Click: detail analysis)").pack(side=tk.LEFT, padx=(20, 0))
        
        # Quality map and vector controls
        vector_frame = ttk.Frame(control_frame)
        vector_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gen_quality_btn = ttk.Button(vector_frame, text="Generate Quality Map",
                                          command=self.compute_current_quality_map)
        self.gen_quality_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.gen_turbulence_btn = ttk.Button(vector_frame, text="Generate Turbulence Map",
                                             command=self.compute_current_turbulence_map)
        self.gen_turbulence_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Third panel display mode
        ttk.Label(vector_frame, text="Third Panel:").pack(side=tk.LEFT, padx=(20, 5))
        self.third_panel_mode = tk.StringVar(value="quality")
        quality_radio = ttk.Radiobutton(vector_frame, text="Quality", variable=self.third_panel_mode, 
                                        value="quality", command=self.update_display)
        quality_radio.pack(side=tk.LEFT)
        turbulence_radio = ttk.Radiobutton(vector_frame, text="Turbulence", variable=self.third_panel_mode, 
                                           value="turbulence", command=self.update_display)
        turbulence_radio.pack(side=tk.LEFT)
        
        # Vector mode controls
        ttk.Label(vector_frame, text="Flow Vectors:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.vector_mode_var = tk.StringVar(value="original")
        original_radio = ttk.Radiobutton(vector_frame, text="Original", variable=self.vector_mode_var, 
                                        value="original", command=self.on_vector_mode_change)
        original_radio.pack(side=tk.LEFT, padx=(0, 5))
        
        # Check if LOD data is available before showing LOD option
        lod_available = self.check_lod_availability()
        if lod_available:
            lod_radio = ttk.Radiobutton(vector_frame, text="LOD", variable=self.vector_mode_var, 
                                       value="lod", command=self.on_vector_mode_change)
            lod_radio.pack(side=tk.LEFT, padx=(0, 10))
            
            # LOD level selection
            ttk.Label(vector_frame, text="LOD Level:").pack(side=tk.LEFT, padx=(10, 5))
            
            self.lod_level_var = tk.IntVar(value=0)
            lod_scale = ttk.Scale(vector_frame, from_=0, to=self.max_lod_levels-1, 
                                 orient=tk.HORIZONTAL, variable=self.lod_level_var,
                                 command=self.on_lod_level_change, length=100)
            lod_scale.pack(side=tk.LEFT, padx=(0, 5))
            
            self.lod_level_label = ttk.Label(vector_frame, text="0")
            self.lod_level_label.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.mouse_label = ttk.Label(status_frame, text="")
        self.mouse_label.pack(side=tk.RIGHT)
        
        self.zoom_label = ttk.Label(status_frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.RIGHT, padx=(20, 10))
        
        # Exit detail analysis button (initially hidden)
        self.exit_detail_btn = ttk.Button(status_frame, text="Exit Detail Analysis", 
                                         command=self.exit_detail_analysis_mode)
        # Don't pack initially - will be shown only in detail mode
        
        # Correction frame
        correction_frame = ttk.Frame(control_frame)
        correction_frame.pack(fill=tk.X, pady=(10, 0))

        # First row - basic correction buttons
        correction_buttons_frame = ttk.Frame(correction_frame)
        correction_buttons_frame.pack(fill=tk.X, pady=(0, 5))

        self.correct_errors_btn = ttk.Button(correction_buttons_frame, text="Correct Current Frame",
                                             command=self.correct_all_errors)
        self.correct_errors_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.correct_all_frames_btn = ttk.Button(correction_buttons_frame, text="Correct All Frames",
                                                 command=self.correct_all_frames_sequentially)
        self.correct_all_frames_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Add a separator
        ttk.Separator(correction_buttons_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=(10,10), fill='y')

        # Second row - range correction
        correction_range_frame = ttk.Frame(correction_frame)
        correction_range_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(correction_range_frame, text="Correct frames from:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.start_frame_var = tk.IntVar(value=0)
        self.start_frame_entry = ttk.Entry(correction_range_frame, textvariable=self.start_frame_var, width=8)
        self.start_frame_entry.pack(side=tk.LEFT, padx=(0, 2))

        # Quick fill button for start frame
        ttk.Button(correction_range_frame, text="Current", width=8, 
                  command=self.set_start_to_current).pack(side=tk.LEFT, padx=(2, 5))

        ttk.Label(correction_range_frame, text="to:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.end_frame_var = tk.IntVar(value=min(99, self.max_pairs - 1))
        self.end_frame_entry = ttk.Entry(correction_range_frame, textvariable=self.end_frame_var, width=8)
        self.end_frame_entry.pack(side=tk.LEFT, padx=(0, 2))

        # Quick fill button for end frame
        ttk.Button(correction_range_frame, text="Current", width=8, 
                  command=self.set_end_to_current).pack(side=tk.LEFT, padx=(2, 10))

        self.correct_range_btn = ttk.Button(correction_range_frame, text="Correct Range",
                                           command=self.correct_frames_range)
        self.correct_range_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Quick range button
        ttk.Button(correction_range_frame, text="Current→End", width=12, 
                  command=self.set_current_to_end_range).pack(side=tk.LEFT, padx=(5, 10))

        # Add help text
        ttk.Label(correction_range_frame, text=f"(0-{self.max_pairs - 1})", foreground="gray").pack(side=tk.LEFT)

        # Third row - other controls
        correction_other_frame = ttk.Frame(correction_frame)
        correction_other_frame.pack(fill=tk.X)

        self.run_taa_btn = ttk.Button(correction_other_frame, text="Run TAA with Corrected Flow", command=self.run_taa_processor)
        self.run_taa_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.save_corrected_var = tk.BooleanVar(value=True)
        save_check = ttk.Checkbutton(
            correction_other_frame,
            text="Save Corrected Flow",
            variable=self.save_corrected_var,
        )
        save_check.pack(side=tk.LEFT, padx=(10, 5))

        self.multithreaded_var = tk.BooleanVar(value=False)
        multithread_check = ttk.Checkbutton(
            correction_other_frame,
            text="Multithreaded",
            variable=self.multithreaded_var,
        )
        multithread_check.pack(side=tk.LEFT, padx=(10, 5))

        self.highlight_errors_var = tk.BooleanVar(value=False)
        highlight_check = ttk.Checkbutton(
            correction_other_frame,
            text="Highlight Correctable Errors",
            variable=self.highlight_errors_var,
            command=self.toggle_error_highlighting
        )
        highlight_check.pack(side=tk.LEFT, padx=(10, 5))



        # Fourth row - progress
        correction_progress_frame = ttk.Frame(correction_frame)
        correction_progress_frame.pack(fill=tk.X, pady=(5, 0))

        self.progress_label = ttk.Label(correction_progress_frame, text="")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5))

        self.progressbar = ttk.Progressbar(correction_progress_frame, orient=tk.HORIZONTAL,
                                           length=300, mode='determinate')
        self.progressbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def resize_frame_for_display(self, frame):
        """Resize frame to fit display with zoom while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not initialized yet, use default
            canvas_width = 800
            canvas_height = 600
        
        # Reserve space for controls and spacing (three frames + spacing + margins)
        available_width = canvas_width - 40  # 20px margin on each side
        available_height = (canvas_height - 150) // 3  # Space for three frames plus controls
        
        # Calculate base scale to fit width
        base_scale_w = available_width / w
        base_scale_h = available_height / h
        base_scale = min(base_scale_w, base_scale_h)
        
        # Apply zoom factor
        final_scale = base_scale * self.zoom_factor
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        # Ensure minimum size
        if new_w < 50 or new_h < 50:
            min_scale = max(50 / w, 50 / h)
            final_scale = min_scale
            new_w = int(w * final_scale)
            new_h = int(h * final_scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        return resized, final_scale
    
    def update_display(self):
        """Update the display with current frame pair"""
        if self.current_pair >= self.max_pairs:
            return
            
        # Load frames
        frame1 = self.frames[self.current_pair]
        frame2 = self.frames[self.current_pair + 1]
        
        # Store original frames for color comparison
        self.frame1 = frame1
        self.frame2 = frame2
        
        # Load flow data
        self.current_flow = self.load_flow_data(self.current_pair)
        
        # Get quality frame
        if self.current_pair in self.quality_maps:
            quality_frame = self.quality_maps[self.current_pair]
        else:
            # If not cached, show a black frame. User must generate it manually.
            quality_frame = np.zeros_like(frame1)
        
        # Determine what to display in the third panel
        third_panel_mode = self.third_panel_mode.get()
        if third_panel_mode == "turbulence":
            third_panel_image = self.turbulence_maps.get(self.current_pair, np.zeros_like(frame1))
            third_panel_title = "Flow Turbulence Map"
        else: # Default to quality map
            third_panel_image = quality_frame
            third_panel_title = "Flow Quality Map"
        
        # Highlight errors if the toggle is active
        if hasattr(self, 'highlight_errors_var') and self.highlight_errors_var.get() and self.current_pair in self.quality_maps:
            quality_frame = quality_frame.copy()  # Work on a copy
            # Highlight all correctable errors (red component > 0) as white
            error_pixels_y, error_pixels_x = np.where(quality_frame[:, :, 0] > 0)
            quality_frame[error_pixels_y, error_pixels_x] = [255, 255, 255]

            # Then, highlight improved pixels as yellow
            improved_pixels = self.improved_correction_pixels.get(self.current_pair)
            if improved_pixels:
                if improved_pixels: # Ensure not empty
                    improved_y = [coord[1] for coord in improved_pixels]
                    improved_x = [coord[0] for coord in improved_pixels]
                    quality_frame[np.array(improved_y), np.array(improved_x)] = [255, 255, 0] # Yellow

            # Finally, re-color pixels that failed previous correction as purple
            failed_pixels = self.failed_correction_pixels.get(self.current_pair)
            if failed_pixels:
                if failed_pixels: # Ensure not empty
                    failed_y = [coord[1] for coord in failed_pixels]
                    failed_x = [coord[0] for coord in failed_pixels]
                    quality_frame[np.array(failed_y), np.array(failed_x)] = [255, 0, 128] # Purple

            # If we are in quality mode, use the highlighted frame
            if third_panel_mode == "quality":
                third_panel_image = quality_frame

        # Resize frames for display
        frame1_display, self.display_scale = self.resize_frame_for_display(frame1)
        frame2_display, _ = self.resize_frame_for_display(frame2)
        third_panel_display, _ = self.resize_frame_for_display(third_panel_image)
        
        # Store original frame dimensions for coordinate mapping
        self.orig_height, self.orig_width = frame1.shape[:2]
        self.display_height, self.display_width = frame1_display.shape[:2]
        
        # Convert to PIL Images
        pil_frame1 = Image.fromarray(frame1_display)
        pil_frame2 = Image.fromarray(frame2_display)
        pil_third_panel = Image.fromarray(third_panel_display)
        
        # Convert to PhotoImage
        self.photo1 = ImageTk.PhotoImage(pil_frame1)
        self.photo2 = ImageTk.PhotoImage(pil_frame2)
        self.photo_third_panel = ImageTk.PhotoImage(pil_third_panel)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate positions for centering with pan offset
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not initialized yet
            self.root.after(100, self.update_display)
            return
            
        x_offset = max(0, (canvas_width - self.display_width) // 2) + self.pan_offset_x
        
        # Position frames vertically with pan offset
        y1_offset = 10 + self.pan_offset_y
        y2_offset = y1_offset + self.display_height + 20
        y3_offset = y2_offset + self.display_height + 20
        
        # Store positions for mouse coordinate mapping
        self.frame1_x = x_offset
        self.frame1_y = y1_offset
        self.frame2_x = x_offset
        self.frame2_y = y2_offset
        self.frame3_x = x_offset
        self.frame3_y = y3_offset
        
        # Draw frames
        self.canvas.create_image(x_offset, y1_offset, anchor=tk.NW, image=self.photo1, tags="frame1")
        self.canvas.create_image(x_offset, y2_offset, anchor=tk.NW, image=self.photo2, tags="frame2")
        self.canvas.create_image(x_offset, y3_offset, anchor=tk.NW, image=self.photo_third_panel, tags="frame3")
        
        # Draw flow vectors as white lines over first frame
        self.draw_flow_vectors(x_offset, y1_offset)
        
        # Add frame labels (top-left corner)
        frame1_num = self.start_frame + self.current_pair
        frame2_num = self.start_frame + self.current_pair + 1
        
        self.canvas.create_text(x_offset, y1_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair} (#{frame1_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y2_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair + 1} (#{frame2_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y3_offset - 5, anchor=tk.SW, 
                               text=third_panel_title, fill="white", font=("Arial", 10))
        
        # Add frame numbers in corners
        # Top-left corners
        self.canvas.create_text(x_offset + 5, y1_offset + 5, anchor=tk.NW, 
                               text=str(frame1_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y2_offset + 5, anchor=tk.NW, 
                               text=str(frame2_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y3_offset + 5, anchor=tk.NW, 
                               text="Info", fill="yellow", font=("Arial", 12, "bold"))
        
        # Top-right corners  
        self.canvas.create_text(x_offset + self.display_width - 5, y1_offset + 5, anchor=tk.NE, 
                               text=f"Pair {self.current_pair}", fill="cyan", font=("Arial", 10, "bold"))
        self.canvas.create_text(x_offset + self.display_width - 5, y2_offset + 5, anchor=tk.NE, 
                               text=f"Next", fill="cyan", font=("Arial", 10, "bold"))
        
        if self.detail_analysis_mode:
            quality_text = "Detail Analysis Mode"
        else:
            if third_panel_mode == "turbulence":
                quality_text = "Hot=Turbulent, Cold=Laminar"
            else:
                quality_text = f"Red=Bad, Green=Good"
        
        self.canvas.create_text(x_offset + self.display_width - 5, y3_offset + 5, anchor=tk.NE, 
                               text=quality_text, fill="cyan", font=("Arial", 10, "bold"))
        
        # Redraw the quality map marker if it exists, regardless of mode
        if self.quality_map_marker_coords:
            radius = 5
            x, y = self.quality_map_marker_coords
            self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                outline="yellow", width=2, tags=self.quality_map_marker_tag
            )

        # Draw detail analysis overlays if in detail mode
        if self.detail_analysis_mode:
            self.draw_detail_analysis_overlays(x_offset, y1_offset, x_offset, y2_offset)
            self.draw_detail_analysis_view()
        
        # Update labels
        self.frame_label.config(text=f"{self.current_pair}/{self.max_pairs-1}")
        
        # Update status with current mode info
        if self.current_flow is not None:
            flow_status = f"Flow loaded: {self.current_flow.shape}"
            
            # Detail analysis mode status
            if self.detail_analysis_mode:
                data = self.detail_analysis_data
                px, py = data['source_pixel']
                analysis_status = f"Detail Analysis: Pixel ({px},{py}), LOD{data['lod_level']}, Confidence={data['confidence']:.3f}"
                self.status_label.config(text=f"{flow_status} | {analysis_status}")
                return
            
            # Quality map status
            if self.current_pair in self.quality_maps:
                quality_status = "Quality map: Generated"
            else:
                quality_status = "Quality map: Not Generated"
            
            # Vector mode status
            if self.use_lod_for_vectors:
                vector_status = f"Vectors: LOD{self.current_lod_level}"
            else:
                vector_status = "Vectors: Original"
            
            self.status_label.config(text=f"{flow_status} | {quality_status} | {vector_status}")
        else:
            self.status_label.config(text="No flow data")
    
    def update_display_quick(self):
        """Quick update display without quality map computation (for dragging)"""
        if self.current_pair >= self.max_pairs:
            return
            
        # Load frames
        frame1 = self.frames[self.current_pair]
        frame2 = self.frames[self.current_pair + 1]
        # Store original frames for color comparison
        self.frame1 = frame1
        self.frame2 = frame2
        
        # Load flow data
        self.current_flow = self.load_flow_data(self.current_pair)
        
        # Use black frame for quality (no computation during drag, or if disabled)
        quality_frame = np.zeros_like(frame1)
        
        # Resize frames for display
        frame1_display, self.display_scale = self.resize_frame_for_display(frame1)
        frame2_display, _ = self.resize_frame_for_display(frame2)
        quality_display, _ = self.resize_frame_for_display(quality_frame)
        
        # Store original frame dimensions for coordinate mapping
        self.orig_height, self.orig_width = frame1.shape[:2]
        self.display_height, self.display_width = frame1_display.shape[:2]
        
        # Convert to PIL Images
        pil_frame1 = Image.fromarray(frame1_display)
        pil_frame2 = Image.fromarray(frame2_display)
        pil_quality = Image.fromarray(quality_display)
        
        # Convert to PhotoImage
        self.photo1 = ImageTk.PhotoImage(pil_frame1)
        self.photo2 = ImageTk.PhotoImage(pil_frame2)
        self.photo_quality = ImageTk.PhotoImage(pil_quality)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate positions for centering with pan offset
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not initialized yet
            self.root.after(100, self.update_display_quick)
            return
            
        x_offset = max(0, (canvas_width - self.display_width) // 2) + self.pan_offset_x
        
        # Position frames vertically with pan offset
        y1_offset = 10 + self.pan_offset_y
        y2_offset = y1_offset + self.display_height + 20
        y3_offset = y2_offset + self.display_height + 20
        
        # Store positions for mouse coordinate mapping
        self.frame1_x = x_offset
        self.frame1_y = y1_offset
        self.frame2_x = x_offset
        self.frame2_y = y2_offset
        self.frame3_x = x_offset
        self.frame3_y = y3_offset
        
        # Draw frames
        self.canvas.create_image(x_offset, y1_offset, anchor=tk.NW, image=self.photo1, tags="frame1")
        self.canvas.create_image(x_offset, y2_offset, anchor=tk.NW, image=self.photo2, tags="frame2")
        self.canvas.create_image(x_offset, y3_offset, anchor=tk.NW, image=self.photo_quality, tags="frame3")
        
        # Draw flow vectors as white lines over first frame
        self.draw_flow_vectors(x_offset, y1_offset)
        
        # Add frame labels (top-left corner)
        frame1_num = self.start_frame + self.current_pair
        frame2_num = self.start_frame + self.current_pair + 1
        
        self.canvas.create_text(x_offset, y1_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair} (#{frame1_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y2_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair + 1} (#{frame2_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y3_offset - 5, anchor=tk.SW, 
                               text=f"Flow Quality Map", fill="white", font=("Arial", 10))
        
        # Add frame numbers in corners
        # Top-left corners
        self.canvas.create_text(x_offset + 5, y1_offset + 5, anchor=tk.NW, 
                               text=str(frame1_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y2_offset + 5, anchor=tk.NW, 
                               text=str(frame2_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y3_offset + 5, anchor=tk.NW, 
                               text="Quality", fill="yellow", font=("Arial", 12, "bold"))
        
        # Top-right corners  
        self.canvas.create_text(x_offset + self.display_width - 5, y1_offset + 5, anchor=tk.NE, 
                               text=f"Pair {self.current_pair}", fill="cyan", font=("Arial", 10, "bold"))
        self.canvas.create_text(x_offset + self.display_width - 5, y2_offset + 5, anchor=tk.NE, 
                               text=f"Next", fill="cyan", font=("Arial", 10, "bold"))
        drag_text = f"Dragging... (Quality Map Paused)"
        self.canvas.create_text(x_offset + self.display_width - 5, y3_offset + 5, anchor=tk.NE, 
                               text=drag_text, fill="orange", font=("Arial", 10, "bold"))
        
        # Update labels
        self.frame_label.config(text=f"{self.current_pair}/{self.max_pairs-1}")
        
        # Update status for dragging
        if self.current_flow is not None:
            vector_status = f"Vectors: LOD{self.current_lod_level}" if self.use_lod_for_vectors else "Vectors: Original"
            self.status_label.config(text=f"Flow loaded: {self.current_flow.shape} | Quality map: Paused (dragging) | {vector_status}")
        else:
            self.status_label.config(text="No flow data")
    
    def on_slider_press(self, event):
        """Handle slider press (start of drag)"""
        self.slider_dragging = True
    
    def on_slider_release(self, event):
        """Handle slider release (end of drag)"""
        self.slider_dragging = False
        # Trigger update after drag ends
        self.update_display()
    
    def on_slider_change(self, val):
        """Called when the frame slider is moved."""
        new_pair = int(float(val))

        if new_pair == self.current_pair:
            return # Avoid unnecessary updates

        self.current_pair = new_pair

        # Only update display during dragging, no computations
        self.update_display_quick()
    
    def on_mouse_move(self, event):
        """Handle mouse movement over canvas"""
        # Guard against calls before the first display update
        if not hasattr(self, 'frame1_x'):
            return

        # Clear previous arrow (but keep flow vectors)
        self.canvas.delete("flow_arrow")
        
        if self.current_flow is None:
            return
            
        # Check if mouse is over first frame (accounting for pan offset)
        if (self.frame1_x <= event.x <= self.frame1_x + self.display_width and
            self.frame1_y <= event.y <= self.frame1_y + self.display_height):
            
            # Convert display coordinates to original frame coordinates
            display_x = event.x - self.frame1_x
            display_y = event.y - self.frame1_y
            
            orig_x = int(display_x / self.display_scale)
            orig_y = int(display_y / self.display_scale)
            
            # Clamp to frame bounds
            orig_x = max(0, min(orig_x, self.orig_width - 1))
            orig_y = max(0, min(orig_y, self.orig_height - 1))
            
            # Get flow vector, handling different flow and frame resolutions
            if self.current_flow is not None:
                fh, fw = self.current_flow.shape[:2]
                
                # Calculate scale factors
                scale_x = fw / self.orig_width
                scale_y = fh / self.orig_height
                
                # Map to flow coordinates
                flow_x_coord = int(orig_x * scale_x)
                flow_y_coord = int(orig_y * scale_y)
                
                # Clamp to flow bounds
                flow_x_coord = max(0, min(flow_x_coord, fw - 1))
                flow_y_coord = max(0, min(flow_y_coord, fh - 1))
                
                # Get flow vector and scale it back to frame resolution
                flow_x = self.current_flow[flow_y_coord, flow_x_coord, 0] / scale_x
                flow_y = self.current_flow[flow_y_coord, flow_x_coord, 1] / scale_y
                
                # Calculate target position in original coordinates
                target_orig_x = orig_x - flow_x
                target_orig_y = orig_y - flow_y
                
                # Convert source pixel position to canvas coordinates on first frame
                source_display_x = orig_x * self.display_scale
                source_display_y = orig_y * self.display_scale
                canvas_source_x = self.frame1_x + source_display_x
                canvas_source_y = self.frame1_y + source_display_y
                
                # Convert target position to display coordinates on second frame
                target_display_x = target_orig_x * self.display_scale
                target_display_y = target_orig_y * self.display_scale
                canvas_target_x = self.frame2_x + target_display_x
                canvas_target_y = self.frame2_y + target_display_y
                
                # Check color consistency to determine arrow color and get details
                arrow_color, src_color, target_color, similarity = self.get_arrow_color_details(orig_x, orig_y, target_orig_x, target_orig_y)
                
                # Draw arrow from exact pixel on first frame to corresponding pixel on second frame
                self.draw_arrow(canvas_source_x, canvas_source_y, canvas_target_x, canvas_target_y, arrow_color)
                
                # Update mouse info in status bar (general info)
                self.mouse_label.config(
                    text=(f"Pixel ({orig_x},{orig_y}) → Flow ({flow_x:.1f},{flow_y:.1f}) | Similarity: {similarity:.3f}")
                )
                
                # --- Draw detailed color info with squares next to the arrow ---
                info_box_x = canvas_source_x + 20
                info_box_y = canvas_source_y
                square_size = 32

                # Convert RGB to hex for tkinter
                src_hex_color = f"#{src_color[0]:02x}{src_color[1]:02x}{src_color[2]:02x}"
                tgt_hex_color = f"#{target_color[0]:02x}{target_color[1]:02x}{target_color[2]:02x}"
                
                # Draw source color square
                self.canvas.create_rectangle(
                    info_box_x, info_box_y, 
                    info_box_x + square_size, info_box_y + square_size,
                    fill=src_hex_color, outline="white", width=1, tags="flow_arrow"
                )

                # Draw target color square
                self.canvas.create_rectangle(
                    info_box_x + square_size + 5, info_box_y, 
                    info_box_x + square_size * 2 + 5, info_box_y + square_size,
                    fill=tgt_hex_color, outline="white", width=1, tags="flow_arrow"
                )
                
                # Draw similarity text below the squares
                text_y = info_box_y + square_size + 5
                text_x_centered = info_box_x + (square_size * 2 + 5) / 2
                info_text = f"{similarity:.3f}"
                
                # Background for text
                self.canvas.create_text(text_x_centered + 1, text_y + 1, text=info_text, 
                                        anchor=tk.N, fill="black", font=("Arial", 10, "bold"), 
                                        tags="flow_arrow")
                # Foreground text
                self.canvas.create_text(text_x_centered, text_y, text=info_text, 
                                        anchor=tk.N, fill="white", font=("Arial", 10, "bold"), 
                                        tags="flow_arrow")
                
                self.canvas.tag_raise("flow_arrow") # Ensure it's drawn on top

            else:
                self.mouse_label.config(text="")
        else:
            self.mouse_label.config(text="")
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving canvas"""
        self.canvas.delete("flow_arrow")
        self.mouse_label.config(text="")
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        # Determine scroll direction
        if event.delta > 0 or event.num == 4:  # Scroll up / zoom in
            zoom_change = self.zoom_step
        elif event.delta < 0 or event.num == 5:  # Scroll down / zoom out
            zoom_change = -self.zoom_step
        else:
            return
        
        # Update zoom factor
        new_zoom = self.zoom_factor + zoom_change
        new_zoom = max(self.min_zoom, min(new_zoom, self.max_zoom))
        
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
            self.update_display()
    
    def on_double_click(self, event):
        """Handle double-click to reset zoom and position"""
        changed = False
        if self.zoom_factor != 1.0:
            self.zoom_factor = 1.0
            self.zoom_label.config(text="Zoom: 100%")
            changed = True
        if self.pan_offset_x != 0 or self.pan_offset_y != 0:
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            changed = True
        if changed:
            self.update_display()
    
    def zoom_in(self):
        """Zoom in by zoom_step"""
        new_zoom = min(self.zoom_factor + self.zoom_step, self.max_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
            self.update_display()
    
    def zoom_out(self):
        """Zoom out by zoom_step"""
        new_zoom = max(self.zoom_factor - self.zoom_step, self.min_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
            self.update_display()
    
    def zoom_reset(self):
        """Reset zoom to 100%"""
        if self.zoom_factor != 1.0:
            self.zoom_factor = 1.0
            self.zoom_label.config(text="Zoom: 100%")
            self.update_display()
    
    def reset_position(self):
        """Reset pan position to center"""
        if self.pan_offset_x != 0 or self.pan_offset_y != 0:
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self.update_display()
    
    def toggle_error_highlighting(self):
        """Toggles the highlighting of correctable errors on the quality map."""
        self.update_display()

    def on_vector_mode_change(self):
        """Handle vector mode radio button change"""
        mode = self.vector_mode_var.get()
        self.use_lod_for_vectors = (mode == "lod")
        self.update_display()
    
    def on_lod_level_change(self, value):
        """Handle LOD level slider change"""
        self.current_lod_level = int(float(value))
        if hasattr(self, 'lod_level_label'):
            self.lod_level_label.config(text=str(self.current_lod_level))
        if self.use_lod_for_vectors:
            self.update_display()
    
    def on_window_resize(self, event):
        """Handle window resize to update frame display"""
        # Only handle resize events for the main window, not child widgets
        if event.widget == self.root:
            # Delay update to avoid too frequent redraws during resize
            if hasattr(self, '_resize_timer'):
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(200, self.update_display)
    
    def on_middle_click(self, event):
        """Handle middle mouse button press to start panning"""
        self.is_panning = True
        self.last_pan_x = event.x
        self.last_pan_y = event.y
        self.canvas.config(cursor="fleur")  # Change cursor to indicate panning
    
    def on_middle_release(self, event):
        """Handle middle mouse button release to stop panning"""
        self.is_panning = False
        self.canvas.config(cursor="")  # Reset cursor
    
    def on_middle_drag(self, event):
        """Handle middle mouse drag for panning"""
        if self.is_panning:
            # Calculate drag distance
            dx = event.x - self.last_pan_x
            dy = event.y - self.last_pan_y
            
            # Update pan offset
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            
            # Update last position
            self.last_pan_x = event.x
            self.last_pan_y = event.y
            
            # Update display
            self.update_display()
    
    def calculate_pixel_quality(self, src_color, target_color):
        """Calculate quality score for a single pixel pair"""
        return correction_worker.calculate_pixel_quality(src_color, target_color)
    
    def generate_quality_frame(self, frame1, frame2, flow):
        """Generate a quality visualization frame"""
        if flow is None:
            # Return black frame if no flow data
            return np.zeros_like(frame1)
        
        h, w = frame1.shape[:2]
        fh, fw = flow.shape[:2]
        quality_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate scale factors if flow and frame dimensions don't match
        scale_x = w / fw if fw > 0 else 1.0
        scale_y = h / fh if fh > 0 else 1.0
        
        # Resize flow to match frame dimensions if needed
        if (fh != h or fw != w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            # Scale flow vectors by the resize ratio
            flow[:, :, 0] *= scale_x
            flow[:, :, 1] *= scale_y
        
        # Vectorized approach for better performance
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Get flow vectors
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        
        # Calculate target positions
        target_x = x_coords - flow_x
        target_y = y_coords - flow_y
        
        # Create masks for valid targets
        valid_mask = ((target_x >= 0) & (target_x < w) & 
                     (target_y >= 0) & (target_y < h))
        
        # Process valid pixels
        valid_y, valid_x = np.where(valid_mask)
        
        if len(valid_y) > 0:
            # Get source colors
            src_colors = frame1[valid_y, valid_x]
            
            # Get target coordinates (rounded to integers)
            tgt_x = np.round(target_x[valid_y, valid_x]).astype(int)
            tgt_y = np.round(target_y[valid_y, valid_x]).astype(int)
            
            # Ensure target coordinates are within bounds
            tgt_x = np.clip(tgt_x, 0, w-1)
            tgt_y = np.clip(tgt_y, 0, h-1)
            
            # Get target colors
            tgt_colors = frame2[tgt_y, tgt_x]
            
            # Calculate similarities for all valid pixels at once
            similarities = []
            for i in range(len(valid_y)):
                sim = self.calculate_pixel_quality(src_colors[i], tgt_colors[i])
                similarities.append(sim)
            similarities = np.array(similarities)
            
            # Color coding
            good_mask = similarities > self.GOOD_QUALITY_THRESHOLD
            
            # Good quality pixels - green
            good_indices = np.where(good_mask)[0]
            if len(good_indices) > 0:
                intensities = (255 * (similarities[good_indices] - 0.5) * 2).astype(int) # Scale green intensity
                quality_frame[valid_y[good_indices], valid_x[good_indices]] = np.column_stack([
                    np.zeros_like(intensities), np.clip(intensities, 0, 255), np.zeros_like(intensities)
                ])
            
            # Bad quality pixels - red
            bad_indices = np.where(~good_mask)[0]
            if len(bad_indices) > 0:
                intensities = (255 * (1.0 - similarities[bad_indices])).astype(int)
                quality_frame[valid_y[bad_indices], valid_x[bad_indices]] = np.column_stack([
                    intensities, np.zeros_like(intensities), np.zeros_like(intensities)
                ])
        
        # Out of bounds pixels - red
        invalid_y, invalid_x = np.where(~valid_mask)
        if len(invalid_y) > 0:
            quality_frame[invalid_y, invalid_x] = [255, 0, 0]
        
        return quality_frame
    
    def generate_quality_frame_fast(self, frame1, frame2, flow):
        """Generate a quality visualization frame - optimized version"""
        return correction_worker.generate_quality_frame_fast(frame1, frame2, flow, self.GOOD_QUALITY_THRESHOLD)
    
    def get_arrow_color(self, src_x, src_y, target_x, target_y):
        """Determine arrow color based on color consistency between source and target pixels"""
        color, _, _, _ = self.get_arrow_color_details(src_x, src_y, target_x, target_y)
        return color

    def get_arrow_color_details(self, src_x, src_y, target_x, target_y):
        """Get arrow color and detailed color information for the hover label."""
        # Default values for out-of-bounds cases
        default_color = "red"
        default_src_color = (0,0,0)
        default_tgt_color = (0,0,0)
        default_similarity = 0.0

        # Check bounds for both frames
        if (src_x < 0 or src_x >= self.orig_width or src_y < 0 or src_y >= self.orig_height or
            target_x < 0 or target_x >= self.orig_width or target_y < 0 or target_y >= self.orig_height):
            return default_color, default_src_color, default_tgt_color, default_similarity
        
        # Get pixel colors from both frames
        src_color = self.frame1[int(src_y), int(src_x)]
        target_color = self.frame2[int(target_y), int(target_x)]
        
        # Calculate quality using shared method
        overall_similarity = self.calculate_pixel_quality(src_color, target_color)
        
        # Threshold using the centralized function
        arrow_color = "green" if self._is_good_quality(overall_similarity) else "red"
        
        return arrow_color, tuple(src_color), tuple(target_color), overall_similarity

    def draw_flow_vectors(self, frame_x_offset, frame_y_offset):
        """Draw flow vectors as white lines over the first frame"""
        # Choose flow data source based on current mode
        if self.use_lod_for_vectors:
            # Use LOD data
            lod_flow = self.load_lod_data(self.current_pair, self.current_lod_level)
            if lod_flow is None:
                return  # No LOD data available
            flow_data = lod_flow
        else:
            # Use original flow data
            if self.current_flow is None:
                return
            flow_data = self.current_flow
        
        # Step size based on mode
        if self.use_lod_for_vectors:
            # For LOD: use step of 20 pixels from frame size
            frame_step = 20
            # Get LOD flow dimensions
            lod_h, lod_w = flow_data.shape[:2]
            # Calculate step in LOD coordinates
            lod_scale_x = lod_w / self.orig_width
            lod_scale_y = lod_h / self.orig_height
            step_x = max(1, int(frame_step * lod_scale_x))
            step_y = max(1, int(frame_step * lod_scale_y))
        else:
            # For original flow: use uniform step in flow coordinates
            step_x = step_y = 25
        
        # Get flow dimensions
        flow_h, flow_w = flow_data.shape[:2]
        
        # Calculate scale factors between flow and frame
        flow_to_frame_scale_x = self.orig_width / flow_w
        flow_to_frame_scale_y = self.orig_height / flow_h
        
        # Draw vectors with step size
        for y in range(0, flow_h, step_y):
            for x in range(0, flow_w, step_x):
                # Get flow vector
                flow_x = flow_data[y, x, 0]
                flow_y = flow_data[y, x, 1]
                
                # Skip very small vectors
                magnitude = np.sqrt(flow_x**2 + flow_y**2)
                if magnitude < 0.3:
                    continue
                
                # Convert flow coordinates to frame coordinates
                frame_x = x * flow_to_frame_scale_x
                frame_y = y * flow_to_frame_scale_y
                
                # Convert to display coordinates
                start_x = frame_x_offset + frame_x * self.display_scale
                start_y = frame_y_offset + frame_y * self.display_scale
                
                # Scale flow vector to frame coordinates, then to display coordinates
                scaled_flow_x = flow_x * flow_to_frame_scale_x
                scaled_flow_y = flow_y * flow_to_frame_scale_y
                end_x = start_x + scaled_flow_x * self.display_scale
                end_y = start_y + scaled_flow_y * self.display_scale
                
                # Draw flow vector as white line
                self.canvas.create_line(start_x, start_y, end_x, end_y, 
                                      fill="white", width=1, tags="flow_vectors")
                
                # Draw small circle at start point
                self.canvas.create_oval(start_x-1, start_y-1, start_x+1, start_y+1, 
                                      fill="white", outline="white", tags="flow_vectors")

    def draw_arrow(self, x1, y1, x2, y2, color="red", tags="flow_arrow"):
        """Draw arrow from (x1,y1) to (x2,y2) with specified color and tags"""
        # Clamp target to canvas bounds
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x2 = max(0, min(x2, canvas_width))
        y2 = max(0, min(y2, canvas_height))
        
        # Calculate arrow properties
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 2:  # Too short to draw
            return
            
        # Normalize
        if length > 0:
            dx /= length
            dy /= length
        
        # Arrow head size
        head_length = min(10, length * 0.2)
        head_angle = 0.5  # radians
        
        # Arrow head points
        head_x1 = x2 - head_length * (dx * np.cos(head_angle) - dy * np.sin(head_angle))
        head_y1 = y2 - head_length * (dx * np.sin(head_angle) + dy * np.cos(head_angle))
        head_x2 = x2 - head_length * (dx * np.cos(-head_angle) - dy * np.sin(-head_angle))
        head_y2 = y2 - head_length * (dx * np.sin(-head_angle) + dy * np.cos(-head_angle))
        
        # Draw arrow line
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags=tags)
        
        # Draw arrow head
        self.canvas.create_line(x2, y2, head_x1, head_y1, fill=color, width=2, tags=tags)
        self.canvas.create_line(x2, y2, head_x2, head_y2, fill=color, width=2, tags=tags)
        
        # Draw circle at start point
        self.canvas.create_oval(x1-3, y1-3, x1+3, y1+3, fill=color, outline="white", tags=tags)
    
    def draw_detail_analysis_view(self):
        """Draw the detail analysis panel directly on the canvas, combining the best of all previous versions."""
        if not self.detail_analysis_data:
            return

        canvas = self.canvas
        data = self.detail_analysis_data
        
        # --- 1. Sizing and Scaling ---
        display_size = 200  # Size for each of the three panels
        border_size = 2
        
        # Helper to draw a square marker for a pixel
        def draw_pixel_marker(image, coords, color, size=2):
            # Coordinates are relative to the image top-left
            cv2.rectangle(image, (coords[0] - size//2, coords[1] - size//2), (coords[0] + size//2, coords[1] + size//2), color, -1)
        
        # Helper to draw a thin rectangle frame
        def draw_thin_frame(image, top_left, bottom_right, color):
            cv2.rectangle(image, top_left, bottom_right, color, 1)

        # --- 2. Prepare Source Panel ---
        source_region, _ = self.extract_region(self.frame1, data['source_pixel'][0], data['source_pixel'][1], self.detail_analysis_region_size)
        source_scaled = cv2.resize(source_region, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        
        # Draw center pixel marker
        center_coords_scaled = (display_size // 2, display_size // 2)
        draw_pixel_marker(source_scaled, center_coords_scaled, (255, 255, 0), size=4) # Yellow
        
        # Draw template frame
        template_w_scaled = int(self.template_radius * 2 * (display_size / (self.detail_analysis_region_size * 2)))
        tl = (center_coords_scaled[0] - template_w_scaled // 2, center_coords_scaled[1] - template_w_scaled // 2)
        br = (center_coords_scaled[0] + template_w_scaled // 2, center_coords_scaled[1] + template_w_scaled // 2)
        draw_thin_frame(source_scaled, tl, br, (255, 255, 0)) # Yellow

        # --- 3. Prepare Target Panel ---
        target_region, target_bounds = self.extract_region(self.frame2, data['lod_target'][0], data['lod_target'][1], self.detail_analysis_region_size)
        target_scaled = cv2.resize(target_region, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

        def absolute_to_scaled_local(abs_coords):
            """Converts absolute frame coordinates to local coordinates in the scaled panel."""
            if abs_coords is None or abs_coords[0] is None: return None
            scale = display_size / (self.detail_analysis_region_size * 2)
            local_x = (abs_coords[0] - target_bounds[0]) * scale
            local_y = (abs_coords[1] - target_bounds[1]) * scale
            # Check if inside the scaled display
            if 0 <= local_x < display_size and 0 <= local_y < display_size:
                return (int(local_x), int(local_y))
            return None

        # Draw pixel markers for all stages
        marker_positions = {
            'original': (data['original_flow'], (255, 165, 0)),
            'lod': (data['lod_target'], (255, 0, 0)),
            'coarse': (data['coarse_result']['target'], (0, 255, 0)),
            'fine': (data['fine_result']['target'] if data['fine_result'] else None, (0, 255, 255))
        }
        for name, (coords, color) in marker_positions.items():
            scaled_pos = absolute_to_scaled_local(coords)
            if scaled_pos:
                draw_pixel_marker(target_scaled, scaled_pos, color, size=4)
        
        # Draw best match frame
        if data['fine_result']:
            fine_pos_scaled = absolute_to_scaled_local(data['fine_result']['target'])
            if fine_pos_scaled:
                template_w_scaled = int(self.template_radius * 2 * (display_size / (self.detail_analysis_region_size * 2)))
                half_w_scaled = template_w_scaled // 2
                tl = (fine_pos_scaled[0] - half_w_scaled, fine_pos_scaled[1] - half_w_scaled)
                br = (fine_pos_scaled[0] + half_w_scaled, fine_pos_scaled[1] + half_w_scaled)

                draw_thin_frame(target_scaled, tl, br, (0, 255, 255)) # Cyan

        # --- 4. Prepare Difference Panel ---
        coarse_res = data['coarse_result']
        dx, dy, angle = coarse_res['phase_shift'][0], coarse_res['phase_shift'][1], coarse_res.get('angle', 0)
        
        # Scale shift values to the display size
        scale_factor = display_size / (self.detail_analysis_region_size * 2)
        shift_dx = dx * scale_factor
        shift_dy = dy * scale_factor
        
        # Create transformation matrix
        center_scaled = (display_size / 2, display_size / 2)
        M_rot = cv2.getRotationMatrix2D(center_scaled, -angle, 1.0)
        M_trans = np.float32([[1, 0, shift_dx], [0, 1, shift_dy]])
        
        # Apply transformations to target region
        transformed_target = cv2.warpAffine(target_scaled, M_rot, (display_size, display_size))
        transformed_target = cv2.warpAffine(transformed_target, M_trans, (display_size, display_size))
        
        # Create a mask to handle non-overlapping areas
        mask = np.ones_like(target_scaled, dtype=np.uint8) * 255
        rotated_mask = cv2.warpAffine(mask, M_rot, (display_size, display_size), flags=cv2.INTER_NEAREST)
        transformed_mask = cv2.warpAffine(rotated_mask, M_trans, (display_size, display_size), flags=cv2.INTER_NEAREST)
        
        # Calculate difference and apply mask
        diff = cv2.absdiff(source_scaled, transformed_target)
        diff[transformed_mask < 255] = 0
        diff_panel = np.clip(diff * 2, 0, 255).astype(np.uint8)
        
        # Draw alignment frames on the difference panel
        draw_thin_frame(diff_panel, (0, 0), (display_size - 1, display_size - 1), (255, 255, 255)) # White
        
        src_pts = np.float32([[0,0], [display_size,0], [display_size,display_size], [0,display_size]]).reshape(-1,1,2)
        M_combined = np.dot(M_trans, np.vstack([M_rot, [0, 0, 1]]))
        transformed_pts = cv2.transform(src_pts, M_combined).astype(np.int32)
        cv2.polylines(diff_panel, [transformed_pts], True, (0, 255, 0), 1) # Green

        # --- 5. Add Borders & Finalize Images ---
        source_bordered = cv2.copyMakeBorder(source_scaled, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[100,100,100])
        target_bordered = cv2.copyMakeBorder(target_scaled, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[100,100,100])
        diff_bordered = cv2.copyMakeBorder(diff_panel, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[100,100,100])

        # --- 6. Canvas Drawing ---
        base_y = self.frame3_y + self.display_height + 40
        spacing = 20
        region_h, region_w = source_bordered.shape[:2]

        x1 = spacing
        x2 = x1 + region_w + spacing
        x3 = x2 + region_w + spacing
        x4 = x3 + region_w + spacing # New column for color swatches

        # Convert to PhotoImage and store references
        self.analysis_photo1 = ImageTk.PhotoImage(Image.fromarray(source_bordered))
        self.analysis_photo2 = ImageTk.PhotoImage(Image.fromarray(target_bordered))
        self.analysis_photo3 = ImageTk.PhotoImage(Image.fromarray(diff_bordered))

        canvas.create_image(x1, base_y, anchor=tk.NW, image=self.analysis_photo1, tags="detail_analysis")
        canvas.create_image(x2, base_y, anchor=tk.NW, image=self.analysis_photo2, tags="detail_analysis")
        canvas.create_image(x3, base_y, anchor=tk.NW, image=self.analysis_photo3, tags="detail_analysis")
        
        # --- 7. Draw Color Swatches Panel ---
        swatch_size = 50
        swatch_y_spacing = 15
        font_details = ("Arial", 8)
        
        def get_color_at(frame, coords):
            if coords is None or coords[0] is None: return None
            h, w = frame.shape[:2]
            x, y = int(coords[0]), int(coords[1])
            if 0 <= x < w and 0 <= y < h:
                return frame[y, x]
            return None

        def draw_swatch(x, y, label, color):
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" if color is not None else "#333333"
            outline_color = "white" if color is not None else "gray"
            canvas.create_rectangle(x, y, x + swatch_size, y + swatch_size, 
                                    fill=hex_color, outline=outline_color, tags="detail_analysis")
            canvas.create_text(x + swatch_size / 2, y + swatch_size + 8, 
                               text=label, anchor=tk.N, fill="white", 
                               font=font_details, tags="detail_analysis")

        # Get colors
        source_color = get_color_at(self.frame1, data['source_pixel'])
        coarse_color = get_color_at(self.frame2, data['coarse_result']['target'])
        fine_color = get_color_at(self.frame2, data['fine_result']['target'] if data['fine_result'] else None)
        
        # Draw swatches
        swatch_y = base_y
        draw_swatch(x4, swatch_y, "Source Pixel", source_color)
        
        swatch_y += swatch_size + swatch_y_spacing + 10
        draw_swatch(x4, swatch_y, "Coarse Target", coarse_color)
        
        if data['fine_result']:
            swatch_y += swatch_size + swatch_y_spacing + 10
            draw_swatch(x4, swatch_y, "Fine Target", fine_color)

        # --- 8. Labels and Legends ---

        legend_y_offset = base_y + region_h + 15
        
        canvas.create_text(x1, base_y - 5, text="Source Region (Template)", anchor=tk.SW, fill="white", tags="detail_analysis")
        canvas.create_text(x1, legend_y_offset,      text="■ Yellow Marker: Selected Pixel", anchor=tk.NW, fill="yellow", font=font_details, tags="detail_analysis")
        canvas.create_text(x1, legend_y_offset + 12, text="■ Yellow Frame: 11x11 Template Area", anchor=tk.NW, fill="yellow", font=font_details, tags="detail_analysis")
        
        canvas.create_text(x2, base_y - 5, text="Target Region (Search Area)", anchor=tk.SW, fill="white", tags="detail_analysis")
        canvas.create_text(x2, legend_y_offset,      text="■ Orange: Original Target", anchor=tk.NW, fill="orange", font=font_details, tags="detail_analysis")
        canvas.create_text(x2, legend_y_offset + 12, text="■ Red: LOD Target (Search Center)", anchor=tk.NW, fill="red", font=font_details, tags="detail_analysis")
        canvas.create_text(x2, legend_y_offset + 24, text="■ Green: Coarse Target (PhaseCorr)", anchor=tk.NW, fill="lime", font=font_details, tags="detail_analysis")
        
        if data['fine_result']:
            canvas.create_text(x2, legend_y_offset + 36, text="■ Cyan Marker: Fine Target (NCC)", anchor=tk.NW, fill="cyan", font=font_details, tags="detail_analysis")
            canvas.create_text(x2, legend_y_offset + 48, text="■ Cyan Frame: Best Match Found", anchor=tk.NW, fill="cyan", font=font_details, tags="detail_analysis")

        canvas.create_text(x3, base_y - 5, text="Coarse Alignment Difference", anchor=tk.SW, fill="white", tags="detail_analysis")
        canvas.create_text(x3, legend_y_offset,      text="■ White Frame: Source Position", anchor=tk.NW, fill="white", font=font_details, tags="detail_analysis")
        canvas.create_text(x3, legend_y_offset + 12, text="■ Green Frame: Target Aligned", anchor=tk.NW, fill="lime", font=font_details, tags="detail_analysis")
    
    def draw_detail_analysis_overlays(self, frame1_x, frame1_y, frame2_x, frame2_y):
        """Draw overlay vectors for detail analysis mode on the main frames."""
        if not self.detail_analysis_data:
            return

        data = self.detail_analysis_data
        source_x, source_y = data['source_pixel']

        # Calculate original target
        orig_flow_x, orig_flow_y = data['original_flow']
        orig_target_x = source_x - orig_flow_x
        orig_target_y = source_y - orig_flow_y
        
        # Get LOD target
        lod_target_x, lod_target_y = data['lod_target']
        
        # Get Coarse target
        coarse_target_x, coarse_target_y = data['coarse_result']['target']
        
        # Get Fine target if available
        fine_target_x, fine_target_y = (None, None)
        if data['fine_result']:
            fine_target_x, fine_target_y = data['fine_result']['target']

        def to_display_coords(orig_x, orig_y, frame_x_offset, frame_y_offset):
            if orig_x is None or orig_y is None:
                return None, None
            return (frame_x_offset + orig_x * self.display_scale,
                    frame_y_offset + orig_y * self.display_scale)
        
        # Convert points to canvas coordinates
        src_display_x, src_display_y = to_display_coords(source_x, source_y, frame1_x, frame1_y)
        orig_tgt_display_x, orig_tgt_display_y = to_display_coords(orig_target_x, orig_target_y, frame2_x, frame2_y)
        lod_tgt_display_x, lod_tgt_display_y = to_display_coords(lod_target_x, lod_target_y, frame2_x, frame2_y)
        coarse_tgt_display_x, coarse_tgt_display_y = to_display_coords(coarse_target_x, coarse_target_y, frame2_x, frame2_y)
        fine_tgt_display_x, fine_tgt_display_y = to_display_coords(fine_target_x, fine_target_y, frame2_x, frame2_y)

        # Draw Original Flow Vector (Red)
        self.draw_arrow(src_display_x, src_display_y, orig_tgt_display_x, orig_tgt_display_y, 
                        color="red", tags="detail_analysis")
        
        # Draw LOD Flow Vector (Orange)
        self.draw_arrow(src_display_x, src_display_y, lod_tgt_display_x, lod_tgt_display_y, 
                        color="orange", tags="detail_analysis")
        
        # Draw Coarse Corrected Flow Vector (Green)
        self.draw_arrow(src_display_x, src_display_y, coarse_tgt_display_x, coarse_tgt_display_y, 
                        color="lime", tags="detail_analysis")

        # Draw Fine Corrected Flow Vector (Blue)
        if fine_tgt_display_x:
            self.draw_arrow(src_display_x, src_display_y, fine_tgt_display_x, fine_tgt_display_y, 
                            color="cyan", tags="detail_analysis")

        # Draw source point circle
        self.canvas.create_oval(src_display_x-3, src_display_y-3, src_display_x+3, src_display_y+3, 
                               fill="yellow", outline="white", tags="detail_analysis")

    def _run_correction_for_frame_index(self, frame_idx, update_progress=False):
        """
        Runs the core correction logic for a single frame index.
        This is the engine for both single-frame and batch corrections.
        Returns a dictionary with results, or None if correction cannot proceed.
        """
        if self.load_flow_data(frame_idx) is None:
            print(f"Skipping frame {frame_idx}: No flow data.")
            return None

        # --- Initial Setup ---
        frame1 = self.frames[frame_idx]
        frame2 = self.frames[frame_idx + 1]
        flow = self.load_flow_data(frame_idx).copy()
        h, w = frame1.shape[:2]
        fh, fw = flow.shape[:2]

        # --- Step 1: Get/Generate Existing Quality Map ---
        quality_map = self.quality_maps.get(frame_idx)
        if quality_map is None:
            if 'cuda' in str(self.processor.device):
                quality_map = self.generate_quality_frame_gpu(frame1, frame2, flow)
            else:
                quality_map = self.generate_quality_frame_fast(frame1, frame2, flow)

        # --- Step 2: Extract Pixels for Correction ---
        bad_pixels_y, bad_pixels_x = np.where(quality_map[:, :, 0] > 0)
        bad_pixels_coords = list(zip(bad_pixels_x, bad_pixels_y))
        initial_error_count = len(bad_pixels_coords)

        if not bad_pixels_coords:
            return {'flow': flow, 'initial': 0, 'final': 0, 'improved': 0, 'failed': 0}

        if update_progress:
            self.progressbar.config(maximum=initial_error_count, value=0)

        # --- Step 3: Prepare LOD Data ---
        lod_level, lod_flow = self.get_highest_available_lod(frame_idx)
        if lod_flow is None:
            messagebox.showerror("Error", f"Frame {frame_idx}: No LOD data available for correction.")
            return None

        # --- Step 4: Perform Correction ---
        flow_data = flow
        scale_x_frame_to_flow = fw / w if w > 0 else 1.0
        scale_y_frame_to_flow = fh / h if h > 0 else 1.0
        lod_h, lod_w = lod_flow.shape[:2]
        lod_scale_x_frame_to_lod = lod_w / w
        lod_scale_y_frame_to_lod = lod_h / h
        
        improved_pixels_set = set()

        for i, (orig_x, orig_y) in enumerate(bad_pixels_coords):
            flow_x_coord = int(orig_x * scale_x_frame_to_flow)
            flow_y_coord = int(orig_y * scale_y_frame_to_flow)
            flow_x_coord = max(0, min(flow_x_coord, fw - 1))
            flow_y_coord = max(0, min(flow_y_coord, fh - 1))
            original_flow_x = flow_data[flow_y_coord, flow_x_coord, 0] / scale_x_frame_to_flow
            original_flow_y = flow_data[flow_y_coord, flow_x_coord, 1] / scale_y_frame_to_flow
            original_target_x, original_target_y = orig_x - original_flow_x, orig_y - original_flow_y
            
            original_similarity = 0.0
            if (0 <= original_target_x < w and 0 <= original_target_y < h):
                original_similarity = self.calculate_pixel_quality(frame1[orig_y, orig_x], frame2[int(original_target_y), int(original_target_x)])

            lod_x = max(0, min(int(orig_x * lod_scale_x_frame_to_lod), lod_w - 1))
            lod_y = max(0, min(int(orig_y * lod_scale_y_frame_to_lod), lod_h - 1))
            lod_flow_x = lod_flow[lod_y, lod_x, 0] / lod_scale_x_frame_to_lod
            lod_flow_y = lod_flow[lod_y, lod_x, 1] / lod_scale_y_frame_to_lod
            
            coarse_result = self._perform_coarse_correction(frame1, frame2, (orig_x, orig_y), (lod_flow_x, lod_flow_y))
            final_flow = coarse_result['flow']
            final_similarity = coarse_result['similarity']

            if coarse_result['similarity'] < self.FINE_CORRECTION_THRESHOLD:
                fine_result = self._perform_fine_correction(frame1, frame2, (orig_x, orig_y), coarse_result['target'])
                if fine_result and fine_result['similarity'] > coarse_result['similarity']:
                    final_flow = fine_result['flow']
                    final_similarity = fine_result['similarity']

            if self._is_good_quality(final_similarity) or (final_similarity > original_similarity):
                final_flow_x_scaled = final_flow[0] * scale_x_frame_to_flow
                final_flow_y_scaled = final_flow[1] * scale_y_frame_to_flow
                flow_y_coord, flow_x_coord = int(orig_y * scale_y_frame_to_flow), int(orig_x * scale_x_frame_to_flow)
                flow_y_coord, flow_x_coord = max(0, min(flow_y_coord, fh - 1)), max(0, min(flow_x_coord, fw - 1))
                flow[flow_y_coord, flow_x_coord] = [final_flow_x_scaled, final_flow_y_scaled]

                if not self._is_good_quality(final_similarity):
                    improved_pixels_set.add((orig_x, orig_y))
            
            if update_progress and (i + 1) % 50 == 0:
                self.progressbar.config(value=i + 1)
                self.root.update_idletasks()
        
        # --- Step 5: Create New Quality Map ---
        if 'cuda' in str(self.processor.device):
            new_quality_map = self.generate_quality_frame_gpu(frame1, frame2, flow)
        else:
            new_quality_map = self.generate_quality_frame_fast(frame1, frame2, flow)
            
        # --- Step 6: Get Final Error Count & Update Caches ---
        final_error_y, final_error_x = np.where(new_quality_map[:, :, 0] > 0)
        final_error_count = len(final_error_y)
        final_bad_pixels_set = set(zip(final_error_x, final_error_y))
        
        self.improved_correction_pixels[frame_idx] = improved_pixels_set.intersection(final_bad_pixels_set)
        self.failed_correction_pixels[frame_idx] = final_bad_pixels_set.difference(self.improved_correction_pixels[frame_idx])
        
        self.flow_data_cache[frame_idx] = flow
        self.quality_maps[frame_idx] = new_quality_map

        return {
            'flow': flow,
            'initial': initial_error_count,
            'final': final_error_count,
            'improved': len(self.improved_correction_pixels[frame_idx]),
            'failed': len(self.failed_correction_pixels[frame_idx])
        }

    def correct_all_errors(self):
        """
        Corrects all poor quality flow vectors for the current frame pair synchronously,
        and updates the UI.
        """
        # --- UI Setup ---
        self.correct_errors_btn.config(state=tk.DISABLED)
        self.correct_all_frames_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Starting correction...")
        self.root.update_idletasks()
        
        # --- Run Correction ---
        results = self._run_correction_for_frame_index(self.current_pair, update_progress=True)
        
        # --- UI Cleanup and Reporting ---
        self.correct_errors_btn.config(state=tk.NORMAL)
        self.correct_all_frames_btn.config(state=tk.NORMAL)
        self.progressbar.config(value=0)
        
        if results is None:
            self.progress_label.config(text="Correction aborted (no data).")
            # The helper function already shows a messagebox for critical errors
            return
            
        # --- Update Main Display and Cache ---
        # The helper function already updated the caches. We just need to refresh the view.
        self.current_flow = results['flow'] # Update the flow being used by the visualizer
        self.update_display()
        
        # --- Save Corrected Flow ---
        saved_path = None
        if self.save_corrected_var.get():
            try:
                original_flow_path = Path(self.flow_files[self.current_pair])
                corrected_flow_dir = Path(self.flow_dir).with_name(Path(self.flow_dir).name + "_corrected")
                os.makedirs(corrected_flow_dir, exist_ok=True)
                new_flow_path = corrected_flow_dir / original_flow_path.name
                
                if new_flow_path.suffix == '.flo':
                    writeFlow(str(new_flow_path), results['flow'])
                elif new_flow_path.suffix == '.npz':
                    np.savez_compressed(str(new_flow_path), flow=results['flow'])
                
                saved_path = new_flow_path
                print(f"Saved corrected flow to {saved_path}")

            except Exception as e:
                print(f"Error saving corrected flow file: {e}")
                messagebox.showerror("Save Error", f"Could not save the corrected flow file.\n{e}")

        # --- Final Message ---
        final_message = (
            f"Correction complete. "
            f"Errors: {results['initial']} -> {results['final']}. "
            f"Improved: {results['improved']}, Failed to fix: {results['failed']}."
        )
        if saved_path:
            final_message += f"\n\nCorrected flow saved to:\n{saved_path}"
        
        self.progress_label.config(text=final_message)
        messagebox.showinfo("Success", final_message)
        self.root.after(5000, lambda: self.progress_label.config(text=""))

    def correct_all_frames_sequentially(self):
        """
        Corrects all poor quality flow vectors for ALL frame pairs sequentially.
        """
        if not messagebox.askokcancel("Confirm Batch Correction", 
            f"This will start a batch correction for all {self.max_pairs} frame pairs. "
            "The UI will be unresponsive until it completes. This may take a long time.\n\nContinue?"):
            return
            
        self._correct_frames_in_range(0, self.max_pairs - 1, "all frames")

    def set_start_to_current(self):
        """Set start frame to current frame"""
        self.start_frame_var.set(self.current_pair)
    
    def set_end_to_current(self):
        """Set end frame to current frame"""
        self.end_frame_var.set(self.current_pair)
    
    def set_current_to_end_range(self):
        """Set range from current frame to the last frame"""
        self.start_frame_var.set(self.current_pair)
        self.end_frame_var.set(self.max_pairs - 1)

    def correct_frames_range(self):
        """
        Corrects poor quality flow vectors for a specified range of frame pairs.
        """
        try:
            start_frame = self.start_frame_var.get()
            end_frame = self.end_frame_var.get()
        except tk.TclError:
            messagebox.showerror("Invalid Input", "Please enter valid frame numbers.")
            return
        
        # Validate range
        if start_frame < 0 or end_frame >= self.max_pairs:
            messagebox.showerror("Invalid Range", 
                f"Frame range must be between 0 and {self.max_pairs - 1}.")
            return
        
        if start_frame > end_frame:
            messagebox.showerror("Invalid Range", 
                "Start frame must be less than or equal to end frame.")
            return
        
        frame_count = end_frame - start_frame + 1
        if not messagebox.askokcancel("Confirm Range Correction", 
            f"This will start a batch correction for frames {start_frame} to {end_frame} "
            f"({frame_count} frame pairs). "
            "The UI will be unresponsive until it completes. This may take a long time.\n\nContinue?"):
            return
            
        self._correct_frames_in_range(start_frame, end_frame, f"frames {start_frame}-{end_frame}")

    def _correct_frames_in_range(self, start_frame, end_frame, description):
        """
        Dispatcher for batch correction. Runs either single-threaded or multi-threaded
        based on the UI checkbox.
        """
        if self.multithreaded_var.get():
            self._correct_frames_in_range_multithreaded(start_frame, end_frame, description)
        else:
            self._correct_frames_in_range_singlethreaded(start_frame, end_frame, description)

    def _correct_frames_in_range_singlethreaded(self, start_frame, end_frame, description):
        """
        Internal method to correct frames in a specified range.
        
        Args:
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (inclusive)
            description: Description for UI messages
        """
        frame_count = end_frame - start_frame + 1
        
        # --- UI Setup ---
        self.correct_errors_btn.config(state=tk.DISABLED)
        self.correct_all_frames_btn.config(state=tk.DISABLED)
        self.correct_range_btn.config(state=tk.DISABLED)
        self.progressbar.config(maximum=frame_count, value=0)
        
        total_initial_errors, total_final_errors, total_improved, total_failed = 0, 0, 0, 0
        frames_processed, frames_skipped = 0, 0
        batch_start_time = time.time()
        
        # --- Log Header ---
        print(f"\n=== Batch Correction Started for {description} ===")
        print(f"Processing {frame_count} frame(s): {start_frame} to {end_frame}")
        print("Frame    : Initial Corrected Failed Improved Success%   Time")
        print("-" * 65)
        
        # --- Main Loop ---
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            frame_start_time = time.time()
            
            self.progress_label.config(text=f"Processing frame {frame_idx} ({i + 1}/{frame_count})...")
            self.progressbar.config(value=i + 1)
            self.root.update_idletasks()
            
            results = self._run_correction_for_frame_index(frame_idx)
            
            frame_duration = time.time() - frame_start_time
            
            if results is None:
                frames_skipped += 1
                print(f"Frame {frame_idx:4d}: SKIPPED (no flow data available) - {frame_duration:.2f}s")
                continue
            
            frames_processed += 1
            total_initial_errors += results['initial']
            total_final_errors += results['final']
            total_improved += results['improved']
            total_failed += results['failed']
            
            # Calculate correction statistics for this frame
            corrected_vectors = results['initial'] - results['final']
            failed_to_correct = results['failed']
            improvement_rate = (corrected_vectors / results['initial'] * 100) if results['initial'] > 0 else 0
            
            # Log detailed frame statistics
            print(f"Frame {frame_idx:4d}: "
                  f"Initial errors: {results['initial']:4d}, "
                  f"Corrected: {corrected_vectors:4d}, "
                  f"Failed: {failed_to_correct:4d}, "
                  f"Improved: {results['improved']:4d}, "
                  f"Success rate: {improvement_rate:5.1f}% - "
                  f"{frame_duration:.2f}s")
            
            if self.save_corrected_var.get() and results['flow'] is not None:
                try:
                    original_flow_path = Path(self.flow_files[frame_idx])
                    corrected_flow_dir = Path(self.flow_dir).with_name(Path(self.flow_dir).name + "_corrected")
                    os.makedirs(corrected_flow_dir, exist_ok=True)
                    new_flow_path = corrected_flow_dir / original_flow_path.name
                    
                    if new_flow_path.suffix == '.flo':
                        writeFlow(str(new_flow_path), results['flow'])
                    elif new_flow_path.suffix == '.npz':
                        np.savez_compressed(str(new_flow_path), flow=results['flow'])
                except Exception as e:
                    print(f"Error saving corrected flow for frame {frame_idx}: {e}")
        
        batch_duration = time.time() - batch_start_time
        
        # --- Log Footer ---
        print("-" * 65)
        total_corrected = total_initial_errors - total_final_errors
        overall_success_rate = (total_corrected / total_initial_errors * 100) if total_initial_errors > 0 else 0
        avg_time_per_frame = batch_duration / frame_count if frame_count > 0 else 0
        
        print(f"=== Batch Correction Complete ===")
        print(f"Total time: {batch_duration:.2f}s ({avg_time_per_frame:.2f}s/frame)")
        print(f"Frames processed: {frames_processed}/{frame_count}")
        print(f"Frames skipped: {frames_skipped}")
        print(f"Initial errors: {total_initial_errors}")
        print(f"Final errors: {total_final_errors}")
        print(f"Successfully corrected: {total_corrected}")
        print(f"Partially improved: {total_improved}")
        print(f"Failed to correct: {total_failed}")
        print(f"Overall success rate: {overall_success_rate:.1f}%")
        print("=" * 40)
        
        # --- Reporting ---
        self.update_display()
        
        summary_message = (
            f"Batch correction complete for {description} in {batch_duration:.2f}s.\n\n"
            f"Frames Processed: {frames_processed}/{frame_count}\n"
            f"Frames Skipped (no data): {frames_skipped}\n"
            f"Average time per frame: {avg_time_per_frame:.2f}s\n\n"
            f"Total Initial Errors: {total_initial_errors:,}\n"
            f"Total Final Errors: {total_final_errors:,}\n"
            f"Successfully Corrected: {total_corrected:,}\n"
            f"  - Partially improved: {total_improved:,}\n"
            f"  - Failed to fix: {total_failed:,}\n"
            f"Overall Success Rate: {overall_success_rate:.1f}%"
        )
        
        if self.save_corrected_var.get() and frames_processed > 0:
             corrected_flow_dir = Path(self.flow_dir).with_name(Path(self.flow_dir).name + "_corrected")
             summary_message += f"\n\nCorrected flows saved to:\n{corrected_flow_dir.resolve()}"

        messagebox.showinfo("Batch Complete", summary_message)
        
        # --- UI Cleanup ---
        self.progress_label.config(text="")
        self.progressbar.config(value=0)
        self.correct_errors_btn.config(state=tk.NORMAL)
        self.correct_all_frames_btn.config(state=tk.NORMAL)
        self.correct_range_btn.config(state=tk.NORMAL)

    def _correct_frames_in_range_multithreaded(self, start_frame, end_frame, description):
        """
        Corrects frames in a specified range using a thread pool.
        """
        frame_indices = list(range(start_frame, end_frame + 1))
        frame_count = len(frame_indices)
        
        # --- UI and Worker Setup ---
        self.correct_errors_btn.config(state=tk.DISABLED)
        self.correct_all_frames_btn.config(state=tk.DISABLED)
        self.correct_range_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
        num_workers = os.cpu_count() or 1
        min_chunk_size = 10
        
        # Calculate chunk size, ensuring it's not too small
        chunk_size = max(min_chunk_size, math.ceil(frame_count / num_workers))
        chunks = [frame_indices[i:i + chunk_size] for i in range(0, frame_count, chunk_size)]
        
        # --- Log Header ---
        print(f"\n=== Multithreaded Batch Correction Started for {description} ===")
        print(f"Processing {frame_count} frames on {len(chunks)} workers.")

        corrected_flow_dir = Path(self.flow_dir).with_name(Path(self.flow_dir).name + "_corrected")
        corrected_flow_dir.mkdir(exist_ok=True)
        print(f"Corrected flows will be saved to: {corrected_flow_dir.resolve()}")

        batch_start_time = time.time()
        all_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Prepare arguments for each worker
            worker_args = []
            for i, chunk in enumerate(chunks):
                arg_tuple = (
                    i + 1,                     # worker_id
                    chunk,                     # frame_indices
                    self.frames,               # All video frames
                    self.flow_data_cache,      # Full flow cache
                    self.lod_data_cache,       # Full LOD cache
                    str(self.processor.device),# Device string
                    self.max_lod_levels,       # Max LOD levels
                    self.flow_files,           # List of original flow files
                    {                          # Constants
                        'GOOD_QUALITY_THRESHOLD': self.GOOD_QUALITY_THRESHOLD,
                        'FINE_CORRECTION_THRESHOLD': self.FINE_CORRECTION_THRESHOLD,
                        'DETAIL_ANALYSIS_REGION_SIZE': self.detail_analysis_region_size,
                        'TEMPLATE_RADIUS': self.template_radius,
                        'SEARCH_RADIUS': self.search_radius,
                    }
                )
                worker_args.append(arg_tuple)
            
            # Submit jobs and track progress with tqdm
            futures = [executor.submit(correction_worker.worker_process, *args) for args in worker_args]
            
            self.progress_label.config(text="Workers processing frames...")
            self.progressbar.config(maximum=len(futures), value=0)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    worker_results = future.result()
                    if worker_results:
                        all_results.extend(worker_results)
                    self.progressbar.step(1)
                    self.root.update_idletasks()
                except Exception as e:
                    print(f"A worker failed with an exception: {e}")
                    import traceback
                    traceback.print_exc()
        
        batch_duration = time.time() - batch_start_time
        
        # --- Aggregate Results and Verify Output ---
        print("\n--- Verifying Output and Aggregating Results ---")
        total_initial_errors = sum(r['initial'] for r in all_results if r)
        total_final_errors = sum(r['final'] for r in all_results if r)
        total_improved = sum(r['improved'] for r in all_results if r)
        total_failed = sum(r['failed'] for r in all_results if r)
        frames_processed = len(all_results) - sum(1 for r in all_results if r and r['skipped'])
        frames_skipped = sum(1 for r in all_results if r and r['skipped'])
        
        missing_files = []
        for frame_idx in frame_indices:
            try:
                original_flow_path = Path(self.flow_files[frame_idx])
                expected_output_path = corrected_flow_dir / original_flow_path.name
                if not expected_output_path.exists():
                    missing_files.append((frame_idx, str(expected_output_path)))
            except IndexError:
                print(f"Warning: Verification failed for frame index {frame_idx}. Index out of range for flow files list.")
        
        if not missing_files:
            print("Verification successful: All expected output files were found.")
        else:
            print(f"Verification FAILED: Found {len(missing_files)} missing output files.")
            for idx, path in missing_files[:10]:
                print(f"  - Missing: Frame {idx} at {path}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more.")

        # --- Log Footer ---
        total_corrected = total_initial_errors - total_final_errors
        overall_success_rate = (total_corrected / total_initial_errors * 100) if total_initial_errors > 0 else 0
        avg_time_per_frame = batch_duration / frame_count if frame_count > 0 else 0

        print("-" * 65)
        print(f"=== Multithreaded Batch Correction Complete ===")
        print(f"Total time: {batch_duration:.2f}s ({avg_time_per_frame:.2f}s/frame)")
        
        # --- Reporting ---
        summary_message = (
            f"Multithreaded correction complete for {description} in {batch_duration:.2f}s.\n\n"
            f"Frames Processed: {frames_processed}/{frame_count} on {len(chunks)} workers.\n"
            f"Frames Skipped: {frames_skipped}\n"
            f"Total Initial Errors: {total_initial_errors:,}\n"
            f"Total Final Errors: {total_final_errors:,}\n"
            f"Successfully Corrected: {total_corrected:,}\n"
            f"Overall Success Rate: {overall_success_rate:.1f}%"
        )
        
        # Add verification result to summary message
        if not missing_files:
            summary_message += "\n\nVerification successful: All corrected flow files were saved correctly."
        else:
            summary_message += f"\n\nVERIFICATION FAILED: {len(missing_files)} corrected flow files are missing."
            summary_message += "\nPlease check the console output for a list of missing files."

        if self.save_corrected_var.get() and frames_processed > 0:
             summary_message += f"\n\nCorrected flows saved to:\n{corrected_flow_dir.resolve()}"
             summary_message += "\n\nNOTE: The current view has NOT been updated. Please restart the visualizer on the '_corrected' directory to see the results."

        messagebox.showinfo("Batch Complete", summary_message)
        
        # --- UI Cleanup ---
        self.progress_label.config(text="")
        self.progressbar.config(value=0)
        self.correct_errors_btn.config(state=tk.NORMAL)
        self.correct_all_frames_btn.config(state=tk.NORMAL)
        self.correct_range_btn.config(state=tk.NORMAL)

    def generate_quality_frame_gpu(self, frame1, frame2, flow):
        """
        Generate a quality visualization frame using GPU (PyTorch) for acceleration.
        """
        return correction_worker.generate_quality_frame_gpu(frame1, frame2, flow, self.processor.device, self.GOOD_QUALITY_THRESHOLD)
        
    def run_taa_processor(self):
        """
        Constructs and runs the flow_processor.py command with TAA options,
        using the corrected flow cache.
        """
        # 1. Check for corrected cache
        flow_dir_path = Path(self.flow_dir)
        if flow_dir_path.name.endswith("_corrected"):
            # Already using a corrected cache directory
            corrected_flow_dir = flow_dir_path
        else:
            # Need to find/generate corrected cache directory
            corrected_flow_dir = flow_dir_path.with_name(flow_dir_path.name + "_corrected")
            
        if not corrected_flow_dir.exists():
            if flow_dir_path.name.endswith("_corrected"):
                # The specified corrected cache directory doesn't exist
                messagebox.showwarning("Cache Not Found",
                    f"The corrected flow cache directory does not exist:\n{corrected_flow_dir.resolve()}\n\n"
                    "Please check that the path is correct or run corrections to generate the cache."
                )
            else:
                # Need to generate corrected cache first
                messagebox.showwarning("Cache Not Found",
                    "The corrected flow cache directory does not exist.\n\n"
                    f"Please run 'Correct All Frames' first to generate it at:\n{corrected_flow_dir.resolve()}"
                )
            return

        # 2. Get video parameters
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps == 0:
                raise ValueError("Could not determine video FPS.")

            start_time = self.start_frame / fps
            # Using len(self.frames) is correct because it's the number of frames loaded into the visualizer
            duration = len(self.frames) / fps 
        except Exception as e:
            messagebox.showerror("Error", f"Could not get video parameters: {e}")
            return
            
        # 3. Construct command
        command = [
            sys.executable,  # Use the same python interpreter
            "flow_processor.py",
            "--input", str(self.video_path),
            "--start-time", f"{start_time:.2f}",
            "--duration", f"{duration:.2f}",
            "--taa",
            "--skip-lods",
            "--tile",
            "--flow-format", "hsv",
            "--use-flow-cache", str(corrected_flow_dir.resolve())
        ]
        
        # Add TAA compression emulation flag if enabled

        
        # 4. Execute command
        try:
            print("--- Running TAA Processor ---")
            print(f"Command: {' '.join(command)}")
            
            # Show an info box to the user
            messagebox.showinfo("Starting TAA Processor",
                "A new terminal window will open to run the TAA process.\n\n"
                "This application will remain open. You can monitor the progress in the new window."
            )

            # Platform-specific execution
            if sys.platform == "win32":
                subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # For macOS/Linux, might need to use xterm or another terminal emulator
                command_str = ' '.join(f'"{c}"' for c in command)
                term_command = ['xterm', '-e', command_str]
                try:
                    subprocess.Popen(term_command)
                except FileNotFoundError:
                    # If xterm is not available
                    messagebox.showwarning("Terminal not found", "Could not find 'xterm' to open a new terminal. Please run the following command manually in your terminal:\n\n" + ' '.join(command))

        except Exception as e:
            error_msg = f"Failed to start TAA processor: {e}\n\nCommand was:\n{' '.join(command)}"
            print(error_msg)
            messagebox.showerror("Execution Error", error_msg)
            
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

    def compute_current_turbulence_map(self):
        """
        Computes the turbulence map for the currently selected frame pair.
        """
        frame_idx = self.current_pair
        
        try:
            flow = self.load_flow_data(frame_idx)
            
            if flow is None:
                messagebox.showerror("Error", f"No flow data available for frame {frame_idx}.")
                return

            print(f"Generating turbulence map for frame {frame_idx}...")
            start_time = time.time()
            
            # Generate the map
            turbulence_map = self.generate_turbulence_map(flow)

            self.turbulence_maps[frame_idx] = turbulence_map
            duration = time.time() - start_time
            print(f"Turbulence map generation finished in {duration:.4f}s")

            self.update_display() # Refresh UI with the new map

        except Exception as e:
            print(f"Error during turbulence map generation: {e}")
            messagebox.showerror("Error", f"An error occurred during turbulence map generation:\n{e}")

    def generate_turbulence_map(self, flow, kernel_size=25):
        """
        Generate a map visualizing flow turbulence (local vector variance).
        A high value indicates high variance in flow vectors in the neighborhood.
        """
        if flow is None:
            return np.zeros_like(self.frame1)

        # We need to resize the flow to match the frame dimensions for visualization
        h, w = self.orig_height, self.orig_width
        fh, fw = flow.shape[:2]
        # Ensure fh and fw are not zero to avoid division by zero
        if fh == 0 or fw == 0:
            return np.zeros_like(self.frame1)
            
        if (fh != h or fw != w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            # Scale flow vectors
            flow[:, :, 0] *= w / fw
            flow[:, :, 1] *= h / fh
        
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        # Use OpenCV's boxFilter for fast local mean calculation.
        # This is equivalent to convolving with a normalized kernel.
        mean_x = cv2.boxFilter(flow_x, -1, (kernel_size, kernel_size), normalize=True, borderType=cv2.BORDER_REFLECT)
        mean_y = cv2.boxFilter(flow_y, -1, (kernel_size, kernel_size), normalize=True, borderType=cv2.BORDER_REFLECT)
        
        mean_x2 = cv2.boxFilter(flow_x**2, -1, (kernel_size, kernel_size), normalize=True, borderType=cv2.BORDER_REFLECT)
        mean_y2 = cv2.boxFilter(flow_y**2, -1, (kernel_size, kernel_size), normalize=True, borderType=cv2.BORDER_REFLECT)

        # Variance = E[X^2] - E[X]^2
        var_x = mean_x2 - mean_x**2
        var_y = mean_y2 - mean_y**2
        
        # Total variance (magnitude of variance vector)
        # Add a small epsilon to avoid issues with sqrt of negative numbers due to precision
        total_variance = np.sqrt(np.maximum(0, var_x) + np.maximum(0, var_y))
        
        # Normalize the map for visualization
        # Using percentile to be robust against outliers
        min_val = np.percentile(total_variance, 5)
        max_val = np.percentile(total_variance, 95)
        
        if max_val - min_val > 1e-6:
            normalized_variance = (total_variance - min_val) / (max_val - min_val)
            normalized_variance = np.clip(normalized_variance, 0, 1)
        else:
            normalized_variance = np.zeros_like(total_variance)

        # Apply a colormap
        heatmap = (normalized_variance * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def generate_quality_frame(self, frame1, frame2, flow):
        """Generate a quality visualization frame"""
        if flow is None:
            # Return black frame if no flow data
            return np.zeros_like(frame1)
        
        h, w = frame1.shape[:2]
        fh, fw = flow.shape[:2]
        quality_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate scale factors if flow and frame dimensions don't match
        scale_x = w / fw if fw > 0 else 1.0
        scale_y = h / fh if fh > 0 else 1.0
        
        # Resize flow to match frame dimensions if needed
        if (fh != h or fw != w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            # Scale flow vectors by the resize ratio
            flow[:, :, 0] *= scale_x
            flow[:, :, 1] *= scale_y
        
        # Vectorized approach for better performance
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Get flow vectors
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        
        # Calculate target positions
        target_x = x_coords - flow_x
        target_y = y_coords - flow_y
        
        # Create masks for valid targets
        valid_mask = ((target_x >= 0) & (target_x < w) & 
                     (target_y >= 0) & (target_y < h))
        
        # Process valid pixels
        valid_y, valid_x = np.where(valid_mask)
        
        if len(valid_y) > 0:
            # Get source colors
            src_colors = frame1[valid_y, valid_x]
            
            # Get target coordinates (rounded to integers)
            tgt_x = np.round(target_x[valid_y, valid_x]).astype(int)
            tgt_y = np.round(target_y[valid_y, valid_x]).astype(int)
            
            # Ensure target coordinates are within bounds
            tgt_x = np.clip(tgt_x, 0, w-1)
            tgt_y = np.clip(tgt_y, 0, h-1)
            
            # Get target colors
            tgt_colors = frame2[tgt_y, tgt_x]
            
            # Calculate similarities for all valid pixels at once
            similarities = []
            for i in range(len(valid_y)):
                sim = self.calculate_pixel_quality(src_colors[i], tgt_colors[i])
                similarities.append(sim)
            similarities = np.array(similarities)
            
            # Color coding
            good_mask = similarities > self.GOOD_QUALITY_THRESHOLD
            
            # Good quality pixels - green
            good_indices = np.where(good_mask)[0]
            if len(good_indices) > 0:
                intensities = (255 * (similarities[good_indices] - 0.5) * 2).astype(int) # Scale green intensity
                quality_frame[valid_y[good_indices], valid_x[good_indices]] = np.column_stack([
                    np.zeros_like(intensities), np.clip(intensities, 0, 255), np.zeros_like(intensities)
                ])
            
            # Bad quality pixels - red
            bad_indices = np.where(~good_mask)[0]
            if len(bad_indices) > 0:
                intensities = (255 * (1.0 - similarities[bad_indices])).astype(int)
                quality_frame[valid_y[bad_indices], valid_x[bad_indices]] = np.column_stack([
                    intensities, np.zeros_like(intensities), np.zeros_like(intensities)
                ])
        
        # Out of bounds pixels - red
        invalid_y, invalid_x = np.where(~valid_mask)
        if len(invalid_y) > 0:
            quality_frame[invalid_y, invalid_x] = [255, 0, 0]
        
        return quality_frame
    
def main():
    parser = argparse.ArgumentParser(description='Interactive Optical Flow Visualizer')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--flow-dir', required=True, help='Directory containing optical flow files')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame number')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames')
    
    # Model parameters
    parser.add_argument('--model', choices=['videoflow', 'memflow'], default='videoflow',
                        help='Choose optical flow model: videoflow or memflow')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Custom path to model weights (only for MemFlow)')
    parser.add_argument('--stage', choices=['sintel', 'things', 'kitti'], default='sintel',
                        help='Training stage/dataset for MemFlow')
    parser.add_argument('--vf-dataset', choices=['sintel', 'things', 'kitti'], default='sintel',
                        help='Dataset for VideoFlow model')
    parser.add_argument('--vf-architecture', choices=['mof', 'bof'], default='mof',
                        help='VideoFlow architecture: mof (MOFNet) or bof (BOFNet)')
    parser.add_argument('--vf-variant', choices=['standard', 'noise'], default='standard',
                        help='VideoFlow model variant: standard or noise')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
        
    if not os.path.exists(args.flow_dir):
        print(f"Error: Flow directory not found: {args.flow_dir}")
        return
    
    try:
        visualizer = FlowVisualizer(args.video, args.flow_dir, args.start_frame, args.max_frames,
                                   flow_model=args.model, model_path=args.model_path, stage=args.stage,
                                   vf_dataset=args.vf_dataset, vf_architecture=args.vf_architecture,
                                   vf_variant=args.vf_variant)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 