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

# VideoFlow modules are now handled by processing.VideoFlowInference

# Import our modules
from config import DeviceManager
from video import VideoInfo, FrameExtractor
from encoding import FlowEncoderFactory, encode_flow
from effects import TAAProcessor, apply_taa_effect
from storage import FlowCacheManager
from visualization import VideoComposer, create_side_by_side, add_text_overlay, create_video_grid
from filtering import AdaptiveOpticalFlowKalmanFilter
from processing.flow_inference import VideoFlowInference
from processing.memflow_inference import MemFlowInference


class VideoFlowProcessor:
    def __init__(self, device='auto', fast_mode=False, tile_mode=False, sequence_length=5, flow_smoothing=0.0,
                 kalman_process_noise=0.01, kalman_measurement_noise=0.1, kalman_prediction_confidence=0.7,
                 kalman_motion_model='constant_acceleration', kalman_outlier_threshold=3.0, kalman_min_track_length=3,
                 flow_model='videoflow', model_path=None, stage='sintel', 
                 vf_dataset='sintel', vf_architecture='mof', vf_variant='standard',
                 taa_emulate_compression=False):
        """Initialize optical flow processor with model selection support"""
        # Initialize device manager
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device(device)
            
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self.flow_smoothing = flow_smoothing
        self.flow_model = flow_model.lower()
        self.stage = stage
        
        # VideoFlow specific parameters
        self.vf_dataset = vf_dataset
        self.vf_architecture = vf_architecture
        self.vf_variant = vf_variant
        
        self.previous_smoothed_flow = None  # For temporal flow smoothing
        
        # Initialize inference engine based on selected model
        if self.flow_model == 'memflow':
            # Determine model path for MemFlow
            if model_path is None:
                model_path = f'MemFlow_ckpt/MemFlowNet_{stage}.pth'
            
            self.inference_engine = MemFlowInference(
                device=self.device,
                model_path=model_path,
                stage=stage,
                sequence_length=sequence_length
            )
            print(f"Using MemFlow model: {model_path}")
        elif self.flow_model == 'videoflow':
            # VideoFlow with configurable dataset and architecture
            self.inference_engine = VideoFlowInference(
                device=self.device,
                fast_mode=fast_mode,
                tile_mode=tile_mode,
                sequence_length=sequence_length,
                dataset=vf_dataset,
                architecture=vf_architecture,
                variant=vf_variant
            )
            model_variant_str = f"_{vf_variant}" if vf_variant == 'noise' and vf_dataset == 'things' else ""
            print(f"Using VideoFlow model: {vf_architecture.upper()}_{vf_dataset}{model_variant_str}.pth")
        else:
            raise ValueError(f"Unsupported flow model: {flow_model}. Choose 'videoflow' or 'memflow'")
        
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
        
        # TAA compression emulation setting
        self.taa_emulate_compression = taa_emulate_compression
        
        # TAA processors for consistent state management
        self.taa_flow_processor = TAAProcessor(alpha=0.1, emulate_compression=taa_emulate_compression)
        self.taa_simple_processor = TAAProcessor(alpha=0.1)
        
        # Initialize storage manager
        self.cache_manager = FlowCacheManager()
        
        # Initialize video composer
        self.video_composer = VideoComposer()
        
        print(f"Optical Flow Processor initialized - Device: {self.device}")
        self.device_manager.print_device_info()
        print(f"Model: {self.flow_model.upper()}")
        print(f"Fast mode: {fast_mode}")
        if self.flow_model == 'videoflow':
            print(f"Tile mode: {tile_mode}")
        else:
            print(f"Tile mode: Not used (MemFlow processes full frames)")
        print(f"Sequence length: {sequence_length} frames")
        if flow_smoothing > 0:
            print(f"Flow smoothing: {flow_smoothing:.2f} (color-consistency stabilization enabled)")
        
    def load_model(self):
        """Load optical flow model (VideoFlow or MemFlow)"""
        self.inference_engine.load_model()
        
    def load_videoflow_model(self):
        """Load VideoFlow MOF model (legacy method - use load_model instead)"""
        if self.flow_model != 'videoflow':
            raise RuntimeError("Cannot call load_videoflow_model when using MemFlow. Use load_model() instead.")
        self.inference_engine.load_model()
        
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
        """Calculate tile grid for fixed square tiles (optimized for VideoFlow MOF model)"""
        return self.inference_engine.calculate_tile_grid(width, height, tile_size)
    
    def extract_tile(self, frame, tile_info):
        """Extract a tile from the frame without padding"""
        return self.inference_engine.extract_tile(frame, tile_info)
    
    def compute_optical_flow_tiled(self, frames, frame_idx, tile_pbar=None, overall_pbar=None):
        """Compute optical flow using tile-based processing with 1280x1280 square tiles"""
        return self.inference_engine.compute_optical_flow_tiled(frames, frame_idx, tile_pbar, overall_pbar)
        
    def prepare_frame_sequence(self, frames, frame_idx):
        """Prepare frame sequence for VideoFlow MOF model"""
        return self.inference_engine.prepare_frame_sequence(frames, frame_idx)
        
    def compute_optical_flow_with_progress(self, frames, frame_idx, tile_pbar=None):
        """Compute optical flow with progress updates for tile processing"""
        return self.inference_engine.compute_optical_flow_with_progress(frames, frame_idx, tile_pbar)
        
    def compute_optical_flow(self, frames, frame_idx):
        """Compute optical flow using VideoFlow model"""
        return self.inference_engine.compute_optical_flow(frames, frame_idx)
    
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
        """Save optical flow in Middlebury .flo format (lossless)"""
        return self.cache_manager.file_handler.save_flow_flo(flow, filename)
    
    def save_flow_npz(self, flow, filename, frame_idx=None, metadata=None):
        """Save optical flow in NumPy .npz format (lossless, compressed)"""
        return self.cache_manager.file_handler.save_flow_npz(flow, filename, frame_idx, metadata)
    
    def save_optical_flow_files(self, flow, base_filename, frame_idx, save_format):
        """Save optical flow in specified format(s)"""
        return self.cache_manager.save_optical_flow_files(flow, base_filename, frame_idx, save_format)
    
    def load_flow_flo(self, filename):
        """Load optical flow from Middlebury .flo format"""
        return self.cache_manager.file_handler.load_flow_flo(filename)
    
    def load_flow_npz(self, filename):
        """Load optical flow from NumPy .npz format"""
        return self.cache_manager.file_handler.load_flow_npz(filename)
    
    def generate_flow_cache_path(self, input_path, start_frame, max_frames, sequence_length, fast_mode, tile_mode):
        """Generate cache directory path based on video processing parameters and model configuration"""
        return self.cache_manager.generate_cache_path(
            input_path, start_frame, max_frames, sequence_length, fast_mode, tile_mode,
            model=self.flow_model, dataset=self.vf_dataset if self.flow_model == 'videoflow' else self.stage,
            architecture=self.vf_architecture if self.flow_model == 'videoflow' else 'none',
            variant=self.vf_variant if self.flow_model == 'videoflow' else 'none'
        )
    
    def check_flow_cache_exists(self, cache_dir, max_frames):
        """Check if complete flow cache exists for the requested number of frames"""
        return self.cache_manager.check_cache_exists(cache_dir, max_frames)
    
    def load_cached_flow(self, cache_dir, frame_idx, format_type='auto'):
        """Load cached optical flow for specific frame"""
        return self.cache_manager.load_cached_flow(cache_dir, frame_idx, format_type)
    
    def generate_flow_lods(self, flow_data, num_lods=5):
        """Generate Level-of-Detail (LOD) pyramid for flow data"""
        return self.cache_manager.lod_generator.generate_lods(flow_data, num_lods)
    
    def save_flow_lods(self, lods, cache_dir, frame_idx):
        """Save LOD pyramid for a frame"""
        return self.cache_manager.save_flow_lods(lods, cache_dir, frame_idx)
    
    def load_flow_lod(self, cache_dir, frame_idx, lod_level=0):
        """Load specific LOD level for a frame"""
        return self.cache_manager.load_flow_lod(cache_dir, frame_idx, lod_level)
    
    def check_flow_lods_exist(self, cache_dir, max_frames, num_lods=5):
        """Check if LOD pyramid exists for all frames"""
        return self.cache_manager.check_flow_lods_exist(cache_dir, max_frames, num_lods)
    
    def generate_lods_for_cache(self, cache_dir, max_frames, num_lods=5):
        """Generate LOD pyramids for all frames in cache"""
        return self.cache_manager.generate_lods_for_cache(cache_dir, max_frames, num_lods)

    def analyze_lod_cache_statistics(self, cache_dir, max_frames, num_lods=5):
        """
        Analyze and report detailed LOD cache statistics
        
        Args:
            cache_dir: Cache directory path
            max_frames: Number of frames to analyze
            num_lods: Expected number of LOD levels
        """
        print("\n--- LOD Cache Statistics ---")
        
        lod_dir = os.path.join(cache_dir, 'lods')
        if not os.path.exists(lod_dir):
            print("LOD directory not found - no LOD data available.")
            print("---------------------------\n")
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
                lod_file = os.path.join(lod_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
                
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
        
        print("---------------------------\n")

    def encode_hsv_format(self, flow, width, height):
        """Encode optical flow in HSV format using encoding module"""
        return encode_flow(flow, width, height, 'hsv')
    
    def encode_gamedev_format(self, flow, width, height):
        """Encode optical flow in gamedev format using encoding module"""
        return encode_flow(flow, width, height, 'gamedev')
    
    def encode_motion_vectors_format(self, flow, width, height):
        """Encode optical flow in motion vectors format using encoding module"""
        return encode_flow(flow, width, height, 'motion-vectors')
    
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
        """Add text overlay to frame"""
        return self.video_composer.add_text_overlay(frame, text, position, font_scale, color, thickness)
        
    def create_side_by_side(self, original, flow_viz, vertical=False, flow_only=False, 
                           taa_frame=None, taa_simple_frame=None, model_name="VideoFlow", fast_mode=False, flow_format="gamedev"):
        """Create side-by-side, top-bottom, flow-only, or TAA visualization with text overlays"""
        return self.video_composer.create_side_by_side(
            original, flow_viz, vertical, flow_only, 
            taa_frame, taa_simple_frame, model_name, fast_mode, flow_format
        )
        
    def generate_output_filename(self, input_path, output_dir, start_time=None, duration=None, 
                                start_frame=0, max_frames=1000, vertical=False, flow_only=False, taa=False, lossless=False, uncompressed=False):
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
        
        if uncompressed:
            parts.append("uncompressed")
        elif lossless:
            parts.append("lossless")
        
        # Join parts and add extension
        # MJPG codec requires AVI container, MP4 container doesn't support it
        extension = ".avi" if lossless or uncompressed else ".avi"  # Use AVI for MJPG compatibility
        filename = "_".join(parts) + extension
        return os.path.join(results_dir, filename)
    
    def process_video(self, input_path, output_path, max_frames=1000, start_frame=0, 
                     start_time=None, duration=None, vertical=False, flow_only=False, taa=False, flow_format='gamedev', 
                     save_flow=None, force_recompute=False, use_flow_cache=None, auto_play=True,
                     taa_compare=False, skip_lods=False, lossless=False, uncompressed=False):
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
                start_frame, max_frames, vertical, flow_only, taa, lossless=lossless, uncompressed=uncompressed
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
            cache_exists, cached_flow_format, missing_frames = self.check_flow_cache_exists(flow_cache_dir, len(frames))
            if cache_exists:
                use_cached_flow = True
                print(f"Using optical flow cache from: {flow_cache_dir} (format: {cached_flow_format})")
            else:
                if not os.path.exists(flow_cache_dir):
                     error_message = (
                        f"Error: The specified cache directory does not exist.\n"
                        f"  Directory: {flow_cache_dir}\n"
                        "  Please provide a valid path for '--use-flow-cache'."
                    )
                else:
                    error_message = (
                        f"Error: The specified cache directory is incomplete.\n"
                        f"  Directory: {flow_cache_dir}\n"
                        f"  Reason: Found {len(frames) - len(missing_frames)} of {len(frames)} required flow files.\n"
                    )
                    if len(missing_frames) < 20:
                        error_message += f"  Missing frame indices: {missing_frames}\n"
                    else:
                        error_message += f"  Missing {len(missing_frames)} frames, including: {missing_frames[:10]}...\n"
                    error_message += "  Please check the directory or remove '--use-flow-cache' to generate a new cache."

                print(error_message, file=sys.stderr)
                sys.exit(1)
        else:
            # Generate automatic cache directory
            flow_cache_dir = self.generate_flow_cache_path(
                input_path, start_frame, len(frames), self.sequence_length, 
                self.fast_mode, self.tile_mode
            )
            
            if not force_recompute:
                cache_exists, cached_flow_format, _ = self.check_flow_cache_exists(flow_cache_dir, len(frames))
                if cache_exists:
                    use_cached_flow = True
                    print(f"Found existing optical flow cache: {flow_cache_dir} (format: {cached_flow_format})")
                else:
                    print(f"No existing cache found, will compute and save to: {flow_cache_dir}")
            else:
                print(f"Force recompute enabled, will overwrite cache: {flow_cache_dir}")
        
        # Check and generate LOD pyramids if needed
        if use_cached_flow and not skip_lods:
            # Check if LOD pyramids exist
            lods_exist = self.check_flow_lods_exist(flow_cache_dir, len(frames))
            if not lods_exist:
                print("LOD pyramids not found, generating...")
                self.generate_lods_for_cache(flow_cache_dir, len(frames))
                print("LOD pyramids generated successfully!")
            else:
                print("LOD pyramids found in cache")
            
            # Show detailed LOD statistics
            self.analyze_lod_cache_statistics(flow_cache_dir, len(frames))
        elif skip_lods:
            print("Skipping LOD pyramid check/generation (--skip-lods enabled)")
        
        # Load optical flow model only if we need to compute flow
        if not use_cached_flow:
            self.load_model()
        
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
        if uncompressed:
            fourcc = 0
            print("Using uncompressed video codec. Output will be .avi and file size will be very large.")
        elif lossless:
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            print("Using lossless FFV1 codec (ensure you have ffmpeg installed). Output will be .avi")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print("Using MJPG codec. Output will be .avi for compatibility.")
        
        if taa_compare:
            # For 6 videos (orig, flow, 4x TAA), use 2x3 grid (2 cols, 3 rows) for more square aspect
            grid_cols = 2
            grid_rows = 3
            canvas_w = grid_cols * width
            canvas_h = int(canvas_w / (4/3))  # Target 4:3 aspect ratio instead of 16:9
            output_size = (canvas_w, canvas_h)
        elif flow_only:
            output_size = (width, height * 2)  # Vertical stack: Original on top, Flow on bottom
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
                self.cache_manager.save_flow_to_cache(raw_flow, flow_cache_dir, i, cache_save_format)
            
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
            elif flow_format == 'motion-vectors':
                flow_viz = self.encode_motion_vectors_format(flow, width, height)
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
                    taa_result = self.apply_taa_effect(frames[i], previous_flow, None, alpha=0.1, use_flow=True)
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
        if not use_cached_flow and flow_cache_dir and not skip_lods:
            print("Generating LOD pyramids for computed flow...")
            self.generate_lods_for_cache(flow_cache_dir, len(frames))
            print("LOD pyramids generated!")
            
            # Show detailed LOD statistics for newly generated pyramids
            self.analyze_lod_cache_statistics(flow_cache_dir, len(frames))
        elif not use_cached_flow and skip_lods:
            print("Skipping LOD pyramid generation for computed flow (--skip-lods enabled)")
        
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
        """Arrange multiple video frames into a grid with a target aspect ratio"""
        return self.video_composer.create_video_grid(frames_dict, grid_shape, target_aspect)

def main():
    parser = argparse.ArgumentParser(description='Optical Flow Processor (VideoFlow/MemFlow)')
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
    parser.add_argument('--taa-emulate-compression', action='store_true',
                       help='Emulate motion vectors compression/decompression in TAA processing (uint8 RG format)')
    parser.add_argument('--flow-format', choices=['gamedev', 'hsv', 'torchvision', 'motion-vectors'], default='gamedev',
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
    parser.add_argument('--skip-lods', action='store_true',
                       help='Skip LOD (Level-of-Detail) pyramid generation/loading for faster processing')
    parser.add_argument('--lossless', action='store_true',
                        help='Save the output video using a lossless codec (FFV1 in .avi container)')
    parser.add_argument('--uncompressed', action='store_true',
                        help='Save the output video completely uncompressed (raw frames in .avi container)')
    parser.add_argument('--model', choices=['videoflow', 'memflow'], default='videoflow',
                        help='Choose optical flow model: videoflow (VideoFlow MOF) or memflow (MemFlow)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Custom path to model weights (only for MemFlow)')
    parser.add_argument('--stage', choices=['sintel', 'things', 'kitti'], default='sintel',
                        help='Training stage/dataset (default: sintel, affects model selection for MemFlow)')
    
    # VideoFlow specific options
    parser.add_argument('--vf-dataset', choices=['sintel', 'things', 'kitti'], default='sintel',
                        help='Dataset for VideoFlow model (default: sintel)')
    parser.add_argument('--vf-architecture', choices=['mof', 'bof'], default='mof',
                        help='VideoFlow architecture: mof (MOFNet) or bof (BOFNet) (default: mof)')
    parser.add_argument('--vf-variant', choices=['standard', 'noise'], default='standard',
                        help='VideoFlow model variant: standard or noise (288960noise) (default: standard)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    # Check dependencies based on selected model
    if args.model == 'videoflow':
        if not os.path.exists('VideoFlow'):
            print("Error: VideoFlow repository not found. Please run:")
            print("git clone https://github.com/XiaoyuShi97/VideoFlow.git")
            return
        
        # Build VideoFlow model filename based on parameters
        arch_upper = args.vf_architecture.upper()
        dataset = args.vf_dataset
        if args.vf_variant == 'noise' and args.vf_dataset == 'things':
            model_filename = f"{arch_upper}_{dataset}_288960noise.pth"
        else:
            model_filename = f"{arch_upper}_{dataset}.pth"
        
        vf_model_path = f"VideoFlow_ckpt/{model_filename}"
        
        if not os.path.exists(vf_model_path):
            print(f"Error: VideoFlow model weights not found: {vf_model_path}")
            print("Available VideoFlow models in VideoFlow_ckpt/:")
            if os.path.exists('VideoFlow_ckpt'):
                for f in os.listdir('VideoFlow_ckpt'):
                    if f.endswith('.pth'):
                        print(f"  - {f}")
            else:
                print("  No VideoFlow_ckpt directory found")
            return
    elif args.model == 'memflow':
        if not os.path.exists('MemFlow'):
            print("Error: MemFlow repository not found. Please ensure MemFlow is properly integrated.")
            return
            
        # Check for MemFlow weights
        if args.model_path is None:
            model_path = f'MemFlow_ckpt/MemFlowNet_{args.stage}.pth'
        else:
            model_path = args.model_path
            
        if not os.path.exists(model_path):
            print(f"Error: MemFlow model weights not found: {model_path}")
            print(f"Available MemFlow models in MemFlow_ckpt/:")
            if os.path.exists('MemFlow_ckpt'):
                for f in os.listdir('MemFlow_ckpt'):
                    if f.endswith('.pth'):
                        print(f"  - {f}")
            else:
                print("  No MemFlow_ckpt directory found")
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
                                      kalman_min_track_length=args.kalman_min_track_length,
                                      flow_model=args.model, model_path=args.model_path, stage=args.stage,
                                      vf_dataset=args.vf_dataset, vf_architecture=args.vf_architecture, 
                                      vf_variant=args.vf_variant, taa_emulate_compression=args.taa_emulate_compression)
        
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
        cache_exists, cached_flow_format, missing_frames = processor.check_flow_cache_exists(flow_cache_dir, len(frames))
        
        if not cache_exists or args.force_recompute:
            print(f"Computing optical flow and saving to cache: {flow_cache_dir}")
            
            # Load model and compute flow
            processor.load_model()
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
                processor.cache_manager.save_flow_to_cache(flow, flow_cache_dir, i, 'npz')
                
                pbar.update(1)
            
            pbar.close()
            print("Flow computation completed!")
            
            # Generate LOD pyramids for the computed flow
            if not args.skip_lods:
                print("Generating LOD pyramids...")
                processor.generate_lods_for_cache(flow_cache_dir, len(frames))
                print("LOD pyramids generated!")
            else:
                print("Skipping LOD pyramid generation (--skip-lods enabled)")
        else:
            print(f"Using existing flow cache: {flow_cache_dir}")
            
            # Check and generate LOD pyramids if needed
            if not args.skip_lods:
                lods_exist = processor.check_flow_lods_exist(flow_cache_dir, len(frames))
                if not lods_exist:
                    print("LOD pyramids not found, generating...")
                    processor.generate_lods_for_cache(flow_cache_dir, len(frames))
                    print("LOD pyramids generated successfully!")
                else:
                    print("LOD pyramids found in cache")
                
                # Show detailed LOD statistics
                processor.analyze_lod_cache_statistics(flow_cache_dir, len(frames))
            else:
                print("Skipping LOD pyramid check/generation (--skip-lods enabled)")
        
        # Launch interactive visualizer
        print("Launching interactive flow visualizer...")
        import subprocess
        import sys
        
        visualizer_cmd = [
            sys.executable, "flow_visualizer.py",
            "--video", args.input,
            "--flow-dir", flow_cache_dir,
            "--start-frame", str(start_frame),
            "--max-frames", str(len(frames)),
            "--model", args.model,
            "--stage", args.stage,
            "--vf-dataset", args.vf_dataset,
            "--vf-architecture", args.vf_architecture,
            "--vf-variant", args.vf_variant
        ]
        
        # Add model path if specified
        if args.model_path:
            visualizer_cmd.extend(["--model-path", args.model_path])
        
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
                                          sequence_length=args.sequence_length, flow_smoothing=0.0,
                                          flow_model=args.model, model_path=args.model_path, stage=args.stage,
                                          vf_dataset=args.vf_dataset, vf_architecture=args.vf_architecture, 
                                          vf_variant=args.vf_variant, taa_emulate_compression=False)
        
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
                                  kalman_min_track_length=args.kalman_min_track_length,
                                  flow_model=args.model, model_path=args.model_path, stage=args.stage,
                                  vf_dataset=args.vf_dataset, vf_architecture=args.vf_architecture, 
                                  vf_variant=args.vf_variant, taa_emulate_compression=args.taa_emulate_compression)
    
    try:
        # Create output filename with frame/time range if not specified
        if args.output == 'videoflow_result.mp4':  # Default output pattern (will be changed to .avi)
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
                args.output = f"videoflow_{start_time_str}_{duration_str}{mode}.avi"
            else:
                # Use frame-based naming
                end_frame = args.start_frame + args.frames - 1
                args.output = f"videoflow_{args.start_frame:06d}_{end_frame:06d}{mode}.avi"
        
        processor.process_video(args.input, args.output, max_frames=args.frames, start_frame=args.start_frame,
                              start_time=args.start_time, duration=args.duration, vertical=args.vertical, 
                              flow_only=args.flow_only, taa=args.taa, flow_format=args.flow_format, save_flow=args.save_flow,
                              force_recompute=args.force_recompute, use_flow_cache=args.use_flow_cache, 
                              auto_play=not args.no_autoplay,
                              taa_compare=args.taa_compare, skip_lods=args.skip_lods,
                              lossless=args.lossless, uncompressed=args.uncompressed)
        
        model_name = args.model.upper()
        if not args.no_autoplay and not args.taa_compare:
            print(f"\n✓ {model_name} processing completed successfully! Video should open automatically.")
        else:
            print(f"\n✓ {model_name} processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 