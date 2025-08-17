#!/usr/bin/env python3
"""
VideoFlow Optical Flow Processor

Pure VideoFlow implementation for optical flow generation with gamedev encoding.
Processes only first 1000 frames of the video.
"""

import os
import sys

# Path fix for portable execution: Add the script's directory to sys.path
# This is necessary because the portable Python environment may not include the CWD.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

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
from processing import FlowProcessorFactory


class ReversedVideoFlowComputer:
    """
    Wrapper class for computing optical flow on reversed video while maintaining original frame order.
    
    This class handles the complexity of:
    1. Reversing video frames for flow computation
    2. Computing forward optical flow on reversed video (effectively backward flow)
    3. Mapping results back to original frame indices for caching and visualization
    
    The key insight: computing forward flow on reversed video gives us backward flow for original video.
    """
    
    def __init__(self, inference_engine):
        """
        Initialize with the optical flow inference engine
        
        Args:
            inference_engine: VideoFlow or MemFlow inference engine
        """
        self.inference_engine = inference_engine
        
    def compute_optical_flow_for_frame(self, original_frames, original_frame_idx, tile_pbar=None, overall_pbar=None):
        """
        Compute optical flow for a specific frame using reversed video processing
        
        Args:
            original_frames: List of frames in original order [frame0, frame1, ..., frameN]
            original_frame_idx: Index of frame in original sequence (0 to N-1)
            tile_pbar: Optional progress bar for tile processing
            overall_pbar: Optional progress bar for overall progress
            
        Returns:
            Optical flow for the specified frame in original sequence
        """
        # Special case: frame0 cannot have backward flow (no frame-1 exists)
        # Solution: copy flow from frame1 to frame0
        if original_frame_idx == 0:
            if len(original_frames) == 1:
                # Only one frame - create zero flow
                height, width = original_frames[0].shape[:2]
                return np.zeros((height, width, 2), dtype=np.float32)
            else:
                # Use flow from frame1 (copy it for frame0)
                return self.compute_optical_flow_for_frame(original_frames, 1, tile_pbar, overall_pbar)
        
        # Create reversed frames for flow computation
        reversed_frames = original_frames[::-1]  # [frameN, frameN-1, ..., frame0]
        
        # Map original frame index to reversed frame index
        reversed_frame_idx = len(original_frames) - 1 - original_frame_idx
        
        # Compute flow using reversed frames and reversed index
        # This gives us forward flow on reversed video = backward flow on original video
        flow = self.inference_engine.compute_optical_flow_tiled(
            reversed_frames, reversed_frame_idx, tile_pbar, overall_pbar
        )
        
        return flow
        
    def compute_optical_flow_with_progress_for_frame(self, original_frames, original_frame_idx, tile_pbar=None):
        """
        Compute optical flow with progress updates for a specific frame
        
        Args:
            original_frames: List of frames in original order
            original_frame_idx: Index of frame in original sequence
            tile_pbar: Optional progress bar for tile processing
            
        Returns:
            Optical flow for the specified frame in original sequence
        """
        # Special case: frame0 cannot have backward flow (no frame-1 exists)
        # Solution: use the flow from frame1 → frame0 (copy flow from frame1)
        if original_frame_idx == 0:
            if len(original_frames) == 1:
                # Only one frame - create zero flow
                height, width = original_frames[0].shape[:2]
                return np.zeros((height, width, 2), dtype=np.float32)
            else:
                # Use flow from frame1 (frame1 → frame0 is what we want for frame0)
                return self.compute_optical_flow_with_progress_for_frame(original_frames, 1, tile_pbar)
        
        # Create reversed frames for flow computation
        reversed_frames = original_frames[::-1]
        
        # Map original frame index to reversed frame index
        reversed_frame_idx = len(original_frames) - 1 - original_frame_idx
        
        # Compute flow using reversed frames and reversed index
        flow = self.inference_engine.compute_optical_flow_with_progress(
            reversed_frames, reversed_frame_idx, tile_pbar
        )
        
        return flow


class VideoFlowProcessor:
    def __init__(self, device='auto', fast_mode=False, tile_mode=False, sequence_length=5,
                 flow_model='videoflow', model_path=None, stage='sintel', 
                 vf_dataset='sintel', vf_architecture='mof', vf_variant='standard',
                 motion_vectors_clamp_range=32.0, flow_input=None, enable_long_term=False):
        """Initialize optical flow processor with model selection support"""
        # Initialize device manager
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device(device)
            
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self.flow_model = flow_model.lower()
        self.stage = stage
        
        # VideoFlow specific parameters
        self.vf_dataset = vf_dataset
        self.vf_architecture = vf_architecture
        self.vf_variant = vf_variant
        
        # Initialize inference engine using factory pattern
        if self.flow_model == 'memflow':
            # MemFlow parameters
            self.inference_engine = FlowProcessorFactory.create_inference(
                model='memflow',
                device=self.device,
                fast_mode=fast_mode,
                tile_mode=tile_mode,
                sequence_length=sequence_length,
                stage=stage,
                model_path=model_path,
                enable_long_term=enable_long_term
            )
            # For MemFlow, the actual model path will be determined by the core
            # We'll show a placeholder here and update it after model loading
            model_path_display = model_path or f'MemFlow_ckpt/MemFlowNet_{stage}.pth'
            print(f"Using MemFlow model: {model_path_display} (will be auto-selected)")
            if enable_long_term:
                print(f"Long-term memory: ENABLED")
            else:
                print(f"Long-term memory: DISABLED (default)")
        elif self.flow_model == 'videoflow':
            # VideoFlow parameters
            self.inference_engine = FlowProcessorFactory.create_inference(
                model='videoflow',
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
        
        # TAA compression emulation setting

        
        # Motion vectors encoding parameters
        self.motion_vectors_clamp_range = motion_vectors_clamp_range
        
        # Flow input video path (for TAA comparison with external flow)
        self.flow_input = flow_input
        
        # Store enable_long_term for cache path generation
        self.enable_long_term = enable_long_term
        
        # TAA processors for consistent state management
        self.taa_flow_processor = TAAProcessor(alpha=0.1)
        self.taa_simple_processor = TAAProcessor(alpha=0.1)
        self.taa_external_processor = TAAProcessor(alpha=0.1)
        
        # Initialize storage manager
        self.cache_manager = FlowCacheManager()
        
        # Initialize video composer
        self.video_composer = VideoComposer()
        
        # Initialize reversed video flow computer for backward optical flow computation
        self.reversed_flow_computer = None  # Will be initialized when model is loaded
        
        print(f"Optical Flow Processor initialized - Device: {self.device}")
        self.device_manager.print_device_info()
        print(f"Model: {self.flow_model.upper()}")
        print(f"Fast mode: {fast_mode}")
        if self.flow_model == 'videoflow':
            print(f"Tile mode: {tile_mode}")
        else:
            print(f"Tile mode: Not used (MemFlow processes full frames)")
        print(f"Sequence length: {sequence_length} frames")
        if self.flow_model == 'memflow':
            print(f"Long-term memory: {'Enabled' if enable_long_term else 'Disabled'}")
        
    def load_model(self):
        """Load optical flow model (VideoFlow or MemFlow)"""
        self.inference_engine.load_model()
        
        # Initialize reversed video flow computer after model is loaded
        self.reversed_flow_computer = ReversedVideoFlowComputer(self.inference_engine)
        print(f"[Model] Initialized reversed video flow computation wrapper")
        
        # For MemFlow, show the actual loaded model path
        if self.flow_model == 'memflow' and hasattr(self.inference_engine, 'model_path'):
            print(f"MemFlow model actually loaded: {self.inference_engine.model_path}")
        
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
        # Add long-term memory info for MemFlow
        extra_params = {}
        if self.flow_model == 'memflow':
            extra_params['long_term'] = 'lt' if self.enable_long_term else 'st'
        
        return self.cache_manager.generate_cache_path(
            input_path, start_frame, max_frames, sequence_length, fast_mode, tile_mode,
            model=self.flow_model, dataset=self.vf_dataset if self.flow_model == 'videoflow' else self.stage,
            architecture=self.vf_architecture if self.flow_model == 'videoflow' else 'none',
            variant=self.vf_variant if self.flow_model == 'videoflow' else 'none',
            extra_params=extra_params
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
    
    def encode_motion_vectors_rg8_format(self, flow, width, height):
        """Encode optical flow in motion vectors RG8 format using encoding module"""
        from encoding.flow_encoders import FlowEncoderFactory
        encoder = FlowEncoderFactory.create_encoder('motion-vectors-rg8', clamp_range=self.motion_vectors_clamp_range)
        return encoder.encode(flow, width, height)
    
    def encode_motion_vectors_rgb8_format(self, flow, width, height):
        """Encode optical flow in motion vectors RGB8 format using encoding module"""
        from encoding.flow_encoders import FlowEncoderFactory
        encoder = FlowEncoderFactory.create_encoder('motion-vectors-rgb8', clamp_range=self.motion_vectors_clamp_range)
        return encoder.encode(flow, width, height)
    

    
    def encode_torchvision_format(self, flow, width, height):
        """Encode optical flow using torchvision format using encoding module"""
        return encode_flow(flow, width, height, 'torchvision')
    
    def extract_flow_from_video(self, video_path, max_frames=1000, start_frame=0, 
                               start_time=None, duration=None, flow_format='motion-vectors-rg8'):
        """
        Extract and decode motion vectors from bottom half of video frames
        
        Args:
            video_path: Path to video with encoded flow in bottom half
            max_frames: Maximum number of frames to extract
            start_frame: Starting frame number
            start_time: Starting time in seconds
            duration: Duration in seconds
            flow_format: Format of encoded flow ('motion-vectors-rg8' or 'motion-vectors-rgb8')
            
        Returns:
            List of decoded flow arrays (H, W, 2) for each frame
        """
        # Extract frames from flow video
        frame_extractor = FrameExtractor(video_path, fast_mode=self.fast_mode)
        flow_frames, fps, width, height, actual_start = frame_extractor.extract_frames(
            max_frames=max_frames, 
            start_frame=start_frame,
            start_time=start_time,
            duration=duration
        )
        
        # Calculate original frame dimensions (top half)
        original_height = height // 2
        
        decoded_flows = []
        
        print(f"Extracting flow from {len(flow_frames)} frames...")
        
        for i, frame in enumerate(flow_frames):
            # Extract bottom half containing encoded flow
            encoded_flow = frame[original_height:, :, :]
            
            # Decode flow based on format
            if flow_format == 'motion-vectors-rg8':
                from encoding.flow_encoders import decode_motion_vectors
                decoded_flow = decode_motion_vectors(encoded_flow, 
                                                   clamp_range=self.motion_vectors_clamp_range, 
                                                   format_variant='rg8')
            elif flow_format == 'motion-vectors-rgb8':
                from encoding.flow_encoders import decode_motion_vectors
                decoded_flow = decode_motion_vectors(encoded_flow, 
                                                   clamp_range=self.motion_vectors_clamp_range, 
                                                   format_variant='rgb8')
            else:
                raise ValueError(f"Unsupported flow format: {flow_format}")
            
            decoded_flows.append(decoded_flow)
        
        return decoded_flows
    
    def create_difference_overlay(self, original_flow, decoded_flow, magnitude_threshold=0.9):
        """
        Create difference overlay between original and decoded flow using radar colors
        
        Args:
            original_flow: Original optical flow (H, W, 2)
            decoded_flow: Decoded optical flow (H, W, 2)
            magnitude_threshold: Threshold for significant differences (not used in new version)
            
        Returns:
            Difference overlay image (H, W, 3) with radar color coding and legend overlay at bottom
        """
        import cv2
        
        # Calculate flow differences
        diff_flow = original_flow - decoded_flow
        
        # Calculate magnitude of differences
        diff_magnitude = np.sqrt(diff_flow[:, :, 0]**2 + diff_flow[:, :, 1]**2)
        
        # Create overlay image
        h, w = diff_flow.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Define error levels and corresponding radar colors (RGB)
        error_levels = [0.1, 0.5, 1.0, 2.0, 4.0]
        radar_colors = [
            (0, 255, 0),      # Green (very small errors)
            (255, 255, 0),    # Yellow (small errors)
            (255, 165, 0),    # Orange (medium errors)
            (255, 0, 0),      # Red (large errors)
            (255, 0, 255),    # Magenta (very large errors)
        ]
        
        # Apply radar colors based on error magnitude
        for i, (level, color) in enumerate(zip(error_levels, radar_colors)):
            if i == 0:
                # First level: errors <= 0.001
                mask = diff_magnitude <= level
            elif i == len(error_levels) - 1:
                # Last level: errors > 0.2
                mask = diff_magnitude > error_levels[i-1]
            else:
                # Middle levels: previous_level < errors <= current_level
                mask = (diff_magnitude > error_levels[i-1]) & (diff_magnitude <= level)
            
            overlay[mask] = color
        
        # Create compact legend with small colored squares
        legend_height = 25
        legend_y_start = h - legend_height
        
        # Position legend squares in bottom-left corner
        square_size = 12
        square_spacing = 45
        start_x = 10
        start_y = h - 20
        
        # Draw color squares with white border and numbers
        for i, (level, color) in enumerate(zip(error_levels, radar_colors)):
            x_pos = start_x + i * square_spacing
            
            # Draw white border (square + 2 pixels)
            cv2.rectangle(overlay, (x_pos - 1, start_y - square_size - 1), 
                         (x_pos + square_size + 1, start_y + 1), (255, 255, 255), -1)
            
            # Draw colored square
            cv2.rectangle(overlay, (x_pos, start_y - square_size), 
                         (x_pos + square_size, start_y), color, -1)
            
            # Add number label
            if i == len(error_levels) - 1:
                label = f">{error_levels[i-1]:.3f}"
            else:
                label = f"{level:.3f}"
            
            # Position number to the right of square
            text_x = x_pos + square_size + 3
            text_y = start_y - 4
            
            # Add black shadow
            cv2.putText(overlay, label, (text_x + 1, text_y + 1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            # Add white text
            cv2.putText(overlay, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return overlay
    
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
        
    def create_side_by_side(self, original, flow_viz, flow_only=False, 
                           taa_frame=None, taa_simple_frame=None, model_name="VideoFlow", fast_mode=False, flow_format="gamedev"):
        """Create side-by-side, flow-only, or TAA visualization with text overlays"""
        return self.video_composer.create_side_by_side(
            original, flow_viz, flow_only, 
            taa_frame, taa_simple_frame, model_name, fast_mode, flow_format
        )
        

    def generate_output_filename(self, input_path, output_dir, start_time=None, duration=None, 
                                start_frame=0, max_frames=1000, flow_only=False, taa=False, uncompressed=False, flow_format='gamedev', fps=30.0):
        """Generate automatic output filename based on parameters"""
        from storage.filename_generator import generate_output_filepath
        
        # Add long-term memory info for MemFlow
        extra_params = {}
        if self.flow_model == 'memflow' and self.enable_long_term:
            extra_params['long_term'] = 'lt'
        
        return generate_output_filepath(
            input_path=input_path,
            output_dir=output_dir,
            start_time=start_time,
            duration=duration,
            start_frame=start_frame,
            max_frames=max_frames,
            flow_only=flow_only,
            taa=taa,
            fast_mode=self.fast_mode,
            tile_mode=self.tile_mode,
            uncompressed=uncompressed,
            flow_format=flow_format,
            motion_vectors_clamp_range=self.motion_vectors_clamp_range,
            fps=fps,
            model=self.flow_model,
            stage=self.stage,
            vf_dataset=self.vf_dataset,
            vf_architecture=self.vf_architecture,
            vf_variant=self.vf_variant,
            extra_params=extra_params
        )
    
    def process_video(self, input_path, output_path, max_frames=1000, start_frame=0, 
                     start_time=None, duration=None, flow_only=False, taa=False, flow_format='gamedev', 
                     save_flow=None, force_recompute=False, use_flow_cache=None, auto_play=True,
                     skip_lods=False, uncompressed=False, flow_input=None):
        """Main processing function"""
        import os
        from video.frame_extractor import FrameExtractor
        
        # Flow input processing (override from self.flow_input if provided)
        if flow_input is None:
            flow_input = self.flow_input
        
        # Check flow input and prepare configuration
        if flow_input is not None:
            if not os.path.exists(flow_input):
                raise ValueError(f"Flow input video not found: {flow_input}")
            
            if not taa:
                print("Warning: --flow-input requires --taa to be enabled. Enabling TAA mode.")
                taa = True
            
            print(f"[Configuration] Flow Input Processing:")
            print(f"  Flow Input Video: {flow_input}")
            print(f"  Flow Format: {flow_format}")
            print(f"  Clamp Range: ±{self.motion_vectors_clamp_range} pixels")
            print(f"  Mode: TAA comparison with external flow")
            print()
        

        
        # Handle time-based parameters
        fps = None
        if start_time is not None or duration is not None:
            fps = self.get_video_fps(input_path)
            print(f"Video FPS: {fps:.2f}")
            
            if start_time is not None:
                start_frame = self.time_to_frame(start_time, fps)
                print(f"Start time: {start_time}s -> frame {start_frame}")
            
            if duration is not None:
                max_frames = self.time_to_frame(duration, fps)
                print(f"Duration: {duration}s -> {max_frames} frames")
        
        # Extract frames first to get accurate FPS
        frame_extractor = FrameExtractor(input_path, fast_mode=self.fast_mode)
        frames, fps, width, height, actual_start = frame_extractor.extract_frames(
            max_frames=max_frames, 
            start_frame=start_frame,
            start_time=start_time,
            duration=duration
        )
        
        print(f"[Processing] Extracted {len(frames)} frames from video")
        print(f"[Processing] Will compute backward optical flow using reversed video processing")
        print(f"              (Forward flow on reversed video = Backward flow on original video)")
        
        # Check if output_path is a directory and generate filename if needed
        if os.path.isdir(output_path):
            output_path = self.generate_output_filename(
                input_path, output_path, start_time, duration, 
                start_frame, max_frames, flow_only, taa, uncompressed=uncompressed, flow_format=flow_format, fps=fps
            )
            print(f"Auto-generated output filename: {os.path.basename(output_path)}")
        
        print(f"Processing: {input_path} -> {output_path}")
        print(f"Frame range: {start_frame} to {start_frame + max_frames - 1}")
        print(f"Video FPS: {fps:.2f}")
        
        # Log encoder configuration
        print(f"\n[Configuration] Flow Encoding:")
        if flow_format == 'hsv':
            print(f"  Format: HSV color space visualization")
        elif flow_format == 'torchvision':
            print(f"  Format: TorchVision color wheel visualization")
        elif flow_format == 'motion-vectors-rg8':
            print(f"  Format: Motion Vectors RG8 (normalized XY in RG channels)")
            print(f"  Clamp Range: ±{self.motion_vectors_clamp_range} pixels")
        elif flow_format == 'motion-vectors-rgb8':
            print(f"  Format: Motion Vectors RGB8 (direction+magnitude)")
            print(f"  Clamp Range: ±{self.motion_vectors_clamp_range} pixels")
        else:
            print(f"  Format: GameDev RG channels (legacy)")
        
        if taa:
            print(f"\n[Configuration] TAA Processing:")
            print(f"  TAA Enabled: True")
            print(f"  TAA Encoder: Native float32 precision")
        

        
        print()  # Add blank line for readability
        
        # Extract flow from external video if provided
        decoded_flows = None
        if flow_input is not None:
            print(f"[Flow Input] Extracting flow from external video...")
            
            # Get info about flow input video using VideoInfo (no frame extraction)
            from video.video_info import VideoInfo
            flow_video_info = VideoInfo(flow_input)
            flow_info = flow_video_info.get_info()
            flow_input_total_frames = flow_info['total_frames']
            
            # We need exactly len(frames) flow frames
            frames_needed = len(frames)
            flow_frames_to_extract = min(frames_needed, flow_input_total_frames)
            
            print(f"  Flow input has {flow_input_total_frames} frames")
            print(f"  Main video has {len(frames)} frames")
            print(f"  Will extract {flow_frames_to_extract} flow frames starting from frame 0")
            
            # Extract flow from external video (always start from frame 0)
            decoded_flows = self.extract_flow_from_video(
                flow_input, flow_frames_to_extract, 0, None, None, flow_format
            )
            
            if len(decoded_flows) == 0:
                raise ValueError("No flow data could be extracted from flow input video")
            
            # If flow input is shorter than main video, extend with last frame
            if len(decoded_flows) < frames_needed:
                last_flow = decoded_flows[-1]
                original_length = len(decoded_flows)
                while len(decoded_flows) < frames_needed:
                    decoded_flows.append(last_flow.copy())
                print(f"  Warning: Flow input shorter than main video. Extended from {original_length} to {len(decoded_flows)} frames using last frame.")
            
            # If flow input is longer than main video, trim to match
            elif len(decoded_flows) > frames_needed:
                decoded_flows = decoded_flows[:frames_needed]
                print(f"  Flow input longer than main video. Trimmed to {len(decoded_flows)} frames.")
            
            print(f"  Successfully prepared {len(decoded_flows)} flow frames")
            print()
        
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
        elif os.path.exists(output_path) and os.path.isdir(output_path):
            # Check if output directory is an existing flow cache
            cache_exists, cached_flow_format, missing_frames = self.check_flow_cache_exists(output_path, len(frames))
            if cache_exists:
                print(f"Detected existing flow cache at output path: {output_path}")
                flow_cache_dir = output_path
                use_cached_flow = True
                # Generate new output path for video since output_path is now used as cache
                output_dir = os.path.dirname(output_path)
                cache_name = os.path.basename(output_path)
                video_name = f"{cache_name}_taa_output.avi"
                output_path = os.path.join(output_dir, video_name)
                print(f"Video will be saved to: {output_path}")
            else:
                print(f"Output directory exists but is not a valid flow cache, generating new cache...")
                # Generate automatic cache directory
                flow_cache_dir = self.generate_flow_cache_path(
                    input_path, start_frame, len(frames), self.sequence_length, 
                    self.fast_mode, self.tile_mode
                )
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
            print(f"[Configuration] Optical Flow Model:")
            print(f"  Model: {self.flow_model}")
            if hasattr(self, 'stage'):
                print(f"  Stage: {self.stage}")
            if self.flow_model == 'videoflow':
                print(f"  Architecture: {self.vf_architecture}")
                print(f"  Dataset: {self.vf_dataset}")
                print(f"  Variant: {self.vf_variant}")
            if self.fast_mode:
                print(f"  Fast Mode: Enabled (lower resolution, fewer iterations)")
            if self.tile_mode:
                print(f"  Tile Mode: Enabled (process in tiles for large frames)")
            print(f"  Device: {self.device}")
            print(f"  Processing: Reversed video for backward optical flow")
            print()
            
            self.load_model()
            
            # Verify reversed flow computer is initialized
            if self.reversed_flow_computer is None:
                raise RuntimeError("Reversed flow computer not initialized after model loading")
        
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
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print("Using MJPG codec. Output will be .avi for compatibility.")
        
        if flow_only:
            output_size = (width, height * 2)  # Stack: Original on top, Flow on bottom
        elif taa:
            if flow_input is not None:
                # Flow input mode: 6 videos in 2x3 grid
                output_size = (width * 2, height * 3)  # 2x3 grid: double width, triple height
            else:
                # Standard TAA mode: 4 videos
                output_size = (width * 2, height * 2)  # 2x2 grid: double width, double height
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
        self.taa_external_processor.reset_history()

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
            main_pbar = tqdm(total=len(frames), desc="Flow processing", 
                           unit="frame", ncols=100, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            tile_pbar = None
            overall_tile_pbar = None
        
        for i in range(len(frames)):
            start_time = time.time()
            
            # Get optical flow (from cache or compute using reversed video)
            if use_cached_flow:
                # Load from cache (raw flow before stabilization)
                raw_flow = self.load_cached_flow(flow_cache_dir, i, cached_flow_format)
            else:
                # Compute optical flow using REVERSED VIDEO processing
                # This computes forward flow on reversed video = backward flow on original video
                # The result is saved with original frame index for proper caching/visualization
                if self.tile_mode:
                    # Reset overall tile progress for each frame
                    overall_tile_pbar.reset()
                    raw_flow = self.reversed_flow_computer.compute_optical_flow_for_frame(
                        frames, i, tile_pbar, overall_tile_pbar
                    )
                else:
                    raw_flow = self.reversed_flow_computer.compute_optical_flow_for_frame(
                        frames, i
                    )
                
                # Save raw flow to cache BEFORE stabilization
                self.cache_manager.save_flow_to_cache(raw_flow, flow_cache_dir, i, cache_save_format)
            
            # Save raw optical flow if explicitly requested (before stabilization)
            if save_flow is not None and flow_base_filename is not None:
                self.save_optical_flow_files(raw_flow, flow_base_filename, i, save_flow)
            
            # No stabilization - use raw flow
            flow = raw_flow
            
            # Encode optical flow based on selected format
            if i == 0:  # Log encoder info only once
                if flow_format == 'hsv':
                    print(f"[Encoder] Using HSV color space encoder")
                elif flow_format == 'torchvision':
                    print(f"[Encoder] Using TorchVision color wheel encoder")
                elif flow_format == 'motion-vectors-rg8':
                    print(f"[Encoder] Using Motion Vectors RG8 encoder (clamp_range={self.motion_vectors_clamp_range})")
                elif flow_format == 'motion-vectors-rgb8':
                    print(f"[Encoder] Using Motion Vectors RGB8 encoder (direction+magnitude format, clamp_range={self.motion_vectors_clamp_range})")
                else:
                    print(f"[Encoder] Using GameDev RG channel encoder")
            
            # Use the current frame's flow for visualization
            if flow_format == 'hsv':
                flow_viz = self.encode_hsv_format(flow, width, height)
            elif flow_format == 'torchvision':
                flow_viz = self.encode_torchvision_format(flow, width, height)
            elif flow_format == 'motion-vectors-rg8':
                flow_viz = self.encode_motion_vectors_rg8_format(flow, width, height)
            elif flow_format == 'motion-vectors-rgb8':
                flow_viz = self.encode_motion_vectors_rgb8_format(flow, width, height)
            else:
                flow_viz = self.encode_gamedev_format(flow, width, height)
            
            # Apply TAA effects if requested
            taa_frame = None
            taa_simple_frame = None
            taa_external_frame = None  # TAA with external flow
            difference_overlay = None  # Difference overlay
            external_flow_viz = None  # External flow visualization

            if taa:
                # Normal TAA processing with current computed flow
                taa_result = self.taa_flow_processor.apply_taa(
                    current_frame=frames[i],
                    flow_pixels=flow,
                    previous_taa_frame=None,
                    alpha=0.1,
                    use_flow=True,
                    sequence_id='flow_taa'
                )
                taa_frame = taa_result
                
                # Apply TAA with external flow if provided
                if flow_input is not None and i < len(decoded_flows):
                    external_flow = decoded_flows[i]
                    taa_external_result = self.taa_external_processor.apply_taa(
                        current_frame=frames[i],
                        flow_pixels=external_flow,
                        previous_taa_frame=None,
                        alpha=0.1,
                        use_flow=True,
                        sequence_id='external_taa'
                    )
                    taa_external_frame = taa_external_result
                    
                    # Create visualization of external flow (using same format as original flow)
                    if flow_format == 'hsv':
                        external_flow_viz = self.encode_hsv_format(external_flow, width, height)
                    elif flow_format == 'torchvision':
                        external_flow_viz = self.encode_torchvision_format(external_flow, width, height)
                    elif flow_format == 'motion-vectors-rg8':
                        external_flow_viz = self.encode_motion_vectors_rg8_format(external_flow, width, height)
                    elif flow_format == 'motion-vectors-rgb8':
                        external_flow_viz = self.encode_motion_vectors_rgb8_format(external_flow, width, height)
                    else:
                        external_flow_viz = self.encode_gamedev_format(external_flow, width, height)
                    
                    # Create difference overlay between original and external flow
                    difference_overlay = self.create_difference_overlay(flow, external_flow, magnitude_threshold=0.5)

                # Apply simple TAA (no flow) with alpha=0.1
                taa_simple_result = self.taa_simple_processor.apply_taa(
                    current_frame=frames[i],
                    flow_pixels=None,
                    previous_taa_frame=None,
                    alpha=0.1,
                    use_flow=False,
                    sequence_id='simple_taa'
                )
                taa_simple_frame = taa_simple_result
                
            # Create combined frame
            if flow_input is not None and taa_external_frame is not None and difference_overlay is not None:
                # Flow input mode: create 6-video grid
                # Use external flow visualization if available, otherwise fall back to original flow
                display_flow_viz = external_flow_viz if external_flow_viz is not None else flow_viz
                combined = self.create_6_video_grid(
                    frames[i], display_flow_viz, taa_frame, taa_simple_frame, 
                    taa_external_frame, difference_overlay
                )
            else:
                if self.flow_model == 'memflow':
                    model_name = f"MemFlow ({self.stage})"
                else:  # videoflow
                    variant_str = f"/{self.vf_variant}" if self.vf_variant == 'noise' and self.vf_dataset == 'things' else ""
                    model_name = f"{self.vf_architecture.upper()} ({self.vf_dataset}{variant_str})"
                combined = self.create_side_by_side(frames[i], flow_viz, flow_only=flow_only, 
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
                main_pbar.set_description(f"Flow processing (ETA: {eta_seconds:.0f}s)")
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
    
    def create_6_video_grid(self, original_frame, flow_viz, taa_frame, taa_simple_frame, 
                           taa_external_frame, difference_overlay):
        """
        Create 6-video grid layout for flow input mode
        
        Args:
            original_frame: Original video frame (RGB)
            flow_viz: External flow visualization (RGB) - shows the loaded flow visualization
            taa_frame: TAA with original flow (RGB, may be float)
            taa_simple_frame: TAA without flow (RGB, may be float)
            taa_external_frame: TAA with external flow (RGB, may be float)
            difference_overlay: Difference overlay between flows (RGB)
            
        Returns:
            Combined frame with 6 videos (BGR for video output)
        """
        import cv2
        h, w = original_frame.shape[:2]
        
        # Convert all frames to BGR format and ensure proper data types
        # (matching the processing done in create_side_by_side)
        orig_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
        flow_bgr = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)
        taa_bgr = cv2.cvtColor(np.clip(taa_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        taa_simple_bgr = cv2.cvtColor(np.clip(taa_simple_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        taa_external_bgr = cv2.cvtColor(np.clip(taa_external_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        diff_bgr = cv2.cvtColor(difference_overlay, cv2.COLOR_RGB2BGR)
        
        # 2x3 grid: 2 columns, 3 rows
        combined = np.zeros((h * 3, w * 2, 3), dtype=np.uint8)
        
        # Top row
        combined[0:h, 0:w, :] = orig_bgr
        combined[0:h, w:2*w, :] = flow_bgr
        
        # Middle row
        combined[h:2*h, 0:w, :] = taa_bgr
        combined[h:2*h, w:2*w, :] = taa_simple_bgr
        
        # Bottom row
        combined[2*h:3*h, 0:w, :] = taa_external_bgr
        combined[2*h:3*h, w:2*w, :] = diff_bgr
        
        # Add text labels
        combined = self.add_text_overlay(combined, "Original", (10, 10))
        combined = self.add_text_overlay(combined, "External Flow", (w + 10, 10))
        combined = self.add_text_overlay(combined, "TAA + Original Flow", (10, h + 10))
        combined = self.add_text_overlay(combined, "TAA Simple", (w + 10, h + 10))
        combined = self.add_text_overlay(combined, "TAA + External Flow", (10, 2*h + 10))
        combined = self.add_text_overlay(combined, "Flow Difference", (w + 10, 2*h + 10))
        
        return combined

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
    parser.add_argument('--flow-only', action='store_true',
                       help='Output only optical flow visualization (no original video)')
    parser.add_argument('--taa', action='store_true',
                       help='Add TAA (Temporal Anti-Aliasing) effect visualization using inverted optical flow from previous frame')
    parser.add_argument('--flow-input', type=str, default=None,
                       help='Input video with encoded motion vectors in bottom half (for TAA comparison with external flow)')
    parser.add_argument('--flow-format', choices=['gamedev', 'hsv', 'torchvision', 'motion-vectors-rg8', 'motion-vectors-rgb8'], default='gamedev',
                       help='Optical flow encoding format: gamedev (RG channels), hsv (standard visualization), torchvision (color wheel), motion-vectors-rg8 (RG channels normalized), motion-vectors-rgb8 (direction+magnitude)')
    parser.add_argument('--motion-vectors-clamp-range', type=float, default=32.0,
                       help='Clamp range for motion-vectors encoding formats (default: 32.0)')
    parser.add_argument('--tile', action='store_true',
                       help='Enable tile-based processing: split frames into 1280x1280 square tiles (optimal for VideoFlow MOF model)')
    parser.add_argument('--sequence-length', type=int, default=5,
                       help='Number of frames to use in sequence for VideoFlow processing (default: 5, recommended: 5-9)')
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
    parser.add_argument('--skip-lods', action='store_true',
                       help='Skip LOD (Level-of-Detail) pyramid generation/loading for faster processing')
    parser.add_argument('--uncompressed', action='store_true',
                        help='Save the output video completely uncompressed (raw frames in .avi container)')
    parser.add_argument('--model', choices=['videoflow', 'memflow'], default='videoflow',
                        help='Choose optical flow model: videoflow (VideoFlow MOF) or memflow (MemFlow)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Custom path to model weights (only for MemFlow)')
    parser.add_argument('--stage', choices=['sintel', 'things', 'kitti'], default='sintel',
                        help='Training stage/dataset (default: sintel, affects model selection for MemFlow)')
    parser.add_argument('--enable-long-term', action='store_true',
                        help='Enable long-term memory for MemFlow (only affects MemFlow model)')
    
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
                                      sequence_length=args.sequence_length,
                                      flow_model=args.model, model_path=args.model_path, stage=args.stage,
                                      vf_dataset=args.vf_dataset, vf_architecture=args.vf_architecture, 
                                      vf_variant=args.vf_variant,
                                      motion_vectors_clamp_range=args.motion_vectors_clamp_range,
                                      flow_input=args.flow_input,
                                      enable_long_term=args.enable_long_term)
        
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
        elif os.path.exists(args.output) and os.path.isdir(args.output):
            # Check if output directory is an existing flow cache
            cache_exists, cached_flow_format, missing_frames = processor.check_flow_cache_exists(args.output, len(frames))
            if cache_exists:
                print(f"Detected existing flow cache at output path: {args.output}")
                flow_cache_dir = args.output
            else:
                print(f"Output directory exists but is not a valid flow cache, generating new cache...")
                flow_cache_dir = processor.generate_flow_cache_path(
                    args.input, start_frame, len(frames), args.sequence_length, 
                    args.fast, args.tile
                )
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
                                          sequence_length=args.sequence_length,
                                          flow_model=args.model, model_path=args.model_path, stage=args.stage,
                                          vf_dataset=args.vf_dataset, vf_architecture=args.vf_architecture, 
                                          vf_variant=args.vf_variant,
                                          motion_vectors_clamp_range=args.motion_vectors_clamp_range,
                                          flow_input=args.flow_input,
                                          enable_long_term=args.enable_long_term)
        
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
                                  sequence_length=args.sequence_length,
                                  flow_model=args.model, model_path=args.model_path, stage=args.stage,
                                  vf_dataset=args.vf_dataset, vf_architecture=args.vf_architecture, 
                                  vf_variant=args.vf_variant,
                                  motion_vectors_clamp_range=args.motion_vectors_clamp_range,
                                  flow_input=args.flow_input,
                                  enable_long_term=args.enable_long_term)
    
    try:
        # Create output filename with frame/time range if not specified
        if args.output == 'videoflow_result.mp4':  # Default output pattern (will be changed to .avi)
            # Default output name, add range info
            mode = ""
            if args.fast:
                mode += "_fast"
            if args.tile:
                mode += "_tile"

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
                              start_time=args.start_time, duration=args.duration, 
                              flow_only=args.flow_only, taa=args.taa, flow_format=args.flow_format, save_flow=args.save_flow,
                              force_recompute=args.force_recompute, use_flow_cache=args.use_flow_cache, 
                              auto_play=not args.no_autoplay,
                              skip_lods=args.skip_lods,
                              uncompressed=args.uncompressed, flow_input=args.flow_input)
        
        model_name = args.model.upper()
        if not args.no_autoplay:
            print(f"\n[SUCCESS] {model_name} processing completed successfully! Video should open automatically.")
        else:
            print(f"\n[SUCCESS] {model_name} processing completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 