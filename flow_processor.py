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

# Import VideoFlow modules
from core.Networks import build_network
from utils.utils import InputPadder
from configs.multiframes_sintel_submission import get_cfg

class VideoFlowProcessor:
    def __init__(self, device='auto', fast_mode=False):
        """Initialize VideoFlow processor with pure VideoFlow implementation"""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = device
            
        self.fast_mode = fast_mode
        self.model = None
        self.input_padder = None
        self.cfg = None
        
        print(f"VideoFlow Processor initialized - Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Fast mode: {fast_mode}")
        
    def load_videoflow_model(self):
        """Load VideoFlow MOF model"""
        print("Loading VideoFlow MOF model...")
        
        # Get VideoFlow configuration
        print("  Getting configuration...")
        self.cfg = get_cfg()
        
        # Apply fast mode optimizations
        if self.fast_mode:
            print("  Applying fast mode optimizations...")
            self.cfg.decoder_depth = 6  # Reduce from default 12
            self.cfg.corr_levels = 3    # Reduce correlation levels
            self.cfg.corr_radius = 3    # Reduce correlation radius
            print(f"    Decoder depth: {self.cfg.decoder_depth}")
            print(f"    Correlation levels: {self.cfg.corr_levels}")
            print(f"    Correlation radius: {self.cfg.corr_radius}")
        
        print(f"  Model path: {self.cfg.model}")
        
        # Check if model weights exist
        model_path = self.cfg.model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        # Build network
        print("  Building network...")
        self.model = build_network(self.cfg)
        print("  Network built successfully")
        
        # Load pre-trained weights
        print("  Loading weights...")
        checkpoint = torch.load(model_path, map_location=self.device)
        print(f"  Checkpoint loaded, keys count: {len(checkpoint.keys())}")
        
        # Remove 'module.' prefix from keys if present (for models trained with DataParallel)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            print("  Removing 'module.' prefix from checkpoint keys...")
            checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
        
        print("  Loading state dict...")
        self.model.load_state_dict(checkpoint)
        
        # Move to device and set evaluation mode
        print(f"  Moving model to {self.device}...")
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ VideoFlow MOF model loaded successfully")
        
    def get_video_fps(self, video_path):
        """Get video FPS for time calculations"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    
    def time_to_frame(self, time_seconds, fps):
        """Convert time in seconds to frame number"""
        return int(time_seconds * fps)
    
    def extract_frames(self, video_path, max_frames=1000, start_frame=0):
        """Extract frames from video starting at start_frame"""
        end_frame = start_frame + max_frames
        print(f"Extracting frames {start_frame} to {end_frame-1} from {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check bounds
        if start_frame >= total_frames:
            raise ValueError(f"Start frame {start_frame} exceeds total frames {total_frames}")
        
        actual_end = min(end_frame, total_frames)
        frames_to_extract = actual_end - start_frame
        
        # Apply fast mode resolution reduction
        if self.fast_mode:
            # More aggressive resolution reduction for fast mode
            # Target maximum 256x256, but maintain aspect ratio
            max_dimension = 256
            scale_factor = min(max_dimension / orig_width, max_dimension / orig_height)
            
            # Don't upscale if already small
            if scale_factor > 1.0:
                scale_factor = 1.0
            
            # Apply additional reduction for large videos
            if max(orig_width, orig_height) > 512:
                scale_factor = min(scale_factor, 0.25)  # Quarter size for very large videos
            elif max(orig_width, orig_height) > 256:
                scale_factor = min(scale_factor, 0.5)   # Half size for medium videos
            
            width = int(orig_width * scale_factor)
            height = int(orig_height * scale_factor)
            
            # Ensure dimensions are even (required for some codecs) and minimum 64x64
            width = max(64, width - (width % 2))
            height = max(64, height - (height % 2))
            
            print(f"Fast mode: aggressive resolution reduction from {orig_width}x{orig_height} to {width}x{height} (scale: {scale_factor:.2f})")
        else:
            width = orig_width
            height = orig_height
        
        print(f"Video: {orig_width}x{orig_height} -> {width}x{height}, {fps:.1f}fps")
        print(f"Extracting frames {start_frame}-{actual_end-1} ({frames_to_extract} frames) from {total_frames} total frames")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frames = []
        pbar = tqdm(total=frames_to_extract, desc="Extracting frames")
        
        for i in range(frames_to_extract):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if in fast mode
            if self.fast_mode:
                frame_rgb = cv2.resize(frame_rgb, (width, height))
            
            frames.append(frame_rgb)
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        print(f"✓ Extracted {len(frames)} frames")
        return frames, fps, width, height, start_frame
        
    def prepare_frame_sequence(self, frames, frame_idx):
        """Prepare 5-frame sequence for VideoFlow MOF model"""
        # Multi-frame: use 5 consecutive frames centered around current frame
        start_idx = max(0, frame_idx - 2)
        end_idx = min(len(frames), frame_idx + 3)
        sequence = frames[start_idx:end_idx]
        
        # Pad to exactly 5 frames
        while len(sequence) < 5:
            if start_idx == 0:
                sequence.insert(0, sequence[0])
            else:
                sequence.append(sequence[-1])
        
        # Ensure exactly 5 frames
        sequence = sequence[:5]
        
        # Convert to tensors (same format as VideoFlow inference.py)
        tensors = []
        for frame in sequence:
            # Convert to tensor and normalize to [0,1], then change HWC to CHW
            tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
            tensors.append(tensor)
        
        # Stack frames and add batch dimension
        batch = torch.stack(tensors).unsqueeze(0).to(self.device)
        return batch
        
    def compute_optical_flow(self, frames, frame_idx):
        """Compute optical flow using VideoFlow model"""
        # Prepare frame sequence
        frame_batch = self.prepare_frame_sequence(frames, frame_idx)
        
        # Create input padder
        padder = InputPadder(frame_batch.shape[-2:])
        frame_batch_padded = padder.pad(frame_batch)
        
        # Run VideoFlow inference (following their inference structure)
        with torch.no_grad():
            # VideoFlow forward pass
            flow_predictions, _ = self.model(frame_batch_padded, {})
            
            # Unpad results
            flow_tensor = padder.unpad(flow_predictions)
            
            # Get the middle flow (index 2 out of 0-4 for 5 frames)
            # Since we want flow for the center frame
            middle_idx = flow_tensor.shape[1] // 2
            flow_tensor = flow_tensor[0, middle_idx]  # Remove batch dim and get middle flow
            
            # Convert to numpy: CHW -> HWC  
            flow_np = flow_tensor.permute(1, 2, 0).cpu().numpy()
            
        return flow_np
        
    def encode_gamedev_format(self, flow, width, height):
        """
        Encode optical flow in gamedev format:
        - Normalize flow by image dimensions
        - Scale and clamp to [-20, +20] range  
        - Map to [0, 1] where 0 = -20, 1 = +20
        - Store in RG channels (R=horizontal, G=vertical)
        """
        # Normalize flow by image dimensions
        norm_flow = flow.copy()
        norm_flow[:, :, 0] /= width    # Horizontal flow
        norm_flow[:, :, 1] /= height   # Vertical flow
        
        # Scale to make motion visible
        norm_flow *= 200
        
        # Clamp to [-20, +20] range
        clamped = np.clip(norm_flow, -20, 20)
        
        # Map [-20, +20] to [0, 1]: 0 = -20, 1 = +20
        encoded = (clamped + 20) / 40
        encoded = np.clip(encoded, 0, 1)
        
        # Create RGB image
        h, w = flow.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, 0] = encoded[:, :, 0]  # R channel: horizontal flow
        rgb[:, :, 1] = encoded[:, :, 1]  # G channel: vertical flow
        rgb[:, :, 2] = 0.0               # B channel: unused
        
        # Convert to 8-bit, handle NaN and inf values
        rgb_8bit = rgb * 255
        rgb_8bit = np.nan_to_num(rgb_8bit, nan=0.0, posinf=255.0, neginf=0.0)
        return rgb_8bit.astype(np.uint8)
        
    def create_side_by_side(self, original, flow_viz, vertical=False, flow_only=False):
        """Create side-by-side, top-bottom, or flow-only visualization"""
        # Ensure same dimensions
        h, w = original.shape[:2]
        if flow_viz.shape[:2] != (h, w):
            flow_viz = cv2.resize(flow_viz, (w, h))
        
        # Convert to BGR for video writing
        flow_bgr = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)
        
        if flow_only:
            # Return only optical flow
            return flow_bgr
        
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        if vertical:
            # Concatenate vertically (top-bottom)
            return np.concatenate([orig_bgr, flow_bgr], axis=0)
        else:
            # Concatenate horizontally (side-by-side)
            return np.concatenate([orig_bgr, flow_bgr], axis=1)
        
    def process_video(self, input_path, output_path, max_frames=1000, start_frame=0, 
                     start_time=None, duration=None, vertical=False, flow_only=False):
        """Main processing function"""
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
        
        print(f"Processing: {input_path} -> {output_path}")
        print(f"Frame range: {start_frame} to {start_frame + max_frames - 1}")
        
        # Load VideoFlow model
        self.load_videoflow_model()
        
        # Extract frames
        frames, fps, width, height, actual_start = self.extract_frames(input_path, max_frames=max_frames, start_frame=start_frame)
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if flow_only:
            output_size = (width, height)  # Flow only: original dimensions
        elif vertical:
            output_size = (width, height * 2)  # Vertical: same width, double height
        else:
            output_size = (width * 2, height)  # Horizontal: double width, same height
        out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        
        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")
        
        # Process each frame
        print("Processing frames with VideoFlow...")
        
        import time
        frame_times = []
        
        # Create progress bar
        pbar = tqdm(total=len(frames), desc="VideoFlow processing", 
                   unit="frame", ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i in range(len(frames)):
            start_time = time.time()
            
            # Compute optical flow using VideoFlow
            flow = self.compute_optical_flow(frames, i)
            
            # Encode in gamedev format
            flow_viz = self.encode_gamedev_format(flow, width, height)
            
            # Create combined frame (side-by-side, top-bottom, or flow-only)
            combined = self.create_side_by_side(frames[i], flow_viz, vertical=vertical, flow_only=flow_only)
            
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
            pbar.set_description(f"VideoFlow processing (ETA: {eta_seconds:.0f}s)")
            pbar.update(1)
        
        pbar.close()
        out.release()
        
        print(f"✓ VideoFlow processing completed!")
        print(f"✓ Output saved: {output_path}")
        if flow_only:
            print("Output: Optical flow only (VideoFlow + gamedev encoding)")
        elif vertical:
            print("Top: Original video")
            print("Bottom: Optical flow (VideoFlow + gamedev encoding)")
        else:
            print("Left side: Original video")
            print("Right side: Optical flow (VideoFlow + gamedev encoding)")
        print("  R channel: Horizontal flow (-20 to +20)")
        print("  G channel: Vertical flow (-20 to +20)")

def main():
    parser = argparse.ArgumentParser(description='VideoFlow Optical Flow Processor')
    parser.add_argument('--input', default='big_buck_bunny_720p_h264.mov',
                       help='Input video file')
    parser.add_argument('--output', default='videoflow_result.mp4',
                       help='Output video file')
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    if not os.path.exists('VideoFlow'):
        print("Error: VideoFlow repository not found. Please run:")
        print("git clone https://github.com/XiaoyuShi97/VideoFlow.git")
        return
        
    if not os.path.exists('VideoFlow_ckpt/MOF_sintel.pth'):
        print("Error: VideoFlow model weights not found.")
        print("Please download MOF_sintel.pth from:")
        print("https://github.com/XiaoyuShi97/VideoFlow")
        print("and place it in VideoFlow_ckpt/")
        return
    
    processor = VideoFlowProcessor(device=args.device, fast_mode=args.fast)
    
    try:
        # Create output filename with frame/time range if not specified
        if args.output == 'videoflow_result.mp4':
            # Default output name, add range info
            mode = ""
            if args.fast:
                mode += "_fast"
            if args.vertical:
                mode += "_vertical"
            if args.flow_only:
                mode += "_flow_only"
            
            if args.start_time is not None or args.duration is not None:
                # Use time-based naming
                fps = processor.get_video_fps(args.input)
                start_frame = processor.time_to_frame(args.start_time, fps) if args.start_time is not None else args.start_frame
                max_frames = processor.time_to_frame(args.duration, fps) if args.duration is not None else args.frames
                end_frame = start_frame + max_frames - 1
                
                start_time_str = f"{args.start_time:.1f}s" if args.start_time is not None else f"{start_frame}f"
                duration_str = f"{args.duration:.1f}s" if args.duration is not None else f"{max_frames}f"
                args.output = f"videoflow_{start_time_str}_{duration_str}{mode}.mp4"
            else:
                # Use frame-based naming
                end_frame = args.start_frame + args.frames - 1
                args.output = f"videoflow_{args.start_frame:06d}_{end_frame:06d}{mode}.mp4"
        
        processor.process_video(args.input, args.output, max_frames=args.frames, start_frame=args.start_frame,
                              start_time=args.start_time, duration=args.duration, vertical=args.vertical, 
                              flow_only=args.flow_only)
        print("\n✓ VideoFlow processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 