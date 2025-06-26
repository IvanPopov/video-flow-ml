"""
Frame Extractor - video frame extraction with fast mode support
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from .video_info import VideoInfo


class FrameExtractor:
    """Video frame extractor with fast mode and time-based extraction"""
    
    def __init__(self, video_path: str, fast_mode: bool = False):
        """
        Initialize frame extractor
        
        Args:
            video_path: Path to video file
            fast_mode: Enable aggressive resolution reduction
        """
        self.video_info = VideoInfo(video_path)
        self.fast_mode = fast_mode
    
    def calculate_fast_mode_dimensions(self, orig_width: int, orig_height: int) -> Tuple[int, int, float]:
        """
        Calculate dimensions for fast mode processing
        
        Args:
            orig_width: Original video width
            orig_height: Original video height
            
        Returns:
            Tuple of (new_width, new_height, scale_factor)
        """
        if not self.fast_mode:
            return orig_width, orig_height, 1.0
        
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
        
        return width, height, scale_factor
    
    def extract_frames(self, 
                      max_frames: int = 1000, 
                      start_frame: int = 0,
                      start_time: Optional[float] = None,
                      duration: Optional[float] = None) -> Tuple[List[np.ndarray], float, int, int, int]:
        """
        Extract frames from video
        
        Args:
            max_frames: Maximum number of frames to extract
            start_frame: Starting frame number (0-based)
            start_time: Starting time in seconds (overrides start_frame)
            duration: Duration in seconds (overrides max_frames)
            
        Returns:
            Tuple of (frames_list, fps, width, height, actual_start_frame)
        """
        info = self.video_info.get_info()
        fps = info['fps']
        orig_width = info['width']
        orig_height = info['height']
        total_frames = info['total_frames']
        
        # Handle time-based parameters
        if start_time is not None:
            start_frame = self.video_info.time_to_frame(start_time)
            print(f"Start time: {start_time}s -> frame {start_frame}")
        
        if duration is not None:
            max_frames = self.video_info.time_to_frame(duration)
            print(f"Duration: {duration}s -> {max_frames} frames")
        
        # Validate and adjust frame range
        start_frame, frames_to_extract = self.video_info.validate_frame_range(start_frame, max_frames)
        end_frame = start_frame + frames_to_extract
        
        # Calculate dimensions for fast mode
        width, height, scale_factor = self.calculate_fast_mode_dimensions(orig_width, orig_height)
        
        if self.fast_mode:
            print(f"Fast mode: aggressive resolution reduction from {orig_width}x{orig_height} to {width}x{height} (scale: {scale_factor:.2f})")
        
        # Open video capture
        cap = cv2.VideoCapture(str(self.video_info.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_info.video_path}")
        
        try:
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Extract frames
            frames = []
            pbar = tqdm(total=frames_to_extract, desc="Extracting frames")
            
            for i in range(frames_to_extract):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could only extract {i} frames out of {frames_to_extract}")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if in fast mode
                if self.fast_mode and scale_factor != 1.0:
                    frame_rgb = cv2.resize(frame_rgb, (width, height))
                
                frames.append(frame_rgb)
                pbar.update(1)
            
            pbar.close()
            
            print(f"Frame range: {start_frame} to {start_frame + len(frames) - 1}")
            
            return frames, fps, width, height, start_frame
            
        finally:
            cap.release()
    
    def extract_time_range(self, 
                          start_time: float, 
                          duration: float) -> Tuple[List[np.ndarray], float, int, int, int]:
        """
        Extract frames for a specific time range
        
        Args:
            start_time: Starting time in seconds
            duration: Duration in seconds
            
        Returns:
            Tuple of (frames_list, fps, width, height, actual_start_frame)
        """
        return self.extract_frames(start_time=start_time, duration=duration)
    
    def get_frame_at_time(self, time_seconds: float) -> np.ndarray:
        """
        Extract a single frame at specific time
        
        Args:
            time_seconds: Time in seconds
            
        Returns:
            Single frame as numpy array
        """
        frame_number = self.video_info.time_to_frame(time_seconds)
        
        cap = cv2.VideoCapture(str(self.video_info.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_info.video_path}")
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Cannot read frame at time {time_seconds}s (frame {frame_number})")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply fast mode if enabled
            if self.fast_mode:
                info = self.video_info.get_info()
                width, height, _ = self.calculate_fast_mode_dimensions(info['width'], info['height'])
                frame_rgb = cv2.resize(frame_rgb, (width, height))
            
            return frame_rgb
            
        finally:
            cap.release()
    
    def print_extraction_info(self, frames_count: int, start_frame: int, fps: float):
        """Print extraction information"""
        info = self.video_info.get_info()
        
        print(f"Video properties: {info['width']}x{info['height']} @ {fps:.2f} FPS")
        print(f"Extracting {frames_count} frames starting from frame {start_frame}")
        
        if self.fast_mode:
            width, height, scale = self.calculate_fast_mode_dimensions(info['width'], info['height'])
            if scale != 1.0:
                print(f"Fast mode: {info['width']}x{info['height']} -> {width}x{height} (scale: {scale:.2f})") 