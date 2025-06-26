"""
Video Info - video properties and information extraction
"""

import cv2
from typing import Tuple, Dict, Any
from pathlib import Path


class VideoInfo:
    """Video information extractor and utilities"""
    
    def __init__(self, video_path: str):
        """
        Initialize with video path
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self._info_cache = None
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive video information
        
        Returns:
            Dictionary with video properties
        """
        if self._info_cache is not None:
            return self._info_cache
            
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        try:
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration_seconds': None,
                'path': str(self.video_path)
            }
            
            # Calculate duration if FPS is available
            if info['fps'] > 0:
                info['duration_seconds'] = info['total_frames'] / info['fps']
            
            self._info_cache = info
            return info
            
        finally:
            cap.release()
    
    def get_fps(self) -> float:
        """Get video FPS"""
        return self.get_info()['fps']
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get video dimensions (width, height)"""
        info = self.get_info()
        return info['width'], info['height']
    
    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return self.get_info()['total_frames']
    
    def get_duration(self) -> float:
        """Get video duration in seconds"""
        duration = self.get_info()['duration_seconds']
        if duration is None:
            raise ValueError("Cannot calculate duration: invalid FPS")
        return duration
    
    def time_to_frame(self, time_seconds: float) -> int:
        """
        Convert time in seconds to frame number
        
        Args:
            time_seconds: Time in seconds
            
        Returns:
            Frame number (0-based)
        """
        fps = self.get_fps()
        if fps <= 0:
            raise ValueError("Cannot convert time to frame: invalid FPS")
        return int(time_seconds * fps)
    
    def frame_to_time(self, frame_number: int) -> float:
        """
        Convert frame number to time in seconds
        
        Args:
            frame_number: Frame number (0-based)
            
        Returns:
            Time in seconds
        """
        fps = self.get_fps()
        if fps <= 0:
            raise ValueError("Cannot convert frame to time: invalid FPS")
        return frame_number / fps
    
    def validate_frame_range(self, start_frame: int, frame_count: int) -> Tuple[int, int]:
        """
        Validate and adjust frame range
        
        Args:
            start_frame: Starting frame number
            frame_count: Number of frames to extract
            
        Returns:
            Tuple of (adjusted_start_frame, adjusted_frame_count)
        """
        total_frames = self.get_frame_count()
        
        # Validate start frame
        if start_frame < 0:
            start_frame = 0
        elif start_frame >= total_frames:
            raise ValueError(f"Start frame {start_frame} exceeds total frames {total_frames}")
        
        # Adjust frame count
        max_frames = total_frames - start_frame
        adjusted_frame_count = min(frame_count, max_frames)
        
        return start_frame, adjusted_frame_count
    
    def print_info(self):
        """Print video information to console"""
        info = self.get_info()
        
        print(f"Video: {info['path']}")
        print(f"Dimensions: {info['width']}x{info['height']}")
        print(f"FPS: {info['fps']:.2f}")
        print(f"Total frames: {info['total_frames']}")
        if info['duration_seconds']:
            print(f"Duration: {info['duration_seconds']:.2f}s")
    
    def reset_cache(self):
        """Reset cached video information"""
        self._info_cache = None 