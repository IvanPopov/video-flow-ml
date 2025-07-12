"""
Synthetic Video Generator for Optical Flow Testing

Generates synthetic videos with moving objects and precise ground truth data
for testing optical flow models.
"""

import os
import sys
import numpy as np
import cv2
import math
from typing import Dict, Any, Tuple

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)


class SyntheticVideoGenerator:
    """Generate synthetic video with moving ball for optical flow testing"""
    
    def __init__(self, width: int = 504, height: int = 216, fps: int = 30):
        """
        Initialize video generator
        
        Args:
            width: Video width in pixels (default: 504 for 21:9 aspect ratio)
            height: Video height in pixels (default: 216 for 21:9 aspect ratio)
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.ball_radius = 20
        self.ball_color = (255, 255, 255)  # White ball
        self.bg_color = (64, 64, 64)       # Dark gray background
        
        # Motion parameters (will be set based on speed)
        self.max_velocity = 0.0  # pixels per frame
        self.motion_period = 0.0  # frames for one complete cycle
        
    def set_motion_parameters(self, speed: str):
        """Set motion parameters based on speed setting"""
        if speed == 'slow':
            self.max_velocity = 2.0    # 2 pixels per frame
            self.motion_period = 120   # 4 seconds at 30fps
        elif speed == 'medium':
            self.max_velocity = 5.0    # 5 pixels per frame
            self.motion_period = 90    # 3 seconds at 30fps
        elif speed == 'fast':
            self.max_velocity = 10.0   # 10 pixels per frame
            self.motion_period = 60    # 2 seconds at 30fps
        else:
            raise ValueError(f"Unknown speed: {speed}")
    
    def set_custom_motion_parameters(self, max_velocity: float, motion_period: float):
        """Set custom motion parameters"""
        self.max_velocity = max_velocity
        self.motion_period = motion_period
    
    def calculate_position_and_velocity(self, frame_idx: int, total_frames: int) -> Tuple[float, float, float, float]:
        """
        Calculate ball position and velocity for given frame
        
        Args:
            frame_idx: Current frame index (0-based)
            total_frames: Total number of frames
            
        Returns:
            Tuple of (x_position, y_position, x_velocity, y_velocity)
        """
        # Normalize time to [0, 1] over the video duration
        t = frame_idx / (total_frames - 1)
        
        # Create sinusoidal motion with multiple cycles
        cycles = total_frames / self.motion_period
        phase = 2 * math.pi * cycles * t
        
        # Position: sinusoidal motion in X, centered in Y
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Motion range (keep ball within frame bounds)
        motion_range = self.width // 2 - self.ball_radius - 20
        
        x_pos = center_x + motion_range * math.sin(phase)
        y_pos = center_y  # Keep Y position fixed for simpler analysis
        
        # Velocity: derivative of position
        x_velocity = motion_range * math.cos(phase) * (2 * math.pi * cycles / (total_frames - 1))
        y_velocity = 0.0  # No Y motion
        
        return x_pos, y_pos, x_velocity, y_velocity
    
    def generate_video(self, output_path: str, total_frames: int, speed: str = None, 
                      custom_params: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Generate synthetic video with moving ball
        
        Args:
            output_path: Path to output video file
            total_frames: Number of frames to generate
            speed: Motion speed ('slow', 'medium', 'fast') - ignored if custom_params provided
            custom_params: Tuple of (max_velocity, motion_period) for custom motion
            
        Returns:
            Dictionary with ground truth data
        """
        if custom_params:
            self.set_custom_motion_parameters(custom_params[0], custom_params[1])
            speed_name = f"custom_{custom_params[0]:.1f}px_{custom_params[1]:.0f}period"
        else:
            if speed is None:
                raise ValueError("Must provide either speed or custom_params")
            self.set_motion_parameters(speed)
            speed_name = speed
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                'speed': speed_name,
                'max_velocity': self.max_velocity,
                'motion_period': self.motion_period
            },
            'frames': []
        }
        
        print(f"Generating {total_frames} frames with {speed_name} motion...")
        
        for frame_idx in range(total_frames):
            # Calculate ball position and velocity
            x_pos, y_pos, x_vel, y_vel = self.calculate_position_and_velocity(frame_idx, total_frames)
            
            # Create frame
            frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
            
            # Draw ball
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius, self.ball_color, -1)
            
            # Add slight anti-aliasing for smoother motion
            cv2.circle(frame, (int(x_pos), int(y_pos)), self.ball_radius + 1, 
                      (self.ball_color[0]//2, self.ball_color[1]//2, self.ball_color[2]//2), 1)
            
            # Write frame
            writer.write(frame)
            
            # Store ground truth
            ground_truth['frames'].append({
                'frame_idx': frame_idx,
                'ball_center': (x_pos, y_pos),
                'ball_velocity': (x_vel, y_vel),
                'ball_radius': self.ball_radius
            })
        
        writer.release()
        
        print(f"Video generated: {output_path}")
        print(f"Motion parameters: max_velocity={self.max_velocity:.1f} px/frame, period={self.motion_period} frames")
        
        return ground_truth 