"""
Video Composer - handles video composition, text overlays, and grid layouts
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Union


class VideoComposer:
    """Main class for video composition operations"""
    
    def __init__(self):
        """Initialize video composer"""
        pass
    
    def add_text_overlay(self, frame: np.ndarray, text: str, position: Union[str, Tuple[int, int]] = 'top-left', 
                        font_scale: float = 0.4, color: Tuple[int, int, int] = (255, 255, 255), 
                        thickness: int = 1) -> np.ndarray:
        """
        Add text overlay to frame
        
        Args:
            frame: Input frame (BGR format for OpenCV)
            text: Text to add
            position: 'top-left', 'top-right', 'bottom-left', 'bottom-right' or tuple (x, y)
            font_scale: Size of the font
            color: Text color (BGR)
            thickness: Text thickness
        
        Returns:
            Frame with text overlay
        """
        if frame is None:
            return frame
            
        frame_with_text = frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Calculate position
        margin = 5
        if isinstance(position, tuple):
            # Direct coordinates provided
            pos = position
        elif position == 'top-left':
            pos = (margin, text_size[1] + margin)
        elif position == 'top-right':
            pos = (w - text_size[0] - margin, text_size[1] + margin)
        elif position == 'bottom-left':
            pos = (margin, h - margin)
        elif position == 'bottom-right':
            pos = (w - text_size[0] - margin, h - margin)
        else:
            pos = (margin, text_size[1] + margin)  # Default to top-left
        
        # Add black outline for better visibility
        cv2.putText(frame_with_text, text, pos, font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Add white text
        cv2.putText(frame_with_text, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
        
        return frame_with_text
    
    def create_side_by_side(self, original: np.ndarray, flow_viz: np.ndarray, 
                          vertical: bool = False, flow_only: bool = False,
                          taa_frame: Optional[np.ndarray] = None, 
                          taa_simple_frame: Optional[np.ndarray] = None,
                          model_name: str = "VideoFlow", fast_mode: bool = False, 
                          flow_format: str = "gamedev") -> np.ndarray:
        """Create side-by-side, top-bottom, flow-only, or TAA visualization with text overlays"""
        # Ensure same dimensions
        h, w = original.shape[:2]
        if flow_viz.shape[:2] != (h, w):
            flow_viz = cv2.resize(flow_viz, (w, h))
        
        # Convert to BGR for video writing
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        flow_bgr = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)
        
        if flow_only:
            # Return only optical flow without text overlays
            return flow_bgr
        
        # Add text overlays (only when not flow_only)
        mode_text = " (Fast)" if fast_mode else ""
        
        orig_bgr = self.add_text_overlay(orig_bgr, f"Original{mode_text}", 'top-left')
        flow_bgr = self.add_text_overlay(flow_bgr, f"Optical Flow{mode_text}", 'top-left')
        flow_bgr = self.add_text_overlay(flow_bgr, f"{model_name} ({flow_format.upper()})", 'bottom-left')
        
        if taa_frame is not None and taa_simple_frame is not None:
            # Both TAA modes: flow-based and simple
            taa_bgr = cv2.cvtColor(np.clip(taa_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            taa_simple_bgr = cv2.cvtColor(np.clip(taa_simple_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Add TAA text overlays
            taa_bgr = self.add_text_overlay(taa_bgr, "TAA + Inv.Flow", 'top-left')
            taa_bgr = self.add_text_overlay(taa_bgr, "Alpha: 0.1", 'bottom-left')
            
            taa_simple_bgr = self.add_text_overlay(taa_simple_bgr, "TAA Simple", 'top-left')
            taa_simple_bgr = self.add_text_overlay(taa_simple_bgr, "Alpha: 0.1", 'bottom-left')
            
            if vertical:
                # Stack vertically: Original | Flow | TAA+Flow | TAA Simple
                return np.concatenate([orig_bgr, flow_bgr, taa_bgr, taa_simple_bgr], axis=0)
            else:
                # Create 2x2 grid layout
                # Top row: Original | Flow
                top_row = np.concatenate([orig_bgr, flow_bgr], axis=1)
                # Bottom row: TAA+Flow | TAA Simple
                bottom_row = np.concatenate([taa_bgr, taa_simple_bgr], axis=1)
                # Stack rows vertically
                return np.concatenate([top_row, bottom_row], axis=0)
        elif taa_frame is not None:
            # Single TAA mode (backward compatibility)
            taa_bgr = cv2.cvtColor(np.clip(taa_frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            taa_bgr = self.add_text_overlay(taa_bgr, "TAA + Inv.Flow", 'top-left')
            taa_bgr = self.add_text_overlay(taa_bgr, "Alpha: 0.1", 'bottom-left')
            
            if vertical:
                # Stack vertically: Original | Flow | TAA
                return np.concatenate([orig_bgr, flow_bgr, taa_bgr], axis=0)
            else:
                # Stack horizontally: Original | Flow | TAA
                return np.concatenate([orig_bgr, flow_bgr, taa_bgr], axis=1)
        else:
            if vertical:
                # Concatenate vertically (top-bottom)
                return np.concatenate([orig_bgr, flow_bgr], axis=0)
            else:
                # Concatenate horizontally (side-by-side)
                return np.concatenate([orig_bgr, flow_bgr], axis=1)
    
    def create_video_grid(self, frames_dict: Dict[str, np.ndarray], 
                         grid_shape: Tuple[int, int], target_aspect: float = 16/9) -> np.ndarray:
        """
        Arrange multiple video frames into a grid with a target aspect ratio.
        
        Args:
            frames_dict: Dictionary {'label': frame} of frames to arrange.
            grid_shape: (rows, cols) for the grid layout.
            target_aspect: Target aspect ratio for the final output.
            
        Returns:
            A single frame containing the grid.
        """
        if not frames_dict:
            return None
            
        rows, cols = grid_shape
        
        # Get dimensions from the first frame
        first_frame = next(iter(frames_dict.values()))
        h, w = first_frame.shape[:2]
        
        # Calculate canvas size to match target aspect ratio
        canvas_w = cols * w
        canvas_h = int(canvas_w / target_aspect)
        
        # Create black canvas
        grid_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Calculate cell size and position
        cell_h = h
        cell_w = w
        
        y_offset = (canvas_h - rows * cell_h) // 2
        x_offset = (canvas_w - cols * cell_w) // 2
        
        frames = list(frames_dict.items())
        
        for i in range(rows * cols):
            if i >= len(frames):
                break
                
            label, frame = frames[i]
            
            row = i // cols
            col = i % cols
            
            y_start = y_offset + row * cell_h
            x_start = x_offset + col * cell_w
            
            # Convert frame to BGR format for video output (handle different input formats)
            if label == 'Flow Viz':
                # Flow visualization is in RGB format, convert to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif 'TAA-' in label:
                # TAA frames might be in float format, ensure uint8 and convert RGB to BGR
                frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
                if len(frame_uint8.shape) == 3 and frame_uint8.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_uint8
            else:
                # Original frames are typically in RGB format, convert to BGR
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
            
            # Add text label to frame with multi-line support
            labeled_frame = frame_bgr.copy()
            lines = label.split('\n')
            font_scale = 0.7  # Increase font size
            thickness = 2
            line_height = 30  # Increase line height
            start_y = 25
            
            # Add dark background for better readability
            max_text_width = 0
            for line in lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                max_text_width = max(max_text_width, text_size[0])
            
            # Draw semi-transparent dark background
            overlay = labeled_frame.copy()
            cv2.rectangle(overlay, (0, 0), (max_text_width + 15, len(lines) * line_height + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, labeled_frame, 0.3, 0, labeled_frame)
            
            for line_idx, line in enumerate(lines):
                y_pos = start_y + line_idx * line_height
                # Add black outline for contrast
                cv2.putText(labeled_frame, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # Add white text
                cv2.putText(labeled_frame, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Place frame on canvas
            if y_start + cell_h <= canvas_h and x_start + cell_w <= canvas_w:
                grid_canvas[y_start:y_start+cell_h, x_start:x_start+cell_w] = labeled_frame
                
        return grid_canvas
    



# Convenience functions for direct usage
def add_text_overlay(frame: np.ndarray, text: str, position: Union[str, Tuple[int, int]] = 'top-left', 
                    font_scale: float = 0.4, color: Tuple[int, int, int] = (255, 255, 255), 
                    thickness: int = 1) -> np.ndarray:
    """Add text overlay to frame"""
    composer = VideoComposer()
    return composer.add_text_overlay(frame, text, position, font_scale, color, thickness)


def create_side_by_side(original: np.ndarray, flow_viz: np.ndarray, 
                       vertical: bool = False, flow_only: bool = False,
                       taa_frame: Optional[np.ndarray] = None, 
                       taa_simple_frame: Optional[np.ndarray] = None,
                       model_name: str = "VideoFlow", fast_mode: bool = False, 
                       flow_format: str = "gamedev") -> np.ndarray:
    """Create side-by-side composition"""
    composer = VideoComposer()
    return composer.create_side_by_side(original, flow_viz, vertical, flow_only, 
                                      taa_frame, taa_simple_frame, model_name, 
                                      fast_mode, flow_format)


def create_video_grid(frames_dict: Dict[str, np.ndarray], 
                     grid_shape: Tuple[int, int], target_aspect: float = 16/9) -> np.ndarray:
    """Create video grid layout"""
    composer = VideoComposer()
    return composer.create_video_grid(frames_dict, grid_shape, target_aspect) 