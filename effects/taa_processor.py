"""
TAA Processor - Temporal Anti-Aliasing effects with optical flow
"""

import numpy as np
import cv2
from typing import Optional, Tuple

# Import motion vectors compression/decompression functions
try:
    from encoding.flow_encoders import encode_motion_vectors, decode_motion_vectors
except ImportError:
    # Fallback if encoding module is not available
    def encode_motion_vectors(flow, clamp_range=64.0, format_variant='rg8'):
        raise ImportError("Motion vectors encoding not available")
    def decode_motion_vectors(encoded_flow, clamp_range=64.0, format_variant='rg8'):
        raise ImportError("Motion vectors decoding not available")


class TAAProcessor:
    """
    Temporal Anti-Aliasing processor with optical flow support
    
    Provides both simple temporal blending and motion-compensated TAA
    using optical flow for accurate pixel reprojection.
    """
    
    def __init__(self, alpha: float = 0.1, bilateral_sigma_color: float = 25.0):
        """
        Initialize TAA processor
        
        Args:
            alpha: Blending weight (0.0 = full history, 1.0 = no history)
            bilateral_sigma_color: Color similarity sigma for bilateral filtering.
                                   Lower values are more selective.
        """
        self.alpha = alpha
        self.bilateral_sigma_color = bilateral_sigma_color
        self.history = {}  # Store history for multiple sequences
    
    def apply_taa(self, 
                  current_frame: np.ndarray,
                  flow_pixels: Optional[np.ndarray] = None,
                  previous_taa_frame: Optional[np.ndarray] = None,
                  alpha: Optional[float] = None,
                  use_flow: bool = True,
                  use_bilateral: bool = True,
                  sequence_id: str = 'default') -> np.ndarray:
        """
        Apply TAA effect with or without optical flow
        
        Args:
            current_frame: Current frame (RGB, 0-255)
            flow_pixels: Inverted optical flow from previous frame (HWC, normalized)
            previous_taa_frame: Previous TAA result frame (if None, uses internal history)
            alpha: Blending weight (if None, uses instance default)
            use_flow: Whether to use optical flow for reprojection
            use_bilateral: Whether to use bilateral filtering for reprojection
            sequence_id: Identifier for sequence (for multi-sequence processing)
        
        Returns:
            TAA processed frame (float32)
        """
        alpha = alpha if alpha is not None else self.alpha
        
        # Use provided previous frame or get from history
        if previous_taa_frame is None:
            previous_taa_frame = self.history.get(sequence_id)
        
        if previous_taa_frame is None:
            # First frame, no history
            result = current_frame.astype(np.float32)
            self.history[sequence_id] = result
            return result
        
        current_float = current_frame.astype(np.float32)
        
        if not use_flow or flow_pixels is None:
            # Simple temporal blending without flow (basic TAA)
            result = alpha * current_float + (1 - alpha) * previous_taa_frame
        else:
            # Flow-based TAA (motion-compensated)
            result = self._apply_flow_based_taa(
                current_float, flow_pixels, previous_taa_frame, alpha, use_bilateral
            )
        
        # Update history
        self.history[sequence_id] = result
        return result
    

    def _apply_flow_based_taa(self,
                             current_frame: np.ndarray,
                             flow_pixels: np.ndarray,
                             previous_taa_frame: np.ndarray,
                             alpha: float,
                             use_bilateral: bool) -> np.ndarray:
        """
        Apply flow-based TAA with motion compensation
        
        Args:
            current_frame: Current frame (float32)
            flow_pixels: Optical flow pixels (HWC)
            previous_taa_frame: Previous TAA frame (float32)
            alpha: Blending weight
            use_bilateral: Whether to use bilateral filtering
            
        Returns:
            TAA processed frame
        """

        h, w = current_frame.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Calculate previous pixel positions using flow
        prev_x = x_coords + flow_pixels[:, :, 0]
        prev_y = y_coords + flow_pixels[:, :, 1]
        
        # Identify pixels where the flow vector points out of bounds
        out_of_bounds_mask = (prev_x < 0) | (prev_x >= w) | (prev_y < 0) | (prev_y >= h)
        
        # Handle NaN and inf values
        prev_x = np.nan_to_num(prev_x, nan=0.0, posinf=w-1, neginf=0.0)
        prev_y = np.nan_to_num(prev_y, nan=0.0, posinf=h-1, neginf=0.0)
        
        # Clamp coordinates to valid range
        prev_x = np.clip(prev_x, 0, w - 1)
        prev_y = np.clip(prev_y, 0, h - 1)
        
        if use_bilateral:
            reprojected = self._bilateral_reprojection_sample(
                previous_taa_frame, prev_x, prev_y, current_frame
            )
        else:
            reprojected = self._bilinear_sample(previous_taa_frame, prev_x, prev_y)
        
        # Exponential moving average (TAA blending)
        taa_result = alpha * current_frame + (1 - alpha) * reprojected
        
        # For out-of-bounds pixels, set to green and disable smoothing
        # green_color = np.array([0.0, 255.0, 0.0], dtype=np.float32)
        #if np.any(out_of_bounds_mask):
        #    taa_result[out_of_bounds_mask] = green_color
        
        return taa_result
    
    def _bilateral_reprojection_sample(self,
                                      image: np.ndarray,
                                      x_coords: np.ndarray,
                                      y_coords: np.ndarray,
                                      current_frame: np.ndarray) -> np.ndarray:
        """
        Perform bilateral reprojection from image at given coordinates,
        weighted by color similarity to the current frame.
        
        Args:
            image: Source image (H, W, C), i.e., previous TAA frame
            x_coords: X coordinates for sampling
            y_coords: Y coordinates for sampling
            current_frame: Current frame for color similarity check
            
        Returns:
            Sampled image
        """
        h, w = image.shape[:2]
        
        # Get integer coordinates for the 4 neighbors
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        
        # Clamp to ensure x1, y1 are within bounds
        x0 = np.clip(x0, 0, w - 2)
        y0 = np.clip(y0, 0, h - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        # Interpolation weights (spatial)
        wx = (x_coords - x0)[..., np.newaxis]
        wy = (y_coords - y0)[..., np.newaxis]
        
        # Get the four neighbors from the previous frame
        p00 = image[y0, x0]
        p01 = image[y0, x1]
        p10 = image[y1, x0]
        p11 = image[y1, x1]
        
        # --- Color similarity weighting ---
        current_lum = np.mean(current_frame, axis=2)
        
        def get_color_weight(neighbor_pixel):
            lum_diff = current_lum - np.mean(neighbor_pixel, axis=2)
            # Use a small epsilon to avoid division by zero if sigma is 0
            sigma_sq = self.bilateral_sigma_color**2 * 0.1
            return np.exp(-lum_diff**2 / (2 * sigma_sq + 1e-6))[..., np.newaxis]

        w00_color = get_color_weight(p00)
        w01_color = get_color_weight(p01)
        w10_color = get_color_weight(p10)
        w11_color = get_color_weight(p11)

        # Combine spatial and color weights
        w00 = (1 - wx) * (1 - wy) * w00_color
        w01 = wx * (1 - wy) * w01_color
        w10 = (1 - wx) * wy * w10_color
        w11 = wx * wy * w11_color
        
        total_weight = w00 + w01 + w10 + w11
        # Avoid division by zero
        total_weight = np.where(total_weight == 0, 1e-6, total_weight)
        
        # Weighted average of neighbors
        reprojected = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11) / total_weight
        
        return reprojected
    
    def _bilinear_sample(self,
                        image: np.ndarray,
                        x_coords: np.ndarray,
                        y_coords: np.ndarray) -> np.ndarray:
        """
        Perform bilinear sampling from image at given coordinates
        
        Args:
            image: Source image (H, W, C)
            x_coords: X coordinates for sampling
            y_coords: Y coordinates for sampling
            
        Returns:
            Sampled image
        """
        h, w = image.shape[:2]
        
        # Get integer coordinates
        x0 = np.floor(x_coords).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(y_coords).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        
        # Ensure indices are valid
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        
        # Interpolation weights
        wx = x_coords - x0
        wy = y_coords - y0
        
        # Sample with bilinear interpolation
        result = np.zeros_like(image, dtype=np.float32)
        
        for c in range(image.shape[2]):  # For each channel
            result[:, :, c] = (
                image[y0, x0, c] * (1 - wx) * (1 - wy) +
                image[y0, x1, c] * wx * (1 - wy) +
                image[y1, x0, c] * (1 - wx) * wy +
                image[y1, x1, c] * wx * wy
            )
        
        return result
    
    def apply_simple_taa(self,
                        current_frame: np.ndarray,
                        previous_taa_frame: Optional[np.ndarray] = None,
                        alpha: Optional[float] = None,
                        sequence_id: str = 'simple') -> np.ndarray:
        """
        Apply simple TAA without optical flow (basic temporal blending)
        
        Args:
            current_frame: Current frame (RGB, 0-255)
            previous_taa_frame: Previous TAA result frame
            alpha: Blending weight
            sequence_id: Identifier for sequence
            
        Returns:
            TAA processed frame (float32)
        """
        return self.apply_taa(
            current_frame=current_frame,
            flow_pixels=None,
            previous_taa_frame=previous_taa_frame,
            alpha=alpha,
            use_flow=False,
            use_bilateral=False,
            sequence_id=sequence_id
        )
    
    def reset_history(self, sequence_id: Optional[str] = None):
        """
        Reset TAA history
        
        Args:
            sequence_id: Specific sequence to reset (if None, resets all)
        """
        if sequence_id is None:
            self.history.clear()
        else:
            self.history.pop(sequence_id, None)
    
    def get_history(self, sequence_id: str = 'default') -> Optional[np.ndarray]:
        """
        Get TAA history for a sequence
        
        Args:
            sequence_id: Sequence identifier
            
        Returns:
            Previous TAA frame or None if no history
        """
        return self.history.get(sequence_id)
    
    def set_alpha(self, alpha: float):
        """
        Set default alpha blending weight
        
        Args:
            alpha: Blending weight (0.0 = full history, 1.0 = no history)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self.alpha = alpha


class TAAComparisonProcessor:
    """
    TAA comparison processor for side-by-side comparison of different TAA methods
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize comparison processor
        
        Args:
            alpha: Default blending weight
        """
        self.flow_taa = TAAProcessor(alpha)
        self.simple_taa = TAAProcessor(alpha)  # Simple TAA doesn't use flow
    
    def apply_comparison(self,
                        current_frame: np.ndarray,
                        flow_pixels: Optional[np.ndarray] = None,
                        alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply both flow-based and simple TAA for comparison
        
        Args:
            current_frame: Current frame (RGB, 0-255)
            flow_pixels: Optical flow pixels
            alpha: Blending weight
            
        Returns:
            Tuple of (flow_taa_result, simple_taa_result)
        """
        # Apply flow-based TAA
        flow_result = self.flow_taa.apply_taa(
            current_frame=current_frame,
            flow_pixels=flow_pixels,
            alpha=alpha,
            use_flow=True,
            use_bilateral=True,
            sequence_id='flow'
        )
        
        # Apply simple TAA
        simple_result = self.simple_taa.apply_simple_taa(
            current_frame=current_frame,
            alpha=alpha,
            sequence_id='simple'
        )
        
        return flow_result, simple_result
    
    def reset_history(self):
        """Reset history for both processors"""
        self.flow_taa.reset_history()
        self.simple_taa.reset_history()
    
    def set_alpha(self, alpha: float):
        """Set alpha for both processors"""
        self.flow_taa.set_alpha(alpha)
        self.simple_taa.set_alpha(alpha)


def apply_taa_effect(current_frame: np.ndarray,
                    flow_pixels: Optional[np.ndarray] = None,
                    previous_taa_frame: Optional[np.ndarray] = None,
                    alpha: float = 0.1,
                    use_flow: bool = True) -> np.ndarray:
    """
    Convenience function to apply TAA effect
    
    Args:
        current_frame: Current frame (RGB, 0-255)
        flow_pixels: Inverted optical flow from previous frame
        previous_taa_frame: Previous TAA result frame
        alpha: Blending weight (0.0 = full history, 1.0 = no history)
        use_flow: Whether to use optical flow for reprojection
    
    Returns:
        TAA processed frame (float32)
    """
    processor = TAAProcessor(alpha)
    return processor.apply_taa(
        current_frame=current_frame,
        flow_pixels=flow_pixels,
        previous_taa_frame=previous_taa_frame,
        alpha=alpha,
        use_flow=use_flow
    ) 