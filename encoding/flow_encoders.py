"""
Flow Encoders - optical flow visualization and encoding formats
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional


class FlowEncoder(ABC):
    """Abstract base class for flow encoders"""
    
    @abstractmethod
    def encode(self, flow: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Encode optical flow to RGB image
        
        Args:
            flow: Optical flow array (H, W, 2)
            width: Image width
            height: Image height
            
        Returns:
            RGB image (H, W, 3) with values in [0, 255]
        """
        pass


class HSVFlowEncoder(FlowEncoder):
    """
    HSV flow encoder - standard visualization format
    - Hue: Flow direction (angle)
    - Saturation: Flow magnitude (normalized)
    - Value: Constant brightness
    """
    
    def encode(self, flow: np.ndarray, width: int, height: int) -> np.ndarray:
        """Encode optical flow in HSV format"""
        # Handle NaN and inf values first
        flow = np.nan_to_num(flow, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        
        # Normalize angle to [0, 2π] and convert to hue [0, 180] for OpenCV
        hue = (angle + np.pi) / (2 * np.pi) * 180
        hue = np.clip(hue, 0, 180).astype(np.uint8)
        
        # Normalize magnitude for saturation
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            saturation = (magnitude / max_magnitude * 255).astype(np.uint8)
        else:
            saturation = np.zeros_like(magnitude, dtype=np.uint8)
        
        # Set constant value (brightness)
        value = np.full_like(magnitude, 255, dtype=np.uint8)
        
        # Create HSV image
        hsv = np.stack([hue, saturation, value], axis=2)
        
        # Convert HSV to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb


class GamedevFlowEncoder(FlowEncoder):
    """
    Gamedev flow encoder - RG channels format
    - Normalize flow by image dimensions
    - Scale and clamp to [-20, +20] range
    - Map to [0, 1] where 0 = -20, 1 = +20
    - Store in RG channels (R=horizontal, G=vertical)
    """
    
    def __init__(self, scale_factor: float = 200.0, clamp_range: float = 20.0):
        """
        Initialize gamedev encoder
        
        Args:
            scale_factor: Scale factor to make motion visible
            clamp_range: Clamp range for flow values
        """
        self.scale_factor = scale_factor
        self.clamp_range = clamp_range
    
    def encode(self, flow: np.ndarray, width: int, height: int) -> np.ndarray:
        """Encode optical flow in gamedev format"""
        # Normalize flow by image dimensions
        norm_flow = flow.copy()
        norm_flow[:, :, 0] /= width    # Horizontal flow
        norm_flow[:, :, 1] /= height   # Vertical flow
        
        # Scale to make motion visible
        norm_flow *= self.scale_factor
        
        # Clamp to [-clamp_range, +clamp_range] range
        clamped = np.clip(norm_flow, -self.clamp_range, self.clamp_range)
        
        # Map [-clamp_range, +clamp_range] to [0, 1]
        encoded = (clamped + self.clamp_range) / (2 * self.clamp_range)
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


class TorchvisionFlowEncoder(FlowEncoder):
    """
    Torchvision flow encoder - color wheel format
    Uses torchvision.utils.flow_to_image for consistent visualization
    """
    
    def __init__(self, fallback_encoder: Optional[FlowEncoder] = None):
        """
        Initialize torchvision encoder
        
        Args:
            fallback_encoder: Encoder to use if torchvision is not available
        """
        self.fallback_encoder = fallback_encoder or HSVFlowEncoder()
        self._torch_available = None
        self._flow_to_image = None
    
    def _check_torchvision(self):
        """Check if torchvision is available and cache the result"""
        if self._torch_available is None:
            try:
                import torch
                from torchvision.utils import flow_to_image
                self._torch_available = True
                self._flow_to_image = flow_to_image
                self._torch = torch
            except ImportError:
                self._torch_available = False
                print("Warning: torchvision not available, falling back to HSV format")
        
        return self._torch_available
    
    def encode(self, flow: np.ndarray, width: int, height: int) -> np.ndarray:
        """Encode optical flow using torchvision format"""
        if not self._check_torchvision():
            return self.fallback_encoder.encode(flow, width, height)
        
        # Convert numpy flow to torch tensor
        # torchvision expects flow in CHW format (channels first)
        flow_tensor = self._torch.from_numpy(flow).permute(2, 0, 1).float()  # HWC -> CHW
        
        # Add batch dimension if needed
        if flow_tensor.dim() == 3:
            flow_tensor = flow_tensor.unsqueeze(0)  # Add batch dimension: CHW -> BCHW
        
        # Use torchvision's flow_to_image function
        # This creates a color wheel visualization similar to Middlebury flow dataset
        with self._torch.no_grad():
            flow_image_tensor = self._flow_to_image(flow_tensor)
        
        # Remove batch dimension and convert back to numpy
        if flow_image_tensor.dim() == 4:
            flow_image_tensor = flow_image_tensor.squeeze(0)  # Remove batch: BCHW -> CHW
        
        # Convert from CHW to HWC and to numpy
        flow_image_np = flow_image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # torchvision returns values in [0, 1] range, convert to [0, 255]
        flow_image_np = (flow_image_np * 255).astype(np.uint8)
        
        return flow_image_np


class FlowEncoderFactory:
    """Factory for creating flow encoders"""
    
    _encoders = {
        'hsv': HSVFlowEncoder,
        'gamedev': GamedevFlowEncoder,
        'torchvision': TorchvisionFlowEncoder
    }
    
    @classmethod
    def create_encoder(cls, format_name: str, **kwargs) -> FlowEncoder:
        """
        Create a flow encoder by format name
        
        Args:
            format_name: Format name ('hsv', 'gamedev', 'torchvision')
            **kwargs: Additional arguments for encoder initialization
            
        Returns:
            FlowEncoder instance
            
        Raises:
            ValueError: If format_name is not supported
        """
        format_name = format_name.lower()
        
        if format_name not in cls._encoders:
            available_formats = ', '.join(cls._encoders.keys())
            raise ValueError(f"Unsupported format '{format_name}'. Available formats: {available_formats}")
        
        encoder_class = cls._encoders[format_name]
        return encoder_class(**kwargs)
    
    @classmethod
    def get_available_formats(cls):
        """Get list of available encoding formats"""
        return list(cls._encoders.keys())
    
    @classmethod
    def register_encoder(cls, format_name: str, encoder_class: type):
        """
        Register a new encoder class
        
        Args:
            format_name: Format name
            encoder_class: Encoder class (must inherit from FlowEncoder)
        """
        if not issubclass(encoder_class, FlowEncoder):
            raise ValueError("Encoder class must inherit from FlowEncoder")
        
        cls._encoders[format_name.lower()] = encoder_class


def encode_flow(flow: np.ndarray, width: int, height: int, format_name: str = 'gamedev') -> np.ndarray:
    """
    Convenience function to encode flow with specified format
    
    Args:
        flow: Optical flow array (H, W, 2)
        width: Image width
        height: Image height
        format_name: Encoding format ('hsv', 'gamedev', 'torchvision')
        
    Returns:
        RGB image (H, W, 3) with values in [0, 255]
    """
    encoder = FlowEncoderFactory.create_encoder(format_name)
    return encoder.encode(flow, width, height) 