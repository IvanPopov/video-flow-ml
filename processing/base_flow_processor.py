"""
Base classes for unified optical flow processing API

This module provides abstract base classes that define a common interface
for both VideoFlow and MemFlow processing pipelines. This ensures consistent
API across different optical flow models.

Architecture:
- BaseFlowCore: Low-level model operations (loading, inference)
- BaseFlowProcessor: High-level processing pipeline (frame sequence, tiles)
- BaseFlowInference: Compatibility layer (maintains backward compatibility)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm


class BaseFlowCore(ABC):
    """
    Abstract base class for low-level optical flow model operations
    
    This class defines the interface for core model operations:
    - Model loading and configuration
    - Direct tensor-to-tensor optical flow computation
    - Device management and memory monitoring
    """
    
    def __init__(self, device: str, fast_mode: bool = False, **kwargs):
        """
        Initialize core flow engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            **kwargs: Model-specific configuration parameters
        """
        self.device = device
        self.fast_mode = fast_mode
        self.model = None
        self.cfg = None
    
    @abstractmethod
    def load_model(self) -> str:
        """
        Load optical flow model
        
        Returns:
            Path to the loaded model
        """
        pass
    
    @abstractmethod
    def compute_flow_from_tensor(self, frame_batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow from pre-processed tensor batch
        
        Args:
            frame_batch_tensor: Pre-processed tensor [B, T, C, H, W]
                              - B=1 (batch size)
                              - T=sequence_length
                              - C=3 (RGB channels)
                              - H, W: frame dimensions
                              - Values normalized to [0,1]
                              - Device: must match self.device
        
        Returns:
            flow_tensor: Raw optical flow tensor [2, H, W]
                        - 2 channels: [flow_x, flow_y]
                        - H, W: same as input frame dimensions
        """
        pass
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if self.model is None:
            return {"status": "not_loaded", "model_type": self.__class__.__name__}
        
        return {
            "status": "loaded",
            "model_type": self.__class__.__name__,
            "device": str(self.device),
            "fast_mode": self.fast_mode
        }
    
    def get_device(self) -> str:
        """Get current device"""
        return str(self.device)
    
    def set_eval_mode(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "device": str(self.device)
            }
        return {"device": str(self.device), "memory_info": "CPU device"}


class BaseFlowProcessor(ABC):
    """
    Abstract base class for high-level optical flow processing
    
    This class defines the interface for complete processing pipelines:
    - Frame sequence preparation from numpy arrays
    - Tile-based processing for large frames
    - Progress tracking integration
    - Format conversions and validation
    """
    
    def __init__(self, device: str, fast_mode: bool = False, tile_mode: bool = False, 
                 sequence_length: int = 5, **kwargs):
        """
        Initialize optical flow processor
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            tile_mode: Enable tile-based processing for large frames
            sequence_length: Number of frames to use in sequence for inference
            **kwargs: Model-specific configuration parameters
        """
        self.device = device
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self.core = None  # Will be set by subclasses
    
    @abstractmethod
    def load_model(self):
        """Load optical flow model using core engine"""
        pass
    
    @abstractmethod
    def prepare_frame_sequence(self, frames: List[np.ndarray], frame_idx: int) -> torch.Tensor:
        """
        Prepare frame sequence for model inference
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            
        Returns:
            frame_batch: Tensor in model format [1, T, 3, H, W], values 0.0-1.0
        """
        pass
    
    def compute_optical_flow(self, frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        """
        Compute optical flow using the model
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            
        Returns:
            flow_np: Optical flow as numpy array [H, W, 2], values in pixels
        """
        if not self.core.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare frame sequence
        frame_batch = self.prepare_frame_sequence(frames, frame_idx)
        
        # Use core engine for inference
        flow_tensor = self.core.compute_flow_from_tensor(frame_batch)
        
        # Convert to numpy: CHW -> HWC
        flow_np = flow_tensor.permute(1, 2, 0).cpu().numpy()
        
        return flow_np
    
    @staticmethod
    def calculate_tile_grid(width: int, height: int, tile_size: int = 1280) -> Tuple[int, int, int, int, List[Dict]]:
        """
        Calculate tile grid for fixed square tiles
        
        Args:
            width, height: Original frame dimensions
            tile_size: Fixed tile size (default: 1280x1280)
            
        Returns:
            (tile_width, tile_height, cols, rows, tiles_info)
        """
        import numpy as np
        
        # Use fixed square tiles
        tile_width = tile_size
        tile_height = tile_size
        
        # Calculate number of tiles needed
        cols = int(np.ceil(width / tile_width))
        rows = int(np.ceil(height / tile_height))
        
        # Calculate actual tile positions
        tiles_info = []
        for row in range(rows):
            for col in range(cols):
                # Calculate tile position
                x = col * tile_width
                y = row * tile_height
                
                # Calculate actual tile size (edge tiles might be smaller)
                actual_width = min(tile_width, width - x)
                actual_height = min(tile_height, height - y)
                
                tiles_info.append({
                    'x': x, 'y': y,
                    'width': actual_width, 'height': actual_height,
                    'col': col, 'row': row
                })
        
        return tile_width, tile_height, cols, rows, tiles_info
    
    def extract_tile(self, frame: np.ndarray, tile_info: Dict) -> np.ndarray:
        """Extract a tile from the frame without padding"""
        x, y = tile_info['x'], tile_info['y']
        w, h = tile_info['width'], tile_info['height']
        
        # Extract the tile from frame
        tile = frame[y:y+h, x:x+w]
        
        return tile
    
    def compute_optical_flow_with_progress(self, frames: List[np.ndarray], frame_idx: int, 
                                         tile_pbar: Optional[tqdm] = None) -> np.ndarray:
        """
        Compute optical flow with progress updates for tile processing
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            tile_pbar: Optional progress bar for tile processing updates
            
        Returns:
            flow_np: Optical flow as numpy array [H, W, 2], values in pixels
        """
        # Default implementation: just calls compute_optical_flow
        # Subclasses can override for more sophisticated progress tracking
        return self.compute_optical_flow(frames, frame_idx)
    
    def compute_optical_flow_tiled(self, frames: List[np.ndarray], frame_idx: int,
                                  tile_pbar: Optional[tqdm] = None, 
                                  overall_pbar: Optional[tqdm] = None) -> np.ndarray:
        """
        Compute optical flow using tile-based processing with square tiles
        
        Args:
            frames: List of frames
            frame_idx: Current frame index
            tile_pbar: Progress bar for current tile processing
            overall_pbar: Progress bar for overall tiles progress
            
        Returns:
            Full-resolution optical flow
        """
        if not self.tile_mode:
            return self.compute_optical_flow_with_progress(frames, frame_idx, tile_pbar)
        
        # Default implementation: not tiled
        # Subclasses should override for actual tile processing
        return self.compute_optical_flow_with_progress(frames, frame_idx, tile_pbar)
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.core is not None and self.core.is_model_loaded()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if self.core is None:
            return {"status": "not_initialized", "processor_type": self.__class__.__name__}
        
        info = self.core.get_model_info()
        info.update({
            "processor_type": self.__class__.__name__,
            "tile_mode": self.tile_mode,
            "sequence_length": self.sequence_length
        })
        
        return info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if self.core is None:
            return {"device": str(self.device), "memory_info": "Core not initialized"}
        
        return self.core.get_memory_usage()
    
    def set_tile_mode(self, enabled: bool):
        """Enable or disable tile-based processing"""
        self.tile_mode = enabled
    
    def set_sequence_length(self, length: int):
        """Set the sequence length for multi-frame processing"""
        self.sequence_length = length
    
    def validate_frames(self, frames: List[np.ndarray], frame_idx: int) -> bool:
        """Validate frame input format and parameters"""
        if not isinstance(frames, list):
            raise ValueError("frames must be a list of numpy arrays")
        
        if len(frames) == 0:
            raise ValueError("frames list cannot be empty")
        
        if frame_idx < 0 or frame_idx >= len(frames):
            raise ValueError(f"frame_idx {frame_idx} out of range [0, {len(frames)-1}]")
        
        # Check frame format
        frame = frames[frame_idx]
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame at index {frame_idx} is not a numpy array")
        
        if len(frame.shape) != 3:
            raise ValueError(f"Frame at index {frame_idx} must have 3 dimensions [H, W, C], got {len(frame.shape)}")
        
        if frame.shape[2] != 3:
            raise ValueError(f"Frame at index {frame_idx} must have 3 channels, got {frame.shape[2]}")
        
        return True


class BaseFlowInference(ABC):
    """
    Abstract base class for optical flow inference compatibility layer
    
    This class provides backward compatibility with existing code while using
    the new modular architecture internally. It delegates operations to
    the appropriate processor and core modules.
    """
    
    def __init__(self, device: str, fast_mode: bool = False, tile_mode: bool = False, 
                 sequence_length: int = 5, **kwargs):
        """
        Initialize optical flow inference engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            tile_mode: Enable tile-based processing for large frames
            sequence_length: Number of frames to use in sequence for inference
            **kwargs: Model-specific configuration parameters
        """
        self.device = device
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self._processor = None  # Will be set by subclasses
        
        # Legacy attributes for backward compatibility
        self.model = None  # Will be set when model is loaded
        self.cfg = None    # Will be set when model is loaded
    
    @abstractmethod
    def load_model(self):
        """Load optical flow model"""
        pass
    
    def calculate_tile_grid(self, width: int, height: int, tile_size: int = 1280) -> Tuple[int, int, int, int, List[Dict]]:
        """Calculate tile grid for fixed square tiles"""
        return BaseFlowProcessor.calculate_tile_grid(width, height, tile_size)
    
    def extract_tile(self, frame: np.ndarray, tile_info: Dict) -> np.ndarray:
        """Extract a tile from the frame without padding"""
        return self._processor.extract_tile(frame, tile_info)
    
    def prepare_frame_sequence(self, frames: List[np.ndarray], frame_idx: int) -> torch.Tensor:
        """Prepare frame sequence for model inference"""
        return self._processor.prepare_frame_sequence(frames, frame_idx)
    
    def compute_optical_flow(self, frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        """Compute optical flow using the model"""
        return self._processor.compute_optical_flow(frames, frame_idx)
    
    def compute_optical_flow_with_progress(self, frames: List[np.ndarray], frame_idx: int, 
                                         tile_pbar: Optional[tqdm] = None) -> np.ndarray:
        """Compute optical flow with progress updates for tile processing"""
        return self._processor.compute_optical_flow_with_progress(frames, frame_idx, tile_pbar)
    
    def compute_optical_flow_tiled(self, frames: List[np.ndarray], frame_idx: int,
                                  tile_pbar: Optional[tqdm] = None, 
                                  overall_pbar: Optional[tqdm] = None) -> np.ndarray:
        """Compute optical flow using tile-based processing"""
        return self._processor.compute_optical_flow_tiled(frames, frame_idx, tile_pbar, overall_pbar)
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._processor.is_model_loaded()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        info = self._processor.get_model_info()
        
        # Add compatibility layer information
        if info["status"] == "loaded":
            info["compatibility_layer"] = self.__class__.__name__
        
        return info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        return self._processor.get_memory_usage()
    
    def get_core_engine(self):
        """Get direct access to core engine"""
        return self._processor.core
    
    def get_processor(self):
        """Get direct access to processor"""
        return self._processor
    
    def validate_frames(self, frames: List[np.ndarray], frame_idx: int) -> bool:
        """Validate frame input format and parameters"""
        return self._processor.validate_frames(frames, frame_idx)
    
    def set_tile_mode(self, enabled: bool):
        """Enable or disable tile-based processing"""
        self.tile_mode = enabled
        self._processor.set_tile_mode(enabled)
    
    def set_sequence_length(self, length: int):
        """Set the sequence length for multi-frame processing"""
        self.sequence_length = length
        self._processor.set_sequence_length(length) 