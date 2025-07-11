"""
VideoFlow Processor - High-level optical flow processing module

This module provides high-level VideoFlow operations:
- Frame sequence preparation and management
- Tile-based processing for large frames
- Progress tracking and coordination
- Format conversions (numpy <-> tensor)
- Input validation and error handling

This module uses VideoFlowCore for actual model inference and provides
a complete processing pipeline for practical applications.
"""

import torch
import numpy as np
from .videoflow_core import VideoFlowCore


class VideoFlowProcessor:
    """
    High-level VideoFlow processor for optical flow computation
    
    This class provides a complete processing pipeline:
    - Frame sequence preparation from numpy arrays
    - Tile-based processing for large frames
    - Progress tracking integration
    - Format conversions and validation
    - Error handling and recovery
    
    Uses VideoFlowCore internally for actual model inference.
    """
    
    def __init__(self, device, fast_mode=False, tile_mode=False, sequence_length=5,
                 dataset='sintel', architecture='mof', variant='standard'):
        """
        Initialize VideoFlow processor
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            tile_mode: Enable tile-based processing for large frames
            sequence_length: Number of frames to use in sequence for inference
            dataset: Training dataset ('sintel', 'things', 'kitti')
            architecture: Model architecture ('mof' for MOFNet, 'bof' for BOFNet)
            variant: Model variant ('standard' or 'noise' for things_288960noise)
        """
        self.device = device
        self.fast_mode = fast_mode
        self.tile_mode = tile_mode
        self.sequence_length = sequence_length
        self.dataset = dataset
        self.architecture = architecture
        self.variant = variant
        
        # Initialize core inference engine with model configuration
        self.core = VideoFlowCore(device, fast_mode, dataset, architecture, variant)
        
        print(f"VideoFlow Processor initialized:")
        print(f"  Device: {device}")
        print(f"  Fast mode: {fast_mode}")
        print(f"  Tile mode: {tile_mode}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Dataset: {dataset}")
        print(f"  Architecture: {architecture.upper()}")
        print(f"  Variant: {variant}")
    
    def load_model(self):
        """Load VideoFlow MOF model using core engine"""
        model_path = self.core.load_model()
        print(f"VideoFlow model loaded successfully from: {model_path}")
    
    @staticmethod
    def calculate_tile_grid(width, height, tile_size=1280):
        """
        Calculate tile grid for fixed square tiles (optimized for VideoFlow MOF model)
        
        Args:
            width, height: Original frame dimensions
            tile_size: Fixed tile size (default: 1280x1280, optimal for MOF model)
            
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
    
    def extract_tile(self, frame, tile_info):
        """Extract a tile from the frame without padding"""
        x, y = tile_info['x'], tile_info['y']
        w, h = tile_info['width'], tile_info['height']
        
        # Extract the tile from frame
        tile = frame[y:y+h, x:x+w]
        
        return tile
    
    def prepare_frame_sequence(self, frames, frame_idx):
        """
        Prepare frame sequence for VideoFlow MOF model
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            
        Returns:
            frame_batch: Tensor in VideoFlow format [1, T, 3, H, W], values 0.0-1.0
        """
        # Multi-frame: use consecutive frames centered around current frame
        half_seq = self.sequence_length // 2
        start_idx = max(0, frame_idx - half_seq)
        end_idx = min(len(frames), frame_idx + half_seq + 1)
        sequence = frames[start_idx:end_idx]
        
        # Pad to exactly sequence_length frames
        while len(sequence) < self.sequence_length:
            if start_idx == 0:
                sequence.insert(0, sequence[0])
            else:
                sequence.append(sequence[-1])
        
        # Ensure exactly sequence_length frames
        sequence = sequence[:self.sequence_length]

        # Convert to tensors (same format as VideoFlow inference.py)
        tensors = []
        for frame in sequence:
            # Convert to tensor and normalize to [0,1], then change HWC to CHW
            if frame.dtype == np.uint8:
                tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
            else:
                # Already float, assume it's in correct range
                tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
            tensors.append(tensor)
        
        # Stack frames and add batch dimension
        batch = torch.stack(tensors).unsqueeze(0).to(self.device)
        return batch
    
    def compute_optical_flow(self, frames, frame_idx):
        """
        Compute optical flow using VideoFlow model
        
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
    
    def compute_optical_flow_with_progress(self, frames, frame_idx, tile_pbar=None):
        """
        Compute optical flow with progress updates for tile processing
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            tile_pbar: Optional progress bar for tile processing updates
            
        Returns:
            flow_np: Optical flow as numpy array [H, W, 2], values in pixels
        """
        if not self.core.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if tile_pbar is not None:
            tile_pbar.set_description("Preparing frames")
            tile_pbar.reset()
        
        # Prepare frame sequence
        frame_batch = self.prepare_frame_sequence(frames, frame_idx)
        
        if tile_pbar is not None:
            tile_pbar.set_description("Running VideoFlow")
            tile_pbar.update(2)
        
        # Use core engine for inference
        flow_tensor = self.core.compute_flow_from_tensor(frame_batch)
        
        if tile_pbar is not None:
            tile_pbar.set_description("Processing output")
            tile_pbar.update(1)
        
        # Convert to numpy: CHW -> HWC  
        flow_np = flow_tensor.permute(1, 2, 0).cpu().numpy()
        
        if tile_pbar is not None:
            tile_pbar.set_description("Completed")
            tile_pbar.update(1)
        
        return flow_np
    
    def compute_optical_flow_tiled(self, frames, frame_idx, tile_pbar=None, overall_pbar=None):
        """
        Compute optical flow using tile-based processing with 1280x1280 square tiles
        
        Args:
            frames: List of numpy arrays in RGB format [H, W, 3], values 0-255
            frame_idx: Index of current frame to process
            tile_pbar: Progress bar for current tile processing
            overall_pbar: Progress bar for overall tiles progress
            
        Returns:
            Full-resolution optical flow as numpy array [H, W, 2]
        """
        if not self.tile_mode:
            # Use standard processing if tile mode is disabled
            return self.compute_optical_flow(frames, frame_idx)
        
        current_frame = frames[frame_idx]
        height, width = current_frame.shape[:2]
        
        # Calculate tile grid
        tile_width, tile_height, cols, rows, tiles_info = VideoFlowProcessor.calculate_tile_grid(width, height)
        
        # Initialize full flow map
        full_flow = np.zeros((height, width, 2), dtype=np.float32)
        
        # Process each tile
        for i, tile_info in enumerate(tiles_info):
            # Update overall progress bar description
            if overall_pbar is not None:
                overall_pbar.set_description(f"Tile {i+1}/{len(tiles_info)} ({tile_info['width']}x{tile_info['height']})")
            
            # Extract tile from all frames in sequence
            tile_frames = []
            for frame in frames:
                tile = self.extract_tile(frame, tile_info)
                tile_frames.append(tile)
            
            # Compute flow for this tile with tile progress bar
            tile_flow = self.compute_optical_flow_with_progress(tile_frames, frame_idx, tile_pbar)
            
            # Place tile flow back into full flow map
            x, y = tile_info['x'], tile_info['y']
            w, h = tile_info['width'], tile_info['height']
            
            # Place flow back into full flow map
            full_flow[y:y+h, x:x+w] = tile_flow
            
            # Update overall progress
            if overall_pbar is not None:
                overall_pbar.update(1)
        
        return full_flow
    
    def is_model_loaded(self):
        """Check if VideoFlow model is loaded"""
        return self.core.is_model_loaded()
    
    def get_model_info(self):
        """Get information about loaded model"""
        core_info = self.core.get_model_info()
        
        # Add processor-specific information
        if core_info["status"] == "loaded":
            core_info.update({
                "tile_mode": self.tile_mode,
                "sequence_length": self.sequence_length,
                "processor_type": "VideoFlowProcessor"
            })
        
        return core_info
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        return self.core.get_memory_usage()
    
    def validate_frames(self, frames, frame_idx):
        """
        Validate frame input format and parameters
        
        Args:
            frames: List of numpy arrays
            frame_idx: Frame index to process
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(frames, list):
            raise ValueError("Frames must be a list of numpy arrays")
        
        if len(frames) == 0:
            raise ValueError("Frames list cannot be empty")
        
        if frame_idx < 0 or frame_idx >= len(frames):
            raise ValueError(f"Frame index {frame_idx} out of range [0, {len(frames)-1}]")
        
        # Check first frame format
        sample_frame = frames[0]
        if not isinstance(sample_frame, np.ndarray):
            raise ValueError("Frames must be numpy arrays")
        
        if len(sample_frame.shape) != 3:
            raise ValueError(f"Frames must be 3D arrays [H,W,C], got shape {sample_frame.shape}")
        
        if sample_frame.shape[2] != 3:
            raise ValueError(f"Frames must have 3 color channels, got {sample_frame.shape[2]}")
        
        # Check data type and range
        if sample_frame.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValueError(f"Unsupported frame dtype: {sample_frame.dtype}")
        
        if sample_frame.dtype == np.uint8:
            if sample_frame.max() > 255 or sample_frame.min() < 0:
                raise ValueError("uint8 frames must be in range [0, 255]")
        elif sample_frame.dtype in [np.float32, np.float64]:
            if sample_frame.max() > 1.0 or sample_frame.min() < 0.0:
                # Allow some tolerance for floating point
                if sample_frame.max() <= 255.0 and sample_frame.min() >= 0.0:
                    pass  # Probably 0-255 range in float format
                else:
                    raise ValueError("Float frames must be in range [0.0, 1.0] or [0.0, 255.0]")
    
    def set_tile_mode(self, enabled):
        """Enable or disable tile-based processing"""
        self.tile_mode = enabled
    
    def set_sequence_length(self, length):
        """Set the sequence length for multi-frame processing"""
        if length < 1 or length > 10:
            raise ValueError("Sequence length must be between 1 and 10")
        self.sequence_length = length 