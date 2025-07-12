#!/usr/bin/env python3
"""
Simplified MemFlow Processor

Based on original MemFlow inference.py but simplified for integration.
No isolated processes, no complex architecture - just direct processing.
Output format matches VideoFlow: optical flow as pixel velocities.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from tqdm import tqdm

# MemFlow imports will be done lazily in load_model method to avoid conflicts


class MemFlowProcessor:
    """
    Simplified MemFlow processor that follows original inference.py logic.
    
    - No isolated processes
    - Direct integration with MemFlow
    - Same output format as VideoFlow (pixel velocities)
    - Maximum quality processing
    """
    
    def __init__(self, device='cuda', model_path='MemFlow_ckpt/MemFlowNet_sintel.pth', 
                 stage='sintel', sequence_length=3):
        """
        Initialize MemFlow processor.
        
        Args:
            device: Processing device ('cuda' or 'cpu')
            model_path: Path to MemFlow model weights
            stage: Training stage configuration ('sintel', 'things', 'kitti')
            sequence_length: Number of frames to use for processing
        """
        self.device = device
        self.model_path = model_path
        self.stage = stage
        self.sequence_length = max(2, sequence_length)
        self.model = None
        self.processor = None
        self.cfg = None
        
        print(f"[MemFlow] Processor initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model: {model_path}")
        print(f"  Stage: {stage}")
        print(f"  Sequence length: {self.sequence_length}")
    
    def load_model(self):
        """Load MemFlow model using original approach"""
        if self.model is not None:
            print("[MemFlow] Model already loaded")
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MemFlow model not found: {self.model_path}")
        
        # Setup MemFlow paths and imports
        memflow_root = os.path.abspath(os.path.join(os.getcwd(), 'MemFlow'))
        
        # Add all necessary paths to sys.path
        paths_to_add = [
            memflow_root,
            os.path.join(memflow_root, 'core'),
            os.path.join(memflow_root, 'inference'),
            os.path.join(memflow_root, 'core', 'Networks'),
            os.path.join(memflow_root, 'core', 'Networks', 'MemFlowNet'),
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Change working directory to MemFlow temporarily for relative imports
        old_cwd = os.getcwd()
        os.chdir(memflow_root)
        
        try:
            # Create complete module structure for MemFlow
            import importlib.util
            import types
            
            # Create core package
            if 'core' not in sys.modules:
                core_package = types.ModuleType('core')
                core_package.__path__ = [os.path.join(memflow_root, 'core')]
                sys.modules['core'] = core_package
            
            # Create core.Networks package
            if 'core.Networks' not in sys.modules:
                networks_package = types.ModuleType('core.Networks')
                networks_package.__path__ = [os.path.join(memflow_root, 'core', 'Networks')]
                networks_package.__package__ = 'core'
                sys.modules['core.Networks'] = networks_package
            
            # Create MemFlowNet package
            memflownet_path = os.path.join(memflow_root, 'core', 'Networks', 'MemFlowNet')
            memflownet_package = types.ModuleType('core.Networks.MemFlowNet')
            memflownet_package.__path__ = [memflownet_path]
            memflownet_package.__package__ = 'core.Networks'
            
            # Load all modules in MemFlowNet directory
            memflownet_modules = ['memory_util', 'corr', 'gma', 'cnn', 'update', 'sk', 'sk2', 'MemFlow', 'MemFlow_P']
            
            for module_name in memflownet_modules:
                module_path = os.path.join(memflownet_path, f'{module_name}.py')
                if os.path.exists(module_path):
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f'core.Networks.MemFlowNet.{module_name}', 
                            module_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        module.__package__ = 'core.Networks.MemFlowNet'
                        spec.loader.exec_module(module)
                        
                        # Add to package and sys.modules
                        setattr(memflownet_package, module_name, module)
                        sys.modules[f'core.Networks.MemFlowNet.{module_name}'] = module
                    except Exception as e:
                        print(f"Warning: Could not load {module_name}: {e}")
            
            # Register the MemFlowNet package
            sys.modules['core.Networks.MemFlowNet'] = memflownet_package
            
            # Load the Networks __init__.py and add it to core.Networks
            networks_init_path = os.path.join(memflow_root, 'core', 'Networks', '__init__.py')
            spec = importlib.util.spec_from_file_location("core.Networks", networks_init_path)
            networks_module = importlib.util.module_from_spec(spec)
            networks_module.__package__ = 'core.Networks'  # Important: set the correct package context
            networks_module.__path__ = [os.path.join(memflow_root, 'core', 'Networks')]
            networks_module.MemFlowNet = memflownet_package  # Add MemFlowNet to Networks
            spec.loader.exec_module(networks_module)
            
            # Update sys.modules
            sys.modules['core.Networks'] = networks_module
            
            # Get build_network function
            build_network = networks_module.build_network
            
            from utils.utils import InputPadder, forward_interpolate
            from inference_core_skflow import InferenceCore
            
            # Store InputPadder for later use
            self.InputPadder = InputPadder
            
            # Load configuration
            if self.stage == 'sintel':
                from configs.sintel_memflownet import get_cfg
            elif self.stage == 'things':
                from configs.things_memflownet import get_cfg
            elif self.stage == 'kitti':
                from configs.kitti_memflownet import get_cfg
            else:
                raise ValueError(f"Unsupported stage: {self.stage}")
            
            self.cfg = get_cfg()
            # Use absolute path to model
            abs_model_path = os.path.abspath(os.path.join(old_cwd, self.model_path))
            self.cfg.restore_ckpt = abs_model_path
            
            # Build and load model
            self.model = build_network(self.cfg).to(self.device)
            
            print(f"[MemFlow] Loading checkpoint from: {abs_model_path}")
            ckpt = torch.load(abs_model_path, map_location='cpu')
            ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
            
            # Handle module prefix
            if 'module' in list(ckpt_model.keys())[0]:
                for key in list(ckpt_model.keys()):
                    ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            
            self.model.load_state_dict(ckpt_model, strict=True)
            self.model.eval()
            
            # Create inference processor
            self.processor = InferenceCore(self.model, config=self.cfg)
            
            print(f"[MemFlow] Model loaded successfully:")
            print(f"  Path: {self.model_path}")
            print(f"  Stage: {self.stage}")
            print(f"  Device: {self.device}")
            
        finally:
            # Restore original working directory
            os.chdir(old_cwd)
    
    def compute_optical_flow(self, frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        """
        Compute optical flow for a specific frame.
        
        Args:
            frames: List of video frames as numpy arrays [H, W, C]
            frame_idx: Target frame index for flow computation
            
        Returns:
            Optical flow as numpy array [H, W, 2] - pixel velocities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare frame sequence
        frames_tensor = self._prepare_frames(frames, frame_idx)
        
        # Compute optical flow
        with torch.no_grad():
            # Create padder
            padder = self.InputPadder(frames_tensor.shape)
            frames_padded = padder.pad(frames_tensor)
            
            # Normalize to [-1, 1] range
            frames_normalized = 2 * (frames_padded / 255.0) - 1.0
            
            # Process frame pair (last two frames)
            frame_pair = frames_normalized[:, -2:]  # Take last 2 frames
            
            # Compute flow
            flow_low, flow_pred = self.processor.step(
                frame_pair, 
                end=True,
                add_pe=('rope' in self.cfg and self.cfg.rope),
                flow_init=None
            )
            
            # Unpad result
            flow_result = padder.unpad(flow_pred[0])
            
            # Convert to numpy [H, W, 2] - pixel velocities
            flow_numpy = flow_result.permute(1, 2, 0).cpu().numpy()
            
            return flow_numpy
    
    def _prepare_frames(self, frames: List[np.ndarray], frame_idx: int) -> torch.Tensor:
        """
        Prepare frame sequence for MemFlow processing.
        
        Args:
            frames: List of video frames as numpy arrays [H, W, C]
            frame_idx: Target frame index for flow computation
            
        Returns:
            Preprocessed tensor [1, T, C, H, W]
        """
        # Determine frame indices for sequence
        total_frames = len(frames)
        end_idx = frame_idx + 1
        start_idx = max(0, end_idx - self.sequence_length)
        
        # Get frame sequence
        frame_sequence = frames[start_idx:end_idx]
        
        # Pad with first frame if needed
        while len(frame_sequence) < self.sequence_length:
            frame_sequence.insert(0, frame_sequence[0])
        
        # Convert to tensors
        tensor_frames = []
        for frame in frame_sequence:
            # Ensure frame is RGB uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Convert to tensor [C, H, W]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            tensor_frames.append(frame_tensor)
        
        # Stack into [T, C, H, W] then add batch dimension
        sequence_tensor = torch.stack(tensor_frames, dim=0)  # [T, C, H, W]
        sequence_tensor = sequence_tensor.unsqueeze(0)       # [1, T, C, H, W]
        
        return sequence_tensor.to(self.device)
    
    def compute_optical_flow_tiled(self, frames: List[np.ndarray], frame_idx: int,
                                 tile_pbar: Optional[tqdm] = None, 
                                 overall_pbar: Optional[tqdm] = None) -> np.ndarray:
        """
        Compute optical flow (compatibility method - no actual tiling).
        
        Args:
            frames: List of video frames as numpy arrays [H, W, C]
            frame_idx: Target frame index for flow computation
            tile_pbar: Progress bar for tiles (updated for compatibility)
            overall_pbar: Progress bar for overall progress (updated for compatibility)
            
        Returns:
            Optical flow as numpy array [H, W, 2] - pixel velocities
        """
        # Update progress bars
        if tile_pbar is not None:
            tile_pbar.set_description("MemFlow processing")
            tile_pbar.reset(total=1)
            tile_pbar.update(1)
        
        if overall_pbar is not None:
            overall_pbar.set_description("MemFlow full-frame")
            overall_pbar.reset(total=1)
            overall_pbar.update(1)
        
        return self.compute_optical_flow(frames, frame_idx)
    
    @staticmethod
    def calculate_tile_grid(width: int, height: int, tile_size: int = 1280) -> tuple:
        """
        Calculate tile grid (compatibility method - no actual tiling).
            
        Returns:
            Tuple compatible with VideoFlow tile interface
        """
        tiles_info = [{
            'x': 0, 'y': 0,
            'width': width, 'height': height,
            'tile_idx': 0
        }]
        
        return width, height, 1, 1, tiles_info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if self.device.startswith('cuda'):
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'device': self.device
            }
        else:
            return {'device': 'cpu', 'note': 'CPU memory tracking not available'}
    
    def cleanup(self):
        """Clean up resources"""
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        self.model = None
        self.processor = None 