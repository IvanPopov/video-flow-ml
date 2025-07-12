"""
MemFlow Core - Low-level optical flow computation module

This module provides the core MemFlow model operations:
- Model loading and configuration
- Direct optical flow computation from prepared tensor batches
- Minimal dependencies and clean interface

This is a low-level module that expects pre-processed tensor inputs
and returns raw optical flow tensors. It does NOT handle:
- Frame sequence preparation
- Tile-based processing
- Progress tracking
- Input validation beyond basic tensor checks

For high-level operations, use MemFlowProcessor instead.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any

from .base_flow_processor import BaseFlowCore


class MemFlowCore(BaseFlowCore):
    """
    Core MemFlow inference engine for optical flow computation
    
    This class provides only the essential MemFlow operations:
    - Model loading with configuration
    - Direct tensor-to-tensor optical flow computation
    - Minimal overhead and dependencies
    
    Input requirements:
    - Pre-processed tensor batches in MemFlow format: [B, T, C, H, W]
    - Normalized to [0,1] float32 values
    - RGB channel order
    
    Output:
    - Raw optical flow tensor in format [2, H, W] (2 channels: x, y flow)
    - No padding, no format conversion
    """
    
    def __init__(self, device: str, fast_mode: bool = False, stage: str = 'sintel', 
                 model_path: str = None, enable_long_term: bool = False, **kwargs):
        """
        Initialize MemFlow core engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode (currently not implemented for MemFlow)
            stage: Training stage/dataset ('sintel', 'things', 'kitti')
            model_path: Custom path to model weights
            enable_long_term: Enable long-term memory (default: False)
            **kwargs: Additional configuration parameters
        """
        super().__init__(device, fast_mode, **kwargs)
        
        self.stage = stage
        self.model_path = model_path
        self.enable_long_term = enable_long_term
        self.processor = None
        self.InputPadder = None
        
        # Generate default model path if not provided
        if self.model_path is None:
            self.model_path = f'MemFlow_ckpt/MemFlowNet_{stage}.pth'
    
    def load_model(self) -> str:
        """Load MemFlow model with original approach"""
        if self.model is not None:
            print("[MemFlow Core] Model already loaded")
            return self.model_path
        
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
        
        # Monkey patch the assertion if long-term memory is enabled
        # This must happen BEFORE any MemFlow modules are imported
        if self.enable_long_term:
            # Temporarily modify the memory manager file to disable the assertion
            memory_manager_path = os.path.join(memflow_root, 'inference', 'memory_manager_skflow.py')
            if os.path.exists(memory_manager_path):
                # Read the file
                with open(memory_manager_path, 'r') as f:
                    content = f.read()
                
                # Store original content
                self._original_memory_manager_content = content
                
                # Comment out the assertion line
                modified_content = content.replace(
                    'assert self.enable_long_term == False',
                    '# assert self.enable_long_term == False  # Temporarily disabled for long-term memory'
                )
                
                # Write back the modified content
                with open(memory_manager_path, 'w') as f:
                    f.write(modified_content)
                
                print("[MemFlow Core] Temporarily disabled long-term memory assertion")
        
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
            networks_module.__package__ = 'core.Networks'
            networks_module.__path__ = [os.path.join(memflow_root, 'core', 'Networks')]
            networks_module.MemFlowNet = memflownet_package
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
            
            # Enable warm start for better sequential processing
            self.cfg.warm_start = True
            
            # Enable long-term memory if requested
            if self.enable_long_term:
                self.cfg.enable_long_term = True
                print(f"[MemFlow Core] Long-term memory enabled")
            else:
                self.cfg.enable_long_term = False
                print(f"[MemFlow Core] Long-term memory disabled (default)")
            
            # Adjust decoder depth for better speed/accuracy trade-off
            # Reduce from 15 to 8 for faster inference while maintaining quality
            self.cfg.val_decoder_depth = 8
            
            # Build and load model
            self.model = build_network(self.cfg).to(self.device)
            
            print(f"[MemFlow Core] Loading checkpoint from: {abs_model_path}")
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
            
            print(f"[MemFlow Core] Model loaded successfully:")
            print(f"  Path: {self.model_path}")
            print(f"  Stage: {self.stage}")
            print(f"  Device: {self.device}")
            print(f"  Long-term memory: {'Enabled' if self.enable_long_term else 'Disabled'}")
            if self.fast_mode:
                print(f"  Fast Mode: Enabled (note: not implemented for MemFlow)")
            
            return self.model_path
            
        finally:
            # Restore original working directory
            os.chdir(old_cwd)
            
            # Restore original memory manager file if it was modified
            if hasattr(self, '_original_memory_manager_content'):
                memory_manager_path = os.path.join(memflow_root, 'inference', 'memory_manager_skflow.py')
                if os.path.exists(memory_manager_path):
                    with open(memory_manager_path, 'w') as f:
                        f.write(self._original_memory_manager_content)
                    print("[MemFlow Core] Restored original memory manager file")
    
    def compute_flow_from_tensor(self, frame_batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow from pre-processed tensor batch
        
        Args:
            frame_batch_tensor: Pre-processed tensor [B, T, C, H, W]
                              - B=1 (batch size)
                              - T=sequence_length (typically 2-3 for MemFlow)
                              - C=3 (RGB channels)
                              - H, W: frame dimensions
                              - Values normalized to [0,1]
                              - Device: must match self.device
        
        Returns:
            flow_tensor: Raw optical flow tensor [2, H, W]
                        - 2 channels: [flow_x, flow_y]
                        - H, W: same as input frame dimensions
        
        Raises:
            RuntimeError: If model not loaded
            ValueError: If tensor format is incorrect
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("MemFlow model not loaded. Call load_model() first.")
        
        # Validate input tensor
        if not isinstance(frame_batch_tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        # Check device compatibility
        tensor_device_str = str(frame_batch_tensor.device)
        model_device_str = str(self.device)
        
        if tensor_device_str.startswith('cuda') and model_device_str == 'cuda':
            pass  # Compatible
        elif model_device_str.startswith('cuda') and tensor_device_str == 'cuda':
            pass  # Compatible
        elif tensor_device_str != model_device_str:
            raise ValueError(f"Input tensor device ({frame_batch_tensor.device}) doesn't match model device ({self.device})")
        
        if len(frame_batch_tensor.shape) != 5:
            raise ValueError(f"Input tensor must have 5 dimensions [B,T,C,H,W], got {len(frame_batch_tensor.shape)}")
        
        if frame_batch_tensor.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {frame_batch_tensor.shape[0]}")
        
        if frame_batch_tensor.shape[2] != 3:
            raise ValueError(f"Must have 3 color channels, got {frame_batch_tensor.shape[2]}")
        
        # MemFlow expects at least 2 frames
        if frame_batch_tensor.shape[1] < 2:
            raise ValueError(f"MemFlow requires at least 2 frames, got {frame_batch_tensor.shape[1]}")
        
        # Get sequence length
        sequence_length = frame_batch_tensor.shape[1]
        
        # Use InputPadder for spatial padding
        padder = self.InputPadder(frame_batch_tensor.shape[-2:])
        
        # Pad entire sequence
        padded_tensor = padder.pad(frame_batch_tensor)
        
        # Convert to MemFlow format: values in [-1, 1]
        images = 2.0 * padded_tensor - 1.0
        
        # Initialize variables for sequential processing
        flow_prev = None
        final_flow = None
        
        # Process frames sequentially to build up memory
        with torch.no_grad():
            # Clear memory only at the beginning (not between frames!)
            self.processor.clear_memory()
            
            # Process consecutive frame pairs
            for ti in range(sequence_length - 1):
                # Get current frame pair
                frame_pair = images[:, ti:ti + 2, :, :, :]  # [B, 2, C, H, W]
                
                # Check if this is the last frame pair
                is_end = (ti == sequence_length - 2)
                
                # MemFlow inference using step method
                flow_low, flow_pre = self.processor.step(
                    frame_pair, 
                    end=is_end, 
                    flow_init=flow_prev,
                    add_pe=False  # Disable positional encoding for now
                )
                
                # Store the final flow (from the last frame pair)
                if is_end:
                    final_flow = flow_pre
                
                # Prepare flow initialization for next iteration (warm start)
                if hasattr(self.cfg, 'warm_start') and self.cfg.warm_start and not is_end:
                    # Use forward_interpolate to prepare flow_init for next frame
                    from utils.utils import forward_interpolate
                    flow_prev = forward_interpolate(flow_low[0])[None].cuda()
                else:
                    flow_prev = None
            
            # Ensure we have a final flow
            if final_flow is None:
                raise RuntimeError("Failed to compute flow - no final flow generated")
            
            # Unpad the result
            final_flow = padder.unpad(final_flow)
            
            # Remove batch dimension: [B, 2, H, W] -> [2, H, W]
            final_flow = final_flow[0]
            
            return final_flow
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        info = super().get_model_info()
        if self.model is not None:
            info.update({
                "model_type": "MemFlowCore",
                "stage": self.stage,
                "model_path": self.model_path,
                "architecture": "MemFlow",
                "framework": "MemFlow",
                "enable_long_term": self.enable_long_term
            })
        return info 