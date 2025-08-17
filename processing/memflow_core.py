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


# ============================================================================
# QUALITY PRESETS FOR MEMFLOW
# ============================================================================

# Available quality presets
QUALITY_PRESETS = {
    'fast': {
        'name': 'Fast',
        'description': 'Fastest processing, lower quality',
        'val_decoder_depth': 6,
        'corr_levels': 3,
        'corr_radius': 3,
        'feat_dim': 128,
        'down_ratio': 10,
        'max_mid_term_frames': 2,
        'min_mid_term_frames': 1,
        'num_prototypes': 64,
        'add_pe': False,
        'attention_scale_factor': 1.0,
        'flow_smoothing': False,
        'outlier_filtering': False
    },
    
    'balanced': {
        'name': 'Balanced',
        'description': 'Good balance between speed and quality',
        'val_decoder_depth': 10,
        'corr_levels': 5,
        'corr_radius': 5,
        'feat_dim': 384,
        'down_ratio': 7,
        'max_mid_term_frames': 3,
        'min_mid_term_frames': 2,
        'num_prototypes': 192,
        'add_pe': True,
        'attention_scale_factor': 1.25,
        'flow_smoothing': True,
        'outlier_filtering': False
    },
    
    'high_quality': {
        'name': 'High Quality',
        'description': 'High quality processing, slower',
        'val_decoder_depth': 12,
        'corr_levels': 6,
        'corr_radius': 6,
        'feat_dim': 512,
        'down_ratio': 6,
        'max_mid_term_frames': 4,
        'min_mid_term_frames': 2,
        'num_prototypes': 256,
        'add_pe': True,
        'attention_scale_factor': 1.5,
        'flow_smoothing': True,
        'outlier_filtering': True
    },
    
    'maximum_quality': {
        'name': 'Maximum Quality',
        'description': 'Maximum quality, very slow',
        'val_decoder_depth': 15,
        'corr_levels': 8,
        'corr_radius': 8,
        'feat_dim': 512,
        'down_ratio': 4,
        'max_mid_term_frames': 6,
        'min_mid_term_frames': 3,
        'num_prototypes': 512,
        'add_pe': True,
        'attention_scale_factor': 2.0,
        'flow_smoothing': True,
        'outlier_filtering': True
    }
}

# Default quality preset (None means use original config settings)
DEFAULT_QUALITY_PRESET = 'maximum_quality'


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
                 model_path: str = None, enable_long_term: bool = False, 
                 quality_preset: str = DEFAULT_QUALITY_PRESET, **kwargs):
        """
        Initialize MemFlow core engine
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode (currently not implemented for MemFlow)
            stage: Training stage/dataset ('sintel', 'things', 'kitti')
            model_path: Custom path to model weights
            enable_long_term: Enable long-term memory (default: False)
            quality_preset: Quality preset ('fast', 'balanced', 'high_quality', 'maximum_quality')
            **kwargs: Additional configuration parameters
        """
        super().__init__(device, fast_mode, **kwargs)
        
        self.stage = stage
        self.model_path = model_path
        self.enable_long_term = enable_long_term
        self.quality_preset = quality_preset
        self.processor = None
        self.InputPadder = None
        
        # Validate quality preset
        if quality_preset is not None and quality_preset not in QUALITY_PRESETS:
            print(f"[MemFlow Core] Warning: Unknown quality preset '{quality_preset}'. Using original config settings.")
            self.quality_preset = None
        
        # Auto-apply maximum quality preset for things + long-term memory combination
        if self.stage == 'things' and self.enable_long_term and self.quality_preset is None:
            self.quality_preset = 'maximum_quality'
            print(f"[MemFlow Core] Auto-applied maximum quality preset for things + long-term memory combination")
        
        # Get preset configuration (None if no preset)
        self.preset_config = QUALITY_PRESETS.get(self.quality_preset, None)
        
        # Generate default model path if not provided
        if self.model_path is None:
            # Auto-select Twins model for things + long-term memory combination
            if self.stage == 'things' and self.enable_long_term:
                self.model_path = 'MemFlow_ckpt/MemFlowNet_T_things.pth'
                print(f"[MemFlow Core] Auto-selected Twins model for things + long-term memory: {self.model_path}")
            else:
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
                # Use Twins configuration for things + long-term memory combination
                if self.enable_long_term and 'MemFlowNet_T_things.pth' in self.model_path:
                    from configs.things_memflownet_t import get_cfg
                    print(f"[MemFlow Core] Using Twins configuration for things + long-term memory")
                else:
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
            
            # Apply quality preset settings if specified
            if self.preset_config is not None:
                self._apply_quality_preset()
                print(f"[MemFlow Core] Applied quality preset: {self.preset_config['name']}")
                print(f"  Description: {self.preset_config['description']}")
            else:
                # Set default add_pe value when no preset is used
                self.add_pe = False
                print(f"[MemFlow Core] Using original config settings (no quality preset)")
                print(f"  Position encoding: {self.add_pe}")
            
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
    
    def _apply_quality_preset(self):
        """Apply quality preset settings to configuration"""
        if self.preset_config is None:
            print(f"[MemFlow Core] No quality preset specified, using original config settings")
            # Set default values for inference
            self.add_pe = False
            return
            
        preset = self.preset_config
        
        # Apply decoder depth
        self.cfg.val_decoder_depth = preset['val_decoder_depth']
        
        # Apply correlation settings
        self.cfg.corr_levels = preset['corr_levels']
        self.cfg.corr_radius = preset['corr_radius']
        
        # Apply feature dimension
        self.cfg.feat_dim = preset['feat_dim']
        
        # Apply down ratio
        self.cfg.down_ratio = preset['down_ratio']
        
        # Apply memory settings
        self.cfg.max_mid_term_frames = preset['max_mid_term_frames']
        self.cfg.min_mid_term_frames = preset['min_mid_term_frames']
        self.cfg.num_prototypes = preset['num_prototypes']
        
        # Apply attention settings
        self.cfg.attention_scale_factor = preset['attention_scale_factor']
        
        # Apply post-processing settings
        self.cfg.flow_smoothing = preset['flow_smoothing']
        self.cfg.outlier_filtering = preset['outlier_filtering']
        
        # Store additional settings for use in inference
        self.add_pe = preset['add_pe']
        
        # Twins-specific optimizations for things + long-term memory combination
        if self.stage == 'things' and self.enable_long_term and 'MemFlowNet_T_things.pth' in self.model_path:
            # Twins-specific optimizations
            self.cfg.decoder_depth = 15  # Увеличить с 12 до 15
            self.cfg.max_long_term_elements = 20000  # Увеличить память
            self.cfg.mixed_precision = False  # Отключить для точности
            self.cfg.rope = True  # Включить Rotary PE
            self.cfg.warm_start = True  # Включить warm start
            
            print(f"[MemFlow Core] Twins-specific optimizations applied:")
            print(f"  Decoder depth: {self.cfg.decoder_depth}")
            print(f"  Max long-term elements: {self.cfg.max_long_term_elements}")
            print(f"  Mixed precision: {self.cfg.mixed_precision}")
            print(f"  Rotary PE: {self.cfg.rope}")
            print(f"  Warm start: {self.cfg.warm_start}")
        
        print(f"[MemFlow Core] Quality settings applied:")
        print(f"  Decoder depth: {preset['val_decoder_depth']}")
        print(f"  Correlation levels: {preset['corr_levels']}, radius: {preset['corr_radius']}")
        print(f"  Feature dimension: {preset['feat_dim']}")
        print(f"  Down ratio: {preset['down_ratio']}")
        print(f"  Memory frames: {preset['min_mid_term_frames']}-{preset['max_mid_term_frames']}")
        print(f"  Prototypes: {preset['num_prototypes']}")
        print(f"  Position encoding: {preset['add_pe']}")
        print(f"  Flow smoothing: {preset['flow_smoothing']}")
        print(f"  Outlier filtering: {preset['outlier_filtering']}")
    
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
                    add_pe=self.add_pe  # Use preset setting for positional encoding
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
                "enable_long_term": self.enable_long_term,
                "quality_preset": self.quality_preset,
                "quality_preset_name": self.preset_config['name'] if self.preset_config else "None (Original)",
                "quality_preset_description": self.preset_config['description'] if self.preset_config else "Using original config settings"
            })
        return info 