"""
Flow Processor Factory - Factory class for creating optical flow processors

This module provides a factory pattern for creating optical flow processors
with unified interface and configuration management.
"""

from typing import Dict, Any, Union
from .base_flow_processor import BaseFlowProcessor, BaseFlowInference
from .videoflow_processor import VideoFlowProcessor
from .videoflow_inference import VideoFlowInference
from .memflow_processor import MemFlowProcessor
from .memflow_inference import MemFlowInference


class FlowProcessorFactory:
    """
    Factory class for creating optical flow processors
    
    This factory provides a unified interface for creating different types of
    optical flow processors (VideoFlow, MemFlow) with consistent configuration.
    """
    
    SUPPORTED_MODELS = {
        'videoflow': {
            'processor': VideoFlowProcessor,
            'inference': VideoFlowInference,
            'description': 'VideoFlow MOF/BOF models with tile support',
            'supports_tile_mode': True,
            'supports_fast_mode': True,
            'recommended_sequence_length': 5,
            'datasets': ['sintel', 'things', 'kitti'],
            'architectures': ['mof', 'bof'],
            'variants': ['standard', 'noise']
        },
        'memflow': {
            'processor': MemFlowProcessor,
            'inference': MemFlowInference,
            'description': 'MemFlow models with full-frame processing',
            'supports_tile_mode': False,
            'supports_fast_mode': False,
            'recommended_sequence_length': 3,
            'datasets': ['sintel', 'things', 'kitti'],
            'architectures': ['memflow'],
            'variants': ['standard']
        }
    }
    
    @staticmethod
    def create_processor(model: str, device: str = 'cuda', fast_mode: bool = False,
                        tile_mode: bool = False, sequence_length: int = None,
                        **kwargs) -> BaseFlowProcessor:
        """
        Create a flow processor instance
        
        Args:
            model: Model type ('videoflow' or 'memflow')
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            tile_mode: Enable tile-based processing for large frames
            sequence_length: Number of frames to use in sequence for inference
            **kwargs: Additional model-specific parameters
            
        Returns:
            BaseFlowProcessor instance
            
        Raises:
            ValueError: If model is not supported or configuration is invalid
        """
        model = model.lower()
        
        if model not in FlowProcessorFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. "
                           f"Supported models: {list(FlowProcessorFactory.SUPPORTED_MODELS.keys())}")
        
        model_config = FlowProcessorFactory.SUPPORTED_MODELS[model]
        
        # Set default sequence length if not provided
        if sequence_length is None:
            sequence_length = model_config['recommended_sequence_length']
        
        # Validate configuration
        if tile_mode and not model_config['supports_tile_mode']:
            print(f"Warning: {model} doesn't support tile mode. Disabling tile mode.")
            tile_mode = False
        
        if fast_mode and not model_config['supports_fast_mode']:
            print(f"Warning: {model} doesn't support fast mode. Disabling fast mode.")
            fast_mode = False
        
        # Create processor
        processor_class = model_config['processor']
        
        if model == 'videoflow':
            # VideoFlow-specific parameters
            dataset = kwargs.get('dataset', 'sintel')
            architecture = kwargs.get('architecture', 'mof')
            variant = kwargs.get('variant', 'standard')
            
            # Validate VideoFlow parameters
            if dataset not in model_config['datasets']:
                raise ValueError(f"Unsupported dataset for VideoFlow: {dataset}. "
                               f"Supported: {model_config['datasets']}")
            
            if architecture not in model_config['architectures']:
                raise ValueError(f"Unsupported architecture for VideoFlow: {architecture}. "
                               f"Supported: {model_config['architectures']}")
            
            if variant not in model_config['variants']:
                raise ValueError(f"Unsupported variant for VideoFlow: {variant}. "
                               f"Supported: {model_config['variants']}")
            
            return processor_class(device, fast_mode, tile_mode, sequence_length,
                                 dataset, architecture, variant)
        
        elif model == 'memflow':
            # MemFlow-specific parameters
            stage = kwargs.get('stage', 'sintel')
            model_path = kwargs.get('model_path', None)
            enable_long_term = kwargs.get('enable_long_term', False)
            
            # Validate MemFlow parameters
            if stage not in model_config['datasets']:
                raise ValueError(f"Unsupported stage for MemFlow: {stage}. "
                               f"Supported: {model_config['datasets']}")
            
            return processor_class(device, fast_mode, tile_mode, sequence_length,
                                 stage, model_path, enable_long_term)
        
        else:
            raise ValueError(f"Internal error: Unknown model type: {model}")
    
    @staticmethod
    def create_inference(model: str, device: str = 'cuda', fast_mode: bool = False,
                        tile_mode: bool = False, sequence_length: int = None,
                        **kwargs) -> BaseFlowInference:
        """
        Create a flow inference instance (compatibility layer)
        
        Args:
            model: Model type ('videoflow' or 'memflow')
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            fast_mode: Enable fast mode with reduced model complexity
            tile_mode: Enable tile-based processing for large frames
            sequence_length: Number of frames to use in sequence for inference
            **kwargs: Additional model-specific parameters
            
        Returns:
            BaseFlowInference instance
            
        Raises:
            ValueError: If model is not supported or configuration is invalid
        """
        model = model.lower()
        
        if model not in FlowProcessorFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. "
                           f"Supported models: {list(FlowProcessorFactory.SUPPORTED_MODELS.keys())}")
        
        model_config = FlowProcessorFactory.SUPPORTED_MODELS[model]
        
        # Set default sequence length if not provided
        if sequence_length is None:
            sequence_length = model_config['recommended_sequence_length']
        
        # Validate configuration
        if tile_mode and not model_config['supports_tile_mode']:
            print(f"Warning: {model} doesn't support tile mode. Disabling tile mode.")
            tile_mode = False
        
        if fast_mode and not model_config['supports_fast_mode']:
            print(f"Warning: {model} doesn't support fast mode. Disabling fast mode.")
            fast_mode = False
        
        # Create inference layer
        inference_class = model_config['inference']
        
        if model == 'videoflow':
            # VideoFlow-specific parameters
            dataset = kwargs.get('dataset', 'sintel')
            architecture = kwargs.get('architecture', 'mof')
            variant = kwargs.get('variant', 'standard')
            
            # Validate VideoFlow parameters
            if dataset not in model_config['datasets']:
                raise ValueError(f"Unsupported dataset for VideoFlow: {dataset}. "
                               f"Supported: {model_config['datasets']}")
            
            if architecture not in model_config['architectures']:
                raise ValueError(f"Unsupported architecture for VideoFlow: {architecture}. "
                               f"Supported: {model_config['architectures']}")
            
            if variant not in model_config['variants']:
                raise ValueError(f"Unsupported variant for VideoFlow: {variant}. "
                               f"Supported: {model_config['variants']}")
            
            return inference_class(device, fast_mode, tile_mode, sequence_length,
                                 dataset, architecture, variant)
        
        elif model == 'memflow':
            # MemFlow-specific parameters
            stage = kwargs.get('stage', 'sintel')
            model_path = kwargs.get('model_path', None)
            enable_long_term = kwargs.get('enable_long_term', False)
            
            # Validate MemFlow parameters
            if stage not in model_config['datasets']:
                raise ValueError(f"Unsupported stage for MemFlow: {stage}. "
                               f"Supported: {model_config['datasets']}")
            
            return inference_class(device, fast_mode, tile_mode, sequence_length,
                                 stage, model_path, enable_long_term)
        
        else:
            raise ValueError(f"Internal error: Unknown model type: {model}")
    
    @staticmethod
    def get_model_info(model: str = None) -> Dict[str, Any]:
        """
        Get information about supported models
        
        Args:
            model: Specific model to get info for, or None for all models
            
        Returns:
            Dictionary with model information
        """
        if model is None:
            return FlowProcessorFactory.SUPPORTED_MODELS
        
        model = model.lower()
        if model not in FlowProcessorFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. "
                           f"Supported models: {list(FlowProcessorFactory.SUPPORTED_MODELS.keys())}")
        
        return FlowProcessorFactory.SUPPORTED_MODELS[model]
    
    @staticmethod
    def list_supported_models() -> list:
        """Get list of supported model names"""
        return list(FlowProcessorFactory.SUPPORTED_MODELS.keys())
    
    @staticmethod
    def create_auto(device: str = 'cuda', prefer_tile_mode: bool = True,
                   prefer_fast_mode: bool = False, sequence_length: int = None,
                   **kwargs) -> BaseFlowProcessor:
        """
        Create a processor with automatic model selection
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device)
            prefer_tile_mode: Prefer models that support tile mode
            prefer_fast_mode: Prefer models that support fast mode
            sequence_length: Number of frames to use in sequence for inference
            **kwargs: Additional model-specific parameters
            
        Returns:
            BaseFlowProcessor instance
        """
        # Default to VideoFlow if no specific preference
        if prefer_tile_mode:
            model = 'videoflow'
            tile_mode = True
        else:
            model = 'videoflow'  # Still default to VideoFlow
            tile_mode = False
        
        print(f"Auto-selecting model: {model}")
        print(f"  Tile mode: {tile_mode}")
        print(f"  Fast mode: {prefer_fast_mode}")
        
        return FlowProcessorFactory.create_processor(
            model, device, prefer_fast_mode, tile_mode, sequence_length, **kwargs
        )
    
    @staticmethod
    def validate_model_config(model: str, **kwargs) -> bool:
        """
        Validate model configuration without creating instance
        
        Args:
            model: Model type ('videoflow' or 'memflow')
            **kwargs: Configuration parameters to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        model = model.lower()
        
        if model not in FlowProcessorFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        
        model_config = FlowProcessorFactory.SUPPORTED_MODELS[model]
        
        # Check tile mode support
        if kwargs.get('tile_mode', False) and not model_config['supports_tile_mode']:
            raise ValueError(f"{model} doesn't support tile mode")
        
        # Check fast mode support
        if kwargs.get('fast_mode', False) and not model_config['supports_fast_mode']:
            raise ValueError(f"{model} doesn't support fast mode")
        
        # Model-specific validation
        if model == 'videoflow':
            dataset = kwargs.get('dataset', 'sintel')
            architecture = kwargs.get('architecture', 'mof')
            variant = kwargs.get('variant', 'standard')
            
            if dataset not in model_config['datasets']:
                raise ValueError(f"Unsupported dataset for VideoFlow: {dataset}")
            
            if architecture not in model_config['architectures']:
                raise ValueError(f"Unsupported architecture for VideoFlow: {architecture}")
            
            if variant not in model_config['variants']:
                raise ValueError(f"Unsupported variant for VideoFlow: {variant}")
        
        elif model == 'memflow':
            stage = kwargs.get('stage', 'sintel')
            
            if stage not in model_config['datasets']:
                raise ValueError(f"Unsupported stage for MemFlow: {stage}")
        
        return True 