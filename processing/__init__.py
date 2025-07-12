"""
Processing module for optical flow computation.

This module contains components for:
- VideoFlow model loading and management  
- MemFlow model loading and management
- Optical flow computation using VideoFlow/MemFlow models
- Tile-based processing for large frames

Unified architecture:

Base classes:
- BaseFlowCore: Abstract base class for low-level model operations
- BaseFlowProcessor: Abstract base class for high-level processing pipeline
- BaseFlowInference: Abstract base class for compatibility layer

VideoFlow:
- VideoFlowCore: Low-level model operations (minimal dependencies)
- VideoFlowProcessor: High-level processing pipeline (tiles, progress, validation)
- VideoFlowInference: Compatibility layer (maintains backward compatibility)

MemFlow:
- MemFlowCore: Low-level model operations (minimal dependencies)
- MemFlowProcessor: High-level processing pipeline (compatibility with VideoFlow interface)
- MemFlowInference: Compatibility layer (maintains backward compatibility)
"""

# Base classes
from .base_flow_processor import BaseFlowCore, BaseFlowProcessor, BaseFlowInference

# VideoFlow components
from .videoflow_inference import VideoFlowInference
from .videoflow_core import VideoFlowCore  
from .videoflow_processor import VideoFlowProcessor

# MemFlow components
from .memflow_core import MemFlowCore
from .memflow_processor import MemFlowProcessor
from .memflow_inference import MemFlowInference

# Factory
from .flow_processor_factory import FlowProcessorFactory

__all__ = [
    # Base classes
    'BaseFlowCore',            # Abstract base class for low-level operations
    'BaseFlowProcessor',       # Abstract base class for high-level processing
    'BaseFlowInference',       # Abstract base class for compatibility layer
    
    # VideoFlow components
    'VideoFlowInference',      # Compatibility layer (recommended for existing code)
    'VideoFlowCore',           # Low-level model operations  
    'VideoFlowProcessor',      # High-level processing pipeline
    
    # MemFlow components
    'MemFlowInference',        # Compatibility layer (recommended for existing code)
    'MemFlowCore',             # Low-level model operations
    'MemFlowProcessor',        # High-level processing pipeline
    
    # Factory
    'FlowProcessorFactory'     # Factory for creating processors with unified interface
] 