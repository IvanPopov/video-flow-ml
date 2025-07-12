"""
Processing module for optical flow computation.

This module contains components for:
- VideoFlow model loading and management  
- MemFlow model loading and management
- Optical flow computation using VideoFlow/MemFlow models
- Tile-based processing for large frames

Simplified architecture:

VideoFlow:
- VideoFlowCore: Low-level model operations (minimal dependencies)
- VideoFlowProcessor: High-level processing pipeline (tiles, progress, validation)
- VideoFlowInference: Compatibility layer (maintains backward compatibility)

MemFlow:
- MemFlowProcessor: Direct integration with MemFlow (based on original inference.py)
- MemFlowInference: Compatibility layer (maintains backward compatibility)
"""

from .flow_inference import VideoFlowInference
from .videoflow_core import VideoFlowCore  
from .videoflow_processor import VideoFlowProcessor
from .memflow_processor import MemFlowProcessor
from .memflow_inference import MemFlowInference

__all__ = [
    # VideoFlow components
    'VideoFlowInference',      # Compatibility layer (recommended for existing code)
    'VideoFlowCore',           # Low-level model operations  
    'VideoFlowProcessor',      # High-level processing pipeline
    
    # MemFlow components
    'MemFlowInference',        # Compatibility layer (recommended for existing code)
    'MemFlowProcessor'         # Direct integration processor
] 