"""
Processing module for VideoFlow optical flow computation.

This module contains components for:
- VideoFlow model loading and management
- Optical flow computation using VideoFlow models
- Tile-based processing for large frames

New modular architecture:
- VideoFlowCore: Low-level model operations (minimal dependencies)
- VideoFlowProcessor: High-level processing pipeline (tiles, progress, validation)
- VideoFlowInference: Compatibility layer (maintains backward compatibility)
"""

from .flow_inference import VideoFlowInference
from .videoflow_core import VideoFlowCore  
from .videoflow_processor import VideoFlowProcessor

__all__ = [
    'VideoFlowInference',      # Compatibility layer (recommended for existing code)
    'VideoFlowCore',           # Low-level model operations  
    'VideoFlowProcessor'       # High-level processing pipeline
] 