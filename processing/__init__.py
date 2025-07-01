"""
Processing module for VideoFlow optical flow computation.

This module contains components for:
- VideoFlow model loading and management
- Optical flow computation using VideoFlow models
- Tile-based processing for large frames
"""

from .flow_inference import VideoFlowInference

__all__ = [
    'VideoFlowInference'
] 