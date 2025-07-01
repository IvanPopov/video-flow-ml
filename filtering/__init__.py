"""
Filtering module for optical flow stabilization and smoothing.

This module contains components for:
- Adaptive Kalman filtering for optical flow
- Flow stabilization algorithms
- Temporal smoothing techniques
"""

from .kalman_filter import AdaptiveOpticalFlowKalmanFilter

__all__ = [
    'AdaptiveOpticalFlowKalmanFilter'
] 