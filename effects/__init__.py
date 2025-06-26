"""
Effects module - visual effects processing for optical flow
"""

from .taa_processor import (
    TAAProcessor,
    TAAComparisonProcessor,
    apply_taa_effect
)

__all__ = [
    'TAAProcessor',
    'TAAComparisonProcessor', 
    'apply_taa_effect'
] 