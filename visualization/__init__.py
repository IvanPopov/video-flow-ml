"""
Visualization module - video composition and output generation
"""

from .video_composer import VideoComposer, create_side_by_side, add_text_overlay, create_video_grid

__all__ = [
    'VideoComposer',
    'create_side_by_side',
    'add_text_overlay', 
    'create_video_grid'
] 