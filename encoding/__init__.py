"""
Encoding module - optical flow visualization and encoding formats
"""

from .flow_encoders import (
    FlowEncoder,
    HSVFlowEncoder,
    GamedevFlowEncoder,
    TorchvisionFlowEncoder,
    MotionVectorsFlowEncoder,
    FlowEncoderFactory,
    encode_flow
)

__all__ = [
    'FlowEncoder',
    'HSVFlowEncoder',
    'GamedevFlowEncoder',
    'TorchvisionFlowEncoder',
    'MotionVectorsFlowEncoder',
    'FlowEncoderFactory',
    'encode_flow'
] 