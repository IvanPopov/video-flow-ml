"""
Common test components for optical flow testing

This package contains reusable components for testing optical flow models:
- SyntheticVideoGenerator: Generate test videos with known ground truth
- OpticalFlowAnalyzer: Analyze flow accuracy against ground truth
- TestRunner: Base class for test execution
"""

from .synthetic_video import SyntheticVideoGenerator
from .flow_analyzer import OpticalFlowAnalyzer
from .test_runner import BaseTestRunner

__all__ = ['SyntheticVideoGenerator', 'OpticalFlowAnalyzer', 'BaseTestRunner'] 