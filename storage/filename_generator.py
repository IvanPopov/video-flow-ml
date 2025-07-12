#!/usr/bin/env python3
"""
Output Filename Generator

Utility module for generating consistent output filenames for video processing.
Used by both flow_processor.py and gui_runner.py to avoid code duplication.
"""

import os
from typing import Optional


def generate_output_filename(input_path: str, 
                           start_time: Optional[float] = None,
                           duration: Optional[float] = None,
                           start_frame: int = 0,
                           max_frames: int = 1000,
                           flow_only: bool = False,
                           taa: bool = False,
                           fast_mode: bool = False,
                           tile_mode: bool = False,
                           uncompressed: bool = False,
                           flow_format: str = 'gamedev',
                           motion_vectors_clamp_range: float = 32.0,
                           fps: float = 30.0,
                           model: str = 'videoflow',
                           stage: str = 'sintel',
                           vf_dataset: str = 'sintel',
                           vf_architecture: str = 'mof',
                           vf_variant: str = 'standard',
                           extra_params: Optional[dict] = None) -> str:
    """
    Generate automatic output filename based on processing parameters.
    
    Args:
        input_path: Path to input video file
        start_time: Starting time in seconds (None if not time-based)
        duration: Duration in seconds (None if not time-based)
        start_frame: Starting frame number
        max_frames: Maximum number of frames to process
        flow_only: Whether processing in flow-only mode
        taa: Whether TAA is enabled
        fast_mode: Whether fast mode is enabled
        tile_mode: Whether tile mode is enabled
        uncompressed: Whether using uncompressed output
        flow_format: Flow visualization format
        motion_vectors_clamp_range: Clamp range for motion vectors formats
        fps: Video frame rate for filename
        model: Model type ('videoflow' or 'memflow')
        stage: Stage/dataset for MemFlow model
        vf_dataset: Dataset for VideoFlow model
        vf_architecture: Architecture for VideoFlow model
        vf_variant: Variant for VideoFlow model
        
    Returns:
        Generated filename (without directory path)
    """
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Build filename parts
    parts = [base_name]
    
    # Add model information
    if model == 'videoflow':
        # VideoFlow model: architecture + dataset + variant
        if vf_variant == 'noise' and vf_dataset == 'things':
            model_part = f"VF_{vf_architecture.upper()}_{vf_dataset}_noise"
        else:
            model_part = f"VF_{vf_architecture.upper()}_{vf_dataset}"
    elif model == 'memflow':
        # MemFlow model: stage
        model_part = f"MF_{stage}"
        # Add long-term memory info - default to 'st' if not specified
        if extra_params and 'long_term' in extra_params:
            model_part += f"_{extra_params['long_term']}"
        else:
            # Default to short-term memory for MemFlow
            model_part += "_st"
    else:
        # Fallback
        model_part = model.upper()
    
    parts.append(model_part)
    
    # Add time/frame info
    if start_time is not None:
        parts.append(f"{start_time}s")
    elif start_frame > 0:
        parts.append(f"f{start_frame}")
        
    if duration is not None:
        parts.append(f"{duration}s")
    elif max_frames != 1000:
        parts.append(f"{max_frames}f")
    
    # Add mode info
    if fast_mode:
        parts.append("fast")
    
    if tile_mode:
        parts.append("tile")
    
    if flow_only:
        # Add format information for flow-only mode (avoid repeating "flow")
        if flow_format != 'gamedev':
            # Clean up format name: replace dashes with underscores, remove redundant "flow"
            clean_format = flow_format.replace('-', '_').replace('_flow', '').replace('flow_', '')
            if flow_format.startswith('motion-vectors'):
                # Add motion vectors format with clamp range
                parts.append(f"{clean_format}_{int(motion_vectors_clamp_range)}")
            else:
                parts.append(clean_format)
        else:
            parts.append("gamedev")
    elif taa:
        parts.append("taa")
    
    # Add FPS information
    parts.append(f"{fps:.0f}fps")
    
    # Add codec information
    if uncompressed:
        parts.append("uncompressed_I420")  # Raw I420 codec
    else:
        parts.append("MJPG")  # Default MJPEG
    
    # Join parts and add extension
    # MJPG codec requires AVI container, MP4 container doesn't support it
    extension = ".avi"  # Use AVI for MJPG compatibility
    filename = "_".join(parts) + extension
    
    return filename


def generate_output_filepath(input_path: str,
                           output_dir: str,
                           start_time: Optional[float] = None,
                           duration: Optional[float] = None,
                           start_frame: int = 0,
                           max_frames: int = 1000,
                           flow_only: bool = False,
                           taa: bool = False,
                           fast_mode: bool = False,
                           tile_mode: bool = False,
                           uncompressed: bool = False,
                           flow_format: str = 'gamedev',
                           motion_vectors_clamp_range: float = 32.0,
                           fps: float = 30.0,
                           model: str = 'videoflow',
                           stage: str = 'sintel',
                           vf_dataset: str = 'sintel',
                           vf_architecture: str = 'mof',
                           vf_variant: str = 'standard',
                           extra_params: Optional[dict] = None) -> str:
    """
    Generate complete output filepath (directory + filename).
    
    Args:
        input_path: Path to input video file
        output_dir: Output directory path
        (other args same as generate_output_filename)
        
    Returns:
        Complete filepath (directory + filename)
    """
    filename = generate_output_filename(
        input_path=input_path,
        start_time=start_time,
        duration=duration,
        start_frame=start_frame,
        max_frames=max_frames,
        flow_only=flow_only,
        taa=taa,
        fast_mode=fast_mode,
        tile_mode=tile_mode,
        uncompressed=uncompressed,
        flow_format=flow_format,
        motion_vectors_clamp_range=motion_vectors_clamp_range,
        fps=fps,
        model=model,
        stage=stage,
        vf_dataset=vf_dataset,
        vf_architecture=vf_architecture,
        vf_variant=vf_variant,
        extra_params=extra_params
    )
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return os.path.join(output_dir, filename)


def generate_cache_directory(input_path: str, 
                           start_frame: int = 0,
                           max_frames: int = 1000,
                           sequence_length: int = 5,
                           fast_mode: bool = False,
                           tile_mode: bool = False,
                           model: str = 'videoflow',
                           dataset: str = 'things',
                           architecture: str = 'mof',
                           variant: str = 'noise',
                           extra_params: Optional[dict] = None) -> str:
    """
    Generate cache directory path based on video processing parameters and model configuration.
    
    Args:
        input_path: Path to input video file
        start_frame: Starting frame number
        max_frames: Maximum number of frames to process
        sequence_length: Sequence length for models
        fast_mode: Whether fast mode is enabled
        tile_mode: Whether tile mode is enabled
        model: Model type ('videoflow' or 'memflow')
        dataset: Dataset for model
        architecture: Architecture for VideoFlow models ('mof' or 'bof')
        variant: Variant type ('standard', 'noise', etc.)
        
    Returns:
        Generated cache directory path
    """
    import os
    from pathlib import Path
    
    video_name = Path(input_path).stem
    
    # Model configuration parameters - include all parameters for uniqueness
    model_params = [model]
    if model == 'videoflow':
        # Always include all VideoFlow parameters to ensure unique cache paths
        model_params.append(architecture)
        model_params.append(dataset)
        model_params.append(variant)
    elif model == 'memflow':
        # Always include dataset for MemFlow
        model_params.append(dataset)
        # Add long-term memory info - default to 'st' if not specified
        if extra_params and 'long_term' in extra_params:
            model_params.append(extra_params['long_term'])
        else:
            # Default to short-term memory for MemFlow
            model_params.append('st')
    
    # Processing parameters
    cache_params = [
        f"seq{sequence_length}",
        f"start{start_frame}",
        f"frames{max_frames}"
    ]
    
    if fast_mode:
        cache_params.append("fast")
    if tile_mode:
        cache_params.append("tile")
    
    # Combine model and processing parameters
    model_id = "_".join(model_params)
    cache_id = "_".join(cache_params)
    cache_dir_name = f"{video_name}_flow_cache_{model_id}_{cache_id}"
    
    cache_path = Path(input_path).parent / cache_dir_name
    return str(cache_path) 