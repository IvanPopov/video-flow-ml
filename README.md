# VideoFlow Optical Flow Processor

VideoFlow implementation for generating optical flow from video with gamedev format encoding.

## Description

This project processes video files to generate optical flow visualization using the VideoFlow neural network:
- **Left side**: Original video frames
- **Right side**: Optical flow in gamedev format

## Gamedev Format Encoding

Optical flow is encoded into RG channels:
- Flow vectors are normalized relative to image resolution
- Values are clamped to range [-20, +20]
- Encoded as: 0 = -20, 1 = +20
- R channel: horizontal flow
- G channel: vertical flow
- B channel: unused (0)

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Windows (setup.bat) or manual installation

### Quick Setup

1. **Clone the repository with submodules:**
```bash
git clone --recursive https://github.com/your-repo/video-flow-ml.git
cd video-flow-ml
```

2. **Run the setup script:**
```bash
setup.bat
```

This will:
- Check for Python and NVIDIA GPU
- Initialize and update git submodules
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Test the installation

3. **Activate the environment:**
```bash
venv_video_flow\Scripts\activate
```

### Repository Structure

```
video-flow-ml/
├── VideoFlow/                    # VideoFlow source code (git submodule)
├── VideoFlow_ckpt/               # Pre-trained models
│   ├── MOF_sintel.pth           # Multi-frame model for Sintel
│   ├── MOF_kitti.pth            # Multi-frame model for KITTI
│   └── BOF_sintel.pth           # Bi-directional model
├── flow_processor.py            # Main processing script
├── setup.bat                    # Installation script
├── requirements.txt             # Python dependencies
├── .gitmodules                  # Git submodule configuration
└── README.md                    # This file
```

## Usage

### Verify Installation

```bash
python check_cuda.py
```

### Basic Usage

```bash
python flow_processor.py --input your_video.mp4 --output result.mp4
```

### Advanced Options

```bash
# Fast processing mode
python flow_processor.py --input video.mp4 --output result.mp4 --fast

# Tile-based processing for better quality
python flow_processor.py --input video.mp4 --output result.mp4 --tile

# Flow visualization only
python flow_processor.py --input video.mp4 --output result.mp4 --flow-only

# Vertical layout
python flow_processor.py --input video.mp4 --output result.mp4 --vertical

# Use more frames for better accuracy (recommended for complex scenes)
python flow_processor.py --input video.mp4 --output result.mp4 --sequence-length 7

# Save raw optical flow data for further processing
python flow_processor.py --input video.mp4 --output result.mp4 --save-flow flo
python flow_processor.py --input video.mp4 --output result.mp4 --save-flow npz
python flow_processor.py --input video.mp4 --output result.mp4 --save-flow both

# Optical flow caching (automatic - speeds up repeated processing)
python flow_processor.py --input video.mp4 --output result1.mp4 --flow-format gamedev
python flow_processor.py --input video.mp4 --output result2.mp4 --flow-format hsv  # Uses cached flow

# Force recompute flow cache
python flow_processor.py --input video.mp4 --output result.mp4 --force-recompute

# Use specific flow cache directory
python flow_processor.py --input video.mp4 --output result.mp4 --use-flow-cache /path/to/cache

# Interactive flow analysis (GUI with mouse-over arrows)
python flow_processor.py --input video.mp4 --interactive --duration 5 --start-time 10
```

### Parameters

- `--input`: Input video file path
- `--output`: Output video file path (default: videoflow_result.mp4)
- `--device`: Processing device (auto, cuda, cpu)
- `--frames`: Maximum frames to process (default: 1000)
- `--fast`: Enable fast mode (lower quality, faster processing)
- `--tile`: Enable tile-based processing for better quality
- `--sequence-length`: Number of frames in sequence (default: 5, recommended: 5-9)
- `--save-flow`: Save raw optical flow data without compression loss
  - `flo`: Middlebury .flo format (standard, widely supported)
  - `npz`: NumPy .npz format (compressed, includes metadata)
  - `both`: Save in both formats
- `--force-recompute`: Force recomputation of optical flow even if cached data exists
- `--use-flow-cache PATH`: Use optical flow from specific cache directory instead of computing
- `--interactive`: Launch interactive flow visualizer instead of creating video output
- `--flow-only`: Output only optical flow visualization

## Technical Details

### VideoFlow Multi-frame Optical Flow (MOF)

The implementation uses VideoFlow MOF model which:
- Analyzes configurable sequences of frames (default: 5, supports 3-9+)
- Generates dense, high-quality optical flow
- Supports multiple pre-trained models (Sintel, KITTI)

### Automatic Flow Caching

The processor automatically caches computed optical flow to speed up repeated processing:

- **Cache Location**: Next to input video file
- **Cache Naming**: Includes parameters that affect raw optical flow computation
- **Smart Reuse**: Automatically detects and reuses compatible cached flow data
- **Parameter Sensitivity**: Only parameters affecting raw flow create separate caches

#### Cache Directory Structure
```
video_flow_cache_seq5_start0_frames100_tile/
├── flow_frame_000000.npz    # Frame 0 optical flow
├── flow_frame_000001.npz    # Frame 1 optical flow
└── ...                      # Additional frames
```

#### Cache Behavior
- **First Run**: Computes and caches optical flow
- **Subsequent Runs**: Automatically uses cached data if core parameters match
- **Core Parameters**: sequence-length, start-frame, frames, fast, tile (affect optical flow computation)
- **Non-Cache Parameters**: flow-format, taa (applied after loading cache)
- **Force Recompute**: `--force-recompute` flag bypasses cache

#### Parameters Affecting Cache
**Create separate caches:**
- `--sequence-length`: Different sequence lengths produce different flow
- `--start-frame` / `--frames`: Different frame ranges
- `--fast`: Changes resolution and model parameters
- `--tile`: Changes processing method

**Reuse same cache:**
- `--flow-format`: Only affects visualization encoding
- `--taa`: Only affects output layout

### Interactive Flow Visualizer

The `--interactive` mode launches a GUI application for detailed optical flow analysis:

#### Features
- **Frame Navigation**: Slider to browse through frame pairs
- **Mouse Interaction**: Hover over pixels to see flow vectors as arrows
- **Real-time Feedback**: Shows source pixel, flow vector, and target coordinates
- **Dual Frame View**: Current and next frame displayed vertically
- **Automatic Caching**: Computes flow if needed, reuses existing cache

#### Usage
```bash
# Launch interactive visualizer
python flow_processor.py --input video.mp4 --interactive

# Interactive with specific time range
python flow_processor.py --input video.mp4 --interactive --start-time 30 --duration 10

# Interactive with tile mode for better quality
python flow_processor.py --input video.mp4 --interactive --tile --sequence-length 7
```

#### Interface
- **Top Frame**: Current frame (hover to see flow arrows)
- **Bottom Frame**: Next frame (shows where pixels move to)
- **Red Arrow**: Flow vector from hovered pixel to target location
- **Frame Slider**: Navigate through all frame pairs
- **Zoom Controls**: Mouse wheel to zoom in/out, buttons for precise control
- **Status Bar**: Shows pixel coordinates, flow values, target position, and zoom level

#### Controls
- **Mouse Wheel**: Zoom in/out (10% steps, range 10%-500%)
- **Middle Mouse Button**: Hold and drag to pan zoomed frames
- **Double-Click**: Reset zoom and position to default
- **Zoom Buttons**: +/- buttons for precise zoom control
- **Center Button**: Reset pan position to center
- **Hover**: Move mouse over top frame to see flow vectors as arrows
- **Slider**: Navigate through frame pairs
- **Window Resize**: Frames automatically scale to fit window width

#### Frame Information
- **Corner Numbers**: Yellow numbers show absolute frame numbers from video
- **Pair Labels**: Cyan labels show relative pair numbers
- **Statistics Bar**: Shows video name, cache directory, frame range, and counts
- **Time Synchronization**: Displays exact frames from specified time range

### Raw Flow Data Saving

The processor can save uncompressed optical flow data for maximum precision:

#### .flo Format (Middlebury)
- **Standard**: Widely supported by optical flow tools
- **Lossless**: Full float32 precision maintained
- **Structure**: Header (magic + dimensions) + raw flow data
- **Usage**: Research, benchmarking, tool interoperability

#### .npz Format (NumPy)
- **Compressed**: Uses NumPy's compression for smaller files
- **Metadata**: Includes frame index, shape, data type information
- **Flexible**: Easy to load and process in Python
- **Usage**: Data analysis, custom processing pipelines

```python
# Loading saved flow data
processor = VideoFlowProcessor()

# Load .flo file
flow_data = processor.load_flow_flo('flow_frame_000001.flo')

# Load .npz file with metadata
npz_data = processor.load_flow_npz('flow_frame_000001.npz')
flow_data = npz_data['flow']
frame_idx = npz_data['frame_idx']
```

#### Example Usage

```bash
# First run - computes and caches optical flow
python flow_processor.py --input video.mp4 --output gamedev.mp4 --flow-format gamedev

# Second run - uses cached flow (much faster)
python flow_processor.py --input video.mp4 --output hsv.mp4 --flow-format hsv

# Save additional flow data while using cache
python flow_processor.py --input video.mp4 --output result.mp4 --save-flow flo
```

### Gamedev Encoding

```python
# Normalize relative to image size
normalized_flow[:, :, 0] /= image_width   # Horizontal component
normalized_flow[:, :, 1] /= image_height  # Vertical component

# Scale and clamp to [-20, +20]
scaled_flow = normalized_flow * 200
clamped_flow = np.clip(scaled_flow, -20, 20)

# Encode to [0, 1]: 0 = -20, 1 = +20
encoded_flow = (clamped_flow + 20) / 40

# Store in RG channels
rgb_image[:, :, 0] = encoded_flow[:, :, 0]  # R: horizontal flow
rgb_image[:, :, 1] = encoded_flow[:, :, 1]  # G: vertical flow
rgb_image[:, :, 2] = 0.0                    # B: unused
```

## System Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU (8GB+ VRAM recommended)
- 16GB+ RAM recommended

## Supported Formats

- **Input**: MP4, MOV, AVI and other video formats
- **Output**: 
  - **Video**: MP4 format
  - **Flow data**: .flo (Middlebury), .npz (NumPy compressed)

### Output Structure

The processor creates automatic caches and optional explicit flow saves:

```
video_directory/
├── video.mp4                                    # Original video
├── video_flow_cache_seq5_start0_frames100/      # Automatic cache (sequence=5)
│   ├── flow_frame_000000.npz                   # Cached flow data
│   └── ...
├── video_flow_cache_seq7_start0_frames100/      # Different cache (sequence=7)
│   ├── flow_frame_000000.npz
│   └── ...
└── results/
    ├── videoflow_result.mp4                     # Processed video
    └── videoflow_result_flow/                   # Explicit flow save (--save-flow)
        ├── flow_frame_000000.flo                # Frame 0 flow (if --save-flow flo)
        ├── flow_frame_000000.npz                # Frame 0 flow (if --save-flow npz)
        └── ...
```

## Output Examples

The output video contains:
- **Left panel**: Original video frames
- **Right panel**: Optical flow visualization:
  - Black pixels = no movement
  - Red tones = rightward movement
  - Green tones = downward movement
  - Yellow tones = diagonal movement

## Troubleshooting

### CUDA Issues
Run `python check_cuda.py` to verify GPU setup.

### Memory Issues
Use `--fast` or `--tile` flags for large videos.

### Updating VideoFlow Submodule
To update the VideoFlow submodule to the latest version:
```bash
git submodule update --remote VideoFlow
```

## References

- [VideoFlow GitHub](https://github.com/XiaoyuShi97/VideoFlow)
- [VideoFlow Paper (ICCV 2023)](https://arxiv.org/abs/2303.08340) 