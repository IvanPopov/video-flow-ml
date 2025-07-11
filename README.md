# VideoFlow Optical Flow Processor

VideoFlow implementation for generating optical flow from video with gamedev format encoding.

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
- Create virtual environment in `.venv` folder
- Install PyTorch with CUDA support (if GPU detected)
- Install all dependencies
- Test the installation

3. **Activate the environment (required for each session):**

**Option 1 - Automatic activation script:**
```bash
activate.bat
```

**Option 2 - Manual activation:**
```bash
.venv\Scripts\activate
```

**Note:** You need to activate the virtual environment every time you start a new terminal session before using the project.

### Repository Structure

```
video-flow-ml/
├── VideoFlow/                    # VideoFlow source code (git submodule)
├── VideoFlow_ckpt/               # Pre-trained models
│   ├── MOF_sintel.pth           # Multi-frame model for Sintel
│   ├── MOF_kitti.pth            # Multi-frame model for KITTI
│   └── BOF_sintel.pth           # Bi-directional model
├── .venv/                       # Virtual environment (created by setup.bat)
├── flow_processor.py            # Main processing script
├── gui_runner.py                # GUI application
├── check_cuda.py                # CUDA verification script
├── setup.bat                    # Installation script
├── activate.bat                 # Environment activation script
├── requirements.txt             # Python dependencies
├── .gitmodules                  # Git submodule configuration
└── README.md                    # This file
```

## Usage

**Important:** Before running any commands, make sure to activate the virtual environment:
```bash
activate.bat
```

### Verify Installation

```bash
python check_cuda.py
```

### Basic Usage

**Command Line Interface (CLI):**
```bash
python flow_processor.py --input your_video.mp4 --output result.mp4
```

**Graphical User Interface (GUI):**
```bash
python gui_runner.py
```

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