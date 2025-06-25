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
```

### Parameters

- `--input`: Input video file path
- `--output`: Output video file path (default: videoflow_result.mp4)
- `--device`: Processing device (auto, cuda, cpu)
- `--frames`: Maximum frames to process (default: 1000)
- `--fast`: Enable fast mode (lower quality, faster processing)
- `--tile`: Enable tile-based processing for better quality
- `--sequence-length`: Number of frames in sequence (default: 5, recommended: 5-9)
- `--flow-only`: Output only optical flow visualization
- `--vertical`: Stack videos vertically instead of horizontally

## Technical Details

### VideoFlow Multi-frame Optical Flow (MOF)

The implementation uses VideoFlow MOF model which:
- Analyzes configurable sequences of frames (default: 5, supports 3-9+)
- Generates dense, high-quality optical flow
- Supports multiple pre-trained models (Sintel, KITTI)

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
- **Output**: MP4

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