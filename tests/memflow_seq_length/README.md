# MemFlow Sequence Length Optimization Test

This test system determines the optimal sequence length for MemFlow optical flow processing by analyzing accuracy and performance across different sequence configurations.

## Overview

The test generates synthetic videos with moving balls and evaluates MemFlow performance with specific sequence lengths: 3, 5, 10, 15, 25, and 50 frames. It measures:

- **Accuracy**: Velocity error, direction error, pixel-level accuracy
- **Speed**: Processing time for each sequence length
- **Memory efficiency**: Implicit in processing performance
- **Optimal configuration**: Balanced accuracy/speed recommendations

## Test Parameters

### Sequence Lengths
- **Default Set**: 3, 5, 10, 15, 25, 50 frames
- **Custom Set**: Configurable via `--seq-lengths` parameter
- **Recommended**: Test 3-15 for balanced results, 25-50 for memory analysis

### Motion Types
- **Slow**: 2 pixels/frame max velocity, 4-second motion period
- **Medium**: 5 pixels/frame max velocity, 3-second motion period  
- **Fast**: 10 pixels/frame max velocity, 2-second motion period

### Video Parameters
- **Resolution**: 504x216 pixels (21:9 aspect ratio)
- **Frame rate**: 30 fps
- **Ball radius**: 20 pixels
- **Default frames**: 120 frames (4 seconds duration)
- **Default duration**: 4 seconds at 30 fps

## Usage

### Quick Start (Batch Files)
```batch
# Quick test (seq lengths 3,5,10 - 90 frames)
run_quick_test.bat

# Full test (seq lengths 3,5,10,15,25,50 - 120 frames)  
run_full_test.bat

# Test all motion types (all seq lengths - 120 frames each)
run_all_motions.bat
```

### Manual Execution
```bash
# Basic test with default sequence lengths
python seq_length_test.py --motion medium --frames 120

# Custom sequence lengths
python seq_length_test.py --motion fast --frames 90 --seq-lengths 3,5,10

# Extended test with long sequences
python seq_length_test.py --motion slow --frames 150 --seq-lengths 3,10,25,50
```

### Command Line Options
- `--motion`: Motion speed (`slow`, `medium`, `fast`) - **required**
- `--frames`: Number of frames to generate (default: 120)
- `--fps`: Frames per second (default: 30)
- `--seq-lengths`: Comma-separated sequence lengths (default: 3,5,10,15,25,50)

## Output Files

All files are generated in the `temp/` directory:

### Results Files
- `seq_length_results_{motion}_{frames}f.json` - Complete test results and analysis
- `ground_truth_{motion}_{frames}f.json` - Mathematical ground truth data
- `test_video_{motion}_{frames}f.mp4` - Generated synthetic video

### Flow Cache Directories
- `test_video_{motion}_{frames}f_flow_cache_memflow_sintel_seq{N}_*` - MemFlow caches for each sequence length

## Test Process

1. **Video Generation**: Creates synthetic video with moving ball and known motion parameters (4 seconds duration)
2. **Sequence Testing**: Runs MemFlow with each specified sequence length
3. **Performance Analysis**: Measures accuracy and processing time for each configuration
4. **Optimization**: Identifies best configurations for accuracy, speed, and overall performance
5. **Reporting**: Generates detailed recommendations and comparative analysis

## Results Analysis

### Accuracy Metrics
- **Velocity Error**: Euclidean distance between predicted and ground truth velocity vectors
- **Direction Error**: Angular difference between velocity vectors (degrees)
- **Magnitude Error**: Absolute difference in velocity magnitudes
- **Pixel Accuracy**: Percentage of frames within 1px, 2px, and 5px thresholds

### Performance Metrics
- **Processing Time**: Total time to process the video sequence
- **Composite Score**: Weighted combination of accuracy (70%) and speed (30%)

### Recommendations
- **Best Accuracy**: Sequence length with highest pixel-level accuracy
- **Best Speed**: Sequence length with fastest processing time
- **Best Overall**: Optimal balance of accuracy and speed

## Expected Patterns

### Typical Results
- **Short sequences (3-5)**: Fast but potentially less accurate
- **Medium sequences (10-15)**: Good balance of speed and accuracy
- **Long sequences (25-50)**: Higher accuracy but slower processing and memory usage

### Memory Considerations
- Longer sequences require more memory for temporal information
- MemFlow's memory management should improve with sequence length up to a point
- Diminishing returns typically occur after 10-15 frames
- Very long sequences (25+) may hit memory limits

## Common Sequence Length Recommendations

Based on typical results:

### For Production Use
- **Real-time applications**: 3-5 frames (balance speed/accuracy)
- **High accuracy needs**: 10-15 frames (optimal memory utilization)
- **Batch processing**: 15-25 frames (maximum accuracy)

### For Different Motion Types
- **Slow motion**: Longer sequences (15-25) often beneficial
- **Fast motion**: Shorter sequences (5-10) may be sufficient
- **Mixed motion**: 10-15 frames provide good general performance

## Troubleshooting

### Common Issues
1. **Out of memory**: Reduce maximum sequence length or frame count
2. **Slow performance**: Use smaller frame count for testing
3. **Cache conflicts**: Different sequence lengths create separate caches automatically
4. **Missing results**: Check temp/ directory for error logs
5. **Sequence too long**: Sequence length must be less than total frames

### Performance Tips
- Start with quick test (90 frames, seq 3,5,10) to verify setup
- Use medium motion for initial testing (most representative)
- Monitor memory usage for sequence lengths >20
- Test with shorter videos first before running full 4-second tests

## Integration with Other Tests

### Shared Components
- Uses common `SyntheticVideoGenerator` from `tests/common/`
- Uses common `OpticalFlowAnalyzer` from `tests/common/`
- Compatible with `velocity_check` test framework

### Complementary Testing
- Run `velocity_check` tests first to verify basic model functionality
- Use sequence length results to optimize `velocity_check` configurations
- Results inform production MemFlow processor settings

## Results Interpretation

### Good Performance Indicators
- Consistent accuracy improvement with longer sequences (up to a point)
- Processing time scaling linearly with sequence length
- Clear optimal range (typically 10-20 frames)

### Potential Issues
- Accuracy degradation with longer sequences: Memory saturation
- Exponential time scaling: Memory management problems
- No accuracy improvement: Model not utilizing temporal information

## Extending the Tests

### Additional Analysis
- Test with different video resolutions
- Evaluate memory usage directly
- Test with different motion patterns (circular, accelerating)
- Compare with VideoFlow sequence length effects

### Custom Motion Parameters
Modify test to use custom motion by editing `SyntheticVideoGenerator`:
```python
# Custom motion: 3 px/frame, 60-frame period
generator.set_custom_motion_parameters(3.0, 60.0)
```

### Custom Sequence Lengths
```bash
# Test specific sequence lengths
python seq_length_test.py --motion medium --seq-lengths 5,20,40

# Test very long sequences (ensure sufficient frames)
python seq_length_test.py --motion slow --frames 200 --seq-lengths 50,75,100
``` 