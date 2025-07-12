# Velocity Check Test System

This test system generates synthetic videos with moving balls and analyzes the accuracy of optical flow models (VideoFlow and MemFlow) by comparing their predictions against ground truth velocities.

## Overview

The test creates a synthetic video with a white ball moving in a sinusoidal pattern on a dark background. The ball's position and velocity are calculated mathematically, providing perfect ground truth data for comparison.

## Test Parameters

### Motion Speeds
- **Slow**: 2 pixels/frame max velocity, 4-second motion period
- **Medium**: 5 pixels/frame max velocity, 3-second motion period  
- **Fast**: 10 pixels/frame max velocity, 2-second motion period

### Video Parameters
- Resolution: 640x480 pixels
- Frame rate: 30 fps
- Ball radius: 20 pixels
- Default frames: 60 frames

## Usage

### Quick Start (Batch Files)
```batch
# Run individual tests
run_slow.bat     # Test with slow motion
run_medium.bat   # Test with medium motion
run_fast.bat     # Test with fast motion

# Run all tests
run_all.bat      # Run all three tests consecutively
```

### Manual Execution
```bash
# Run specific test
python velocity_test.py --speed slow --frames 60 --fps 30

# Custom parameters
python velocity_test.py --speed medium --frames 120 --fps 60
```

## Output Files

All files are generated in the `temp/` directory:

### Video Files
- `test_video_slow.mp4` - Synthetic video with slow motion
- `test_video_medium.mp4` - Synthetic video with medium motion  
- `test_video_fast.mp4` - Synthetic video with fast motion

### Ground Truth Files
- `ground_truth_slow.json` - Mathematical ground truth for slow motion
- `ground_truth_medium.json` - Mathematical ground truth for medium motion
- `ground_truth_fast.json` - Mathematical ground truth for fast motion

### Results Files
- `results_slow.json` - Analysis results for slow motion
- `results_medium.json` - Analysis results for medium motion
- `results_fast.json` - Analysis results for fast motion

### Flow Cache Directories
- `test_video_slow_flow_cache_videoflow_*` - VideoFlow optical flow cache
- `test_video_slow_flow_cache_memflow_*` - MemFlow optical flow cache
- (Similar for medium and fast)

## Test Process

1. **Video Generation**: Creates synthetic video with moving ball
2. **Flow Computation**: Runs both VideoFlow and MemFlow models
3. **Analysis**: Compares predicted flow vectors against ground truth
4. **Reporting**: Generates detailed accuracy statistics

## Accuracy Metrics

### Error Measurements
- **Velocity Error**: Euclidean distance between predicted and ground truth velocity vectors
- **Direction Error**: Angular difference between velocity vectors (in degrees)
- **Magnitude Error**: Absolute difference in velocity magnitudes

### Accuracy Thresholds
- **< 1px**: Percentage of frames with velocity error less than 1 pixel
- **< 2px**: Percentage of frames with velocity error less than 2 pixels
- **< 5px**: Percentage of frames with velocity error less than 5 pixels

## Expected Results

### VideoFlow
- Should provide accurate velocity predictions
- Typical accuracy: 80-95% within 1-2 pixels
- Good direction and magnitude consistency

### MemFlow
- May show timing issues (flow for wrong frame)
- Potentially higher velocity errors
- May require temporal offset correction

## Actual Test Results

**⚠️ IMPORTANT FINDINGS**: Both models show significant accuracy issues with synthetic test data:

### VideoFlow Results (Major Issues Found)
- **Mean Error**: 27.94 px/frame
- **Direction Error**: 169.7° (nearly opposite direction)
- **Accuracy**: 0% (no frames within 2px threshold)
- **Issue**: Complete direction reversal - predicts leftward movement when ball moves rightward
- **Root Cause**: Coordinate system interpretation problems

### MemFlow Results (Better but Still Issues)
- **Mean Error**: 11.72 px/frame (significantly better than VideoFlow)
- **Direction Error**: 85.8° (moderate direction issues)
- **Accuracy**: 0% (still no frames within 2px threshold)
- **Issue**: Temporal offset problems - early frames much worse than later frames
- **Root Cause**: Model warmup or temporal sequence indexing issues

**Conclusion**: Both models require debugging before production use. See `FINDINGS.md` for detailed analysis and recommendations.

## Troubleshooting

### Common Issues
1. **Missing Models**: Ensure VideoFlow and MemFlow models are available
2. **CUDA Errors**: Check GPU availability and CUDA installation
3. **Flow Cache Not Found**: Verify flow processor completed successfully
4. **Import Errors**: Ensure project root is in Python path

### Debug Mode
Add `--frames 10` to run shorter tests for debugging:
```bash
python velocity_test.py --speed slow --frames 10 --fps 30
```

## Analysis Interpretation

### Good Results
- Low velocity error (< 1-2 pixels)
- Low direction error (< 10 degrees)
- High accuracy percentages (> 80%)

### Potential Issues
- High velocity error: Model not tracking motion correctly
- High direction error: Flow vectors pointing wrong direction
- Low accuracy percentages: Systematic prediction errors

### Temporal Offset Issues
If MemFlow shows consistent offset patterns, the model may be:
- Predicting flow for previous frame
- Using wrong temporal sequence
- Requiring different frame indexing

## Extending the Tests

### Custom Motion Patterns
Modify `SyntheticVideoGenerator.calculate_position_and_velocity()` to test:
- Linear motion
- Circular motion
- Acceleration/deceleration
- Multiple objects

### Additional Metrics
Extend `OpticalFlowAnalyzer` to measure:
- Temporal consistency
- Spatial smoothness
- Edge preservation
- Motion blur handling 