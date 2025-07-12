# MemFlow Long-Term Memory Test

This test evaluates the effectiveness of MemFlow's long-term memory feature by comparing short-term (st) and long-term (lt) memory configurations on scenarios where temporal consistency is crucial.

## Test Scenario: Ball Behind Obstacles

The test generates synthetic video with a moving ball that periodically goes behind rectangular obstacles. This scenario is designed to demonstrate the importance of long-term memory:

- **Short-term memory (st)**: May lose track of the ball when it's occluded for several frames
- **Long-term memory (lt)**: Should maintain better tracking by remembering the ball's trajectory before occlusion

## Test Features

### Video Generation
- Ball moves in a sinusoidal pattern across the screen
- Rectangular obstacles are placed at strategic positions
- Ball becomes occluded for 10-15 frames at a time
- Total video length: 180 frames (6 seconds at 30fps)

### Evaluation Metrics
- **Occlusion Recovery**: How quickly the model recovers tracking after occlusion
- **Trajectory Consistency**: How well the model maintains the ball's path during occlusion
- **Velocity Accuracy**: Comparison of predicted vs actual ball velocity
- **Processing Time**: Performance comparison between st and lt modes

### Test Configurations
- **Short-term memory**: `long_term=st` (default MemFlow behavior)
- **Long-term memory**: `long_term=lt` (enhanced memory mode)
- **Dataset**: sintel (optimized for general motion)

## Usage

### Quick Test (30 seconds)
```bash
run_quick_test.bat
```

### Full Test (2-3 minutes)
```bash
run_full_test.bat
```

### Custom Test
```bash
python long_term_test.py --frames 180 --fps 30 --dataset sintel
```

## Expected Results

### Long-term Memory Advantages
- Better tracking during occlusion periods
- More consistent velocity predictions
- Improved trajectory continuity
- Higher accuracy in occlusion recovery

### Performance Considerations
- Long-term memory may be slightly slower due to additional memory management
- Memory usage will be higher with long-term mode
- The trade-off between accuracy and performance should be evaluated

## Output Files

- `test_video_occlusion_180f.mp4`: Generated test video
- `ground_truth_occlusion_180f.json`: Ground truth data
- `long_term_results_occlusion_180f.json`: Test results
- `comparison_plots.png`: Visual comparison of st vs lt performance

## Analysis

The test provides detailed analysis of:
1. **Occlusion Periods**: Performance during ball occlusion
2. **Recovery Time**: How many frames needed to re-acquire tracking
3. **Trajectory Deviation**: How much the predicted path deviates from ground truth
4. **Overall Accuracy**: Combined metrics for both visible and occluded periods

## Recommendations

Based on test results, recommendations are provided for:
- When to use long-term memory
- Optimal sequence lengths for different scenarios
- Performance vs accuracy trade-offs
- Memory usage considerations 