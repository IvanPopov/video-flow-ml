# Velocity Test Findings Report

## Summary

This report summarizes the findings from synthetic velocity tests comparing VideoFlow and MemFlow optical flow models. The tests used mathematically precise ground truth data to identify accuracy issues and temporal inconsistencies.

## Test Setup

- **Synthetic Video**: White ball moving sinusoidally on dark background
- **Resolution**: 640x480 pixels
- **Motion**: Horizontal movement only (Y=0), X velocities 14-16 px/frame
- **Ground Truth**: Mathematically calculated positions and velocities
- **Test Duration**: 10 frames at 30 fps

## Key Findings

### 1. VideoFlow Issues

**Problem**: Complete direction reversal
- **Mean Error**: 27.94 px/frame
- **Direction Error**: 169.7° (nearly opposite direction)
- **Accuracy**: 0% (no frames within 2px threshold)

**Specific Issues**:
- Ground truth shows rightward movement (+16 px/frame)
- VideoFlow predicts leftward movement (-16 px/frame)
- Almost perfect magnitude but completely wrong direction
- Suggests coordinate system interpretation problems

**Frame Analysis**:
```
Frame 0: GT=(16.3, 0.0) → Pred=(-0.1, -0.2)  [Small movement]
Frame 1: GT=(16.3, 0.0) → Pred=(-0.1, 0.1)   [Small movement]
Frame 2: GT=(16.2, 0.0) → Pred=(-16.0, 0.0)  [Sudden large opposite movement]
```

### 2. MemFlow Issues

**Problem**: Temporal offset and warmup issues
- **Mean Error**: 11.72 px/frame (significantly better than VideoFlow)
- **Direction Error**: 85.8° (moderate direction issues)
- **Accuracy**: 0% (still no frames within 2px threshold)

**Specific Issues**:
- Early frames: 17.3 px/frame error
- Later frames: 7.3 px/frame error
- Suggests temporal offset or model warmup problems
- Better magnitude accuracy in later frames

**Frame Analysis**:
```
Frame 0: GT=(16.3, 0.0) → Pred=(-1.1, -0.2)  [Small leftward movement]
Frame 1: GT=(16.3, 0.0) → Pred=(-1.1, -0.2)  [Small leftward movement]
Frame 2: GT=(16.2, 0.0) → Pred=(-1.2, 0.2)   [Small leftward movement]
Frame 4: GT=(15.9, 0.0) → Pred=(11.1, -2.6)  [Better rightward movement]
Frame 5: GT=(15.6, 0.0) → Pred=(8.5, -3.8)   [Rightward movement]
```

## Root Cause Analysis

### VideoFlow Problems

1. **Coordinate System Issue**: The model consistently predicts opposite X-direction
2. **Temporal Inconsistency**: Sudden jump from small to large predictions
3. **Possible Causes**:
   - Wrong coordinate system convention (image vs world coordinates)
   - Incorrect frame ordering in sequence
   - Model trained with different coordinate assumptions

### MemFlow Problems

1. **Temporal Offset**: Clear pattern of early frames being worse
2. **Warmup Issue**: Model may need several frames to "warm up"
3. **Possible Causes**:
   - Memory initialization problems
   - Temporal sequence indexing issues
   - Model designed for longer sequences

## Recommendations

### Immediate Actions

1. **Fix VideoFlow Coordinate System**:
   - Investigate X-axis coordinate conventions
   - Check frame ordering in sequence preparation
   - Verify model input/output coordinate systems

2. **Fix MemFlow Temporal Handling**:
   - Investigate memory initialization
   - Check frame indexing in sequence
   - Consider temporal offset correction

3. **Validation**:
   - Run tests with more frames to confirm patterns
   - Test with different motion patterns (vertical, circular)
   - Compare with reference implementations

### Long-term Improvements

1. **Model Calibration**:
   - Develop coordinate system validation tests
   - Create temporal consistency benchmarks
   - Implement model-specific corrections

2. **Testing Framework**:
   - Extend synthetic tests to cover more scenarios
   - Add real-world validation data
   - Implement automated regression testing

## Technical Details

### Test Environment
- **GPU**: NVIDIA GeForce RTX 4060 Ti
- **CUDA**: Available and used
- **Models**: 
  - VideoFlow: MOF_sintel.pth
  - MemFlow: MemFlowNet_sintel.pth

### Generated Files
- `test_video_slow.mp4`: Synthetic test video
- `ground_truth_slow.json`: Mathematical ground truth
- `results_slow.json`: Detailed analysis results
- Flow caches: NPZ files with raw optical flow data

## Conclusion

Both models show significant accuracy issues with the synthetic test case:

- **VideoFlow**: Systematic direction reversal (coordinate system bug)
- **MemFlow**: Temporal offset issues but better overall accuracy

The synthetic testing approach successfully identified specific technical issues that would be difficult to detect with real-world data. These findings suggest that both models require debugging of their coordinate systems and temporal handling before they can be considered reliable for production use.

**Priority**: Address VideoFlow coordinate system issues first (complete direction reversal), then investigate MemFlow temporal offset problems.

## Next Steps

1. Debug VideoFlow coordinate system implementation
2. Investigate MemFlow temporal sequence handling
3. Run extended tests with fixed implementations
4. Validate fixes with real-world data
5. Implement automated testing in CI/CD pipeline 