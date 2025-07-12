# Video Flow ML Tests

This directory contains various tests for the video-flow-ml project, designed to validate optical flow models and identify potential issues.

## Test Suites

### Velocity Check Tests (`velocity_check/`)

Synthetic video tests that generate mathematically precise ground truth data to validate optical flow accuracy.

**Purpose**: Detect coordinate system issues, temporal inconsistencies, and accuracy problems in optical flow models.

**Status**: ⚠️ **CRITICAL ISSUES FOUND**
- VideoFlow: Complete direction reversal (coordinate system bug)
- MemFlow: Temporal offset issues but better overall accuracy

**Quick Start**:
```bash
# Run basic test
cd tests/velocity_check
python velocity_test.py --speed slow --frames 10 --fps 30

# Or use batch files
run_slow.bat      # Test with slow motion
run_medium.bat    # Test with medium motion
run_fast.bat      # Test with fast motion
run_all.bat       # Run all tests
```

**View Results**:
```bash
python show_results.py
# Or use: show_results.bat
```

See `velocity_check/FINDINGS.md` for detailed analysis and recommendations.

## Test Results Summary

| Model | Mean Error | Direction Error | Accuracy | Status |
|-------|------------|----------------|----------|--------|
| VideoFlow | 27.94 px/frame | 169.7° | 0% | ❌ Critical Issues |
| MemFlow | 11.72 px/frame | 85.8° | 0% | ⚠️ Needs Debugging |

## Recommendations

1. **Priority 1**: Fix VideoFlow coordinate system (complete direction reversal)
2. **Priority 2**: Debug MemFlow temporal sequence handling
3. **Priority 3**: Implement automated regression testing
4. **Priority 4**: Extend test coverage to more motion patterns

## Adding New Tests

To add a new test suite:

1. Create a new directory under `tests/`
2. Add `__init__.py` and main test script
3. Create batch files for easy execution
4. Update this README with test description
5. Add findings to the results summary

## Test Environment

- **GPU**: NVIDIA GeForce RTX 4060 Ti
- **CUDA**: Available and used
- **Python**: 3.10+
- **Dependencies**: See `requirements.txt` in project root

## Running All Tests

```bash
# From project root
python -m pytest tests/  # If using pytest
# Or run individual test suites manually
```

## Troubleshooting

### Common Issues

1. **Unicode Errors**: Windows console may have issues with special characters
2. **CUDA Memory**: Ensure sufficient GPU memory for model loading
3. **Path Issues**: Run tests from correct directory or use absolute paths
4. **Model Loading**: Ensure model weights are available in correct locations

### Debug Mode

Use debug mode for faster iteration:
```bash
python velocity_test.py --speed slow --frames 5 --fps 30
```

## Contributing

When adding tests:
- Follow existing naming conventions
- Include both programmatic and batch file interfaces
- Generate detailed reports with findings
- Update documentation with results
- Consider synthetic data for precise validation 