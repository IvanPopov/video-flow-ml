# Optical Flow Testing Framework

This directory contains a comprehensive testing framework for optical flow models, specifically designed for VideoFlow and MemFlow evaluation using synthetic ground truth data.

## Overview

The framework uses synthetic videos with moving objects and mathematically precise ground truth to identify accuracy issues, optimize configurations, and validate optical flow implementations.

## Test Suites

### 🎯 [`velocity_check/`](velocity_check/) - Model Accuracy Validation
**Purpose**: Validate optical flow model accuracy across different motion speeds

**Features**:
- Tests VideoFlow (Sintel/Things) and MemFlow models
- Evaluates slow, medium, and fast motion scenarios  
- Identifies direction errors, magnitude issues, and temporal problems
- Provides accuracy metrics (velocity error, direction error, pixel-level accuracy)

**Quick Start**:
```bash
cd velocity_check
run_fast.bat        # Test fast motion
run_all.bat         # Test all motion speeds
```

**Key Results**: Successfully identified and fixed critical MemFlow direction bugs, improving accuracy by 12-18x.

### 🔧 [`memflow_seq_length/`](memflow_seq_length/) - Sequence Length Optimization  
**Purpose**: Determine optimal sequence length for MemFlow processing

**Features**:
- Tests sequence lengths from 2-12 frames
- Evaluates accuracy vs processing speed trade-offs
- Provides recommendations for different use cases
- Measures memory efficiency and temporal coherence

**Quick Start**:
```bash
cd memflow_seq_length  
run_quick_test.bat     # Test seq lengths 2-6 (quick)
run_full_test.bat      # Test seq lengths 2-10 (full)
```

**Expected Results**: Identifies optimal sequence lengths (typically 4-7 frames) for different motion types.

### 🛠️ [`common/`](common/) - Shared Components
**Purpose**: Reusable components for all test suites

**Components**:
- `SyntheticVideoGenerator` - Creates test videos with known ground truth
- `OpticalFlowAnalyzer` - Analyzes flow accuracy against ground truth
- `BaseTestRunner` - Base class for test execution

## Architecture

```
tests/
├── common/                    # Shared components
│   ├── synthetic_video.py    # Video generation
│   ├── flow_analyzer.py      # Accuracy analysis  
│   └── test_runner.py        # Base test runner
├── velocity_check/           # Model validation tests
└── memflow_seq_length/       # Sequence optimization tests
```

## Common Features

### Synthetic Video Generation
- **Resolution**: 504x216 pixels (21:9 aspect ratio, optimized for speed)
- **Motion**: Sinusoidal ball movement with precise mathematical ground truth
- **Speeds**: Slow (2px/frame), Medium (5px/frame), Fast (10px/frame)
- **Customizable**: Frame count, FPS, motion parameters

### Accuracy Analysis
- **Velocity Error**: Euclidean distance between predicted and ground truth
- **Direction Error**: Angular difference in degrees
- **Pixel Accuracy**: Percentage within 1px, 2px, 5px thresholds
- **Statistical Analysis**: Mean, std deviation, frame-by-frame analysis

### Model Integration
- **VideoFlow**: Sintel and Things models with dataset-specific configurations
- **MemFlow**: Optimized settings with memory management and warm start
- **Automatic Cache Management**: Proper isolation between different test runs

## Usage Patterns

### For Model Validation
1. Run `velocity_check` tests first to verify basic functionality
2. Check for direction errors, temporal issues, accuracy problems
3. Use results to debug and fix model implementations

### For Optimization  
1. Use `memflow_seq_length` tests to find optimal sequence lengths
2. Consider accuracy vs speed trade-offs for your use case
3. Apply findings to production configurations

### For Development
1. Create new test suites by extending `BaseTestRunner`
2. Reuse `SyntheticVideoGenerator` and `OpticalFlowAnalyzer`
3. Follow established patterns for consistency

## Results Interpretation

### Good Performance Indicators
- **Low velocity error**: <2px for accurate motion tracking
- **Low direction error**: <10° for correct motion direction
- **High pixel accuracy**: >80% within 2px threshold
- **Consistent performance**: Similar results across different speeds

### Common Issues
- **Direction reversal**: 180° error indicates coordinate system problems
- **Temporal offset**: Early frames worse than later frames
- **Memory saturation**: Accuracy degradation with longer sequences
- **Model warmup**: First few frames showing poor performance

## Best Practices

### Test Design
- Start with quick tests (few frames, limited sequence lengths)
- Use medium motion for initial validation (most representative)
- Test edge cases (very slow/fast motion) separately
- Validate fixes with multiple motion types

### Performance Optimization
- Use 21:9 aspect ratio videos for 65% faster processing
- Enable MemFlow warm start for temporal consistency
- Configure appropriate sequence lengths based on use case
- Monitor memory usage for long sequences

### Result Analysis
- Focus on consistent trends across multiple tests
- Look for systematic errors (direction, magnitude, temporal)
- Compare relative performance between models
- Use synthetic tests to validate real-world improvements

## Integration with Production

### Configuration Recommendations
Based on test results:
- **MemFlow sequence length**: 5-6 frames for optimal accuracy
- **VideoFlow model selection**: Sintel for fast motion, Things for general use
- **Memory management**: Enable warm start and proper sequence handling
- **Error tolerance**: <2px for high-accuracy applications, <5px for real-time

### Validation Pipeline
1. **Synthetic Tests**: Use this framework for initial validation
2. **Real-world Tests**: Validate on actual video data
3. **Performance Tests**: Measure processing speed and memory usage
4. **Regression Tests**: Ensure changes don't break existing functionality

## Extending the Framework

### Adding New Test Suites
1. Create new directory under `tests/`
2. Inherit from `BaseTestRunner` 
3. Reuse `SyntheticVideoGenerator` and `OpticalFlowAnalyzer`
4. Follow established naming and structure patterns

### Custom Analyses
- Extend `OpticalFlowAnalyzer` for domain-specific metrics
- Modify `SyntheticVideoGenerator` for different motion patterns
- Add new batch files for common use cases
- Create specialized documentation

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure Python path includes project root
2. **Cache conflicts**: Use `--force-recompute` flag
3. **Memory issues**: Reduce frame count or sequence length  
4. **Model loading**: Check model file paths and permissions

### Performance Tips
- Use SSD storage for faster video generation and processing
- Monitor GPU memory usage during tests
- Parallelize tests across different motion types
- Cache intermediate results when appropriate

## Contributing

When adding new tests or features:
1. Maintain compatibility with existing test suites
2. Update this README with new capabilities
3. Add appropriate documentation to new test directories
4. Ensure tests are reproducible and well-documented
5. Follow established patterns for consistency

## Dependencies

- **Core**: NumPy, OpenCV, PyTorch (for model integration)
- **Analysis**: Mathematical libraries for ground truth calculations
- **Video**: CV2 for video generation and processing  
- **Models**: VideoFlow and MemFlow frameworks

## License

Same as parent project - see project root for details. 