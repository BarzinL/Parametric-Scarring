# Component Testing for Parametric Scarring Experiment

This directory contains test scripts to verify each component before running the full experiment. These tests ensure that all components are working correctly and help identify any issues early.

## Test Components

The following components are tested:

1. **Audio Synthesis** (`test_audio_synthesis.py`)
   - Tests phoneme audio generation
   - Validates audio properties (duration, sample rate, normalization)
   - Checks formant characteristics
   - Verifies error handling

2. **Circular Mask** (`test_circular_mask.py`)
   - Tests circular region creation
   - Validates mask properties (shape, size, position)
   - Checks edge cases (different radii, positions, grid sizes)
   - Verifies device compatibility

3. **Equilibrium Reset** (`test_equilibrium_reset.py`)
   - Tests substrate reset to equilibrium
   - Validates pattern formation after reset
   - Checks stability detection
   - Verifies different parameter configurations

4. **Audio Injection (Short)** (`test_audio_injection.py`)
   - Tests short audio injection into substrate
   - Validates injection effects on substrate state
   - Checks different injection parameters
   - Verifies phoneme discrimination

## Usage

### Prerequisites

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests

#### Option 1: Run All Tests (Recommended)

Use the master test script to run all component tests:

```bash
python run_tests.py
```

Or run directly from the tests directory:

```bash
python tests/run_all_tests.py
```

This will:
- Check dependencies and environment
- Run all component tests in sequence
- Provide a summary of results
- Tell you if the system is ready for the full experiment

#### Option 2: Run Individual Tests

Run specific component tests:

```bash
# Test audio synthesis
python tests/test_audio_synthesis.py

# Test circular mask creation
python tests/test_circular_mask.py

# Test equilibrium reset
python tests/test_equilibrium_reset.py

# Test audio injection
python tests/test_audio_injection.py

# Test phoneme pipeline
python tests/test_phoneme_pipeline.py
```

### Test Output

Each test script:
- Prints detailed progress information
- Saves visualizations to `test_results/[component]/` directory
- Returns exit code 0 on success, 1 on failure
- Provides clear error messages for debugging

## Interpreting Results

### Success Indicators

- ✓ indicates a successful test
- All tests should show "✓✓✓ [COMPONENT] TEST: ALL PASSED"
- Master script should show "✓✓✓ ALL COMPONENT TESTS PASSED"

### Failure Indicators

- ✗ indicates a failed test
- Error messages will be displayed
- Check the `test_results/` directory for debug information
- Fix issues before running the full experiment

## Test Details

### Audio Synthesis Tests

1. **Basic synthesis**: Generates phonemes and checks basic properties
2. **File saving**: Verifies audio files can be saved
3. **Different durations**: Tests various audio lengths
4. **Different sample rates**: Tests various sampling rates
5. **Formant characteristics**: Checks phoneme differentiation
6. **Error handling**: Tests invalid input handling
7. **Visualization**: Creates waveform and spectrum plots

### Circular Mask Tests

1. **Basic mask creation**: Creates circular region and checks properties
2. **Different positions**: Tests various center positions
3. **Different radii**: Tests various circle sizes
4. **Edge cases**: Tests boundary conditions
5. **Different grid sizes**: Tests various substrate dimensions
6. **Device compatibility**: Tests CPU and GPU if available
7. **Performance**: Measures execution time
8. **Visualization**: Creates mask visualizations

### Equilibrium Reset Tests

1. **Basic reset**: Resets substrate and checks state
2. **Different durations**: Tests various reset lengths
3. **Reset after perturbation**: Tests recovery from disturbances
4. **Different parameters**: Tests various RD parameters
5. **Stability detection**: Tests equilibrium detection
6. **Visualization**: Creates before/after images
7. **Device compatibility**: Tests CPU and GPU if available
8. **Performance**: Measures reset time

### Audio Injection Tests

1. **Basic injection**: Injects audio and checks effects
2. **Different strengths**: Tests various injection strengths
3. **Different phonemes**: Tests various phoneme types
4. **Different regions**: Tests various injection locations
5. **Different step ratios**: Tests various temporal resolutions
6. **Visualization**: Creates injection evolution images
7. **Device compatibility**: Tests CPU and GPU if available
8. **Performance**: Measures injection time
9. **Edge cases**: Tests empty and very short audio

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
2. **CUDA errors**: Tests will fall back to CPU if GPU unavailable
3. **Timeout errors**: Increase timeout in `run_all_tests.py` if needed
4. **Permission errors**: Check write permissions for output directories

### Debug Tips

1. Check individual test outputs for specific error messages
2. Examine generated visualizations in `test_results/` directories
3. Verify all dependencies are installed correctly
4. Ensure sufficient disk space for test outputs

## Running the Full Experiment

After all tests pass:

```bash
python experiments/1a_rev1_direct_acoustic.py
```

If tests fail, fix the issues before running the full experiment to avoid wasted computation time.

## Test File Structure

```
.
├── test_audio_synthesis.py      # Audio synthesis tests
├── test_circular_mask.py        # Circular mask tests
├── test_equilibrium_reset.py    # Equilibrium reset tests
├── test_audio_injection.py      # Audio injection tests
├── run_all_tests.py             # Master test script
├── TESTING_README.md            # This file
└── test_results/                # Test outputs (created during testing)
    ├── audio_synthesis/
    ├── circular_mask/
    ├── equilibrium_reset/
    └── audio_injection/