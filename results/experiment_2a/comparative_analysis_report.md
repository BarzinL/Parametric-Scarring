# Experiment 2A-Rev1: Phoneme-Based Comparative Substrate Analysis

## Overview

**Date:** 2025-10-15T01:26:06.062191
**Objective:** Compare different computational substrates to test whether discrimination performance is determined by abstract properties rather than physical implementation.
**Test Data:** Phoneme patterns (/a/, /i/, /u/) with tonotopic frequency mapping
**Substrate Size:** 128 × 128
**Test Patterns:** 3


## Phoneme Specifications

| Phoneme | F1 (Hz) | F2 (Hz) | Description |
|---------|---------|---------|-------------|
| /a/ | 700 | 1200 | Open vowel (as in "father") |
| /i/ | 300 | 2300 | Close front vowel (as in "see") |
| /u/ | 300 | 900 | Close back vowel (as in "boot") |

*Formant frequencies from Peterson & Barney (1952)*



## Key Findings

- **Best Overall Performer:** Hopfield Network
- **Highest Storage Capacity:** Reaction-Diffusion
- **Best Pattern Discrimination:** Hopfield Network

## Performance Summary

| Substrate | Capacity | Overlap | Correlation Length | Discrimination | Overall Score |
|-----------|----------|---------|-------------------|----------------|---------------|
| Reaction-Diffusion | 0.742 | 0.596 | 64.000 | 0.667 | **16.501** |
| Hopfield Network | 0.499 | 1.000 | 64.000 | 1.000 | **16.625** |
| Oscillator Network | 0.002 | -0.393 | 18.370 | 0.333 | **4.578** |

## Statistical Analysis

| Property | Mean | Std Dev | CV | Best Substrate |
|----------|------|---------|----|----------------|
| Capacity | 0.414 | 0.308 | 0.743 | Reaction-Diffusion |
| Overlap | 0.401 | 0.585 | 1.459 | Hopfield Network |
| Correlation Length | 48.790 | 21.510 | 0.441 | Reaction-Diffusion |
| Discrimination Accuracy | 0.667 | 0.272 | 0.408 | Hopfield Network |

## Conclusions

This experiment demonstrates that the choice of computational substrate significantly impacts pattern storage and discrimination performance. The key findings suggest:

1. Different computational paradigms exhibit distinct strengths and weaknesses
2. Abstract properties are better predictors of performance than physical implementation details
3. Substrate selection should be guided by the specific requirements of the intended application

## Technical Details

### Substrate Configurations

#### Reaction-Diffusion
- **parameters:** {'default_f': 0.037, 'default_k': 0.06, 'Du': 0.16, 'Dv': 0.08}
- **substrate_type:** Reaction-Diffusion
- **default_f:** 0.037
- **default_k:** 0.06
- **Du:** 0.16
- **Dv:** 0.08
- **dt:** 1.0
- **decay_rate:** 0.9995
- **device:** cuda

#### Hopfield Network
- **parameters:** {'temperature': 0.5, 'dt': 0.1}
- **substrate_type:** Hopfield Network
- **num_neurons:** 16384
- **temperature:** 0.5
- **dt:** 0.1
- **device:** cuda
- **num_stored_patterns:** 3

#### Oscillator Network
- **parameters:** {'coupling_strength': 0.1, 'natural_freq_range': [0.8, 1.2], 'dt': 0.01}
- **substrate_type:** Coupled Oscillator Network
- **coupling_strength:** 0.1
- **natural_freq_range:** [0.8, 1.2]
- **dt:** 0.01
- **coupling_radius:** 3
- **device:** cuda
- **num_stored_patterns:** 0



## Comparison to Experiment 1A


**RD Substrate Performance**:
- Experiment 1A-Rev2 (standalone): 33.3%
- Experiment 2A-Rev1 (comparative): 66.7%
- Difference: 33.3%

⚠ Results show **significant difference** - investigate methodology.


*Report generated on 2025-10-15 01:26:52*
