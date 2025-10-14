# Experiment 2A: Comparative Substrate Analysis

## Overview

**Date:** 2025-10-14T06:11:54.815217
**Objective:** Compare different computational substrates to test whether discrimination performance is determined by abstract properties rather than physical implementation.
**Substrate Size:** 128 × 128
**Test Patterns:** 5

## Key Findings

- **Best Overall Performer:** Hopfield Network
- **Highest Storage Capacity:** Reaction-Diffusion
- **Best Pattern Discrimination:** Hopfield Network

## Performance Summary

| Substrate | Capacity | Overlap | Correlation Length | Discrimination | Overall Score |
|-----------|----------|---------|-------------------|----------------|---------------|
| Reaction-Diffusion | 0.581 | 0.585 | 0.000 | 0.400 | **0.392** |
| Hopfield Network | 0.395 | 1.000 | 0.000 | 1.000 | **0.599** |
| Oscillator Network | 0.002 | -0.001 | 0.000 | 0.200 | **0.050** |

## Statistical Analysis

| Property | Mean | Std Dev | CV | Best Substrate |
|----------|------|---------|----|----------------|
| Capacity | 0.326 | 0.242 | 0.741 | Reaction-Diffusion |
| Overlap | 0.528 | 0.411 | 0.778 | Hopfield Network |
| Correlation Length | 0.000 | 0.000 | 0.000 | Reaction-Diffusion |
| Discrimination Accuracy | 0.533 | 0.340 | 0.637 | Hopfield Network |

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
- **num_stored_patterns:** 5

#### Oscillator Network
- **parameters:** {'coupling_strength': 0.1, 'natural_freq_range': [0.8, 1.2], 'dt': 0.01}
- **substrate_type:** Coupled Oscillator Network
- **coupling_strength:** 0.1
- **natural_freq_range:** [0.8, 1.2]
- **dt:** 0.01
- **coupling_radius:** 3
- **device:** cuda
- **num_stored_patterns:** 5


*Report generated on 2025-10-14 06:15:31*
