"""
Analysis tools for Experiment 2A: Comparative Substrate Analysis.

This module provides property measurement and analysis functions
for comparing different computational substrates.
"""

from .properties import (
    measure_capacity,
    measure_overlap,
    measure_correlation_length,
    measure_dynamics,
    compute_similarity_matrix,
    analyze_discrimination_performance,
    compare_substrates,
    measure_substrate_properties
)

__all__ = [
    'measure_capacity',
    'measure_overlap',
    'measure_correlation_length',
    'measure_dynamics',
    'compute_similarity_matrix',
    'analyze_discrimination_performance',
    'compare_substrates',
    'measure_substrate_properties'
]