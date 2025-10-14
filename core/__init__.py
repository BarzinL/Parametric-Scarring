"""
Core module for computational substrate experiments.

This module provides the core functionality for implementing and analyzing
different computational substrates for pattern storage and discrimination.
"""

# Import key classes and functions for easy access
from .substrate import GrayScottSubstrate
from .patterns import (
    create_phoneme_pattern,
    create_geometric_pattern,
    generate_random_pattern,
    synthesize_phoneme_audio
)
from .metrics import pattern_similarity
from .scarring import apply_scarring
from .visualization import save_state

__all__ = [
    'GrayScottSubstrate',
    'create_phoneme_pattern',
    'create_geometric_pattern', 
    'generate_random_pattern',
    'synthesize_phoneme_audio',
    'pattern_similarity',
    'apply_scarring',
    'save_state'
]