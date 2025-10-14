"""
Computational substrates for Experiment 2A: Comparative Substrate Analysis.

This module provides different computational substrate implementations:
- Reaction-Diffusion substrate
- Hopfield Network substrate  
- Oscillator Network substrate
"""

from .base import ComputationalSubstrate
from .rd_substrate import RDSubstrate
from .hopfield_substrate import HopfieldSubstrate
from .oscillator_substrate import OscillatorSubstrate

__all__ = [
    'ComputationalSubstrate',
    'RDSubstrate', 
    'HopfieldSubstrate',
    'OscillatorSubstrate'
]