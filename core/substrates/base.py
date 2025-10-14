"""
Abstract base class for computational substrates.

This module defines the common interface that all substrate implementations
must follow for Experiment 2A: Comparative Substrate Analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class ComputationalSubstrate(ABC):
    """
    Abstract base class for computational substrates.
    
    All substrate implementations must inherit from this class and implement
    the required methods to ensure consistent behavior across different
    computational paradigms.
    """
    
    def __init__(self, size: Tuple[int, int], **kwargs):
        """
        Initialize the computational substrate.
        
        Args:
            size: Spatial dimensions (height, width) of the substrate
            **kwargs: Additional substrate-specific parameters
        """
        self.size = size
        self.height, self.width = size
        self.state = None
        self.time_step = 0
        self.parameters = kwargs
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the substrate state.
        
        This method should set up the initial conditions for the substrate,
        including any internal state variables and parameters.
        """
        pass
    
    @abstractmethod
    def inject_pattern(self, pattern: np.ndarray, location: Optional[Tuple[int, int]] = None) -> None:
        """
        Inject a pattern into the substrate.
        
        Args:
            pattern: 2D pattern array to inject
            location: Optional (x, y) location for targeted injection
        """
        pass
    
    @abstractmethod
    def evolve(self, steps: int = 1) -> None:
        """
        Evolve the substrate dynamics for a given number of steps.
        
        Args:
            steps: Number of evolution steps to perform
        """
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the substrate.
        
        Returns:
            Current substrate state as 2D array
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the substrate to initial conditions.
        """
        pass
    
    @abstractmethod
    def get_dynamics(self) -> Dict[str, Any]:
        """
        Get substrate dynamics information.
        
        Returns:
            Dictionary containing dynamics metrics
        """
        pass
    
    def store_pattern(self, pattern: np.ndarray, label: str = "") -> None:
        """
        Store a pattern in the substrate's memory.
        
        Default implementation calls inject_pattern, but substrates
        can override this for specialized storage mechanisms.
        
        Args:
            pattern: Pattern to store
            label: Optional label for the pattern
        """
        self.inject_pattern(pattern)
    
    def recall_pattern(self, cue: np.ndarray) -> np.ndarray:
        """
        Recall a pattern from the substrate using a cue.
        
        Default implementation returns the current state, but substrates
        can override this for specialized recall mechanisms.
        
        Args:
            cue: Cue pattern to trigger recall
            
        Returns:
            Recalled pattern
        """
        self.inject_pattern(cue)
        self.evolve(steps=10)  # Allow some evolution for recall
        return self.get_state()
    
    def apply_scar(self, scar_pattern: np.ndarray, strength: float = 1.0) -> None:
        """
        Apply a scar to the substrate.
        
        Default implementation modifies parameters, but substrates
        can override this for specialized scarring mechanisms.
        
        Args:
            scar_pattern: Pattern representing the scar
            strength: Strength of the scar (0.0 to 1.0)
        """
        # Default implementation - substrates should override
        pass
    
    def measure_property(self, property_name: str) -> float:
        """
        Measure a specific property of the substrate.
        
        Args:
            property_name: Name of the property to measure
            
        Returns:
            Measured property value
        """
        # Default implementation - substrates can override
        return 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the substrate.
        
        Returns:
            Dictionary containing substrate information
        """
        return {
            'type': self.__class__.__name__,
            'size': self.size,
            'time_step': self.time_step,
            'parameters': self.parameters
        }