"""
Reaction-Diffusion substrate wrapper for Experiment 2A.

This module provides a wrapper around the existing Gray-Scott substrate
to make it compatible with the ComputationalSubstrate interface.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from .base import ComputationalSubstrate
from ..substrate import GrayScottSubstrate


class RDSubstrate(ComputationalSubstrate):
    """
    Reaction-Diffusion substrate wrapper.
    
    This class wraps the existing GrayScottSubstrate to provide
    a consistent interface for comparative analysis.
    """
    
    def __init__(self, size: Tuple[int, int], **kwargs):
        """
        Initialize RD substrate.
        
        Args:
            size: Spatial dimensions (height, width) of the substrate
            **kwargs: Additional parameters including:
                - default_f: Default feed rate (default: 0.037)
                - default_k: Default kill rate (default: 0.060)
                - Du: Diffusion rate for U (default: 0.16)
                - Dv: Diffusion rate for V (default: 0.08)
                - dt: Time step size (default: 1.0)
                - decay_rate: V decay per timestep (default: 0.9995)
                - device: torch.device (default: auto-detect)
        """
        super().__init__(size, **kwargs)
        
        # Extract parameters with defaults
        self.default_f = kwargs.get('default_f', 0.037)
        self.default_k = kwargs.get('default_k', 0.060)
        self.Du = kwargs.get('Du', 0.16)
        self.Dv = kwargs.get('Dv', 0.08)
        self.dt = kwargs.get('dt', 1.0)
        self.decay_rate = kwargs.get('decay_rate', 0.9995)
        self.device = kwargs.get('device', None)
        
        # Initialize the underlying Gray-Scott substrate
        self.rd_substrate = GrayScottSubstrate(
            width=self.width,
            height=self.height,
            default_f=self.default_f,
            default_k=self.default_k,
            Du=self.Du,
            Dv=self.Dv,
            dt=self.dt,
            decay_rate=self.decay_rate,
            device=self.device
        )
        
        # Initialize state
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize the substrate state."""
        self.rd_substrate.reset_state()
        self.rd_substrate.reset_parameters()
        self.time_step = 0
        self.state = self.rd_substrate.get_state()
    
    def inject_pattern(self, pattern: np.ndarray, location: Optional[Tuple[int, int]] = None) -> None:
        """
        Inject a pattern into the substrate.
        
        Args:
            pattern: 2D pattern array to inject
            location: Optional (x, y) location for targeted injection
        """
        # Convert numpy array to torch tensor
        if isinstance(pattern, np.ndarray):
            pattern_tensor = torch.from_numpy(pattern).float().to(self.rd_substrate.device)
        else:
            pattern_tensor = pattern
            
        # Ensure pattern has correct shape
        if pattern_tensor.shape != (self.height, self.width):
            pattern_tensor = torch.nn.functional.interpolate(
                pattern_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        if location is not None:
            # Targeted injection at specific location
            x, y = location
            # Create a circular mask around the location
            radius = min(20, min(self.width, self.height) // 4)
            y_grid, x_grid = torch.meshgrid(
                torch.arange(self.height, device=self.rd_substrate.device),
                torch.arange(self.width, device=self.rd_substrate.device),
                indexing="ij"
            )
            dist = torch.sqrt((x_grid - x)**2 + (y_grid - y)**2)
            mask = (dist < radius).float()
            
            # Apply pattern with mask
            self.rd_substrate.V += pattern_tensor * mask * 0.5
        else:
            # Global injection
            self.rd_substrate.V += pattern_tensor * 0.5
        
        # Clamp to valid range
        self.rd_substrate.V.clamp_(0, 1)
        self.state = self.rd_substrate.get_state()
    
    def evolve(self, steps: int = 1) -> None:
        """
        Evolve the substrate dynamics for a given number of steps.
        
        Args:
            steps: Number of evolution steps to perform
        """
        for _ in range(steps):
            self.rd_substrate.simulate_step()
            self.time_step += 1
        
        self.state = self.rd_substrate.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the substrate.
        
        Returns:
            Current substrate state as 2D numpy array (V field)
        """
        state_dict = self.rd_substrate.get_state()
        return state_dict['V'].cpu().numpy()
    
    def reset(self) -> None:
        """Reset the substrate to initial conditions."""
        self.initialize()
    
    def get_dynamics(self) -> Dict[str, Any]:
        """
        Get substrate dynamics information.
        
        Returns:
            Dictionary containing dynamics metrics
        """
        return {
            'time_step': self.time_step,
            'stability': self.rd_substrate.is_stable(),
            'scar_divergence': self.rd_substrate.get_scar_divergence(),
            'v_std': torch.std(self.rd_substrate.V).item(),
            'v_mean': torch.mean(self.rd_substrate.V).item(),
            'u_std': torch.std(self.rd_substrate.U).item(),
            'u_mean': torch.mean(self.rd_substrate.U).item()
        }
    
    def store_pattern(self, pattern: np.ndarray, label: str = "") -> None:
        """
        Store a pattern by creating a parameter scar.
        
        Args:
            pattern: Pattern to store
            label: Optional label for the pattern
        """
        # Convert pattern to torch tensor
        if isinstance(pattern, np.ndarray):
            pattern_tensor = torch.from_numpy(pattern).float().to(self.rd_substrate.device)
        else:
            pattern_tensor = pattern
            
        # Ensure correct shape
        if pattern_tensor.shape != (self.height, self.width):
            pattern_tensor = torch.nn.functional.interpolate(
                pattern_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Create parameter scar by modifying F and K based on pattern
        # Higher pattern values increase f (feed rate) and decrease k (kill rate)
        scar_strength = 0.01  # Modulate scar strength
        
        self.rd_substrate.F += pattern_tensor * scar_strength
        self.rd_substrate.K -= pattern_tensor * scar_strength * 0.5
        
        # Clamp parameters to reasonable ranges
        self.rd_substrate.F.clamp_(0.01, 0.08)
        self.rd_substrate.K.clamp_(0.035, 0.08)
        
        self.state = self.rd_substrate.get_state()
    
    def recall_pattern(self, cue: np.ndarray) -> np.ndarray:
        """
        Recall a pattern using a cue.
        
        Args:
            cue: Cue pattern to trigger recall
            
        Returns:
            Recalled pattern
        """
        # Inject cue and evolve to allow pattern completion
        self.inject_pattern(cue)
        self.evolve(steps=50)  # Allow more evolution for pattern completion
        return self.get_state()
    
    def apply_scar(self, scar_pattern: np.ndarray, strength: float = 1.0) -> None:
        """
        Apply a scar to the substrate.
        
        Args:
            scar_pattern: Pattern representing the scar
            strength: Strength of the scar (0.0 to 1.0)
        """
        # Convert pattern to torch tensor
        if isinstance(scar_pattern, np.ndarray):
            scar_tensor = torch.from_numpy(scar_pattern).float().to(self.rd_substrate.device)
        else:
            scar_tensor = scar_pattern
            
        # Ensure correct shape
        if scar_tensor.shape != (self.height, self.width):
            scar_tensor = torch.nn.functional.interpolate(
                scar_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Apply scar by modifying parameters
        scar_strength = 0.02 * strength
        
        self.rd_substrate.F += scar_tensor * scar_strength
        self.rd_substrate.K -= scar_tensor * scar_strength * 0.5
        
        # Clamp parameters
        self.rd_substrate.F.clamp_(0.01, 0.08)
        self.rd_substrate.K.clamp_(0.035, 0.08)
        
        self.state = self.rd_substrate.get_state()
    
    def measure_property(self, property_name: str) -> float:
        """
        Measure a specific property of the substrate.
        
        Args:
            property_name: Name of the property to measure
            
        Returns:
            Measured property value
        """
        if property_name == 'capacity':
            # Measure as inverse of parameter divergence
            return 1.0 / (1.0 + self.rd_substrate.get_scar_divergence())
        elif property_name == 'overlap':
            # Measure pattern overlap (simplified)
            return torch.mean(self.rd_substrate.V).item()
        elif property_name == 'correlation_length':
            # Estimate correlation length from spatial autocorrelation
            v_field = self.rd_substrate.V.cpu().numpy()
            # Simple approximation: use standard deviation as proxy
            return np.std(v_field)
        elif property_name == 'dynamics':
            # Return dynamics measure
            return self.rd_substrate.is_stable()
        else:
            return 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the substrate.
        
        Returns:
            Dictionary containing substrate information
        """
        info = super().get_info()
        info.update({
            'substrate_type': 'Reaction-Diffusion',
            'default_f': self.default_f,
            'default_k': self.default_k,
            'Du': self.Du,
            'Dv': self.Dv,
            'dt': self.dt,
            'decay_rate': self.decay_rate,
            'device': str(self.rd_substrate.device)
        })
        return info