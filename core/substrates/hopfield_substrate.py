"""
Hopfield Network substrate for Experiment 2A.

This module implements a Hopfield network as a computational substrate
for pattern storage and recall.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from .base import ComputationalSubstrate


class HopfieldSubstrate(ComputationalSubstrate):
    """
    Hopfield Network substrate.
    
    Implements a continuous Hopfield network for pattern storage and recall
    with energy-based dynamics.
    """
    
    def __init__(self, size: Tuple[int, int], **kwargs):
        """
        Initialize Hopfield substrate.
        
        Args:
            size: Spatial dimensions (height, width) of the substrate
            **kwargs: Additional parameters including:
                - num_neurons: Total number of neurons (default: height*width)
                - temperature: Temperature parameter for dynamics (default: 0.5)
                - dt: Time step size (default: 0.1)
                - device: torch.device (default: auto-detect)
        """
        super().__init__(size, **kwargs)
        
        # Extract parameters with defaults
        self.num_neurons = kwargs.get('num_neurons', self.height * self.width)
        self.temperature = kwargs.get('temperature', 0.5)
        self.dt = kwargs.get('dt', 0.1)
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize network state and weights
        self.state = torch.zeros(self.num_neurons, device=self.device)
        self.weights = torch.zeros(self.num_neurons, self.num_neurons, device=self.device)
        self.stored_patterns = []
        
        # Create coordinate mapping for 2D visualization
        self.y_coords = torch.arange(self.height, device=self.device).repeat_interleave(self.width)
        self.x_coords = torch.arange(self.width, device=self.device).repeat(self.height)
        
        # Initialize to random state
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize the substrate state."""
        # Initialize to small random values
        self.state = torch.randn(self.num_neurons, device=self.device) * 0.1
        self.weights.zero_()
        self.stored_patterns = []
        self.time_step = 0
    
    def inject_pattern(self, pattern: np.ndarray, location: Optional[Tuple[int, int]] = None) -> None:
        """
        Inject a pattern into the substrate.
        
        Args:
            pattern: 2D pattern array to inject
            location: Optional (x, y) location for targeted injection
        """
        # Convert numpy array to torch tensor
        if isinstance(pattern, np.ndarray):
            pattern_tensor = torch.from_numpy(pattern).float().to(self.device)
        else:
            pattern_tensor = pattern
            
        # Flatten to 1D
        pattern_flat = pattern_tensor.flatten()
        if len(pattern_flat) != self.num_neurons:
            # Resize if needed
            pattern_flat = F.interpolate(
                pattern_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).flatten()
        
        if location is not None:
            # Targeted injection at specific location
            x, y = location
            # Create a Gaussian mask around the location
            radius = min(20, min(self.width, self.height) // 4)
            
            # Calculate distances from location for each neuron
            dist_x = self.x_coords - x
            dist_y = self.y_coords - y
            dist = torch.sqrt(dist_x**2 + dist_y**2)
            mask = torch.exp(-(dist**2) / (2 * radius**2))
            
            # Apply pattern with mask
            self.state = self.state * (1 - mask) + pattern_flat * mask
        else:
            # Global injection
            self.state = pattern_flat
        
        # Clamp to reasonable range
        self.state.clamp_(-1, 1)
    
    def evolve(self, steps: int = 1) -> None:
        """
        Evolve the substrate dynamics for a given number of steps.
        
        Args:
            steps: Number of evolution steps to perform
        """
        for _ in range(steps):
            # Hopfield dynamics: dx/dt = -x + tanh(Wx / T)
            activation = torch.matmul(self.weights, self.state) / self.temperature
            self.state = self.state + self.dt * (-self.state + torch.tanh(activation))
            self.time_step += 1
        
        # Clamp to reasonable range
        self.state.clamp_(-1, 1)
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the substrate.
        
        Returns:
            Current substrate state as 2D numpy array
        """
        # Reshape to 2D
        state_2d = self.state.reshape(self.height, self.width)
        return state_2d.cpu().numpy()
    
    def reset(self) -> None:
        """Reset the substrate to initial conditions."""
        self.initialize()
    
    def get_dynamics(self) -> Dict[str, Any]:
        """
        Get substrate dynamics information.
        
        Returns:
            Dictionary containing dynamics metrics
        """
        # Calculate energy
        energy = -0.5 * torch.matmul(self.state, torch.matmul(self.weights, self.state))
        
        # Calculate activity measures
        mean_activity = torch.mean(self.state).item()
        std_activity = torch.std(self.state).item()
        
        # Check for convergence (small state changes)
        if hasattr(self, 'prev_state'):
            state_change = torch.norm(self.state - self.prev_state).item()
        else:
            state_change = float('inf')
        
        self.prev_state = self.state.clone()
        
        return {
            'time_step': self.time_step,
            'energy': energy.item(),
            'mean_activity': mean_activity,
            'std_activity': std_activity,
            'state_change': state_change,
            'num_stored_patterns': len(self.stored_patterns),
            'converged': state_change < 0.001
        }
    
    def store_pattern(self, pattern: np.ndarray, label: str = "") -> None:
        """
        Store a pattern using Hebbian learning.
        
        Args:
            pattern: Pattern to store
            label: Optional label for the pattern
        """
        # Convert and flatten pattern
        if isinstance(pattern, np.ndarray):
            pattern_tensor = torch.from_numpy(pattern).float().to(self.device)
        else:
            pattern_tensor = pattern
            
        pattern_flat = pattern_tensor.flatten()
        if len(pattern_flat) != self.num_neurons:
            pattern_flat = F.interpolate(
                pattern_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).flatten()
        
        # Normalize pattern
        pattern_flat = pattern_flat / torch.norm(pattern_flat)
        
        # Hebbian learning: W = W + (1/N) * p * p^T
        # Remove self-connections
        outer_product = torch.outer(pattern_flat, pattern_flat)
        self.weights += outer_product / self.num_neurons
        
        # Zero out diagonal (no self-connections)
        self.weights.fill_diagonal_(0)
        
        # Store pattern for reference
        self.stored_patterns.append({
            'pattern': pattern_flat.cpu().numpy(),
            'label': label
        })
    
    def recall_pattern(self, cue: np.ndarray) -> np.ndarray:
        """
        Recall a pattern using a cue.
        
        Args:
            cue: Cue pattern to trigger recall
            
        Returns:
            Recalled pattern
        """
        # Inject cue and evolve to convergence
        self.inject_pattern(cue)
        
        # Evolve until convergence or max steps
        max_steps = 100
        for _ in range(max_steps):
            prev_state = self.state.clone()
            self.evolve(steps=1)
            
            # Check for convergence
            if torch.norm(self.state - prev_state) < 0.001:
                break
        
        return self.get_state()
    
    def apply_scar(self, scar_pattern: np.ndarray, strength: float = 1.0) -> None:
        """
        Apply a scar to the substrate by modifying weights.
        
        Args:
            scar_pattern: Pattern representing the scar
            strength: Strength of the scar (0.0 to 1.0)
        """
        # Convert and flatten pattern
        if isinstance(scar_pattern, np.ndarray):
            scar_tensor = torch.from_numpy(scar_pattern).float().to(self.device)
        else:
            scar_tensor = scar_pattern
            
        scar_flat = scar_tensor.flatten()
        if len(scar_flat) != self.num_neurons:
            scar_flat = F.interpolate(
                scar_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).flatten()
        
        # Apply scar by modifying weights based on pattern
        # Create asymmetric weight modification to represent scar
        scar_strength = 0.1 * strength
        weight_modification = torch.outer(scar_flat, scar_flat) * scar_strength
        
        # Apply modification with some asymmetry
        self.weights += weight_modification
        self.weights -= weight_modification.T * 0.5  # Add asymmetry
        
        # Zero out diagonal
        self.weights.fill_diagonal_(0)
    
    def measure_property(self, property_name: str) -> float:
        """
        Measure a specific property of the substrate.
        
        Args:
            property_name: Name of the property to measure
            
        Returns:
            Measured property value
        """
        if property_name == 'capacity':
            # Estimate capacity as fraction of stored patterns to total neurons
            return len(self.stored_patterns) / self.num_neurons
        elif property_name == 'overlap':
            # Measure pattern overlap (average correlation with stored patterns)
            if len(self.stored_patterns) == 0:
                return 0.0
            
            overlaps = []
            for stored in self.stored_patterns:
                stored_pattern = torch.from_numpy(stored['pattern']).float().to(self.device)
                correlation = torch.corrcoef(torch.stack([self.state, stored_pattern]))[0, 1]
                overlaps.append(correlation.item())
            
            return np.mean(overlaps)
        elif property_name == 'correlation_length':
            # Estimate correlation length from spatial structure
            state_2d = self.state.reshape(self.height, self.width)
            # Simple approximation: use spatial coherence
            return torch.std(state_2d).item()
        elif property_name == 'dynamics':
            # Return energy as dynamics measure
            energy = -0.5 * torch.matmul(self.state, torch.matmul(self.weights, self.state))
            return energy.item()
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
            'substrate_type': 'Hopfield Network',
            'num_neurons': self.num_neurons,
            'temperature': self.temperature,
            'dt': self.dt,
            'device': str(self.device),
            'num_stored_patterns': len(self.stored_patterns)
        })
        return info