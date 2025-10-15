"""
Coupled Oscillator Network substrate for Experiment 2A.

This module implements a network of coupled oscillators as a computational substrate
for pattern storage and recall with synchronization-based dynamics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from .base import ComputationalSubstrate


class OscillatorSubstrate(ComputationalSubstrate):
    """
    Coupled Oscillator Network substrate.
    
    Implements a network of Kuramoto oscillators with spatial coupling
    for pattern storage and recall through synchronization dynamics.
    """
    
    def __init__(self, size: Tuple[int, int], **kwargs):
        """
        Initialize Oscillator substrate.
        
        Args:
            size: Spatial dimensions (height, width) of the substrate
            **kwargs: Additional parameters including:
                - coupling_strength: Coupling strength between oscillators (default: 0.1)
                - natural_freq_range: Range of natural frequencies (default: (0.8, 1.2))
                - dt: Time step size (default: 0.01)
                - coupling_radius: Radius of coupling neighborhood (default: 3)
                - device: torch.device (default: auto-detect)
        """
        super().__init__(size, **kwargs)
        
        # Extract parameters with defaults
        self.coupling_strength = kwargs.get('coupling_strength', 0.1)
        self.natural_freq_range = kwargs.get('natural_freq_range', (0.8, 1.2))
        self.dt = kwargs.get('dt', 0.01)
        self.coupling_radius = kwargs.get('coupling_radius', 3)
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize oscillator states
        self.phases = torch.zeros(self.height, self.width, device=self.device)
        self.natural_frequencies = torch.zeros(self.height, self.width, device=self.device)
        self.coupling_matrix = None
        
        # Stored patterns for memory
        self.stored_patterns = []
        
        # Create coupling matrix
        self._create_coupling_matrix()
        
        # Initialize to random state
        self.initialize()
    
    def _create_coupling_matrix(self) -> None:
        """Create spatial coupling matrix for oscillators."""
        # Create distance matrix
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij"
        )
        
        # Calculate pairwise distances
        dist_x = x_coords.unsqueeze(2) - x_coords.unsqueeze(0).unsqueeze(1)
        dist_y = y_coords.unsqueeze(2) - y_coords.unsqueeze(0).unsqueeze(1)
        distances = torch.sqrt(dist_x**2 + dist_y**2)
        
        # Create coupling matrix (1 if within radius, 0 otherwise)
        self.coupling_matrix = (distances <= self.coupling_radius).float()
        
        # Remove self-coupling
        self.coupling_matrix.view(-1)[::self.height * self.width + 1] = 0
        
        # Normalize by number of neighbors
        neighbor_counts = self.coupling_matrix.sum(dim=(1, 2), keepdim=True)
        neighbor_counts[neighbor_counts == 0] = 1  # Avoid division by zero
        self.coupling_matrix = self.coupling_matrix / neighbor_counts
    
    def initialize(self) -> None:
        """Initialize the substrate state."""
        # Random initial phases
        self.phases = torch.rand(self.height, self.width, device=self.device) * 2 * np.pi
        
        # Random natural frequencies within specified range
        freq_min, freq_max = self.natural_freq_range
        self.natural_frequencies = torch.rand(self.height, self.width, device=self.device) * (freq_max - freq_min) + freq_min
        
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
            
        # Ensure correct shape
        if pattern_tensor.shape != (self.height, self.width):
            pattern_tensor = F.interpolate(
                pattern_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Convert pattern to phase perturbations
        # Higher pattern values correspond to phase shifts
        phase_shift = pattern_tensor * np.pi
        
        if location is not None:
            # Targeted injection at specific location
            x, y = location
            # Create a Gaussian mask around the location
            radius = min(20, min(self.width, self.height) // 4)
            
            y_grid, x_grid = torch.meshgrid(
                torch.arange(self.height, device=self.device),
                torch.arange(self.width, device=self.device),
                indexing="ij"
            )
            dist = torch.sqrt((x_grid - x)**2 + (y_grid - y)**2)
            mask = torch.exp(-(dist**2) / (2 * radius**2))
            
            # Apply phase shift with mask
            self.phases = self.phases + phase_shift * mask
        else:
            # Global injection
            self.phases = self.phases + phase_shift
        
        # Wrap phases to [0, 2π]
        self.phases = self.phases % (2 * np.pi)
    
    def evolve(self, steps: int = 1) -> None:
        """
        Evolve the substrate dynamics for a given number of steps.
        
        Args:
            steps: Number of evolution steps to perform
        """
        for _ in range(steps):
            # Kuramoto model dynamics: dθ/dt = ω + K * Σ sin(θj - θi)
            
            # Calculate phase differences
            phase_diff = self.phases.unsqueeze(2) - self.phases.unsqueeze(0).unsqueeze(1)
            
            # Calculate coupling term
            coupling_term = self.coupling_strength * torch.sum(
                self.coupling_matrix * torch.sin(phase_diff),
                dim=(1, 2)
            )
            
            # Update phases
            self.phases = self.phases + (self.natural_frequencies + coupling_term) * self.dt
            
            # Wrap phases to [0, 2π]
            self.phases = self.phases % (2 * np.pi)
            
            self.time_step += 1
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the substrate.
        
        Returns:
            Current substrate state as 2D numpy array
            (represented as complex exponentials of phases)
        """
        # Convert phases to complex representation
        complex_state = torch.exp(1j * self.phases)
        return complex_state.real.cpu().numpy()
    
    def reset(self) -> None:
        """Reset the substrate to initial conditions."""
        self.initialize()
    
    def get_dynamics(self) -> Dict[str, Any]:
        """
        Get substrate dynamics information.
        
        Returns:
            Dictionary containing dynamics metrics
        """
        # Calculate order parameter (measure of synchronization)
        complex_phases = torch.exp(1j * self.phases)
        order_parameter = torch.abs(torch.mean(complex_phases)).item()
        
        # Calculate mean phase
        mean_phase = torch.angle(torch.mean(complex_phases)).item()
        
        # Calculate phase variance
        phase_variance = torch.var(self.phases).item()
        
        # Check for synchronization
        synchronized = order_parameter > 0.7
        
        return {
            'time_step': self.time_step,
            'order_parameter': order_parameter,
            'mean_phase': mean_phase,
            'phase_variance': phase_variance,
            'synchronized': synchronized,
            'num_stored_patterns': len(self.stored_patterns),
            'mean_frequency': torch.mean(self.natural_frequencies).item()
        }
    
    def store_pattern_spatial(self, pattern: np.ndarray, label: str = "") -> None:
        """
        Store a 2D spatial pattern by converting to frequency representation.
        
        Args:
            pattern: 2D numpy array
            label: Pattern label
        """
        # Flatten pattern
        flat = pattern.flatten()
        
        # Take FFT to get frequency components
        fft = np.fft.fft(flat)
        freqs = np.fft.fftfreq(len(flat))
        
        # Map frequency components to oscillator natural frequencies
        # Each oscillator responds to its resonant frequency
        
        # Reset oscillators
        self.reset()
        
        # Drive oscillators based on spectral content
        for i in range(self.height * self.width):
            # Find closest frequency in pattern spectrum
            target_freq = self.natural_frequencies.flatten()[i].item() / (2 * np.pi)  # Convert to Hz
            
            # Find power at this frequency
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            power = np.abs(fft[freq_idx])
            
            # Set initial oscillator amplitude based on power
            flat_i = i % (self.height * self.width)
            self.phases.flatten()[flat_i] = torch.tensor(power * 0.01, device=self.device)  # Scale down
        
        # Let oscillators couple and settle
        self.stabilize(n_steps=500)
    
    def stabilize(self, n_steps: int = 500) -> None:
        """
        Let the oscillator network stabilize.
        
        Args:
            n_steps: Number of evolution steps for stabilization
        """
        self.evolve(steps=n_steps)

    def store_pattern(self, pattern: np.ndarray, label: str = "") -> None:
        """Override to handle spatial patterns."""
        if isinstance(pattern, np.ndarray) and pattern.ndim == 2:
            self.store_pattern_spatial(pattern, label)
        else:
            super().store_pattern(pattern, label)
    
    def recall_pattern(self, cue: np.ndarray) -> np.ndarray:
        """
        Recall a pattern using a cue.
        
        Args:
            cue: Cue pattern to trigger recall
            
        Returns:
            Recalled pattern
        """
        # Inject cue and evolve to synchronization
        self.inject_pattern(cue)
        
        # Evolve until synchronization or max steps
        max_steps = 1000
        for _ in range(max_steps):
            self.evolve(steps=10)
            
            # Check for synchronization
            dynamics = self.get_dynamics()
            if dynamics['synchronized']:
                break
        
        return self.get_state()
    
    def apply_scar(self, scar_pattern: np.ndarray, strength: float = 1.0) -> None:
        """
        Apply a scar to the substrate by modifying coupling strengths.
        
        Args:
            scar_pattern: Pattern representing the scar
            strength: Strength of the scar (0.0 to 1.0)
        """
        # Convert pattern to torch tensor
        if isinstance(scar_pattern, np.ndarray):
            scar_tensor = torch.from_numpy(scar_pattern).float().to(self.device)
        else:
            scar_tensor = scar_pattern
            
        # Ensure correct shape
        if scar_tensor.shape != (self.height, self.width):
            scar_tensor = F.interpolate(
                scar_tensor.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Apply scar by modifying coupling matrix locally
        scar_strength = 0.1 * strength
        
        # Create spatial modulation of coupling
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij"
        )
        
        # Modify coupling based on scar pattern
        for i in range(self.height):
            for j in range(self.width):
                # Calculate distance from current point
                dist = torch.sqrt((x_grid - j)**2 + (y_grid - i)**2)
                
                # Create Gaussian influence based on scar pattern
                influence = scar_tensor[i, j] * torch.exp(-(dist**2) / (2 * self.coupling_radius**2))
                
                # Modify coupling matrix
                self.coupling_matrix[i, j] += influence * scar_strength
        
        # Ensure coupling matrix remains valid
        self.coupling_matrix.clamp_(0, 1)
        
        # Remove self-coupling
        self.coupling_matrix.view(-1)[::self.height * self.width + 1] = 0
        
        # Renormalize
        neighbor_counts = self.coupling_matrix.sum(dim=(1, 2), keepdim=True)
        neighbor_counts[neighbor_counts == 0] = 1
        self.coupling_matrix = self.coupling_matrix / neighbor_counts
    
    def measure_property(self, property_name: str) -> float:
        """
        Measure a specific property of the substrate.
        
        Args:
            property_name: Name of the property to measure
            
        Returns:
            Measured property value
        """
        if property_name == 'capacity':
            # Estimate capacity based on frequency diversity
            return torch.std(self.natural_frequencies).item()
        elif property_name == 'overlap':
            # Measure synchronization overlap
            complex_phases = torch.exp(1j * self.phases)
            return torch.abs(torch.mean(complex_phases)).item()
        elif property_name == 'correlation_length':
            # Estimate correlation length from phase coherence
            # Calculate spatial correlation of phases
            sin_phases = torch.sin(self.phases)
            cos_phases = torch.cos(self.phases)
            
            # Simple approximation: use phase coherence
            return torch.std(sin_phases).item() + torch.std(cos_phases).item()
        elif property_name == 'dynamics':
            # Return order parameter as dynamics measure
            complex_phases = torch.exp(1j * self.phases)
            return torch.abs(torch.mean(complex_phases)).item()
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
            'substrate_type': 'Coupled Oscillator Network',
            'coupling_strength': self.coupling_strength,
            'natural_freq_range': self.natural_freq_range,
            'dt': self.dt,
            'coupling_radius': self.coupling_radius,
            'device': str(self.device),
            'num_stored_patterns': len(self.stored_patterns)
        })
        return info