"""
substrate.py - Abstract computational medium for learning

Current implementation: Gray-Scott reaction-diffusion system
Future: Could be replaced with other PDEs, SDEs, or novel dynamics

The "substrate" is the physical medium that:
- Self-organizes into patterns
- Can be modified (scarred) to store memory
- Operates in continuous space/time
"""

import torch
import torch.nn.functional as F


class GrayScottSubstrate:
    """
    Gray-Scott reaction-diffusion system.

    Two chemicals (U and V) diffuse and react:
        ∂U/∂t = Du∇²U - UV² + f(1-U)
        ∂V/∂t = Dv∇²V + UV² - (k+f)V

    Parameters f and k control pattern formation (spots, stripes, mazes, etc.)
    """

    def __init__(
        self,
        width=256,
        height=256,
        default_f=0.037,
        default_k=0.060,
        Du=0.16,
        Dv=0.08,
        dt=1.0,
        decay_rate=0.9995,
        device=None,
    ):
        """
        Initialize Gray-Scott substrate.

        Args:
            width: Grid width
            height: Grid height
            default_f: Default feed rate (controls U replenishment)
            default_k: Default kill rate (controls V removal)
            Du: Diffusion rate for U
            Dv: Diffusion rate for V
            dt: Time step size
            decay_rate: V decay per timestep (prevents runaway growth)
            device: torch.device (cuda/cpu), auto-detected if None
        """
        self.width = width
        self.height = height
        self.default_f = default_f
        self.default_k = default_k
        self.Du = Du
        self.Dv = Dv
        self.dt = dt
        self.decay_rate = decay_rate

        # Auto-detect device if not specified
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Initialize parameter fields (F and K are spatially varying)
        self.F = torch.full((height, width), default_f, device=device)
        self.K = torch.full((height, width), default_k, device=device)

        # Initialize chemical concentrations
        self.U = torch.ones(height, width, device=device)
        self.V = torch.zeros(height, width, device=device)

        # Create coordinate grids (useful for pattern generation)
        self.y_grid, self.x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        # Laplacian kernel for computing ∇²
        # This is a discrete approximation: weighted average of neighbors minus center
        self.laplacian_kernel = torch.tensor(
            [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]], device=device
        ).reshape(1, 1, 3, 3)

    def simulate_step(self):
        """
        Execute one timestep of Gray-Scott dynamics.

        Updates U and V in-place based on:
        - Diffusion (spreading via Laplacian)
        - Reaction (UV² interaction)
        - Feed/kill (f and k parameters)
        """
        # Pad with circular boundary conditions (toroidal topology)
        # This means the left edge wraps to right edge, top to bottom
        U_padded = F.pad(
            self.U.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular"
        )
        V_padded = F.pad(
            self.V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular"
        )

        # Compute Laplacian (∇²) via convolution
        lap_U = F.conv2d(U_padded, self.laplacian_kernel).squeeze()
        lap_V = F.conv2d(V_padded, self.laplacian_kernel).squeeze()

        # Reaction term: U and V interact quadratically (UV²)
        uvv = self.U * self.V * self.V

        # Gray-Scott equations (Euler integration)
        self.U = self.U + (self.Du * lap_U - uvv + self.F * (1 - self.U)) * self.dt
        self.V = self.V + (self.Dv * lap_V + uvv - (self.K + self.F) * self.V) * self.dt

        # Apply decay to V (prevents runaway growth)
        self.V *= self.decay_rate

        # Clamp to valid concentration range [0, 1]
        self.U.clamp_(0, 1)
        self.V.clamp_(0, 1)

    def reset_state(self, U_init=None, V_init=None):
        """
        Reset U and V to initial conditions.

        Args:
            U_init: Initial U state (default: all 1.0)
            V_init: Initial V state (default: all 0.0)
        """
        if U_init is None:
            self.U.fill_(1.0)
        else:
            self.U.copy_(U_init)

        if V_init is None:
            self.V.fill_(0.0)
        else:
            self.V.copy_(V_init)

    def reset_parameters(self):
        """Reset F and K to default values (erase all scars)."""
        self.F.fill_(self.default_f)
        self.K.fill_(self.default_k)

    def get_state(self):
        """
        Get current state as dictionary.

        Returns:
            dict with keys: 'U', 'V', 'F', 'K'
        """
        return {
            "U": self.U.clone(),
            "V": self.V.clone(),
            "F": self.F.clone(),
            "K": self.K.clone(),
        }

    def set_state(self, state_dict):
        """
        Restore state from dictionary.

        Args:
            state_dict: Dict with keys 'U', 'V', 'F', 'K'
        """
        self.U.copy_(state_dict["U"])
        self.V.copy_(state_dict["V"])
        self.F.copy_(state_dict["F"])
        self.K.copy_(state_dict["K"])

    def get_scar_divergence(self):
        """
        Measure total parameter divergence from baseline.

        Returns:
            float: Sum of absolute deviations in F and K
        """
        f_div = torch.abs(self.F - self.default_f).sum().item()
        k_div = torch.abs(self.K - self.default_k).sum().item()
        return f_div + k_div
