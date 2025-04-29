import numpy as np
from typing import Tuple, Optional, Dict, Any

class KolmogorovTurbulence:
    """Concrete model for generating Kolmogorov turbulence phase screens."""
    def __init__(self, grid_size: int, pixel_scale_m: float, r0_m: float, L0_m: float = 100.0, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.pixel_scale_m = pixel_scale_m
        self.r0_m = r0_m
        self.L0_m = L0_m
        self.rng = np.random.default_rng(seed)

    def generate_phase_screen(self) -> np.ndarray:
        N = self.grid_size
        delta = self.pixel_scale_m
        fx = np.fft.fftfreq(N, delta)
        fy = np.fft.fftfreq(N, delta)
        FX, FY = np.meshgrid(fx, fy)
        f = np.sqrt(FX**2 + FY**2)
        # Von Karman spectrum (Kolmogorov for L0 -> inf)
        PSD_phi = 0.023 * self.r0_m ** (-5/3) / (f**2 + 1/self.L0_m**2) ** (11/6)
        PSD_phi[0, 0] = 0  # Remove piston
        cn = (self.rng.normal(size=(N, N)) + 1j * self.rng.normal(size=(N, N)))
        cn *= np.sqrt(PSD_phi / 2)
        phase_screen = np.fft.ifft2(cn).real * (N * delta) ** 2
        return phase_screen

    def frozen_flow(self, phase_screen: np.ndarray, vx_mps: float, vy_mps: float, dt_s: float) -> np.ndarray:
        """Shift phase screen according to wind (Taylor frozen flow)."""
        shift_x = int(np.round(vx_mps * dt_s / self.pixel_scale_m))
        shift_y = int(np.round(vy_mps * dt_s / self.pixel_scale_m))
        return np.roll(np.roll(phase_screen, shift_y, axis=0), shift_x, axis=1)

class MultiLayerTurbulence:
    """Concrete model for multi-layer atmospheric turbulence."""
    def __init__(self, layers: Dict[str, Dict[str, Any]]):
        """
        Args:
            layers: Dict of layer names to parameter dicts (grid_size, pixel_scale_m, r0_m, L0_m, vx_mps, vy_mps)
        """
        self.layers = {}
        for name, params in layers.items():
            self.layers[name] = {
                'model': KolmogorovTurbulence(
                    grid_size=params['grid_size'],
                    pixel_scale_m=params['pixel_scale_m'],
                    r0_m=params['r0_m'],
                    L0_m=params.get('L0_m', 100.0),
                    seed=params.get('seed', None)
                ),
                'vx_mps': params.get('vx_mps', 0.0),
                'vy_mps': params.get('vy_mps', 0.0)
            }

    def generate_combined_phase_screen(self, dt_s: float = 0.0, prev_screens: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        combined = None
        screens = {}
        for name, layer in self.layers.items():
            if prev_screens and name in prev_screens:
                screen = layer['model'].frozen_flow(prev_screens[name], layer['vx_mps'], layer['vy_mps'], dt_s)
            else:
                screen = layer['model'].generate_phase_screen()
            screens[name] = screen
            if combined is None:
                combined = np.zeros_like(screen)
            combined += screen
        return combined, screens

class TurbulenceCompensator:
    """Concrete compensation technique for atmospheric turbulence (phase conjugation)."""
    def __init__(self, n_actuators=32, influence_function_sigma=0.5, control_gain=0.6):
        """
        Initialize turbulence compensator with deformable mirror parameters.
        
        Args:
            n_actuators: Number of actuators across the deformable mirror
            influence_function_sigma: Width of actuator influence function (Gaussian sigma)
            control_gain: Gain factor for control loop (0-1)
        """
        self.n_actuators = n_actuators
        self.influence_function_sigma = influence_function_sigma
        self.control_gain = control_gain
        self.actuator_positions = None
        self.influence_functions = None
        self._initialize_dm()
        
    def _initialize_dm(self):
        """Initialize deformable mirror actuator positions and influence functions."""
        # Create grid of actuator positions
        x = np.linspace(-1, 1, self.n_actuators)
        y = np.linspace(-1, 1, self.n_actuators)
        self.actuator_positions = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        
        # Pre-compute influence functions (will be applied to higher-resolution phase maps)
        self.influence_functions = []
        
    def phase_conjugation(self, phase_map: np.ndarray) -> np.ndarray:
        """Return conjugate phase map for compensation."""
        return -phase_map
    
    def modal_decomposition(self, phase_map: np.ndarray, n_modes: int = 20) -> np.ndarray:
        """
        Decompose phase map into Zernike modes and reconstruct with limited modes.
        
        Args:
            phase_map: Input phase map to decompose
            n_modes: Number of Zernike modes to use in reconstruction
            
        Returns:
            Reconstructed phase map using limited modes
        """
        # Simplified implementation - in practice would use Zernike polynomials
        # This is a placeholder for the actual implementation
        h, w = phase_map.shape
        y, x = np.mgrid[-1:1:h*1j, -1:1:w*1j]
        r = np.sqrt(x**2 + y**2)
        mask = r <= 1.0
        
        # For demonstration, just apply a low-pass filter as approximation
        from scipy import ndimage
        filtered = ndimage.gaussian_filter(phase_map, sigma=max(h,w)/(n_modes*np.pi))
        return filtered * mask

    def apply_dm(self, phase_map: np.ndarray, dm_commands: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply deformable mirror correction to input phase map.
        
        Args:
            phase_map: Input phase map to correct
            dm_commands: Optional pre-computed DM commands, if None will compute from phase_map
            
        Returns:
            Corrected phase map after DM application
        """
        if dm_commands is None:
            dm_commands = self.compute_dm_commands(phase_map)
            
        # Apply influence functions to create DM shape
        dm_shape = self._commands_to_shape(dm_commands, phase_map.shape)
        
        # Apply correction (subtract DM shape from input phase)
        corrected_phase = phase_map - dm_shape
        return corrected_phase
    
    def compute_dm_commands(self, phase_map: np.ndarray) -> np.ndarray:
        """
        Compute optimal DM commands to correct input phase map.
        
        Args:
            phase_map: Phase map to correct
            
        Returns:
            Array of DM actuator commands
        """
        # For simplicity, use zonal control approach
        # Sample phase at actuator locations
        h, w = phase_map.shape
        y, x = np.mgrid[-1:1:h*1j, -1:1:w*1j]
        
        commands = []
        for pos in self.actuator_positions:
            # Find nearest point in phase map
            i = int((pos[1] + 1) * (h-1) / 2)
            j = int((pos[0] + 1) * (w-1) / 2)
            i = max(0, min(h-1, i))
            j = max(0, min(w-1, j))
            commands.append(-phase_map[i, j] * self.control_gain)
            
        return np.array(commands)
    
    def _commands_to_shape(self, commands: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert DM commands to a phase shape using influence functions.
        
        Args:
            commands: Array of actuator commands
            output_shape: Shape of output phase map
            
        Returns:
            Phase map representing DM shape
        """
        h, w = output_shape
        y, x = np.mgrid[-1:1:h*1j, -1:1:w*1j]
        points = np.vstack((x.flatten(), y.flatten())).T
        
        # Initialize output shape
        dm_shape = np.zeros(output_shape)
        
        # Apply each actuator influence function
        for i, (pos, cmd) in enumerate(zip(self.actuator_positions, commands)):
            # Gaussian influence function
            dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            influence = np.exp(-0.5 * (dist / self.influence_function_sigma)**2)
            dm_shape += cmd * influence
            
        return dm_shape

    def strehl_ratio(self, residual_phase: np.ndarray) -> float:
        """Compute Strehl ratio from residual phase error."""
        # Mask to pupil area
        h, w = residual_phase.shape
        y, x = np.mgrid[-1:1:h*1j, -1:1:w*1j]
        r = np.sqrt(x**2 + y**2)
        mask = r <= 1.0
        
        # Compute RMS within pupil
        masked_phase = residual_phase * mask
        valid_points = np.sum(mask)
        if valid_points > 0:
            rms = np.sqrt(np.sum((masked_phase)**2) / valid_points)
        else:
            rms = np.inf
            
        # Maréchal approximation for Strehl ratio
        return float(np.exp(-(2*np.pi*rms)**2))
