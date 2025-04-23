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
    def __init__(self):
        pass

    def phase_conjugation(self, phase_map: np.ndarray) -> np.ndarray:
        """Return conjugate phase map for compensation."""
        return -phase_map

    def apply_dm(self, phase_map: np.ndarray, dm: Any) -> np.ndarray:
        """Compute DM commands to compensate input phase map."""
        return dm.compute_commands(phase_map)

    def strehl_ratio(self, residual_phase: np.ndarray) -> float:
        """Compute Strehl ratio from residual phase error."""
        rms = np.std(residual_phase)
        return float(np.exp(-rms ** 2))
