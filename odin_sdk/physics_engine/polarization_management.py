import numpy as np
from typing import Dict, Any, Optional

class BirefringenceFiber:
    """Models polarization evolution in fiber with stress-induced birefringence."""
    def __init__(self, length_m: float, delta_n: float, orientation_rad: float = 0.0):
        self.length_m = length_m
        self.delta_n = delta_n  # Birefringence (n_fast - n_slow)
        self.orientation_rad = orientation_rad  # Fast axis orientation (radians)
    def retardance(self, wavelength_nm: float) -> float:
        # Phase retardance between fast and slow axes
        wl_m = wavelength_nm * 1e-9
        return 2 * np.pi * self.delta_n * self.length_m / wl_m
    def jones_matrix(self, wavelength_nm: float) -> np.ndarray:
        # Jones matrix for birefringent fiber segment
        phi = self.retardance(wavelength_nm)
        theta = self.orientation_rad
        J = np.array([
            [np.cos(theta)**2 + np.sin(theta)**2 * np.exp(1j*phi),
             np.cos(theta)*np.sin(theta)*(1 - np.exp(1j*phi))],
            [np.cos(theta)*np.sin(theta)*(1 - np.exp(1j*phi)),
             np.sin(theta)**2 + np.cos(theta)**2 * np.exp(1j*phi)]
        ])
        return J
    def propagate(self, input_jones: np.ndarray, wavelength_nm: float) -> np.ndarray:
        J = self.jones_matrix(wavelength_nm)
        return J @ input_jones

class PolarizationController:
    """Manages polarization state with birefringence compensation."""
    def __init__(self, fiber_segments: Optional[list] = None):
        self.fiber_segments = fiber_segments or []
    def add_segment(self, segment: BirefringenceFiber):
        self.fiber_segments.append(segment)
    def compensate(self, input_jones: np.ndarray, wavelength_nm: float) -> np.ndarray:
        state = input_jones
        for seg in self.fiber_segments:
            state = seg.propagate(state, wavelength_nm)
        return state
    def get_total_jones(self, wavelength_nm: float) -> np.ndarray:
        J = np.eye(2, dtype=complex)
        for seg in self.fiber_segments:
            J = seg.jones_matrix(wavelength_nm) @ J
        return J

# Example usage:
# fiber = BirefringenceFiber(10, 3e-5, orientation_rad=np.pi/4)
# controller = PolarizationController([fiber])
# input_pol = np.array([1, 0])  # Horizontal polarization
# output_pol = controller.compensate(input_pol, 1064)
# print(output_pol)
