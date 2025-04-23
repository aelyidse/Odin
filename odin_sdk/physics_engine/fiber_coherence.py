import numpy as np
from typing import Dict, Any, Optional

class FiberAmplifierCoherenceModel:
    """
    Models spatiotemporal coherence effects in high-power fiber amplifiers.
    Includes mutual coherence function, temporal coherence length, and spatial coherence area.
    """
    def __init__(self, wavelength_nm: float, bandwidth_nm: float, core_radius_um: float, NA: float, pulse_duration_ps: Optional[float] = None):
        self.wavelength_nm = wavelength_nm
        self.bandwidth_nm = bandwidth_nm
        self.core_radius_um = core_radius_um
        self.NA = NA
        self.pulse_duration_ps = pulse_duration_ps
        self._compute_derived()
    def _compute_derived(self):
        c = 2.99792458e8  # m/s
        wl = self.wavelength_nm * 1e-9
        bw = self.bandwidth_nm * 1e-9
        self.temporal_coherence_length_m = wl**2 / (bw * c) if bw > 0 else np.inf
        self.temporal_coherence_time_s = self.temporal_coherence_length_m / c
        self.spatial_coherence_area_m2 = np.pi * (self.core_radius_um*1e-6)**2 * (1 - np.cos(np.arcsin(self.NA)))
        if self.pulse_duration_ps:
            self.spectral_width_Hz = 0.44 / (self.pulse_duration_ps * 1e-12)
        else:
            self.spectral_width_Hz = bw * c / wl**2
    def mutual_coherence(self, r1: np.ndarray, r2: np.ndarray, tau: float) -> float:
        """
        Compute the mutual coherence function between two points r1, r2 (in meters) and time delay tau (in seconds).
        Assumes Gaussian statistics and stationary source.
        """
        delta_r = np.linalg.norm(r1 - r2)
        spatial = np.exp(-delta_r**2 / (2 * self.spatial_coherence_area_m2))
        temporal = np.exp(-(tau**2) / (2 * self.temporal_coherence_time_s**2))
        return spatial * temporal
    def as_dict(self) -> Dict[str, Any]:
        return {
            'wavelength_nm': self.wavelength_nm,
            'bandwidth_nm': self.bandwidth_nm,
            'core_radius_um': self.core_radius_um,
            'NA': self.NA,
            'temporal_coherence_length_m': self.temporal_coherence_length_m,
            'temporal_coherence_time_s': self.temporal_coherence_time_s,
            'spatial_coherence_area_m2': self.spatial_coherence_area_m2,
            'spectral_width_Hz': self.spectral_width_Hz
        }

# Example usage:
# model = FiberAmplifierCoherenceModel(1064, 1, 10, 0.14, pulse_duration_ps=1)
# coh = model.mutual_coherence(np.array([0,0]), np.array([0.01,0]), 1e-13)
# print(model.as_dict())
