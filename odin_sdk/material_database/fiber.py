from typing import Dict, Tuple, Optional, List
import numpy as np

class FiberCharacteristics:
    """Concrete data structure for fiber properties and configuration."""
    def __init__(self,
                 core_diameter_um: float,
                 cladding_diameter_um: float,
                 numerical_aperture: float,
                 length_m: float,
                 pm_type: Optional[str] = None,
                 birefringence: Optional[float] = None,
                 doping_profile: Optional[Dict[str, float]] = None,
                 connector_type: Optional[str] = None,
                 connector_alignment_deg: Optional[float] = None,
                 attenuation_db_per_km: Optional[float] = None,
                 polarization_extinction_ratio_db: Optional[float] = None,
                 additional_params: Optional[Dict[str, float]] = None):
        self.core_diameter_um = core_diameter_um
        self.cladding_diameter_um = cladding_diameter_um
        self.numerical_aperture = numerical_aperture
        self.length_m = length_m
        self.pm_type = pm_type  # e.g., 'PANDA', 'bow-tie', None
        self.birefringence = birefringence  # Delta n
        self.doping_profile = doping_profile or {}  # e.g., {'Yb': 5e25}
        self.connector_type = connector_type
        self.connector_alignment_deg = connector_alignment_deg
        self.attenuation_db_per_km = attenuation_db_per_km
        self.polarization_extinction_ratio_db = polarization_extinction_ratio_db
        self.additional_params = additional_params or {}

    def mode_field_diameter(self) -> float:
        """Estimate mode field diameter (MFD) based on core and NA (approximate formula)."""
        # MFD ≈ 0.65*core_diameter + 1.619*lambda/NA (for single-mode, lambda in um)
        lambda_um = self.additional_params.get('wavelength_um', 1.06)
        return 0.65 * self.core_diameter_um + 1.619 * lambda_um / self.numerical_aperture

class FiberThermalProfile:
    """Thermal profile for fiber or component: temperature as a function of position and time."""
    def __init__(self,
                 positions_m: np.ndarray,
                 times_s: Optional[np.ndarray] = None,
                 temperature_C: Optional[np.ndarray] = None):
        """
        Args:
            positions_m: 1D array of positions along fiber/component (meters)
            times_s: Optional 1D array of time points (seconds)
            temperature_C: 1D or 2D array of temperatures (C). Shape: (len(positions),) or (len(positions), len(times))
        """
        self.positions_m = positions_m
        self.times_s = times_s
        self.temperature_C = temperature_C  # Shape: (positions,) or (positions, times)

    def get_profile(self, time_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Get temperature profile along fiber/component at a specific time index."""
        if self.temperature_C is None:
            raise ValueError("No temperature data available.")
        if self.times_s is None or self.temperature_C.ndim == 1:
            return self.positions_m, self.temperature_C
        return self.positions_m, self.temperature_C[:, time_idx]

    def get_history(self, position_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get temperature history at a specific position index."""
        if self.temperature_C is None or self.times_s is None or self.temperature_C.ndim == 1:
            raise ValueError("No time-dependent data available.")
        return self.times_s, self.temperature_C[position_idx, :]
