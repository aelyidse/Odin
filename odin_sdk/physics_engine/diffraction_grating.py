import numpy as np
from typing import Dict, Any, Optional, Tuple

class DiffractionGrating:
    """Concrete simulation of a diffraction grating with temperature effects."""
    def __init__(self,
                 groove_density_lpm: float,
                 blaze_angle_deg: float,
                 substrate_material: str,
                 coating_material: Optional[str] = None,
                 grating_length_mm: float = 25.0,
                 thermal_expansion_coeff: float = 5e-6,
                 base_temp_C: float = 25.0,
                 base_period_nm: Optional[float] = None,
                 refractive_index_fn: Optional[Any] = None):
        """
        Args:
            groove_density_lpm: Grooves per millimeter
            blaze_angle_deg: Blaze angle (degrees)
            substrate_material: Grating substrate material name
            coating_material: Optional coating material name
            grating_length_mm: Physical length of grating (mm)
            thermal_expansion_coeff: Linear expansion coefficient (1/K)
            base_temp_C: Reference temperature (C)
            base_period_nm: Optional explicit period at base_temp (nm)
            refractive_index_fn: Optional function n(lambda, T) for coating/material
        """
        self.groove_density_lpm = groove_density_lpm
        self.blaze_angle_deg = blaze_angle_deg
        self.substrate_material = substrate_material
        self.coating_material = coating_material
        self.grating_length_mm = grating_length_mm
        self.thermal_expansion_coeff = thermal_expansion_coeff
        self.base_temp_C = base_temp_C
        self.base_period_nm = base_period_nm or (1e6 / groove_density_lpm)
        self.refractive_index_fn = refractive_index_fn

    def period_nm(self, temp_C: float) -> float:
        """Return grating period (nm) at temp_C, accounting for thermal expansion."""
        delta_T = temp_C - self.base_temp_C
        return self.base_period_nm * (1 + self.thermal_expansion_coeff * delta_T)

    def diffraction_angle(self, wavelength_nm: float, order: int, incident_angle_deg: float, temp_C: float = 25.0) -> float:
        """Compute diffraction angle (deg) for given wavelength, order, and incident angle at temp_C."""
        d = self.period_nm(temp_C) * 1e-9  # m
        lam = wavelength_nm * 1e-9
        theta_i = np.deg2rad(incident_angle_deg)
        # Grating equation: d(sin(theta_i) + sin(theta_m)) = m*lambda
        arg = order * lam / d - np.sin(theta_i)
        if np.abs(arg) > 1:
            raise ValueError("No real solution for diffraction angle (total internal reflection or order out of range)")
        theta_m = np.arcsin(arg)
        return np.rad2deg(theta_m)

    def blaze_efficiency(self, wavelength_nm: float, order: int, temp_C: float = 25.0) -> float:
        """Estimate grating efficiency at blaze wavelength (scalar theory, ignores polarization)."""
        blaze_angle = np.deg2rad(self.blaze_angle_deg)
        d = self.period_nm(temp_C) * 1e-9
        lam = wavelength_nm * 1e-9
        # Scalar efficiency model: max at blaze wavelength
        blaze_wavelength = 2 * d * np.sin(blaze_angle)
        efficiency = np.exp(-((lam - blaze_wavelength) / (0.1 * blaze_wavelength)) ** 2)
        return float(np.clip(efficiency, 0, 1))

    def angular_dispersion(self, wavelength_nm: float, order: int, incident_angle_deg: float, temp_C: float = 25.0) -> float:
        """Compute angular dispersion d(theta_m)/d(lambda) in deg/nm."""
        d = self.period_nm(temp_C) * 1e-9
        lam = wavelength_nm * 1e-9
        theta_i = np.deg2rad(incident_angle_deg)
        arg = order * lam / d - np.sin(theta_i)
        if np.abs(arg) > 1:
            return 0.0
        dtheta_dlambda = order / (d * np.cos(np.arcsin(arg))) * 1e-9  # rad/m
        return np.rad2deg(dtheta_dlambda)
