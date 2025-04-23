import numpy as np
from typing import Callable, Dict, Any, Optional

class ThermalLensingModel:
    """
    Models thermal lensing in fiber amplifiers with spatially varying dopant concentration.
    Computes temperature profile, refractive index change, and induced lens focal length.
    """
    def __init__(self, fiber_radius_um: float, fiber_length_m: float, pump_power_w: float, absorption_coeff: Callable[[float], float], dopant_profile: Callable[[float], float], dn_dT: float = 1.2e-5, thermal_conductivity: float = 1.4, ambient_temp_C: float = 25.0):
        self.fiber_radius_um = fiber_radius_um
        self.fiber_length_m = fiber_length_m
        self.pump_power_w = pump_power_w
        self.absorption_coeff = absorption_coeff  # function of radius (um)
        self.dopant_profile = dopant_profile      # function of radius (um)
        self.dn_dT = dn_dT
        self.thermal_conductivity = thermal_conductivity
        self.ambient_temp_C = ambient_temp_C
        self._compute_profiles()
    def _compute_profiles(self):
        r = np.linspace(0, self.fiber_radius_um, 100)
        dr = r[1] - r[0]
        # Absorbed power density profile [W/m^3]
        absorption = np.array([self.absorption_coeff(ri) for ri in r])
        dopant = np.array([self.dopant_profile(ri) for ri in r])
        absorbed_power_density = absorption * dopant * self.pump_power_w / (np.pi * (self.fiber_radius_um*1e-6)**2 * self.fiber_length_m)
        # Steady-state radial temperature profile (cylindrical symmetry, 1D)
        k = self.thermal_conductivity
        temp_profile = np.zeros_like(r)
        for i, ri in enumerate(r):
            if ri == 0:
                temp_profile[i] = self.ambient_temp_C
            else:
                # Integrate power density from center to ri
                Q = np.trapz(absorbed_power_density[:i+1]*r[:i+1], r[:i+1]) * 2 * np.pi * self.fiber_length_m
                temp_profile[i] = self.ambient_temp_C + Q / (2 * np.pi * k * ri * 1e-6 * self.fiber_length_m)
        self.r_um = r
        self.temp_profile_C = temp_profile
        # Refractive index change profile
        self.dn_profile = self.dn_dT * (temp_profile - self.ambient_temp_C)
        # Effective focal length (thin lens approximation)
        dn_dr = np.gradient(self.dn_profile, self.r_um*1e-6)
        self.focal_length_m = 1 / (2 * dn_dr[-1] / self.fiber_radius_um*1e-6) if dn_dr[-1] != 0 else np.inf
    def get_profiles(self) -> Dict[str, Any]:
        return {
            'radius_um': self.r_um.tolist(),
            'temperature_C': self.temp_profile_C.tolist(),
            'dn': self.dn_profile.tolist(),
            'focal_length_m': float(self.focal_length_m)
        }

# Example usage:
# absorption_coeff = lambda r: 1.0  # Flat or function of r
# dopant_profile = lambda r: 1.0 if r < 8 else 0.1  # Step or graded
# model = ThermalLensingModel(10, 5, 1000, absorption_coeff, dopant_profile)
# print(model.get_profiles())
