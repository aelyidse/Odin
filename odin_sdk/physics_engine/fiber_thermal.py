import numpy as np
from typing import Dict, Any, Optional
from material_database.fiber import FiberCharacteristics, FiberThermalProfile
from material_database.material import OpticalMaterial

class FiberThermalManagement:
    """Concrete simulation of thermal management for fiber components."""
    def __init__(self,
                 fiber: FiberCharacteristics,
                 material: OpticalMaterial,
                 length_m: float,
                 dz: float = 1e-3,
                 cooling_coeff_W_m2K: float = 20.0,
                 ambient_temp_C: float = 25.0):
        """
        Args:
            fiber: FiberCharacteristics instance
            material: OpticalMaterial instance (for thermal conductivity, specific heat, etc.)
            length_m: Fiber/component length (meters)
            dz: Spatial discretization (meters)
            cooling_coeff_W_m2K: Convective cooling coefficient (W/m^2/K)
            ambient_temp_C: Ambient/coolant temperature (C)
        """
        self.fiber = fiber
        self.material = material
        self.length_m = length_m
        self.dz = dz
        self.cooling_coeff = cooling_coeff_W_m2K
        self.ambient_temp_C = ambient_temp_C
        self.z = np.arange(0, length_m + dz, dz)
        self.radius_m = fiber.core_diameter_um * 1e-6 / 2
        self.area_m2 = np.pi * self.radius_m ** 2

    def heat_source_profile(self, pump_absorption_W_m: np.ndarray, signal_absorption_W_m: np.ndarray, quantum_defect: float) -> np.ndarray:
        """Compute distributed heat source profile along the fiber (W/m)."""
        # Total heat: pump absorption + signal absorption, scaled by quantum defect
        return quantum_defect * (pump_absorption_W_m + signal_absorption_W_m)

    def steady_state_profile(self, heat_profile_W_m: np.ndarray) -> np.ndarray:
        """Compute steady-state temperature profile along the fiber (1D finite difference)."""
        k = self.material.thermal_conductivity(self.ambient_temp_C)  # W/(m K)
        h = self.cooling_coeff
        dz = self.dz
        N = len(self.z)
        T = np.full(N, self.ambient_temp_C)
        for _ in range(1000):  # Fixed-point iteration
            T_new = T.copy()
            for i in range(1, N-1):
                d2T = (T[i-1] - 2*T[i] + T[i+1]) / dz**2
                cooling = h * (T[i] - self.ambient_temp_C) / (k / self.radius_m)
                T_new[i] = T[i] + 0.05 * (heat_profile_W_m[i] / (k * self.area_m2) - d2T - cooling)
            if np.max(np.abs(T_new - T)) < 1e-3:
                break
            T = T_new
        return T

    def transient_profile(self, heat_profile_W_m: np.ndarray, times_s: np.ndarray, initial_temp_C: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute transient temperature profile along the fiber (explicit finite difference)."""
        k = self.material.thermal_conductivity(self.ambient_temp_C)
        c = self.material.specific_heat(self.ambient_temp_C)
        rho = self.material.density
        dz = self.dz
        dt = times_s[1] - times_s[0]
        N = len(self.z)
        T = np.full(N, self.ambient_temp_C) if initial_temp_C is None else initial_temp_C.copy()
        result = np.zeros((N, len(times_s)))
        result[:, 0] = T
        for t_idx in range(1, len(times_s)):
            T_new = T.copy()
            for i in range(1, N-1):
                d2T = (T[i-1] - 2*T[i] + T[i+1]) / dz**2
                cooling = self.cooling_coeff * (T[i] - self.ambient_temp_C) / (k / self.radius_m)
                T_new[i] = T[i] + dt * (heat_profile_W_m[i] / (rho * c * self.area_m2) - d2T - cooling)
            result[:, t_idx] = T_new
            T = T_new
        return result

    def to_thermal_profile(self, temp_profile: np.ndarray, times_s: Optional[np.ndarray] = None) -> FiberThermalProfile:
        return FiberThermalProfile(positions_m=self.z, times_s=times_s, temperature_C=temp_profile)
