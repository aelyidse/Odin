import numpy as np
from typing import Dict, Any

class ThermalModel:
    """Concrete thermal modeling system for high-power components (steady-state and transient)."""
    def __init__(self, material_props: Dict[str, float], geometry: Dict[str, float], cooling: Dict[str, Any]):
        """
        Args:
            material_props: Dictionary with keys 'thermal_conductivity', 'specific_heat', 'density', etc.
            geometry: Dictionary with keys 'length', 'width', 'height', 'surface_area', 'volume', etc.
            cooling: Dictionary with cooling parameters (type, convection coeff, coolant temp, etc.)
        """
        self.material_props = material_props
        self.geometry = geometry
        self.cooling = cooling

    def steady_state_temperature(self, power_input: float, ambient_temp: float) -> float:
        """Compute steady-state maximum temperature under continuous power load.
        Args:
            power_input: Power dissipated (W)
            ambient_temp: Ambient/coolant temperature (C)
        Returns:
            Maximum component temperature (C)
        """
        # Q = h*A*(T-T_ambient) --> T = Q/(h*A) + T_ambient
        h = self.cooling.get('convection_coeff', 10.0)  # W/(m^2 K)
        A = self.geometry.get('surface_area', 1.0)       # m^2
        T = power_input / (h * A) + ambient_temp
        return T

    def transient_temperature(self, power_input: float, ambient_temp: float, t: float, initial_temp: float = None) -> float:
        """Compute temperature at time t for a step power input (lumped capacitance model).
        Args:
            power_input: Power dissipated (W)
            ambient_temp: Ambient/coolant temperature (C)
            t: Time since power applied (s)
            initial_temp: Starting component temperature (C), defaults to ambient
        Returns:
            Component temperature at time t (C)
        """
        # Lumped capacitance: T(t) = T_ambient + (T0-T_ambient)*exp(-t/tau) + Q/(h*A)*(1-exp(-t/tau))
        h = self.cooling.get('convection_coeff', 10.0)
        A = self.geometry.get('surface_area', 1.0)
        V = self.geometry.get('volume', 1.0)
        rho = self.material_props.get('density', 8000.0)
        c = self.material_props.get('specific_heat', 500.0)
        tau = rho * c * V / (h * A)  # time constant (s)
        T0 = initial_temp if initial_temp is not None else ambient_temp
        T_inf = power_input / (h * A) + ambient_temp
        T = T_inf + (T0 - T_inf) * np.exp(-t / tau)
        return T

    def temperature_profile(self, power_map: np.ndarray, ambient_temp: float, time: float = None) -> np.ndarray:
        """Compute spatial temperature profile for distributed heat sources.
        Args:
            power_map: 2D or 3D array of power density (W/m^3)
            ambient_temp: Ambient/coolant temperature (C)
            time: Optional time for transient profile (s)
        Returns:
            Array of temperatures (C)
        """
        # For simplicity, steady-state solution using finite difference (Dirichlet BC at boundaries)
        k = self.material_props.get('thermal_conductivity', 100.0)  # W/(m K)
        dx = self.geometry.get('dx', 1e-3)
        shape = power_map.shape
        temp = np.full(shape, ambient_temp)
        # Iterative solver for steady-state: k*laplacian(T) + power = 0
        for _ in range(500):  # Fixed-point iteration
            lap = (
                np.roll(temp, 1, axis=0) + np.roll(temp, -1, axis=0) +
                np.roll(temp, 1, axis=1) + np.roll(temp, -1, axis=1) - 4 * temp
            ) / dx**2
            temp = temp + 0.1 * (power_map / k - lap)
        return temp
