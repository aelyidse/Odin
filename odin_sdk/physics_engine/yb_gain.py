import numpy as np
from typing import Dict, Any, Callable, Optional

class YbDopingProfile:
    """Concrete model for ytterbium doping concentration profile in fiber core."""
    def __init__(self, profile_fn: Callable[[float], float], core_radius_um: float):
        """
        Args:
            profile_fn: Function returning Yb concentration (ions/m^3) as function of radius (um)
            core_radius_um: Core radius (microns)
        """
        self.profile_fn = profile_fn
        self.core_radius_um = core_radius_um

    def concentration(self, r_um: float) -> float:
        """Return Yb concentration at radius r (microns)."""
        if r_um > self.core_radius_um:
            return 0.0
        return self.profile_fn(r_um)

    def average_concentration(self, n_samples: int = 100) -> float:
        """Compute average Yb concentration in the core."""
        rs = np.linspace(0, self.core_radius_um, n_samples)
        vals = np.array([self.concentration(r) * 2 * np.pi * r for r in rs])
        return vals.sum() / (np.pi * self.core_radius_um ** 2)

class YbFiberGainModel:
    """Concrete model for Yb-doped fiber gain and population inversion."""
    def __init__(self,
                 doping_profile: YbDopingProfile,
                 sigma_abs_pump: float,
                 sigma_em_signal: float,
                 sigma_abs_signal: float,
                 pump_wavelength_nm: float,
                 signal_wavelength_nm: float,
                 pump_power_W: float,
                 signal_power_W: float,
                 fiber_length_m: float,
                 quantum_efficiency: float = 0.9,
                 background_loss_db_per_m: float = 0.01):
        """
        Args:
            doping_profile: YbDopingProfile instance
            sigma_abs_pump: Absorption cross-section at pump wavelength (m^2)
            sigma_em_signal: Emission cross-section at signal wavelength (m^2)
            sigma_abs_signal: Absorption cross-section at signal wavelength (m^2)
            pump_wavelength_nm: Pump wavelength (nm)
            signal_wavelength_nm: Signal wavelength (nm)
            pump_power_W: Input pump power (W)
            signal_power_W: Input signal power (W)
            fiber_length_m: Fiber length (m)
            quantum_efficiency: Fraction of absorbed pump converted to signal
            background_loss_db_per_m: Background loss (dB/m)
        """
        self.doping_profile = doping_profile
        self.sigma_abs_pump = sigma_abs_pump
        self.sigma_em_signal = sigma_em_signal
        self.sigma_abs_signal = sigma_abs_signal
        self.pump_wavelength_nm = pump_wavelength_nm
        self.signal_wavelength_nm = signal_wavelength_nm
        self.pump_power_W = pump_power_W
        self.signal_power_W = signal_power_W
        self.fiber_length_m = fiber_length_m
        self.quantum_efficiency = quantum_efficiency
        self.background_loss_db_per_m = background_loss_db_per_m

    def small_signal_gain(self, z: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute small-signal gain (1/m) along the fiber using average Yb concentration."""
        avg_N = self.doping_profile.average_concentration()
        sigma_e = self.sigma_em_signal
        sigma_a = self.sigma_abs_signal
        if z is None:
            z = np.linspace(0, self.fiber_length_m, 100)
        # Simple exponential decay of pump (no ASE, single pass)
        pump_abs_coeff = avg_N * self.sigma_abs_pump
        pump = self.pump_power_W * np.exp(-pump_abs_coeff * z)
        # Population inversion (approximate, steady-state)
        N2 = avg_N * (1 - np.exp(-pump_abs_coeff * z)) * self.quantum_efficiency
        gain = (sigma_e * N2 - sigma_a * (avg_N - N2))
        # Subtract background loss (convert dB/m to 1/m)
        alpha_bg = self.background_loss_db_per_m / 4.343
        return gain - alpha_bg

    def saturated_gain(self, input_signal_W: float, z: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute saturated gain (1/m) along the fiber for given input signal power."""
        avg_N = self.doping_profile.average_concentration()
        sigma_e = self.sigma_em_signal
        sigma_a = self.sigma_abs_signal
        if z is None:
            z = np.linspace(0, self.fiber_length_m, 100)
        # Simple saturation model
        Isat = 1.0 / (sigma_e + sigma_a)  # Saturation intensity (arbitrary units)
        S = input_signal_W / Isat
        pump_abs_coeff = avg_N * self.sigma_abs_pump
        pump = self.pump_power_W * np.exp(-pump_abs_coeff * z)
        N2 = avg_N * (1 - np.exp(-pump_abs_coeff * z)) * self.quantum_efficiency / (1 + S)
        gain = (sigma_e * N2 - sigma_a * (avg_N - N2))
        alpha_bg = self.background_loss_db_per_m / 4.343
        return gain - alpha_bg
