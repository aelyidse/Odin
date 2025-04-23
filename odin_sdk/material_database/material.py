from typing import Dict, Callable, Tuple, Optional
import numpy as np

class OpticalMaterial:
    """Concrete schema for optical materials with temperature-dependent properties."""
    def __init__(self,
                 name: str,
                 refractive_index_fn: Callable[[float, float], float],
                 absorption_coeff_fn: Callable[[float, float], float],
                 thermal_conductivity_fn: Callable[[float], float],
                 specific_heat_fn: Callable[[float], float],
                 density: float,
                 wavelength_range: Tuple[float, float],
                 temp_range: Tuple[float, float],
                 metadata: Optional[Dict[str, str]] = None):
        """
        Args:
            name: Material name/identifier
            refractive_index_fn: Function n(lambda, T) -> n (wavelength [nm], temperature [C])
            absorption_coeff_fn: Function alpha(lambda, T) -> 1/m
            thermal_conductivity_fn: Function k(T) -> W/(m K)
            specific_heat_fn: Function c(T) -> J/(kg K)
            density: Density (kg/m^3)
            wavelength_range: (lambda_min, lambda_max) in nm
            temp_range: (T_min, T_max) in C
            metadata: Optional dictionary for additional fields
        """
        self.name = name
        self.refractive_index_fn = refractive_index_fn
        self.absorption_coeff_fn = absorption_coeff_fn
        self.thermal_conductivity_fn = thermal_conductivity_fn
        self.specific_heat_fn = specific_heat_fn
        self.density = density
        self.wavelength_range = wavelength_range
        self.temp_range = temp_range
        self.metadata = metadata or {}

    def refractive_index(self, wavelength_nm: float, temp_C: float) -> float:
        return self.refractive_index_fn(wavelength_nm, temp_C)

    def absorption_coeff(self, wavelength_nm: float, temp_C: float) -> float:
        return self.absorption_coeff_fn(wavelength_nm, temp_C)

    def thermal_conductivity(self, temp_C: float) -> float:
        return self.thermal_conductivity_fn(temp_C)

    def specific_heat(self, temp_C: float) -> float:
        return self.specific_heat_fn(temp_C)

    def in_range(self, wavelength_nm: float, temp_C: float) -> bool:
        return (self.wavelength_range[0] <= wavelength_nm <= self.wavelength_range[1] and
                self.temp_range[0] <= temp_C <= self.temp_range[1])

class TabulatedOpticalMaterial(OpticalMaterial):
    """Tabulated optical material with property interpolation."""
    def __init__(self,
                 name: str,
                 refractive_index_table: np.ndarray,
                 absorption_coeff_table: np.ndarray,
                 wavelengths: np.ndarray,
                 temperatures: np.ndarray,
                 thermal_conductivity_table: np.ndarray,
                 specific_heat_table: np.ndarray,
                 density: float,
                 metadata: Optional[Dict[str, str]] = None):
        """
        Args:
            refractive_index_table: 2D array (wavelength x temp)
            absorption_coeff_table: 2D array (wavelength x temp)
            wavelengths: 1D array of wavelengths (nm)
            temperatures: 1D array of temperatures (C)
            thermal_conductivity_table: 1D array (temp)
            specific_heat_table: 1D array (temp)
        """
        from scipy.interpolate import RegularGridInterpolator
        self._n_interp = RegularGridInterpolator((wavelengths, temperatures), refractive_index_table)
        self._a_interp = RegularGridInterpolator((wavelengths, temperatures), absorption_coeff_table)
        self._k_interp = RegularGridInterpolator((temperatures,), thermal_conductivity_table)
        self._c_interp = RegularGridInterpolator((temperatures,), specific_heat_table)
        super().__init__(
            name=name,
            refractive_index_fn=lambda l, t: float(self._n_interp([[l, t]])),
            absorption_coeff_fn=lambda l, t: float(self._a_interp([[l, t]])),
            thermal_conductivity_fn=lambda t: float(self._k_interp([[t]])),
            specific_heat_fn=lambda t: float(self._c_interp([[t]])),
            density=density,
            wavelength_range=(wavelengths[0], wavelengths[-1]),
            temp_range=(temperatures[0], temperatures[-1]),
            metadata=metadata
        )
