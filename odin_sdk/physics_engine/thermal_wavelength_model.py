import numpy as np
from typing import Dict, Any, Optional, Callable

class ThermalWavelengthModel:
    """
    Models the relationship between temperature and wavelength in optical systems.
    
    Accounts for thermal expansion, thermo-optic effects, and mechanical strain
    to predict wavelength shifts with sub-nanometer precision.
    """
    def __init__(self, 
                 center_wavelength_nm: float,
                 dn_dT: float = 1.2e-5,  # Thermo-optic coefficient
                 alpha: float = 5.5e-7,  # Thermal expansion coefficient
                 cavity_length_mm: float = 10.0,
                 reference_temp_c: float = 25.0):
        """
        Initialize thermal wavelength model.
        
        Args:
            center_wavelength_nm: Center wavelength at reference temperature
            dn_dT: Thermo-optic coefficient (1/°C)
            alpha: Thermal expansion coefficient (1/°C)
            cavity_length_mm: Optical cavity length (mm)
            reference_temp_c: Reference temperature (°C)
        """
        self.center_wavelength_nm = center_wavelength_nm
        self.dn_dT = dn_dT
        self.alpha = alpha
        self.cavity_length_mm = cavity_length_mm
        self.reference_temp_c = reference_temp_c
        
        # Derived parameters
        self.wavelength_temp_sensitivity = self._calculate_sensitivity()
        
    def _calculate_sensitivity(self) -> float:
        """
        Calculate wavelength sensitivity to temperature.
        
        Returns:
            Sensitivity in nm/°C
        """
        # Combined effect of thermo-optic and thermal expansion
        # For a typical laser cavity: dλ/dT = λ(dn/dT/n + α)
        # Assuming n ≈ 1.5 for typical optical materials
        n = 1.5
        sensitivity = self.center_wavelength_nm * (self.dn_dT/n + self.alpha)
        return sensitivity
    
    def predict_wavelength(self, temperature_c: float) -> float:
        """
        Predict wavelength at given temperature.
        
        Args:
            temperature_c: Current temperature (°C)
            
        Returns:
            Predicted wavelength (nm)
        """
        delta_t = temperature_c - self.reference_temp_c
        wavelength_shift = delta_t * self.wavelength_temp_sensitivity
        return self.center_wavelength_nm + wavelength_shift
    
    def required_temperature(self, target_wavelength_nm: float) -> float:
        """
        Calculate required temperature for target wavelength.
        
        Args:
            target_wavelength_nm: Desired wavelength (nm)
            
        Returns:
            Required temperature (°C)
        """
        if self.wavelength_temp_sensitivity == 0:
            return self.reference_temp_c
            
        wavelength_shift = target_wavelength_nm - self.center_wavelength_nm
        delta_t = wavelength_shift / self.wavelength_temp_sensitivity
        return self.reference_temp_c + delta_t
    
    def thermal_response(self, 
                        initial_temp_c: float,
                        target_temp_c: float,
                        time_s: float,
                        time_constant_s: float) -> float:
        """
        Model thermal response over time (first-order system).
        
        Args:
            initial_temp_c: Initial temperature (°C)
            target_temp_c: Target temperature (°C)
            time_s: Time since change (s)
            time_constant_s: Thermal time constant (s)
            
        Returns:
            Current temperature (°C)
        """
        # First-order thermal response
        delta_t = target_temp_c - initial_temp_c
        response = initial_temp_c + delta_t * (1 - np.exp(-time_s / time_constant_s))
        return response
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for serialization."""
        return {
            'center_wavelength_nm': self.center_wavelength_nm,
            'dn_dT': self.dn_dT,
            'alpha': self.alpha,
            'cavity_length_mm': self.cavity_length_mm,
            'reference_temp_c': self.reference_temp_c,
            'wavelength_temp_sensitivity': self.wavelength_temp_sensitivity
        }