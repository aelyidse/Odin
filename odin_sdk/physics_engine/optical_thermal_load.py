import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from material_database.material import OpticalMaterial

class OpticalThermalLoad:
    """
    Simulates thermal loading effects in optical components under high power.
    Models absorption, heat distribution, and thermal lensing effects.
    """
    def __init__(self,
                 material: OpticalMaterial,
                 thickness_mm: float,
                 diameter_mm: float,
                 cooling_coeff_W_m2K: float = 15.0,
                 ambient_temp_C: float = 25.0,
                 spatial_resolution_mm: float = 0.1):
        """
        Initialize optical thermal load simulator.
        
        Args:
            material: OpticalMaterial instance
            thickness_mm: Component thickness (mm)
            diameter_mm: Component diameter (mm)
            cooling_coeff_W_m2K: Convective cooling coefficient (W/m²/K)
            ambient_temp_C: Ambient temperature (°C)
            spatial_resolution_mm: Spatial discretization (mm)
        """
        self.material = material
        self.thickness_m = thickness_mm * 1e-3
        self.radius_m = diameter_mm * 1e-3 / 2
        self.cooling_coeff = cooling_coeff_W_m2K
        self.ambient_temp_C = ambient_temp_C
        
        # Setup spatial grid
        self.dr = spatial_resolution_mm * 1e-3  # m
        self.nr = int(self.radius_m / self.dr) + 1
        self.r = np.linspace(0, self.radius_m, self.nr)
        
        # Material properties at ambient temperature
        self.k = material.thermal_conductivity(ambient_temp_C)  # W/(m·K)
        self.c = material.specific_heat(ambient_temp_C)  # J/(kg·K)
        self.rho = material.density  # kg/m³
        
    def calculate_absorption(self, 
                            power_W: float, 
                            wavelength_nm: float, 
                            beam_radius_mm: float) -> np.ndarray:
        """
        Calculate absorbed power density distribution.
        
        Args:
            power_W: Incident beam power (W)
            wavelength_nm: Beam wavelength (nm)
            beam_radius_mm: 1/e² beam radius (mm)
            
        Returns:
            Absorbed power density (W/m³) as function of radius
        """
        beam_radius_m = beam_radius_mm * 1e-3
        # Get absorption coefficient at this wavelength and ambient temp
        alpha = self.material.absorption_coeff(wavelength_nm, self.ambient_temp_C)  # 1/m
        
        # Gaussian beam intensity profile
        intensity = power_W / (np.pi * beam_radius_m**2) * np.exp(-2 * (self.r / beam_radius_m)**2)  # W/m²
        
        # Absorbed power density (Beer-Lambert law)
        absorbed_density = alpha * intensity  # W/m³
        
        return absorbed_density
    
    def steady_state_temperature(self, absorbed_power_density: np.ndarray) -> np.ndarray:
        """
        Calculate steady-state temperature profile using finite difference method.
        
        Args:
            absorbed_power_density: Absorbed power density (W/m³) as function of radius
            
        Returns:
            Temperature profile (°C) as function of radius
        """
        # Initialize temperature array
        T = np.full(self.nr, self.ambient_temp_C)
        
        # Finite difference solution of heat equation with source term
        # d²T/dr² + (1/r)·dT/dr + q/(k) = 0
        for _ in range(1000):  # Iterative solution
            T_new = T.copy()
            
            # Interior points
            for i in range(1, self.nr-1):
                # Second derivative approximation
                d2T = (T[i-1] - 2*T[i] + T[i+1]) / self.dr**2
                
                # First derivative approximation (central difference)
                dT = (T[i+1] - T[i-1]) / (2 * self.dr)
                
                # Source term
                q = absorbed_power_density[i]
                
                # Heat equation with source
                if self.r[i] > 0:  # Avoid division by zero at r=0
                    T_new[i] = T[i] + 0.1 * (d2T + dT/self.r[i] + q/self.k)
                else:
                    T_new[i] = T[i] + 0.1 * (2*d2T + q/self.k)  # At r=0, use symmetry
            
            # Boundary conditions
            # At r=0: dT/dr = 0 (symmetry)
            T_new[0] = T_new[1]
            
            # At r=R: -k·dT/dr = h·(T-T_ambient) (convective cooling)
            dT_dr = (T[-2] - T[-1]) / self.dr
            T_new[-1] = T_new[-2] + self.dr * self.cooling_coeff * (T[-1] - self.ambient_temp_C) / self.k
            
            # Check convergence
            if np.max(np.abs(T_new - T)) < 1e-3:
                break
                
            T = T_new
            
        return T
    
    def calculate_thermal_lensing(self, temperature_profile: np.ndarray, wavelength_nm: float) -> Dict[str, Any]:
        """
        Calculate thermal lensing effects from temperature profile.
        
        Args:
            temperature_profile: Temperature profile (°C) as function of radius
            wavelength_nm: Beam wavelength (nm)
            
        Returns:
            Dictionary with thermal lensing parameters
        """
        # Get temperature-dependent refractive index
        n_ambient = self.material.refractive_index(wavelength_nm, self.ambient_temp_C)
        
        # Calculate dn/dT (approximate using finite difference)
        delta_T = 5.0  # °C
        n_delta = self.material.refractive_index(wavelength_nm, self.ambient_temp_C + delta_T)
        dn_dT = (n_delta - n_ambient) / delta_T  # 1/°C
        
        # Calculate optical path difference (OPD)
        delta_T_profile = temperature_profile - self.ambient_temp_C
        opd = dn_dT * delta_T_profile * self.thickness_m  # m
        
        # Fit parabola to estimate focal length
        # For a parabolic OPD: OPD(r) = r²/(2f)
        # where f is the focal length
        valid_idx = ~np.isnan(opd)
        if np.sum(valid_idx) > 3:  # Need at least 3 points for quadratic fit
            p = np.polyfit(self.r[valid_idx]**2, opd[valid_idx], 1)
            focal_length = 1/(2*p[0]) if abs(p[0]) > 1e-10 else float('inf')
        else:
            focal_length = float('inf')
        
        # Calculate peak-to-valley wavefront distortion
        p2v = np.max(opd) - np.min(opd)
        
        # Calculate RMS wavefront error
        rms_wavefront = np.std(opd)
        
        return {
            'focal_length_m': float(focal_length),
            'peak_to_valley_m': float(p2v),
            'rms_wavefront_m': float(rms_wavefront),
            'opd_profile_m': opd.tolist(),
            'dn_dT': float(dn_dT)
        }
    
    def simulate(self, power_W: float, wavelength_nm: float, beam_radius_mm: float) -> Dict[str, Any]:
        """
        Run complete thermal simulation and return results.
        
        Args:
            power_W: Incident beam power (W)
            wavelength_nm: Beam wavelength (nm)
            beam_radius_mm: 1/e² beam radius (mm)
            
        Returns:
            Dictionary with simulation results
        """
        # Calculate absorption
        absorbed_power = self.calculate_absorption(power_W, wavelength_nm, beam_radius_mm)
        
        # Calculate temperature profile
        temperature = self.steady_state_temperature(absorbed_power)
        
        # Calculate thermal lensing
        lensing = self.calculate_thermal_lensing(temperature, wavelength_nm)
        
        # Calculate total absorbed power
        total_absorbed = np.trapz(2 * np.pi * self.r * absorbed_power, self.r) * self.thickness_m
        
        # Calculate maximum temperature
        max_temp = np.max(temperature)
        
        return {
            'max_temperature_C': float(max_temp),
            'temperature_profile_C': temperature.tolist(),
            'radial_positions_m': self.r.tolist(),
            'total_absorbed_power_W': float(total_absorbed),
            'thermal_lensing': lensing,
            'material': self.material.name,
            'wavelength_nm': wavelength_nm,
            'input_power_W': power_W
        }