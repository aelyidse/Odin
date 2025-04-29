import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize

class SpectralBeamEfficiencyOptimizer:
    """
    Optimizes spectral beam combining efficiency by adjusting grating angles,
    wavelength spacing, and beam alignment parameters.
    """
    def __init__(self, 
                 initial_wavelengths_nm: List[float],
                 grating_line_density: float = 1200.0,  # lines/mm
                 grating_blaze_angle_deg: float = 17.5,
                 target_efficiency: float = 0.95):
        """
        Initialize the spectral beam efficiency optimizer.
        
        Args:
            initial_wavelengths_nm: List of initial laser wavelengths (nm)
            grating_line_density: Diffraction grating line density (lines/mm)
            grating_blaze_angle_deg: Grating blaze angle (degrees)
            target_efficiency: Target combining efficiency (0-1)
        """
        self.wavelengths_nm = np.array(initial_wavelengths_nm)
        self.n_channels = len(initial_wavelengths_nm)
        self.grating_line_density = grating_line_density
        self.grating_blaze_angle_deg = grating_blaze_angle_deg
        self.target_efficiency = target_efficiency
        
        # Initial parameters
        self.grating_angle_deg = 45.0  # Initial grating angle
        self.beam_angles_deg = np.zeros(self.n_channels)  # Incident angles
        self.beam_positions_mm = np.zeros(self.n_channels)  # Beam positions on grating
        
        # Performance metrics
        self.current_efficiency = 0.0
        self.diffraction_losses = np.zeros(self.n_channels)
        self.alignment_errors = np.zeros(self.n_channels)
        
        # Initialize beam angles based on grating equation
        self._initialize_beam_angles()
    
    def _initialize_beam_angles(self):
        """Calculate initial beam angles using the grating equation."""
        # Grating equation: sin(θᵢ) + sin(θₘ) = m·λ/d
        # where θᵢ is incident angle, θₘ is diffraction angle, m is order, λ is wavelength, d is grating period
        d = 1.0 / self.grating_line_density  # grating period in mm
        m = 1  # First diffraction order
        
        # For spectral beam combining, we want all diffracted beams to exit at the same angle
        # So we solve for the incident angles that achieve this
        diffraction_angle_deg = self.grating_angle_deg
        diffraction_angle_rad = np.radians(diffraction_angle_deg)
        
        for i, wavelength_nm in enumerate(self.wavelengths_nm):
            wavelength_mm = wavelength_nm * 1e-6  # Convert nm to mm
            # Solve grating equation for incident angle
            sin_theta_i = m * wavelength_mm / d - np.sin(diffraction_angle_rad)
            # Clamp to valid range
            sin_theta_i = np.clip(sin_theta_i, -1.0, 1.0)
            incident_angle_rad = np.arcsin(sin_theta_i)
            self.beam_angles_deg[i] = np.degrees(incident_angle_rad)
    
    def calculate_efficiency(self, 
                            grating_angle_deg: float, 
                            beam_angles_deg: np.ndarray,
                            wavelengths_nm: np.ndarray) -> float:
        """
        Calculate the spectral beam combining efficiency for given parameters.
        
        Args:
            grating_angle_deg: Grating angle (degrees)
            beam_angles_deg: Array of incident beam angles (degrees)
            wavelengths_nm: Array of beam wavelengths (nm)
            
        Returns:
            Combining efficiency (0-1)
        """
        # Convert to radians
        grating_angle_rad = np.radians(grating_angle_deg)
        beam_angles_rad = np.radians(beam_angles_deg)
        
        # Grating parameters
        d = 1.0 / self.grating_line_density  # grating period in mm
        m = 1  # First diffraction order
        
        # Calculate diffraction angles for each beam
        diffraction_angles_rad = np.zeros(self.n_channels)
        for i, wavelength_nm in enumerate(wavelengths_nm):
            wavelength_mm = wavelength_nm * 1e-6  # Convert nm to mm
            # Grating equation
            sin_theta_m = m * wavelength_mm / d - np.sin(beam_angles_rad[i])
            # Check if solution is valid (avoid evanescent orders)
            if abs(sin_theta_m) > 1.0:
                self.diffraction_losses[i] = 1.0
                diffraction_angles_rad[i] = 0.0
            else:
                self.diffraction_losses[i] = 0.0
                diffraction_angles_rad[i] = np.arcsin(sin_theta_m)
        
        # Calculate angular deviation from target diffraction angle
        target_angle_rad = grating_angle_rad
        angular_deviations_rad = diffraction_angles_rad - target_angle_rad
        
        # Calculate efficiency based on angular deviations
        # Using a Gaussian model for efficiency vs. angular error
        angular_tolerance_rad = np.radians(0.01)  # 0.01 degree tolerance
        efficiency_factors = np.exp(-(angular_deviations_rad / angular_tolerance_rad)**2)
        
        # Apply diffraction losses
        efficiency_factors *= (1.0 - self.diffraction_losses)
        
        # Overall efficiency is the product of individual efficiencies
        # weighted by relative power (assuming equal power for now)
        overall_efficiency = np.mean(efficiency_factors)
        
        # Store for diagnostics
        self.alignment_errors = np.degrees(angular_deviations_rad)
        self.current_efficiency = overall_efficiency
        
        return overall_efficiency
    
    def optimize(self, max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize grating angle and beam angles to maximize combining efficiency.
        
        Args:
            max_iterations: Maximum number of optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        # Define the objective function (negative efficiency to minimize)
        def objective(params):
            grating_angle = params[0]
            beam_angles = params[1:self.n_channels+1]
            return -self.calculate_efficiency(grating_angle, beam_angles, self.wavelengths_nm)
        
        # Initial parameters: [grating_angle, beam_angle_1, beam_angle_2, ...]
        initial_params = np.concatenate(([self.grating_angle_deg], self.beam_angles_deg))
        
        # Bounds for parameters
        # Grating angle: 30 to 60 degrees
        # Beam angles: -70 to 70 degrees
        bounds = [(30.0, 60.0)] + [(-70.0, 70.0)] * self.n_channels
        
        # Run optimization
        result = minimize(
            objective, 
            initial_params, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        self.grating_angle_deg = optimized_params[0]
        self.beam_angles_deg = optimized_params[1:self.n_channels+1]
        
        # Calculate final efficiency
        final_efficiency = self.calculate_efficiency(
            self.grating_angle_deg, 
            self.beam_angles_deg,
            self.wavelengths_nm
        )
        
        # Return results
        return {
            'optimized_grating_angle_deg': float(self.grating_angle_deg),
            'optimized_beam_angles_deg': self.beam_angles_deg.tolist(),
            'final_efficiency': float(final_efficiency),
            'alignment_errors_deg': self.alignment_errors.tolist(),
            'diffraction_losses': self.diffraction_losses.tolist(),
            'convergence_success': result.success,
            'iterations': result.nit
        }
    
    def optimize_wavelength_spacing(self, 
                                   min_spacing_nm: float = 0.5, 
                                   max_spacing_nm: float = 5.0) -> Dict[str, Any]:
        """
        Optimize wavelength spacing between channels to maximize combining efficiency.
        
        Args:
            min_spacing_nm: Minimum allowed wavelength spacing (nm)
            max_spacing_nm: Maximum allowed wavelength spacing (nm)
            
        Returns:
            Dictionary with optimization results
        """
        # Define the objective function
        def objective(spacings_nm):
            # Generate wavelengths from base wavelength and spacings
            base_wavelength = self.wavelengths_nm[0]
            wavelengths = np.zeros(self.n_channels)
            wavelengths[0] = base_wavelength
            for i in range(1, self.n_channels):
                wavelengths[i] = wavelengths[i-1] + spacings_nm[i-1]
            
            # Optimize angles for these wavelengths
            self.wavelengths_nm = wavelengths
            self._initialize_beam_angles()
            result = self.optimize(max_iterations=20)
            
            # Return negative efficiency (for minimization)
            return -result['final_efficiency']
        
        # Initial spacings (equal spacing)
        if self.n_channels > 1:
            initial_spacing = (self.wavelengths_nm[-1] - self.wavelengths_nm[0]) / (self.n_channels - 1)
            initial_spacings = np.ones(self.n_channels - 1) * initial_spacing
            
            # Bounds for spacings
            bounds = [(min_spacing_nm, max_spacing_nm)] * (self.n_channels - 1)
            
            # Run optimization
            result = minimize(
                objective, 
                initial_spacings, 
                method='L-BFGS-B',
                bounds=bounds
            )
            
            # Extract optimized spacings
            optimized_spacings = result.x
            
            # Generate final wavelengths
            base_wavelength = self.wavelengths_nm[0]
            optimized_wavelengths = np.zeros(self.n_channels)
            optimized_wavelengths[0] = base_wavelength
            for i in range(1, self.n_channels):
                optimized_wavelengths[i] = optimized_wavelengths[i-1] + optimized_spacings[i-1]
            
            # Update wavelengths and re-optimize angles
            self.wavelengths_nm = optimized_wavelengths
            self._initialize_beam_angles()
            angle_result = self.optimize()
            
            return {
                'optimized_wavelengths_nm': self.wavelengths_nm.tolist(),
                'optimized_spacings_nm': optimized_spacings.tolist(),
                'optimized_grating_angle_deg': angle_result['optimized_grating_angle_deg'],
                'optimized_beam_angles_deg': angle_result['optimized_beam_angles_deg'],
                'final_efficiency': angle_result['final_efficiency'],
                'convergence_success': result.success
            }
        else:
            # Only one channel, no spacing to optimize
            return {
                'optimized_wavelengths_nm': self.wavelengths_nm.tolist(),
                'optimized_spacings_nm': [],
                'optimized_grating_angle_deg': self.grating_angle_deg,
                'optimized_beam_angles_deg': self.beam_angles_deg.tolist(),
                'final_efficiency': self.current_efficiency,
                'convergence_success': True
            }