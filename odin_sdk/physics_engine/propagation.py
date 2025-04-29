from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class LaserPropagationModel(ABC):
    """Abstract base class for laser propagation models in various media."""
    @abstractmethod
    def propagate(self, beam: np.ndarray, distance: float, params: Dict[str, Any]) -> np.ndarray:
        """Propagate a laser beam through a given distance with specified parameters.
        Args:
            beam: Initial beam field (complex amplitude or intensity distribution)
            distance: Propagation distance in meters
            params: Dictionary of propagation/environment parameters
        Returns:
            Propagated beam field
        """
        pass

class FreeSpacePropagation(LaserPropagationModel):
    """Model for free-space Gaussian beam propagation."""
    @abstractmethod
    def propagate(self, beam: np.ndarray, distance: float, params: Dict[str, Any]) -> np.ndarray:
        pass

class AtmosphericAttenuation(LaserPropagationModel):
    """Model for atmospheric absorption and scattering losses (Beer-Lambert law, Rayleigh/Mie)."""
    @abstractmethod
    def propagate(self, beam: np.ndarray, distance: float, params: Dict[str, Any]) -> np.ndarray:
        pass

class TurbulencePropagation(LaserPropagationModel):
    """Model for random phase screen turbulence effects (using Cn^2, Kolmogorov spectrum)."""
    
    def __init__(self, turbulence_model=None, compensator=None):
        """
        Initialize turbulence propagation model.
        
        Args:
            turbulence_model: Model for generating turbulence phase screens
            compensator: Optional TurbulenceCompensator for AO correction
        """
        self.turbulence_model = turbulence_model
        self.compensator = compensator
        self.prev_screens = None
        self.accumulated_time = 0.0
    
    def propagate(self, beam: np.ndarray, distance: float, params: Dict[str, Any]) -> np.ndarray:
        """
        Propagate beam through turbulent atmosphere with optional compensation.
        
        Args:
            beam: Input complex beam amplitude
            distance: Propagation distance in meters
            params: Dictionary of parameters including:
                - dt_s: Time step since last propagation (for frozen flow)
                - cn2: Refractive index structure parameter
                - compensate: Whether to apply AO compensation
                - wavelength: Beam wavelength in meters
                
        Returns:
            Propagated complex beam amplitude
        """
        # Get time step for frozen flow
        dt_s = params.get('dt_s', 0.0)
        self.accumulated_time += dt_s
        
        # Generate or update turbulence phase screens
        if self.turbulence_model:
            if isinstance(self.turbulence_model, MultiLayerTurbulence):
                phase_screen, self.prev_screens = self.turbulence_model.generate_combined_phase_screen(
                    dt_s=dt_s, prev_screens=self.prev_screens)
            else:
                # Single layer turbulence
                phase_screen = self.turbulence_model.generate_phase_screen()
        else:
            # Create simple phase screen based on Cn2 if no model provided
            cn2 = params.get('cn2', 1e-14)  # Default Cn2 value
            wavelength = params.get('wavelength', 1.0e-6)  # Default wavelength (1 μm)
            r0 = 0.185 * wavelength**(6/5) * (cn2 * distance)**(-3/5)  # Fried parameter
            
            # Simple phase screen generation
            grid_size = beam.shape[0]
            phase_screen = np.random.normal(0, 1, beam.shape) * (wavelength / (2*np.pi*r0))
        
        # Apply compensation if requested and compensator available
        if params.get('compensate', False) and self.compensator:
            # Get wavefront measurement (with sensing error if specified)
            sensing_error = params.get('sensing_error', 0.0)
            if sensing_error > 0:
                sensed_phase = phase_screen + np.random.normal(0, sensing_error, phase_screen.shape)
            else:
                sensed_phase = phase_screen
                
            # Apply compensation
            correction = self.compensator.phase_conjugation(sensed_phase)
            
            # Apply correction with specified efficiency
            efficiency = params.get('correction_efficiency', 1.0)
            phase_screen = phase_screen + efficiency * correction
            
            # Store metrics if requested
            if 'metrics' in params:
                params['metrics']['strehl'] = self.compensator.strehl_ratio(phase_screen)
                params['metrics']['rms_wavefront'] = np.std(phase_screen)
        
        # Apply phase screen to beam
        phase_term = np.exp(1j * phase_screen)
        beam_after_turbulence = beam * phase_term
        
        # Propagate through free space after turbulence
        # This would typically call a free space propagation method
        # For simplicity, we'll just return the beam with turbulence applied
        return beam_after_turbulence

class ThermalBloomingPropagation(LaserPropagationModel):
    """Model for nonlinear propagation due to thermal blooming (air heating, refractive index change)."""
    @abstractmethod
    def propagate(self, beam: np.ndarray, distance: float, params: Dict[str, Any]) -> np.ndarray:
        pass

class MultiLayerAtmospherePropagation(LaserPropagationModel):
    """Model for multi-layer atmosphere with altitude-dependent parameters."""
    @abstractmethod
    def propagate(self, beam: np.ndarray, distance: float, params: Dict[str, Any]) -> np.ndarray:
        pass
