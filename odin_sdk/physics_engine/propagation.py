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
    @abstractmethod
    def propagate(self, beam: np.ndarray, distance: float, params: Dict[str, Any]) -> np.ndarray:
        pass

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
