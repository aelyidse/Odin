from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class FiberOpticsModel(ABC):
    """Abstract base class for fiber optics simulation with polarization modeling."""
    @abstractmethod
    def propagate(self, field: np.ndarray, length: float, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate a field through the fiber, returning final field and polarization state.
        Args:
            field: Input field (Jones vector or Stokes parameters)
            length: Fiber length in meters
            params: Dictionary of fiber and environmental parameters
        Returns:
            Tuple of (output field, output polarization state)
        """
        pass

class PMFiberModel(FiberOpticsModel):
    """Polarization-maintaining fiber model (e.g., PANDA fiber)."""
    @abstractmethod
    def propagate(self, field: np.ndarray, length: float, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        pass

class BirefringenceModel(FiberOpticsModel):
    """Model for birefringence and polarization evolution in fiber."""
    @abstractmethod
    def propagate(self, field: np.ndarray, length: float, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        pass

class PolarizationDependentLossModel(FiberOpticsModel):
    """Model for polarization-dependent loss and extinction ratio."""
    @abstractmethod
    def propagate(self, field: np.ndarray, length: float, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        pass

class FiberBraggGratingModel(FiberOpticsModel):
    """Model for fiber Bragg grating effects on polarization and wavelength selection."""
    @abstractmethod
    def propagate(self, field: np.ndarray, length: float, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        pass
