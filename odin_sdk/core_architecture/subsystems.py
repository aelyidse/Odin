from abc import abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from .component import Component

class LaserUnit(Component[np.ndarray]):
    """Abstract base class for laser unit subsystems in ODIN SDK."""
    
    @abstractmethod
    def set_power_level(self, power_watts: float) -> bool:
        """Set the output power level of the laser unit.
        
        Args:
            power_watts: Desired power output in watts
        Returns:
            True if power level was set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_wavelength(self) -> float:
        """Get the current operating wavelength of the laser.
        
        Returns:
            Wavelength in nanometers
        """
        pass
    
    @abstractmethod
    def adjust_fiber_tip(self, x_offset: float, y_offset: float) -> bool:
        """Adjust the fiber tip position for beam direction control.
        
        Args:
            x_offset: Displacement in x-axis (micrometers)
            y_offset: Displacement in y-axis (micrometers)
        Returns:
            True if adjustment successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_thermal_status(self) -> Dict[str, float]:
        """Get thermal status of critical laser components.
        
        Returns:
            Dictionary with component names and their temperatures in Celsius
        """
        pass

class BeamCombining(Component[np.ndarray]):
    """Abstract base class for beam combining subsystems in ODIN SDK."""
    
    @abstractmethod
    def add_laser_source(self, laser_unit: LaserUnit, wavelength: float, angle: float) -> bool:
        """Add a laser source to the beam combining system.
        
        Args:
            laser_unit: Laser unit component instance
            wavelength: Operating wavelength of the laser in nanometers
            angle: Incident angle on diffraction grating in degrees
        Returns:
            True if source added successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_combining_efficiency(self) -> float:
        """Get the current beam combining efficiency.
        
        Returns:
            Efficiency as a percentage (0-100)
        """
        pass
    
    @abstractmethod
    def adjust_grating_angle(self, angle_delta: float) -> bool:
        """Adjust the diffraction grating angle to optimize combining.
        
        Args:
            angle_delta: Angle adjustment in millidegrees
        Returns:
            True if adjustment successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_spectral_profile(self) -> List[Tuple[float, float]]:
        """Get the spectral profile of the combined beam.
        
        Returns:
            List of (wavelength, power) tuples for spectral components
        """
        pass

class ControlSystem(Component[Dict[str, float]]):
    """Abstract base class for control subsystems in ODIN SDK."""
    
    @abstractmethod
    def register_component(self, component: Component, dependencies: Optional[List[Component]] = None) -> bool:
        """Register a component with the control system.
        
        Args:
            component: Component to register
            dependencies: Optional list of components this one depends on
        Returns:
            True if registration successful, False otherwise
        """
        pass
    
    @abstractmethod
    def set_control_parameters(self, component_name: str, parameters: Dict[str, float]) -> bool:
        """Set control parameters for a specific component.
        
        Args:
            component_name: Name of component to configure
            parameters: Dictionary of parameter names and values
        Returns:
            True if parameters set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Dict[str, float]]:
        """Get status metrics for all controlled components.
        
        Returns:
            Nested dictionary of component names and their status metrics
        """
        pass
    
    @abstractmethod
    def execute_control_loop(self, delta_time: float) -> bool:
        """Execute one iteration of the control loop.
        
        Args:
            delta_time: Time since last control loop in seconds
        Returns:
            True if control loop executed successfully, False otherwise
        """
        pass
