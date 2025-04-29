import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from scipy.optimize import minimize

from .thermal_wavelength_model import ThermalWavelengthModel
from .wavelength_control import WavelengthController

class FBGTemperatureController:
    """
    Fiber Bragg Grating (FBG) temperature control system with high precision wavelength stability.
    
    Implements closed-loop temperature control to maintain FBG wavelength stability
    within sub-nanometer precision for optical sensing and communication applications.
    """
    def __init__(self, 
                 center_wavelength_nm: float,
                 bragg_sensitivity_nm_per_C: float = 0.01,
                 initial_temp_C: float = 25.0,
                 target_stability_nm: float = 0.01,
                 control_interval_s: float = 0.1,
                 thermal_time_constant_s: float = 2.0,
                 max_temp_C: float = 80.0,
                 min_temp_C: float = 15.0):
        """
        Initialize FBG temperature controller.
        
        Args:
            center_wavelength_nm: Target center wavelength (nm)
            bragg_sensitivity_nm_per_C: FBG wavelength shift per degree C
            initial_temp_C: Initial temperature (C)
            target_stability_nm: Required wavelength stability (nm)
            control_interval_s: Control loop interval (s)
            thermal_time_constant_s: Thermal response time constant (s)
            max_temp_C: Maximum allowed temperature (C)
            min_temp_C: Minimum allowed temperature (C)
        """
        self.center_wavelength_nm = center_wavelength_nm
        self.bragg_sensitivity = bragg_sensitivity_nm_per_C
        self.current_temp_C = initial_temp_C
        self.target_stability_nm = target_stability_nm
        self.control_interval_s = control_interval_s
        self.thermal_time_constant_s = thermal_time_constant_s
        self.max_temp_C = max_temp_C
        self.min_temp_C = min_temp_C
        
        # Initialize thermal wavelength model
        self.thermal_model = ThermalWavelengthModel(
            center_wavelength_nm=center_wavelength_nm,
            reference_temp_c=initial_temp_C
        )
        
        # Initialize wavelength controller
        self.wavelength_controller = WavelengthController(
            setpoint_nm=center_wavelength_nm,
            initial_nm=center_wavelength_nm,
            bragg_sensitivity_nm_per_C=bragg_sensitivity_nm_per_C,
            temp_C=initial_temp_C,
            max_drift_nm=target_stability_nm
        )
        
        # PID controller parameters
        self.kp = 2.0  # Proportional gain
        self.ki = 0.5  # Integral gain
        self.kd = 0.1  # Derivative gain
        
        # Controller state
        self.setpoint_temp_C = initial_temp_C
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_update_time = time.time()
        
        # Performance tracking
        self.history: List[Dict[str, float]] = []
        self.max_history_length = 1000
        
    def set_target_wavelength(self, wavelength_nm: float) -> None:
        """
        Set target wavelength and calculate required temperature.
        
        Args:
            wavelength_nm: Target wavelength (nm)
        """
        self.center_wavelength_nm = wavelength_nm
        self.wavelength_controller.setpoint_nm = wavelength_nm
        
        # Calculate required temperature using thermal model
        self.setpoint_temp_C = self.thermal_model.required_temperature(wavelength_nm)
        self.setpoint_temp_C = np.clip(self.setpoint_temp_C, self.min_temp_C, self.max_temp_C)
        
        # Reset controller state
        self.error_integral = 0.0
        self.last_error = 0.0
        
    def update(self, measured_wavelength_nm: Optional[float] = None, 
               measured_temp_C: Optional[float] = None) -> Dict[str, float]:
        """
        Update control loop with measurements and compute heater/cooler output.
        
        Args:
            measured_wavelength_nm: Measured FBG wavelength (nm), if available
            measured_temp_C: Measured temperature (C), if available
            
        Returns:
            Dict with control outputs and status
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Use measured values if available, otherwise use model predictions
        current_temp = measured_temp_C if measured_temp_C is not None else self.current_temp_C
        
        if measured_wavelength_nm is not None:
            current_wavelength = measured_wavelength_nm
            # Update wavelength controller with measurement
            self.wavelength_controller.step(dt, current_temp, 0.0)
        else:
            # Predict wavelength from temperature
            current_wavelength = self.thermal_model.predict_wavelength(current_temp)
        
        # Calculate temperature error
        error = self.setpoint_temp_C - current_temp
        
        # PID control
        if dt > 0:
            # Update integral term with anti-windup
            self.error_integral += error * dt
            self.error_integral = np.clip(self.error_integral, -10.0, 10.0)
            
            # Calculate derivative term
            derivative = (error - self.last_error) / dt if dt > 0 else 0.0
            self.last_error = error
            
            # Calculate control output
            control_output = (
                self.kp * error + 
                self.ki * self.error_integral + 
                self.kd * derivative
            )
        else:
            control_output = self.kp * error
        
        # Clip control output to valid range (-1 to 1)
        # Negative values activate cooling, positive values activate heating
        control_output = np.clip(control_output, -1.0, 1.0)
        
        # Calculate predicted temperature change
        if control_output > 0:
            # Heating
            max_heating_rate_C_per_s = 2.0  # Maximum heating rate
            temp_change = control_output * max_heating_rate_C_per_s * self.control_interval_s
        else:
            # Cooling
            max_cooling_rate_C_per_s = 1.0  # Maximum cooling rate
            temp_change = control_output * max_cooling_rate_C_per_s * self.control_interval_s
        
        # Update current temperature (simulated or real system would use actual measurements)
        self.current_temp_C = current_temp + temp_change
        self.current_temp_C = np.clip(self.current_temp_C, self.min_temp_C, self.max_temp_C)
        
        # Calculate wavelength error
        wavelength_error = self.center_wavelength_nm - current_wavelength
        
        # Record state
        state = {
            'timestamp': current_time,
            'temperature_C': self.current_temp_C,
            'setpoint_temp_C': self.setpoint_temp_C,
            'temp_error_C': error,
            'wavelength_nm': current_wavelength,
            'setpoint_wavelength_nm': self.center_wavelength_nm,
            'wavelength_error_nm': wavelength_error,
            'control_output': control_output
        }
        
        self.history.append(state)
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
        
        return state
    
    def get_stability_metrics(self, window: int = 100) -> Dict[str, float]:
        """
        Calculate stability metrics over recent history.
        
        Args:
            window: Number of recent measurements to consider
            
        Returns:
            Dict with stability metrics
        """
        if not self.history or len(self.history) < 2:
            return {
                'temp_stability_C': 0.0,
                'wavelength_stability_nm': 0.0,
                'mean_temp_C': self.current_temp_C,
                'mean_wavelength_nm': self.center_wavelength_nm
            }
        
        # Get recent history
        recent = self.history[-window:] if len(self.history) > window else self.history
        
        # Extract data
        temperatures = [entry['temperature_C'] for entry in recent]
        wavelengths = [entry['wavelength_nm'] for entry in recent]
        
        # Calculate metrics
        temp_stability = float(np.std(temperatures))
        wavelength_stability = float(np.std(wavelengths))
        mean_temp = float(np.mean(temperatures))
        mean_wavelength = float(np.mean(wavelengths))
        
        return {
            'temp_stability_C': temp_stability,
            'wavelength_stability_nm': wavelength_stability,
            'mean_temp_C': mean_temp,
            'mean_wavelength_nm': mean_wavelength
        }
    
    def is_stable(self) -> bool:
        """Check if wavelength is stable within target stability."""
        metrics = self.get_stability_metrics()
        return metrics['wavelength_stability_nm'] <= self.target_stability_nm