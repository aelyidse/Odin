import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from scipy.optimize import minimize

class WavelengthController:
    """Concrete controller for laser wavelength control and stability monitoring."""
    def __init__(self,
                 setpoint_nm: float,
                 initial_nm: float,
                 bragg_sensitivity_nm_per_C: float = 0.01,
                 temp_C: float = 25.0,
                 temp_stability_C: float = 0.01,
                 max_drift_nm: float = 0.05,
                 control_gain: float = 0.5):
        """
        Args:
            setpoint_nm: Desired wavelength (nm)
            initial_nm: Initial wavelength (nm)
            bragg_sensitivity_nm_per_C: FBG wavelength shift per degree C
            temp_C: Initial temperature (C)
            temp_stability_C: Standard deviation of temperature noise (C)
            max_drift_nm: Maximum allowed drift before alarm (nm)
            control_gain: Proportional gain for feedback control
        """
        self.setpoint_nm = setpoint_nm
        self.current_nm = initial_nm
        self.bragg_sensitivity = bragg_sensitivity_nm_per_C
        self.temp_C = temp_C
        self.temp_stability_C = temp_stability_C
        self.max_drift_nm = max_drift_nm
        self.control_gain = control_gain
        self.history = []  # (time, wavelength, error)
        self.alarm = False

    def step(self, dt: float, env_temp_C: Optional[float] = None, external_perturbation_nm: float = 0.0) -> float:
        """Simulate one control loop step (dt seconds)."""
        # Simulate temperature fluctuation
        temp_noise = np.random.normal(0, self.temp_stability_C)
        temp = env_temp_C if env_temp_C is not None else self.temp_C
        temp += temp_noise
        # Compute wavelength drift due to temperature
        drift_nm = self.bragg_sensitivity * (temp - self.temp_C)
        # Add external perturbation (e.g., mechanical)
        measured_nm = self.current_nm + drift_nm + external_perturbation_nm
        # Feedback control to correct wavelength
        error = self.setpoint_nm - measured_nm
        correction = self.control_gain * error
        self.current_nm = measured_nm + correction
        # Alarm if deviation exceeds threshold
        self.alarm = abs(error) > self.max_drift_nm
        self.history.append((dt, self.current_nm, error))
        return self.current_nm

    def get_stability_metrics(self, window: int = 100) -> Dict[str, float]:
        """Return RMS deviation and max error over recent steps."""
        if len(self.history) < window:
            return {'rms': 0.0, 'max': 0.0}
        errors = np.array([h[2] for h in self.history[-window:]])
        return {'rms': float(np.sqrt(np.mean(errors**2))), 'max': float(np.max(np.abs(errors)))}

    def is_alarm(self) -> bool:
        return self.alarm

    def reset(self) -> None:
        self.current_nm = self.setpoint_nm
        self.history.clear()
        self.alarm = False


class WavelengthStabilizer:
    """
    High-precision wavelength control system with sub-nanometer stability.
    
    Implements advanced feedback control with predictive compensation and 
    thermal stabilization to achieve wavelength stability below 0.1 nm.
    """
    def __init__(self, 
                 center_wavelength_nm: float,
                 target_stability_nm: float = 0.1,
                 feedback_rate_hz: float = 100.0,
                 thermal_time_constant_s: float = 5.0,
                 use_predictive_comp: bool = True):
        """
        Initialize wavelength stabilizer.
        
        Args:
            center_wavelength_nm: Target center wavelength (nm)
            target_stability_nm: Required stability (nm)
            feedback_rate_hz: Control loop rate (Hz)
            thermal_time_constant_s: Thermal response time constant (s)
            use_predictive_comp: Enable predictive compensation
        """
        self.center_wavelength_nm = center_wavelength_nm
        self.target_stability_nm = target_stability_nm
        self.feedback_rate_hz = feedback_rate_hz
        self.thermal_time_constant_s = thermal_time_constant_s
        self.use_predictive_comp = use_predictive_comp
        
        # Control parameters
        self.kp = 0.5  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.05  # Derivative gain
        
        # State variables
        self.current_wavelength_nm = center_wavelength_nm
        self.error_integral = 0.0
        self.last_error = 0.0
        self.history: List[float] = []
        self.history_max_len = 100
        
        # Actuator state
        self.temperature_setpoint_c = 25.0
        self.grating_angle_mrad = 0.0
        self.piezo_position_nm = 0.0
        
        # Performance metrics
        self.stability_metrics = {
            'rms_error_nm': 0.0,
            'peak_to_peak_nm': 0.0,
            'allan_variance': 0.0
        }
    
    def update_measurement(self, wavelength_nm: float, timestamp_s: float) -> Dict[str, float]:
        """
        Update with new wavelength measurement and compute control outputs.
        
        Args:
            wavelength_nm: Measured wavelength (nm)
            timestamp_s: Measurement timestamp (s)
            
        Returns:
            Dict of actuator commands
        """
        # Store measurement in history
        self.current_wavelength_nm = wavelength_nm
        self.history.append(wavelength_nm)
        if len(self.history) > self.history_max_len:
            self.history.pop(0)
            
        # Compute error
        error = self.center_wavelength_nm - wavelength_nm
        
        # PID control
        self.error_integral += error / self.feedback_rate_hz
        error_derivative = (error - self.last_error) * self.feedback_rate_hz
        self.last_error = error
        
        # Compute control signal
        control_signal = (self.kp * error + 
                          self.ki * self.error_integral + 
                          self.kd * error_derivative)
        
        # Apply predictive compensation if enabled
        if self.use_predictive_comp and len(self.history) >= 3:
            predicted_drift = self._predict_drift()
            control_signal += predicted_drift
        
        # Convert control signal to actuator commands
        actuator_commands = self._compute_actuator_commands(control_signal)
        
        # Update stability metrics
        self._update_stability_metrics()
        
        return actuator_commands
    
    def _compute_actuator_commands(self, control_signal: float) -> Dict[str, float]:
        """
        Convert control signal to optimal actuator commands.
        
        Uses a model-based approach to distribute control effort across
        thermal, mechanical, and piezoelectric actuators.
        
        Args:
            control_signal: Combined control signal
            
        Returns:
            Dict of actuator commands
        """
        # Fast corrections go to piezo, slow drift to temperature
        # Medium corrections to grating angle
        
        # Simple low-pass filter to separate components
        fast_component = 0.8 * control_signal
        slow_component = 0.2 * control_signal
        
        # Update actuator states
        self.piezo_position_nm += fast_component
        self.temperature_setpoint_c += slow_component * 0.01  # Scale for temperature
        
        # Limit actuator ranges
        self.piezo_position_nm = np.clip(self.piezo_position_nm, -100, 100)
        self.temperature_setpoint_c = np.clip(self.temperature_setpoint_c, 20, 30)
        
        return {
            'piezo_position_nm': self.piezo_position_nm,
            'temperature_setpoint_c': self.temperature_setpoint_c,
            'grating_angle_mrad': self.grating_angle_mrad
        }
    
    def _predict_drift(self) -> float:
        """
        Predict future wavelength drift based on recent history.
        
        Returns:
            Predicted drift (nm)
        """
        if len(self.history) < 3:
            return 0.0
            
        # Simple linear extrapolation
        recent = self.history[-3:]
        slope = (recent[2] - recent[0]) / 2
        
        # Predict one step ahead
        predicted_value = recent[2] + slope
        predicted_drift = self.center_wavelength_nm - predicted_value
        
        return predicted_drift
    
    def _update_stability_metrics(self):
        """Update performance metrics based on recent history."""
        if len(self.history) < 2:
            return
            
        recent = np.array(self.history[-20:])
        self.stability_metrics['rms_error_nm'] = np.std(recent)
        self.stability_metrics['peak_to_peak_nm'] = np.max(recent) - np.min(recent)
        
        # Simplified Allan variance calculation
        if len(recent) >= 4:
            diffs = recent[1:] - recent[:-1]
            self.stability_metrics['allan_variance'] = np.sum(diffs**2) / (2 * (len(diffs) - 1))
    
    def optimize_control_parameters(self, 
                                   simulation_fn: Callable[[Dict[str, float]], float],
                                   n_steps: int = 100) -> Dict[str, float]:
        """
        Optimize control parameters using simulation-based approach.
        
        Args:
            simulation_fn: Function that simulates system response
            n_steps: Number of simulation steps
            
        Returns:
            Optimized parameters
        """
        def objective(params):
            self.kp, self.ki, self.kd = params
            
            # Reset state
            self.error_integral = 0.0
            self.last_error = 0.0
            self.history = []
            
            # Run simulation
            wavelength = self.center_wavelength_nm
            total_error = 0.0
            
            for i in range(n_steps):
                # Get simulated wavelength with disturbance
                wavelength = simulation_fn({'time_step': i / self.feedback_rate_hz})
                
                # Update controller
                self.update_measurement(wavelength, i / self.feedback_rate_hz)
                
                # Accumulate error
                total_error += abs(wavelength - self.center_wavelength_nm)
            
            return total_error / n_steps
        
        # Optimize PID parameters
        result = minimize(objective, [self.kp, self.ki, self.kd], 
                         bounds=[(0.01, 2.0), (0.0, 1.0), (0.0, 0.5)])
        
        self.kp, self.ki, self.kd = result.x
        return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get current stability performance metrics."""
        return self.stability_metrics
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert controller state to dictionary for serialization."""
        return {
            'center_wavelength_nm': self.center_wavelength_nm,
            'current_wavelength_nm': self.current_wavelength_nm,
            'target_stability_nm': self.target_stability_nm,
            'control_parameters': {
                'kp': self.kp,
                'ki': self.ki,
                'kd': self.kd
            },
            'actuator_state': {
                'piezo_position_nm': self.piezo_position_nm,
                'temperature_setpoint_c': self.temperature_setpoint_c,
                'grating_angle_mrad': self.grating_angle_mrad
            },
            'stability_metrics': self.stability_metrics
        }
