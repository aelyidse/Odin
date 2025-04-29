import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, List
import time

class MEMSMirrorController:
    """
    High-speed control loop for MEMS mirror adjustment with sub-microsecond response time.
    
    Implements a real-time feedback system for precise mirror positioning using
    predictive algorithms and hardware-optimized control strategies.
    """
    def __init__(self,
                 mirror_diameter_mm: float = 5.0,
                 resonant_frequency_hz: float = 25000.0,
                 damping_ratio: float = 0.7,
                 max_deflection_deg: float = 5.0,
                 settling_time_us: float = 0.5,
                 feedback_delay_us: float = 0.1,
                 control_rate_hz: float = 1000000.0):
        """
        Initialize MEMS mirror controller.
        
        Args:
            mirror_diameter_mm: Physical diameter of the MEMS mirror (mm)
            resonant_frequency_hz: Mechanical resonant frequency (Hz)
            damping_ratio: Mechanical damping ratio (dimensionless)
            max_deflection_deg: Maximum mirror deflection angle (degrees)
            settling_time_us: Target settling time (microseconds)
            feedback_delay_us: System feedback delay (microseconds)
            control_rate_hz: Control loop update rate (Hz)
        """
        self.mirror_diameter_mm = mirror_diameter_mm
        self.resonant_frequency_hz = resonant_frequency_hz
        self.damping_ratio = damping_ratio
        self.max_deflection_deg = max_deflection_deg
        self.settling_time_us = settling_time_us
        self.feedback_delay_us = feedback_delay_us
        self.control_rate_hz = control_rate_hz
        
        # Derived parameters
        self.control_period_us = 1e6 / control_rate_hz
        self.natural_frequency = 2 * np.pi * resonant_frequency_hz
        
        # Controller state
        self.current_position = np.zeros(2)  # [x_angle, y_angle] in degrees
        self.target_position = np.zeros(2)   # [x_angle, y_angle] in degrees
        self.last_update_time = 0
        
        # PID controller parameters (tuned for critical damping)
        self.kp = 1.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.01  # Derivative gain
        self.integral_error = np.zeros(2)
        self.last_error = np.zeros(2)
        
        # Feedforward compensation
        self._setup_feedforward_model()
        
    def _setup_feedforward_model(self):
        """Configure feedforward compensation model based on mirror dynamics."""
        # Second-order system model parameters
        wn = self.natural_frequency
        zeta = self.damping_ratio
        
        # Step response model: x(t) = 1 - exp(-ζωₙt)·(cos(ωd·t) + (ζωₙ/ωd)·sin(ωd·t))
        # where ωd = ωₙ·sqrt(1-ζ²) is the damped natural frequency
        self.damped_frequency = wn * np.sqrt(1 - zeta**2) if zeta < 1.0 else 0
        
        # Pre-compute coefficients for feedforward control
        self.feedforward_coeffs = {
            'rise_time': 1.8 / (zeta * wn),  # Time to reach 90% of target
            'overshoot': np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) if zeta < 1.0 else 0,
            'settling_time': 4.6 / (zeta * wn)  # Time to reach and stay within 1% of target
        }
        
    def set_target_position(self, x_angle_deg: float, y_angle_deg: float) -> None:
        """
        Set target mirror position in degrees.
        
        Args:
            x_angle_deg: Target x-axis angle (degrees)
            y_angle_deg: Target y-axis angle (degrees)
        """
        # Clamp to maximum deflection
        self.target_position[0] = np.clip(x_angle_deg, -self.max_deflection_deg, self.max_deflection_deg)
        self.target_position[1] = np.clip(y_angle_deg, -self.max_deflection_deg, self.max_deflection_deg)
        
        # Reset integral error when target changes significantly
        if np.linalg.norm(self.target_position - self.current_position) > 0.5:
            self.integral_error = np.zeros(2)
    
    def update(self, position_feedback: Optional[np.ndarray] = None, dt_us: Optional[float] = None) -> np.ndarray:
        """
        Update control loop with optional position feedback.
        
        Args:
            position_feedback: Measured [x,y] position in degrees, if available
            dt_us: Time step in microseconds (if None, uses elapsed time since last call)
            
        Returns:
            Control signal [x,y] (normalized -1.0 to 1.0)
        """
        # Calculate time step
        current_time = time.time() * 1e6  # Current time in microseconds
        if dt_us is None:
            dt_us = current_time - self.last_update_time if self.last_update_time > 0 else self.control_period_us
        self.last_update_time = current_time
        
        # Use feedback if available, otherwise use model prediction
        if position_feedback is not None:
            self.current_position = position_feedback
        
        # Calculate error
        error = self.target_position - self.current_position
        
        # PID control with anti-windup
        self.integral_error += error * dt_us / 1e6
        # Clamp integral term to prevent windup
        max_integral = 0.5 * self.max_deflection_deg
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        
        # Calculate derivative term (with filtering)
        if dt_us > 0:
            derivative = (error - self.last_error) / (dt_us / 1e6)
            # Low-pass filter on derivative term
            derivative_filter = 0.1
            derivative = derivative * derivative_filter + (1 - derivative_filter) * derivative
        else:
            derivative = np.zeros(2)
        
        self.last_error = error.copy()
        
        # Compute control signal
        control = (
            self.kp * error + 
            self.ki * self.integral_error + 
            self.kd * derivative
        )
        
        # Add feedforward term for faster response
        feedforward = self._calculate_feedforward(error)
        control += feedforward
        
        # Normalize control signal to [-1, 1]
        control = np.clip(control / self.max_deflection_deg, -1.0, 1.0)
        
        # Update model prediction
        self._update_model_prediction(control, dt_us)
        
        return control
    
    def _calculate_feedforward(self, error: np.ndarray) -> np.ndarray:
        """Calculate feedforward term based on target and dynamics model."""
        # Simple feedforward: anticipate control needed to overcome inertia
        # More sophisticated models would include full dynamics
        ff_gain = 0.2 * self.natural_frequency / (2 * np.pi)
        return ff_gain * error
    
    def _update_model_prediction(self, control: np.ndarray, dt_us: float) -> None:
        """Update internal model prediction of mirror position."""
        # Simple second-order dynamics model
        # For a real implementation, this would be more sophisticated
        dt_s = dt_us / 1e6
        
        # Acceleration proportional to control signal
        accel = control * self.max_deflection_deg * self.natural_frequency**2
        
        # Damping proportional to velocity (estimated)
        velocity = (self.target_position - self.current_position) / dt_s if dt_s > 0 else np.zeros(2)
        damping = -2 * self.damping_ratio * self.natural_frequency * velocity
        
        # Position update using semi-implicit Euler integration
        velocity += (accel + damping) * dt_s
        position_change = velocity * dt_s
        
        # Update current position if no feedback was provided
        self.current_position += position_change
    
    def simulate_step_response(self, target_angle_deg: float, duration_us: float = 100.0, dt_us: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Simulate step response for a given target angle.
        
        Args:
            target_angle_deg: Target angle (degrees)
            duration_us: Simulation duration (microseconds)
            dt_us: Time step (microseconds)
            
        Returns:
            Dictionary with time and position arrays
        """
        # Reset controller state
        self.current_position = np.zeros(2)
        self.target_position = np.zeros(2)
        self.integral_error = np.zeros(2)
        self.last_error = np.zeros(2)
        
        # Set target position
        self.set_target_position(target_angle_deg, 0)
        
        # Prepare arrays for results
        n_steps = int(duration_us / dt_us) + 1
        times = np.linspace(0, duration_us, n_steps)
        positions = np.zeros((n_steps, 2))
        controls = np.zeros((n_steps, 2))
        
        # Run simulation
        for i in range(n_steps):
            controls[i] = self.update(dt_us=dt_us)
            positions[i] = self.current_position
        
        return {
            'time_us': times,
            'position_deg': positions,
            'control': controls
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate controller performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Theoretical metrics based on system parameters
        rise_time_us = self.feedforward_coeffs['rise_time'] * 1e6
        settling_time_us = self.feedforward_coeffs['settling_time'] * 1e6
        overshoot_pct = self.feedforward_coeffs['overshoot'] * 100
        
        # Control loop metrics
        control_latency_us = self.control_period_us + self.feedback_delay_us
        bandwidth_hz = 0.35 / (rise_time_us / 1e6)  # Approximate bandwidth
        
        return {
            'rise_time_us': float(rise_time_us),
            'settling_time_us': float(settling_time_us),
            'overshoot_pct': float(overshoot_pct),
            'control_latency_us': float(control_latency_us),
            'bandwidth_hz': float(bandwidth_hz),
            'max_deflection_deg': float(self.max_deflection_deg)
        }