import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class BeamJitterCompensator:
    """
    Real-time beam jitter compensation algorithm that detects and corrects for 
    beam pointing instabilities using feedback control.
    """
    def __init__(self, 
                 sampling_rate_hz: float = 1000.0,
                 control_bandwidth_hz: float = 100.0,
                 max_correction_urad: float = 500.0,
                 filter_length: int = 10):
        """
        Initialize beam jitter compensator.
        
        Args:
            sampling_rate_hz: Sampling rate of the beam position sensor (Hz)
            control_bandwidth_hz: Control loop bandwidth (Hz)
            max_correction_urad: Maximum correction angle (microradians)
            filter_length: Length of moving average filter for jitter estimation
        """
        self.sampling_rate_hz = sampling_rate_hz
        self.control_bandwidth_hz = control_bandwidth_hz
        self.max_correction_urad = max_correction_urad
        self.filter_length = filter_length
        
        # Internal state
        self.position_history = []  # Recent beam position measurements
        self.correction_angles = np.zeros(2)  # Current x,y correction angles (urad)
        self.reference_position = np.zeros(2)  # Target beam position
        
        # Control parameters
        self.proportional_gain = 0.7
        self.integral_gain = 0.3
        self.derivative_gain = 0.05
        
        # Error accumulation for integral term
        self.error_integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        
        # Performance metrics
        self.jitter_before_compensation = 0.0
        self.jitter_after_compensation = 0.0
        self.correction_history = []
    
    def set_reference_position(self, position: np.ndarray):
        """Set the target beam position."""
        self.reference_position = position.copy()
        
    def update(self, measured_position: np.ndarray) -> np.ndarray:
        """
        Process new beam position measurement and calculate correction.
        
        Args:
            measured_position: Current beam position [x, y]
            
        Returns:
            Correction angles [x_angle, y_angle] in microradians
        """
        # Store position in history
        self.position_history.append(measured_position.copy())
        if len(self.position_history) > self.filter_length:
            self.position_history.pop(0)
        
        # Calculate error (reference - measured)
        error = self.reference_position - measured_position
        
        # Calculate jitter before compensation (standard deviation of recent positions)
        if len(self.position_history) >= 3:
            positions = np.array(self.position_history)
            self.jitter_before_compensation = np.mean(np.std(positions, axis=0))
        
        # PID control
        dt = 1.0 / self.sampling_rate_hz
        
        # Proportional term
        p_term = self.proportional_gain * error
        
        # Integral term (with anti-windup)
        self.error_integral += error * dt
        max_integral = self.max_correction_urad / self.integral_gain
        self.error_integral = np.clip(self.error_integral, -max_integral, max_integral)
        i_term = self.integral_gain * self.error_integral
        
        # Derivative term (with filtering)
        if len(self.position_history) > 1:
            d_term = self.derivative_gain * (error - self.prev_error) / dt
        else:
            d_term = np.zeros_like(error)
        
        self.prev_error = error.copy()
        
        # Calculate correction
        correction = p_term + i_term + d_term
        
        # Apply bandwidth limitation (simple low-pass filter)
        alpha = 2 * np.pi * self.control_bandwidth_hz * dt
        self.correction_angles = self.correction_angles * (1 - alpha) + correction * alpha
        
        # Clip to maximum correction
        self.correction_angles = np.clip(
            self.correction_angles, 
            -self.max_correction_urad, 
            self.max_correction_urad
        )
        
        # Store correction for history
        self.correction_history.append(self.correction_angles.copy())
        if len(self.correction_history) > self.filter_length:
            self.correction_history.pop(0)
        
        # Estimate jitter after compensation (simulated)
        compensated_positions = [
            p + c * 1e-6 * self.sampling_rate_hz / 1000.0  # Simple model of correction effect
            for p, c in zip(self.position_history[-min(len(self.position_history), 5):], 
                           self.correction_history[-min(len(self.correction_history), 5):])
        ]
        if len(compensated_positions) >= 3:
            self.jitter_after_compensation = np.mean(np.std(np.array(compensated_positions), axis=0))
        
        return self.correction_angles
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        reduction_pct = 0.0
        if self.jitter_before_compensation > 0:
            reduction_pct = 100 * (1 - self.jitter_after_compensation / self.jitter_before_compensation)
            
        return {
            'jitter_before_urad': float(self.jitter_before_compensation * 1e6),
            'jitter_after_urad': float(self.jitter_after_compensation * 1e6),
            'jitter_reduction_pct': float(reduction_pct),
            'correction_x_urad': float(self.correction_angles[0]),
            'correction_y_urad': float(self.correction_angles[1]),
            'correction_rms_urad': float(np.sqrt(np.mean(np.square(self.correction_angles))))
        }
    
    def reset(self):
        """Reset compensator state."""
        self.position_history = []
        self.correction_angles = np.zeros(2)
        self.error_integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.correction_history = []
        self.jitter_before_compensation = 0.0
        self.jitter_after_compensation = 0.0

# Example usage:
# compensator = BeamJitterCompensator(sampling_rate_hz=1000.0)
# compensator.set_reference_position(np.array([0.0, 0.0]))
# 
# # In control loop:
# for i in range(100):
#     # Simulate beam jitter
#     jitter = np.random.normal(0, 1e-6, size=2)  # 1 microradian jitter
#     measured_position = np.array([0.0, 0.0]) + jitter
#     
#     # Get correction
#     correction = compensator.update(measured_position)
#     
#     # Apply correction to steering mirror
#     # steering_mirror.apply_correction(correction)
#     
# # Get performance metrics
# metrics = compensator.get_metrics()
# print(f"Jitter reduction: {metrics['jitter_reduction_pct']:.1f}%")