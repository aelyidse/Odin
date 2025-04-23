from typing import Optional
import numpy as np

class PIDController:
    """Generic PID controller with anti-windup and gain scheduling."""
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0.0, output_limits: Optional[tuple] = None, dt: float = 1e-3,
                 gain_schedule_fn: Optional[callable] = None, anti_windup: str = "clamp"):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits  # (min, max)
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.last_output = 0.0
        self.gain_schedule_fn = gain_schedule_fn  # Function: (setpoint, measurement, error) -> (Kp, Ki, Kd)
        self.anti_windup = anti_windup  # 'clamp' or 'backcalc'
        self.awu_beta = 0.8  # Back-calculation gain (if used)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.last_output = 0.0

    def step(self, measurement: float) -> float:
        error = self.setpoint - measurement
        # Gain scheduling
        if self.gain_schedule_fn is not None:
            self.Kp, self.Ki, self.Kd = self.gain_schedule_fn(self.setpoint, measurement, error)
        # PID terms
        self.integral += error * self.dt
        derivative = 0.0
        if self.prev_measurement is not None:
            derivative = (measurement - self.prev_measurement) / self.dt
        output = self.Kp * error + self.Ki * self.integral - self.Kd * derivative
        unclamped_output = output
        # Anti-windup
        if self.output_limits is not None:
            output = float(np.clip(output, self.output_limits[0], self.output_limits[1]))
            # Integral clamping
            if self.anti_windup == "clamp":
                if self.Ki != 0:
                    if output != unclamped_output:
                        # Prevent further windup
                        self.integral -= error * self.dt  # Undo last integration
            # Back-calculation
            elif self.anti_windup == "backcalc":
                if self.Ki != 0:
                    self.integral += self.awu_beta * (output - unclamped_output) / self.Ki
        self.prev_error = error
        self.prev_measurement = measurement
        self.last_output = output
        return output

class ThermalStabilizer:
    """Thermal stabilization using PID control."""
    def __init__(self, pid: PIDController):
        self.pid = pid
    def control(self, current_temp: float) -> float:
        """Return actuator command (e.g., heater/cooler power)."""
        return self.pid.step(current_temp)

class MechanicalStabilizer:
    """Mechanical stabilization (e.g., vibration or pointing) using PID control."""
    def __init__(self, pid: PIDController):
        self.pid = pid
    def control(self, current_position: float) -> float:
        """Return actuator command (e.g., stage, gimbal, or piezo voltage)."""
        return self.pid.step(current_position)
