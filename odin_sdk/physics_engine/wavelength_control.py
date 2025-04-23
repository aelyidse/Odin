import numpy as np
from typing import Dict, Any, Optional

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
