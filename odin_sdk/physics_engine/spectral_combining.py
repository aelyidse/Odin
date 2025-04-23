import numpy as np
from typing import List, Dict, Any, Optional

class PhaseLockController:
    """Implements phase locking feedback for spectral beam combining."""
    def __init__(self, n_channels: int, feedback_gain: float = 0.1):
        self.n_channels = n_channels
        self.feedback_gain = feedback_gain
        self.phases = np.zeros(n_channels)  # Current phase offsets [rad]
        self.target_phases = np.zeros(n_channels)  # Target phases (e.g., 0 for in-phase)
    def update(self, measured_phases: np.ndarray) -> np.ndarray:
        """Feedback update: adjusts phase actuators to minimize phase error."""
        phase_error = self.target_phases - measured_phases
        self.phases += self.feedback_gain * phase_error
        return self.phases.copy()

class SpectralCombiner:
    """Spectral beam combining with phase locking feedback."""
    def __init__(self, wavelengths_nm: List[float], initial_phases: Optional[List[float]] = None, feedback_gain: float = 0.1):
        self.wavelengths_nm = np.array(wavelengths_nm)
        self.n_channels = len(wavelengths_nm)
        self.phases = np.array(initial_phases) if initial_phases is not None else np.zeros(self.n_channels)
        self.phase_lock = PhaseLockController(self.n_channels, feedback_gain)
    def combine(self, amplitudes: List[float], measured_phases: List[float], n_feedback_steps: int = 10) -> Dict[str, Any]:
        amplitudes = np.array(amplitudes)
        measured_phases = np.array(measured_phases)
        # Feedback loop
        for _ in range(n_feedback_steps):
            self.phases = self.phase_lock.update(measured_phases)
            measured_phases = self.measure_phases(amplitudes, self.phases)
        # Compute combined field and efficiency
        E = np.sum(amplitudes * np.exp(1j * self.phases))
        efficiency = np.abs(E)**2 / np.sum(amplitudes**2)
        return {
            'combined_field': E,
            'efficiency': efficiency,
            'final_phases': self.phases.tolist()
        }
    def measure_phases(self, amplitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
        # In a real system, this would use interferometric feedback; here we simulate perfect measurement
        return phases + np.random.normal(0, 0.01, size=phases.shape)  # Add small noise

# Example usage:
# combiner = SpectralCombiner([1064, 1070, 1080], feedback_gain=0.2)
# result = combiner.combine([1,1,1], [0.1, -0.2, 0.05], n_feedback_steps=20)
# print(result)
