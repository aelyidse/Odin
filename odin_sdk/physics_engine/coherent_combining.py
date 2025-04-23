import numpy as np
from typing import List, Dict, Any, Optional

class CoherentBeamCombiner:
    """
    Models coherent beam combination with sub-wavelength phase precision.
    Supports arbitrary number of channels, amplitude, phase, and polarization state.
    """
    def __init__(self, wavelengths_nm: List[float], initial_phases: Optional[List[float]] = None, polarizations: Optional[List[np.ndarray]] = None):
        self.wavelengths_nm = np.array(wavelengths_nm)
        self.n_channels = len(wavelengths_nm)
        self.phases = np.array(initial_phases) if initial_phases is not None else np.zeros(self.n_channels)
        # Each polarization is a Jones vector (2D complex)
        if polarizations is not None:
            self.polarizations = [np.array(p, dtype=complex) for p in polarizations]
        else:
            self.polarizations = [np.array([1,0], dtype=complex) for _ in range(self.n_channels)]  # Default: horizontal
    def combine(self, amplitudes: List[float], phases: Optional[List[float]] = None, polarizations: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        amplitudes = np.array(amplitudes)
        if phases is not None:
            self.phases = np.array(phases)
        if polarizations is not None:
            pols = [np.array(p, dtype=complex) for p in polarizations]
        else:
            pols = self.polarizations
        # Sub-wavelength phase precision: use full complex field sum
        E_total = np.zeros(2, dtype=complex)  # Jones vector sum
        for i in range(self.n_channels):
            wl = self.wavelengths_nm[i]
            phase = self.phases[i] % (2*np.pi)
            # Apply phase to Jones vector
            E = amplitudes[i] * pols[i] * np.exp(1j * phase)
            E_total += E
        intensity = np.abs(E_total[0])**2 + np.abs(E_total[1])**2
        efficiency = intensity / np.sum(amplitudes**2)
        phase_error = np.std([p % (2*np.pi) for p in self.phases])
        return {
            'combined_field': E_total,
            'intensity': intensity,
            'efficiency': efficiency,
            'final_phases': self.phases.tolist(),
            'phase_error_std_rad': phase_error
        }

# Example usage:
# wavelengths = [1064, 1064, 1064]
# amplitudes = [1.0, 1.0, 1.0]
# phases = [0.0, np.pi/4, np.pi/2]
# pols = [np.array([1,0]), np.array([0,1]), np.array([1,1])/np.sqrt(2)]
# combiner = CoherentBeamCombiner(wavelengths, phases, pols)
# result = combiner.combine(amplitudes)
# print(result)
