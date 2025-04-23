import numpy as np
from typing import Dict, Any, Tuple
from .fiber import PMFiberModel

class PANDAFiberModel(PMFiberModel):
    """Concrete simulation of PANDA-type polarization-maintaining fiber propagation."""
    def __init__(self,
                 length_m: float,
                 birefringence: float,
                 extinction_ratio_db: float,
                 core_diameter_um: float,
                 cladding_diameter_um: float,
                 stress_element_distance_um: float,
                 wavelength_nm: float,
                 temp_C: float = 25.0,
                 crosstalk_per_m: float = 1e-6,
                 additional_params: Dict[str, Any] = None):
        """
        Args:
            length_m: Fiber length (meters)
            birefringence: Delta n (typ. 3e-4 to 5e-4)
            extinction_ratio_db: Polarization extinction ratio (dB)
            core_diameter_um: Core diameter (microns)
            cladding_diameter_um: Cladding diameter (microns)
            stress_element_distance_um: Distance between stress-applying parts (microns)
            wavelength_nm: Operating wavelength (nm)
            temp_C: Temperature (C)
            crosstalk_per_m: Power fraction coupled per meter (for imperfection/loss)
            additional_params: Optional dictionary for extra fields
        """
        self.length_m = length_m
        self.birefringence = birefringence
        self.extinction_ratio_db = extinction_ratio_db
        self.core_diameter_um = core_diameter_um
        self.cladding_diameter_um = cladding_diameter_um
        self.stress_element_distance_um = stress_element_distance_um
        self.wavelength_nm = wavelength_nm
        self.temp_C = temp_C
        self.crosstalk_per_m = crosstalk_per_m
        self.additional_params = additional_params or {}

    def propagate(self, field: np.ndarray, length: float, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            field: Input Jones vector (2,) or Stokes vector (4,)
            length: Propagation length (meters, <= self.length_m)
            params: Dict with optional 'delta_n', 'temp_C', 'crosstalk_per_m', etc.
        Returns:
            Tuple (output_field, output_polarization_state)
        """
        # Use Jones formalism for polarization evolution
        delta_n = params.get('delta_n', self.birefringence)
        temp_C = params.get('temp_C', self.temp_C)
        crosstalk = params.get('crosstalk_per_m', self.crosstalk_per_m)
        lam = params.get('wavelength_nm', self.wavelength_nm) * 1e-9  # m
        L = min(length, self.length_m)
        # Phase delay between axes
        delta_beta = 2 * np.pi * delta_n / lam
        phi = delta_beta * L
        # Jones matrix for ideal PM fiber (no crosstalk)
        J = np.array([[np.exp(-1j * phi / 2), 0],
                      [0, np.exp(1j * phi / 2)]], dtype=np.complex128)
        # Add crosstalk (coupling between axes, simple exponential loss model)
        if crosstalk > 0:
            eta = np.exp(-crosstalk * L)
            J = eta * J + (1 - eta) * np.eye(2)
        # Apply Jones matrix
        out_field = J @ field
        # Compute output Stokes vector
        S0 = np.abs(out_field[0])**2 + np.abs(out_field[1])**2
        S1 = np.abs(out_field[0])**2 - np.abs(out_field[1])**2
        S2 = 2 * np.real(out_field[0] * np.conj(out_field[1]))
        S3 = 2 * np.imag(out_field[0] * np.conj(out_field[1]))
        stokes = np.array([S0, S1, S2, S3])
        return out_field, stokes
