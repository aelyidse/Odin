import numpy as np
from typing import Dict, Any, Optional

class RayleighScattering:
    """Computes Rayleigh scattering cross-section and attenuation with spectral dependence."""
    @staticmethod
    def cross_section(wavelength_nm: float, n: float = 1.0003, depolarization: float = 0.035) -> float:
        # Rayleigh cross-section per molecule (Bucholtz, 1995)
        # wavelength in nm, n: refractive index, depolarization: King factor
        wl = wavelength_nm * 1e-9  # nm to m
        N_s = 2.547e25  # molecules/m^3 at STP
        K = (6 + 3 * depolarization) / (6 - 7 * depolarization)
        sigma = (24 * np.pi**3 * (n**2 - 1)**2) / (wl**4 * N_s**2 * (n**2 + 2)**2) * K
        return sigma  # m^2 per molecule
    @staticmethod
    def attenuation(length_m: float, wavelength_nm: float, n: float = 1.0003, depolarization: float = 0.035, number_density: float = 2.547e25) -> float:
        sigma = RayleighScattering.cross_section(wavelength_nm, n, depolarization)
        tau = sigma * number_density * length_m
        return np.exp(-tau)  # transmission fraction

class MieScattering:
    """Computes Mie scattering efficiency and attenuation for polydisperse aerosols with spectral dependence."""
    @staticmethod
    def mie_efficiency(size_param: float, m: complex) -> float:
        # Placeholder: full Mie calculation is complex (see Bohren & Huffman)
        # Use empirical fit for spherical particles (van de Hulst, 1957)
        # Q ~ 2 for x >> 1, Q ~ 4x^4/3 for x << 1
        if size_param < 0.1:
            return 4 * size_param**4 / 3
        elif size_param > 10:
            return 2.0
        else:
            return 2 - 2 * np.exp(-4 * size_param / 3)
    @staticmethod
    def attenuation(length_m: float, wavelength_nm: float, r_eff_um: float, n_medium: float, m_particle: complex, number_density: float) -> float:
        wl = wavelength_nm * 1e-9  # nm to m
        r_eff = r_eff_um * 1e-6    # um to m
        x = 2 * np.pi * r_eff / wl
        Q = MieScattering.mie_efficiency(x, m_particle)
        sigma = Q * np.pi * r_eff**2
        tau = sigma * number_density * length_m
        return np.exp(-tau)  # transmission fraction
    @staticmethod
    def spectral_curve(length_m: float, wavelengths_nm: np.ndarray, r_eff_um: float, n_medium: float, m_particle: complex, number_density: float) -> np.ndarray:
        return np.array([
            MieScattering.attenuation(length_m, wl, r_eff_um, n_medium, m_particle, number_density)
            for wl in wavelengths_nm
        ])

# Example usage:
# rayleigh_trans = RayleighScattering.attenuation(1000, 532, n=1.0003)
# mie_trans = MieScattering.attenuation(1000, 1064, r_eff_um=0.5, n_medium=1.0, m_particle=1.5+0.01j, number_density=1e8)
# spectrum = MieScattering.spectral_curve(1000, np.linspace(400, 1100, 100), 0.5, 1.0, 1.5+0.01j, 1e8)
