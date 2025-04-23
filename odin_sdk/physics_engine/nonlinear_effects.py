import numpy as np
from typing import Dict, Any, Optional

class SBSModel:
    """Stimulated Brillouin Scattering (SBS) threshold and spectral broadening."""
    @staticmethod
    def sbs_threshold(power_w: float, effective_area_m2: float, gain_length_m: float, g_B: float = 5e-11, n_eff: float = 1.45) -> float:
        # SBS threshold power (Agrawal, Nonlinear Fiber Optics)
        # g_B: Brillouin gain coefficient [m/W], effective_area: [m^2], gain_length: [m], n_eff: refractive index
        return 21 * effective_area_m2 / (g_B * gain_length_m)
    @staticmethod
    def sbs_gain(power_w: float, effective_area_m2: float, gain_length_m: float, g_B: float = 5e-11) -> float:
        return g_B * power_w * gain_length_m / effective_area_m2
    @staticmethod
    def sbs_broadening(bandwidth_hz: float, acoustic_lifetime_s: float = 1e-9) -> float:
        # SBS linewidth broadening due to modulation
        return bandwidth_hz + 1/(2*np.pi*acoustic_lifetime_s)

class SRSModel:
    """Stimulated Raman Scattering (SRS) threshold and spectral evolution."""
    @staticmethod
    def srs_threshold(power_w: float, effective_area_m2: float, gain_length_m: float, g_R: float = 1e-13) -> float:
        # SRS threshold power (Agrawal)
        return 16 * effective_area_m2 / (g_R * gain_length_m)
    @staticmethod
    def srs_gain(power_w: float, effective_area_m2: float, gain_length_m: float, g_R: float = 1e-13) -> float:
        return g_R * power_w * gain_length_m / effective_area_m2
    @staticmethod
    def srs_spectrum_shift(wavelength_nm: float) -> float:
        # Raman shift in silica ~13.2 THz
        c = 2.99792458e8
        shift_hz = 13.2e12
        wl_m = wavelength_nm * 1e-9
        freq = c / wl_m
        return c / (freq - shift_hz) * 1e9  # nm

class FWMModel:
    """Four-Wave Mixing (FWM) efficiency and spectral products."""
    @staticmethod
    def fwm_efficiency(power_w: float, gamma: float, length_m: float, delta_beta: float) -> float:
        # FWM efficiency (Agrawal)
        # gamma: nonlinear coefficient [1/W/m], delta_beta: phase mismatch [1/m]
        if delta_beta == 0:
            return (gamma * power_w * length_m)**2
        else:
            return (gamma * power_w * np.sinc(delta_beta * length_m / (2*np.pi)))**2
    @staticmethod
    def fwm_spectral_products(wavelengths_nm: np.ndarray) -> np.ndarray:
        # Returns all possible FWM-generated wavelengths (degenerate and non-degenerate)
        wl = wavelengths_nm
        products = []
        n = len(wl)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and i != k and j != k:
                        wl_fwm = 1/(1/wl[i] + 1/wl[j] - 1/wl[k])
                        products.append(wl_fwm)
        return np.unique(products)

class NonlinearSpectralEffects:
    """Aggregates nonlinear effects (SBS, SRS, FWM) for spectral modeling."""
    def __init__(self, fiber_params: Dict[str, Any]):
        self.fiber_params = fiber_params
    def compute(self, power_w: float, wavelengths_nm: np.ndarray, bandwidth_hz: float) -> Dict[str, Any]:
        eff_area = self.fiber_params['effective_area_m2']
        length = self.fiber_params['length_m']
        n_eff = self.fiber_params.get('n_eff', 1.45)
        gamma = self.fiber_params.get('gamma', 1.0)
        # SBS
        sbs_thresh = SBSModel.sbs_threshold(power_w, eff_area, length)
        sbs_gain = SBSModel.sbs_gain(power_w, eff_area, length)
        sbs_broad = SBSModel.sbs_broadening(bandwidth_hz)
        # SRS
        srs_thresh = SRSModel.srs_threshold(power_w, eff_area, length)
        srs_gain = SRSModel.srs_gain(power_w, eff_area, length)
        srs_shift = SRSModel.srs_spectrum_shift(np.mean(wavelengths_nm))
        # FWM
        fwm_eff = FWMModel.fwm_efficiency(power_w, gamma, length, delta_beta=0)
        fwm_products = FWMModel.fwm_spectral_products(wavelengths_nm)
        return {
            'SBS': {'threshold_w': sbs_thresh, 'gain': sbs_gain, 'broadening_hz': sbs_broad},
            'SRS': {'threshold_w': srs_thresh, 'gain': srs_gain, 'shifted_wavelength_nm': srs_shift},
            'FWM': {'efficiency': fwm_eff, 'products_nm': fwm_products.tolist()}
        }

# Example usage:
# params = {'effective_area_m2': 50e-12, 'length_m': 10, 'n_eff': 1.45, 'gamma': 1.2}
# model = NonlinearSpectralEffects(params)
# result = model.compute(1000, np.array([1064, 1070, 1080]), 1e9)
# print(result)
