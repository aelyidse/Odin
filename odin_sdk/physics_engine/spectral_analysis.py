import numpy as np
from typing import List, Dict, Any, Tuple

def beam_combining_efficiency(input_spectra: List[Tuple[np.ndarray, np.ndarray]], combined_spectrum: Tuple[np.ndarray, np.ndarray], window_nm: float = 2.0) -> float:
    """
    Calculate beam combining efficiency as the ratio of in-band combined power to total input power.
    Args:
        input_spectra: List of (wavelengths, powers) arrays for each input beam
        combined_spectrum: (wavelengths, powers) array for the combined beam
        window_nm: Wavelength window for in-band power calculation (nm)
    Returns:
        Efficiency (0-1)
    """
    # Total input power (sum over all beams)
    total_input = sum(np.trapz(p, w) for w, p in input_spectra)
    # For each input, find in-band power in combined spectrum
    combined_w, combined_p = combined_spectrum
    in_band_power = 0.0
    for w, p in input_spectra:
        for peak in w[np.where(p == np.max(p))]:
            idx = np.where((combined_w >= peak - window_nm/2) & (combined_w <= peak + window_nm/2))[0]
            if idx.size > 0:
                in_band_power += np.trapz(combined_p[idx], combined_w[idx])
    return float(in_band_power / total_input) if total_input > 0 else 0.0

def spectral_overlap(input_spectra: List[Tuple[np.ndarray, np.ndarray]], combined_spectrum: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Calculate normalized spectral overlap between input beams and combined beam.
    Args:
        input_spectra: List of (wavelengths, powers) arrays
        combined_spectrum: (wavelengths, powers) array
    Returns:
        Overlap metric (0-1)
    """
    combined_w, combined_p = combined_spectrum
    overlap = 0.0
    for w, p in input_spectra:
        interp_combined = np.interp(w, combined_w, combined_p, left=0, right=0)
        overlap += np.trapz(np.minimum(p, interp_combined), w)
    total_input = sum(np.trapz(p, w) for w, p in input_spectra)
    return float(overlap / total_input) if total_input > 0 else 0.0

def spectral_mismatch(input_spectra: List[Tuple[np.ndarray, np.ndarray]], combined_spectrum: Tuple[np.ndarray, np.ndarray]) -> List[float]:
    """
    Calculate mismatch (fractional loss) for each input beam in the combined spectrum.
    Args:
        input_spectra: List of (wavelengths, powers) arrays
        combined_spectrum: (wavelengths, powers) array
    Returns:
        List of mismatch fractions (0-1) for each input
    """
    combined_w, combined_p = combined_spectrum
    mismatches = []
    for w, p in input_spectra:
        interp_combined = np.interp(w, combined_w, combined_p, left=0, right=0)
        input_power = np.trapz(p, w)
        matched_power = np.trapz(np.minimum(p, interp_combined), w)
        mismatches.append(float(1.0 - matched_power / input_power) if input_power > 0 else 1.0)
    return mismatches
