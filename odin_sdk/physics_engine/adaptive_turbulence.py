import numpy as np
from typing import Callable, Tuple, List, Dict, Any

def hufnagel_valley_cn2(z: float) -> float:
    """Standard Hufnagel-Valley model for Cn^2 as a function of altitude z [m]."""
    # Parameters: ground-level turbulence, wind speed, tropopause strength
    Cn2_0 = 1.7e-14  # m^(-2/3) at ground
    V = 21.0         # m/s wind speed
    A = 5.94e-53     # tropopause term
    return (A * (V/27)**2 * z**10 * np.exp(-z/1000)
            + 2.7e-16 * np.exp(-z/1500)
            + Cn2_0 * np.exp(-z/100))

def adaptive_altitude_grid(z_min: float, z_max: float, cn2_fn: Callable[[float], float], min_points: int = 20, max_points: int = 200, rel_tol: float = 0.05) -> np.ndarray:
    """
    Generate an adaptive grid in altitude, refining where Cn^2(z) has high variation.
    Args:
        z_min, z_max: altitude range [m]
        cn2_fn: function returning Cn^2 at altitude
        min_points: minimum grid points
        max_points: maximum grid points
        rel_tol: relative tolerance for local Cn^2 variation
    Returns:
        np.ndarray of altitudes
    """
    grid = np.linspace(z_min, z_max, min_points)
    for _ in range(10):
        new_grid = [grid[0]]
        for i in range(1, len(grid)):
            z0, z1 = grid[i-1], grid[i]
            c0, c1 = cn2_fn(z0), cn2_fn(z1)
            zm = 0.5 * (z0 + z1)
            cm = cn2_fn(zm)
            interp = 0.5 * (c0 + c1)
            if abs(cm - interp) / max(cm, interp, 1e-20) > rel_tol and len(grid) < max_points:
                new_grid.append(zm)
            new_grid.append(z1)
        grid = np.unique(np.array(new_grid))
        if len(grid) >= max_points:
            break
    return np.sort(grid)

class AdaptiveTurbulenceProfile:
    """Adaptive sampling of Cn^2(z) for atmospheric turbulence modeling."""
    def __init__(self, z_min: float, z_max: float, cn2_fn: Callable[[float], float] = hufnagel_valley_cn2, rel_tol: float = 0.05):
        self.z_min = z_min
        self.z_max = z_max
        self.cn2_fn = cn2_fn
        self.rel_tol = rel_tol
        self.altitudes = adaptive_altitude_grid(z_min, z_max, cn2_fn, rel_tol=rel_tol)
        self.cn2_profile = np.array([cn2_fn(z) for z in self.altitudes])
    def as_dict(self) -> Dict[str, Any]:
        return {'altitudes_m': self.altitudes.tolist(), 'Cn2_profile': self.cn2_profile.tolist()}

# Example usage:
# profile = AdaptiveTurbulenceProfile(0, 20000)
# print(profile.as_dict())
