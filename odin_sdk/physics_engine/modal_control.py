import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.special import eval_genlaguerre

class ZernikePolynomials:
    """
    Generates Zernike polynomials for modal wavefront control on a unit disk.
    """
    @staticmethod
    def noll_indices(j: int) -> Tuple[int, int]:
        # Noll's sequential indices to (n, m)
        n = 0
        j1 = j - 1
        while j1 > n:
            n += 1
            j1 -= n
        m = -n + 2 * j1
        return n, m
    @staticmethod
    def zernike(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Radial part
        R = np.zeros_like(rho)
        for k in range((n - abs(m)) // 2 + 1):
            c = (-1)**k * np.math.factorial(n - k) / (
                np.math.factorial(k) * np.math.factorial((n + abs(m))//2 - k) * np.math.factorial((n - abs(m))//2 - k))
            R += c * rho**(n - 2*k)
        if m >= 0:
            return R * np.cos(m * theta)
        else:
            return R * np.sin(-m * theta)
    @staticmethod
    def basis(n_modes: int, grid_shape: Tuple[int, int]) -> np.ndarray:
        ny, nx = grid_shape
        y, x = np.linspace(-1, 1, ny), np.linspace(-1, 1, nx)
        xx, yy = np.meshgrid(x, y)
        rho = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        mask = rho <= 1
        basis = []
        for j in range(1, n_modes + 1):
            n, m = ZernikePolynomials.noll_indices(j)
            Z = np.zeros_like(rho)
            Z[mask] = ZernikePolynomials.zernike(n, m, rho[mask], theta[mask])
            basis.append(Z)
        return np.array(basis)

class ModalWavefrontController:
    """
    Modal control using Zernike or other orthogonal basis functions.
    Projects wavefronts onto modal basis and computes actuator commands.
    """
    def __init__(self, n_modes: int, grid_shape: Tuple[int, int], basis_fn: Optional[callable] = None):
        self.n_modes = n_modes
        self.grid_shape = grid_shape
        self.basis_fn = basis_fn or (lambda n, g: ZernikePolynomials.basis(n, g))
        self.basis = self.basis_fn(n_modes, grid_shape)
    def analyze(self, wavefront: np.ndarray) -> np.ndarray:
        # Project wavefront onto modal basis (least squares)
        phi = wavefront.ravel()
        B = self.basis.reshape(self.n_modes, -1).T
        coeffs, *_ = np.linalg.lstsq(B, phi, rcond=None)
        return coeffs
    def synthesize(self, coeffs: np.ndarray) -> np.ndarray:
        # Reconstruct wavefront from modal coefficients
        return np.tensordot(coeffs, self.basis, axes=1)
    def as_dict(self, coeffs: np.ndarray) -> Dict[str, Any]:
        return {'coefficients': coeffs.tolist()}

# Example usage:
# ctrl = ModalWavefrontController(10, (32,32))
# wf = np.random.randn(32,32)
# coeffs = ctrl.analyze(wf)
# wf_rec = ctrl.synthesize(coeffs)
