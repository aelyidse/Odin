import numpy as np
from typing import Tuple, Dict, Any

class ZonalWavefrontReconstructor:
    """
    Advanced zonal wavefront reconstruction using local slope measurements (e.g., Shack-Hartmann).
    Supports arbitrary grid geometries and robust least-squares integration.
    """
    def __init__(self, grid_shape: Tuple[int, int]):
        self.ny, self.nx = grid_shape
        self.grid_shape = grid_shape
    def reconstruct(self, slopes_x: np.ndarray, slopes_y: np.ndarray) -> np.ndarray:
        """
        Reconstruct wavefront from local slopes (zonal approach).
        slopes_x, slopes_y: measured slopes (ny, nx) arrays
        Returns: reconstructed wavefront (ny, nx)
        """
        ny, nx = self.ny, self.nx
        # Build integration matrix (finite difference, least squares)
        N = ny * nx
        A = []
        b = []
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                # x-slope: phi(i+1,j) - phi(i,j) = dx * slope_x
                if i < nx - 1:
                    row = np.zeros(N)
                    row[idx] = -1
                    row[idx + 1] = 1
                    A.append(row)
                    b.append(slopes_x[j, i])
                # y-slope: phi(i,j+1) - phi(i,j) = dy * slope_y
                if j < ny - 1:
                    row = np.zeros(N)
                    row[idx] = -1
                    row[idx + nx] = 1
                    A.append(row)
                    b.append(slopes_y[j, i])
        # Fix piston (mean zero)
        row = np.zeros(N)
        row[:] = 1.0 / N
        A.append(row)
        b.append(0.0)
        A = np.vstack(A)
        b = np.array(b)
        # Least squares solution
        phi_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
        phi = phi_vec.reshape((ny, nx))
        return phi
    def as_dict(self, phi: np.ndarray) -> Dict[str, Any]:
        return {'wavefront': phi.tolist()}

# Example usage:
# recon = ZonalWavefrontReconstructor((10,10))
# slopes_x = np.random.randn(10,10)
# slopes_y = np.random.randn(10,10)
# phi = recon.reconstruct(slopes_x, slopes_y)
# print(phi)
