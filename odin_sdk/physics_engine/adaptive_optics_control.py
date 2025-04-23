import numpy as np
from typing import Tuple, Optional

class DeformableMirror:
    """Concrete model and control algorithm for a deformable mirror (DM)."""
    def __init__(self, n_actuators_x: int, n_actuators_y: int, max_stroke_um: float = 8.0):
        self.nx = n_actuators_x
        self.ny = n_actuators_y
        self.max_stroke_um = max_stroke_um
        self.actuator_commands = np.zeros((self.ny, self.nx))  # in microns

    def compute_commands(self, wavefront_error: np.ndarray, influence_fn: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute actuator commands to minimize wavefront error.
        Args:
            wavefront_error: 2D array of phase error (microns)
            influence_fn: Optional influence function matrix (ny*nx, ny*nx)
        Returns:
            actuator_commands: 2D array (ny, nx) in microns
        """
        # Flatten wavefront error and solve least-squares for actuator commands
        wf_flat = wavefront_error.flatten()
        n_act = self.nx * self.ny
        if influence_fn is None:
            influence_fn = np.eye(n_act)  # Assume direct mapping
        # Least-squares solution
        cmds, _, _, _ = np.linalg.lstsq(influence_fn, wf_flat, rcond=None)
        self.actuator_commands = cmds.reshape((self.ny, self.nx))
        # Clip to physical stroke
        self.actuator_commands = np.clip(self.actuator_commands, -self.max_stroke_um, self.max_stroke_um)
        return self.actuator_commands

    def apply(self) -> np.ndarray:
        """Return the phase map applied by the DM (microns)."""
        return self.actuator_commands.copy()

class LCSLM:
    """Concrete model and control algorithm for a liquid crystal spatial light modulator (LC-SLM)."""
    def __init__(self, n_pixels_x: int, n_pixels_y: int, max_phase_rad: float = 2 * np.pi):
        self.nx = n_pixels_x
        self.ny = n_pixels_y
        self.max_phase_rad = max_phase_rad
        self.pixel_commands = np.zeros((self.ny, self.nx))  # in radians

    def compute_commands(self, wavefront_error: np.ndarray) -> np.ndarray:
        """
        Compute pixel commands to minimize wavefront error.
        Args:
            wavefront_error: 2D array of phase error (radians)
        Returns:
            pixel_commands: 2D array (ny, nx) in radians
        """
        # Directly map negative wavefront error to pixel commands
        cmds = -wavefront_error
        self.pixel_commands = np.clip(cmds, -self.max_phase_rad, self.max_phase_rad)
        return self.pixel_commands

    def apply(self) -> np.ndarray:
        """Return the phase map applied by the LC-SLM (radians)."""
        return self.pixel_commands.copy()
