import numpy as np
from typing import Callable, Optional, Dict, Any

class ModelPredictiveController:
    """
    Model Predictive Control (MPC) for thermal stabilization.
    Supports linear/quasi-linear plant models, input and output constraints, and receding horizon optimization.
    """
    def __init__(self, plant_model: Callable[[np.ndarray, float], float],
                 horizon: int = 10, dt: float = 1.0,
                 u_limits: Optional[tuple] = None, y_limits: Optional[tuple] = None):
        """
        plant_model: function (state, input) -> next_state
        horizon: prediction horizon (steps)
        dt: timestep [s]
        u_limits: (umin, umax) actuator constraints
        y_limits: (ymin, ymax) output constraints
        """
        self.plant_model = plant_model
        self.horizon = horizon
        self.dt = dt
        self.u_limits = u_limits
        self.y_limits = y_limits
        self.last_u = 0.0
    def optimize(self, x0: float, setpoint: float, Q: float = 1.0, R: float = 0.01) -> float:
        """
        Solve the receding horizon control problem for the next control input.
        x0: current state (temperature)
        setpoint: desired state
        Q: output tracking weight
        R: input effort weight
        Returns: optimal control input for the next step
        """
        # Discretize input space for simple optimization (can replace with cvxpy for full QP)
        if self.u_limits is not None:
            u_candidates = np.linspace(self.u_limits[0], self.u_limits[1], 21)
        else:
            u_candidates = np.linspace(-10, 10, 21)
        best_u = u_candidates[0]
        best_cost = np.inf
        for u0 in u_candidates:
            x = x0
            cost = 0.0
            u = u0
            for t in range(self.horizon):
                x = self.plant_model(x, u)
                y = x
                if self.y_limits is not None:
                    y = np.clip(y, self.y_limits[0], self.y_limits[1])
                cost += Q * (y - setpoint)**2 + R * (u**2)
                # Simple assumption: hold input constant over horizon
            if cost < best_cost:
                best_cost = cost
                best_u = u0
        self.last_u = best_u
        return best_u

class ThermalMPCStabilizer:
    """
    Thermal stabilization using model predictive control (MPC).
    """
    def __init__(self, mpc: ModelPredictiveController):
        self.mpc = mpc
    def control(self, current_temp: float, setpoint: float) -> float:
        """Return actuator command (e.g., heater/cooler power) from MPC."""
        return self.mpc.optimize(current_temp, setpoint)

# Example usage:
# def plant(x, u):
#     # Simple first-order thermal plant: x[k+1] = x[k] + dt/tau * (-(x[k]-T_env)+K*u)
#     tau = 100.0
#     K = 0.5
#     T_env = 25.0
#     dt = 1.0
#     return x + dt/tau * (-(x-T_env) + K*u)
# mpc = ModelPredictiveController(plant, horizon=15, dt=1.0, u_limits=(0,10), y_limits=(20,80))
# stabilizer = ThermalMPCStabilizer(mpc)
# u = stabilizer.control(40, 60)
# print(u)
