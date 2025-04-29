import numpy as np
from typing import Callable, Optional, Dict, Any

class PredictiveAtmosphericCompensator:
    """
    Predictive atmospheric compensation using temporal forecasting and control.
    Uses recent wavefront/turbulence measurements to estimate and pre-correct future distortions.
    """
    def __init__(self, prediction_horizon: int = 3, model_fn: Optional[Callable] = None, 
                 buffer_size: int = 10, use_modal: bool = True, n_modes: int = 20):
        """
        Initialize predictive compensator.
        
        Args:
            prediction_horizon: Number of time steps to predict ahead
            model_fn: Custom prediction model function (if None, uses default)
            buffer_size: Maximum size of history buffer
            use_modal: Whether to use modal decomposition for prediction
            n_modes: Number of modes to use if modal decomposition is enabled
        """
        self.prediction_horizon = prediction_horizon
        self.model_fn = model_fn or self.default_model
        self.history = []  # Store recent wavefront/turbulence states
        self.buffer_size = buffer_size
        self.use_modal = use_modal
        self.n_modes = n_modes
        self.modal_coeffs_history = []  # For modal prediction
        self.turbulence_compensator = None  # Will be set if modal decomposition used
        
    def set_turbulence_compensator(self, compensator):
        """Set turbulence compensator for modal decomposition."""
        self.turbulence_compensator = compensator
        
    def update(self, measurement: np.ndarray):
        """
        Add a new wavefront/turbulence measurement (2D array).
        
        Args:
            measurement: New wavefront measurement (phase map)
        """
        if self.use_modal and self.turbulence_compensator:
            # Decompose into modes before storing
            modal_phase = self.turbulence_compensator.modal_decomposition(
                measurement, n_modes=self.n_modes)
            self.history.append(modal_phase.copy())
            
            # Also store raw measurement for reference
            if len(self.history) > self.buffer_size:
                self.history.pop(0)
        else:
            # Store raw measurement
            self.history.append(measurement.copy())
            if len(self.history) > self.buffer_size:
                self.history.pop(0)
    
    def predict(self) -> np.ndarray:
        """
        Predict the next wavefront/turbulence state.
        
        Returns: 
            predicted 2D array (same shape as input)
        """
        return self.model_fn(self.history)
    
    @staticmethod
    def default_model(history: list) -> np.ndarray:
        """
        Default prediction model using linear extrapolation.
        
        Args:
            history: List of previous wavefront measurements
            
        Returns:
            Predicted next wavefront
        """
        if len(history) < 2:
            return history[-1] if history else np.zeros((1,1))
            
        # Linear extrapolation: 2*last - second_last
        return 2*history[-1] - history[-2]
    
    def ar_model(self, history: list, order: int = 3) -> np.ndarray:
        """
        Autoregressive prediction model.
        
        Args:
            history: List of previous wavefront measurements
            order: AR model order
            
        Returns:
            Predicted next wavefront
        """
        if len(history) < order + 1:
            return self.default_model(history)
            
        # Simplified AR model - in practice would use proper AR coefficient estimation
        # This is a weighted average of recent history with more weight on recent frames
        weights = np.array([0.5, 0.3, 0.15, 0.05])[:min(order, 4)]
        weights = weights / np.sum(weights)
        
        prediction = np.zeros_like(history[-1])
        for i, w in enumerate(weights):
            if i < len(history):
                prediction += w * history[-(i+1)]
                
        return prediction
    
    def kalman_predict(self, history: list) -> np.ndarray:
        """
        Kalman filter-based prediction (simplified implementation).
        
        Args:
            history: List of previous wavefront measurements
            
        Returns:
            Predicted next wavefront
        """
        if len(history) < 3:
            return self.default_model(history)
            
        # Simplified Kalman prediction - in practice would use proper Kalman filter
        # This estimates velocity and applies it to current state
        current = history[-1]
        velocity = (history[-1] - history[-2])
        acceleration = (history[-1] - 2*history[-2] + history[-3])
        
        # Predict using constant acceleration model
        prediction = current + velocity + 0.5 * acceleration
        return prediction
    
    def compensate(self, current_wavefront: np.ndarray) -> np.ndarray:
        """
        Compute the pre-correction to apply for predictive compensation.
        
        Args:
            current_wavefront: Current wavefront measurement
            
        Returns: 
            correction phase map (2D array)
        """
        # Update history with current measurement
        self.update(current_wavefront)
        
        # Predict future wavefront
        predicted = self.predict()
        
        # Return negative of prediction as correction
        return -predicted
    
    def as_dict(self, correction: np.ndarray) -> Dict[str, Any]:
        """Convert correction to dictionary for serialization."""
        return {
            'correction': correction.tolist(),
            'prediction_horizon': self.prediction_horizon,
            'history_length': len(self.history),
            'use_modal': self.use_modal,
            'n_modes': self.n_modes if self.use_modal else 0
        }

# Example usage:
# comp = PredictiveAtmosphericCompensator(prediction_horizon=5)
# for t in range(10):
#     wf = np.random.randn(8,8)  # Simulate measured wavefront
#     comp.update(wf)
#     correction = comp.compensate(wf)
#     print(correction)
