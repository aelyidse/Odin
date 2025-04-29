import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize

class PolarizationStateVerifier:
    """
    Verifies and monitors polarization state maintenance through optical systems.
    Provides real-time analysis of polarization degradation and compensation feedback.
    """
    def __init__(self, 
                 reference_jones_vector: np.ndarray,
                 tolerance_threshold: float = 0.95,
                 measurement_wavelength_nm: float = 1064.0):
        """
        Initialize polarization state verifier.
        
        Args:
            reference_jones_vector: Reference Jones vector (normalized)
            tolerance_threshold: Minimum acceptable polarization overlap (0-1)
            measurement_wavelength_nm: Wavelength for measurements
        """
        # Normalize reference Jones vector
        self.reference_jones = reference_jones_vector / np.linalg.norm(reference_jones_vector)
        self.tolerance_threshold = tolerance_threshold
        self.wavelength_nm = measurement_wavelength_nm
        
        # Tracking metrics
        self.history = []
        self.current_overlap = 1.0
        self.current_extinction_ratio = float('inf')
        self.alarm_active = False
    
    def calculate_overlap(self, measured_jones: np.ndarray) -> float:
        """
        Calculate overlap between reference and measured polarization states.
        
        Args:
            measured_jones: Measured Jones vector
            
        Returns:
            Overlap metric (0-1, where 1 is perfect)
        """
        # Normalize measured Jones vector
        measured_norm = measured_jones / np.linalg.norm(measured_jones)
        
        # Calculate overlap (fidelity)
        overlap = np.abs(np.vdot(self.reference_jones, measured_norm))**2
        return float(overlap)
    
    def calculate_extinction_ratio(self, measured_jones: np.ndarray) -> float:
        """
        Calculate polarization extinction ratio.
        
        Args:
            measured_jones: Measured Jones vector
            
        Returns:
            Extinction ratio in dB
        """
        # Normalize measured Jones vector
        measured_norm = measured_jones / np.linalg.norm(measured_jones)
        
        # Calculate orthogonal component
        projection = np.vdot(self.reference_jones, measured_norm) * self.reference_jones
        orthogonal = measured_norm - projection
        
        # Calculate extinction ratio
        p_parallel = np.abs(projection)**2
        p_orthogonal = np.abs(orthogonal)**2
        
        if p_orthogonal == 0:
            return float('inf')
        
        extinction_ratio = 10 * np.log10(p_parallel / p_orthogonal)
        return float(extinction_ratio)
    
    def verify(self, measured_jones: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Verify polarization state against reference.
        
        Args:
            measured_jones: Measured Jones vector
            timestamp: Optional measurement timestamp
            
        Returns:
            Verification results
        """
        # Calculate metrics
        overlap = self.calculate_overlap(measured_jones)
        extinction_ratio = self.calculate_extinction_ratio(measured_jones)
        
        # Check if within tolerance
        is_maintained = overlap >= self.tolerance_threshold
        
        # Update alarm state
        self.alarm_active = not is_maintained
        
        # Update current metrics
        self.current_overlap = overlap
        self.current_extinction_ratio = extinction_ratio
        
        # Store in history
        result = {
            'timestamp': timestamp if timestamp is not None else len(self.history),
            'overlap': overlap,
            'extinction_ratio': extinction_ratio,
            'is_maintained': is_maintained,
            'alarm_active': self.alarm_active
        }
        self.history.append(result)
        
        return result
    
    def get_compensation_suggestion(self, measured_jones: np.ndarray) -> Dict[str, Any]:
        """
        Calculate suggested compensation to restore polarization state.
        
        Args:
            measured_jones: Measured Jones vector
            
        Returns:
            Compensation parameters
        """
        # Only calculate if polarization maintenance is poor
        if self.calculate_overlap(measured_jones) >= self.tolerance_threshold:
            return {'required': False, 'parameters': {}}
        
        # Define objective function for optimization
        def objective(params):
            # Simple model: phase retardation and rotation
            phase_retardation = params[0]
            rotation_angle = params[1]
            
            # Create compensation Jones matrix
            cos_rot = np.cos(rotation_angle)
            sin_rot = np.sin(rotation_angle)
            exp_phase = np.exp(1j * phase_retardation)
            
            jones_matrix = np.array([
                [cos_rot, -sin_rot],
                [sin_rot, cos_rot]
            ]) @ np.array([
                [1, 0],
                [0, exp_phase]
            ])
            
            # Apply compensation
            compensated = jones_matrix @ measured_jones
            
            # Calculate overlap with reference
            return -self.calculate_overlap(compensated)  # Negative for minimization
        
        # Run optimization
        initial_params = [0.0, 0.0]  # Initial phase and rotation
        bounds = [(0, 2*np.pi), (0, np.pi)]  # Bounds for phase and rotation
        
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Extract optimized parameters
        optimized_params = result.x
        phase_retardation = optimized_params[0]
        rotation_angle = optimized_params[1]
        
        # Calculate expected improvement
        expected_overlap = -result.fun
        
        return {
            'required': True,
            'parameters': {
                'phase_retardation_rad': float(phase_retardation),
                'rotation_angle_rad': float(rotation_angle)
            },
            'expected_overlap': float(expected_overlap),
            'improvement': float(expected_overlap - self.calculate_overlap(measured_jones))
        }
    
    def get_stability_metrics(self, window: int = 20) -> Dict[str, float]:
        """
        Calculate stability metrics over recent history.
        
        Args:
            window: Number of recent measurements to consider
            
        Returns:
            Stability metrics
        """
        if not self.history:
            return {'mean_overlap': 1.0, 'std_overlap': 0.0, 'min_overlap': 1.0}
        
        recent = self.history[-window:] if len(self.history) > window else self.history
        overlaps = [entry['overlap'] for entry in recent]
        
        return {
            'mean_overlap': float(np.mean(overlaps)),
            'std_overlap': float(np.std(overlaps)),
            'min_overlap': float(np.min(overlaps)),
            'stability_score': float(np.mean(overlaps) / (1.0 + np.std(overlaps)))
        }