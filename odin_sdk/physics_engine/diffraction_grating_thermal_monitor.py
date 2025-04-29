import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from .diffraction_grating import DiffractionGrating

class DiffractionGratingThermalMonitor:
    """
    Monitors thermal stability of diffraction gratings and provides alerts
    when thermal variations affect wavelength precision or diffraction efficiency.
    """
    def __init__(self, 
                 grating: DiffractionGrating,
                 reference_wavelength_nm: float = 1064.0,
                 max_wavelength_drift_nm: float = 0.1,
                 max_angle_drift_mrad: float = 0.05,
                 sampling_interval_s: float = 1.0,
                 alert_threshold: float = 0.8,
                 history_length: int = 100):
        """
        Initialize diffraction grating thermal monitor.
        
        Args:
            grating: DiffractionGrating instance to monitor
            reference_wavelength_nm: Reference wavelength for monitoring
            max_wavelength_drift_nm: Maximum acceptable wavelength drift
            max_angle_drift_mrad: Maximum acceptable diffraction angle drift in mrad
            sampling_interval_s: Time between measurements in seconds
            alert_threshold: Threshold (0-1) for triggering alerts
            history_length: Number of measurements to keep in history
        """
        self.grating = grating
        self.reference_wavelength_nm = reference_wavelength_nm
        self.max_wavelength_drift_nm = max_wavelength_drift_nm
        self.max_angle_drift_mrad = max_angle_drift_mrad
        self.sampling_interval_s = sampling_interval_s
        self.alert_threshold = alert_threshold
        
        # Reference values at base temperature
        self.reference_temp_C = grating.base_temp_C
        self.reference_period_nm = grating.period_nm(self.reference_temp_C)
        self.reference_angle_deg = grating.diffraction_angle(
            reference_wavelength_nm, 1, 45.0, self.reference_temp_C)
        
        # Monitoring state
        self.current_temp_C = self.reference_temp_C
        self.is_stable = True
        self.alert_active = False
        self.stability_score = 1.0
        
        # History tracking
        self.history: List[Dict[str, Any]] = []
        self.history_length = history_length
        self.last_measurement_time = 0
    
    def measure(self, current_temp_C: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Measure thermal stability at current temperature.
        
        Args:
            current_temp_C: Current temperature in Celsius
            timestamp: Optional measurement timestamp
            
        Returns:
            Measurement results
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Update current temperature
        self.current_temp_C = current_temp_C
        
        # Calculate current grating period
        current_period_nm = self.grating.period_nm(current_temp_C)
        period_drift_nm = current_period_nm - self.reference_period_nm
        
        # Calculate current diffraction angle
        current_angle_deg = self.grating.diffraction_angle(
            self.reference_wavelength_nm, 1, 45.0, current_temp_C)
        angle_drift_deg = current_angle_deg - self.reference_angle_deg
        angle_drift_mrad = angle_drift_deg * 1000.0 / 57.3  # Convert deg to mrad
        
        # Calculate wavelength drift due to thermal expansion
        # Δλ/λ = Δd/d for first order
        wavelength_drift_nm = self.reference_wavelength_nm * (period_drift_nm / self.reference_period_nm)
        
        # Calculate stability metrics
        period_stability = 1.0 - abs(period_drift_nm) / (self.max_wavelength_drift_nm * 2)
        angle_stability = 1.0 - abs(angle_drift_mrad) / self.max_angle_drift_mrad
        wavelength_stability = 1.0 - abs(wavelength_drift_nm) / self.max_wavelength_drift_nm
        
        # Overall stability score (weighted average)
        self.stability_score = 0.4 * period_stability + 0.3 * angle_stability + 0.3 * wavelength_stability
        self.stability_score = max(0.0, min(1.0, self.stability_score))
        
        # Update stability state
        self.is_stable = self.stability_score > self.alert_threshold
        
        # Check if alert should be active
        prev_alert = self.alert_active
        self.alert_active = not self.is_stable
        
        # Create measurement record
        measurement = {
            'timestamp': timestamp,
            'temperature_C': current_temp_C,
            'period_nm': current_period_nm,
            'period_drift_nm': period_drift_nm,
            'angle_deg': current_angle_deg,
            'angle_drift_mrad': angle_drift_mrad,
            'wavelength_drift_nm': wavelength_drift_nm,
            'stability_score': self.stability_score,
            'is_stable': self.is_stable,
            'alert_active': self.alert_active,
            'alert_changed': prev_alert != self.alert_active
        }
        
        # Add to history
        self.history.append(measurement)
        if len(self.history) > self.history_length:
            self.history.pop(0)
            
        self.last_measurement_time = timestamp
        
        return measurement
    
    def get_stability_metrics(self, window: int = 20) -> Dict[str, float]:
        """
        Get stability metrics over recent history window.
        
        Args:
            window: Number of recent measurements to consider
            
        Returns:
            Dictionary of stability metrics
        """
        if not self.history:
            return {
                'mean_stability': 1.0,
                'min_stability': 1.0,
                'temperature_variance': 0.0,
                'drift_rate_nm_per_min': 0.0
            }
            
        # Get recent measurements
        recent = self.history[-window:] if len(self.history) > window else self.history
        
        # Calculate metrics
        stability_scores = [m['stability_score'] for m in recent]
        temperatures = [m['temperature_C'] for m in recent]
        
        # Calculate drift rate if we have timestamps
        drift_rate = 0.0
        if len(recent) >= 2:
            start_time = recent[0]['timestamp']
            end_time = recent[-1]['timestamp']
            start_drift = recent[0]['wavelength_drift_nm']
            end_drift = recent[-1]['wavelength_drift_nm']
            
            if end_time > start_time:
                time_diff_minutes = (end_time - start_time) / 60.0
                if time_diff_minutes > 0:
                    drift_rate = (end_drift - start_drift) / time_diff_minutes
        
        return {
            'mean_stability': float(np.mean(stability_scores)),
            'min_stability': float(np.min(stability_scores)),
            'temperature_variance': float(np.var(temperatures)),
            'drift_rate_nm_per_min': float(drift_rate)
        }
    
    def predict_drift(self, future_temp_C: float) -> Dict[str, float]:
        """
        Predict drift if temperature changes to specified value.
        
        Args:
            future_temp_C: Future temperature to predict for
            
        Returns:
            Dictionary with predicted drift values
        """
        # Calculate predicted period
        predicted_period_nm = self.grating.period_nm(future_temp_C)
        period_drift_nm = predicted_period_nm - self.reference_period_nm
        
        # Calculate predicted diffraction angle
        predicted_angle_deg = self.grating.diffraction_angle(
            self.reference_wavelength_nm, 1, 45.0, future_temp_C)
        angle_drift_deg = predicted_angle_deg - self.reference_angle_deg
        angle_drift_mrad = angle_drift_deg * 1000.0 / 57.3
        
        # Calculate predicted wavelength drift
        wavelength_drift_nm = self.reference_wavelength_nm * (period_drift_nm / self.reference_period_nm)
        
        return {
            'predicted_period_nm': float(predicted_period_nm),
            'predicted_period_drift_nm': float(period_drift_nm),
            'predicted_angle_deg': float(predicted_angle_deg),
            'predicted_angle_drift_mrad': float(angle_drift_mrad),
            'predicted_wavelength_drift_nm': float(wavelength_drift_nm)
        }