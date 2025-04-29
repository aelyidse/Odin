import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import minimize

class BeamAlignmentVerification:
    """
    Verifies beam alignment using angle sensing data to ensure optimal 
    beam coupling and system performance.
    
    Features:
    - Real-time angle deviation monitoring
    - Alignment stability analysis
    - Automatic alert generation for misalignment
    - Historical trend analysis
    """
    
    def __init__(self, 
                 reference_angles_deg: Dict[str, float],
                 max_angle_deviation_mrad: float = 0.1,
                 stability_window: int = 20,
                 alert_threshold: float = 0.8,
                 sampling_interval_s: float = 0.5):
        """
        Initialize beam alignment verification system.
        
        Args:
            reference_angles_deg: Reference angles for each axis {'x': angle_x, 'y': angle_y}
            max_angle_deviation_mrad: Maximum acceptable angle deviation in milliradians
            stability_window: Number of measurements to use for stability calculations
            alert_threshold: Threshold (0-1) for triggering alignment alerts
            sampling_interval_s: Time between measurements in seconds
        """
        self.reference_angles = reference_angles_deg
        self.max_angle_deviation_mrad = max_angle_deviation_mrad
        self.stability_window = stability_window
        self.alert_threshold = alert_threshold
        self.sampling_interval_s = sampling_interval_s
        
        # Monitoring state
        self.current_angles = reference_angles_deg.copy()
        self.is_aligned = True
        self.alert_active = False
        self.alignment_score = 1.0
        
        # History tracking
        self.history: List[Dict[str, Any]] = []
        self.last_measurement_time = time.time()
        
    def verify_alignment(self, 
                        measured_angles_deg: Dict[str, float], 
                        timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Verify beam alignment using measured angle data.
        
        Args:
            measured_angles_deg: Measured angles for each axis {'x': angle_x, 'y': angle_y}
            timestamp: Optional measurement timestamp
            
        Returns:
            Verification results including alignment status and metrics
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Update current angles
        self.current_angles = measured_angles_deg.copy()
        
        # Calculate angle deviations in milliradians
        deviations_mrad = {}
        for axis, ref_angle in self.reference_angles.items():
            if axis in measured_angles_deg:
                # Convert degree deviation to milliradians
                deviation_deg = measured_angles_deg[axis] - ref_angle
                deviations_mrad[axis] = deviation_deg * (np.pi / 180) * 1000
            else:
                deviations_mrad[axis] = 0.0
                
        # Calculate alignment metrics
        axis_scores = {}
        for axis, deviation in deviations_mrad.items():
            # Normalize deviation to score (1.0 = perfect alignment, 0.0 = max deviation)
            axis_scores[axis] = max(0.0, 1.0 - abs(deviation) / self.max_angle_deviation_mrad)
            
        # Overall alignment score (average of axis scores)
        self.alignment_score = sum(axis_scores.values()) / len(axis_scores) if axis_scores else 1.0
        
        # Update alignment state
        prev_aligned = self.is_aligned
        self.is_aligned = self.alignment_score > self.alert_threshold
        
        # Check if alert should be active
        prev_alert = self.alert_active
        self.alert_active = not self.is_aligned
        
        # Calculate RMS deviation
        rms_deviation_mrad = np.sqrt(np.mean([d**2 for d in deviations_mrad.values()]))
        
        # Create measurement record
        measurement = {
            'timestamp': timestamp,
            'angles_deg': measured_angles_deg,
            'deviations_mrad': deviations_mrad,
            'axis_scores': axis_scores,
            'alignment_score': self.alignment_score,
            'rms_deviation_mrad': rms_deviation_mrad,
            'is_aligned': self.is_aligned,
            'alert_active': self.alert_active,
            'alert_changed': prev_alert != self.alert_active
        }
        
        # Add to history
        self.history.append(measurement)
        if len(self.history) > 100:  # Limit history length
            self.history.pop(0)
            
        self.last_measurement_time = timestamp
        
        return measurement
    
    def get_stability_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate alignment stability metrics over recent history.
        
        Args:
            window: Number of recent measurements to consider (defaults to stability_window)
            
        Returns:
            Dictionary of stability metrics
        """
        if window is None:
            window = self.stability_window
            
        if not self.history or len(self.history) < 2:
            return {
                'mean_alignment_score': 1.0,
                'alignment_stability': 1.0,
                'max_deviation_mrad': 0.0,
                'rms_deviation_mrad': 0.0
            }
            
        # Get recent measurements
        recent = self.history[-window:] if len(self.history) > window else self.history
        
        # Calculate metrics
        alignment_scores = [m['alignment_score'] for m in recent]
        rms_deviations = [m['rms_deviation_mrad'] for m in recent]
        
        # Extract all axis deviations
        all_deviations = []
        for m in recent:
            all_deviations.extend(list(m['deviations_mrad'].values()))
        
        return {
            'mean_alignment_score': float(np.mean(alignment_scores)),
            'alignment_stability': float(1.0 - np.std(alignment_scores)),
            'max_deviation_mrad': float(np.max(np.abs(all_deviations))) if all_deviations else 0.0,
            'rms_deviation_mrad': float(np.mean(rms_deviations))
        }
    
    def predict_drift(self, time_horizon_s: float = 60.0) -> Dict[str, Any]:
        """
        Predict alignment drift over specified time horizon.
        
        Args:
            time_horizon_s: Time horizon for prediction in seconds
            
        Returns:
            Dictionary with drift predictions
        """
        if len(self.history) < 10:
            return {
                'predicted_drift_mrad': {axis: 0.0 for axis in self.reference_angles},
                'predicted_alignment_score': self.alignment_score,
                'time_to_misalignment_s': float('inf')
            }
        
        # Calculate drift rates for each axis
        drift_rates = {}
        for axis in self.reference_angles:
            # Get recent measurements with timestamps
            measurements = [(m['timestamp'], m['deviations_mrad'].get(axis, 0.0)) 
                           for m in self.history if axis in m.get('deviations_mrad', {})]
            
            if len(measurements) < 5:
                drift_rates[axis] = 0.0
                continue
                
            # Simple linear regression to find drift rate
            times = np.array([m[0] for m in measurements])
            deviations = np.array([m[1] for m in measurements])
            
            # Normalize times to start from 0
            times = times - times[0]
            
            if np.all(times == 0):
                drift_rates[axis] = 0.0
                continue
                
            # Calculate drift rate (mrad/s)
            slope = np.polyfit(times, deviations, 1)[0]
            drift_rates[axis] = slope
        
        # Predict future deviations
        predicted_deviations = {
            axis: self.current_angles.get(axis, 0.0) + rate * time_horizon_s
            for axis, rate in drift_rates.items()
        }
        
        # Calculate predicted alignment score
        predicted_scores = {}
        for axis, deviation in predicted_deviations.items():
            predicted_scores[axis] = max(0.0, 1.0 - abs(deviation) / self.max_angle_deviation_mrad)
        
        predicted_score = sum(predicted_scores.values()) / len(predicted_scores) if predicted_scores else 1.0
        
        # Estimate time to misalignment
        time_to_misalignment = float('inf')
        if self.alignment_score > self.alert_threshold:
            # Currently aligned, calculate time until score drops below threshold
            worst_axis = min(drift_rates.items(), key=lambda x: abs(x[1]), default=(None, 0))
            if worst_axis[0] and abs(worst_axis[1]) > 1e-6:
                current_deviation = self.current_angles.get(worst_axis[0], 0.0)
                max_allowed_deviation = self.max_angle_deviation_mrad * (1.0 - self.alert_threshold)
                remaining_deviation = max_allowed_deviation - abs(current_deviation)
                if remaining_deviation > 0 and worst_axis[1] != 0:
                    time_to_misalignment = remaining_deviation / abs(worst_axis[1])
        
        return {
            'predicted_drift_mrad': drift_rates,
            'predicted_deviations_mrad': predicted_deviations,
            'predicted_alignment_score': predicted_score,
            'time_to_misalignment_s': time_to_misalignment
        }
    
    def get_correction_angles(self) -> Dict[str, float]:
        """
        Calculate correction angles needed to restore alignment.
        
        Returns:
            Dictionary of correction angles for each axis
        """
        corrections = {}
        for axis, ref_angle in self.reference_angles.items():
            current = self.current_angles.get(axis, ref_angle)
            corrections[axis] = ref_angle - current
            
        return corrections