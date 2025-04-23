import numpy as np
import time
from typing import Dict, Any, List, Callable, Optional

class PerformanceDegradationAlert:
    """Represents an alert for detected performance degradation."""
    def __init__(self, metric: str, value: float, threshold: float, timestamp: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.details = details or {}
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'details': self.details
        }

class PerformanceMonitor:
    """Monitors system performance and detects degradation in real time."""
    def __init__(self, thresholds: Dict[str, float], window_size: int = 20):
        self.thresholds = thresholds  # e.g., {'beam_efficiency': 0.8, 'thermal_margin': 10.0}
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {k: [] for k in thresholds}
        self.alerts: List[PerformanceDegradationAlert] = []
    def record(self, metric: str, value: float):
        if metric not in self.history:
            self.history[metric] = []
        self.history[metric].append(value)
        if len(self.history[metric]) > self.window_size:
            self.history[metric] = self.history[metric][-self.window_size:]
        self.check_degradation(metric)
    def check_degradation(self, metric: str):
        if metric not in self.thresholds:
            return
        values = self.history[metric]
        if not values:
            return
        current = values[-1]
        if metric in ['beam_efficiency', 'thermal_margin']:
            if current < self.thresholds[metric]:
                alert = PerformanceDegradationAlert(
                    metric=metric,
                    value=current,
                    threshold=self.thresholds[metric],
                    details={'history': values[-self.window_size:]}
                )
                self.alerts.append(alert)
        # Add more metric-specific logic as needed
    def get_alerts(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.alerts]

# Example usage:
# monitor = PerformanceMonitor({'beam_efficiency': 0.85, 'thermal_margin': 12.0})
# for v in [0.9, 0.88, 0.84, 0.82, 0.8]: monitor.record('beam_efficiency', v)
# print(monitor.get_alerts())
