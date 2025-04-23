import numpy as np
import time
from typing import Dict, Any, List, Callable, Optional

class MaintenanceAlert:
    """Represents a predictive maintenance alert or recommendation."""
    def __init__(self, subsystem: str, issue: str, score: float, timestamp: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        self.subsystem = subsystem
        self.issue = issue
        self.score = score
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.details = details or {}
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subsystem': self.subsystem,
            'issue': self.issue,
            'score': self.score,
            'timestamp': self.timestamp,
            'details': self.details
        }

class PredictiveMaintenanceEngine:
    """Analyzes digital twin state/history for predictive maintenance."""
    def __init__(self):
        self.models: Dict[str, Callable[[List[Dict[str, Any]]], Optional[MaintenanceAlert]]] = {}
        self.alerts: List[MaintenanceAlert] = []

    def register_model(self, subsystem: str, model_fn: Callable[[List[Dict[str, Any]]], Optional[MaintenanceAlert]]):
        """Register a predictive model for a subsystem."""
        self.models[subsystem] = model_fn

    def analyze(self, twin_history: List[Any]):
        """Run all registered models on the digital twin's history."""
        for subsystem, model_fn in self.models.items():
            alert = model_fn([h[1].get(subsystem, {}) for h in twin_history])
            if alert:
                self.alerts.append(alert)

    def get_alerts(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.alerts]

# Example model: simple threshold or trend detection
# def laser_temp_model(history):
#     temps = [h.get('temp', 0) for h in history if 'temp' in h]
#     if len(temps) > 10 and np.mean(temps[-10:]) > 70:
#         return MaintenanceAlert('laser', 'overtemp_trend', score=1.0, details={'mean_last10': float(np.mean(temps[-10:]))})
#     return None
# engine = PredictiveMaintenanceEngine()
# engine.register_model('laser', laser_temp_model)
# engine.analyze(twin.get_history())
# print(engine.get_alerts())
