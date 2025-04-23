import numpy as np
from typing import Dict, Any, List, Optional, Callable

class ExplainableAnomalyDetector:
    """
    Detects anomalies in telemetry/state using statistical and explainable AI approaches.
    Provides human-interpretable explanations for detected anomalies.
    """
    def __init__(self, baseline_stats: Optional[Dict[str, Dict[str, float]]] = None, feature_names: Optional[List[str]] = None):
        """
        baseline_stats: dict of {feature: {'mean': val, 'std': val}}
        feature_names: list of features to monitor
        """
        self.baseline_stats = baseline_stats or {}
        self.feature_names = feature_names
        self.last_explanation = None
    def fit_baseline(self, data: List[Dict[str, Any]]):
        """
        Fit baseline statistics (mean, std) for each feature from historical data.
        """
        if not data:
            return
        self.feature_names = self.feature_names or list(data[0].keys())
        for feat in self.feature_names:
            vals = np.array([d[feat] for d in data if feat in d])
            self.baseline_stats[feat] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals) + 1e-6)}
    def detect(self, state: Dict[str, Any], threshold: float = 3.0) -> bool:
        """
        Returns True if anomaly detected. Sets self.last_explanation.
        """
        anomalies = []
        for feat, stats in self.baseline_stats.items():
            if feat in state:
                z = abs((state[feat] - stats['mean']) / stats['std'])
                if z > threshold:
                    anomalies.append((feat, z, state[feat], stats['mean'], stats['std']))
        if anomalies:
            self.last_explanation = self._explain(anomalies)
            return True
        self.last_explanation = None
        return False
    def _explain(self, anomalies: List) -> str:
        """
        Generate a human-readable explanation for the detected anomalies.
        """
        lines = ["Anomaly detected:"]
        for feat, z, val, mean, std in anomalies:
            lines.append(f"Feature '{feat}' is {z:.2f} std devs from mean (value={val}, mean={mean}, std={std})")
        return "\n".join(lines)
    def get_last_explanation(self) -> Optional[str]:
        return self.last_explanation

# Example usage:
# detector = ExplainableAnomalyDetector()
# detector.fit_baseline([{'pos': 1, 'vel': 2}, {'pos': 1.1, 'vel': 2.1}])
# is_anomaly = detector.detect({'pos': 5, 'vel': 2})
# print(detector.get_last_explanation())
