import numpy as np
from typing import Dict, Any, List, Optional

class SensorFusionResult:
    """Structured result of fused environmental awareness, including uncertainty quantification."""
    def __init__(self, fused_state: Dict[str, Any], sources: Dict[str, Any], timestamp: float, fused_cov: Optional[np.ndarray] = None, confidence: Optional[Dict[str, float]] = None):
        self.fused_state = fused_state
        self.sources = sources
        self.timestamp = timestamp
        self.fused_cov = fused_cov  # Fused covariance matrix (if available)
        self.confidence = confidence  # Per-state confidence (optional)
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fused_state': self.fused_state,
            'sources': self.sources,
            'timestamp': self.timestamp,
            'fused_cov': self.fused_cov.tolist() if self.fused_cov is not None else None,
            'confidence': self.confidence
        }

class SensorFusionEngine:
    """Fuses multi-modal sensor data for local environmental awareness. Supports hierarchical fusion across temporal scales and adaptive sensor weighting."""
    def __init__(self, fusion_mode: str = 'weighted_average'):
        self.fusion_mode = fusion_mode  # 'weighted_average', 'bayesian', etc.

    def compute_adaptive_weights(self, sensor_data: Dict[str, Dict[str, Any]], environment: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute adaptive sensor weights based on environmental conditions.
        Args:
            sensor_data: Dict of {sensor_name: ...}
            environment: Dict with keys like 'visibility', 'weather', 'jamming', 'sensor_health'
        Returns:
            Dict of {sensor_name: weight}
        """
        weights = {}
        for name, d in sensor_data.items():
            # Default base weight
            w = 1.0
            # Example: reduce EO weight in low visibility
            if 'eo' in name.lower() and 'visibility' in environment:
                vis = environment['visibility']
                w *= max(min((vis - 0.2) / 0.8, 1.0), 0.0)  # Scale 0 (bad) to 1 (good)
            # Example: reduce radar weight in heavy rain
            if 'radar' in name.lower() and 'weather' in environment:
                if environment['weather'] in ['rain', 'storm']:
                    w *= 0.6
            # Example: reduce weight if sensor is jammed
            if 'jamming' in environment and name in environment['jamming']:
                w *= 0.2
            # Example: scale by sensor health
            if 'sensor_health' in environment and name in environment['sensor_health']:
                w *= environment['sensor_health'][name]
            weights[name] = w
        # Normalize
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
        return weights

    def hierarchical_fuse(self, temporal_sensor_data: Dict[str, Dict[str, Dict[str, Any]]], weights: Optional[Dict[str, float]] = None, timestamp: Optional[float] = None, levels: Optional[List[str]] = None) -> Dict[str, SensorFusionResult]:
        """
        Perform hierarchical fusion across temporal scales.
        Args:
            temporal_sensor_data: Dict of {level: {sensor_name: {'state':..., 'cov':...}}}
            weights: Optional weights for sensors
            timestamp: Fusion time
            levels: Temporal levels, e.g., ['short', 'mid', 'long']
        Returns:
            Dict of {level: SensorFusionResult}, plus 'final' fused result
        """
        if levels is None:
            levels = list(temporal_sensor_data.keys())
        level_results = {}
        # 1. Fuse at each temporal scale
        for lvl in levels:
            level_results[lvl] = self.fuse(temporal_sensor_data[lvl], weights, timestamp)
        # 2. Fuse across levels (meta-fusion)
        meta_sensor_data = {}
        for lvl in levels:
            lvl_res = level_results[lvl]
            meta_sensor_data[lvl] = {
                'state': lvl_res.fused_state,
                'cov': lvl_res.fused_cov
            }
        final_result = self.fuse(meta_sensor_data, weights=None, timestamp=timestamp)
        level_results['final'] = final_result
        return level_results

    def fuse(self, sensor_data: Dict[str, Dict[str, Any]], weights: Optional[Dict[str, float]] = None, timestamp: Optional[float] = None) -> SensorFusionResult:
        """
        Args:
            sensor_data: Dict of {sensor_name: {'state': Dict[str, Any], 'cov': np.ndarray}}
            weights: Optional dict of sensor weights
            timestamp: Fusion time
        Returns:
            SensorFusionResult (with uncertainty quantification)
        """
        if self.fusion_mode == 'weighted_average':
            fused_state, fused_cov, confidence = self._weighted_average_with_uncertainty(sensor_data, weights)
        else:
            fused_state, fused_cov, confidence = self._simple_average_with_uncertainty(sensor_data)
        return SensorFusionResult(fused_state, sensor_data, timestamp or 0.0, fused_cov, confidence)

    def _weighted_average_with_uncertainty(self, sensor_data: Dict[str, Dict[str, Any]], weights: Optional[Dict[str, float]]):
        # Fuse state and propagate uncertainty (assume diagonal covariances)
        keys = set()
        for d in sensor_data.values():
            keys.update(d['state'].keys())
        fused = {}
        fused_cov = {}
        confidence = {}
        for k in keys:
            vals = []
            ws = []
            vars_ = []
            for name, d in sensor_data.items():
                if k in d['state']:
                    vals.append(d['state'][k])
                    w = weights[name] if weights and name in weights else 1.0
                    ws.append(w)
                    if 'cov' in d and d['cov'] is not None:
                        idx = list(d['state'].keys()).index(k) if k in d['state'] else 0
                        # Assume diagonal
                        try:
                            vars_.append(d['cov'].item(idx, idx))
                        except Exception:
                            vars_.append(float(d['cov']))
                    else:
                        vars_.append(1.0)  # Default variance if not provided
            if vals:
                ws = np.array(ws)
                ws = ws / np.sum(ws)
                fused[k] = float(np.average(vals, weights=ws))
                # Propagate uncertainty: weighted sum of variances
                fused_var = float(np.sum(ws ** 2 * np.array(vars_)))
                fused_cov[k] = fused_var
                confidence[k] = float(np.exp(-fused_var))  # Higher confidence for lower variance
        # Assemble diagonal covariance matrix
        cov_arr = np.diag([fused_cov[k] for k in keys]) if fused_cov else None
        return fused, cov_arr, confidence

    def _simple_average_with_uncertainty(self, sensor_data: Dict[str, Dict[str, Any]]):
        keys = set()
        for d in sensor_data.values():
            keys.update(d['state'].keys())
        fused = {}
        fused_cov = {}
        confidence = {}
        for k in keys:
            vals = [d['state'][k] for d in sensor_data.values() if k in d['state']]
            vars_ = [float(d['cov']) if 'cov' in d and d['cov'] is not None else 1.0 for d in sensor_data.values() if k in d['state']]
            if vals:
                fused[k] = float(np.mean(vals))
                fused_var = float(np.mean(vars_))
                fused_cov[k] = fused_var
                confidence[k] = float(np.exp(-fused_var))
        cov_arr = np.diag([fused_cov[k] for k in keys]) if fused_cov else None
        return fused, cov_arr, confidence

# Example usage:
# fusion = SensorFusionEngine()
# result = fusion.fuse({
#     'radar': {'state': {'x': 10, 'y': 5, 'vx': 1.2}, 'cov': ...},
#     'eo': {'state': {'x': 10.1, 'y': 5.2, 'vx': 1.0}, 'cov': ...}
# }, weights={'radar': 0.7, 'eo': 0.3}, timestamp=time.time())
# print(result.to_dict())
