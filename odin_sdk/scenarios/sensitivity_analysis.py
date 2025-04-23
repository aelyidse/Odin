import numpy as np
from typing import Dict, Any, List, Callable, Optional

class SensitivityAnalyzer:
    """
    Performs sensitivity analysis to quantify the contribution of each system component or parameter
    to mission performance or output metrics.
    """
    def __init__(self, base_params: Dict[str, Any], metric_func: Callable[[Dict[str, Any]], float]):
        """
        base_params: dict of all parameters (e.g., scenario, sensor, environment)
        metric_func: function that evaluates performance metric given params
        """
        self.base_params = base_params
        self.metric_func = metric_func
    def analyze(self, components: List[str], delta: float = 0.05) -> Dict[str, float]:
        """
        For each component, perturb its value by +/- delta (relative) and assess impact on metric.
        Returns dict of component: sensitivity score (absolute effect on metric).
        """
        sensitivities = {}
        base_metric = self.metric_func(self.base_params)
        for comp in components:
            params_plus = dict(self.base_params)
            params_minus = dict(self.base_params)
            # Only perturb numeric values
            if isinstance(self.base_params[comp], (int, float)):
                params_plus[comp] = self.base_params[comp] * (1 + delta)
                params_minus[comp] = self.base_params[comp] * (1 - delta)
                metric_plus = self.metric_func(params_plus)
                metric_minus = self.metric_func(params_minus)
                # Central difference approximation
                sensitivity = (metric_plus - metric_minus) / (2 * self.base_params[comp] * delta)
                sensitivities[comp] = abs(sensitivity)
            else:
                sensitivities[comp] = 0.0  # Non-numeric: not supported
        return sensitivities

# Example usage:
# def mission_metric(params):
#     # Example: higher threat count reduces success
#     return 1.0 - 0.1 * params['threat_count']
# base = {'threat_count': 5, 'sensor_range': 10.0, 'weather_severity': 2.0}
# analyzer = SensitivityAnalyzer(base, mission_metric)
# print(analyzer.analyze(['threat_count', 'sensor_range', 'weather_severity']))
