import time
from typing import Dict, Any, List, Optional
import numpy as np

class SystemPerformanceMetrics:
    """Tracks and computes system-level performance metrics for ODIN simulation."""
    def __init__(self):
        self.metrics: Dict[str, List[Any]] = {}
        self.start_time = None
        self.end_time = None

    def start_benchmark(self):
        self.start_time = time.perf_counter()

    def end_benchmark(self):
        self.end_time = time.perf_counter()

    def record(self, metric: str, value: Any):
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)

    def get_metric(self, metric: str) -> List[Any]:
        return self.metrics.get(metric, [])

    def summary(self) -> Dict[str, Any]:
        summary = {}
        for k, v in self.metrics.items():
            arr = np.array(v)
            if arr.dtype.kind in 'fi':
                summary[k] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'last': float(arr[-1])
                }
            else:
                summary[k] = {'count': len(arr), 'last': arr[-1]}
        if self.start_time and self.end_time:
            summary['wall_time_s'] = self.end_time - self.start_time
        return summary

    def reset(self):
        self.metrics.clear()
        self.start_time = None
        self.end_time = None

# Example system-level metrics to record:
# - engagement_success_rate
# - mean_time_to_engage
# - system_response_time
# - resource_utilization
# - tracking_accuracy
# - beam_combining_efficiency
# - thermal_margin
# - control_loop_jitter
# - fault_count
# Metrics can be recorded at each simulation step or at scenario completion.

# Example usage:
# metrics = SystemPerformanceMetrics()
# metrics.start_benchmark()
# ... run simulation, record metrics ...
# metrics.end_benchmark()
# print(metrics.summary())
