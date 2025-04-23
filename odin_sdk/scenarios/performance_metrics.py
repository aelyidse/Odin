from typing import Dict, Any, Optional

class MissionPerformance:
    """
    Tracks and computes performance metrics for a mission scenario, including mission success indicators.
    """
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.metrics = {
            'threats_neutralized': 0,
            'threats_escaped': 0,
            'protected_targets_hit': 0,
            'mission_time_s': 0.0,
            'resources_used': 0.0,
            'false_positives': 0,
            'false_negatives': 0,
            'engagement_success_rate': 0.0,
            'collateral_damage': 0,
            'roe_violations': 0,
        }
        self.success_indicators = {
            'mission_success': False,
            'all_protected_targets_safe': True,
            'roe_compliance': True,
            'within_time_window': True,
            'resource_efficiency': True,
        }
    def update(self, key: str, value: Any):
        if key in self.metrics:
            self.metrics[key] = value
        elif key in self.success_indicators:
            self.success_indicators[key] = value
        else:
            raise KeyError(f"Unknown metric or indicator: {key}")
    def compute_success(self, scenario_params: Dict[str, Any]):
        """
        Evaluate mission success and key indicators based on scenario and metrics.
        """
        # Protected targets
        self.success_indicators['all_protected_targets_safe'] = (self.metrics['protected_targets_hit'] == 0)
        # ROE compliance
        self.success_indicators['roe_compliance'] = (self.metrics['roe_violations'] == 0)
        # Time window
        if 'time_window' in scenario_params:
            # Example: mission_time_s < end - start (in seconds)
            start, end = scenario_params['time_window']['start'], scenario_params['time_window']['end']
            # Simplified: always True for now
            self.success_indicators['within_time_window'] = True
        # Resource efficiency (example: resources_used < threshold)
        self.success_indicators['resource_efficiency'] = (self.metrics['resources_used'] < 1000)  # Example threshold
        # Engagement success
        if (self.metrics['threats_neutralized'] > 0 and
            self.success_indicators['all_protected_targets_safe'] and
            self.success_indicators['roe_compliance'] and
            self.success_indicators['within_time_window']):
            self.success_indicators['mission_success'] = True
        else:
            self.success_indicators['mission_success'] = False
    def report(self) -> Dict[str, Any]:
        return {
            'scenario': self.scenario_name,
            'metrics': self.metrics,
            'success_indicators': self.success_indicators
        }

# Example usage:
# perf = MissionPerformance('urban_defense')
# perf.update('threats_neutralized', 5)
# perf.update('protected_targets_hit', 0)
# perf.compute_success(scenario_params)
# print(perf.report())
