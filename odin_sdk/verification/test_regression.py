import unittest
import os
import json
from system_simulation.full_system_sim import OdinFullSystemSimulator
from system_simulation.scenario_generator import ScenarioGenerator
from system_simulation.performance_metrics import SystemPerformanceMetrics

class TestSystemRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = 'odin_config.json'  # Should exist or be mocked
        cls.log_path = 'odin_regression_log.jsonl'
        cls.gen = ScenarioGenerator(seed=1234)
        cls.metrics = SystemPerformanceMetrics()

    def run_scenario_and_check(self, profile, n_targets, expected_min_success=0.7):
        scenario = self.gen.generate(profile=profile, n_targets=n_targets)
        sim = OdinFullSystemSimulator(self.config_path, self.log_path)
        sim.setup_scenario(scenario)
        self.metrics.start_benchmark()
        sim.run(duration=2.0, dt=0.05)
        self.metrics.end_benchmark()
        log = sim.get_log()
        # Example: check that at least expected_min_success fraction of targets were engaged
        engaged = sum(1 for e in log if e.get('event') == 'beam_allocation' and e['content']['assignments'])
        self.assertGreaterEqual(engaged / n_targets, expected_min_success)
        # Example: check wall time
        summary = self.metrics.summary()
        self.assertLess(summary.get('wall_time_s', 0), 10.0)

    def test_airborne_profile(self):
        self.run_scenario_and_check('airborne', 3)

    def test_swarm_profile(self):
        self.run_scenario_and_check('swarm', 5, expected_min_success=0.6)

    def test_bad_weather_profile(self):
        self.run_scenario_and_check('bad_weather', 2, expected_min_success=0.5)

if __name__ == '__main__':
    unittest.main()
