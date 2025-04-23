import numpy as np
from typing import Dict, Any, List, Optional
from system_simulation.environment import WeatherModel, TerrainModel, EnvironmentalConditionModel

class ScenarioGenerator:
    """Generates scenarios for various engagement profiles in ODIN simulation."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate(self, profile: str = 'default', n_targets: int = 3) -> Dict[str, Any]:
        """
        Args:
            profile: Engagement scenario type (e.g., 'airborne', 'swarm', 'missile', 'mixed')
            n_targets: Number of targets
        Returns:
            Scenario dictionary for simulation setup
        """
        # Weather
        if profile == 'bad_weather':
            weather = WeatherModel(visibility_km=3, humidity_pct=90, temperature_C=5, wind_speed_mps=12, precipitation='rain')
        else:
            weather = WeatherModel()
        # Terrain
        grid = self.rng.uniform(0, 100, (100, 100))
        terrain = TerrainModel(grid, grid_spacing_m=10.0)
        env = EnvironmentalConditionModel(weather, terrain)
        # Targets
        targets = []
        for i in range(n_targets):
            if profile == 'airborne':
                ttype = 'aircraft'
                alt = self.rng.uniform(1000, 10000)
            elif profile == 'swarm':
                ttype = 'uav'
                alt = self.rng.uniform(100, 1000)
            elif profile == 'missile':
                ttype = 'missile'
                alt = self.rng.uniform(50, 2000)
            else:
                ttype = self.rng.choice(['aircraft', 'uav', 'missile'])
                alt = self.rng.uniform(50, 10000)
            targets.append({
                'id': i+1,
                'type': ttype,
                'position': [self.rng.uniform(0, 1000), self.rng.uniform(0, 1000), alt],
                'velocity': [self.rng.uniform(-200, 200), self.rng.uniform(-200, 200), self.rng.uniform(-20, 20)],
                'priority': self.rng.uniform(0, 1)
            })
        # Beams
        beams = []
        for i in range(n_targets):
            beams.append({
                'id': i+1,
                'power_w': self.rng.uniform(500, 5000)
            })
        # Sensor data (mock)
        sensor_data = {
            'radar': {'state': {'x': t['position'][0], 'y': t['position'][1], 'vx': t['velocity'][0]}, 'cov': np.eye(3).tolist()} for t in targets
        }
        scenario = {
            'environment': env.as_dict(),
            'targets': targets,
            'beams': beams,
            'sensor_data': sensor_data,
            'profile': profile
        }
        return scenario

# Example usage:
# gen = ScenarioGenerator()
# scenario = gen.generate(profile='swarm', n_targets=5)
# print(scenario)
