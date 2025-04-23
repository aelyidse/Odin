from typing import Dict, Any, List

class MissionScenario:
    """
    Represents a mission scenario with all relevant parameters for simulation or planning.
    """
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'parameters': self.parameters}

class ScenarioLibrary:
    """
    Library of predefined mission scenarios and adversarial scenario generation.
    Allows retrieval and listing of scenarios by name or type, and creation of adversarial scenarios.
    """
    def __init__(self):
        self.scenarios: List[MissionScenario] = []
        self._build_default_scenarios()

    def generate_adversarial(self, base: str = None, seed: int = None, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Generate an adversarial scenario based on a base scenario or at random.
        Intensity controls the level of challenge (threat surge, deception, jamming, etc).
        """
        import random
        rng = random.Random(seed)
        if base:
            base_scenario = None
            for s in self.scenarios:
                if s.name == base:
                    base_scenario = s
                    break
            if not base_scenario:
                raise ValueError(f"Base scenario '{base}' not found.")
            params = dict(base_scenario.parameters)
        else:
            # Pick a random base scenario
            params = dict(rng.choice(self.scenarios).parameters)
        # Adversarial modifications
        # 1. Threat surge
        n_threats = int(3 + 5 * intensity)
        params['threat_types'] = params.get('threat_types', []) + [rng.choice(['uav','missile','jammer','decoy','boat','vehicle']) for _ in range(n_threats)]
        # 2. Deception: add decoys and ambiguous targets
        if intensity > 0.5:
            params['threat_types'] += ['decoy'] * int(2 * intensity)
        # 3. Electronic attack
        if intensity > 0.7:
            params['electronic_attack'] = {'jamming': True, 'spoofing': rng.random() > 0.5}
        # 4. Unexpected behaviors
        if intensity > 0.3:
            params['unexpected_behaviors'] = rng.sample([
                'pop-up threat', 'coordinated swarm', 'multi-axis attack', 'false positive', 'target switching', 'sensor blackout'
            ], int(2 * intensity))
        # 5. Weather/terrain stressors
        if intensity > 0.4:
            params['weather'] = rng.choice(['fog','rain','storm','hot','cold'])
        scenario_name = f"adversarial_{base or 'random'}_{rng.randint(1000,9999)}"
        scenario = MissionScenario(scenario_name, params)
        self.scenarios.append(scenario)
        return scenario.to_dict()

    def generate_graduated_complexity(self, levels: int = 5, seed: int = None) -> list:
        """
        Generate a list of scenarios with graduated complexity for system testing.
        Levels range from simple (single threat, clear weather) to highly complex (multi-threat, adversarial, environmental stressors).
        """
        import random
        rng = random.Random(seed)
        scenarios = []
        for level in range(1, levels+1):
            base = rng.choice(self.scenarios)
            params = dict(base.parameters)
            # Scale up complexity
            params['threat_types'] = params.get('threat_types', []) + [rng.choice(['uav','missile','jammer','decoy','boat','vehicle']) for _ in range(level)]
            if level > 2:
                params['weather'] = rng.choice(['clear','fog','rain','storm','hot','cold'])
            if level > 3:
                params['electronic_attack'] = {'jamming': level > 3, 'spoofing': level > 4}
                params['unexpected_behaviors'] = rng.sample([
                    'pop-up threat', 'coordinated swarm', 'multi-axis attack', 'false positive', 'target switching', 'sensor blackout'
                ], min(level, 6))
            scenario_name = f"test_complexity_lvl{level}_{rng.randint(1000,9999)}"
            scenario = MissionScenario(scenario_name, params)
            self.scenarios.append(scenario)
            scenarios.append(scenario.to_dict())
        return scenarios
    def _build_default_scenarios(self):
        self.scenarios.append(MissionScenario(
            'urban_defense', {
                'allowed_actions': ['engage', 'track', 'defend'],
                'roe': 'tight',
                'protected_targets': ['Hospital', 'School'],
                'operational_area': ['SectorA', 'SectorB'],
                'time_window': {'start': '06:00', 'end': '20:00'},
                'weather': 'clear',
                'terrain': 'urban',
                'threat_types': ['uav', 'missile'],
            }
        ))
        self.scenarios.append(MissionScenario(
            'maritime_interdiction', {
                'allowed_actions': ['track', 'intercept', 'engage'],
                'roe': 'standard',
                'protected_targets': ['CivilianVessel'],
                'operational_area': ['SeaZone1', 'SeaZone2'],
                'time_window': {'start': '00:00', 'end': '23:59'},
                'weather': 'fog',
                'terrain': 'sea',
                'threat_types': ['boat', 'fast_inshore_attack_craft'],
            }
        ))
        self.scenarios.append(MissionScenario(
            'desert_recon', {
                'allowed_actions': ['recon', 'track', 'support'],
                'roe': 'loose',
                'protected_targets': [],
                'operational_area': ['DesertSector1'],
                'time_window': {'start': '05:00', 'end': '21:00'},
                'weather': 'hot',
                'terrain': 'desert',
                'threat_types': ['vehicle', 'uav'],
            }
        ))
    def get(self, name: str) -> Dict[str, Any]:
        for s in self.scenarios:
            if s.name == name:
                return s.to_dict()
        raise ValueError(f"Scenario '{name}' not found.")
    def list(self) -> List[str]:
        return [s.name for s in self.scenarios]

# Example usage:
# lib = ScenarioLibrary()
# print(lib.list())
# print(lib.get('urban_defense'))
