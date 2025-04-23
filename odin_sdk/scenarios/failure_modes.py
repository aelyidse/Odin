from typing import Dict, Any, List, Callable

class FailureMode:
    """
    Represents a failure mode for a system component, with possible effects and propagation rules.
    """
    def __init__(self, component: str, failure_type: str, probability: float, effects: List[str]):
        self.component = component
        self.failure_type = failure_type  # e.g., 'sensor blackout', 'jammed', 'misclassification'
        self.probability = probability  # Probability of failure in scenario
        self.effects = effects  # List of downstream effects (by name)
    def as_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'failure_type': self.failure_type,
            'probability': self.probability,
            'effects': self.effects
        }

class FailurePropagationEngine:
    """
    Simulates failure mode activation and propagates effects through the system.
    """
    def __init__(self, failure_modes: List[FailureMode], propagation_rules: Dict[str, List[str]]):
        self.failure_modes = failure_modes
        self.propagation_rules = propagation_rules  # effect -> list of downstream effects
    def propagate(self, initial_failures: List[str]) -> Dict[str, Any]:
        """
        Given a list of initial failure effect names, propagate through system.
        Returns dict of all affected components and triggered effects.
        """
        affected = set(initial_failures)
        newly_affected = set(initial_failures)
        while newly_affected:
            next_affected = set()
            for effect in newly_affected:
                for downstream in self.propagation_rules.get(effect, []):
                    if downstream not in affected:
                        next_affected.add(downstream)
            affected.update(next_affected)
            newly_affected = next_affected
        # Map effects to components
        component_map = {}
        for fm in self.failure_modes:
            for eff in fm.effects:
                if eff in affected:
                    component_map.setdefault(fm.component, []).append(eff)
        return {'affected_effects': list(affected), 'component_map': component_map}
    def random_failure_injection(self, seed: int = None) -> Dict[str, Any]:
        """
        Randomly activate failure modes based on their probability and propagate effects.
        """
        import random
        rng = random.Random(seed)
        initial = []
        for fm in self.failure_modes:
            if rng.random() < fm.probability:
                initial.extend(fm.effects)
        return self.propagate(initial)

# Example usage:
# fms = [FailureMode('sensor','blackout',0.1,['sensor_loss']), FailureMode('comm','jammed',0.2,['comm_loss'])]
# rules = {'sensor_loss':['tracking_loss'], 'comm_loss':['no_engage']}
# engine = FailurePropagationEngine(fms, rules)
# print(engine.random_failure_injection())
