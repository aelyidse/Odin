import numpy as np
from typing import List, Dict, Any, Optional

class OptimalBeamAllocator:
    """
    Allocates laser power optimally to multiple targets under total power constraints.
    Supports weighted prioritization (e.g., threat level, mission value) and minimum/maximum per-target bounds.
    """
    def __init__(self, max_total_power: float, min_power: Optional[List[float]] = None, max_power: Optional[List[float]] = None):
        self.max_total_power = max_total_power
        self.min_power = min_power
        self.max_power = max_power
    def allocate(self, priorities: List[float], n_targets: Optional[int] = None) -> List[float]:
        """
        priorities: list of weights (e.g., threat scores)
        Returns: list of allocated powers (sum <= max_total_power)
        """
        if n_targets is None:
            n_targets = len(priorities)
        priorities = np.array(priorities)
        priorities = np.maximum(priorities, 0)
        if np.sum(priorities) == 0:
            # Even allocation if no priorities
            alloc = np.ones(n_targets) / n_targets
        else:
            alloc = priorities / np.sum(priorities)
        power = alloc * self.max_total_power
        # Apply per-target min/max constraints
        if self.min_power is not None:
            power = np.maximum(power, self.min_power)
        if self.max_power is not None:
            power = np.minimum(power, self.max_power)
        # Renormalize if over budget
        total = np.sum(power)
        if total > self.max_total_power:
            power = power * (self.max_total_power / total)
        return power.tolist()
    def as_dict(self, allocation: List[float]) -> Dict[str, Any]:
        return {'allocation': allocation}

# Example usage:
# allocator = OptimalBeamAllocator(max_total_power=1000, min_power=[50,50,50], max_power=[600,600,600])
# priorities = [0.9, 0.7, 0.2]
# allocation = allocator.allocate(priorities)
# print(allocation)
