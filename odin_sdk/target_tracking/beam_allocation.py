import numpy as np
from typing import List, Dict, Any, Optional
from scipy.optimize import linear_sum_assignment

class BeamAllocationOptimizer:
    """Optimizes allocation of multiple beams to multiple targets based on configurable criteria."""
    def __init__(self, criteria: str = 'max_score'):
        self.criteria = criteria  # 'max_score', 'min_time', etc.

    def compute_cost_matrix(self, beams: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> np.ndarray:
        # Lower cost is better for assignment
        n_beams = len(beams)
        n_targets = len(targets)
        cost = np.zeros((n_beams, n_targets))
        for i, beam in enumerate(beams):
            for j, tgt in enumerate(targets):
                # Example: cost = inverse of priority * (required_power / available_power)
                priority = tgt.get('priority_score', 1.0)
                req_power = tgt.get('required_power_w', 1.0)
                avail_power = beam.get('power_w', 1.0)
                if avail_power < req_power:
                    cost[i, j] = 1e6  # infeasible
                else:
                    cost[i, j] = 1.0 / (priority + 1e-3)
        return cost

    def optimize(self, beams: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        cost = self.compute_cost_matrix(beams, targets)
        row_ind, col_ind = linear_sum_assignment(cost)
        assignments = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1e5:
                assignments.append({'beam': beams[i]['id'], 'target': targets[j]['id'], 'cost': float(cost[i, j])})
        return {
            'assignments': assignments,
            'cost_matrix': cost,
            'unassigned_beams': [beams[i]['id'] for i in range(len(beams)) if i not in row_ind],
            'unassigned_targets': [targets[j]['id'] for j in range(len(targets)) if j not in col_ind]
        }
