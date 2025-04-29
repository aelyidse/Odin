import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import linear_sum_assignment

class BeamQualityMetrics:
    """Real-time beam quality metrics calculator with Strehl ratio."""
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_strehl_ratio(self, wavefront_error: np.ndarray) -> float:
        """
        Calculate Strehl ratio from wavefront error.
        
        Args:
            wavefront_error: Array of wavefront phase errors in radians
            
        Returns:
            Strehl ratio (0-1, where 1 is perfect)
        """
        # Strehl ratio approximation: exp(-σ²)
        # where σ² is the variance of the wavefront error in radians²
        variance = np.var(wavefront_error)
        strehl_ratio = np.exp(-variance)
        return float(strehl_ratio)
    
    def calculate_beam_quality(self, 
                              beam_profile: np.ndarray, 
                              wavefront_error: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive beam quality metrics.
        
        Args:
            beam_profile: 2D array of beam intensity profile
            wavefront_error: Optional 2D array of wavefront error in radians
            
        Returns:
            Dictionary of beam quality metrics
        """
        # Normalize beam profile
        normalized_profile = beam_profile / np.max(beam_profile)
        
        # Calculate beam width (second moment)
        y_indices, x_indices = np.indices(normalized_profile.shape)
        x_center = np.sum(x_indices * normalized_profile) / np.sum(normalized_profile)
        y_center = np.sum(y_indices * normalized_profile) / np.sum(normalized_profile)
        
        x_variance = np.sum(((x_indices - x_center) ** 2) * normalized_profile) / np.sum(normalized_profile)
        y_variance = np.sum(((y_indices - y_center) ** 2) * normalized_profile) / np.sum(normalized_profile)
        
        beam_width = 2 * np.sqrt(x_variance + y_variance)
        
        # Calculate M² factor (beam quality factor)
        # Simplified calculation - in practice would need far-field measurements
        m_squared = np.sqrt(x_variance * y_variance) * 4 / beam_width
        
        # Calculate power in bucket (fraction of power within central region)
        bucket_radius = int(beam_width / 2)
        y_grid, x_grid = np.ogrid[:normalized_profile.shape[0], :normalized_profile.shape[1]]
        mask = ((x_grid - x_center)**2 + (y_grid - y_center)**2) <= bucket_radius**2
        power_in_bucket = np.sum(normalized_profile[mask]) / np.sum(normalized_profile)
        
        # Calculate Strehl ratio if wavefront error is provided
        strehl_ratio = 1.0
        if wavefront_error is not None:
            strehl_ratio = self.calculate_strehl_ratio(wavefront_error)
        
        metrics = {
            'strehl_ratio': strehl_ratio,
            'beam_width': float(beam_width),
            'm_squared': float(m_squared),
            'power_in_bucket': float(power_in_bucket),
            'centroid_x': float(x_center),
            'centroid_y': float(y_center)
        }
        
        # Store metrics in history
        self.metrics_history.append(metrics.copy())
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
            
        return metrics
    
    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get history of beam quality metrics."""
        return self.metrics_history
    
    def get_average_metrics(self, window: int = 10) -> Dict[str, float]:
        """Get average metrics over recent history window."""
        if not self.metrics_history:
            return {}
            
        window = min(window, len(self.metrics_history))
        recent = self.metrics_history[-window:]
        
        avg_metrics = {}
        for key in recent[0].keys():
            avg_metrics[key] = sum(m[key] for m in recent) / window
            
        return avg_metrics

class AdaptiveBeamAllocator:
    """
    Adaptive beam allocation algorithm with dynamic priority adjustment.
    
    Optimizes allocation of beams to targets while dynamically adjusting priorities
    based on engagement history, target characteristics, and system performance.
    """
    def __init__(self, 
                 priority_decay_factor: float = 0.95,
                 success_boost_factor: float = 1.2,
                 max_priority: float = 10.0,
                 allocation_strategy: str = 'balanced'):
        """
        Initialize adaptive beam allocator.
        
        Args:
            priority_decay_factor: Factor to decay priorities over time (0-1)
            success_boost_factor: Factor to boost priority after successful engagement
            max_priority: Maximum priority value
            allocation_strategy: Strategy for allocation ('balanced', 'power_efficient', 'max_coverage')
        """
        self.priority_decay_factor = priority_decay_factor
        self.success_boost_factor = success_boost_factor
        self.max_priority = max_priority
        self.allocation_strategy = allocation_strategy
        
        # Track engagement history
        self.target_history: Dict[str, Dict[str, Any]] = {}
        self.beam_history: Dict[str, Dict[str, Any]] = {}
        self.allocation_history: List[Dict[str, Any]] = []
        
    def compute_dynamic_priorities(self, 
                                  targets: List[Dict[str, Any]], 
                                  system_state: Dict[str, Any]) -> List[float]:
        """
        Compute dynamic priorities for targets based on multiple factors.
        
        Args:
            targets: List of target dictionaries
            system_state: Current system state information
            
        Returns:
            List of priority values corresponding to targets
        """
        priorities = []
        
        for target in targets:
            target_id = target['id']
            base_priority = target.get('priority_score', 1.0)
            
            # Initialize target history if new
            if target_id not in self.target_history:
                self.target_history[target_id] = {
                    'engagement_count': 0,
                    'success_rate': 0.0,
                    'time_since_last': system_state.get('current_time', 0),
                    'cumulative_priority': base_priority
                }
            
            # Retrieve history
            history = self.target_history[target_id]
            
            # Calculate time factor (higher priority for neglected targets)
            current_time = system_state.get('current_time', 0)
            time_factor = min(3.0, (current_time - history['time_since_last']) / 10.0)
            
            # Calculate success factor (prioritize targets with lower success rates)
            success_factor = 1.0
            if history['engagement_count'] > 0:
                success_factor = 2.0 - history['success_rate']
            
            # Calculate urgency factor
            urgency = target.get('urgency', 1.0)
            
            # Calculate final priority
            dynamic_priority = base_priority * time_factor * success_factor * urgency
            
            # Apply system-wide priority adjustments
            if 'resource_constraints' in system_state:
                resource_factor = min(1.5, system_state['resource_constraints'])
                dynamic_priority *= resource_factor
                
            # Ensure priority is within bounds
            dynamic_priority = min(self.max_priority, dynamic_priority)
            
            priorities.append(dynamic_priority)
            
        return priorities
    
    def compute_cost_matrix(self, 
                           beams: List[Dict[str, Any]], 
                           targets: List[Dict[str, Any]],
                           dynamic_priorities: List[float],
                           system_state: Dict[str, Any]) -> np.ndarray:
        """
        Compute cost matrix for beam-target assignment with dynamic priorities.
        
        Args:
            beams: List of beam dictionaries
            targets: List of target dictionaries
            dynamic_priorities: Dynamic priority values for targets
            system_state: Current system state
            
        Returns:
            Cost matrix for assignment algorithm
        """
        n_beams = len(beams)
        n_targets = len(targets)
        cost = np.ones((n_beams, n_targets)) * 1e6  # Initialize as infeasible
        
        for i, beam in enumerate(beams):
            beam_power = beam.get('power_w', 1.0)
            beam_quality = beam.get('beam_quality', 1.0)
            
            # Use Strehl ratio if available
            if 'wavefront_error' in beam:
                strehl = self.beam_quality_metrics.calculate_strehl_ratio(beam['wavefront_error'])
                beam_quality = strehl
            
            for j, target in enumerate(targets):
                # Skip if beam cannot physically engage target
                if not self._can_engage(beam, target, system_state):
                    continue
                    
                # Calculate engagement cost factors
                req_power = target.get('required_power_w', 1.0)
                distance = target.get('distance_km', 1.0)
                
                # Power efficiency factor
                if beam_power < req_power:
                    continue  # Infeasible assignment
                power_efficiency = req_power / beam_power
                
                # Distance factor
                distance_factor = min(5.0, distance)
                
                # Priority factor (higher priority = lower cost)
                priority_factor = 1.0 / (dynamic_priorities[j] + 0.1)
                
                # Beam quality factor
                quality_factor = 1.0 / beam_quality
                
                # Calculate final cost based on strategy
                if self.allocation_strategy == 'power_efficient':
                    cost[i, j] = priority_factor * (1.0 - power_efficiency) * quality_factor
                elif self.allocation_strategy == 'max_coverage':
                    cost[i, j] = priority_factor * distance_factor
                else:  # balanced
                    cost[i, j] = priority_factor * distance_factor * (1.0 - power_efficiency) * quality_factor
        
        return cost
    
    def _can_engage(self, beam: Dict[str, Any], target: Dict[str, Any], system_state: Dict[str, Any]) -> bool:
        """Check if beam can physically engage target."""
        # Check power requirements
        if beam.get('power_w', 0) < target.get('required_power_w', float('inf')):
            return False
            
        # Check range limitations
        max_range = beam.get('max_range_km', float('inf'))
        target_distance = target.get('distance_km', float('inf'))
        if target_distance > max_range:
            return False
            
        # Check beam steering limitations
        if 'position' in beam and 'position' in target:
            # Simple angular check
            return True  # Simplified - would check angular limits in real implementation
            
        return True
    
    def optimize(self, 
                beams: List[Dict[str, Any]], 
                targets: List[Dict[str, Any]],
                system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize beam allocation with dynamic priority adjustment.
        
        Args:
            beams: List of beam dictionaries
            targets: List of target dictionaries
            system_state: Current system state
            
        Returns:
            Dictionary with assignment results
        """
        # Compute dynamic priorities
        dynamic_priorities = self.compute_dynamic_priorities(targets, system_state)
        
        # Compute cost matrix
        cost = self.compute_cost_matrix(beams, targets, dynamic_priorities, system_state)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Create assignment list
        assignments = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1e5:  # Only include feasible assignments
                assignments.append({
                    'beam': beams[i]['id'],
                    'target': targets[j]['id'],
                    'cost': float(cost[i, j]),
                    'priority': float(dynamic_priorities[j])
                })
                
                # Update target history
                target_id = targets[j]['id']
                self.target_history[target_id]['time_since_last'] = system_state.get('current_time', 0)
                self.target_history[target_id]['engagement_count'] += 1
        
        # Store allocation history
        self.allocation_history.append({
            'time': system_state.get('current_time', 0),
            'assignments': assignments.copy()
        })
        
        # Trim history if needed
        if len(self.allocation_history) > 100:
            self.allocation_history.pop(0)
        
        return {
            'assignments': assignments,
            'dynamic_priorities': list(zip([t['id'] for t in targets], dynamic_priorities)),
            'cost_matrix': cost,
            'unassigned_beams': [beams[i]['id'] for i in range(len(beams)) if i not in row_ind],
            'unassigned_targets': [targets[j]['id'] for j in range(len(targets)) if j not in col_ind],
            'beam_quality_metrics': beam_metrics
        }
    
    def update_engagement_results(self, 
                                 engagement_results: List[Dict[str, Any]],
                                 system_state: Dict[str, Any]) -> None:
        """
        Update internal state based on engagement results.
        
        Args:
            engagement_results: List of engagement result dictionaries
            system_state: Current system state
        """
        for result in engagement_results:
            target_id = result.get('target_id')
            success = result.get('success', False)
            
            if target_id in self.target_history:
                history = self.target_history[target_id]
                
                # Update success rate
                prev_successes = history['success_rate'] * history['engagement_count']
                history['engagement_count'] += 1
                history['success_rate'] = (prev_successes + (1 if success else 0)) / history['engagement_count']
                
                # Adjust priority based on success
                if success:
                    history['cumulative_priority'] *= self.priority_decay_factor
                else:
                    history['cumulative_priority'] *= self.success_boost_factor
                    history['cumulative_priority'] = min(self.max_priority, history['cumulative_priority'])
    
    def get_allocation_metrics(self) -> Dict[str, Any]:
        """Get metrics about allocation performance."""
        if not self.allocation_history:
            return {'allocation_count': 0}
            
        # Calculate metrics from history
        total_assignments = sum(len(h['assignments']) for h in self.allocation_history)
        unique_targets = set()
        for h in self.allocation_history:
            for a in h['assignments']:
                unique_targets.add(a['target'])
                
        return {
            'allocation_count': len(self.allocation_history),
            'avg_assignments_per_allocation': total_assignments / len(self.allocation_history),
            'unique_targets_engaged': len(unique_targets),
            'target_history': {tid: {k: v for k, v in h.items() if k != 'time_since_last'} 
                              for tid, h in self.target_history.items()}
        }