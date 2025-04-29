import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import minimize

class PowerDistributionOptimizer:
    """
    Optimizes power distribution across multiple simultaneous targets based on
    priority, atmospheric conditions, and system constraints.
    
    Features:
    - Dynamic power allocation based on target priority and engagement requirements
    - Atmospheric loss compensation with predictive modeling
    - Real-time power redistribution during engagement
    - Thermal and power budget management
    """
    
    def __init__(self, 
                 max_total_power_w: float = 1000.0,
                 min_effective_power_w: float = 50.0,
                 thermal_limit_w: float = 1200.0,
                 efficiency_threshold: float = 0.7,
                 allocation_strategy: str = 'priority_weighted'):
        """
        Initialize power distribution optimizer.
        
        Args:
            max_total_power_w: Maximum total output power available (Watts)
            min_effective_power_w: Minimum effective power required per target (Watts)
            thermal_limit_w: System thermal dissipation limit (Watts)
            efficiency_threshold: Minimum acceptable power delivery efficiency
            allocation_strategy: Power allocation strategy ('priority_weighted', 
                                'equal', 'max_coverage', 'efficiency_optimized')
        """
        self.max_total_power_w = max_total_power_w
        self.min_effective_power_w = min_effective_power_w
        self.thermal_limit_w = thermal_limit_w
        self.efficiency_threshold = efficiency_threshold
        self.allocation_strategy = allocation_strategy
        
        # Internal state
        self.current_allocation = {}  # target_id -> power allocation
        self.power_history = []  # Track allocation history
        self.system_efficiency = 1.0  # Current system efficiency
        self.thermal_load = 0.0  # Current thermal load
        
    def calculate_atmospheric_losses(self, 
                                    targets: List[Dict[str, Any]], 
                                    atmospheric_conditions: Dict[str, Any]) -> List[float]:
        """
        Calculate atmospheric transmission factors for each target.
        
        Args:
            targets: List of target dictionaries with position and characteristics
            atmospheric_conditions: Current atmospheric conditions
            
        Returns:
            List of transmission factors (0-1) for each target
        """
        transmission_factors = []
        
        for target in targets:
            # Extract target parameters
            distance_km = target.get('distance_km', 10.0)
            elevation_deg = target.get('elevation_deg', 45.0)
            
            # Extract atmospheric parameters
            visibility_km = atmospheric_conditions.get('visibility_km', 10.0)
            turbulence_cn2 = atmospheric_conditions.get('turbulence_cn2', 1e-14)
            precipitation_rate = atmospheric_conditions.get('precipitation_rate', 0.0)
            
            # Calculate basic Beer-Lambert atmospheric extinction
            extinction_coeff = 3.91 / visibility_km
            
            # Adjust for elevation angle (longer path through atmosphere at low angles)
            if elevation_deg > 0:
                air_mass = 1.0 / np.sin(np.radians(elevation_deg))
            else:
                air_mass = 10.0  # High extinction for negative elevation
                
            # Calculate transmission factor
            transmission = np.exp(-extinction_coeff * distance_km * air_mass)
            
            # Apply additional losses for precipitation if present
            if precipitation_rate > 0:
                rain_loss_factor = np.exp(-0.1 * precipitation_rate * distance_km)
                transmission *= rain_loss_factor
                
            # Apply turbulence-induced losses (simplified model)
            turbulence_loss = 1.0 - 0.2 * np.sqrt(turbulence_cn2 * 1e14) * distance_km / 10.0
            turbulence_loss = max(0.5, turbulence_loss)  # Limit minimum transmission
            
            transmission *= turbulence_loss
            transmission_factors.append(float(transmission))
            
        return transmission_factors
    
    def optimize_power_distribution(self, 
                                   targets: List[Dict[str, Any]], 
                                   atmospheric_conditions: Dict[str, Any],
                                   system_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize power distribution across multiple targets.
        
        Args:
            targets: List of target dictionaries with priority and characteristics
            atmospheric_conditions: Current atmospheric conditions
            system_state: Current system state including available power
            
        Returns:
            Dictionary mapping target IDs to allocated power (Watts)
        """
        n_targets = len(targets)
        if n_targets == 0:
            return {}
            
        # Calculate atmospheric transmission for each target
        transmission_factors = self.calculate_atmospheric_losses(targets, atmospheric_conditions)
        
        # Extract target priorities
        priorities = [target.get('priority', 1.0) for target in targets]
        target_ids = [target.get('id', f"target_{i}") for i, target in enumerate(targets)]
        
        # Available power may be constrained by system state
        available_power = min(
            self.max_total_power_w,
            system_state.get('available_power_w', self.max_total_power_w)
        )
        
        # Calculate power allocation based on selected strategy
        if self.allocation_strategy == 'equal':
            # Equal power to all targets
            base_power = available_power / n_targets
            power_allocation = {target_id: base_power for target_id in target_ids}
            
        elif self.allocation_strategy == 'priority_weighted':
            # Weighted by priority
            total_priority = sum(priorities)
            power_allocation = {
                target_id: available_power * (priority / total_priority)
                for target_id, priority in zip(target_ids, priorities)
            }
            
        elif self.allocation_strategy == 'efficiency_optimized':
            # Optimize for maximum power delivery efficiency
            power_allocation = self._optimize_for_efficiency(
                target_ids, priorities, transmission_factors, available_power
            )
            
        elif self.allocation_strategy == 'max_coverage':
            # Ensure maximum number of targets receive minimum effective power
            power_allocation = self._optimize_for_coverage(
                target_ids, priorities, transmission_factors, available_power
            )
        
        else:
            # Default to priority-weighted
            total_priority = sum(priorities)
            power_allocation = {
                target_id: available_power * (priority / total_priority)
                for target_id, priority in zip(target_ids, priorities)
            }
        
        # Ensure minimum effective power requirements are met where possible
        power_allocation = self._adjust_for_minimum_effective_power(
            power_allocation, transmission_factors, target_ids
        )
        
        # Calculate system metrics
        self._update_system_metrics(power_allocation, transmission_factors, target_ids)
        
        # Store current allocation
        self.current_allocation = power_allocation.copy()
        self.power_history.append(power_allocation.copy())
        if len(self.power_history) > 100:
            self.power_history.pop(0)
            
        return power_allocation
    
    def _optimize_for_efficiency(self, 
                               target_ids: List[str], 
                               priorities: List[float],
                               transmission_factors: List[float], 
                               available_power: float) -> Dict[str, float]:
        """
        Optimize power allocation to maximize overall power delivery efficiency.
        
        Uses transmission factors and priorities to allocate more power to
        targets with better atmospheric transmission.
        """
        # Combine priority and transmission into a single efficiency metric
        efficiency_metrics = [p * t for p, t in zip(priorities, transmission_factors)]
        total_efficiency = sum(efficiency_metrics)
        
        # Allocate power proportionally to efficiency metric
        power_allocation = {
            target_id: available_power * (metric / total_efficiency)
            for target_id, metric in zip(target_ids, efficiency_metrics)
        }
        
        return power_allocation
    
    def _optimize_for_coverage(self, 
                             target_ids: List[str], 
                             priorities: List[float],
                             transmission_factors: List[float], 
                             available_power: float) -> Dict[str, float]:
        """
        Optimize to ensure maximum number of targets receive minimum effective power.
        
        Prioritizes targets by priority and allocates minimum effective power to
        as many targets as possible.
        """
        # Calculate effective power needed at source to achieve minimum at target
        required_powers = [
            self.min_effective_power_w / t if t > 0.1 else float('inf')
            for t in transmission_factors
        ]
        
        # Sort targets by priority (highest first)
        sorted_indices = np.argsort(priorities)[::-1]
        
        # Allocate power to targets in priority order until budget is exhausted
        remaining_power = available_power
        power_allocation = {target_id: 0.0 for target_id in target_ids}
        
        for idx in sorted_indices:
            target_id = target_ids[idx]
            required_power = required_powers[idx]
            
            if required_power <= remaining_power:
                # Allocate required power to this target
                power_allocation[target_id] = required_power
                remaining_power -= required_power
            else:
                # Allocate remaining power to this target
                power_allocation[target_id] = remaining_power
                remaining_power = 0
                break
                
        # Distribute any remaining power proportionally to priority
        if remaining_power > 0:
            # Calculate total priority of targets that received power
            active_targets = [i for i, tid in enumerate(target_ids) if power_allocation[tid] > 0]
            active_priorities = [priorities[i] for i in active_targets]
            total_active_priority = sum(active_priorities)
            
            if total_active_priority > 0:
                # Distribute remaining power proportionally
                for i, target_id in enumerate(target_ids):
                    if power_allocation[target_id] > 0:
                        power_allocation[target_id] += remaining_power * (priorities[i] / total_active_priority)
        
        return power_allocation
    
    def _adjust_for_minimum_effective_power(self, 
                                          power_allocation: Dict[str, float],
                                          transmission_factors: List[float],
                                          target_ids: List[str]) -> Dict[str, float]:
        """
        Adjust power allocation to ensure minimum effective power requirements are met.
        
        Removes targets that cannot meet minimum requirements and redistributes power.
        """
        adjusted_allocation = power_allocation.copy()
        
        # Check which targets will receive effective power below minimum
        below_minimum = []
        for i, target_id in enumerate(target_ids):
            effective_power = adjusted_allocation[target_id] * transmission_factors[i]
            if effective_power < self.min_effective_power_w and adjusted_allocation[target_id] > 0:
                below_minimum.append((i, target_id))
        
        # If any targets are below minimum, redistribute their power
        if below_minimum:
            # Remove power from targets below minimum
            freed_power = 0
            for i, target_id in below_minimum:
                freed_power += adjusted_allocation[target_id]
                adjusted_allocation[target_id] = 0
            
            # Redistribute freed power to remaining targets
            remaining_targets = [tid for tid in target_ids if adjusted_allocation[tid] > 0]
            if remaining_targets:
                additional_power = freed_power / len(remaining_targets)
                for tid in remaining_targets:
                    adjusted_allocation[tid] += additional_power
        
        return adjusted_allocation
    
    def _update_system_metrics(self, 
                             power_allocation: Dict[str, float],
                             transmission_factors: List[float],
                             target_ids: List[str]) -> None:
        """Update system efficiency and thermal load metrics."""
        total_power = sum(power_allocation.values())
        
        # Calculate effective delivered power
        delivered_power = sum(
            power_allocation[tid] * transmission_factors[i]
            for i, tid in enumerate(target_ids)
        )
        
        # Update system efficiency
        if total_power > 0:
            self.system_efficiency = delivered_power / total_power
        else:
            self.system_efficiency = 1.0
            
        # Update thermal load (simplified model)
        self.thermal_load = total_power * (1.0 - self.system_efficiency * 0.8)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        return {
            'total_allocated_power_w': sum(self.current_allocation.values()),
            'system_efficiency': self.system_efficiency,
            'thermal_load_w': self.thermal_load,
            'thermal_headroom_w': self.thermal_limit_w - self.thermal_load,
            'power_utilization': sum(self.current_allocation.values()) / self.max_total_power_w
        }
    
    def reallocate_on_target_loss(self, lost_target_id: str) -> Dict[str, float]:
        """
        Reallocate power when a target is lost or engagement ends.
        
        Args:
            lost_target_id: ID of target that was lost
            
        Returns:
            Updated power allocation
        """
        if lost_target_id not in self.current_allocation:
            return self.current_allocation
            
        # Get power that was allocated to lost target
        freed_power = self.current_allocation[lost_target_id]
        
        # Remove lost target from allocation
        updated_allocation = {k: v for k, v in self.current_allocation.items() if k != lost_target_id}
        
        # Redistribute freed power proportionally to remaining targets
        if updated_allocation and freed_power > 0:
            total_remaining_power = sum(updated_allocation.values())
            if total_remaining_power > 0:
                for target_id in updated_allocation:
                    proportion = updated_allocation[target_id] / total_remaining_power
                    updated_allocation[target_id] += freed_power * proportion
        
        self.current_allocation = updated_allocation
        return updated_allocation