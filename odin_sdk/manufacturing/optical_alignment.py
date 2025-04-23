import numpy as np
from typing import Dict, Any, List, Optional

class AlignmentStep:
    """Represents a single step in the precision alignment procedure."""
    def __init__(self, description: str, target_tolerance_um: float, measurement_method: str, adjustment_method: str):
        self.description = description
        self.target_tolerance_um = target_tolerance_um
        self.measurement_method = measurement_method
        self.adjustment_method = adjustment_method
    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'target_tolerance_um': self.target_tolerance_um,
            'measurement_method': self.measurement_method,
            'adjustment_method': self.adjustment_method
        }

class OpticalAlignmentProcedure:
    """Defines a precision alignment procedure for optical components."""
    def __init__(self, component: str, steps: Optional[List[AlignmentStep]] = None):
        self.component = component
        self.steps = steps or []
    def add_step(self, step: AlignmentStep):
        self.steps.append(step)
    def as_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'steps': [s.to_dict() for s in self.steps]
        }
    def describe(self) -> str:
        lines = [f"Alignment Procedure for {self.component}:"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step.description} (Tolerance: {step.target_tolerance_um} um, Measurement: {step.measurement_method}, Adjustment: {step.adjustment_method})")
        return '\n'.join(lines)

# Example: Alignment procedure for a fiber collimator
fiber_collimator_alignment = OpticalAlignmentProcedure('fiber collimator')
fiber_collimator_alignment.add_step(AlignmentStep(
    description='Initial coarse alignment using visible pilot beam',
    target_tolerance_um=100.0,
    measurement_method='visual inspection or CCD camera',
    adjustment_method='manual micrometer stages'
))
fiber_collimator_alignment.add_step(AlignmentStep(
    description='Fine x/y/z alignment for beam centration',
    target_tolerance_um=5.0,
    measurement_method='beam profiler or knife-edge scan',
    adjustment_method='precision translation stages'
))
fiber_collimator_alignment.add_step(AlignmentStep(
    description='Optimize angular alignment (tip/tilt)',
    target_tolerance_um=2.0,
    measurement_method='far-field pattern analysis',
    adjustment_method='goniometer or tip/tilt stage'
))
fiber_collimator_alignment.add_step(AlignmentStep(
    description='Lock alignment and verify with power meter and wavefront sensor',
    target_tolerance_um=1.0,
    measurement_method='power meter, wavefront sensor',
    adjustment_method='locking screws, adhesive, or active feedback'
))

# Example usage:
# print(fiber_collimator_alignment.describe())
