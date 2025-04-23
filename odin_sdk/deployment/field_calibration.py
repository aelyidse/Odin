import numpy as np
from typing import Dict, Any, List, Optional, Callable

class CalibrationStep:
    """Represents a single step in a field calibration procedure."""
    def __init__(self, description: str, method: str, acceptance_criteria: str):
        self.description = description
        self.method = method  # e.g., 'power meter', 'wavefront sensor', 'reference target'
        self.acceptance_criteria = acceptance_criteria
    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'method': self.method,
            'acceptance_criteria': self.acceptance_criteria
        }

class FieldCalibrationProcedure:
    """Defines a field calibration procedure for a system or subsystem."""
    def __init__(self, component: str, steps: Optional[List[CalibrationStep]] = None):
        self.component = component
        self.steps = steps or []
    def add_step(self, step: CalibrationStep):
        self.steps.append(step)
    def as_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'steps': [s.to_dict() for s in self.steps]
        }
    def describe(self) -> str:
        lines = [f"Field Calibration Procedure for {self.component}:"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step.description} (Method: {step.method}, Acceptance: {step.acceptance_criteria})")
        return '\n'.join(lines)

# Example: Laser pointing calibration
laser_pointing_cal = FieldCalibrationProcedure('laser pointing')
laser_pointing_cal.add_step(CalibrationStep(
    description='Set up reference target at known distance',
    method='survey equipment, rangefinder',
    acceptance_criteria='Target distance error < 0.5 m'
))
laser_pointing_cal.add_step(CalibrationStep(
    description='Align laser to reference target',
    method='beam profiler, visible alignment laser',
    acceptance_criteria='Beam centroid within 1 mm of reference mark'
))
laser_pointing_cal.add_step(CalibrationStep(
    description='Verify pointing repeatability',
    method='multiple firings, record centroid',
    acceptance_criteria='Std deviation < 0.5 mm over 10 shots'
))

# Diagnostic tools
class DiagnosticTool:
    """Base class for field diagnostic tools."""
    def __init__(self, name: str, test_fn: Callable[..., Any]):
        self.name = name
        self.test_fn = test_fn
    def run(self, *args, **kwargs) -> Any:
        return self.test_fn(*args, **kwargs)

# Example: Diagnostic tool for beam quality

def beam_quality_diagnostic(beam_profile: np.ndarray) -> Dict[str, float]:
    intensity = beam_profile
    total_power = np.sum(intensity)
    centroid_x = np.sum(intensity * np.arange(intensity.shape[1])) / total_power
    centroid_y = np.sum(intensity.T * np.arange(intensity.shape[0])) / total_power
    var_x = np.sum(intensity * (np.arange(intensity.shape[1]) - centroid_x)**2) / total_power
    var_y = np.sum(intensity.T * (np.arange(intensity.shape[0]) - centroid_y)**2) / total_power
    m2 = np.sqrt(var_x * var_y)
    return {'centroid_x': float(centroid_x), 'centroid_y': float(centroid_y), 'M2': float(m2)}

beam_quality_tool = DiagnosticTool('beam_quality', beam_quality_diagnostic)

# Example usage:
# print(laser_pointing_cal.describe())
# result = beam_quality_tool.run(np.random.rand(100,100))
# print(result)
