from typing import Dict, Any, List, Optional

class QAProtocolStep:
    """Represents a single step in a quality assurance protocol."""
    def __init__(self, description: str, method: str, acceptance_criteria: str, documentation_required: bool = True):
        self.description = description
        self.method = method  # e.g., 'visual inspection', 'interferometry', 'power meter', 'burn-in'
        self.acceptance_criteria = acceptance_criteria
        self.documentation_required = documentation_required
    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'method': self.method,
            'acceptance_criteria': self.acceptance_criteria,
            'documentation_required': self.documentation_required
        }

class QATestingProcedure:
    """Defines a full QA testing procedure for a component or assembly."""
    def __init__(self, component: str, steps: Optional[List[QAProtocolStep]] = None):
        self.component = component
        self.steps = steps or []
    def add_step(self, step: QAProtocolStep):
        self.steps.append(step)
    def as_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'steps': [s.to_dict() for s in self.steps]
        }
    def describe(self) -> str:
        lines = [f"QA Protocol for {self.component}:"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step.description} (Method: {step.method}, Acceptance: {step.acceptance_criteria}, Documentation: {step.documentation_required})")
        return '\n'.join(lines)

# Example: QA protocol for a fiber laser module
fiber_laser_qa = QATestingProcedure('fiber laser module')
fiber_laser_qa.add_step(QAProtocolStep(
    description='Visual inspection of fiber and connectors',
    method='microscope, visual',
    acceptance_criteria='No contamination, scratches, or defects visible',
    documentation_required=True
))
fiber_laser_qa.add_step(QAProtocolStep(
    description='End-face geometry and cleave quality',
    method='interferometry, end-face imaging',
    acceptance_criteria='Flatness < 0.5 um, angle < 0.5 deg',
    documentation_required=True
))
fiber_laser_qa.add_step(QAProtocolStep(
    description='Output power and efficiency test',
    method='power meter, calibrated photodiode',
    acceptance_criteria='Power > 1 kW, efficiency > 80%',
    documentation_required=True
))
fiber_laser_qa.add_step(QAProtocolStep(
    description='Burn-in and thermal cycling',
    method='automated burn-in station',
    acceptance_criteria='No failure or degradation after 48h/thermal cycles',
    documentation_required=True
))

# Example usage:
# print(fiber_laser_qa.describe())
