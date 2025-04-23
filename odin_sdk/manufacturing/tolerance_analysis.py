import numpy as np
from typing import Dict, Any, List, Tuple

class ToleranceSpec:
    """Defines nominal value and tolerance for a component parameter."""
    def __init__(self, name: str, nominal: float, tol_minus: float, tol_plus: float):
        self.name = name
        self.nominal = nominal
        self.tol_minus = tol_minus
        self.tol_plus = tol_plus
    def limits(self) -> Tuple[float, float]:
        return (self.nominal - self.tol_minus, self.nominal + self.tol_plus)
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'nominal': self.nominal,
            'tol_minus': self.tol_minus,
            'tol_plus': self.tol_plus
        }

class ToleranceAnalysisResult:
    """Stores results of a tolerance analysis simulation."""
    def __init__(self, parameter_results: Dict[str, Dict[str, float]], yield_percent: float):
        self.parameter_results = parameter_results
        self.yield_percent = yield_percent
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameter_results': self.parameter_results,
            'yield_percent': self.yield_percent
        }

class ToleranceAnalyzer:
    """Performs Monte Carlo tolerance analysis for manufacturing variations."""
    def __init__(self, specs: List[ToleranceSpec], n_samples: int = 10000):
        self.specs = specs
        self.n_samples = n_samples

    def analyze(self, test_fn: callable) -> ToleranceAnalysisResult:
        """
        Args:
            test_fn: Callable that takes a dict of {param: value} and returns True if part passes, False otherwise.
        Returns:
            ToleranceAnalysisResult
        """
        samples = {}
        for spec in self.specs:
            samples[spec.name] = np.random.uniform(spec.limits()[0], spec.limits()[1], self.n_samples)
        pass_mask = np.zeros(self.n_samples, dtype=bool)
        for i in range(self.n_samples):
            params = {name: samples[name][i] for name in samples}
            pass_mask[i] = test_fn(params)
        parameter_results = {}
        for spec in self.specs:
            param_samples = samples[spec.name][pass_mask]
            parameter_results[spec.name] = {
                'mean': float(np.mean(param_samples)) if len(param_samples) > 0 else float('nan'),
                'std': float(np.std(param_samples)) if len(param_samples) > 0 else float('nan'),
                'min': float(np.min(param_samples)) if len(param_samples) > 0 else float('nan'),
                'max': float(np.max(param_samples)) if len(param_samples) > 0 else float('nan'),
            }
        yield_percent = 100.0 * np.sum(pass_mask) / self.n_samples
        return ToleranceAnalysisResult(parameter_results, yield_percent)

# Example usage:
# specs = [ToleranceSpec('core_diameter_um', 10.0, 0.2, 0.2), ToleranceSpec('NA', 0.14, 0.01, 0.01)]
# def test_fn(params): return params['core_diameter_um'] > 9.8 and params['NA'] > 0.13
# analyzer = ToleranceAnalyzer(specs, n_samples=10000)
# result = analyzer.analyze(test_fn)
# print(result.to_dict())
