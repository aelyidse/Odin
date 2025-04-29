import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import norm

@dataclass
class OpticalComponentSpec:
    """Specification for an optical component with manufacturing tolerances."""
    name: str
    nominal_value: float
    tolerance: float  # ± tolerance
    unit: str
    distribution: str = "uniform"  # "uniform", "normal", "triangular"
    
    def limits(self) -> Tuple[float, float]:
        """Return the min and max values based on tolerance."""
        return (self.nominal_value - self.tolerance, self.nominal_value + self.tolerance)
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate random samples according to the specified distribution."""
        if self.distribution == "uniform":
            return np.random.uniform(
                self.nominal_value - self.tolerance,
                self.nominal_value + self.tolerance,
                n_samples
            )
        elif self.distribution == "normal":
            # 3-sigma rule: tolerance represents 3 standard deviations
            sigma = self.tolerance / 3
            return np.random.normal(self.nominal_value, sigma, n_samples)
        elif self.distribution == "triangular":
            return np.random.triangular(
                self.nominal_value - self.tolerance,
                self.nominal_value,
                self.nominal_value + self.tolerance,
                n_samples
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")


class OpticalToleranceAnalyzer:
    """
    Analyzes manufacturing tolerances for optical components using Monte Carlo simulation.
    
    This tool helps determine how manufacturing variations affect optical system performance
    and identifies critical parameters that require tighter tolerances.
    """
    
    def __init__(self, specs: List[OpticalComponentSpec], n_samples: int = 10000):
        """
        Initialize the optical tolerance analyzer.
        
        Args:
            specs: List of optical component specifications
            n_samples: Number of Monte Carlo samples to generate
        """
        self.specs = specs
        self.n_samples = n_samples
        self.results = None
        self.parameter_sensitivities = {}
    
    def analyze(self, performance_function: Callable[[Dict[str, float]], float], 
                threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo analysis to determine system performance sensitivity to tolerances.
        
        Args:
            performance_function: Function that takes component parameters and returns a performance metric
            threshold: Optional threshold for acceptable performance
            
        Returns:
            Dictionary with analysis results
        """
        # Generate samples for each parameter
        samples = {}
        for spec in self.specs:
            samples[spec.name] = spec.generate_samples(self.n_samples)
        
        # Evaluate performance for each sample set
        performance_values = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            params = {name: samples[name][i] for name in samples}
            performance_values[i] = performance_function(params)
        
        # Determine pass/fail if threshold is provided
        if threshold is not None:
            pass_mask = performance_values >= threshold
            yield_rate = 100.0 * np.sum(pass_mask) / self.n_samples
        else:
            pass_mask = np.ones(self.n_samples, dtype=bool)
            yield_rate = 100.0
        
        # Calculate statistics for each parameter
        parameter_results = {}
        for spec in self.specs:
            param_samples = samples[spec.name][pass_mask]
            parameter_results[spec.name] = {
                'nominal': spec.nominal_value,
                'tolerance': spec.tolerance,
                'unit': spec.unit,
                'mean': float(np.mean(param_samples)) if len(param_samples) > 0 else float('nan'),
                'std': float(np.std(param_samples)) if len(param_samples) > 0 else float('nan'),
                'min': float(np.min(param_samples)) if len(param_samples) > 0 else float('nan'),
                'max': float(np.max(param_samples)) if len(param_samples) > 0 else float('nan'),
            }
        
        # Calculate performance statistics
        perf_results = {
            'mean': float(np.mean(performance_values)),
            'std': float(np.std(performance_values)),
            'min': float(np.min(performance_values)),
            'max': float(np.max(performance_values)),
            'yield_rate': float(yield_rate)
        }
        
        # Calculate parameter sensitivities
        self._calculate_sensitivities(samples, performance_values)
        
        # Store and return results
        self.results = {
            'parameters': parameter_results,
            'performance': perf_results,
            'sensitivities': self.parameter_sensitivities,
            'n_samples': self.n_samples
        }
        
        return self.results
    
    def _calculate_sensitivities(self, samples: Dict[str, np.ndarray], performance: np.ndarray):
        """Calculate sensitivity of performance to each parameter."""
        for name, param_samples in samples.items():
            # Calculate correlation coefficient
            correlation = np.corrcoef(param_samples, performance)[0, 1]
            
            # Normalize to get sensitivity (0-100%)
            sensitivity = abs(correlation) * 100
            
            self.parameter_sensitivities[name] = {
                'correlation': float(correlation),
                'sensitivity': float(sensitivity)
            }
    
    def plot_results(self, output_file: Optional[str] = None):
        """
        Generate visualization of the analysis results.
        
        Args:
            output_file: Optional file path to save the plot
        """
        if self.results is None:
            raise ValueError("No analysis results available. Run analyze() first.")
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Optical Component Tolerance Analysis', fontsize=16)
        
        # Plot 1: Parameter sensitivities
        sensitivities = [(k, v['sensitivity']) for k, v in self.parameter_sensitivities.items()]
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        names = [s[0] for s in sensitivities]
        values = [s[1] for s in sensitivities]
        
        axs[0, 0].barh(names, values)
        axs[0, 0].set_xlabel('Sensitivity (%)')
        axs[0, 0].set_title('Parameter Sensitivity')
        
        # Plot 2: Performance histogram
        if 'performance' in self.results:
            perf = self.results['performance']
            axs[0, 1].hist(np.linspace(perf['min'], perf['max'], 50), bins=20, alpha=0.7)
            axs[0, 1].axvline(perf['mean'], color='r', linestyle='--', label=f"Mean: {perf['mean']:.3f}")
            axs[0, 1].set_xlabel('Performance Metric')
            axs[0, 1].set_ylabel('Frequency')
            axs[0, 1].set_title('Performance Distribution')
            axs[0, 1].legend()
        
        # Plot 3: Parameter distributions for top 2 sensitive parameters
        if len(sensitivities) >= 2:
            top_params = [s[0] for s in sensitivities[:2]]
            
            for i, param in enumerate(top_params):
                spec = next((s for s in self.specs if s.name == param), None)
                if spec:
                    x = np.linspace(
                        spec.nominal_value - 1.5*spec.tolerance,
                        spec.nominal_value + 1.5*spec.tolerance,
                        100
                    )
                    
                    axs[1, i].hist(
                        np.linspace(
                            self.results['parameters'][param]['min'],
                            self.results['parameters'][param]['max'],
                            50
                        ),
                        bins=20, 
                        alpha=0.7
                    )
                    
                    axs[1, i].axvline(
                        spec.nominal_value, 
                        color='r', 
                        linestyle='--', 
                        label=f"Nominal: {spec.nominal_value}"
                    )
                    
                    axs[1, i].set_xlabel(f"{param} ({spec.unit})")
                    axs[1, i].set_ylabel('Frequency')
                    axs[1, i].set_title(f'Distribution of {param}')
                    axs[1, i].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def generate_report(self) -> str:
        """Generate a text report summarizing the analysis results."""
        if self.results is None:
            return "No analysis results available. Run analyze() first."
        
        report = []
        report.append("=== Optical Component Tolerance Analysis Report ===")
        report.append(f"Number of samples: {self.n_samples}")
        report.append("")
        
        # Performance summary
        perf = self.results['performance']
        report.append("Performance Summary:")
        report.append(f"  Mean: {perf['mean']:.6f}")
        report.append(f"  Standard Deviation: {perf['std']:.6f}")
        report.append(f"  Range: [{perf['min']:.6f}, {perf['max']:.6f}]")
        report.append(f"  Yield Rate: {perf['yield_rate']:.2f}%")
        report.append("")
        
        # Parameter sensitivities
        report.append("Parameter Sensitivities (most to least critical):")
        sensitivities = [(k, v['sensitivity']) for k, v in self.parameter_sensitivities.items()]
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        for name, sensitivity in sensitivities:
            report.append(f"  {name}: {sensitivity:.2f}%")
        report.append("")
        
        # Parameter statistics
        report.append("Parameter Statistics:")
        for name, param in self.results['parameters'].items():
            report.append(f"  {name} ({param['unit']}):")
            report.append(f"    Nominal: {param['nominal']:.6f} ± {param['tolerance']:.6f}")
            report.append(f"    Mean: {param['mean']:.6f}")
            report.append(f"    Std Dev: {param['std']:.6f}")
            report.append(f"    Range: [{param['min']:.6f}, {param['max']:.6f}]")
        
        return "\n".join(report)


# Example usage:
if __name__ == "__main__":
    # Define optical component specifications
    specs = [
        OpticalComponentSpec("lens_radius", 25.4, 0.1, "mm"),
        OpticalComponentSpec("lens_thickness", 5.0, 0.05, "mm"),
        OpticalComponentSpec("refractive_index", 1.5168, 0.001, "", "normal"),
        OpticalComponentSpec("surface_flatness", 0.25, 0.1, "waves"),
        OpticalComponentSpec("coating_thickness", 0.55, 0.05, "μm")
    ]
    
    # Define a simple performance function (example: optical path difference)
    def optical_performance(params):
        # This is a simplified example - replace with actual optical performance calculation
        opd = (
            params["lens_thickness"] * params["refractive_index"] -
            25.4 * 1.5168  # Nominal path length
        )
        return 1.0 - abs(opd) * 10  # Higher is better
    
    # Run analysis
    analyzer = OpticalToleranceAnalyzer(specs, n_samples=5000)
    results = analyzer.analyze(optical_performance, threshold=0.9)
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Plot results
    analyzer.plot_results()