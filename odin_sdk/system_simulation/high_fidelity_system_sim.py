import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json

# Import core system components
from odin_sdk.system_simulation.full_system_sim import OdinFullSystemSimulator
from odin_sdk.system_simulation.performance_metrics import SystemPerformanceMetrics
from odin_sdk.physics_engine.optical_thermal_load import OpticalThermalLoad
from odin_sdk.physics_engine.shack_hartmann import ShackHartmannSensor
from odin_sdk.physics_engine.adaptive_optics_control import DeformableMirror
from odin_sdk.physics_engine.beam_alignment_verification import BeamAlignmentVerification
from odin_sdk.deployment.beam_alignment_calibration import BeamAlignmentCalibration
from odin_sdk.deployment.performance_monitor import PerformanceMonitor, PerformanceDegradationAlert


class HighFidelitySystemSimulator:
    """
    High-fidelity simulation of complete ODIN system performance.
    
    Integrates all subsystems including:
    - Beam alignment and calibration
    - Adaptive optics
    - Thermal effects
    - Target tracking
    - Performance monitoring
    
    Provides comprehensive analysis of system performance under various
    operational conditions and scenarios.
    """
    
    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize high-fidelity system simulator.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for simulation outputs
        """
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize core simulator
        self.core_sim = OdinFullSystemSimulator(
            config_path=config_path,
            log_path=os.path.join(output_dir, "system_log.jsonl")
        )
        
        # Initialize performance metrics
        self.metrics = SystemPerformanceMetrics()
        
        # Initialize subsystem simulators
        self._init_subsystems()
        
        # Simulation state
        self.sim_time = 0.0
        self.sim_step = 0
        self.results_history = []
        
    def _init_subsystems(self):
        """Initialize all subsystem simulators."""
        # Optical subsystems
        self._init_optical_subsystems()
        
        # Thermal subsystems
        self._init_thermal_subsystems()
        
        # Tracking subsystems
        self._init_tracking_subsystems()
        
        # Performance monitoring
        self._init_performance_monitoring()
    
    def _init_optical_subsystems(self):
        """Initialize optical subsystems."""
        # Wavefront sensing
        self.wavefront_sensor = ShackHartmannSensor(
            n_lenslets_x=32,
            n_lenslets_y=32,
            lenslet_pitch_mm=0.3,
            focal_length_mm=5.0,
            detector_pixel_size_um=5.0,
            detector_pixels_x=1024,
            detector_pixels_y=1024,
            noise_std_pix=0.1
        )
        
        # Adaptive optics
        self.deformable_mirror = DeformableMirror(
            n_actuators_x=32,
            n_actuators_y=32,
            max_stroke_um=10.0
        )
        
        # Beam alignment
        self.beam_alignment = BeamAlignmentCalibration(
            wavefront_sensor=self.wavefront_sensor,
            deformable_mirror=self.deformable_mirror,
            target_strehl_ratio=0.9,
            max_iterations=30,
            convergence_threshold=0.001
        )
        
        # Beam verification
        self.beam_verification = BeamAlignmentVerification(
            reference_angles_deg={'x': 0.0, 'y': 0.0},
            max_angle_deviation_mrad=0.2,
            stability_window=30
        )
    
    def _init_thermal_subsystems(self):
        """Initialize thermal subsystems."""
        # Define optical materials for key components
        from dataclasses import dataclass
        
        @dataclass
        class OpticalMaterial:
            name: str
            refractive_index_fn: callable
            absorption_coeff_fn: callable
            thermal_conductivity_fn: callable
            specific_heat_fn: callable
            density: float
            
            def refractive_index(self, wavelength_nm, temp_C):
                return self.refractive_index_fn(wavelength_nm, temp_C)
                
            def absorption_coeff(self, wavelength_nm, temp_C):
                return self.absorption_coeff_fn(wavelength_nm, temp_C)
                
            def thermal_conductivity(self, temp_C):
                return self.thermal_conductivity_fn(temp_C)
                
            def specific_heat(self, temp_C):
                return self.specific_heat_fn(temp_C)
        
        # Define fused silica material
        def fs_index(wl_nm, temp_C):
            # Simplified Sellmeier with thermal dependence
            return 1.45 + 0.00001 * (temp_C - 20)
            
        def fs_absorption(wl_nm, temp_C):
            # Simplified absorption model
            return 0.0001 + 0.00001 * (temp_C - 20)
            
        def fs_conductivity(temp_C):
            return 1.38  # W/(m·K)
            
        def fs_specific_heat(temp_C):
            return 740  # J/(kg·K)
        
        fused_silica = OpticalMaterial(
            name="Fused Silica",
            refractive_index_fn=fs_index,
            absorption_coeff_fn=fs_absorption,
            thermal_conductivity_fn=fs_conductivity,
            specific_heat_fn=fs_specific_heat,
            density=2200  # kg/m³
        )
        
        # Create thermal load simulators for key components
        self.thermal_simulators = {
            "output_window": OpticalThermalLoad(
                material=fused_silica,
                thickness_mm=10.0,
                diameter_mm=100.0,
                cooling_coeff_W_m2K=20.0
            ),
            "focusing_lens": OpticalThermalLoad(
                material=fused_silica,
                thickness_mm=15.0,
                diameter_mm=80.0,
                cooling_coeff_W_m2K=15.0
            )
        }
    
    def _init_tracking_subsystems(self):
        """Initialize tracking subsystems."""
        # These are handled by the core simulator
        pass
    
    def _init_performance_monitoring(self):
        """Initialize performance monitoring."""
        self.performance_monitor = PerformanceMonitor({
            'beam_efficiency': 0.85,
            'thermal_margin': 10.0,
            'strehl_ratio': 0.8,
            'pointing_accuracy_mrad': 0.1
        })
    
    def setup_scenario(self, scenario_config: Dict[str, Any]):
        """
        Set up a simulation scenario.
        
        Args:
            scenario_config: Scenario configuration dictionary
        """
        # Set up core simulator scenario
        self.core_sim.setup_scenario(scenario_config)
        
        # Reset simulation state
        self.sim_time = 0.0
        self.sim_step = 0
        self.results_history = []
        
        # Reset metrics
        self.metrics.reset()
        self.metrics.start_benchmark()
        
        # Log scenario setup
        print(f"Setting up scenario: {scenario_config.get('name', 'Unnamed')}")
    
    def step(self, dt: float = 0.01):
        """
        Advance simulation by one time step.
        
        Args:
            dt: Time step in seconds
        """
        self.sim_time += dt
        self.sim_step += 1
        
        # Step 1: Core system simulation step
        self.core_sim.step(dt)
        
        # Step 2: Optical subsystem simulation
        self._simulate_optical_subsystems(dt)
        
        # Step 3: Thermal subsystem simulation
        self._simulate_thermal_subsystems(dt)
        
        # Step 4: Calculate and record performance metrics
        self._calculate_performance_metrics()
        
        # Step 5: Check for performance degradation
        self._check_performance_degradation()
        
        # Step 6: Record results for this step
        self._record_step_results()
    
    def _simulate_optical_subsystems(self, dt: float):
        """Simulate optical subsystems for one time step."""
        # Get current system state from core simulator
        system_state = self._get_current_system_state()
        
        # Simulate wavefront sensing and correction
        if self.sim_step % 10 == 0:  # Every 10 steps
            # Measure wavefront
            wavefront = self.beam_alignment._measure_wavefront()
            if wavefront is not None:
                # Calculate wavefront error (assuming flat reference)
                wavefront_error = wavefront - np.zeros_like(wavefront)
                
                # Apply correction
                if self.deformable_mirror is not None:
                    self.beam_alignment._apply_wavefront_correction(wavefront_error)
                
                # Calculate Strehl ratio
                strehl_ratio = self.beam_alignment._calculate_strehl_ratio(wavefront_error)
                
                # Record metrics
                self.metrics.record('strehl_ratio', strehl_ratio)
                self.metrics.record('rms_wavefront_error', float(np.std(wavefront_error)))
        
        # Simulate beam alignment verification
        if self.sim_step % 20 == 0:  # Every 20 steps
            # Simulate angle measurements with some noise
            angle_x = system_state.get('pointing_angle_x', 0.0) + np.random.normal(0, 0.05)
            angle_y = system_state.get('pointing_angle_y', 0.0) + np.random.normal(0, 0.05)
            
            # Verify alignment
            verification_result = self.beam_verification.verify_alignment({
                'x': angle_x,
                'y': angle_y
            })
            
            # Record metrics
            self.metrics.record('alignment_score', verification_result['alignment_score'])
            self.metrics.record('rms_angle_deviation_mrad', verification_result['rms_deviation_mrad'])
    
    def _simulate_thermal_subsystems(self, dt: float):
        """Simulate thermal subsystems for one time step."""
        # Get current system state from core simulator
        system_state = self._get_current_system_state()
        
        # Get laser parameters
        power_W = system_state.get('laser_power_W', 1000.0)
        wavelength_nm = system_state.get('wavelength_nm', 1064.0)
        beam_radius_mm = system_state.get('beam_radius_mm', 10.0)
        
        # Only run thermal simulation periodically to save computation
        if self.sim_step % 50 == 0:  # Every 50 steps
            # Simulate thermal effects on key components
            thermal_results = {}
            for component_name, simulator in self.thermal_simulators.items():
                result = simulator.simulate(power_W, wavelength_nm, beam_radius_mm)
                thermal_results[component_name] = result
                
                # Record key metrics
                self.metrics.record(f'max_temp_{component_name}_C', result['max_temperature_C'])
                
                # Calculate thermal lensing impact
                if 'thermal_lensing' in result:
                    focal_length_m = result['thermal_lensing']['focal_length_m']
                    self.metrics.record(f'thermal_focal_length_{component_name}_m', focal_length_m)
                    
                    # Calculate thermal margin (difference from critical temperature)
                    critical_temp = self.config.get('critical_temperatures', {}).get(component_name, 200.0)
                    thermal_margin = critical_temp - result['max_temperature_C']
                    self.metrics.record(f'thermal_margin_{component_name}_C', thermal_margin)
    
    def _calculate_performance_metrics(self):
        """Calculate and record overall system performance metrics."""
        # Get current system state
        system_state = self._get_current_system_state()
        
        # Calculate beam efficiency
        strehl_ratio = self.metrics.get_metric('strehl_ratio')
        if strehl_ratio:
            latest_strehl = strehl_ratio[-1]
            
            # Simplified beam efficiency calculation
            beam_efficiency = latest_strehl * system_state.get('transmission_efficiency', 0.95)
            self.metrics.record('beam_efficiency', beam_efficiency)
        
        # Calculate system response time
        if 'command_time' in system_state and 'response_time' in system_state:
            response_time_ms = (system_state['response_time'] - system_state['command_time']) * 1000
            self.metrics.record('system_response_time_ms', response_time_ms)
        
        # Calculate engagement effectiveness
        if 'target_tracks' in system_state:
            # Simplified engagement effectiveness metric
            track_quality = sum(t.get('quality', 0) for t in system_state['target_tracks'])
            if system_state['target_tracks']:
                avg_track_quality = track_quality / len(system_state['target_tracks'])
                self.metrics.record('tracking_quality', avg_track_quality)
    
    def _check_performance_degradation(self):
        """Check for performance degradation and record alerts."""
        # Record current metrics in performance monitor
        for metric_name in ['beam_efficiency', 'thermal_margin', 'strehl_ratio']:
            metrics = self.metrics.get_metric(metric_name)
            if metrics:
                self.performance_monitor.record(metric_name, metrics[-1])
        
        # Get any new alerts
        alerts = self.performance_monitor.get_alerts()
        if alerts:
            for alert in alerts:
                self.metrics.record('alert', alert)
                print(f"ALERT: {alert['metric']} = {alert['value']:.3f}, threshold = {alert['threshold']:.3f}")
    
    def _record_step_results(self):
        """Record results for the current simulation step."""
        # Create results snapshot
        results = {
            'sim_time': self.sim_time,
            'sim_step': self.sim_step,
            'metrics': {k: v[-1] if v else None for k, v in self.metrics.metrics.items()}
        }
        
        # Add to history
        self.results_history.append(results)
    
    def _get_current_system_state(self) -> Dict[str, Any]:
        """Get current system state from core simulator."""
        # In a real implementation, this would extract state from the core simulator
        # For now, we'll create a synthetic state
        return {
            'laser_power_W': 5000.0 * (1.0 + 0.1 * np.sin(self.sim_time / 10.0)),
            'wavelength_nm': 1064.0,
            'beam_radius_mm': 10.0 + 0.5 * np.sin(self.sim_time / 5.0),
            'pointing_angle_x': 0.01 * np.sin(self.sim_time / 3.0),
            'pointing_angle_y': 0.01 * np.cos(self.sim_time / 3.0),
            'transmission_efficiency': 0.95 - 0.03 * (1.0 - np.exp(-self.sim_time / 100.0)),
            'target_tracks': [
                {'id': 1, 'quality': 0.9 - 0.1 * np.sin(self.sim_time / 20.0)},
                {'id': 2, 'quality': 0.85 + 0.1 * np.sin(self.sim_time / 15.0)}
            ],
            'command_time': self.sim_time - 0.005,
            'response_time': self.sim_time
        }
    
    def run(self, duration: float, dt: float = 0.01):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            dt: Time step in seconds
        """
        print(f"Running simulation for {duration:.1f} seconds with dt={dt:.4f}s")
        
        steps = int(duration / dt)
        for _ in range(steps):
            self.step(dt)
            
            # Print progress every 10%
            if self.sim_step % (steps // 10) == 0:
                progress = 100 * self.sim_time / duration
                print(f"Simulation progress: {progress:.1f}%")
        
        # Finalize metrics
        self.metrics.end_benchmark()
        
        print("Simulation complete")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of simulation results.
        
        Returns:
            Dictionary with simulation results summary
        """
        # Get metrics summary
        metrics_summary = self.metrics.summary()
        
        # Calculate additional summary statistics
        beam_efficiency = self.metrics.get_metric('beam_efficiency')
        strehl_ratio = self.metrics.get_metric('strehl_ratio')
        
        # Calculate stability metrics
        stability_metrics = {}
        for metric_name in ['beam_efficiency', 'strehl_ratio', 'rms_wavefront_error']:
            values = self.metrics.get_metric(metric_name)
            if values and len(values) > 10:
                stability_metrics[f'{metric_name}_stability'] = 1.0 - (np.std(values) / np.mean(values))
        
        # Count alerts by type
        alerts = self.metrics.get_metric('alert')
        alert_counts = {}
        if alerts:
            for alert in alerts:
                metric = alert['metric']
                if metric not in alert_counts:
                    alert_counts[metric] = 0
                alert_counts[metric] += 1
        
        # Combine all results
        return {
            'metrics': metrics_summary,
            'stability': stability_metrics,
            'alerts': alert_counts,
            'simulation_parameters': {
                'duration': self.sim_time,
                'steps': self.sim_step,
                'config_path': self.config_path
            }
        }
    
    def plot_results(self, output_file: Optional[str] = None):
        """
        Generate visualization of simulation results.
        
        Args:
            output_file: Optional file path to save the plot
        """
        if not self.results_history:
            print("No simulation results to plot")
            return
        
        # Extract time series data
        times = [r['sim_time'] for r in self.results_history]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('ODIN System Performance Simulation Results', fontsize=16)
        
        # Plot 1: Beam quality metrics
        ax1 = axs[0, 0]
        metrics_to_plot = ['strehl_ratio', 'beam_efficiency']
        for metric in metrics_to_plot:
            values = [r['metrics'].get(metric) for r in self.results_history]
            # Filter out None values
            valid_times = [t for t, v in zip(times, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            if valid_values:
                ax1.plot(valid_times, valid_values, label=metric)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Quality Metric')
        ax1.set_title('Beam Quality Metrics')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Thermal metrics
        ax2 = axs[0, 1]
        components = ['output_window', 'focusing_lens']
        for component in components:
            values = [r['metrics'].get(f'max_temp_{component}_C') for r in self.results_history]
            valid_times = [t for t, v in zip(times, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            if valid_values:
                ax2.plot(valid_times, valid_values, label=f'{component} temp')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Component Temperatures')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Wavefront error
        ax3 = axs[1, 0]
        values = [r['metrics'].get('rms_wavefront_error') for r in self.results_history]
        valid_times = [t for t, v in zip(times, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        if valid_values:
            ax3.plot(valid_times, valid_values)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('RMS Error')
        ax3.set_title('Wavefront Error')
        ax3.grid(True)
        
        # Plot 4: Pointing accuracy
        ax4 = axs[1, 1]
        values = [r['metrics'].get('rms_angle_deviation_mrad') for r in self.results_history]
        valid_times = [t for t, v in zip(times, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        if valid_values:
            ax4.plot(valid_times, valid_values)
            # Add threshold line
            ax4.axhline(0.2, color='r', linestyle='--', label='Threshold')
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('RMS Deviation (mrad)')
        ax4.set_title('Pointing Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        # Plot 5: System response time
        ax5 = axs[2, 0]
        values = [r['metrics'].get('system_response_time_ms') for r in self.results_history]
        valid_times = [t for t, v in zip(times, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        if valid_values:
            ax5.plot(valid_times, valid_values)
        
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Response Time (ms)')
        ax5.set_title('System Response Time')
        ax5.grid(True)
        
        # Plot 6: Tracking quality
        ax6 = axs[2, 1]
        values = [r['metrics'].get('tracking_quality') for r in self.results_history]
        valid_times = [t for t, v in zip(times, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        if valid_values:
            ax6.plot(valid_times, valid_values)
        
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Quality')
        ax6.set_title('Target Tracking Quality')
        ax6.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if output_file:
            plt.savefig(output_file)
            print(f"Results plot saved to {output_file}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = HighFidelitySystemSimulator(
        config_path="config/system_config.json",
        output_dir="simulation_results"
    )
    
    # Set up scenario
    scenario = {
        "name": "High-power engagement",
        "duration": 10.0,
        "targets": [
            {"id": 1, "range_km": 5.0, "velocity_m_s": 10.0, "priority": 0.9},
            {"id": 2, "range_km": 8.0, "velocity_m_s": 15.0, "priority": 0.7}
        ],
        "environment": {
            "turbulence_cn2": 1e-14,
            "visibility_km": 20.0,
            "wind_speed_m_s": 5.0
        },
        "system_config": {
            "laser_power_W": 5000.0,
            "beam_quality": 1.2,  # M²
            "wavelength_nm": 1064.0
        }
    }
    
    simulator.setup_scenario(scenario)
    
    # Run simulation
    simulator.run(duration=10.0, dt=0.01)
    
    # Get and print results summary
    results = simulator.get_results_summary()
    print("\nSimulation Results Summary:")
    print(f"Wall time: {results['metrics'].get('wall_time_s', 0):.2f} seconds")
    print(f"Mean beam efficiency: {results['metrics'].get('beam_efficiency', {}).get('mean', 0):.4f}")
    print(f"Mean Strehl ratio: {results['metrics'].get('strehl_ratio', {}).get('mean', 0):.4f}")
    
    # Plot results
    simulator.plot_results("simulation_results/performance_plot.png")


def validate_system_accuracy(self) -> Dict[str, Any]:
    """
    Validate system accuracy using precision metrics.
    
    Returns:
        Dictionary with validation results
    """
    # Initialize precision metrics
    precision_metrics = SystemPrecisionMetrics(self.config.get("precision_thresholds"))
    
    # Record current metrics for validation
    if self.metrics.get_metric('strehl_ratio'):
        precision_metrics.record("strehl_ratio", self.metrics.get_metric('strehl_ratio')[-1], "ratio")
        
    if self.metrics.get_metric('rms_wavefront_error'):
        # Convert to nanometers assuming the original is in waves
        wavefront_error_nm = self.metrics.get_metric('rms_wavefront_error')[-1] * 1064  # Assuming 1064nm wavelength
        precision_metrics.record("wavefront_error_nm_rms", wavefront_error_nm, "nm")
        
    if self.metrics.get_metric('rms_angle_deviation_mrad'):
        # Convert mrad to μrad
        pointing_accuracy = self.metrics.get_metric('rms_angle_deviation_mrad')[-1] * 1000
        precision_metrics.record("pointing_accuracy_urad", pointing_accuracy, "μrad")
        
    if self.metrics.get_metric('tracking_quality'):
        # Normalize tracking quality to precision metric
        tracking_quality = self.metrics.get_metric('tracking_quality')[-1]
        tracking_precision = (1 - tracking_quality) * 100  # Convert to μrad error (example conversion)
        precision_metrics.record("tracking_precision_urad", tracking_precision, "μrad")
        
    # Calculate thermal stability from component temperatures
    if self.metrics.get_metric('max_temp_output_window_C') and self.metrics.get_metric('max_temp_focusing_lens_C'):
        temps = []
        if len(self.metrics.get_metric('max_temp_output_window_C')) > 10:
            # Calculate temperature stability over last 10 samples
            window_temps = self.metrics.get_metric('max_temp_output_window_C')[-10:]
            lens_temps = self.metrics.get_metric('max_temp_focusing_lens_C')[-10:]
            window_stability = np.std(window_temps)
            lens_stability = np.std(lens_temps)
            thermal_stability = max(window_stability, lens_stability)
            precision_metrics.record("thermal_stability_C", thermal_stability, "°C")
    
    # System response time
    if self.metrics.get_metric('system_response_time_ms'):
        precision_metrics.record("response_time_ms", self.metrics.get_metric('system_response_time_ms')[-1], "ms")
    
    # Calculate overall system accuracy
    system_accuracy = precision_metrics.calculate_system_accuracy()
    precision_metrics.record("system_accuracy", system_accuracy, "ratio")
    
    # Generate validation summary
    validation_results = precision_metrics.validation_summary()
    
    # Log validation results
    print(f"System Accuracy Validation: {system_accuracy:.4f}")
    print(f"Overall Validation Rate: {validation_results['overall_validation_rate']:.2f}")
    
    return validation_results