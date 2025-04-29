import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Callable

from odin_sdk.physics_engine.shack_hartmann import ShackHartmannSensor
from odin_sdk.physics_engine.adaptive_optics_control import DeformableMirror, LCSLM
from odin_sdk.manufacturing.optical_alignment import OpticalAlignmentProcedure, AlignmentStep
from odin_sdk.deployment.field_calibration import CalibrationStep, FieldCalibrationProcedure
from odin_sdk.deployment.performance_monitor import PerformanceMonitor


class BeamAlignmentCalibration:
    """
    Automated calibration procedure for beam alignment.
    
    This class provides a structured approach to calibrate and align optical beams
    with high precision, using feedback from sensors and adaptive optics.
    """
    
    def __init__(self, 
                 wavefront_sensor: Optional[ShackHartmannSensor] = None,
                 deformable_mirror: Optional[DeformableMirror] = None,
                 target_strehl_ratio: float = 0.85,
                 max_iterations: int = 20,
                 convergence_threshold: float = 0.005,
                 log_results: bool = True):
        """
        Initialize beam alignment calibration system.
        
        Args:
            wavefront_sensor: ShackHartmannSensor instance for wavefront measurement
            deformable_mirror: DeformableMirror instance for wavefront correction
            target_strehl_ratio: Target Strehl ratio for successful calibration
            max_iterations: Maximum number of calibration iterations
            convergence_threshold: Threshold for convergence determination
            log_results: Whether to log calibration results
        """
        self.wavefront_sensor = wavefront_sensor
        self.deformable_mirror = deformable_mirror
        self.target_strehl_ratio = target_strehl_ratio
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.log_results = log_results
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_history = []
        self.current_strehl_ratio = 0.0
        self.current_wavefront_error = None
        self.current_beam_position = np.array([0.0, 0.0])  # x, y in mm
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor({
            'strehl_ratio': self.target_strehl_ratio,
            'beam_position_error': 0.1  # mm
        })
        
        # Create calibration procedure
        self.calibration_procedure = self._create_calibration_procedure()
    
    def _create_calibration_procedure(self) -> FieldCalibrationProcedure:
        """Create the structured calibration procedure."""
        procedure = FieldCalibrationProcedure('beam alignment')
        
        # Step 1: Initial system check
        procedure.add_step(CalibrationStep(
            description='System initialization and sensor check',
            method='self-diagnostic',
            acceptance_criteria='All sensors and actuators operational'
        ))
        
        # Step 2: Coarse alignment
        procedure.add_step(CalibrationStep(
            description='Coarse beam alignment to target',
            method='beam profiler, reference target',
            acceptance_criteria='Beam centroid within 1 mm of reference mark'
        ))
        
        # Step 3: Wavefront measurement
        procedure.add_step(CalibrationStep(
            description='Wavefront measurement and analysis',
            method='Shack-Hartmann sensor',
            acceptance_criteria='Valid wavefront data acquired'
        ))
        
        # Step 4: Adaptive correction
        procedure.add_step(CalibrationStep(
            description='Adaptive optics correction',
            method='deformable mirror, closed-loop control',
            acceptance_criteria=f'Strehl ratio > {self.target_strehl_ratio}'
        ))
        
        # Step 5: Fine alignment
        procedure.add_step(CalibrationStep(
            description='Fine beam position adjustment',
            method='piezo actuators, beam profiler',
            acceptance_criteria='Beam position error < 0.1 mm'
        ))
        
        # Step 6: Verification
        procedure.add_step(CalibrationStep(
            description='Stability verification',
            method='time series measurement',
            acceptance_criteria='Stable performance over 60 seconds'
        ))
        
        return procedure
    
    def run_calibration(self, 
                       reference_wavefront: Optional[np.ndarray] = None,
                       target_position: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run the complete beam alignment calibration procedure.
        
        Args:
            reference_wavefront: Optional reference wavefront (ideal flat)
            target_position: Optional target beam position [x, y] in mm
            
        Returns:
            Dictionary with calibration results
        """
        if self.wavefront_sensor is None:
            raise ValueError("Wavefront sensor is required for calibration")
        
        # Initialize results
        results = {
            'success': False,
            'iterations': 0,
            'final_strehl_ratio': 0.0,
            'beam_position_error': 0.0,
            'convergence_history': [],
            'timestamp': time.time()
        }
        
        # Step 1: System initialization
        print("Step 1: System initialization and sensor check")
        if not self._initialize_system():
            results['failure_reason'] = "System initialization failed"
            return results
        
        # Step 2: Coarse alignment
        print("Step 2: Coarse beam alignment")
        if target_position is not None:
            self._coarse_alignment(target_position)
        
        # Create default reference wavefront if not provided
        if reference_wavefront is None:
            # Create flat wavefront as reference
            reference_wavefront = np.zeros((32, 32))
        
        # Step 3-4: Iterative wavefront correction
        print("Step 3-4: Wavefront measurement and correction")
        for iteration in range(self.max_iterations):
            # Measure current wavefront
            measured_wavefront = self._measure_wavefront()
            if measured_wavefront is None:
                results['failure_reason'] = "Failed to measure wavefront"
                return results
            
            # Calculate wavefront error
            wavefront_error = measured_wavefront - reference_wavefront
            self.current_wavefront_error = wavefront_error
            
            # Calculate Strehl ratio
            strehl_ratio = self._calculate_strehl_ratio(wavefront_error)
            self.current_strehl_ratio = strehl_ratio
            
            # Log progress
            iteration_result = {
                'iteration': iteration,
                'strehl_ratio': strehl_ratio,
                'rms_wavefront_error': float(np.std(wavefront_error)),
                'max_wavefront_error': float(np.max(np.abs(wavefront_error)))
            }
            results['convergence_history'].append(iteration_result)
            
            if self.log_results:
                print(f"  Iteration {iteration}: Strehl ratio = {strehl_ratio:.4f}")
            
            # Check if we've reached target performance
            if strehl_ratio >= self.target_strehl_ratio:
                print(f"  Target Strehl ratio achieved: {strehl_ratio:.4f}")
                break
            
            # Check for convergence
            if iteration > 0:
                prev_strehl = results['convergence_history'][iteration-1]['strehl_ratio']
                if abs(strehl_ratio - prev_strehl) < self.convergence_threshold:
                    print(f"  Convergence detected at iteration {iteration}")
                    break
            
            # Apply correction if we have a deformable mirror
            if self.deformable_mirror is not None:
                self._apply_wavefront_correction(wavefront_error)
        
        # Step 5: Fine alignment
        print("Step 5: Fine beam position adjustment")
        if target_position is not None:
            position_error = self._fine_alignment(target_position)
            results['beam_position_error'] = float(position_error)
        
        # Step 6: Stability verification
        print("Step 6: Stability verification")
        stability_result = self._verify_stability(60)  # 60 seconds
        results['stability'] = stability_result
        
        # Update final results
        results['iterations'] = len(results['convergence_history'])
        results['final_strehl_ratio'] = self.current_strehl_ratio
        results['success'] = (
            self.current_strehl_ratio >= self.target_strehl_ratio and
            stability_result['is_stable']
        )
        
        # Update calibration state
        self.is_calibrated = results['success']
        self.calibration_history.append(results)
        
        return results
    
    def _initialize_system(self) -> bool:
        """Initialize and check all system components."""
        try:
            # Check wavefront sensor
            if self.wavefront_sensor is None:
                print("  Warning: No wavefront sensor configured")
                return False
            
            # Check deformable mirror if available
            if self.deformable_mirror is not None:
                # Reset DM to flat state
                self.deformable_mirror.actuator_commands = np.zeros(
                    (self.deformable_mirror.ny, self.deformable_mirror.nx)
                )
            
            return True
        except Exception as e:
            print(f"  System initialization failed: {str(e)}")
            return False
    
    def _measure_wavefront(self) -> Optional[np.ndarray]:
        """Measure the current wavefront using the wavefront sensor."""
        try:
            # In a real system, this would interface with actual hardware
            # For simulation, we'll create a synthetic wavefront with some aberrations
            
            # Create a synthetic phase map for testing
            # In a real system, this would come from the wavefront sensor
            aperture_mm = 10.0  # 10mm aperture
            phase_map = np.zeros((32, 32))
            
            # Add some synthetic aberrations for testing
            y, x = np.indices(phase_map.shape)
            y = y - phase_map.shape[0] // 2
            x = x - phase_map.shape[1] // 2
            r = np.sqrt(x**2 + y**2) / (phase_map.shape[0] // 2)
            
            # Add defocus and astigmatism
            defocus = 0.5 * (2 * r**2 - 1)
            astigmatism = 0.3 * (r**2) * np.cos(2 * np.arctan2(y, x))
            
            phase_map = defocus + astigmatism
            
            # If we have previous corrections, apply them to simulate feedback
            if hasattr(self, 'previous_correction') and self.previous_correction is not None:
                phase_map -= self.previous_correction * 0.8  # 80% effectiveness
            
            return phase_map
            
        except Exception as e:
            print(f"  Wavefront measurement failed: {str(e)}")
            return None
    
    def _calculate_strehl_ratio(self, wavefront_error: np.ndarray) -> float:
        """Calculate Strehl ratio from wavefront error."""
        # Convert wavefront error to radians if it's not already
        # Assuming wavefront_error is in waves or radians
        
        # Strehl ratio approximation: exp(-σ²)
        # where σ² is the variance of the wavefront error in radians²
        variance = np.var(wavefront_error)
        strehl_ratio = np.exp(-variance)
        
        return float(strehl_ratio)
    
    def _apply_wavefront_correction(self, wavefront_error: np.ndarray) -> bool:
        """Apply correction to the deformable mirror."""
        try:
            if self.deformable_mirror is None:
                return False
            
            # Compute actuator commands to correct the wavefront
            commands = self.deformable_mirror.compute_commands(wavefront_error)
            
            # Store correction for simulation purposes
            self.previous_correction = wavefront_error
            
            return True
        except Exception as e:
            print(f"  Wavefront correction failed: {str(e)}")
            return False
    
    def _coarse_alignment(self, target_position: np.ndarray) -> float:
        """Perform coarse alignment to get beam near target position."""
        # Simulate beam movement to target position with some error
        current_position = np.random.normal(target_position, 0.5)  # 0.5mm std error
        self.current_beam_position = current_position
        
        # Calculate position error
        position_error = np.linalg.norm(current_position - target_position)
        
        print(f"  Coarse alignment complete. Position error: {position_error:.3f} mm")
        return position_error
    
    def _fine_alignment(self, target_position: np.ndarray) -> float:
        """Perform fine alignment to precisely position the beam."""
        # Simulate fine adjustment with piezo actuators
        # In a real system, this would use feedback from a position sensor
        
        # Start from current position after coarse alignment
        current_position = self.current_beam_position
        
        # Simulate iterative adjustment with decreasing error
        for i in range(5):  # 5 iterations of fine adjustment
            # Calculate error vector
            error_vector = target_position - current_position
            
            # Apply correction with some noise
            correction = error_vector * 0.8  # 80% effectiveness
            noise = np.random.normal(0, 0.02, size=2)  # 20μm positioning noise
            
            # Update position
            current_position = current_position + correction + noise
        
        # Update current position
        self.current_beam_position = current_position
        
        # Calculate final position error
        position_error = np.linalg.norm(current_position - target_position)
        
        print(f"  Fine alignment complete. Position error: {position_error:.3f} mm")
        return position_error
    
    def _verify_stability(self, duration_seconds: float, sample_rate: float = 1.0) -> Dict[str, Any]:
        """Verify stability of the aligned beam over time."""
        # Number of samples to collect
        n_samples = int(duration_seconds * sample_rate)
        
        # Storage for measurements
        strehl_history = []
        position_history = []
        
        print(f"  Verifying stability over {duration_seconds} seconds...")
        
        # Simulate measurements over time
        for i in range(n_samples):
            # Simulate small random fluctuations in wavefront and position
            strehl_fluctuation = np.random.normal(0, 0.01)  # Small Strehl fluctuation
            position_fluctuation = np.random.normal(0, 0.01, size=2)  # Small position fluctuation
            
            # Calculate current values with fluctuation
            current_strehl = min(1.0, max(0.0, self.current_strehl_ratio + strehl_fluctuation))
            current_position = self.current_beam_position + position_fluctuation
            
            # Record measurements
            strehl_history.append(current_strehl)
            position_history.append(current_position)
            
            # In a real system, we would wait for the next sample
            # time.sleep(1.0 / sample_rate)
        
        # Calculate stability metrics
        strehl_mean = np.mean(strehl_history)
        strehl_std = np.std(strehl_history)
        
        position_history = np.array(position_history)
        position_std = np.std(position_history, axis=0)
        position_stability = np.mean(position_std)
        
        # Determine if stability criteria are met
        is_stable = (
            strehl_mean >= self.target_strehl_ratio and
            strehl_std < 0.05 and
            position_stability < 0.05
        )
        
        stability_result = {
            'is_stable': is_stable,
            'strehl_mean': float(strehl_mean),
            'strehl_std': float(strehl_std),
            'position_stability_mm': float(position_stability),
            'duration_seconds': duration_seconds,
            'n_samples': n_samples
        }
        
        print(f"  Stability verification {'passed' if is_stable else 'failed'}")
        print(f"  Strehl ratio: {strehl_mean:.4f} ± {strehl_std:.4f}")
        print(f"  Position stability: {position_stability:.4f} mm")
        
        return stability_result
    
    def generate_report(self) -> str:
        """Generate a detailed calibration report."""
        if not self.calibration_history:
            return "No calibration data available."
        
        # Get the most recent calibration
        cal = self.calibration_history[-1]
        
        report = []
        report.append("=== Beam Alignment Calibration Report ===")
        report.append(f"Timestamp: {time.ctime(cal['timestamp'])}")
        report.append(f"Calibration {'successful' if cal['success'] else 'failed'}")
        report.append("")
        
        report.append("Performance Metrics:")
        report.append(f"  Final Strehl Ratio: {cal['final_strehl_ratio']:.4f} (Target: {self.target_strehl_ratio:.4f})")
        if 'beam_position_error' in cal:
            report.append(f"  Beam Position Error: {cal['beam_position_error']:.4f} mm")
        report.append(f"  Iterations Required: {cal['iterations']}")
        report.append("")
        
        if 'stability' in cal:
            stability = cal['stability']
            report.append("Stability Verification:")
            report.append(f"  Duration: {stability['duration_seconds']} seconds")
            report.append(f"  Strehl Stability: {stability['strehl_mean']:.4f} ± {stability['strehl_std']:.4f}")
            report.append(f"  Position Stability: {stability['position_stability_mm']:.4f} mm")
            report.append(f"  Stability Test: {'Passed' if stability['is_stable'] else 'Failed'}")
            report.append("")
        
        if 'failure_reason' in cal:
            report.append(f"Failure Reason: {cal['failure_reason']}")
            report.append("")
        
        report.append("Calibration Procedure:")
        for i, step in enumerate(self.calibration_procedure.steps, 1):
            report.append(f"  Step {i}: {step.description}")
            report.append(f"    Method: {step.method}")
            report.append(f"    Criteria: {step.acceptance_criteria}")
        
        return "\n".join(report)
    
    def plot_calibration_results(self, output_file: Optional[str] = None):
        """Generate visualization of calibration results."""
        if not self.calibration_history:
            print("No calibration data available for plotting.")
            return
        
        # Get the most recent calibration
        cal = self.calibration_history[-1]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Beam Alignment Calibration Results', fontsize=16)
        
        # Plot 1: Convergence history
        if 'convergence_history' in cal and cal['convergence_history']:
            iterations = [d['iteration'] for d in cal['convergence_history']]
            strehl = [d['strehl_ratio'] for d in cal['convergence_history']]
            rms_error = [d['rms_wavefront_error'] for d in cal['convergence_history']]
            
            ax1 = axs[0, 0]
            ax1.plot(iterations, strehl, 'o-', label='Strehl Ratio')
            ax1.axhline(self.target_strehl_ratio, color='r', linestyle='--', 
                       label=f'Target ({self.target_strehl_ratio:.2f})')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Strehl Ratio')
            ax1.set_title('Convergence History')
            ax1.legend()
            ax1.grid(True)
            
            # Add RMS wavefront error on secondary y-axis
            ax1b = ax1.twinx()
            ax1b.plot(iterations, rms_error, 's-', color='green', label='RMS Error')
            ax1b.set_ylabel('RMS Wavefront Error (rad)', color='green')
            ax1b.tick_params(axis='y', labelcolor='green')
        
        # Plot 2: Wavefront error map
        if self.current_wavefront_error is not None:
            ax2 = axs[0, 1]
            im = ax2.imshow(self.current_wavefront_error, cmap='viridis')
            ax2.set_title('Wavefront Error Map')
            plt.colorbar(im, ax=ax2, label='Phase (rad)')
        
        # Plot 3: Stability over time
        if 'stability' in cal and hasattr(self, '_stability_data'):
            # This would be populated in a real implementation
            # For now, we'll create synthetic data
            time_points = np.linspace(0, cal['stability']['duration_seconds'], 
                                     cal['stability']['n_samples'])
            
            # Create synthetic stability data
            np.random.seed(42)  # For reproducibility
            strehl_data = cal['stability']['strehl_mean'] + \
                         np.random.normal(0, cal['stability']['strehl_std'], len(time_points))
            
            ax3 = axs[1, 0]
            ax3.plot(time_points, strehl_data)
            ax3.axhline(self.target_strehl_ratio, color='r', linestyle='--', 
                       label=f'Target ({self.target_strehl_ratio:.2f})')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Strehl Ratio')
            ax3.set_title('Stability Over Time')
            ax3.legend()
            ax3.grid(True)
        
        # Plot 4: Beam position
        if hasattr(self, 'current_beam_position'):
            ax4 = axs[1, 1]
            
            # Create a simple beam profile visualization
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            
            # Gaussian beam centered at current position
            beam_x, beam_y = self.current_beam_position
            beam = np.exp(-((X-beam_x)**2 + (Y-beam_y)**2) / 0.5**2)
            
            im = ax4.imshow(beam, extent=[-5, 5, -5, 5], origin='lower', cmap='hot')
            ax4.set_xlabel('X Position (mm)')
            ax4.set_ylabel('Y Position (mm)')
            ax4.set_title('Beam Position')
            ax4.grid(True)
            
            # Mark the center
            ax4.plot(beam_x, beam_y, 'b+', markersize=10)
            
            # If we have a target position, mark it
            if hasattr(self, 'target_position') and self.target_position is not None:
                target_x, target_y = self.target_position
                ax4.plot(target_x, target_y, 'gx', markersize=10, label='Target')
                ax4.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Create wavefront sensor
    sensor = ShackHartmannSensor(
        n_lenslets_x=10,
        n_lenslets_y=10,
        lenslet_pitch_mm=0.5,
        focal_length_mm=25.0,
        detector_pixel_size_um=5.0,
        detector_pixels_x=1024,
        detector_pixels_y=1024
    )
    
    # Create deformable mirror
    dm = DeformableMirror(
        n_actuators_x=12,
        n_actuators_y=12,
        max_stroke_um=5.0
    )
    
    # Create beam alignment calibration
    calibration = BeamAlignmentCalibration(
        wavefront_sensor=sensor,
        deformable_mirror=dm,
        target_strehl_ratio=0.85
    )
    
    # Run calibration
    target_position = np.array([0.0, 0.0])  # Center alignment
    results = calibration.run_calibration(target_position=target_position)
    
    # Generate and print report
    report = calibration.generate_report()
    print("\n" + report)
    
    # Plot results
    calibration.plot_calibration_results()