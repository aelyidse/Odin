import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import minimize

class VibrationIsolationSystem:
    """
    Models and compensates for environmental vibrations to maintain beam stability.
    
    Features:
    - Real-time vibration measurement and analysis
    - Multi-stage passive and active isolation modeling
    - Adaptive compensation for varying vibration profiles
    - Performance metrics and stability analysis
    """
    
    def __init__(self, 
                 natural_frequencies_hz: List[float] = [5.0, 20.0],
                 damping_ratios: List[float] = [0.1, 0.3],
                 sampling_rate_hz: float = 1000.0,
                 max_correction_amplitude: float = 1.0,
                 compensation_bandwidth_hz: float = 100.0):
        """
        Initialize vibration isolation system.
        
        Args:
            natural_frequencies_hz: Natural frequencies of isolation stages (Hz)
            damping_ratios: Damping ratios of isolation stages
            sampling_rate_hz: Vibration measurement sampling rate (Hz)
            max_correction_amplitude: Maximum correction amplitude
            compensation_bandwidth_hz: Active compensation bandwidth (Hz)
        """
        self.natural_frequencies_hz = natural_frequencies_hz
        self.damping_ratios = damping_ratios
        self.sampling_rate_hz = sampling_rate_hz
        self.max_correction_amplitude = max_correction_amplitude
        self.compensation_bandwidth_hz = compensation_bandwidth_hz
        
        # Derived parameters
        self.n_stages = len(natural_frequencies_hz)
        self.dt = 1.0 / sampling_rate_hz
        
        # System state
        self.input_vibrations = []
        self.output_vibrations = []
        self.compensation_signals = []
        self.is_active = True
        
        # Performance metrics
        self.transmissibility = np.ones(self.n_stages)
        self.isolation_efficiency = 0.0
        self.rms_vibration_input = 0.0
        self.rms_vibration_output = 0.0
        
        # Initialize transfer functions
        self._initialize_transfer_functions()
    
    def _initialize_transfer_functions(self):
        """Initialize transfer functions for each isolation stage."""
        self.transfer_functions = []
        
        for i in range(self.n_stages):
            wn = 2 * np.pi * self.natural_frequencies_hz[i]
            zeta = self.damping_ratios[i]
            
            # Create transfer function for this stage
            def transfer_function(f, wn=wn, zeta=zeta):
                w = 2 * np.pi * f
                s = 1j * w
                # Second-order system transfer function
                tf = 1 / (1 + 2*zeta*(s/wn) + (s/wn)**2)
                return tf
            
            self.transfer_functions.append(transfer_function)
    
    def calculate_transmissibility(self, frequency_hz: float) -> float:
        """
        Calculate system transmissibility at a specific frequency.
        
        Args:
            frequency_hz: Frequency to calculate transmissibility (Hz)
            
        Returns:
            Transmissibility ratio (output/input)
        """
        transmissibility = 1.0
        
        # Calculate through all stages
        for tf in self.transfer_functions:
            transmissibility *= np.abs(tf(frequency_hz))
        
        return float(transmissibility)
    
    def get_transmissibility_curve(self, 
                                  freq_min_hz: float = 0.1, 
                                  freq_max_hz: float = 1000.0,
                                  n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate transmissibility curve over frequency range.
        
        Args:
            freq_min_hz: Minimum frequency (Hz)
            freq_max_hz: Maximum frequency (Hz)
            n_points: Number of points in curve
            
        Returns:
            Dictionary with frequency and transmissibility arrays
        """
        frequencies = np.logspace(np.log10(freq_min_hz), np.log10(freq_max_hz), n_points)
        transmissibility = np.zeros_like(frequencies)
        
        for i, f in enumerate(frequencies):
            transmissibility[i] = self.calculate_transmissibility(f)
        
        return {
            'frequencies_hz': frequencies,
            'transmissibility': transmissibility
        }
    
    def update(self, 
              input_vibration: np.ndarray,
              timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Process new vibration measurement and calculate compensation.
        
        Args:
            input_vibration: Measured vibration signal
            timestamp: Optional measurement timestamp
            
        Returns:
            Dictionary with processed vibration and compensation data
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store input vibration
        self.input_vibrations.append(input_vibration.copy())
        if len(self.input_vibrations) > 1000:
            self.input_vibrations.pop(0)
        
        # Calculate passive isolation effect
        output_vibration = self._apply_passive_isolation(input_vibration)
        
        # Apply active compensation if enabled
        compensation_signal = np.zeros_like(input_vibration)
        if self.is_active:
            compensation_signal = self._calculate_compensation(input_vibration, output_vibration)
            output_vibration = output_vibration - compensation_signal
        
        # Store results
        self.output_vibrations.append(output_vibration.copy())
        self.compensation_signals.append(compensation_signal.copy())
        
        if len(self.output_vibrations) > 1000:
            self.output_vibrations.pop(0)
        if len(self.compensation_signals) > 1000:
            self.compensation_signals.pop(0)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return {
            'timestamp': timestamp,
            'input_vibration': input_vibration,
            'output_vibration': output_vibration,
            'compensation_signal': compensation_signal,
            'isolation_efficiency': self.isolation_efficiency,
            'transmissibility': self.transmissibility.tolist()
        }
    
    def _apply_passive_isolation(self, input_vibration: np.ndarray) -> np.ndarray:
        """
        Apply passive isolation transfer functions to input vibration.
        
        This is a simplified model - a real implementation would use proper
        digital filtering based on the transfer functions.
        
        Args:
            input_vibration: Input vibration signal
            
        Returns:
            Vibration after passive isolation
        """
        # Simple implementation - apply frequency-domain filtering
        # In a real system, this would be more sophisticated
        
        # For simplicity, we'll just scale the input by the DC transmissibility
        # of each stage (this is a very simplified model)
        output_vibration = input_vibration.copy()
        
        for i in range(self.n_stages):
            # Low-frequency attenuation factor (simplified)
            attenuation = 1.0 / (1.0 + (0.5 / self.natural_frequencies_hz[i])**2)
            output_vibration *= attenuation
        
        return output_vibration
    
    def _calculate_compensation(self, 
                              input_vibration: np.ndarray,
                              output_vibration: np.ndarray) -> np.ndarray:
        """
        Calculate active compensation signal to counteract residual vibrations.
        
        Args:
            input_vibration: Input vibration signal
            output_vibration: Vibration after passive isolation
            
        Returns:
            Compensation signal
        """
        # Simple feedforward compensation (more sophisticated algorithms would be used in practice)
        # Estimate the frequency content that needs to be compensated
        
        # For this simplified model, we'll just generate a signal proportional to the output
        # but phase-shifted to provide cancellation
        compensation = -0.8 * output_vibration
        
        # Apply bandwidth limitation
        # In practice, this would be a proper filter
        compensation = np.clip(compensation, -self.max_correction_amplitude, self.max_correction_amplitude)
        
        return compensation
    
    def _update_performance_metrics(self):
        """Update system performance metrics based on recent data."""
        if not self.input_vibrations or not self.output_vibrations:
            return
        
        # Calculate RMS values
        recent_inputs = np.array(self.input_vibrations[-100:])
        recent_outputs = np.array(self.output_vibrations[-100:])
        
        self.rms_vibration_input = np.sqrt(np.mean(np.square(recent_inputs)))
        self.rms_vibration_output = np.sqrt(np.mean(np.square(recent_outputs)))
        
        # Calculate isolation efficiency
        if self.rms_vibration_input > 0:
            self.isolation_efficiency = 1.0 - (self.rms_vibration_output / self.rms_vibration_input)
        else:
            self.isolation_efficiency = 0.0
    
    def set_active_compensation(self, enabled: bool):
        """Enable or disable active compensation."""
        self.is_active = enabled
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'isolation_efficiency': float(self.isolation_efficiency),
            'rms_input': float(self.rms_vibration_input),
            'rms_output': float(self.rms_vibration_output),
            'active_compensation': self.is_active,
            'n_stages': self.n_stages,
            'natural_frequencies_hz': self.natural_frequencies_hz,
            'damping_ratios': self.damping_ratios
        }
    
    def optimize_parameters(self, 
                           vibration_profile: np.ndarray,
                           frequency_profile: np.ndarray) -> Dict[str, Any]:
        """
        Optimize isolation parameters for a given vibration profile.
        
        Args:
            vibration_profile: Power spectral density of vibration
            frequency_profile: Frequency points for vibration profile
            
        Returns:
            Optimized parameters
        """
        # Define objective function for optimization
        def objective(params):
            # Extract parameters
            n_stages = len(params) // 2
            natural_frequencies = params[:n_stages]
            damping_ratios = params[n_stages:]
            
            # Calculate transmissibility at each frequency
            transmissibility = np.ones_like(frequency_profile)
            for i in range(n_stages):
                wn = 2 * np.pi * natural_frequencies[i]
                zeta = damping_ratios[i]
                
                for j, f in enumerate(frequency_profile):
                    w = 2 * np.pi * f
                    s = 1j * w
                    tf = 1 / (1 + 2*zeta*(s/wn) + (s/wn)**2)
                    transmissibility[j] *= abs(tf)
            
            # Calculate weighted transmissibility (weighted by vibration profile)
            weighted_transmissibility = transmissibility * vibration_profile
            
            # Return mean weighted transmissibility (to minimize)
            return np.mean(weighted_transmissibility)
        
        # Initial parameters
        initial_params = np.concatenate((self.natural_frequencies_hz, self.damping_ratios))
        
        # Parameter bounds
        n_stages = self.n_stages
        bounds = [(1.0, 50.0)] * n_stages + [(0.01, 1.0)] * n_stages
        
        # Run optimization
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        optimized_natural_frequencies = optimized_params[:n_stages].tolist()
        optimized_damping_ratios = optimized_params[n_stages:].tolist()
        
        # Update system parameters
        self.natural_frequencies_hz = optimized_natural_frequencies
        self.damping_ratios = optimized_damping_ratios
        
        # Reinitialize transfer functions
        self._initialize_transfer_functions()
        
        return {
            'optimized_natural_frequencies_hz': optimized_natural_frequencies,
            'optimized_damping_ratios': optimized_damping_ratios,
            'objective_value': float(result.fun),
            'convergence_success': result.success
        }