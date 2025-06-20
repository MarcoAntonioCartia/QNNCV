"""
Multimode Coherent State Constellation Circuit

This implementation creates genuine multimode diversity by starting each mode
in a different coherent state |α_i⟩, forming a "constellation" in phase space.

Key innovations:
- Each mode starts at a unique coherent state instead of vacuum
- True multimode utilization from the beginning
- Rich inter-mode quantum correlations
- Genuine quantum diversity instead of single-mode collapse

Based on the 16-mode encoding concept but optimized for QGANs.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MultimodalConstellationCircuit:
    """
    Quantum circuit with multimode coherent state constellation initialization.
    
    This circuit creates genuine multimode diversity by:
    1. Initializing each mode in a unique coherent state |α_i⟩
    2. Creating a "constellation" of starting points in phase space
    3. Enabling true multimode quantum correlations
    4. Preventing single-mode collapse through diverse initialization
    """
    
    def __init__(self, 
                 n_modes: int, 
                 n_layers: int, 
                 cutoff_dim: int = 6,
                 constellation_radius: float = 1.5,
                 enable_correlations: bool = True):
        """
        Initialize multimode constellation circuit.
        
        Args:
            n_modes: Number of quantum modes
            n_layers: Number of variational layers
            cutoff_dim: Fock space cutoff dimension
            constellation_radius: Radius of coherent state constellation
            enable_correlations: Enable inter-mode correlations
        """
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.constellation_radius = constellation_radius
        self.enable_correlations = enable_correlations
        
        # SF Program-Engine components
        self.prog = sf.Program(n_modes)
        self.engine = sf.Engine("tf", backend_options={
            "cutoff_dim": cutoff_dim,
            "eval": True,
            "batch_size": None
        })
        
        # Parameter management
        self.tf_parameters = {}
        self.sf_param_names = []
        self.param_mapping = {}
        
        # Constellation parameters (unique for each mode)
        self.constellation_alphas = self._create_constellation_points()
        
        # Build the enhanced symbolic program
        self._build_constellation_program()
        self._create_tf_parameters()
        
        logger.info(f"🌟 Multimode Constellation Circuit initialized:")
        logger.info(f"   Modes: {n_modes}, Layers: {n_layers}")
        logger.info(f"   Constellation radius: {constellation_radius}")
        logger.info(f"   Inter-mode correlations: {enable_correlations}")
        logger.info(f"   Total parameters: {len(self.sf_param_names)}")
        logger.info(f"   Constellation points: {[f'{α:.3f}' for α in self.constellation_alphas[:3]]}...")
    
    def _create_constellation_points(self) -> List[complex]:
        """
        Create constellation of coherent state starting points using communication theory.
        
        Uses perfect equally-spaced constellation (like QPSK, 16-QAM) for maximum
        orthogonality and mode separation - no randomness!
        
        Returns:
            List of complex alpha values for each mode
        """
        constellation_points = []
        
        for i in range(self.n_modes):
            if self.n_modes == 1:
                # Single mode at origin (fallback)
                alpha = 0.0 + 0.0j
            else:
                # 🌟 COMMUNICATION THEORY CONSTELLATION: Perfect equal spacing
                # No randomness - maximum orthogonality and separation
                angle = 2 * np.pi * i / self.n_modes  # Perfect geometric spacing
                radius = self.constellation_radius    # Fixed radius for all modes
                
                # Complex amplitude: α = r * e^(iθ) 
                alpha = radius * np.exp(1j * angle)
            
            constellation_points.append(alpha)
        
        logger.info(f"Created communication theory constellation with {len(constellation_points)} equally-spaced coherent states")
        logger.info(f"Perfect angular separation: {360/self.n_modes:.1f}° between modes")
        
        return constellation_points
    
    def _build_constellation_program(self) -> None:
        """Build quantum program with multimode constellation initialization."""
        
        with self.prog.context as q:
            
            # 🌟 PHASE 1: COHERENT STATE CONSTELLATION INITIALIZATION
            # Each mode starts at a unique coherent state instead of vacuum
            for mode in range(self.n_modes):
                # Real and imaginary parts as separate parameters for flexibility
                alpha_real = self.prog.params(f'constellation_real_{mode}')
                alpha_imag = self.prog.params(f'constellation_imag_{mode}')
                
                # Create coherent state displacement
                # Note: SF Dgate expects (alpha, phi) or (r, phi) format
                # We'll use amplitude and phase representation
                alpha_magnitude = self.prog.params(f'constellation_mag_{mode}')
                alpha_phase = self.prog.params(f'constellation_phase_{mode}')
                
                ops.Dgate(alpha_magnitude, alpha_phase) | q[mode]
                
                self.sf_param_names.extend([
                    f'constellation_mag_{mode}',
                    f'constellation_phase_{mode}'
                ])
            
            # 🌟 PHASE 2: VARIATIONAL QUANTUM LAYERS
            # Now apply variational transformations on top of constellation
            for layer in range(self.n_layers):
                
                # Inter-mode entangling operations (beam splitters)
                if self.enable_correlations:
                    for mode in range(self.n_modes - 1):
                        # Beam splitter for inter-mode correlations
                        theta_param = self.prog.params(f'bs_theta_{layer}_{mode}')
                        phi_param = self.prog.params(f'bs_phi_{layer}_{mode}')
                        ops.BSgate(theta_param, phi_param) | (q[mode], q[mode + 1])
                        
                        self.sf_param_names.extend([
                            f'bs_theta_{layer}_{mode}',
                            f'bs_phi_{layer}_{mode}'
                        ])
                
                # Individual mode transformations
                for mode in range(self.n_modes):
                    # Squeezing for each mode
                    squeeze_r = self.prog.params(f'squeeze_r_{layer}_{mode}')
                    squeeze_phi = self.prog.params(f'squeeze_phi_{layer}_{mode}')
                    ops.Sgate(squeeze_r, squeeze_phi) | q[mode]
                    
                    # Phase rotation
                    rotation_phi = self.prog.params(f'rotation_{layer}_{mode}')
                    ops.Rgate(rotation_phi) | q[mode]
                    
                    # Additional displacement for input encoding
                    disp_alpha = self.prog.params(f'displacement_alpha_{layer}_{mode}')
                    disp_phi = self.prog.params(f'displacement_phi_{layer}_{mode}')
                    ops.Dgate(disp_alpha, disp_phi) | q[mode]
                    
                    self.sf_param_names.extend([
                        f'squeeze_r_{layer}_{mode}',
                        f'squeeze_phi_{layer}_{mode}',
                        f'rotation_{layer}_{mode}',
                        f'displacement_alpha_{layer}_{mode}',
                        f'displacement_phi_{layer}_{mode}'
                    ])
            
            # 🌟 PHASE 3: FINAL INTER-MODE MIXING
            # Add final mixing layer to enhance correlations
            if self.enable_correlations:
                for mode in range(self.n_modes - 1):
                    final_theta = self.prog.params(f'final_bs_theta_{mode}')
                    ops.BSgate(final_theta) | (q[mode], q[mode + 1])
                    self.sf_param_names.append(f'final_bs_theta_{mode}')
        
        logger.info(f"Constellation program built with {len(self.sf_param_names)} parameters")
    
    def _create_tf_parameters(self) -> None:
        """Create TensorFlow variables for all parameters including constellation."""
        
        mode_idx = 0
        
        for param_name in self.sf_param_names:
            
            # Initialize constellation parameters with unique values
            if 'constellation_mag' in param_name:
                # Magnitude of constellation point
                mode_num = int(param_name.split('_')[-1])
                constellation_alpha = self.constellation_alphas[mode_num]
                initial_value = tf.constant([abs(constellation_alpha)])
                
            elif 'constellation_phase' in param_name:
                # Phase of constellation point
                mode_num = int(param_name.split('_')[-1])
                constellation_alpha = self.constellation_alphas[mode_num]
                initial_value = tf.constant([np.angle(constellation_alpha)])
                
            elif 'squeeze_r' in param_name:
                # Squeezing magnitude: small positive values
                initial_value = tf.random.uniform([1], 0.0, 0.2)
                
            elif 'squeeze_phi' in param_name:
                # Squeezing phase: random
                initial_value = tf.random.uniform([1], 0, 2*np.pi)
                
            elif 'bs_theta' in param_name:
                # Beam splitter mixing angle: around π/4 for balanced mixing
                initial_value = tf.random.normal([1], mean=np.pi/4, stddev=0.2)
                
            elif 'bs_phi' in param_name:
                # Beam splitter phase: random
                initial_value = tf.random.uniform([1], 0, 2*np.pi)
                
            elif 'rotation' in param_name:
                # Phase rotation: random
                initial_value = tf.random.uniform([1], 0, 2*np.pi)
                
            elif 'displacement_alpha' in param_name:
                # Additional displacement magnitude: small
                initial_value = tf.random.normal([1], stddev=0.1)
                
            elif 'displacement_phi' in param_name:
                # Additional displacement phase: random
                initial_value = tf.random.uniform([1], 0, 2*np.pi)
                
            else:
                # Default: small random values
                initial_value = tf.random.normal([1], stddev=0.1)
            
            # Create tf.Variable
            tf_var = tf.Variable(initial_value, name=param_name)
            self.tf_parameters[param_name] = tf_var
            self.param_mapping[param_name] = tf_var
        
        logger.info(f"Created {len(self.tf_parameters)} TensorFlow variables")
    
    def execute(self, 
                input_encoding: Optional[tf.Tensor] = None,
                parameter_modulation: Optional[Dict[str, tf.Tensor]] = None) -> Any:
        """
        Execute quantum circuit with constellation initialization.
        
        Args:
            input_encoding: Input-dependent parameter modulation
            parameter_modulation: Direct parameter modulation
            
        Returns:
            SF quantum state with multimode diversity
        """
        # Create parameter arguments
        args = {}
        
        # Base parameters from tf.Variables
        for param_name, tf_var in self.tf_parameters.items():
            args[param_name] = tf.squeeze(tf_var)
        
        # Apply input encoding to displacement parameters
        if input_encoding is not None:
            args = self._apply_multimode_input_encoding(args, input_encoding)
        
        # Apply direct parameter modulation
        if parameter_modulation is not None:
            for param_name, modulation in parameter_modulation.items():
                if param_name in args:
                    args[param_name] = args[param_name] + tf.squeeze(modulation)
        
        # Execute SF program
        try:
            if self.engine.run_progs:
                self.engine.reset()
            
            result = self.engine.run(self.prog, args=args)
            return result.state
            
        except Exception as e:
            logger.error(f"Constellation circuit execution failed: {e}")
            raise
    
    def _apply_multimode_input_encoding(self, 
                                       args: Dict[str, tf.Tensor], 
                                       input_encoding: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Apply input encoding to enhance multimode diversity.
        
        Args:
            args: Current parameter arguments
            input_encoding: Input encoding tensor
            
        Returns:
            Enhanced parameter arguments with multimode encoding
        """
        # Handle batch processing
        if len(input_encoding.shape) > 1:
            mean_encoding = tf.reduce_mean(input_encoding, axis=0)
        else:
            mean_encoding = input_encoding
        
        encoding_dim = tf.shape(mean_encoding)[0]
        encoding_strength = 0.3  # Moderate strength for stability
        
        # Apply encoding to displacement parameters across all modes and layers
        encoding_idx = 0
        
        for layer in range(self.n_layers):
            for mode in range(self.n_modes):
                
                # Modulate displacement amplitude
                alpha_key = f'displacement_alpha_{layer}_{mode}'
                if alpha_key in args and encoding_idx < encoding_dim:
                    modulation = encoding_strength * mean_encoding[encoding_idx]
                    args[alpha_key] = args[alpha_key] + modulation
                    encoding_idx = (encoding_idx + 1) % encoding_dim
                
                # Modulate displacement phase  
                phi_key = f'displacement_phi_{layer}_{mode}'
                if phi_key in args and encoding_idx < encoding_dim:
                    modulation = encoding_strength * mean_encoding[encoding_idx]
                    args[phi_key] = args[phi_key] + modulation
                    encoding_idx = (encoding_idx + 1) % encoding_dim
        
        return args
    
    def execute_batch(self, input_encodings: tf.Tensor) -> List[Any]:
        """
        Execute batch with genuine multimode diversity preservation.
        
        Args:
            input_encodings: Batch of input encodings [batch_size, encoding_dim]
            
        Returns:
            List of quantum states with multimode diversity
        """
        batch_size = tf.shape(input_encodings)[0]
        quantum_states = []
        
        # 🌟 MULTIMODE CORRELATION PRESERVATION
        # Each sample gets its own multimode signature while preserving correlations
        
        for i in range(batch_size):
            sample_encoding = input_encodings[i]
            
            # Execute with individual sample encoding (no averaging!)
            # This preserves the unique multimode character of each sample
            quantum_state = self.execute(input_encoding=sample_encoding)
            quantum_states.append(quantum_state)
        
        return quantum_states
    
    def extract_measurements(self, state: Any) -> tf.Tensor:
        """
        Extract comprehensive multimode measurements.
        
        Args:
            state: SF quantum state
            
        Returns:
            Comprehensive measurement tensor with all modes represented
        """
        measurements = []
        
        # Extract quadrature measurements for all modes
        for mode in range(self.n_modes):
            # X quadrature (position)
            x_quad = state.quad_expectation(mode, 0)
            if tf.rank(x_quad) > 0:
                x_quad = tf.reduce_mean(x_quad)
            measurements.append(x_quad)
            
            # P quadrature (momentum)
            p_quad = state.quad_expectation(mode, np.pi/2)
            if tf.rank(p_quad) > 0:
                p_quad = tf.reduce_mean(p_quad)
            measurements.append(p_quad)
        
        # Stack measurements
        measurement_tensor = tf.stack(measurements)
        measurement_tensor = tf.reshape(measurement_tensor, [-1])
        measurement_tensor = tf.cast(measurement_tensor, tf.float32)
        
        return measurement_tensor
    
    def extract_batch_measurements(self, quantum_states: List[Any]) -> tf.Tensor:
        """Extract measurements from batch with multimode preservation."""
        batch_measurements = []
        
        for state in quantum_states:
            measurements = self.extract_measurements(state)
            batch_measurements.append(measurements)
        
        return tf.stack(batch_measurements, axis=0)
    
    def get_measurement_dimension(self) -> int:
        """Get dimension of measurement output (2 * n_modes for x,p quadratures)."""
        return 2 * self.n_modes
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables."""
        return list(self.tf_parameters.values())
    
    def get_constellation_info(self) -> Dict[str, Any]:
        """Get information about the coherent state constellation."""
        return {
            'n_modes': self.n_modes,
            'constellation_radius': self.constellation_radius,
            'constellation_points': [
                {'mode': i, 'alpha': complex(self.constellation_alphas[i]), 
                 'magnitude': abs(self.constellation_alphas[i]),
                 'phase': np.angle(self.constellation_alphas[i])}
                for i in range(self.n_modes)
            ],
            'enable_correlations': self.enable_correlations,
            'parameter_count': len(self.tf_parameters),
            'measurement_dim': self.get_measurement_dimension()
        }
    
    @staticmethod
    def get_static_constellation_points(n_modes: int, 
                                       constellation_radius: float = 1.5) -> List[complex]:
        """
        Get static constellation points for pipeline integration.
        
        This method provides the constellation points as static constants
        for use in other quantum circuits without creating a full constellation circuit.
        Perfect for pipeline integration with PureSFQuantumCircuit.
        
        Args:
            n_modes: Number of quantum modes
            constellation_radius: Radius of constellation circle
            
        Returns:
            List of static constellation points (not trainable)
        """
        constellation_points = []
        
        for i in range(n_modes):
            if n_modes == 1:
                alpha = 0.0 + 0.0j
            else:
                # 🌟 COMMUNICATION THEORY CONSTELLATION: Perfect equal spacing
                angle = 2 * np.pi * i / n_modes  # Perfect geometric spacing
                radius = constellation_radius    # Fixed radius for all modes
                alpha = radius * np.exp(1j * angle)
            
            constellation_points.append(alpha)
        
        logger.info(f"Generated static constellation: {n_modes} modes, radius={constellation_radius}")
        logger.info(f"Perfect angular separation: {360/n_modes:.1f}° between modes")
        
        return constellation_points
    
    @staticmethod
    def create_constellation_displacements(constellation_points: List[complex]) -> List[Tuple[float, float]]:
        """
        Convert constellation points to SF displacement parameters.
        
        Args:
            constellation_points: List of complex constellation points
            
        Returns:
            List of (magnitude, phase) tuples for SF Dgate operations
        """
        displacements = []
        
        for alpha in constellation_points:
            magnitude = abs(alpha)
            phase = np.angle(alpha)
            displacements.append((magnitude, phase))
        
        return displacements


def test_constellation_circuit():
    """Test the multimode constellation circuit."""
    print("🌟 Testing Multimode Constellation Circuit...")
    
    # Create constellation circuit
    circuit = MultimodalConstellationCircuit(
        n_modes=4,
        n_layers=2,
        cutoff_dim=6,
        constellation_radius=1.5,
        enable_correlations=True
    )
    
    print(f"Created constellation circuit with {len(circuit.trainable_variables)} parameters")
    
    # Show constellation info
    constellation_info = circuit.get_constellation_info()
    print("🌟 Constellation Points:")
    for point in constellation_info['constellation_points']:
        print(f"   Mode {point['mode']}: α = {point['alpha']:.3f} "
              f"(r={point['magnitude']:.3f}, φ={point['phase']:.3f})")
    
    # Test execution
    try:
        print("\nTesting constellation execution...")
        state = circuit.execute()
        measurements = circuit.extract_measurements(state)
        print(f"✅ Execution successful, measurements shape: {measurements.shape}")
        print(f"   Measurements per mode: {circuit.get_measurement_dimension() // circuit.n_modes}")
        
        # Check measurement diversity across modes
        measurements_per_mode = 2  # x and p quadratures
        mode_variances = []
        
        for mode in range(circuit.n_modes):
            start_idx = mode * measurements_per_mode
            end_idx = start_idx + measurements_per_mode
            mode_measurements = measurements[start_idx:end_idx]
            mode_variance = tf.math.reduce_variance(mode_measurements)
            mode_variances.append(float(mode_variance))
        
        print(f"   Mode variances: {[f'{v:.4f}' for v in mode_variances]}")
        
        # Test batch processing
        print("\nTesting batch processing...")
        input_batch = tf.random.normal([3, 8])  # 3 samples, 8 encoding dims
        quantum_states = circuit.execute_batch(input_batch)
        batch_measurements = circuit.extract_batch_measurements(quantum_states)
        
        print(f"✅ Batch processing successful: {batch_measurements.shape}")
        
        # Analyze multimode diversity
        batch_variance_per_mode = []
        for mode in range(circuit.n_modes):
            start_idx = mode * measurements_per_mode
            end_idx = start_idx + measurements_per_mode
            mode_batch_measurements = batch_measurements[:, start_idx:end_idx]
            mode_batch_variance = tf.math.reduce_variance(mode_batch_measurements)
            batch_variance_per_mode.append(float(mode_batch_variance))
        
        print(f"   Batch mode variances: {[f'{v:.4f}' for v in batch_variance_per_mode]}")
        
        # Check if all modes are active (variance > threshold)
        active_modes = sum(1 for v in batch_variance_per_mode if v > 0.001)
        print(f"   Active modes: {active_modes}/{circuit.n_modes}")
        
        if active_modes == circuit.n_modes:
            print("🎉 SUCCESS: All modes show significant activity!")
        else:
            print(f"⚠️  Only {active_modes} modes active, some diversity may be lost")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_constellation_circuit()
    if success:
        print("\n🌟 Multimode Constellation Circuit ready for integration!")
    else:
        print("\n❌ Constellation Circuit test failed!")
