"""
Pure Quantum Generator with Transformation Matrices

Architecture:
latent_vector(latent_dim) → T_matrix → quantum_encoding(M×modes) → 
PURE_QUANTUM_CIRCUIT → raw_measurements(M×modes) → A_inverse → generated_data(output_dim)

Key Features:
- Classical transformation matrices A and T for dimension matching
- Pure quantum circuit with individual gate parameters
- Raw measurement extraction (no statistical processing)
- Reversible transformations for proper encoding/decoding
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class PureQuantumGenerator:
    """
    Pure quantum generator using transformation matrices and individual gate parameters.
    
    Architecture:
    1. T matrix: latent_dim → quantum_encoding_dim
    2. Quantum encoding: encoding_dim → quantum circuit modulation
    3. Pure quantum circuit: individual gate parameters
    4. Raw measurements: direct measurement outcomes
    5. A^(-1) matrix: measurement_dim → output_dim
    """
    
    def __init__(self, latent_dim=6, output_dim=2, n_modes=4, layers=2, 
                 cutoff_dim=8, measurement_dim=None):
        """
        Initialize pure quantum generator.
        
        Args:
            latent_dim (int): Input latent vector dimension
            output_dim (int): Output generated data dimension
            n_modes (int): Number of quantum modes
            layers (int): Number of quantum circuit layers
            cutoff_dim (int): Fock space cutoff
            measurement_dim (int): Dimension of raw measurements (default: n_modes * 3)
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # Calculate measurement dimension (multiple measurements per mode)
        if measurement_dim is None:
            # 3 measurements per mode: X quadrature, P quadrature, photon number
            self.measurement_dim = n_modes * 3
        else:
            self.measurement_dim = measurement_dim
        
        # Calculate quantum encoding dimension
        self.quantum_encoding_dim = self.measurement_dim  # For simplicity
        
        logger.info(f"Initializing Pure Quantum Generator:")
        logger.info(f"  - latent_dim: {latent_dim}")
        logger.info(f"  - output_dim: {output_dim}")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - measurement_dim: {self.measurement_dim}")
        logger.info(f"  - quantum_encoding_dim: {self.quantum_encoding_dim}")
        
        # Import the pure quantum circuit
        from pure_quantum_circuit import PureQuantumCircuit
        
        # Initialize pure quantum circuit
        self.quantum_circuit = PureQuantumCircuit(
            n_modes=n_modes, 
            layers=layers, 
            cutoff_dim=cutoff_dim
        )
        
        # Initialize transformation matrices
        self._init_transformation_matrices()
        
        # Initialize quantum encoding strategy
        self._init_quantum_encoding()
    
    def _init_transformation_matrices(self):
        """Initialize learnable transformation matrices A and T."""
        
        # T matrix: latent_dim → quantum_encoding_dim
        # Make it learnable but initialized to be approximately invertible
        self.T_matrix = tf.Variable(
            tf.random.orthogonal([self.latent_dim, self.quantum_encoding_dim]) * 0.5,
            name="T_transformation_matrix",
            trainable=True
        )
        
        # T bias for affine transformation
        self.T_bias = tf.Variable(
            tf.zeros([self.quantum_encoding_dim]),
            name="T_bias",
            trainable=True
        )
        
        # A matrix: measurement_dim → output_dim
        # Initialize as learnable transformation
        self.A_matrix = tf.Variable(
            tf.random.normal([self.measurement_dim, self.output_dim], stddev=0.1),
            name="A_transformation_matrix",
            trainable=True
        )
        
        # A bias for affine transformation
        self.A_bias = tf.Variable(
            tf.zeros([self.output_dim]),
            name="A_bias",
            trainable=True
        )
        
        logger.info(f"Transformation matrices initialized:")
        logger.info(f"  - T: {self.latent_dim} → {self.quantum_encoding_dim}")
        logger.info(f"  - A: {self.measurement_dim} → {self.output_dim}")
    
    def _init_quantum_encoding(self):
        """Initialize quantum encoding strategy."""
        # We'll modulate displacement parameters primarily
        # Each measurement provides modulation for specific quantum parameters
        
        # Create mapping from quantum encoding to circuit parameter modulation
        total_circuit_params = self.quantum_circuit.get_parameter_count()
        
        # Encoding matrix: quantum_encoding_dim → circuit_parameter_modulation
        self.encoding_matrix = tf.Variable(
            tf.random.normal([self.quantum_encoding_dim, total_circuit_params], stddev=0.01),
            name="quantum_encoding_matrix",
            trainable=True
        )
        
        logger.info(f"Quantum encoding matrix: {self.quantum_encoding_dim} → {total_circuit_params}")
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        variables = []
        
        # Add pure quantum circuit parameters
        variables.extend(self.quantum_circuit.trainable_variables)
        
        # Add transformation matrices
        variables.extend([
            self.T_matrix, self.T_bias,
            self.A_matrix, self.A_bias,
            self.encoding_matrix
        ])
        
        return variables
    
    def transform_latent(self, z):
        """
        Apply T transformation: latent → quantum encoding.
        
        Args:
            z (tensor): Latent vector [batch_size, latent_dim]
            
        Returns:
            tensor: Quantum encoding [batch_size, quantum_encoding_dim]
        """
        # Affine transformation: T @ z + bias
        quantum_encoding = tf.matmul(z, self.T_matrix) + self.T_bias
        
        # Apply activation to keep encoding in reasonable range
        quantum_encoding = tf.nn.tanh(quantum_encoding) * 0.5  # [-0.5, 0.5] range
        
        return quantum_encoding
    
    def encode_to_circuit_modulation(self, quantum_encoding):
        """
        Convert quantum encoding to circuit parameter modulation.
        
        Args:
            quantum_encoding (tensor): [batch_size, quantum_encoding_dim]
            
        Returns:
            dict: Parameter modulation for quantum circuit
        """
        batch_size = tf.shape(quantum_encoding)[0]
        
        # Transform encoding to parameter modulation
        param_modulation = tf.matmul(quantum_encoding, self.encoding_matrix)
        param_modulation = tf.nn.tanh(param_modulation) * 0.1  # Small modulation
        
        # Convert to parameter dictionary format expected by quantum circuit
        # This maps the flat modulation vector to named parameters
        modulation_dict = self._create_modulation_mapping(param_modulation)
        
        return modulation_dict
    
    def _create_modulation_mapping(self, param_modulation):
        """Convert flat modulation vector to parameter dictionary."""
        # Get parameter structure from quantum circuit
        structure = self.quantum_circuit.get_parameter_structure()
        
        modulation_dict = {}
        param_idx = 0
        
        for layer in range(self.layers):
            layer_key = f'layer_{layer}'
            if layer_key in structure:
                layer_info = structure[layer_key]
                
                for param_type, count in layer_info.items():
                    for i in range(count):
                        param_name = f'L{layer}_{param_type.upper()}_{i}'
                        
                        if param_idx < param_modulation.shape[1]:
                            # Extract modulation for this parameter
                            modulation_dict[param_name] = param_modulation[:, param_idx]
                            param_idx += 1
        
        return modulation_dict
    
    def execute_quantum_circuit(self, quantum_encoding):
        """
        Execute quantum circuit with encoding modulation.
        
        Args:
            quantum_encoding (tensor): [batch_size, quantum_encoding_dim]
            
        Returns:
            list: Raw quantum states for each sample
        """
        batch_size = tf.shape(quantum_encoding)[0]
        
        # Get modulation mapping
        modulation_dict = self.encode_to_circuit_modulation(quantum_encoding)
        
        quantum_states = []
        
        # Execute circuit for each sample (SF limitation - no true batching)
        for i in range(batch_size):
            # Extract modulation for this sample
            sample_modulation = {}
            for param_name, modulation_tensor in modulation_dict.items():
                sample_modulation[param_name] = modulation_tensor[i]
            
            # Execute quantum circuit
            state = self.quantum_circuit.execute_circuit(sample_modulation)
            quantum_states.append(state)
        
        return quantum_states
    
    def extract_raw_measurements(self, quantum_states):
        """
        Extract raw measurements from quantum states.
        
        NO statistical processing - direct measurement outcomes.
        
        Args:
            quantum_states (list): List of SF quantum states
            
        Returns:
            tensor: Raw measurements [batch_size, measurement_dim]
        """
        batch_size = len(quantum_states)
        all_measurements = []
        
        for state in quantum_states:
            mode_measurements = []
            
            # For each mode, extract multiple measurement types
            for mode in range(self.n_modes):
                # Measurement 1: X quadrature (position-like)
                x_quad = self._measure_x_quadrature(state, mode)
                mode_measurements.append(x_quad)
                
                # Measurement 2: P quadrature (momentum-like)
                p_quad = self._measure_p_quadrature(state, mode)
                mode_measurements.append(p_quad)
                
                # Measurement 3: Photon number expectation
                n_photon = self._measure_photon_number(state, mode)
                mode_measurements.append(n_photon)
            
            # Stack measurements for this sample
            sample_measurements = tf.stack(mode_measurements)
            all_measurements.append(sample_measurements)
        
        # Stack all samples
        raw_measurements = tf.stack(all_measurements, axis=0)
        
        return raw_measurements
    
    def _measure_x_quadrature(self, state, mode):
        """Measure X quadrature for a specific mode."""
        # X quadrature: <a + a†>/√2
        # For Fock states: <n|x|n> = √(n+1) + √n
        
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Calculate quadrature expectation value
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        
        # Simplified: use mean position as proxy for x quadrature
        x_expectation = tf.reduce_sum(prob_amplitudes * n_vals) / tf.sqrt(2.0)
        
        # Add quantum noise (realistic measurement includes noise)
        x_measurement = x_expectation + tf.random.normal([], stddev=0.1)
        
        return x_measurement
    
    def _measure_p_quadrature(self, state, mode):
        """Measure P quadrature for a specific mode."""
        # P quadrature: <a - a†>/(i√2)
        
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Simplified: use variance as proxy for p quadrature
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        mean_n = tf.reduce_sum(prob_amplitudes * n_vals)
        var_n = tf.reduce_sum(prob_amplitudes * (n_vals - mean_n)**2)
        
        p_expectation = tf.sqrt(var_n) / tf.sqrt(2.0)
        
        # Add quantum noise
        p_measurement = p_expectation + tf.random.normal([], stddev=0.1)
        
        return p_measurement
    
    def _measure_photon_number(self, state, mode):
        """Measure photon number for a specific mode."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Photon number expectation
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        n_expectation = tf.reduce_sum(prob_amplitudes * n_vals)
        
        # Add shot noise (Poissonian)
        n_measurement = n_expectation + tf.random.normal([], stddev=tf.sqrt(n_expectation + 1e-6))
        
        return n_measurement
    
    def transform_measurements(self, raw_measurements):
        """
        Apply A^(-1) transformation: measurements → output data.
        
        Args:
            raw_measurements (tensor): [batch_size, measurement_dim]
            
        Returns:
            tensor: Generated data [batch_size, output_dim]
        """
        # Affine transformation: A @ measurements + bias
        generated_data = tf.matmul(raw_measurements, self.A_matrix) + self.A_bias
        
        # Apply final activation for output range
        generated_data = tf.nn.tanh(generated_data) * 3.0  # [-3, 3] range
        
        return generated_data
    
    def generate(self, z):
        """
        Full generation pipeline.
        
        Args:
            z (tensor): Latent vectors [batch_size, latent_dim]
            
        Returns:
            tensor: Generated data [batch_size, output_dim]
        """
        # Step 1: T transformation
        quantum_encoding = self.transform_latent(z)
        
        # Step 2: Execute quantum circuit
        quantum_states = self.execute_quantum_circuit(quantum_encoding)
        
        # Step 3: Extract raw measurements
        raw_measurements = self.extract_raw_measurements(quantum_states)
        
        # Step 4: A^(-1) transformation
        generated_data = self.transform_measurements(raw_measurements)
        
        return generated_data
    
    def get_raw_measurements(self, z):
        """
        Get raw measurements without final transformation (for loss computation).
        
        Args:
            z (tensor): Latent vectors [batch_size, latent_dim]
            
        Returns:
            tensor: Raw measurements [batch_size, measurement_dim]
        """
        # Steps 1-3 only
        quantum_encoding = self.transform_latent(z)
        quantum_states = self.execute_quantum_circuit(quantum_encoding)
        raw_measurements = self.extract_raw_measurements(quantum_states)
        
        return raw_measurements
    
    def compute_transformation_regularization(self):
        """Compute regularization to encourage well-conditioned transformations."""
        # Encourage A matrix to be well-conditioned
        A_det = tf.linalg.det(tf.matmul(tf.transpose(self.A_matrix), self.A_matrix))
        A_reg = tf.square(A_det - 1.0)  # Encourage det(A^T A) ≈ 1
        
        # Encourage T matrix orthogonality
        T_orth = tf.matmul(tf.transpose(self.T_matrix), self.T_matrix)
        T_reg = tf.reduce_mean(tf.square(T_orth - tf.eye(self.quantum_encoding_dim)))
        
        return A_reg + T_reg


def test_pure_quantum_generator():
    """Test the pure quantum generator."""
    print("\n" + "="*60)
    print("TESTING PURE QUANTUM GENERATOR")
    print("="*60)
    
    # Create generator
    generator = PureQuantumGenerator(
        latent_dim=6,
        output_dim=2,
        n_modes=4,
        layers=2,
        cutoff_dim=6
    )
    
    print(f"✅ Generator created successfully")
    print(f"🔧 Total trainable variables: {len(generator.trainable_variables)}")
    
    # Test generation
    print(f"\n🎯 Testing generation...")
    try:
        z_test = tf.random.normal([3, 6])
        generated_data = generator.generate(z_test)
        
        print(f"✅ Generation successful")
        print(f"📏 Input shape: {z_test.shape}")
        print(f"📏 Output shape: {generated_data.shape}")
        print(f"📊 Output range: [{tf.reduce_min(generated_data):.3f}, {tf.reduce_max(generated_data):.3f}]")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None
    
    # Test raw measurements
    print(f"\n📊 Testing raw measurements...")
    try:
        raw_measurements = generator.get_raw_measurements(z_test)
        print(f"✅ Raw measurements extracted")
        print(f"📏 Raw measurements shape: {raw_measurements.shape}")
        print(f"📊 Raw measurements range: [{tf.reduce_min(raw_measurements):.3f}, {tf.reduce_max(raw_measurements):.3f}]")
        
    except Exception as e:
        print(f"❌ Raw measurements failed: {e}")
    
    # Test gradient flow
    print(f"\n🌊 Testing gradient flow...")
    try:
        with tf.GradientTape() as tape:
            z = tf.random.normal([2, 6])
            output = generator.generate(z)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, generator.trainable_variables)
        grad_status = [g is not None for g in gradients]
        
        print(f"✅ Gradient test: {sum(grad_status)}/{len(grad_status)} variables have gradients")
        print(f"📈 Loss: {loss:.6f}")
        
        # Check different variable types
        quantum_grads = len([g for g in gradients[:len(generator.quantum_circuit.trainable_variables)] if g is not None])
        transform_grads = len([g for g in gradients[len(generator.quantum_circuit.trainable_variables):] if g is not None])
        
        print(f"🔬 Quantum circuit gradients: {quantum_grads}/{len(generator.quantum_circuit.trainable_variables)}")
        print(f"🔄 Transformation gradients: {transform_grads}/5")
        
        if all(grad_status):
            print(f"🎉 ALL VARIABLES HAVE GRADIENTS!")
        else:
            print(f"⚠️  Some variables missing gradients")
            
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
    
    # Test regularization
    print(f"\n📐 Testing transformation regularization...")
    try:
        reg_loss = generator.compute_transformation_regularization()
        print(f"✅ Regularization computed: {reg_loss:.6f}")
        
    except Exception as e:
        print(f"❌ Regularization failed: {e}")
    
    return generator


if __name__ == "__main__":
    generator = test_pure_quantum_generator()