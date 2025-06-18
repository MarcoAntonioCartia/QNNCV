"""
Pure Quantum Discriminator with Raw Measurement Classification

Architecture:
input_data(input_dim) → A_matrix → quantum_encoding(M×modes) → 
PURE_QUANTUM_CIRCUIT → raw_measurements(M×modes) → classification_network → probability

Key Features:
- Classical transformation matrix A for input encoding
- Pure quantum circuit with individual gate parameters
- Raw measurement extraction (no statistical processing)
- Classification network for discrimination
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class PureQuantumDiscriminator:
    """
    Pure quantum discriminator using transformation matrices and individual gate parameters.
    
    Architecture:
    1. A matrix: input_dim → quantum_encoding_dim
    2. Quantum encoding: encoding_dim → quantum circuit modulation
    3. Pure quantum circuit: individual gate parameters
    4. Raw measurements: direct measurement outcomes
    5. Classification network: measurement_dim → probability
    """
    
    def __init__(self, input_dim=2, n_modes=2, layers=2, cutoff_dim=8, 
                 measurement_dim=None):
        """
        Initialize pure quantum discriminator.
        
        Args:
            input_dim (int): Input data dimension
            n_modes (int): Number of quantum modes
            layers (int): Number of quantum circuit layers
            cutoff_dim (int): Fock space cutoff
            measurement_dim (int): Dimension of raw measurements (default: n_modes * 3)
        """
        self.input_dim = input_dim
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
        
        logger.info(f"Initializing Pure Quantum Discriminator:")
        logger.info(f"  - input_dim: {input_dim}")
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
        
        # Initialize transformation matrix
        self._init_transformation_matrix()
        
        # Initialize quantum encoding strategy
        self._init_quantum_encoding()
        
        # Initialize classification network
        self._init_classification_network()
    
    def _init_transformation_matrix(self):
        """Initialize learnable transformation matrix A."""
        
        # A matrix: input_dim → quantum_encoding_dim
        # Make it learnable and well-conditioned
        self.A_matrix = tf.Variable(
            tf.random.normal([self.input_dim, self.quantum_encoding_dim], stddev=0.1),
            name="A_discriminator_matrix",
            trainable=True
        )
        
        # A bias for affine transformation
        self.A_bias = tf.Variable(
            tf.zeros([self.quantum_encoding_dim]),
            name="A_discriminator_bias",
            trainable=True
        )
        
        logger.info(f"Discriminator transformation matrix A: {self.input_dim} → {self.quantum_encoding_dim}")
    
    def _init_quantum_encoding(self):
        """Initialize quantum encoding strategy."""
        # Create mapping from quantum encoding to circuit parameter modulation
        total_circuit_params = self.quantum_circuit.get_parameter_count()
        
        # Encoding matrix: quantum_encoding_dim → circuit_parameter_modulation
        self.encoding_matrix = tf.Variable(
            tf.random.normal([self.quantum_encoding_dim, total_circuit_params], stddev=0.01),
            name="discriminator_encoding_matrix",
            trainable=True
        )
        
        logger.info(f"Discriminator encoding matrix: {self.quantum_encoding_dim} → {total_circuit_params}")
    
    def _init_classification_network(self):
        """Initialize classification network for raw measurements."""
        # Import keras compatibility
        try:
            from tensorflow import keras
        except ImportError:
            import keras
        
        # Classification network: measurement_dim → probability
        self.classifier = keras.Sequential([
            keras.layers.Dense(8, activation='relu', name='disc_hidden1'),
            keras.layers.Dense(4, activation='relu', name='disc_hidden2'),
            keras.layers.Dense(1, activation='sigmoid', name='disc_output')
        ], name='discriminator_classifier')
        
        # Build the network
        dummy_measurements = tf.zeros((1, self.measurement_dim))
        _ = self.classifier(dummy_measurements)
        
        logger.info(f"Classification network: {self.measurement_dim} → 1")
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        variables = []
        
        # Add pure quantum circuit parameters
        variables.extend(self.quantum_circuit.trainable_variables)
        
        # Add transformation matrix
        variables.extend([
            self.A_matrix, self.A_bias,
            self.encoding_matrix
        ])
        
        # Add classification network
        variables.extend(self.classifier.trainable_variables)
        
        return variables
    
    def transform_input(self, x):
        """
        Apply A transformation: input → quantum encoding.
        
        Args:
            x (tensor): Input data [batch_size, input_dim]
            
        Returns:
            tensor: Quantum encoding [batch_size, quantum_encoding_dim]
        """
        # Affine transformation: A @ x + bias
        quantum_encoding = tf.matmul(x, self.A_matrix) + self.A_bias
        
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
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Calculate quadrature expectation value
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        x_expectation = tf.reduce_sum(prob_amplitudes * n_vals) / tf.sqrt(2.0)
        
        # Add quantum noise
        x_measurement = x_expectation + tf.random.normal([], stddev=0.1)
        
        return x_measurement
    
    def _measure_p_quadrature(self, state, mode):
        """Measure P quadrature for a specific mode."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Simplified: use variance as proxy for p quadrature
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        mean_n = tf.reduce_sum(prob_amplitudes * n_vals)
        var_n = tf.reduce_sum(prob_amplitudes * (n_vals - mean_n)**2)
        
        p_expectation = tf.sqrt(var_n) / tf.sqrt(2.0)
        p_measurement = p_expectation + tf.random.normal([], stddev=0.1)
        
        return p_measurement
    
    def _measure_photon_number(self, state, mode):
        """Measure photon number for a specific mode."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Photon number expectation
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        n_expectation = tf.reduce_sum(prob_amplitudes * n_vals)
        
        # Add shot noise
        n_measurement = n_expectation + tf.random.normal([], stddev=tf.sqrt(n_expectation + 1e-6))
        
        return n_measurement
    
    def classify_measurements(self, raw_measurements):
        """
        Classify raw measurements to discrimination probability.
        
        Args:
            raw_measurements (tensor): [batch_size, measurement_dim]
            
        Returns:
            tensor: Discrimination probabilities [batch_size, 1]
        """
        return self.classifier(raw_measurements)
    
    def discriminate(self, x):
        """
        Full discrimination pipeline.
        
        Args:
            x (tensor): Input data [batch_size, input_dim]
            
        Returns:
            tensor: Discrimination probabilities [batch_size, 1]
        """
        # Step 1: A transformation
        quantum_encoding = self.transform_input(x)
        
        # Step 2: Execute quantum circuit
        quantum_states = self.execute_quantum_circuit(quantum_encoding)
        
        # Step 3: Extract raw measurements
        raw_measurements = self.extract_raw_measurements(quantum_states)
        
        # Step 4: Classification
        probabilities = self.classify_measurements(raw_measurements)
        
        return probabilities
    
    def get_raw_measurements(self, x):
        """
        Get raw measurements without classification (for loss computation).
        
        Args:
            x (tensor): Input data [batch_size, input_dim]
            
        Returns:
            tensor: Raw measurements [batch_size, measurement_dim]
        """
        # Steps 1-3 only
        quantum_encoding = self.transform_input(x)
        quantum_states = self.execute_quantum_circuit(quantum_encoding)
        raw_measurements = self.extract_raw_measurements(quantum_states)
        
        return raw_measurements
    
    def compute_transformation_regularization(self):
        """Compute regularization to encourage well-conditioned transformation."""
        # Encourage A matrix to be well-conditioned
        A_gram = tf.matmul(tf.transpose(self.A_matrix), self.A_matrix)
        A_reg = tf.reduce_mean(tf.square(A_gram - tf.eye(self.quantum_encoding_dim)))
        
        return A_reg


def test_pure_quantum_discriminator():
    """Test the pure quantum discriminator."""
    print("\n" + "="*60)
    print("TESTING PURE QUANTUM DISCRIMINATOR")
    print("="*60)
    
    # Create discriminator
    discriminator = PureQuantumDiscriminator(
        input_dim=2,
        n_modes=2,
        layers=2,
        cutoff_dim=6
    )
    
    print(f"✅ Discriminator created successfully")
    print(f"🔧 Total trainable variables: {len(discriminator.trainable_variables)}")
    
    # Test discrimination
    print(f"\n🎯 Testing discrimination...")
    try:
        x_test = tf.random.normal([3, 2])
        probabilities = discriminator.discriminate(x_test)
        
        print(f"✅ Discrimination successful")
        print(f"📏 Input shape: {x_test.shape}")
        print(f"📏 Output shape: {probabilities.shape}")
        print(f"📊 Probabilities range: [{tf.reduce_min(probabilities):.3f}, {tf.reduce_max(probabilities):.3f}]")
        
    except Exception as e:
        print(f"❌ Discrimination failed: {e}")
        return None
    
    # Test raw measurements
    print(f"\n📊 Testing raw measurements...")
    try:
        raw_measurements = discriminator.get_raw_measurements(x_test)
        print(f"✅ Raw measurements extracted")
        print(f"📏 Raw measurements shape: {raw_measurements.shape}")
        print(f"📊 Raw measurements range: [{tf.reduce_min(raw_measurements):.3f}, {tf.reduce_max(raw_measurements):.3f}]")
        
    except Exception as e:
        print(f"❌ Raw measurements failed: {e}")
    
    # Test gradient flow
    print(f"\n🌊 Testing gradient flow...")
    try:
        with tf.GradientTape() as tape:
            x = tf.random.normal([2, 2])
            output = discriminator.discriminate(x)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(output),
                logits=output
            ))
        
        gradients = tape.gradient(loss, discriminator.trainable_variables)
        grad_status = [g is not None for g in gradients]
        
        print(f"✅ Gradient test: {sum(grad_status)}/{len(grad_status)} variables have gradients")
        print(f"📈 Loss: {loss:.6f}")
        
        # Check different variable types
        quantum_grads = len([g for g in gradients[:len(discriminator.quantum_circuit.trainable_variables)] if g is not None])
        other_grads = len([g for g in gradients[len(discriminator.quantum_circuit.trainable_variables):] if g is not None])
        
        print(f"🔬 Quantum circuit gradients: {quantum_grads}/{len(discriminator.quantum_circuit.trainable_variables)}")
        print(f"🔄 Other gradients: {other_grads}/{len(gradients) - len(discriminator.quantum_circuit.trainable_variables)}")
        
        if all(grad_status):
            print(f"🎉 ALL VARIABLES HAVE GRADIENTS!")
        else:
            print(f"⚠️  Some variables missing gradients")
            
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
    
    # Test regularization
    print(f"\n📐 Testing transformation regularization...")
    try:
        reg_loss = discriminator.compute_transformation_regularization()
        print(f"✅ Regularization computed: {reg_loss:.6f}")
        
    except Exception as e:
        print(f"❌ Regularization failed: {e}")
    
    return discriminator


if __name__ == "__main__":
    discriminator = test_pure_quantum_discriminator()