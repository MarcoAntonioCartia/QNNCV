"""
Quantum Discriminator for QGANs.
This implements both true quantum and quantum-inspired discriminators.
"""

import numpy as np
import tensorflow as tf

# Quantum library imports with fallbacks
try:
    import strawberryfields as sf
    from strawberryfields.ops import *
    SF_AVAILABLE = True
    print("✓ Strawberry Fields available for quantum discriminator")
except ImportError:
    SF_AVAILABLE = False
    print("⚠ Strawberry Fields not available - using quantum-inspired discriminator")

try:
    import pennylane as qml
    PL_AVAILABLE = True
    print("✓ PennyLane available for quantum discriminator")
except ImportError:
    PL_AVAILABLE = False
    print("⚠ PennyLane not available")

class QuantumDiscriminator:
    """
    Quantum Discriminator that can distinguish between real and quantum-generated data.
    
    This implementation provides multiple quantum approaches:
    1. Strawberry Fields continuous variable quantum computing
    2. PennyLane discrete variable quantum computing  
    3. Quantum-inspired classical neural network fallback
    """
    
    def __init__(self, input_dim=2, n_qubits=4, n_layers=3, backend='auto'):
        """
        Initialize quantum discriminator.
        
        Args:
            input_dim (int): Dimension of input data
            n_qubits (int): Number of qubits/qumodes for quantum processing
            n_layers (int): Number of quantum circuit layers
            backend (str): Quantum backend ('strawberry_fields', 'pennylane', 'classical', 'auto')
        """
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Determine backend
        if backend == 'auto':
            if SF_AVAILABLE:
                self.backend = 'strawberry_fields'
            elif PL_AVAILABLE:
                self.backend = 'pennylane'
            else:
                self.backend = 'classical'
        else:
            self.backend = backend
        
        print(f"Initializing quantum discriminator with backend: {self.backend}")
        
        # Initialize based on selected backend
        if self.backend == 'strawberry_fields' and SF_AVAILABLE:
            self._init_strawberry_fields()
        elif self.backend == 'pennylane' and PL_AVAILABLE:
            self._init_pennylane()
        else:
            self._init_classical_quantum_inspired()
    
    def _init_strawberry_fields(self):
        """Initialize Strawberry Fields continuous variable quantum discriminator."""
        self.cutoff_dim = 8
        self.engine = sf.Engine("tf", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Quantum parameters for discrimination
        self.sf_squeeze_params = tf.Variable(
            tf.random.uniform([self.n_qubits], -0.3, 0.3),
            name="sf_squeeze_params",
            trainable=True
        )
        
        self.sf_rotation_params = tf.Variable(
            tf.random.uniform([self.n_qubits, self.n_layers], 0, 2*np.pi),
            name="sf_rotation_params",
            trainable=True
        )
        
        self.sf_displacement_params = tf.Variable(
            tf.random.uniform([self.n_qubits, self.n_layers, 2], -1, 1),
            name="sf_displacement_params",
            trainable=True
        )
        
        # Classical post-processing for final discrimination
        self.sf_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
    
    def _init_pennylane(self):
        """Initialize PennyLane discrete variable quantum discriminator."""
        # Create quantum device
        self.pl_device = qml.device('default.qubit', wires=self.n_qubits)
        
        # Quantum parameters
        self.pl_weights = tf.Variable(
            tf.random.uniform([self.n_layers, self.n_qubits, 3], 0, 2*np.pi),
            name="pl_weights",
            trainable=True
        )
        
        # Input encoding parameters
        self.pl_encoding_weights = tf.Variable(
            tf.random.uniform([self.input_dim, self.n_qubits], 0, 2*np.pi),
            name="pl_encoding_weights",
            trainable=True
        )
        
        # Create quantum neural network
        @qml.qnode(self.pl_device, interface='tf')
        def quantum_circuit(inputs, weights, encoding_weights):
            # Data encoding
            for i in range(self.input_dim):
                for j in range(self.n_qubits):
                    qml.RY(inputs[i] * encoding_weights[i, j], wires=j)
            
            # Quantum layers
            for layer in range(self.n_layers):
                # Entangling layer
                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])  # Circular entanglement
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        self.pl_quantum_circuit = quantum_circuit
        
        # Classical post-processing
        self.pl_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def _init_classical_quantum_inspired(self):
        """Initialize quantum-inspired classical discriminator."""
        # Quantum-inspired parameters that simulate quantum effects
        self.qi_quantum_params = tf.Variable(
            tf.random.normal([self.n_qubits, self.n_layers, 4]),  # [amplitude, phase, entanglement, measurement]
            name="qi_quantum_params",
            trainable=True
        )
        
        # Quantum-inspired processing layers
        self.qi_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_qubits * 2, activation='tanh'),
            tf.keras.layers.Dense(self.n_qubits, activation=lambda x: tf.sin(x))  # Oscillatory like quantum
        ])
        
        # Quantum-inspired interference layers
        self.qi_interference_layers = []
        for layer in range(self.n_layers):
            layer_net = tf.keras.Sequential([
                tf.keras.layers.Dense(self.n_qubits, activation='tanh'),
                tf.keras.layers.Dense(self.n_qubits, activation=lambda x: tf.cos(x))
            ])
            self.qi_interference_layers.append(layer_net)
        
        # Final classification
        self.qi_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        variables = []
        
        if self.backend == 'strawberry_fields':
            variables.extend([
                self.sf_squeeze_params,
                self.sf_rotation_params, 
                self.sf_displacement_params
            ])
            variables.extend(self.sf_classifier.trainable_variables)
            
        elif self.backend == 'pennylane':
            variables.extend([
                self.pl_weights,
                self.pl_encoding_weights
            ])
            variables.extend(self.pl_classifier.trainable_variables)
            
        else:  # classical quantum-inspired
            variables.append(self.qi_quantum_params)
            variables.extend(self.qi_encoder.trainable_variables)
            for layer in self.qi_interference_layers:
                variables.extend(layer.trainable_variables)
            variables.extend(self.qi_classifier.trainable_variables)
        
        return variables
    
    def _discriminate_strawberry_fields(self, x):
        """Discriminate using Strawberry Fields quantum circuit."""
        batch_size = tf.shape(x)[0]
        results = []
        
        for i in range(batch_size):
            sample = x[i]
            
            # Create quantum program
            prog = sf.Program(self.n_qubits)
            
            with prog.context as q:
                # Encode input data into quantum states
                for j in range(min(self.input_dim, self.n_qubits)):
                    # Displacement encoding
                    alpha = sample[j] * 0.5  # Scale input
                    Dgate(alpha, 0) | q[j]
                
                # Apply quantum processing layers
                for layer in range(self.n_layers):
                    # Squeezing operations
                    for j in range(self.n_qubits):
                        Sgate(self.sf_squeeze_params[j]) | q[j]
                    
                    # Rotation operations
                    for j in range(self.n_qubits):
                        Rgate(self.sf_rotation_params[j, layer]) | q[j]
                    
                    # Displacement operations
                    for j in range(self.n_qubits):
                        Dgate(
                            self.sf_displacement_params[j, layer, 0],
                            self.sf_displacement_params[j, layer, 1]
                        ) | q[j]
                    
                    # Beam splitter entanglement
                    for j in range(0, self.n_qubits - 1, 2):
                        BSgate(np.pi/4, 0) | (q[j], q[j + 1])
            
            # Run quantum circuit
            result = self.engine.run(prog)
            
            # Extract quantum measurements
            measurements = []
            for j in range(self.n_qubits):
                x_quad = result.state.quad_expectation(j, 0)
                p_quad = result.state.quad_expectation(j, 1)
                measurements.extend([x_quad, p_quad])
            
            results.append(measurements)
        
        quantum_features = tf.constant(results, dtype=tf.float32)
        return self.sf_classifier(quantum_features)
    
    def _discriminate_pennylane(self, x):
        """Discriminate using PennyLane quantum circuit."""
        # Convert to numpy to avoid TensorFlow graph issues with PennyLane
        x_numpy = x.numpy() if hasattr(x, 'numpy') else x
        batch_size = x_numpy.shape[0]
        quantum_outputs = []
        
        for i in range(batch_size):
            sample = x_numpy[i]
            # Pad or truncate input to match expected size
            if len(sample) < self.input_dim:
                padded_sample = np.concatenate([sample, np.zeros(self.input_dim - len(sample))])
            else:
                padded_sample = sample[:self.input_dim]
            
            # Run quantum circuit (PennyLane handles numpy arrays better)
            qnn_output = self.pl_quantum_circuit(
                padded_sample, 
                self.pl_weights.numpy(), 
                self.pl_encoding_weights.numpy()
            )
            quantum_outputs.append(float(qnn_output))
        
        # Convert back to TensorFlow tensor
        quantum_features = tf.constant(quantum_outputs, dtype=tf.float32)
        quantum_features = tf.reshape(quantum_features, [-1, 1])  # Ensure correct shape
        
        return self.pl_classifier(quantum_features)
    
    def _discriminate_classical_quantum_inspired(self, x):
        """Discriminate using quantum-inspired classical processing."""
        # Encode input through quantum-inspired encoder
        encoded = self.qi_encoder(x)
        
        # Apply quantum-inspired interference layers
        current_state = encoded
        for layer_idx in range(self.n_layers):
            # Get quantum parameters for this layer
            layer_params = self.qi_quantum_params[:, layer_idx, :]
            
            # Process through interference layer
            interference_output = self.qi_interference_layers[layer_idx](current_state)
            
            # Simulate quantum interference patterns
            amplitude_modulation = tf.sin(interference_output * layer_params[:, 0])
            phase_modulation = tf.cos(interference_output * layer_params[:, 1])
            entanglement_effect = tf.sin(current_state + layer_params[:, 2])
            
            # Combine effects (simulating quantum superposition)
            quantum_effect = amplitude_modulation + phase_modulation + 0.1 * entanglement_effect
            
            # Update state with quantum-like evolution
            current_state = current_state + 0.1 * quantum_effect
        
        # Final classification
        return self.qi_classifier(current_state)
    
    def discriminate(self, x):
        """
        Main discrimination method.
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Discrimination scores [batch_size, 1] (0=fake, 1=real)
        """
        if self.backend == 'strawberry_fields':
            return self._discriminate_strawberry_fields(x)
        elif self.backend == 'pennylane':
            # Use py_function to wrap PennyLane for TensorFlow compatibility
            return tf.py_function(
                func=self._discriminate_pennylane_wrapper,
                inp=[x],
                Tout=tf.float32
            )
        else:
            return self._discriminate_classical_quantum_inspired(x)
    
    def _discriminate_pennylane_wrapper(self, x):
        """Wrapper for PennyLane discrimination to work with tf.py_function."""
        result = self._discriminate_pennylane(x)
        # Ensure proper shape for TensorFlow
        if hasattr(result, 'shape'):
            return result
        else:
            return tf.reshape(tf.constant(result, dtype=tf.float32), [-1, 1])
    
    def get_quantum_info(self):
        """Get information about the quantum discriminator."""
        info = {
            'backend': self.backend,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'input_dim': self.input_dim
        }
        
        if self.backend == 'strawberry_fields':
            info['cutoff_dim'] = self.cutoff_dim
            
        elif self.backend == 'pennylane':
            info['device'] = str(self.pl_device)
            
        return info

# Example usage and testing
if __name__ == "__main__":
    print("Testing Quantum Discriminator")
    print("=" * 30)
    
    # Test different backends
    backends = ['classical', 'pennylane', 'strawberry_fields']
    
    for backend in backends:
        print(f"\nTesting {backend} backend:")
        try:
            discriminator = QuantumDiscriminator(
                input_dim=2,
                n_qubits=4,
                n_layers=2,
                backend=backend
            )
            
            # Test discrimination
            test_data = tf.random.normal([5, 2])
            scores = discriminator.discriminate(test_data)
            
            print(f"  ✓ Discrimination scores shape: {scores.shape}")
            print(f"  ✓ Trainable variables: {len(discriminator.trainable_variables)}")
            print(f"  ✓ Quantum info: {discriminator.get_quantum_info()}")
            
        except Exception as e:
            print(f"  ✗ {backend} backend failed: {e}")
    
    print("\n✅ Quantum discriminator testing completed!") 