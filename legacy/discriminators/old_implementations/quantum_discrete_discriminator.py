import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf

class QuantumDiscreteDiscriminator:
    """Qubit-based Quantum Discriminator using PennyLane.
    
    Key Concepts:
    - Angle embedding of classical data into qubits
    - Parametrized quantum circuit for classification
    - Expectation value measurement as decision score
    """
    
    def __init__(self, n_qubits=5, n_layers=2):
        """Initialize quantum circuit.
        
        Args:
            n_qubits (int): Number of qubits (input_dim must match n_qubits).
            n_layers (int): Depth of parametrized quantum circuit.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Quantum device (default simulator)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Quantum circuit definition
        def qc_circuit(x, params):
            """Quantum circuit for discrimination.
            1. Encode input data into qubit rotations
            2. Apply trainable parametrized gates
            3. Measure decision qubit
            """
            # Reshape parameters into layers
            params = tf.reshape(params, (n_layers, n_qubits))
            
            # Encode input data into Y-rotations
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
            
            # Parametrized layers
            for layer in range(n_layers):
                # Y-rotations on all qubits
                for wire in range(n_qubits)):
                    qml.RY(params[layer, wire], wires=wire)
                # Entanglement
                for wire in range(n_qubits)):
                    qml.CZ(wires=[wire, (wire+1) % n_qubits])
            
            # Measure first qubit for classification
            return qml.expval(qml.PauliZ(0))
        
        # Create QNode
        self.qnode = qml.QNode(qc_circuit, self.dev, interface="tf")
        
        # Trainable parameters
        self.params = tf.Variable(
            tf.random.uniform(shape=[n_layers * n_qubits], minval=0, maxval=2*np.pi),
            name="discriminator_weights"
        )

    def discriminate(self, x):
        """Classify input samples using quantum circuit.
        
        Args:
            x (tensor): Input data of shape [batch_size, n_qubits].
        
        Returns:
            tensor: Probability scores (post-processed via sigmoid).
        """
        # Scale input to [-π, π] for angle embedding
        x_scaled = tf.math.scalar_mul(np.pi, x)
        
        # Process each sample in batch
        expectations = tf.stack([self.qnode(x_batch, self.params) for x_batch in x_scaled])
        
        # Convert expectation values (-1 to 1) to probabilities (0 to 1)
        return tf.math.sigmoid(expectations * 2.0)  # Scale to match sigmoid input