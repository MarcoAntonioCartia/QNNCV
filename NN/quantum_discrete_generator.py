import pennylane as qml  # Quantum ML library for qubits  
from pennylane import numpy as np  
import tensorflow as tf  

class QuantumDiscreteGenerator:  
    """Qubit-based Quantum Generator using PennyLane.  

    Key Concepts:  
    - **Qubits**: Discrete-variable quantum systems (e.g., superconducting circuits).  
    - **Angle Embedding**: Encodes classical noise into quantum states via rotation gates.  
    - **Entangling Layers**: Creates qubit entanglement for expressive quantum circuits.  
    """  

    def __init__(self, n_qubits=5, n_layers=3):  
        """Initialize a qubit-based quantum circuit.  

        Args:  
            n_qubits (int): Number of qubits (output_dim = n_qubits).  
            n_layers (int): Depth of the parameterized quantum circuit.  
        """  
        self.n_qubits = n_qubits  
        self.n_layers = n_layers  

        # PennyLane device (default qubit simulator with TensorFlow interface)  
        self.dev = qml.device("default.qubit", wires=n_qubits)  

        # Quantum circuit template  
        def quantum_circuit(z, params):  
            """Parametrized quantum circuit.  
            1. Encodes latent vector `z` into qubit rotations.  
            2. Applies entangling layers with trainable parameters.  
            3. Measures expectation values.  
            """  
            # Reshape params into layers  
            params = tf.reshape(params, (n_layers, n_qubits))  

            # Angle embedding of latent vector `z`  
            qml.AngleEmbedding(z, wires=range(n_qubits), rotation='Y')  

            # Entangling layers with trainable rotations  
            for layer in range(n_layers):  
                # Y-rotations on all qubits  
                for wire in range(n_qubits):  
                    qml.RY(params[layer, wire], wires=wire)  
                # Entangle qubits in a circular pattern  
                for wire in range(n_qubits):  
                    qml.CZ(wires=[wire, (wire+1) % n_qubits])  

            # Measure expectation values (Z on all qubits)  
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]  

        # Create QNode (quantum function + device)  
        self.qnode = qml.QNode(quantum_circuit, self.dev, interface="tf")  

        # Trainable parameters (shape: [n_layers * n_qubits])  
        self.params = tf.Variable(  
            tf.random.uniform(shape=[n_layers * n_qubits], minval=0, maxval=2*np.pi),  
            name="quantum_weights"  
        )  

    def generate(self, z):  
        """Generate samples via quantum circuit execution.  

        Args:  
            z (tensor): Latent noise vector (shape [batch_size, latent_dim]).  

        Returns:  
            samples (tensor): Generated data (expectation values, shape [batch_size, n_qubits]).  
        """  
        # Convert latent vector to angles (normalize to [-pi, pi])  
        z_scaled = tf.math.scalar_mul(np.pi, z)  
        # Execute quantum circuit for all batch elements  
        return tf.stack([self.qnode(z_batch, self.params) for z_batch in z_scaled])  