import strawberryfields as sf  # CV quantum computing library
from strawberryfields.ops import *  # Quantum operations (gates)
import tensorflow as tf  # Integration with TensorFlow for gradients

class QuantumContinuousGenerator:
    """CV Quantum Generator using Strawberry Fields (qumodes).
    
    Key Concepts:
    - Qumodes: Continuous-variable quantum systems (e.g., photonic states).
    - Gaussian Operations: Displacement (Dgate) and Squeezing (Squeezed) to manipulate qumodes.
    - Measurement: Homodyne detection to collapse quantum states into classical data.
    """
    
    def __init__(self, n_qumodes=30, cutoff_dim=5):
        """Initialize a CV quantum circuit.
        
        Args:
            n_qumodes (int): Number of qumodes (matches output dimension).
            cutoff_dim (int): Hilbert space cutoff for simulation (approximates infinite dimensions).
        """
        self.n_qumodes = n_qumodes
        self.cutoff_dim = cutoff_dim
        
        # Quantum simulator backend (TensorFlow for automatic differentiation)
        self.eng = sf.Engine(
            backend="tf", 
            backend_options={"cutoff_dim": cutoff_dim}
        )
        
        # Quantum circuit parameters (displacement magnitudes)
        # These are trainable variables optimized during training.
        self.params = tf.Variable(
            tf.random.normal(shape=[n_qumodes]),  # Initialized randomly
            name="displacement_params"
        )

    def generate(self, z):
        """Generate samples by running the CV quantum circuit.
        
        Args:
            z (tensor): Latent noise vector (classical input; unused here but part of GAN API).
        
        Returns:
            samples (tensor): Generated data (homodyne measurement results).
        """
        # Create a quantum program with `n_qumodes` modes
        prog = sf.Program(self.n_qumodes)
        
        # Build the quantum circuit
        with prog.context as q:  # 'q' is the register of qumodes
            # Apply quantum gates to each qumode
            for i in range(self.n_qumodes):
                # Displacement gate (amplitude set by trainable parameter)
                Dgate(self.params[i]) | q[i]  # Dgate(r, phi): r = displacement magnitude
                
                # Squeezing gate (fixed squeezing for non-Gaussian effects)
                Squeezed(0.1) | q[i]  # Squeezed(r, phi): r = squeezing magnitude
            
            # Measure all qumodes via homodyne detection (X quadrature)
            MeasureHomodyne(0.5) | q  # 0.5*pi/2 phase for X measurement
        
        # Run the quantum circuit
        result = self.eng.run(prog)
        
        # Return measurement results (shape [batch_size, n_qumodes])
        return result.samples  # These are the "generated" samples