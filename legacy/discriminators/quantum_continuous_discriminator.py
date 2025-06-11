import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
import numpy as np

class QuantumContinuousDiscriminator:
    """Enhanced CV Quantum Discriminator using Strawberry Fields.
    
    Sophisticated quantum circuit architecture featuring:
    - Multi-layer quantum processing
    - Input encoding via displacement gates
    - Squeezing for quantum nonlinearity
    - Interferometer for mode coupling
    - Adaptive homodyne measurements
    - Classical post-processing network
    """
    
    def __init__(self, n_qumodes=4, input_dim=4, cutoff_dim=10):
        """Initialize enhanced CV quantum discriminator.
        
        Args:
            n_qumodes (int): Number of quantum modes
            input_dim (int): Dimension of input data
            cutoff_dim (int): Fock space cutoff for simulation
        """
        self.n_qumodes = n_qumodes
        self.input_dim = input_dim
        self.cutoff_dim = cutoff_dim
        
        # Quantum engine with optimized settings
        self.eng = sf.Engine('tf', backend_options={
            'cutoff_dim': cutoff_dim,
            'pure': True,
            'batch_size': None  # Dynamic batch size
        })
        
        # Initialize quantum parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize trainable quantum and classical parameters."""
        
        # Classical preprocessing network (maps input to quantum parameters)
        self.input_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='tanh', name='encoder_dense1'),
            tf.keras.layers.Dense(self.n_qumodes * 2, activation='tanh', name='encoder_output'),  # [r, phi] for displacement
        ])
        
        # Build the encoder network
        dummy_input = tf.zeros((1, self.input_dim))
        _ = self.input_encoder(dummy_input)
        
        # Trainable squeezing parameters for quantum nonlinearity
        self.squeeze_params = tf.Variable(
            tf.random.normal([self.n_qumodes], stddev=0.1),
            name="disc_squeeze_params"
        )
        
        # Note: Using fixed DFT interferometer instead of trainable parameters
        # to avoid symbolic tensor issues with Strawberry Fields
        
        # Learnable measurement angles for adaptive measurements
        self.measurement_angles = tf.Variable(
            tf.random.uniform([self.n_qumodes], maxval=2*np.pi),
            name="measurement_angles"
        )
        
        # Classical post-processing network
        self.output_network = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='relu', name='output_dense1'),
            tf.keras.layers.Dense(4, activation='relu', name='output_dense2'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='output_final'),  # Binary classification
        ])
        
        # Build the output network
        dummy_quantum_input = tf.zeros((1, self.n_qumodes))
        _ = self.output_network(dummy_quantum_input)
    
    @property
    def trainable_variables(self):
        """Return all trainable parameters for optimization."""
        variables = [
            self.squeeze_params, 
            self.measurement_angles
        ]
        variables.extend(self.input_encoder.trainable_variables)
        variables.extend(self.output_network.trainable_variables)
        return variables
    

    
    def _create_circuit_program(self):
        """Create the sophisticated quantum discriminator circuit."""
        prog = sf.Program(self.n_qumodes)
        
        # Create symbolic parameters for the circuit
        squeeze_symbols = [prog.params(f"squeeze_{i}") for i in range(self.n_qumodes)]
        displacement_r_symbols = [prog.params(f"disp_r_{i}") for i in range(self.n_qumodes)]
        displacement_phi_symbols = [prog.params(f"disp_phi_{i}") for i in range(self.n_qumodes)]
        measurement_symbols = [prog.params(f"measure_{i}") for i in range(self.n_qumodes)]
        
        with prog.context as q:
            # Step 1: Input encoding via displacement gates
            for i in range(self.n_qumodes):
                Dgate(displacement_r_symbols[i], displacement_phi_symbols[i]) | q[i]
            
            # Step 2: First squeezing layer for quantum nonlinearity
            for i in range(self.n_qumodes):
                Sgate(squeeze_symbols[i], 0.0) | q[i]
            
            # Step 3: Simple fixed interferometer (Fourier transform)
            # Use a predefined unitary matrix to avoid symbolic tensor issues
            n = self.n_qumodes
            # Create a simple DFT-like unitary matrix
            angles = [2 * np.pi * i * j / n for i in range(n) for j in range(n)]
            dft_matrix = np.array([[np.exp(1j * angles[i*n + j]) / np.sqrt(n) 
                                 for j in range(n)] for i in range(n)])
            
            Interferometer(dft_matrix, mesh='rectangular') | q
            
            # Step 4: Second squeezing layer with different phase
            for i in range(self.n_qumodes):
                Sgate(squeeze_symbols[i] * 0.5, np.pi/4) | q[i]  # Different phase for complexity
            
            # Step 5: Adaptive homodyne measurements
            for i in range(self.n_qumodes):
                MeasureHomodyne(measurement_symbols[i]) | q[i]
        
        return prog
    
    def discriminate(self, x):
        """Discriminate between real and fake samples using quantum circuit.
        
        Args:
            x (tensor): Input samples [batch_size, input_dim]
            
        Returns:
            probabilities (tensor): Probability of being real [batch_size, 1]
        """
        batch_size = tf.shape(x)[0]
        
        # Step 1: Classical preprocessing to quantum parameters
        encoded_params = self.input_encoder(x)  # [batch_size, n_qumodes * 2]
        displacement_r = encoded_params[:, :self.n_qumodes]
        displacement_phi = encoded_params[:, self.n_qumodes:]
        
        # Normalize displacement parameters to prevent circuit instability
        displacement_r = tf.tanh(displacement_r) * 2.0  # Bound to [-2, 2]
        displacement_phi = displacement_phi * np.pi  # Scale to [-π, π]
        
        # Step 2: Create quantum circuit program
        prog = self._create_circuit_program()
        
        # Step 3: Execute quantum circuit for each sample in batch
        all_measurements = []
        
        for i in range(batch_size):
            # Create parameter mapping for this sample
            param_mapping = {
                **{f"squeeze_{j}": self.squeeze_params[j] for j in range(self.n_qumodes)},
                **{f"disp_r_{j}": displacement_r[i, j] for j in range(self.n_qumodes)},
                **{f"disp_phi_{j}": displacement_phi[i, j] for j in range(self.n_qumodes)},
                **{f"measure_{j}": self.measurement_angles[j] for j in range(self.n_qumodes)}
            }
            
            # Reset quantum engine and execute circuit
            if self.eng.run_progs:
                self.eng.reset()
            
            result = self.eng.run(prog, args=param_mapping)
            measurements = tf.cast(result.samples, tf.float32)
            # Ensure measurements are 1D [n_qumodes]
            if len(measurements.shape) > 1:
                measurements = tf.squeeze(measurements, axis=0)
            all_measurements.append(measurements)
        
        # Step 4: Stack quantum measurements from all samples
        quantum_features = tf.stack(all_measurements, axis=0)  # [batch_size, n_qumodes]
        
        # Step 5: Classical post-processing for final classification
        probabilities = self.output_network(quantum_features)
        
        return probabilities
    
    def discriminate_single(self, x_single):
        """Discriminate a single sample (useful for debugging and testing)."""
        x_batch = tf.expand_dims(x_single, 0)
        result = self.discriminate(x_batch)
        return tf.squeeze(result, 0)
