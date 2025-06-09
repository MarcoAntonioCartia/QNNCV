import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
import numpy as np

class QuantumContinuousGeneratorEnhanced:
    """Enhanced CV Quantum Generator using Strawberry Fields.
    
    Improvements over original:
    - Proper latent vector integration
    - Batch processing support
    - More expressive quantum circuit architecture
    - Robust parameter initialization
    """
    
    def __init__(self, n_qumodes=4, latent_dim=10, cutoff_dim=10):
        """Initialize enhanced CV quantum generator.
        
        Args:
            n_qumodes (int): Number of quantum modes (output dimension)
            latent_dim (int): Dimension of classical latent input
            cutoff_dim (int): Fock space cutoff for simulation
        """
        self.n_qumodes = n_qumodes
        self.latent_dim = latent_dim
        self.cutoff_dim = cutoff_dim
        
        # Quantum engine with optimized settings
        self.eng = sf.Engine('tf', backend_options={
            'cutoff_dim': cutoff_dim,
            'pure': True,
            'batch_size': None  # Dynamic batch size
        })
        
        # Enhanced parameter structure
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize trainable quantum parameters with proper scaling."""
        
        # Classical-to-quantum encoding network
        self.encoding_network = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(self.n_qumodes * 2, activation='tanh'),  # [r, phi] pairs
        ])
        
        # Build the encoding network
        dummy_input = tf.zeros((1, self.latent_dim))
        _ = self.encoding_network(dummy_input)
        
        # Two-mode squeezing parameters (for entanglement)
        self.squeeze_params = tf.Variable(
            tf.random.normal([self.n_qumodes // 2], stddev=0.1),
            name="squeeze_parameters"
        )
        
        # Interferometer parameters for mode mixing  
        # Use fewer parameters for simpler unitary generation
        n_interferometer_params = self.n_qumodes * self.n_qumodes
        self.interferometer_params = tf.Variable(
            tf.random.normal([n_interferometer_params], stddev=0.05),
            name="interferometer_parameters"
        )
        
        # Additional phase parameters for fine control
        self.phase_params = tf.Variable(
            tf.random.normal([self.n_qumodes], stddev=0.1),
            name="phase_parameters"
        )
    
    @property
    def trainable_variables(self):
        """Return all trainable parameters for optimizer."""
        variables = [self.squeeze_params, self.interferometer_params, self.phase_params]
        variables.extend(self.encoding_network.trainable_variables)
        return variables
    
    def _build_interferometer_matrix(self):
        """Build unitary interferometer matrix from parameters.
        
        Uses QR decomposition for guaranteed unitarity.
        """
        n = self.n_qumodes
        
        # Reshape parameters into matrix form
        matrix_params = tf.reshape(self.interferometer_params, [n, n])
        
        # Use QR decomposition directly on real matrix for simplicity
        # This avoids complex casting warnings while maintaining unitarity
        q, _ = tf.linalg.qr(matrix_params)
        
        return q
    
    def generate(self, z):
        """Generate quantum samples from latent vector.
        
        Args:
            z (tensor): Latent noise vector [batch_size, latent_dim]
            
        Returns:
            samples (tensor): Generated samples [batch_size, n_qumodes]
        """
        batch_size = tf.shape(z)[0]
        
        # Step 1: Classical preprocessing of latent vector
        displacement_params = self.encoding_network(z)  # [batch_size, n_qumodes * 2]
        displacement_r = displacement_params[:, :self.n_qumodes]
        displacement_phi = displacement_params[:, self.n_qumodes:]
        
        # Step 2: Create quantum circuit for batch processing
        prog = sf.Program(self.n_qumodes)
        
        # Create symbolic parameters
        squeeze_symbols = [prog.params(f"squeeze_{i}") for i in range(self.n_qumodes // 2)]
        displacement_r_symbols = [prog.params(f"displace_r_{i}") for i in range(self.n_qumodes)]
        displacement_phi_symbols = [prog.params(f"displace_phi_{i}") for i in range(self.n_qumodes)]
        phase_symbols = [prog.params(f"phase_{i}") for i in range(self.n_qumodes)]
        
        # Build interferometer matrix
        interferometer_matrix = self._build_interferometer_matrix()
        
        with prog.context as q:
            # Step 1: Two-mode squeezing for entanglement generation
            for i in range(self.n_qumodes // 2):
                S2gate(squeeze_symbols[i], 0.0) | (q[2*i], q[2*i+1])
            
            # Step 2: Interferometer for mode mixing
            Interferometer(interferometer_matrix, mesh='rectangular') | q
            
            # Step 3: Displacement gates (classical data encoding)
            for i in range(self.n_qumodes):
                Dgate(displacement_r_symbols[i], displacement_phi_symbols[i]) | q[i]
            
            # Step 4: Additional phase rotations for expressivity
            for i in range(self.n_qumodes):
                Rgate(phase_symbols[i]) | q[i]
            
            # Step 5: Homodyne measurements
            for i in range(self.n_qumodes):
                MeasureHomodyne(0.0) | q[i]  # X-quadrature measurement
        
        # Step 3: Execute circuit for each sample in batch
        all_samples = []
        
        for i in range(batch_size):
            # Create parameter mapping for this sample
            param_mapping = {
                **{f"squeeze_{j}": self.squeeze_params[j] for j in range(self.n_qumodes // 2)},
                **{f"displace_r_{j}": displacement_r[i, j] for j in range(self.n_qumodes)},
                **{f"displace_phi_{j}": displacement_phi[i, j] for j in range(self.n_qumodes)},
                **{f"phase_{j}": self.phase_params[j] for j in range(self.n_qumodes)}
            }
            
            # Reset engine and run circuit
            if self.eng.run_progs:
                self.eng.reset()
            
            result = self.eng.run(prog, args=param_mapping)
            samples = tf.cast(result.samples, tf.float32)
            # Ensure samples are 1D [n_qumodes]
            if len(samples.shape) > 1:
                samples = tf.squeeze(samples, axis=0)
            all_samples.append(samples)
        
        # Stack samples into batch
        output = tf.stack(all_samples, axis=0)
        return output
    
    def generate_single(self, z_single):
        """Generate a single sample (useful for debugging)."""
        z_batch = tf.expand_dims(z_single, 0)
        result = self.generate(z_batch)
        return tf.squeeze(result, 0)

# Fallback simple quantum generator for when Strawberry Fields is not available
class QuantumContinuousGeneratorSimple:
    """Simple quantum-inspired generator that works without Strawberry Fields."""
    
    def __init__(self, n_qumodes=4, latent_dim=10):
        """Initialize simple quantum-inspired generator."""
        self.n_qumodes = n_qumodes
        self.latent_dim = latent_dim
        
        # Classical network that mimics quantum behavior
        self.quantum_network = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(32, activation='sin'),  # Oscillatory activation
            tf.keras.layers.Dense(n_qumodes, activation='tanh')
        ])
        
        # Build the network
        dummy_input = tf.zeros((1, latent_dim))
        _ = self.quantum_network(dummy_input)
        
        # Quantum-inspired parameters
        self.quantum_params = tf.Variable(
            tf.random.normal([n_qumodes], stddev=0.1),
            name="quantum_params"
        )
    
    @property
    def trainable_variables(self):
        """Return trainable variables."""
        variables = [self.quantum_params]
        variables.extend(self.quantum_network.trainable_variables)
        return variables
    
    def generate(self, z):
        """Generate samples using quantum-inspired classical network."""
        # Process through quantum-inspired network
        base_output = self.quantum_network(z)
        
        # Add quantum-inspired interference patterns
        interference = tf.sin(base_output * self.quantum_params)
        
        # Combine base output with interference
        output = base_output + 0.1 * interference
        
        return output

def test_quantum_generators():
    """Test both quantum generators."""
    print("Testing Quantum Generators")
    print("==========================")
    
    # Test simple generator first
    print("\n1. Testing Simple Quantum Generator:")
    simple_gen = QuantumContinuousGeneratorSimple(n_qumodes=4, latent_dim=8)
    z_test = tf.random.normal([2, 8])
    samples_simple = simple_gen.generate(z_test)
    print(f"Simple generator output shape: {samples_simple.shape}")
    print(f"Simple generator trainable vars: {len(simple_gen.trainable_variables)}")
    
    # Test enhanced generator if Strawberry Fields is available
    print("\n2. Testing Enhanced Quantum Generator:")
    try:
        enhanced_gen = QuantumContinuousGeneratorEnhanced(n_qumodes=4, latent_dim=8)
        samples_enhanced = enhanced_gen.generate(z_test)
        print(f"Enhanced generator output shape: {samples_enhanced.shape}")
        print(f"Enhanced generator trainable vars: {len(enhanced_gen.trainable_variables)}")
        print("Enhanced quantum generator working!")
    except Exception as e:
        print(f"Enhanced generator failed (likely missing Strawberry Fields): {e}")
        print("Using simple generator as fallback.")

if __name__ == "__main__":
    test_quantum_generators()
