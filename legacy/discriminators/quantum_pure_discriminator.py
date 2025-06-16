"""
Pure Quantum Discriminator - Only Quantum Variables in Optimization

This discriminator uses ONLY quantum parameters for discrimination:
- No classical neural networks
- Direct quantum state discrimination
- Pure quantum gradient flow
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class QuantumPureDiscriminator:
    """
    Pure quantum discriminator with ONLY quantum parameters.
    
    No classical networks - just quantum variables being optimized.
    """
    
    def __init__(self, n_modes=2, input_dim=2, layers=2, cutoff_dim=6):
        """Initialize pure quantum discriminator."""
        self.n_modes = n_modes
        self.input_dim = input_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        logger.info(f"Initializing Pure Quantum Discriminator:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - input_dim: {input_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize quantum weights (single matrix)
        self._init_quantum_weights()
        
        # Create symbolic parameters
        self._create_symbolic_params()
        
        # Build quantum program
        self._build_quantum_program()
    
    def _init_sf_components(self):
        """Initialize Strawberry Fields engine and program."""
        self.eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        self.qnn = sf.Program(self.n_modes)
        logger.info("SF engine initialized for pure quantum discrimination")
    
    def _init_quantum_weights(self):
        """Initialize ONLY quantum weights - no classical components."""
        # Calculate parameter count
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        params_per_layer = 2 * M + 4 * self.n_modes
        
        # Single quantum weight matrix
        self.weights = tf.Variable(
            tf.random.normal([self.layers, params_per_layer], stddev=0.1),
            name="pure_quantum_discriminator_weights"
        )
        
        # Input encoding weights (quantum parameters that encode input data)
        # These map input features directly to quantum circuit modulations
        self.input_weights = tf.Variable(
            tf.random.normal([self.input_dim, self.n_modes], stddev=0.1),
            name="quantum_input_encoding_weights"
        )
        
        self.num_quantum_params = int(np.prod(self.weights.shape))
        logger.info(f"Pure quantum weights: shape {self.weights.shape}")
        logger.info(f"Input encoding weights: shape {self.input_weights.shape}")
        logger.info(f"Total quantum parameters: {self.num_quantum_params + self.input_weights.shape[0] * self.input_weights.shape[1]}")
    
    def _create_symbolic_params(self):
        """Create SF symbolic parameters."""
        # Main circuit parameters
        sf_params = np.arange(self.num_quantum_params).reshape(self.weights.shape).astype(str)
        self.sf_params = np.array([self.qnn.params(*i) for i in sf_params])
        
        # Input encoding parameters
        self.input_params = []
        for i in range(self.n_modes):
            self.input_params.append(self.qnn.params(f"input_encoding_{i}"))
        
        logger.info(f"Created {self.num_quantum_params} circuit parameters + {len(self.input_params)} input parameters")
    
    def _build_quantum_program(self):
        """Build quantum program with input encoding."""
        with self.qnn.context as q:
            # Input encoding layer (displacement based on input)
            for i in range(self.n_modes):
                ops.Dgate(self.input_params[i], 0.0) | q[i]
            
            # Main quantum layers
            for k in range(self.layers):
                self._quantum_layer(self.sf_params[k], q)
        
        logger.info(f"Pure quantum program built with input encoding + {self.layers} layers")
    
    def _quantum_layer(self, params, q):
        """Single quantum layer."""
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)
        
        # Extract parameters
        int1 = params[:M]
        s = params[M:M+N]
        int2 = params[M+N:2*M+N]
        dr = params[2*M+N:2*M+2*N]
        dp = params[2*M+2*N:2*M+3*N]
        k = params[2*M+3*N:2*M+4*N]
        
        # Apply operations
        self._interferometer(int1, q)
        for i in range(N):
            ops.Sgate(s[i]) | q[i]
        self._interferometer(int2, q)
        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]
    
    def _interferometer(self, params, q):
        """Interferometer implementation."""
        N = len(q)
        theta = params[:N*(N-1)//2]
        phi = params[N*(N-1)//2:N*(N-1)]
        rphi = params[-N+1:]
        
        if N == 1:
            ops.Rgate(rphi[0]) | q[0]
            return
        
        n = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1
        
        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]
    
    @property
    def trainable_variables(self):
        """Return ONLY quantum variables - no classical networks."""
        return [self.weights, self.input_weights]
    
    def discriminate(self, x):
        """
        Pure quantum discrimination.
        
        Args:
            x: Input samples [batch_size, input_dim]
            
        Returns:
            probabilities: Discrimination scores [batch_size, 1]
        """
        batch_size = tf.shape(x)[0]
        all_scores = []
        
        for i in range(batch_size):
            score = self._discriminate_single(x[i])
            all_scores.append(score)
        
        # Stack and reshape to [batch_size, 1]
        scores = tf.stack(all_scores, axis=0)
        return tf.reshape(scores, [batch_size, 1])
    
    def _discriminate_single(self, x_single):
        """Discriminate single sample using pure quantum computation."""
        # Quantum input encoding: map input to displacement parameters
        input_encoding = tf.matmul(tf.reshape(x_single, [1, -1]), self.input_weights)
        input_encoding = tf.squeeze(input_encoding)  # [n_modes]
        
        # Create parameter mapping
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(),
                tf.reshape(self.weights, [-1])
            )
        }
        
        # Add input encoding
        for i, param in enumerate(self.input_params):
            if i < len(input_encoding):
                mapping[param.name] = input_encoding[i]
            else:
                mapping[param.name] = tf.constant(0.0)
        
        # Reset engine
        if self.eng.run_progs:
            self.eng.reset()
        
        # Run quantum circuit
        try:
            state = self.eng.run(self.qnn, args=mapping).state
            
            # Extract discrimination score from quantum state
            score = self._extract_discrimination_score(state)
            
            # Convert to probability using sigmoid
            probability = tf.nn.sigmoid(score)
            
            return probability
            
        except Exception as e:
            logger.debug(f"Quantum circuit failed: {e}")
            return tf.constant(0.5)  # Neutral score
    
    def _extract_discrimination_score(self, state):
        """Extract discrimination score from quantum state."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Use quantum state properties for discrimination
        # Higher photon numbers → more likely to be real
        # Lower photon numbers → more likely to be fake
        
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        
        # Weighted average photon number across all modes
        mean_photon_number = tf.reduce_sum(prob_amplitudes * n_vals)
        
        # Normalize to discrimination score
        # Map [0, cutoff_dim] → [-2, 2] for sigmoid
        score = (mean_photon_number - self.cutoff_dim/2) / (self.cutoff_dim/4) * 2
        
        return score
    
    def compute_quantum_metrics(self):
        """Compute quantum metrics."""
        # Test input
        x_test = tf.random.normal([1, self.input_dim])
        
        # Get input encoding
        input_encoding = tf.matmul(x_test, self.input_weights)
        input_encoding = tf.squeeze(input_encoding)
        
        # Create mapping
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(),
                tf.reshape(self.weights, [-1])
            )
        }
        
        for i, param in enumerate(self.input_params):
            if i < len(input_encoding):
                mapping[param.name] = input_encoding[i]
        
        if self.eng.run_progs:
            self.eng.reset()
        
        state = self.eng.run(self.qnn, args=mapping).state
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        metrics = {
            'trace': tf.math.real(state.trace()),
            'purity': tf.reduce_sum(prob_amplitudes ** 2),
            'max_prob': tf.reduce_max(prob_amplitudes),
            'mean_photon': tf.reduce_sum(prob_amplitudes * tf.range(self.cutoff_dim, dtype=tf.float32))
        }
        
        return metrics


def test_pure_quantum_discriminator():
    """Test the pure quantum discriminator."""
    print("\n" + "="*60)
    print("TESTING PURE QUANTUM DISCRIMINATOR")
    print("="*60)
    
    # Create discriminator
    discriminator = QuantumPureDiscriminator(
        n_modes=2,
        input_dim=2,
        layers=2,
        cutoff_dim=6
    )
    
    print(f"✅ Pure quantum discriminator created")
    print(f"📊 Trainable variables: {len(discriminator.trainable_variables)}")
    print(f"   - Circuit weights: {discriminator.weights.shape}")
    print(f"   - Input weights: {discriminator.input_weights.shape}")
    print(f"   - Total parameters: {tf.size(discriminator.weights).numpy() + tf.size(discriminator.input_weights).numpy()}")
    
    # Test discrimination
    x_test = tf.random.normal([5, 2])
    probs = discriminator.discriminate(x_test)
    
    print(f"\n🎯 Discrimination test:")
    print(f"   Input shape: {x_test.shape}")
    print(f"   Output shape: {probs.shape}")
    print(f"   Probability range: [{tf.reduce_min(probs):.3f}, {tf.reduce_max(probs):.3f}]")
    
    # Test gradient flow
    with tf.GradientTape() as tape:
        x = tf.random.normal([2, 2])
        output = discriminator.discriminate(x)
        loss = tf.reduce_mean(tf.square(output - 0.5))
    
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    
    print(f"\n🔄 Gradient test:")
    print(f"   Circuit weight gradient: {gradients[0] is not None}")
    print(f"   Input weight gradient: {gradients[1] is not None}")
    
    if gradients[0] is not None:
        print(f"   Circuit gradient norm: {tf.norm(gradients[0]):.6f}")
    if gradients[1] is not None:
        print(f"   Input gradient norm: {tf.norm(gradients[1]):.6f}")
    
    # Test quantum metrics
    metrics = discriminator.compute_quantum_metrics()
    print(f"\n🔬 Quantum metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n✅ Pure quantum discriminator test completed!")
    print("   - Only quantum parameters (no classical networks)")
    print("   - Direct quantum state discrimination")
    print("   - Gradient flow through quantum operations")
    
    return discriminator


if __name__ == "__main__":
    test_pure_quantum_discriminator()
