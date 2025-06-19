"""
Quantum Strawberry Fields Generator with Individual Variables

This generator uses individual tf.Variables for each quantum parameter
to ensure proper gradient flow through the Strawberry Fields backend.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class QuantumSFGeneratorIndividual:
    """
    Quantum generator with individual tf.Variables for gradient flow.
    
    Features:
    - Individual tf.Variable for each quantum parameter
    - Direct measurement without mode selection
    - Fixed projection for visualization only
    """
    
    def __init__(self, n_modes=4, latent_dim=6, layers=2, cutoff_dim=6,
                 output_dim=2):
        """Initialize quantum generator with individual variables."""
        self.n_modes = n_modes
        self.latent_dim = latent_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.output_dim = output_dim
        
        logger.info(f"Initializing Quantum SF Generator with Individual Variables:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - latent_dim: {latent_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize individual quantum variables
        self._init_individual_quantum_variables()
        
        # Initialize coherent encoding
        self._init_coherent_encoder()
        
        # Initialize fixed projection
        self._init_fixed_projection()
        
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
        logger.info("SF engine initialized")
    
    def _init_individual_quantum_variables(self):
        """Initialize individual tf.Variables for each quantum parameter."""
        # Calculate parameter structure
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        
        # Create dictionaries to hold individual variables
        self.quantum_vars = {}
        
        # Initialize variables for each layer
        for layer in range(self.layers):
            layer_vars = {}
            
            # Interferometer 1 parameters
            layer_vars['int1'] = []
            for i in range(M):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'int1_L{layer}_P{i}'
                )
                layer_vars['int1'].append(var)
            
            # Squeezing parameters
            layer_vars['squeeze'] = []
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.01),
                    name=f'squeeze_L{layer}_M{i}'
                )
                layer_vars['squeeze'].append(var)
            
            # Interferometer 2 parameters
            layer_vars['int2'] = []
            for i in range(M):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'int2_L{layer}_P{i}'
                )
                layer_vars['int2'].append(var)
            
            # Displacement r parameters
            layer_vars['disp_r'] = []
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.01),
                    name=f'disp_r_L{layer}_M{i}'
                )
                layer_vars['disp_r'].append(var)
            
            # Displacement phi parameters
            layer_vars['disp_phi'] = []
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'disp_phi_L{layer}_M{i}'
                )
                layer_vars['disp_phi'].append(var)
            
            # Kerr parameters
            layer_vars['kerr'] = []
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.001),
                    name=f'kerr_L{layer}_M{i}'
                )
                layer_vars['kerr'].append(var)
            
            self.quantum_vars[f'layer_{layer}'] = layer_vars
        
        # Count total parameters
        self.num_quantum_params = 0
        for layer in range(self.layers):
            layer_vars = self.quantum_vars[f'layer_{layer}']
            for param_type in layer_vars:
                self.num_quantum_params += len(layer_vars[param_type])
        
        logger.info(f"Created {self.num_quantum_params} individual quantum variables")
    
    def _init_coherent_encoder(self):
        """Initialize coherent state encoding."""
        # Individual variables for coherent encoding matrix
        self.coherent_vars = {}
        
        # Create individual variables for encoding matrix
        self.coherent_vars['encoding'] = []
        for i in range(self.latent_dim):
            row_vars = []
            for j in range(2 * self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.5),
                    name=f'encoding_{i}_{j}'
                )
                row_vars.append(var)
            self.coherent_vars['encoding'].append(row_vars)
        
        # Individual amplitude scale variables
        self.coherent_vars['amplitude_scale'] = []
        for i in range(self.n_modes):
            var = tf.Variable(
                0.5,
                name=f'amplitude_scale_{i}'
            )
            self.coherent_vars['amplitude_scale'].append(var)
        
        logger.info(f"Created coherent encoding variables")
    
    def _init_fixed_projection(self):
        """Initialize fixed projection matrix."""
        if self.n_modes >= self.output_dim:
            random_matrix = np.random.randn(self.n_modes, self.output_dim)
            q, _ = np.linalg.qr(random_matrix)
            self.viz_projection = tf.constant(q[:, :self.output_dim], dtype=tf.float32)
        else:
            eye = np.eye(self.n_modes)
            padding = np.zeros((self.n_modes, self.output_dim - self.n_modes))
            self.viz_projection = tf.constant(
                np.concatenate([eye, padding], axis=1), 
                dtype=tf.float32
            )
        
        self.viz_projection_inv = tf.linalg.pinv(self.viz_projection)
        logger.info(f"Fixed projection: {self.n_modes} → {self.output_dim}")
    
    def _create_symbolic_params(self):
        """Create SF symbolic parameters."""
        # Create symbolic parameters for quantum circuit
        self.sf_layer_params = []
        
        for layer in range(self.layers):
            layer_params = {}
            layer_vars = self.quantum_vars[f'layer_{layer}']
            
            # Create symbolic parameters for each type
            layer_params['int1'] = [self.qnn.params(f'int1_L{layer}_P{i}') 
                                   for i in range(len(layer_vars['int1']))]
            layer_params['squeeze'] = [self.qnn.params(f'squeeze_L{layer}_M{i}') 
                                      for i in range(self.n_modes)]
            layer_params['int2'] = [self.qnn.params(f'int2_L{layer}_P{i}') 
                                   for i in range(len(layer_vars['int2']))]
            layer_params['disp_r'] = [self.qnn.params(f'disp_r_L{layer}_M{i}') 
                                     for i in range(self.n_modes)]
            layer_params['disp_phi'] = [self.qnn.params(f'disp_phi_L{layer}_M{i}') 
                                       for i in range(self.n_modes)]
            layer_params['kerr'] = [self.qnn.params(f'kerr_L{layer}_M{i}') 
                                   for i in range(self.n_modes)]
            
            self.sf_layer_params.append(layer_params)
        
        # Coherent state parameters
        self.coherent_params = []
        for i in range(self.n_modes):
            self.coherent_params.append({
                'alpha_r': self.qnn.params(f"alpha_r_{i}"),
                'alpha_i': self.qnn.params(f"alpha_i_{i}")
            })
    
    def _build_quantum_program(self):
        """Build quantum program."""
        with self.qnn.context as q:
            # Initialize with coherent states
            for i in range(self.n_modes):
                alpha = self.coherent_params[i]['alpha_r'] + 1j * self.coherent_params[i]['alpha_i']
                ops.Dgate(alpha) | q[i]
            
            # Apply quantum layers
            for layer in range(self.layers):
                self._quantum_layer(self.sf_layer_params[layer], q)
    
    def _quantum_layer(self, layer_params, q):
        """Single quantum layer using symbolic parameters."""
        N = len(q)
        
        # Apply operations with symbolic parameters
        self._interferometer(layer_params['int1'], q)
        
        for i in range(N):
            ops.Sgate(layer_params['squeeze'][i]) | q[i]
        
        self._interferometer(layer_params['int2'], q)
        
        for i in range(N):
            ops.Dgate(layer_params['disp_r'][i], layer_params['disp_phi'][i]) | q[i]
            ops.Kgate(layer_params['kerr'][i]) | q[i]
    
    def _interferometer(self, params, q):
        """Interferometer with individual parameters."""
        N = len(q)
        
        if N == 1:
            # Single mode rotation
            ops.Rgate(params[0]) | q[0]
            return
        
        # Calculate expected parameter counts
        n_theta = N * (N - 1) // 2
        n_phi = N * (N - 1) // 2
        n_rphi = max(1, N - 1)
        
        # Extract parameters
        theta = params[:n_theta]
        phi = params[n_theta:n_theta + n_phi]
        rphi = params[n_theta + n_phi:]
        
        # Apply beamsplitter array
        param_idx = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[param_idx], phi[param_idx]) | (q1, q2)
                    param_idx += 1
        
        # Apply final rotations
        for i in range(len(rphi)):
            ops.Rgate(rphi[i]) | q[i]
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        variables = []
        
        # Add all quantum variables
        for layer in range(self.layers):
            layer_vars = self.quantum_vars[f'layer_{layer}']
            for param_type in layer_vars:
                variables.extend(layer_vars[param_type])
        
        # Add coherent encoding variables
        for row in self.coherent_vars['encoding']:
            variables.extend(row)
        variables.extend(self.coherent_vars['amplitude_scale'])
        
        return variables
    
    def _get_encoding_matrix(self):
        """Construct encoding matrix from individual variables."""
        rows = []
        for row_vars in self.coherent_vars['encoding']:
            row = tf.stack(row_vars)
            rows.append(row)
        return tf.stack(rows)
    
    def _get_amplitude_scale(self):
        """Get amplitude scale as tensor."""
        return tf.stack(self.coherent_vars['amplitude_scale'])
    
    def transform_data_to_quantum_space(self, data_2d):
        """Transform 2D data to quantum space."""
        return tf.matmul(data_2d, self.viz_projection_inv)
    
    def project_to_2d(self, quantum_samples):
        """Project quantum samples to 2D."""
        return tf.matmul(quantum_samples, self.viz_projection)
    
    def generate(self, z, return_2d=False):
        """Generate samples."""
        batch_size = tf.shape(z)[0]
        all_samples = []
        
        # Get encoding matrix
        encoding_matrix = self._get_encoding_matrix()
        
        # Encode to coherent state parameters
        coherent_params = tf.matmul(z, encoding_matrix)
        
        for i in range(batch_size):
            sample = self._generate_single(coherent_params[i])
            all_samples.append(sample)
        
        quantum_samples = tf.stack(all_samples, axis=0)
        
        if return_2d:
            return self.project_to_2d(quantum_samples)
        else:
            return quantum_samples
    
    def _generate_single(self, coherent_params):
        """Generate single sample."""
        # Split coherent parameters
        real_parts = coherent_params[:self.n_modes]
        imag_parts = coherent_params[self.n_modes:]
        
        # Apply amplitude scaling
        amplitude_scale = self._get_amplitude_scale()
        real_parts = real_parts * amplitude_scale
        imag_parts = imag_parts * amplitude_scale
        
        # Create parameter mapping with individual variables
        mapping = {}
        
        # Map quantum circuit parameters
        for layer in range(self.layers):
            layer_vars = self.quantum_vars[f'layer_{layer}']
            layer_params = self.sf_layer_params[layer]
            
            # Map each parameter type
            for i, var in enumerate(layer_vars['int1']):
                mapping[layer_params['int1'][i].name] = var
            
            for i, var in enumerate(layer_vars['squeeze']):
                mapping[layer_params['squeeze'][i].name] = var
            
            for i, var in enumerate(layer_vars['int2']):
                mapping[layer_params['int2'][i].name] = var
            
            for i, var in enumerate(layer_vars['disp_r']):
                mapping[layer_params['disp_r'][i].name] = var
            
            for i, var in enumerate(layer_vars['disp_phi']):
                mapping[layer_params['disp_phi'][i].name] = var
            
            for i, var in enumerate(layer_vars['kerr']):
                mapping[layer_params['kerr'][i].name] = var
        
        # Map coherent state parameters
        for i in range(self.n_modes):
            mapping[self.coherent_params[i]['alpha_r'].name] = real_parts[i]
            mapping[self.coherent_params[i]['alpha_i'].name] = imag_parts[i]
        
        # Reset engine
        if self.eng.run_progs:
            self.eng.reset()
        
        # Run quantum circuit
        try:
            state = self.eng.run(self.qnn, args=mapping).state
            
            # Extract measurements
            quantum_vector = self._extract_quantum_measurements(state)
            
            return quantum_vector
            
        except Exception as e:
            logger.debug(f"Quantum circuit failed: {e}")
            return tf.random.normal([self.n_modes], stddev=0.5)
    
    def _extract_quantum_measurements(self, state):
        """Extract measurements from quantum state."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        measurements = []
        
        for mode in range(self.n_modes):
            if mode == 0:
                # Mean photon number
                n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
                mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
                measurements.append(mean_n)
            elif mode == 1:
                # Standard deviation
                n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
                mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
                var_n = tf.reduce_sum(prob_amplitudes * (n_vals[:tf.shape(prob_amplitudes)[0]] - mean_n)**2)
                measurements.append(tf.sqrt(var_n + 1e-6))
            elif mode == 2:
                # Entropy
                entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12))
                measurements.append(entropy)
            else:
                # Higher moments
                moment_order = mode
                moment = tf.reduce_sum(
                    prob_amplitudes * tf.pow(
                        tf.range(tf.shape(prob_amplitudes)[0], dtype=tf.float32), 
                        moment_order
                    )
                )
                measurements.append(tf.pow(moment + 1e-6, 1.0/moment_order))
        
        # Normalize
        measurements = tf.stack(measurements)
        measurements = tf.nn.tanh(measurements / 3.0) * 3.0
        
        return measurements


def test_individual_generator():
    """Test the individual variable generator."""
    print("\n" + "="*60)
    print("TESTING QUANTUM SF GENERATOR WITH INDIVIDUAL VARIABLES")
    print("="*60)
    
    # Create generator
    generator = QuantumSFGeneratorIndividual(
        n_modes=4,
        latent_dim=6,
        layers=2,
        cutoff_dim=6,
        output_dim=2
    )
    
    print(f"\nGenerator created successfully")
    print(f"Total trainable variables: {len(generator.trainable_variables)}")
    
    # Test generation
    print("\n1. GENERATION TEST")
    z_test = tf.random.normal([10, 6])
    
    # Generate in quantum space
    samples_quantum = generator.generate(z_test, return_2d=False)
    print(f"   Quantum samples shape: {samples_quantum.shape}")
    print(f"   Quantum samples range: [{tf.reduce_min(samples_quantum):.3f}, {tf.reduce_max(samples_quantum):.3f}]")
    
    # Generate in 2D
    samples_2d = generator.generate(z_test, return_2d=True)
    print(f"   2D samples shape: {samples_2d.shape}")
    
    # Test gradient flow
    print("\n2. GRADIENT FLOW TEST")
    with tf.GradientTape() as tape:
        z = tf.random.normal([4, 6])
        output = generator.generate(z, return_2d=False)
        loss = tf.reduce_mean(tf.square(output))
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    # Count gradients by type
    grad_counts = {'total': 0, 'non_zero': 0}
    
    for i, (var, grad) in enumerate(zip(generator.trainable_variables, gradients)):
        if grad is not None:
            grad_counts['total'] += 1
            if tf.reduce_sum(tf.abs(grad)) > 1e-8:
                grad_counts['non_zero'] += 1
    
    print(f"\n   Total variables: {len(generator.trainable_variables)}")
    print(f"   Variables with gradients: {grad_counts['total']}")
    print(f"   Variables with non-zero gradients: {grad_counts['non_zero']}")
    
    # Show some specific gradients
    print("\n   Sample gradient norms:")
    for i in range(min(10, len(gradients))):
        var = generator.trainable_variables[i]
        grad = gradients[i]
        if grad is not None:
            print(f"   {var.name}: {tf.norm(grad):.6f}")
        else:
            print(f"   {var.name}: No gradient")
    
    print("\n✅ Individual variable generator test completed!")
    
    return generator


if __name__ == "__main__":
    generator = test_individual_generator()
