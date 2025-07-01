"""
Input-Dependent Quantum Circuit - Killoran-Style Implementation

This implementation follows the Killoran paper approach where the latent vector
directly influences the initial quantum state preparation, creating input-dependent
quantum processing instead of static quantum states.

Key innovations:
1. Input-dependent initial state preparation (coherent states from latent vector)
2. Preserves SF Tutorial gradient flow pattern
3. Creates different quantum states for different inputs
4. Foundation for spatial mode assignment decoder
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


def interferometer(params, q):
    """SF Tutorial interferometer implementation (EXACT COPY for gradient flow)."""
    N = len(q)
    theta = params[: N * (N - 1) // 2]
    phi = params[N * (N - 1) // 2 : N * (N - 1)]
    rphi = params[-N + 1 :]

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


def layer(params, q):
    """SF Tutorial layer implementation (EXACT COPY for gradient flow)."""
    N = len(q)
    M = int(N * (N - 1)) + max(1, N - 1)

    int1 = params[:M]
    s = params[M : M + N]
    int2 = params[M + N : 2 * M + N]
    dr = params[2 * M + N : 2 * M + 2 * N]
    dp = params[2 * M + 2 * N : 2 * M + 3 * N]
    k = params[2 * M + 3 * N : 2 * M + 4 * N]

    # Standard SF Tutorial layer structure
    interferometer(int1, q)
    
    for i in range(N):
        ops.Sgate(s[i]) | q[i]

    interferometer(int2, q)

    for i in range(N):
        ops.Dgate(dr[i], dp[i]) | q[i]
        ops.Kgate(k[i]) | q[i]


def init_weights(modes, layers, active_sd=0.0001, passive_sd=0.1):
    """SF Tutorial weight initialization (EXACT COPY for gradient flow)."""
    M = int(modes * (modes - 1)) + max(1, modes - 1)

    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    dp_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)

    weights = tf.concat(
        [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights], axis=1
    )
    weights = tf.Variable(weights)

    return weights


class InputDependentQuantumCircuit:
    """
    Input-Dependent Quantum Circuit - Killoran-Style Implementation
    
    Key innovation: Uses latent vector to prepare different initial quantum states,
    creating input-dependent quantum processing instead of static states.
    
    Architecture:
    1. Input-dependent coherent state preparation
    2. Standard SF Tutorial QNN layers (preserves gradient flow)
    3. X-quadrature measurement extraction
    """
    
    def __init__(self, 
                 n_modes: int, 
                 n_layers: int, 
                 latent_dim: int,
                 cutoff_dim: int = 6,
                 input_scale_factor: float = 0.1):
        """
        Initialize input-dependent quantum circuit.
        
        Args:
            n_modes: Number of quantum modes
            n_layers: Number of QNN layers
            latent_dim: Dimensionality of input latent vector
            cutoff_dim: Fock space cutoff dimension
            input_scale_factor: Scaling factor for latent vector influence
        """
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.cutoff_dim = cutoff_dim
        self.input_scale_factor = input_scale_factor
        
        # SF Engine (preserved from SF Tutorial)
        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
        
        # QNN weights (preserved from SF Tutorial)
        self.weights = init_weights(n_modes, n_layers)
        self.num_params = np.prod(self.weights.shape)
        
        logger.info(f"Input-Dependent Quantum Circuit initialized:")
        logger.info(f"  Modes: {n_modes}, Layers: {n_layers}, Latent dim: {latent_dim}")
        logger.info(f"  Total QNN parameters: {self.num_params}")
        logger.info(f"  Input scale factor: {input_scale_factor}")
    
    def create_input_dependent_program(self, latent_vector: tf.Tensor) -> Tuple[sf.Program, np.ndarray]:
        """
        Create quantum program with input-dependent initial state.
        
        This is the key innovation: different latent vectors create different
        initial quantum states, enabling input-dependent processing.
        
        Args:
            latent_vector: Input latent vector [latent_dim]
            
        Returns:
            Tuple of (SF Program with input-dependent initial state, sf_params)
        """
        # Create new program for this specific input
        qnn = sf.Program(self.n_modes)
        
        # Create SF symbolic parameters for this program
        sf_params = np.arange(self.num_params).reshape(self.weights.shape).astype(str)
        sf_params = np.array([qnn.params(*i) for i in sf_params])
        
        with qnn.context as q:
            # INNOVATION: Input-dependent initial state preparation
            for mode in range(self.n_modes):
                # Map latent vector components to coherent state amplitudes
                latent_idx = mode % self.latent_dim
                alpha_real = tf.gather(latent_vector, latent_idx) * self.input_scale_factor
                
                # Use next latent component for imaginary part (if available)
                alpha_imag_idx = (mode + 1) % self.latent_dim
                alpha_imag = tf.gather(latent_vector, alpha_imag_idx) * self.input_scale_factor * 0.1
                
                # Prepare input-dependent coherent state
                ops.Coherent(alpha_real, alpha_imag) | q[mode]
            
            # Standard QNN layers (preserves gradient flow)
            for k in range(self.n_layers):
                layer(sf_params[k], q)
        
        return qnn, sf_params
    
    def execute_with_input(self, latent_vector: tf.Tensor) -> Any:
        """
        Execute quantum circuit with input-dependent initial state.
        
        This creates different quantum states for different latent vectors,
        solving the "same state for all inputs" problem.
        
        Args:
            latent_vector: Input latent vector [latent_dim]
            
        Returns:
            SF quantum state (input-dependent)
        """
        # Create input-dependent program
        qnn, sf_params = self.create_input_dependent_program(latent_vector)
        
        # Create parameter mapping (preserves gradient flow)
        mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(self.weights, [-1]))}
        
        # Reset engine if needed
        if self.eng.run_progs:
            self.eng.reset()
        
        # Execute input-dependent program
        state = self.eng.run(qnn, args=mapping).state
        return state
    
    def extract_x_quadratures(self, state: Any) -> tf.Tensor:
        """
        Extract X-quadrature measurements preserving gradients.
        
        Args:
            state: SF quantum state
            
        Returns:
            X-quadrature measurements [n_modes]
        """
        measurements = []
        
        for mode in range(self.n_modes):
            # Extract X-quadrature (preserves gradients)
            x_quad = state.quad_expectation(mode, 0)
            measurements.append(x_quad)
        
        # Stack to tensor
        measurement_tensor = tf.stack(measurements)
        measurement_tensor = tf.cast(measurement_tensor, tf.float32)
        measurement_tensor = tf.reshape(measurement_tensor, [-1])
        
        return measurement_tensor
    
    def process_single_input(self, latent_vector: tf.Tensor) -> tf.Tensor:
        """
        Complete input-dependent quantum processing for single input.
        
        Args:
            latent_vector: Single latent vector [latent_dim]
            
        Returns:
            X-quadrature measurements [n_modes]
        """
        # Execute with input-dependent initial state
        state = self.execute_with_input(latent_vector)
        
        # Extract measurements
        x_quadratures = self.extract_x_quadratures(state)
        
        return x_quadratures
    
    def process_batch(self, latent_batch: tf.Tensor) -> tf.Tensor:
        """
        Process batch of latent vectors (each gets different quantum state).
        
        Args:
            latent_batch: Batch of latent vectors [batch_size, latent_dim]
            
        Returns:
            Batch of X-quadrature measurements [batch_size, n_modes]
        """
        # Convert to numpy for iteration since we need to process each sample individually
        latent_numpy = latent_batch.numpy()
        quantum_outputs = []
        
        for i in range(latent_numpy.shape[0]):
            # Each sample gets its own input-dependent quantum state
            latent_vector = tf.constant(latent_numpy[i], dtype=tf.float32)
            x_quadratures = self.process_single_input(latent_vector)
            quantum_outputs.append(x_quadratures)
        
        return tf.stack(quantum_outputs, axis=0)
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables (QNN weights only)."""
        return [self.weights]
    
    def test_input_dependency(self, test_inputs: tf.Tensor) -> Dict[str, Any]:
        """
        Test that different inputs produce different outputs.
        
        Args:
            test_inputs: Test latent vectors [n_tests, latent_dim]
            
        Returns:
            Analysis of input dependency
        """
        outputs = self.process_batch(test_inputs)
        
        # Check if outputs are different for different inputs
        output_std = tf.math.reduce_std(outputs, axis=0)
        input_dependency = tf.reduce_mean(output_std)
        
        # Check if same input produces same output
        same_input = tf.tile(test_inputs[0:1], [2, 1])  # Duplicate first input
        same_outputs = self.process_batch(same_input)
        consistency = tf.reduce_mean(tf.abs(same_outputs[0] - same_outputs[1]))
        
        return {
            'input_dependency': float(input_dependency),  # Should be > 0
            'consistency': float(consistency),             # Should be ≈ 0
            'output_range': [float(tf.reduce_min(outputs)), float(tf.reduce_max(outputs))],
            'is_input_dependent': float(input_dependency) > 0.01
        }
    
    def test_gradient_flow(self) -> Tuple[float, bool]:
        """
        Test gradient flow through input-dependent circuit.
        
        Returns:
            Tuple of (gradient_flow_percentage, all_gradients_present)
        """
        # Test data
        latent_test = tf.random.normal([4, self.latent_dim])
        
        with tf.GradientTape() as tape:
            outputs = self.process_batch(latent_test)
            # Create target with correct shape (same as outputs)
            target = tf.random.normal(tf.shape(outputs))
            loss = tf.reduce_mean(tf.square(outputs - target))
        
        # Check gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        valid_gradients = [g for g in gradients if g is not None]
        gradient_flow = len(valid_gradients) / len(self.trainable_variables)
        all_present = len(valid_gradients) == len(self.trainable_variables)
        
        return gradient_flow, all_present


def test_input_dependent_circuit():
    """Test the input-dependent quantum circuit implementation."""
    print("🧪 Testing Input-Dependent Quantum Circuit (Phase 1)")
    print("=" * 70)
    
    try:
        # Create circuit
        circuit = InputDependentQuantumCircuit(
            n_modes=3,
            n_layers=2,
            latent_dim=4,
            cutoff_dim=4,
            input_scale_factor=0.1
        )
        
        print(f"\n1. Circuit Initialization:")
        print(f"   Modes: {circuit.n_modes}, Layers: {circuit.n_layers}")
        print(f"   Latent dim: {circuit.latent_dim}")
        print(f"   Parameters: {circuit.num_params}")
        
        # Test input dependency
        print(f"\n2. Testing Input Dependency:")
        test_inputs = tf.random.normal([4, circuit.latent_dim])
        dependency_results = circuit.test_input_dependency(test_inputs)
        
        print(f"   Input dependency: {dependency_results['input_dependency']:.4f} (should be > 0.01)")
        print(f"   Consistency: {dependency_results['consistency']:.6f} (should be ≈ 0)")
        print(f"   Output range: {dependency_results['output_range']}")
        print(f"   Is input dependent: {'✅' if dependency_results['is_input_dependent'] else '❌'}")
        
        # Test gradient flow
        print(f"\n3. Testing Gradient Flow:")
        gradient_flow, all_present = circuit.test_gradient_flow()
        
        print(f"   Gradient flow: {gradient_flow:.1%} (should be 100%)")
        print(f"   All gradients present: {'✅' if all_present else '❌'}")
        
        # Test batch processing
        print(f"\n4. Testing Batch Processing:")
        batch_inputs = tf.random.normal([8, circuit.latent_dim])
        batch_outputs = circuit.process_batch(batch_inputs)
        
        print(f"   Input shape: {batch_inputs.shape}")
        print(f"   Output shape: {batch_outputs.shape}")
        print(f"   Sample output: {batch_outputs[0].numpy()}")
        
        # Success assessment
        input_dependent = dependency_results['is_input_dependent']
        gradient_flow_ok = gradient_flow > 0.99
        consistency_ok = dependency_results['consistency'] < 0.001
        
        success = input_dependent and gradient_flow_ok and consistency_ok
        
        print(f"\n🎯 Phase 1 Test Results:")
        print(f"   Input dependency: {'✅' if input_dependent else '❌'}")
        print(f"   Gradient flow: {'✅' if gradient_flow_ok else '❌'} ({gradient_flow:.1%})")
        print(f"   Consistency: {'✅' if consistency_ok else '❌'} ({dependency_results['consistency']:.6f})")
        print(f"   Overall: {'✅ SUCCESS - Ready for Phase 2' if success else '❌ NEEDS FIXES'}")
        
        return success, dependency_results, gradient_flow
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, 0.0


if __name__ == "__main__":
    test_input_dependent_circuit()
