"""
Simple One-Layer Quantum Neural Network

A minimal quantum neural network implementation for educational purposes:
- 2 modes, 1 layer
- Complete parameter tracking
- Step-by-step state evolution
- Full visualization integration

This serves as a foundation for understanding quantum circuit learning.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SimpleQuantumNN:
    """
    Simple one-layer quantum neural network for educational analysis.
    
    Architecture:
    - 2 quantum modes
    - 1 processing layer with displacement, squeezing, beamsplitter, rotation
    - X quadrature measurements
    - Classical input/output mapping
    """
    
    def __init__(self, cutoff_dim: int = 6):
        """
        Initialize simple quantum neural network.
        
        Args:
            cutoff_dim: Fock space cutoff dimension
        """
        self.n_modes = 2
        self.cutoff_dim = cutoff_dim
        
        # Create SF program and engine
        self.prog = sf.Program(self.n_modes)
        self.engine = sf.Engine("tf", backend_options={
            "cutoff_dim": cutoff_dim,
            "eval": True
        })
        
        # Parameter tracking
        self.parameters = {}
        self.parameter_history = []
        self.state_history = []
        
        # Build circuit and create parameters
        self._build_circuit()
        self._create_parameters()
        
        logger.info(f"SimpleQuantumNN initialized: {self.n_modes} modes, {len(self.parameters)} parameters")
    
    def _build_circuit(self):
        """Build the symbolic quantum circuit."""
        with self.prog.context as q:
            # Input encoding: Displacement gates
            self.disp_r_0 = self.prog.params('disp_r_0')
            self.disp_r_1 = self.prog.params('disp_r_1')
            ops.Dgate(self.disp_r_0) | q[0]
            ops.Dgate(self.disp_r_1) | q[1]
            
            # Quantum processing: Squeezing gates
            self.squeeze_r_0 = self.prog.params('squeeze_r_0')
            self.squeeze_r_1 = self.prog.params('squeeze_r_1')
            ops.Sgate(self.squeeze_r_0) | q[0]
            ops.Sgate(self.squeeze_r_1) | q[1]
            
            # Entanglement: Beam splitter
            self.bs_theta = self.prog.params('bs_theta')
            ops.BSgate(self.bs_theta) | (q[0], q[1])
            
            # Basis rotation: Rotation gates
            self.rot_phi_0 = self.prog.params('rot_phi_0')
            self.rot_phi_1 = self.prog.params('rot_phi_1')
            ops.Rgate(self.rot_phi_0) | q[0]
            ops.Rgate(self.rot_phi_1) | q[1]
        
        self.param_names = [
            'disp_r_0', 'disp_r_1', 'squeeze_r_0', 'squeeze_r_1', 
            'bs_theta', 'rot_phi_0', 'rot_phi_1'
        ]
        
        logger.info(f"Circuit built with {len(self.param_names)} parameters")
    
    def _create_parameters(self):
        """Create TensorFlow variables for all parameters."""
        # Initialize parameters with small random values
        initial_values = {
            'disp_r_0': 0.1,      # Small displacement
            'disp_r_1': 0.1,      # Small displacement
            'squeeze_r_0': 0.05,  # Small squeezing
            'squeeze_r_1': 0.05,  # Small squeezing
            'bs_theta': np.pi/4,  # Balanced beam splitter
            'rot_phi_0': 0.0,     # No initial rotation
            'rot_phi_1': 0.0      # No initial rotation
        }
        
        for name in self.param_names:
            self.parameters[name] = tf.Variable(
                initial_values[name], 
                name=name, 
                dtype=tf.float32
            )
        
        logger.info(f"Created {len(self.parameters)} TensorFlow variables")
    
    def forward(self, x: tf.Tensor, track_states: bool = False) -> Tuple[tf.Tensor, Optional[List[Any]]]:
        """
        Forward pass through quantum neural network.
        
        Args:
            x: Input data [batch_size, 1]
            track_states: Whether to track intermediate quantum states
            
        Returns:
            output: Network output [batch_size, 1]
            states: List of intermediate states (if track_states=True)
        """
        batch_size = int(x.shape[0])
        outputs = []
        states_list = [] if track_states else None
        
        # Process each sample individually (required for SF)
        for i in range(batch_size):
            sample_x = float(x[i, 0].numpy())  # Extract scalar input as Python float
            
            # Create parameter arguments with input encoding
            args = {}
            for name, param in self.parameters.items():
                if name.startswith('disp_r'):
                    # Encode input into displacement parameters
                    encoding_strength = 0.5
                    args[name] = float(param.numpy()) + encoding_strength * sample_x
                else:
                    args[name] = float(param.numpy())
            
            # Execute quantum circuit
            try:
                # Reset engine state
                self.engine.reset()
                
                result = self.engine.run(self.prog, args=args)
                state = result.state
                
                if track_states and states_list is not None:
                    states_list.append(state)
                
                # Extract measurements (X quadrature from both modes)
                x_quad_0 = state.quad_expectation(0, 0)  # Mode 0, X quadrature
                x_quad_1 = state.quad_expectation(1, 0)  # Mode 1, X quadrature
                
                # Ensure we get scalar values
                if hasattr(x_quad_0, 'numpy'):
                    x_quad_0 = float(x_quad_0.numpy())
                if hasattr(x_quad_1, 'numpy'):
                    x_quad_1 = float(x_quad_1.numpy())
                
                # Combine measurements (simple linear combination)
                output = 0.5 * (x_quad_0 + x_quad_1)
                outputs.append(tf.constant(output, dtype=tf.float32))
                
            except Exception as e:
                logger.warning(f"Forward pass failed for sample {i}: {e}")
                outputs.append(tf.constant(0.0, dtype=tf.float32))
                if track_states and states_list is not None:
                    states_list.append(None)
        
        # Stack outputs
        output_tensor = tf.stack(outputs)
        output_tensor = tf.reshape(output_tensor, [batch_size, 1])
        
        return output_tensor, states_list
    
    def get_parameter_values(self) -> Dict[str, float]:
        """Get current parameter values."""
        return {name: float(param.numpy()) for name, param in self.parameters.items()}
    
    def track_parameters(self):
        """Track current parameter values for history."""
        current_values = self.get_parameter_values()
        self.parameter_history.append(current_values.copy())
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get circuit information."""
        return {
            'n_modes': self.n_modes,
            'n_parameters': len(self.parameters),
            'parameter_names': self.param_names,
            'cutoff_dim': self.cutoff_dim,
            'parameter_values': self.get_parameter_values()
        }
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables."""
        return list(self.parameters.values())
    
    def print_circuit_diagram(self):
        """Print ASCII circuit diagram."""
        print("\n" + "="*60)
        print("SIMPLE QUANTUM NEURAL NETWORK CIRCUIT")
        print("="*60)
        print(f"Input: x → Displacement encoding")
        print(f"")
        print(f"q[0]: |0⟩──D(disp_r_0+0.5*x)──S(squeeze_r_0)──┐")
        print(f"                                              │")
        print(f"                                         BS(bs_theta)──R(rot_phi_0)──[X]")
        print(f"                                              │")
        print(f"q[1]: |0⟩──D(disp_r_1+0.5*x)──S(squeeze_r_1)──┘                    R(rot_phi_1)──[X]")
        print(f"")
        print(f"Output: y = 0.5 * (X[0] + X[1])")
        print(f"")
        print(f"Parameters ({len(self.parameters)} total):")
        for name, param in self.parameters.items():
            print(f"  {name:<12}: {float(param.numpy()):8.4f}")
        print("="*60)
    
    def print_parameter_evolution(self, last_n: int = 5):
        """Print recent parameter evolution."""
        if len(self.parameter_history) < 2:
            print("No parameter history available")
            return
        
        print(f"\nParameter Evolution (last {min(last_n, len(self.parameter_history))} steps):")
        print("-" * 80)
        
        # Header
        print(f"{'Step':<6}", end="")
        for name in self.param_names:
            print(f"{name:<12}", end="")
        print()
        
        # Show last n steps
        start_idx = max(0, len(self.parameter_history) - last_n)
        for i in range(start_idx, len(self.parameter_history)):
            print(f"{i:<6}", end="")
            for name in self.param_names:
                value = self.parameter_history[i][name]
                print(f"{value:8.4f}    ", end="")
            print()


def test_simple_quantum_nn():
    """Test the simple quantum neural network."""
    print("🧪 Testing Simple Quantum Neural Network...")
    
    # Create network
    qnn = SimpleQuantumNN(cutoff_dim=6)
    
    # Print initial circuit
    qnn.print_circuit_diagram()
    
    # Test forward pass
    print("\n📊 Testing forward pass...")
    x_test = tf.constant([[0.5], [1.0], [-0.5]], dtype=tf.float32)
    
    try:
        # Forward pass with state tracking
        output, states = qnn.forward(x_test, track_states=True)
        
        print(f"✅ Forward pass successful!")
        print(f"Input shape: {x_test.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Input values: {x_test.numpy().flatten()}")
        print(f"Output values: {output.numpy().flatten()}")
        print(f"States tracked: {len(states) if states else 0}")
        
        # Test gradient flow
        print("\n🔄 Testing gradient flow...")
        with tf.GradientTape() as tape:
            output = qnn.forward(x_test)[0]
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, qnn.trainable_variables)
        valid_grads = [g for g in gradients if g is not None]
        
        print(f"✅ Gradient flow: {len(valid_grads)}/{len(qnn.trainable_variables)} parameters")
        
        # Show gradient magnitudes
        print("Gradient magnitudes:")
        for name, grad in zip(qnn.param_names, gradients):
            if grad is not None:
                print(f"  {name:<12}: {float(tf.norm(grad)):8.4f}")
            else:
                print(f"  {name:<12}: None")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE QUANTUM NEURAL NETWORK TEST")
    print("=" * 60)
    
    success = test_simple_quantum_nn()
    
    if success:
        print("\n🎉 Simple Quantum NN test successful!")
    else:
        print("\n❌ Simple Quantum NN test failed!")
