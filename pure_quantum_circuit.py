"""
Pure Quantum Circuit with Individual Gate Parameters

This module implements a pure quantum approach where each quantum gate 
has individual parameters, completely removing classical neural network 
components from the quantum circuit itself.

Architecture:
- Each BSgate, Sgate, Dgate has individual tf.Variable parameters
- No classical neural networks inside quantum circuits
- Preserves gradient flow through TensorFlow computation graph
- Single SF program to prevent gradient breaking
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class PureQuantumCircuit:
    """
    Pure quantum circuit where each gate has individual parameters.
    
    No classical components - only quantum gates with learnable parameters.
    All parameters are individual tf.Variables for maximum gradient flow.
    """
    
    def __init__(self, n_modes=4, layers=2, cutoff_dim=8):
        """
        Initialize pure quantum circuit.
        
        Args:
            n_modes (int): Number of quantum modes
            layers (int): Number of quantum layers
            cutoff_dim (int): Fock space cutoff dimension
        """
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        logger.info(f"Initializing Pure Quantum Circuit:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize individual quantum gate parameters
        self._init_individual_gate_parameters()
        
        # Create symbolic parameters
        self._create_symbolic_parameters()
        
        # Build quantum program
        self._build_quantum_program()
    
    def _init_sf_components(self):
        """Initialize Strawberry Fields engine and program."""
        self.eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        self.prog = sf.Program(self.n_modes)
        logger.info("SF engine and program initialized")
    
    def _init_individual_gate_parameters(self):
        """Initialize individual tf.Variable for each quantum gate."""
        self.gate_params = {}
        
        # For each layer
        for layer in range(self.layers):
            layer_params = {}
            
            # 1. INTERFEROMETER 1 PARAMETERS
            # Beamsplitter parameters (theta, phi for each mode pair)
            layer_params['bs1_theta'] = []
            layer_params['bs1_phi'] = []
            
            # Calculate number of beamsplitter pairs
            n_bs = self.n_modes * (self.n_modes - 1) // 2
            for i in range(n_bs):
                # Individual parameter for each beamsplitter
                theta_var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'L{layer}_BS1_theta_{i}'
                )
                phi_var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'L{layer}_BS1_phi_{i}'
                )
                layer_params['bs1_theta'].append(theta_var)
                layer_params['bs1_phi'].append(phi_var)
            
            # Rotation phases for interferometer
            layer_params['rot1_phi'] = []
            for i in range(max(1, self.n_modes - 1)):
                rot_var = tf.Variable(
                    tf.random.uniform([], 0, 2*np.pi),
                    name=f'L{layer}_ROT1_phi_{i}'
                )
                layer_params['rot1_phi'].append(rot_var)
            
            # 2. SQUEEZING PARAMETERS
            layer_params['squeeze_r'] = []
            for i in range(self.n_modes):
                squeeze_var = tf.Variable(
                    tf.random.normal([], stddev=0.01),  # Small initial squeezing
                    name=f'L{layer}_SQUEEZE_r_{i}'
                )
                layer_params['squeeze_r'].append(squeeze_var)
            
            # 3. INTERFEROMETER 2 PARAMETERS
            layer_params['bs2_theta'] = []
            layer_params['bs2_phi'] = []
            
            for i in range(n_bs):
                theta_var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'L{layer}_BS2_theta_{i}'
                )
                phi_var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'L{layer}_BS2_phi_{i}'
                )
                layer_params['bs2_theta'].append(theta_var)
                layer_params['bs2_phi'].append(phi_var)
            
            layer_params['rot2_phi'] = []
            for i in range(max(1, self.n_modes - 1)):
                rot_var = tf.Variable(
                    tf.random.uniform([], 0, 2*np.pi),
                    name=f'L{layer}_ROT2_phi_{i}'
                )
                layer_params['rot2_phi'].append(rot_var)
            
            # 4. DISPLACEMENT PARAMETERS
            layer_params['disp_r'] = []
            layer_params['disp_phi'] = []
            for i in range(self.n_modes):
                r_var = tf.Variable(
                    tf.random.normal([], stddev=0.01),  # Small initial displacement
                    name=f'L{layer}_DISP_r_{i}'
                )
                phi_var = tf.Variable(
                    tf.random.uniform([], 0, 2*np.pi),
                    name=f'L{layer}_DISP_phi_{i}'
                )
                layer_params['disp_r'].append(r_var)
                layer_params['disp_phi'].append(phi_var)
            
            # 5. KERR NONLINEARITY PARAMETERS
            layer_params['kerr_kappa'] = []
            for i in range(self.n_modes):
                kerr_var = tf.Variable(
                    tf.random.normal([], stddev=0.001),  # Very small nonlinearity
                    name=f'L{layer}_KERR_kappa_{i}'
                )
                layer_params['kerr_kappa'].append(kerr_var)
            
            self.gate_params[f'layer_{layer}'] = layer_params
        
        # Count total parameters
        total_params = 0
        for layer_key, layer_params in self.gate_params.items():
            for param_type, param_list in layer_params.items():
                total_params += len(param_list)
        
        logger.info(f"Individual gate parameters initialized: {total_params} total parameters")
    
    def _create_symbolic_parameters(self):
        """Create symbolic parameters for SF program."""
        self.symbolic_params = {}
        
        for layer in range(self.layers):
            layer_symbols = {}
            layer_key = f'layer_{layer}'
            
            # Create symbolic parameters matching our gate structure
            # BS1 parameters
            layer_symbols['bs1_theta'] = []
            layer_symbols['bs1_phi'] = []
            n_bs = self.n_modes * (self.n_modes - 1) // 2
            for i in range(n_bs):
                layer_symbols['bs1_theta'].append(self.prog.params(f'L{layer}_BS1_theta_{i}'))
                layer_symbols['bs1_phi'].append(self.prog.params(f'L{layer}_BS1_phi_{i}'))
            
            # ROT1 parameters
            layer_symbols['rot1_phi'] = []
            for i in range(max(1, self.n_modes - 1)):
                layer_symbols['rot1_phi'].append(self.prog.params(f'L{layer}_ROT1_phi_{i}'))
            
            # Squeeze parameters
            layer_symbols['squeeze_r'] = []
            for i in range(self.n_modes):
                layer_symbols['squeeze_r'].append(self.prog.params(f'L{layer}_SQUEEZE_r_{i}'))
            
            # BS2 parameters
            layer_symbols['bs2_theta'] = []
            layer_symbols['bs2_phi'] = []
            for i in range(n_bs):
                layer_symbols['bs2_theta'].append(self.prog.params(f'L{layer}_BS2_theta_{i}'))
                layer_symbols['bs2_phi'].append(self.prog.params(f'L{layer}_BS2_phi_{i}'))
            
            # ROT2 parameters
            layer_symbols['rot2_phi'] = []
            for i in range(max(1, self.n_modes - 1)):
                layer_symbols['rot2_phi'].append(self.prog.params(f'L{layer}_ROT2_phi_{i}'))
            
            # Displacement parameters
            layer_symbols['disp_r'] = []
            layer_symbols['disp_phi'] = []
            for i in range(self.n_modes):
                layer_symbols['disp_r'].append(self.prog.params(f'L{layer}_DISP_r_{i}'))
                layer_symbols['disp_phi'].append(self.prog.params(f'L{layer}_DISP_phi_{i}'))
            
            # Kerr parameters
            layer_symbols['kerr_kappa'] = []
            for i in range(self.n_modes):
                layer_symbols['kerr_kappa'].append(self.prog.params(f'L{layer}_KERR_kappa_{i}'))
            
            self.symbolic_params[layer_key] = layer_symbols
        
        logger.info("Symbolic parameters created")
    
    def _build_quantum_program(self):
        """Build quantum program with individual gate operations."""
        with self.prog.context as q:
            # Apply layers sequentially
            for layer in range(self.layers):
                self._apply_quantum_layer(layer, q)
        
        logger.info(f"Pure quantum program built with {self.layers} layers")
    
    def _apply_quantum_layer(self, layer, q):
        """Apply a single quantum layer with individual gate parameters."""
        layer_key = f'layer_{layer}'
        symbols = self.symbolic_params[layer_key]
        
        # 1. First Interferometer
        self._apply_interferometer(
            symbols['bs1_theta'], 
            symbols['bs1_phi'], 
            symbols['rot1_phi'], 
            q
        )
        
        # 2. Squeezing gates
        for i in range(self.n_modes):
            ops.Sgate(symbols['squeeze_r'][i]) | q[i]
        
        # 3. Second Interferometer
        self._apply_interferometer(
            symbols['bs2_theta'], 
            symbols['bs2_phi'], 
            symbols['rot2_phi'], 
            q
        )
        
        # 4. Displacement gates
        for i in range(self.n_modes):
            ops.Dgate(symbols['disp_r'][i], symbols['disp_phi'][i]) | q[i]
        
        # 5. Kerr nonlinearity
        for i in range(self.n_modes):
            ops.Kgate(symbols['kerr_kappa'][i]) | q[i]
    
    def _apply_interferometer(self, theta_params, phi_params, rot_params, q):
        """Apply interferometer with individual beamsplitter parameters."""
        N = len(q)
        
        if N == 1:
            # Single mode: just rotation
            ops.Rgate(rot_params[0]) | q[0]
            return
        
        # Apply beamsplitter array
        param_idx = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    if param_idx < len(theta_params):
                        ops.BSgate(theta_params[param_idx], phi_params[param_idx]) | (q1, q2)
                        param_idx += 1
        
        # Apply final rotations
        for i in range(min(len(rot_params), len(q))):
            ops.Rgate(rot_params[i]) | q[i]
    
    @property
    def trainable_variables(self):
        """Return all individual gate parameters as trainable variables."""
        variables = []
        
        for layer_key, layer_params in self.gate_params.items():
            for param_type, param_list in layer_params.items():
                variables.extend(param_list)
        
        return variables
    
    def create_parameter_mapping(self, input_modulation=None):
        """
        Create parameter mapping for SF execution.
        
        Args:
            input_modulation (dict): Optional modulation to add to base parameters
            
        Returns:
            dict: Parameter mapping for SF engine
        """
        mapping = {}
        
        for layer in range(self.layers):
            layer_key = f'layer_{layer}'
            gate_params = self.gate_params[layer_key]
            
            # Map each individual parameter
            for param_type, param_list in gate_params.items():
                for i, param_var in enumerate(param_list):
                    param_name = f'L{layer}_{param_type.upper()}_{i}'
                    
                    # Get base parameter value
                    base_value = param_var
                    
                    # Add modulation if provided
                    if input_modulation and param_name in input_modulation:
                        modulated_value = base_value + input_modulation[param_name]
                    else:
                        modulated_value = base_value
                    
                    mapping[param_name] = modulated_value
        
        return mapping
    
    def execute_circuit(self, input_modulation=None):
        """
        Execute quantum circuit with optional input modulation.
        
        Args:
            input_modulation (dict): Optional parameter modulation
            
        Returns:
            SF quantum state
        """
        # Create parameter mapping
        mapping = self.create_parameter_mapping(input_modulation)
        
        # Reset engine if needed
        if self.eng.run_progs:
            self.eng.reset()
        
        # Execute circuit
        state = self.eng.run(self.prog, args=mapping).state
        
        return state
    
    def get_parameter_count(self):
        """Get total number of individual parameters."""
        total = 0
        for layer_params in self.gate_params.values():
            for param_list in layer_params.values():
                total += len(param_list)
        return total
    
    def get_parameter_structure(self):
        """Get detailed parameter structure information."""
        structure = {}
        
        for layer_key, layer_params in self.gate_params.items():
            layer_info = {}
            for param_type, param_list in layer_params.items():
                layer_info[param_type] = len(param_list)
            structure[layer_key] = layer_info
        
        return structure


def test_pure_quantum_circuit():
    """Test the pure quantum circuit implementation."""
    print("\n" + "="*60)
    print("TESTING PURE QUANTUM CIRCUIT")
    print("="*60)
    
    # Create circuit
    circuit = PureQuantumCircuit(n_modes=4, layers=2, cutoff_dim=6)
    
    print(f"✅ Circuit created successfully")
    print(f"📊 Total parameters: {circuit.get_parameter_count()}")
    print(f"🔧 Trainable variables: {len(circuit.trainable_variables)}")
    
    # Test parameter structure
    structure = circuit.get_parameter_structure()
    print(f"\n📋 Parameter structure:")
    for layer_key, layer_info in structure.items():
        print(f"  {layer_key}:")
        for param_type, count in layer_info.items():
            print(f"    {param_type}: {count} parameters")
    
    # Test circuit execution
    print(f"\n🔄 Testing circuit execution...")
    try:
        state = circuit.execute_circuit()
        print(f"✅ Circuit execution successful")
        print(f"📏 State shape: {state.ket().shape}")
        print(f"🎯 State norm: {tf.reduce_sum(tf.abs(state.ket())**2):.4f}")
    except Exception as e:
        print(f"❌ Circuit execution failed: {e}")
        return None
    
    # Test gradient flow
    print(f"\n🌊 Testing gradient flow...")
    try:
        with tf.GradientTape() as tape:
            state = circuit.execute_circuit()
            loss = tf.reduce_mean(tf.abs(state.ket())**2)
        
        gradients = tape.gradient(loss, circuit.trainable_variables)
        grad_status = [g is not None for g in gradients]
        
        print(f"✅ Gradient test: {sum(grad_status)}/{len(grad_status)} parameters have gradients")
        print(f"📈 Loss: {loss:.6f}")
        
        if all(grad_status):
            print(f"🎉 ALL PARAMETERS HAVE GRADIENTS!")
        else:
            print(f"⚠️  Some parameters missing gradients")
            
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
    
    # Test with input modulation
    print(f"\n🎛️  Testing input modulation...")
    try:
        # Create simple modulation
        modulation = {
            'L0_DISP_R_0': tf.constant(0.1),
            'L0_DISP_R_1': tf.constant(-0.1),
        }
        
        state_modulated = circuit.execute_circuit(input_modulation=modulation)
        print(f"✅ Modulated execution successful")
        print(f"📏 Modulated state norm: {tf.reduce_sum(tf.abs(state_modulated.ket())**2):.4f}")
        
    except Exception as e:
        print(f"❌ Modulation test failed: {e}")
    
    return circuit


if __name__ == "__main__":
    circuit = test_pure_quantum_circuit()