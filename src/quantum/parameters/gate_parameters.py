"""
Gate Parameter Manager for Quantum Circuits

This module manages individual gate parameters while maintaining
gradient flow through TensorFlow variables.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class GateParameterManager:
    """
    Manages individual quantum gate parameters.
    
    Creates and organizes tf.Variables for each quantum gate
    while maintaining the structure needed for gradient flow.
    """
    
    def __init__(self, n_modes: int, layers: int):
        """
        Initialize gate parameter manager.
        
        Args:
            n_modes: Number of quantum modes
            layers: Number of circuit layers
        """
        self.n_modes = n_modes
        self.layers = layers
        
        # Initialize gate parameters
        self.gate_params = {}
        self._init_gate_parameters()
        
        logger.info(f"Gate parameter manager initialized: {self.get_parameter_count()} parameters")
    
    def _init_gate_parameters(self) -> None:
        """Initialize individual tf.Variable for each quantum gate."""
        for layer in range(self.layers):
            layer_params = {}
            
            # Calculate number of beamsplitters
            n_bs = self.n_modes * (self.n_modes - 1) // 2
            n_rot = max(1, self.n_modes - 1)
            
            # Interferometer 1 parameters
            layer_params['bs1_theta'] = [
                tf.Variable(tf.random.normal([], stddev=0.1), name=f'L{layer}_BS1_theta_{i}')
                for i in range(n_bs)
            ]
            layer_params['bs1_phi'] = [
                tf.Variable(tf.random.normal([], stddev=0.1), name=f'L{layer}_BS1_phi_{i}')
                for i in range(n_bs)
            ]
            layer_params['rot1_phi'] = [
                tf.Variable(tf.random.uniform([], 0, 2*np.pi), name=f'L{layer}_ROT1_phi_{i}')
                for i in range(n_rot)
            ]
            
            # Squeezing parameters
            layer_params['squeeze_r'] = [
                tf.Variable(tf.random.normal([], stddev=0.01), name=f'L{layer}_SQUEEZE_r_{i}')
                for i in range(self.n_modes)
            ]
            
            # Interferometer 2 parameters
            layer_params['bs2_theta'] = [
                tf.Variable(tf.random.normal([], stddev=0.1), name=f'L{layer}_BS2_theta_{i}')
                for i in range(n_bs)
            ]
            layer_params['bs2_phi'] = [
                tf.Variable(tf.random.normal([], stddev=0.1), name=f'L{layer}_BS2_phi_{i}')
                for i in range(n_bs)
            ]
            layer_params['rot2_phi'] = [
                tf.Variable(tf.random.uniform([], 0, 2*np.pi), name=f'L{layer}_ROT2_phi_{i}')
                for i in range(n_rot)
            ]
            
            # Displacement parameters
            layer_params['disp_r'] = [
                tf.Variable(tf.random.normal([], stddev=0.01), name=f'L{layer}_DISP_r_{i}')
                for i in range(self.n_modes)
            ]
            layer_params['disp_phi'] = [
                tf.Variable(tf.random.uniform([], 0, 2*np.pi), name=f'L{layer}_DISP_phi_{i}')
                for i in range(self.n_modes)
            ]
            
            # Kerr nonlinearity parameters
            layer_params['kerr_kappa'] = [
                tf.Variable(tf.random.normal([], stddev=0.001), name=f'L{layer}_KERR_kappa_{i}')
                for i in range(self.n_modes)
            ]
            
            self.gate_params[f'layer_{layer}'] = layer_params
    
    def create_symbolic_parameters(self, prog: sf.Program) -> Dict[str, Dict[str, List[Any]]]:
        """
        Create symbolic parameters for SF program.
        
        Args:
            prog: Strawberry Fields program
            
        Returns:
            Dictionary of symbolic parameters
        """
        symbolic_params = {}
        
        for layer in range(self.layers):
            layer_symbols = {}
            layer_key = f'layer_{layer}'
            
            # Create symbolic parameters matching gate structure
            n_bs = self.n_modes * (self.n_modes - 1) // 2
            n_rot = max(1, self.n_modes - 1)
            
            layer_symbols['bs1_theta'] = [prog.params(f'L{layer}_bs1_theta_{i}') for i in range(n_bs)]
            layer_symbols['bs1_phi'] = [prog.params(f'L{layer}_bs1_phi_{i}') for i in range(n_bs)]
            layer_symbols['rot1_phi'] = [prog.params(f'L{layer}_rot1_phi_{i}') for i in range(n_rot)]
            
            layer_symbols['squeeze_r'] = [prog.params(f'L{layer}_squeeze_r_{i}') for i in range(self.n_modes)]
            
            layer_symbols['bs2_theta'] = [prog.params(f'L{layer}_bs2_theta_{i}') for i in range(n_bs)]
            layer_symbols['bs2_phi'] = [prog.params(f'L{layer}_bs2_phi_{i}') for i in range(n_bs)]
            layer_symbols['rot2_phi'] = [prog.params(f'L{layer}_rot2_phi_{i}') for i in range(n_rot)]
            
            layer_symbols['disp_r'] = [prog.params(f'L{layer}_disp_r_{i}') for i in range(self.n_modes)]
            layer_symbols['disp_phi'] = [prog.params(f'L{layer}_disp_phi_{i}') for i in range(self.n_modes)]
            
            layer_symbols['kerr_kappa'] = [prog.params(f'L{layer}_kerr_kappa_{i}') for i in range(self.n_modes)]
            
            symbolic_params[layer_key] = layer_symbols
        
        return symbolic_params
    
    def get_parameter_mapping(self, modulation: Optional[Dict[str, tf.Tensor]] = None) -> Dict[str, tf.Tensor]:
        """
        Get parameter mapping for SF execution.
        
        Args:
            modulation: Optional parameter modulation
            
        Returns:
            Dictionary mapping parameter names to values
        """
        mapping = {}
        
        for layer in range(self.layers):
            layer_key = f'layer_{layer}'
            gate_params = self.gate_params[layer_key]
            
            for param_type, param_list in gate_params.items():
                for i, param_var in enumerate(param_list):
                    # Use the exact parameter name format that matches the symbolic parameters
                    param_name = f'L{layer}_{param_type}_{i}'
                    
                    # Get base parameter value
                    base_value = param_var
                    
                    # Add modulation if provided
                    if modulation and param_name in modulation:
                        modulated_value = base_value + modulation[param_name]
                    else:
                        modulated_value = base_value
                    
                    mapping[param_name] = modulated_value
        
        return mapping
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable gate parameters."""
        variables = []
        for layer_params in self.gate_params.values():
            for param_list in layer_params.values():
                variables.extend(param_list)
        return variables
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return len(self.trainable_variables)
    
    def get_parameter_structure(self) -> Dict[str, Dict[str, int]]:
        """Get detailed parameter structure information."""
        structure = {}
        
        for layer_key, layer_params in self.gate_params.items():
            layer_info = {}
            for param_type, param_list in layer_params.items():
                layer_info[param_type] = len(param_list)
            structure[layer_key] = layer_info
        
        return structure
    
    def get_parameter_by_name(self, param_name: str) -> Optional[tf.Variable]:
        """
        Get a specific parameter by name.
        
        Args:
            param_name: Parameter name (e.g., 'L0_BS1_theta_0')
            
        Returns:
            The parameter variable or None if not found
        """
        # Parse parameter name
        parts = param_name.split('_')
        if len(parts) < 4:
            return None
        
        layer = int(parts[0][1:])  # Remove 'L' prefix
        param_type = '_'.join(parts[1:-1]).lower()
        param_idx = int(parts[-1])
        
        layer_key = f'layer_{layer}'
        if layer_key in self.gate_params and param_type in self.gate_params[layer_key]:
            param_list = self.gate_params[layer_key][param_type]
            if param_idx < len(param_list):
                return param_list[param_idx]
        
        return None


class ParameterModulator:
    """
    Handles parameter modulation for input encoding.
    
    Converts input encodings to parameter modulations while
    maintaining gradient flow.
    """
    
    def __init__(self, encoding_dim: int, n_parameters: int):
        """
        Initialize parameter modulator.
        
        Args:
            encoding_dim: Dimension of input encoding
            n_parameters: Total number of circuit parameters
        """
        self.encoding_dim = encoding_dim
        self.n_parameters = n_parameters
        
        # Encoding matrix for modulation
        self.encoding_matrix = tf.Variable(
            tf.random.normal([encoding_dim, n_parameters], stddev=0.01),
            name="parameter_encoding_matrix"
        )
        
        logger.info(f"Parameter modulator initialized: {encoding_dim} → {n_parameters}")
    
    def create_modulation(self, encoding: tf.Tensor, param_names: List[str]) -> Dict[str, tf.Tensor]:
        """
        Create parameter modulation from encoding.
        
        Args:
            encoding: Input encoding tensor [batch_size, encoding_dim]
            param_names: List of parameter names in order
            
        Returns:
            Dictionary of parameter modulations
        """
        # Transform encoding to parameter modulation
        param_modulation = tf.matmul(encoding, self.encoding_matrix)
        param_modulation = tf.nn.tanh(param_modulation) * 0.1  # Small modulation
        
        # Create modulation dictionary
        modulation_dict = {}
        for i, param_name in enumerate(param_names):
            if i < param_modulation.shape[1]:
                modulation_dict[param_name] = param_modulation[:, i]
        
        return modulation_dict
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables."""
        return [self.encoding_matrix]
