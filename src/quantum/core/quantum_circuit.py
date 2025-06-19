"""
Core Quantum Circuit with Gradient-Safe Architecture

This module implements the base quantum circuit that ensures gradient flow
is preserved through a single SF program while maintaining modularity.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Force eager execution for SF gradient compatibility
tf.config.run_functions_eagerly(True)

logger = logging.getLogger(__name__)

class QuantumCircuitBase(ABC):
    """
    Abstract base class for quantum circuits.
    
    Ensures gradient flow by maintaining a single SF program and engine
    while allowing modular construction of circuit components.
    """
    
    def __init__(self, n_modes: int, cutoff_dim: int = 8):
        """
        Initialize quantum circuit base.
        
        Args:
            n_modes: Number of quantum modes
            cutoff_dim: Fock space cutoff dimension
        """
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        
        # Single SF program and engine for gradient preservation
        self.prog = sf.Program(self.n_modes)
        self.eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        
        # Track if circuit has been built
        self._circuit_built = False
        
        logger.info(f"Quantum circuit base initialized: {n_modes} modes, cutoff={cutoff_dim}")
    
    @abstractmethod
    def build_circuit(self) -> None:
        """Build the quantum circuit. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_parameter_mapping(self, modulation: Optional[Dict[str, tf.Tensor]] = None) -> Dict[str, tf.Tensor]:
        """
        Get parameter mapping for circuit execution.
        
        Args:
            modulation: Optional parameter modulation
            
        Returns:
            Dictionary mapping parameter names to values
        """
        pass
    
    @property
    @abstractmethod
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables in the circuit."""
        pass
    
    def execute(self, modulation: Optional[Dict[str, tf.Tensor]] = None) -> Any:
        """
        Execute the quantum circuit.
        
        Args:
            modulation: Optional parameter modulation
            
        Returns:
            Quantum state from SF engine
        """
        # Build circuit if not already built
        if not self._circuit_built:
            self.build_circuit()
            self._circuit_built = True
        
        # Get parameter mapping
        mapping = self.get_parameter_mapping(modulation)
        
        # Reset engine if needed
        if self.eng.run_progs:
            self.eng.reset()
        
        # Execute circuit - single execution point for gradient preservation
        state = self.eng.run(self.prog, args=mapping).state
        
        return state
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the circuit structure."""
        return {
            'n_modes': self.n_modes,
            'cutoff_dim': self.cutoff_dim,
            'circuit_built': self._circuit_built,
            'n_parameters': len(self.trainable_variables)
        }


class PureQuantumCircuit(QuantumCircuitBase):
    """
    Pure quantum circuit with individual gate parameters.
    
    Implements the gradient-safe architecture with modular components
    while maintaining a single SF program.
    """
    
    def __init__(self, n_modes: int, layers: int, cutoff_dim: int = 8):
        """
        Initialize pure quantum circuit.
        
        Args:
            n_modes: Number of quantum modes
            layers: Number of circuit layers
            cutoff_dim: Fock space cutoff dimension
        """
        super().__init__(n_modes, cutoff_dim)
        self.layers = layers
        
        # Initialize parameter manager (modular component)
        from ..parameters.gate_parameters import GateParameterManager
        self.param_manager = GateParameterManager(n_modes, layers)
        
        # Initialize circuit builder (modular component)
        from ..builders.circuit_builder import CircuitBuilder
        self.circuit_builder = CircuitBuilder(self.prog)
        
        logger.info(f"Pure quantum circuit initialized: {layers} layers")
    
    def build_circuit(self) -> None:
        """Build the quantum circuit using modular components."""
        with self.prog.context as q:
            # Get symbolic parameters
            symbolic_params = self.param_manager.create_symbolic_parameters(self.prog)
            
            # Build circuit layers
            for layer in range(self.layers):
                layer_params = symbolic_params[f'layer_{layer}']
                self.circuit_builder.build_layer(q, layer_params, layer)
        
        logger.info(f"Circuit built with {self.param_manager.get_parameter_count()} parameters")
    
    def get_parameter_mapping(self, modulation: Optional[Dict[str, tf.Tensor]] = None) -> Dict[str, tf.Tensor]:
        """Get parameter mapping including optional modulation."""
        # Get the tensor mapping from parameter manager - preserves gradients!
        return self.param_manager.get_parameter_mapping(modulation)
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable gate parameters."""
        return self.param_manager.trainable_variables
    
    def get_parameter_structure(self) -> Dict[str, Dict[str, int]]:
        """Get detailed parameter structure information."""
        return self.param_manager.get_parameter_structure()
