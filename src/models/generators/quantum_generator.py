"""
Quantum Generator using modular architecture.

This module implements a pure quantum generator that uses the modular
components to maintain gradient flow while generating quantum states.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from src.quantum.core import PureQuantumCircuit
from src.quantum.measurements import RawMeasurementExtractor, HolisticMeasurementExtractor
from src.models.transformations import TransformationPair, StaticTransformationMatrix
from src.utils.tensor_utils import safe_tensor_indexing, ensure_tensor

logger = logging.getLogger(__name__)


class QuantumGeneratorBase(ABC):
    """Abstract base class for quantum generators."""
    
    @abstractmethod
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """Generate samples from latent input."""
        pass
    
    @property
    @abstractmethod
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables."""
        pass


class PureQuantumGenerator(QuantumGeneratorBase):
    """
    Pure quantum generator with modular architecture.
    
    Uses ONLY static transformations to ensure ALL learning happens
    through quantum parameters only. No classical neural networks.
    """
    
    def __init__(self, 
                 latent_dim: int = 6,
                 output_dim: int = 2,
                 n_modes: int = 4,
                 layers: int = 2,
                 cutoff_dim: int = 6,
                 measurement_type: str = "raw"):
        """
        Initialize pure quantum generator.
        
        Args:
            latent_dim: Dimension of latent input
            output_dim: Dimension of generated output
            n_modes: Number of quantum modes
            layers: Number of circuit layers
            cutoff_dim: Fock space cutoff
            measurement_type: Type of measurement extraction ('raw', 'holistic')
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # Quantum circuit
        self.circuit = PureQuantumCircuit(
            n_modes=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim
        )
        
        # Measurement extractor
        if measurement_type == "raw":
            self.measurements = RawMeasurementExtractor(
                n_modes=n_modes,
                cutoff_dim=cutoff_dim
            )
        elif measurement_type == "holistic":
            self.measurements = HolisticMeasurementExtractor(
                n_modes=n_modes,
                cutoff_dim=cutoff_dim
            )
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")
        
        # STATIC transformation matrices - NO trainable parameters
        measurement_dim = self.measurements.get_measurement_dim()
        self.transforms = TransformationPair(
            encoder_dim=(latent_dim, n_modes * 3),  # Encode to parameter space
            decoder_dim=(measurement_dim, output_dim),  # Decode measurements
            trainable=False,  # ALWAYS static for pure quantum learning
            name_prefix="generator"
        )
        
        logger.info(f"Pure quantum generator initialized: {latent_dim} → {output_dim}")
        logger.info(f"  Quantum modes: {n_modes}, Layers: {layers}")
        logger.info(f"  Measurement type: {measurement_type}")
        logger.info(f"  Using STATIC transformations (pure quantum learning)")
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate samples from latent input.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, output_dim]
        """
        batch_size = tf.shape(z)[0]
        
        # Encode latent to parameter modulation
        param_encoding = self.transforms.encode(z)
        
        # Execute quantum circuits for each sample - MATCHING DISCRIMINATOR PATTERN
        quantum_states = []
        for i in range(batch_size):
            # Create parameter modulation for this sample - use safe indexing
            sample_encoding = safe_tensor_indexing(param_encoding, i)
            sample_encoding = tf.expand_dims(sample_encoding, 0)  # Add batch dimension
            
            # Get CORRECT parameter mapping keys (not TF variable names)
            base_mapping = self.circuit.get_parameter_mapping()
            param_names = list(base_mapping.keys())
            
            # Create modulation dictionary with CORRECT parameter names
            modulation = {}
            encoding_values = tf.reshape(sample_encoding, [-1])
            for j, name in enumerate(param_names):
                if j < tf.shape(encoding_values)[0]:
                    # Use ensure_tensor for safety
                    modulation[name] = ensure_tensor(encoding_values[j] * 0.001)  # Very small modulation for stability
                else:
                    # Initialize in else branch for TensorFlow AutoGraph compatibility
                    modulation[name] = ensure_tensor(0.0)  # Default to zero modulation
            
            # Execute circuit WITH modulation - this connects input to quantum parameters!
            # This is the KEY fix for gradient flow through quantum circuits
            state = self.circuit.execute(modulation)
            quantum_states.append(state)
        
        # Extract measurements from quantum states
        raw_measurements = self.measurements.extract_measurements(quantum_states)
        
        # Decode measurements to output space
        output = self.transforms.decode(raw_measurements)
        
        return output
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables - ONLY quantum circuit parameters."""
        # Only return quantum circuit variables
        # Transformations are static, measurements have no trainable params
        return self.circuit.trainable_variables
    
    def get_quantum_state_properties(self, z: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Get properties of generated quantum states for analysis.
        
        Args:
            z: Latent input
            
        Returns:
            Dictionary of quantum properties
        """
        # Generate quantum states
        param_encoding = self.transforms.encode(z)
        quantum_states = []
        
        batch_size = tf.shape(z)[0]
        for i in range(batch_size):
            # Use safe tensor indexing
            sample_encoding = safe_tensor_indexing(param_encoding, i)
            sample_encoding = tf.expand_dims(sample_encoding, 0)  # Add batch dimension
            
            # For now, execute circuit without parameter modulation to avoid TF graph issues
            # This allows the analysis to focus on the base quantum circuit behavior
            state = self.circuit.execute({})
            quantum_states.append(state)
        
        # Extract various properties
        properties = {}
        
        # Photon number statistics
        photon_numbers = []
        purities = []
        
        for state in quantum_states:
            ket = state.ket()
            probs = tf.abs(ket) ** 2
            
            # Mean photon number
            n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
            mean_n = tf.reduce_sum(probs * n_vals)
            photon_numbers.append(mean_n)
            
            # Purity
            purity = tf.reduce_sum(probs ** 2)
            purities.append(purity)
        
        properties['mean_photon_number'] = tf.stack(photon_numbers)
        properties['purity'] = tf.stack(purities)
        
        return properties


# Removed HybridQuantumGenerator class entirely
# We only use pure quantum approaches with no classical neural networks
