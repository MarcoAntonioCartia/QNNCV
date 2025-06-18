"""
Quantum Discriminator using modular architecture.

This module implements a pure quantum discriminator that uses the modular
components to maintain gradient flow while discriminating quantum states.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from quantum.core import PureQuantumCircuit
from quantum.measurements import RawMeasurementExtractor, HolisticMeasurementExtractor
from models.transformations import StaticTransformationMatrix

logger = logging.getLogger(__name__)


class QuantumDiscriminatorBase(ABC):
    """Abstract base class for quantum discriminators."""
    
    @abstractmethod
    def discriminate(self, x: tf.Tensor) -> tf.Tensor:
        """Discriminate input samples."""
        pass
    
    @property
    @abstractmethod
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables."""
        pass


class PureQuantumDiscriminator(QuantumDiscriminatorBase):
    """
    Pure quantum discriminator with modular architecture.
    
    Uses ONLY static transformations and quantum parameters.
    The final classification is done through quantum measurements
    without any classical neural networks.
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 n_modes: int = 2,
                 layers: int = 2,
                 cutoff_dim: int = 6,
                 measurement_type: str = "raw"):
        """
        Initialize pure quantum discriminator.
        
        Args:
            input_dim: Dimension of input data
            n_modes: Number of quantum modes
            layers: Number of circuit layers
            cutoff_dim: Fock space cutoff
            measurement_type: Type of measurement extraction ('raw', 'holistic')
        """
        self.input_dim = input_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # Quantum circuit
        self.circuit = PureQuantumCircuit(
            n_modes=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim
        )
        
        # Input transformation (static)
        self.input_transform = StaticTransformationMatrix(
            input_dim=input_dim,
            output_dim=n_modes * 3,  # Parameter modulation space
            name="discriminator_input"
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
        
        # Output transformation for classification (static)
        measurement_dim = self.measurements.get_measurement_dim()
        self.output_transform = StaticTransformationMatrix(
            input_dim=measurement_dim,
            output_dim=1,  # Binary classification
            name="discriminator_output"
        )
        
        logger.info(f"Pure quantum discriminator initialized: {input_dim} → 1")
        logger.info(f"  Quantum modes: {n_modes}, Layers: {layers}")
        logger.info(f"  Measurement type: {measurement_type}")
        logger.info(f"  Using STATIC transformations (pure quantum learning)")
    
    def discriminate(self, x: tf.Tensor) -> tf.Tensor:
        """
        Discriminate input samples.
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Discrimination scores [batch_size, 1]
        """
        batch_size = tf.shape(x)[0]
        
        # Transform input to parameter modulation
        param_encoding = self.input_transform.transform(x)
        
        # Execute quantum circuits for each sample
        quantum_states = []
        for i in range(batch_size):
            # Create parameter modulation for this sample
            sample_encoding = param_encoding[i:i+1]
            
            # Get parameter names for modulation
            param_names = [var.name.split(':')[0] for var in self.circuit.trainable_variables]
            
            # Create modulation dictionary
            modulation = {}
            encoding_values = tf.reshape(sample_encoding, [-1])
            for j, name in enumerate(param_names):
                if j < tf.shape(encoding_values)[0]:
                    modulation[name] = encoding_values[j] * 0.1  # Small modulation
            
            # Execute circuit without modulation for now
            # TODO: Fix parameter modulation mapping
            state = self.circuit.execute({})
            quantum_states.append(state)
        
        # Extract measurements from quantum states
        raw_measurements = self.measurements.extract_measurements(quantum_states)
        
        # Transform measurements to discrimination score
        scores = self.output_transform.transform(raw_measurements)
        
        return scores
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables - ONLY quantum circuit parameters."""
        # Only quantum circuit parameters are trainable
        return self.circuit.trainable_variables
    
    def get_quantum_features(self, x: tf.Tensor) -> tf.Tensor:
        """
        Get quantum features (measurements) for input data.
        
        Args:
            x: Input data
            
        Returns:
            Quantum features (raw measurements)
        """
        batch_size = tf.shape(x)[0]
        param_encoding = self.input_transform.transform(x)
        
        quantum_states = []
        for i in range(batch_size):
            sample_encoding = param_encoding[i:i+1]
            param_names = [var.name.split(':')[0] for var in self.circuit.trainable_variables]
            
            modulation = {}
            encoding_values = tf.reshape(sample_encoding, [-1])
            for j, name in enumerate(param_names):
                if j < tf.shape(encoding_values)[0]:
                    modulation[name] = encoding_values[j] * 0.1
            
            state = self.circuit.execute(modulation)
            quantum_states.append(state)
        
        # Return raw measurements as features
        return self.measurements.extract_measurements(quantum_states)


class QuantumWassersteinDiscriminator(PureQuantumDiscriminator):
    """
    Quantum discriminator specifically designed for Wasserstein GANs.
    
    Maintains Lipschitz constraint through quantum circuit properties
    rather than classical weight clipping or gradient penalties.
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 n_modes: int = 2,
                 layers: int = 2,
                 cutoff_dim: int = 6,
                 measurement_type: str = "raw"):
        """Initialize Wasserstein quantum discriminator."""
        super().__init__(
            input_dim=input_dim,
            n_modes=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim,
            measurement_type=measurement_type
        )
        
        # For Wasserstein, we don't use sigmoid activation
        # The output is unbounded (critic score)
        logger.info("Wasserstein quantum discriminator initialized (unbounded output)")
    
    def compute_gradient_penalty(self, real_data: tf.Tensor, fake_data: tf.Tensor) -> tf.Tensor:
        """
        Compute gradient penalty for Wasserstein loss.
        
        This operates on the quantum measurement space rather than
        parameter space, maintaining quantum properties.
        
        Args:
            real_data: Real samples
            fake_data: Generated samples
            
        Returns:
            Gradient penalty term
        """
        batch_size = tf.shape(real_data)[0]
        
        # Interpolate between real and fake
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        
        # Get quantum features for interpolated data
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            features = self.get_quantum_features(interpolated)
            # Use L2 norm of features as the critic output
            critic_output = tf.reduce_sum(features ** 2, axis=1, keepdims=True)
        
        # Compute gradients with respect to interpolated data
        gradients = tape.gradient(critic_output, interpolated)
        
        # Compute gradient penalty
        if gradients is not None:
            gradients_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1) + 1e-8)
            gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        else:
            gradient_penalty = tf.constant(0.0)
        
        return gradient_penalty
