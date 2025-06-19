"""
Transformation Matrix Manager

This module manages the A and T transformation matrices used in
quantum generators and discriminators for dimension matching.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TransformationMatrixBase(ABC):
    """
    Abstract base class for transformation matrices.
    
    Defines the interface for transformation matrices used in
    quantum neural networks.
    """
    
    @abstractmethod
    def transform(self, x: tf.Tensor) -> tf.Tensor:
        """Apply transformation to input tensor."""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output dimension of transformation."""
        pass
    
    @property
    @abstractmethod
    def trainable_variables(self) -> list:
        """Return trainable variables."""
        pass


class StaticTransformationMatrix(TransformationMatrixBase):
    """
    Static (non-trainable) transformation matrix.
    
    Used when all learning should happen through quantum parameters only.
    """
    
    def __init__(self, input_dim: int, output_dim: int, name: str = "static_transform"):
        """
        Initialize static transformation matrix.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            name: Name for the transformation
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        
        # Create static matrix
        self.matrix = self._create_static_matrix()
        
        logger.info(f"Static transformation matrix '{name}': {input_dim} → {output_dim}")
    
    def _create_static_matrix(self) -> tf.Tensor:
        """Create well-conditioned static matrix."""
        if self.input_dim <= self.output_dim:
            # Expand dimensions
            base_matrix = np.random.randn(self.input_dim, self.input_dim).astype(np.float32)
            u, _, vh = np.linalg.svd(base_matrix)
            orthogonal_base = u @ vh
            
            if self.output_dim > self.input_dim:
                # Pad with zeros
                padding = np.zeros((self.input_dim, self.output_dim - self.input_dim), dtype=np.float32)
                matrix = np.concatenate([orthogonal_base, padding], axis=1)
            else:
                matrix = orthogonal_base
        else:
            # Reduce dimensions
            base_matrix = np.random.randn(self.input_dim, self.input_dim).astype(np.float32)
            u, _, vh = np.linalg.svd(base_matrix)
            orthogonal_base = u @ vh
            matrix = orthogonal_base[:, :self.output_dim]
        
        # Scale for reasonable range
        matrix = matrix * 0.5
        
        # Convert to constant tensor with float32 dtype
        return tf.constant(matrix, dtype=tf.float32, name=f"{self.name}_matrix")
    
    def transform(self, x: tf.Tensor) -> tf.Tensor:
        """Apply static transformation."""
        return tf.matmul(x, self.matrix)
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim
    
    @property
    def trainable_variables(self) -> list:
        """Return empty list (no trainable variables)."""
        return []


class TrainableTransformationMatrix(TransformationMatrixBase):
    """
    Trainable transformation matrix with optional bias.
    
    Used when classical transformation parameters should be learned.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 use_bias: bool = True, 
                 initialization: str = "orthogonal",
                 name: str = "trainable_transform"):
        """
        Initialize trainable transformation matrix.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            use_bias: Whether to include bias term
            initialization: Initialization method ('orthogonal', 'xavier', 'random')
            name: Name for the transformation
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.name = name
        
        # Create trainable matrix
        self.matrix = self._create_trainable_matrix(initialization)
        
        # Create bias if requested
        if use_bias:
            self.bias = tf.Variable(
                tf.zeros([output_dim]),
                name=f"{name}_bias",
                trainable=True
            )
        else:
            self.bias = None
        
        logger.info(f"Trainable transformation matrix '{name}': {input_dim} → {output_dim} (bias={use_bias})")
    
    def _create_trainable_matrix(self, initialization: str) -> tf.Variable:
        """Create trainable matrix with specified initialization."""
        if initialization == "orthogonal":
            # Start with orthogonal initialization
            if self.input_dim == self.output_dim:
                init_value = tf.initializers.Orthogonal()(shape=[self.input_dim, self.output_dim])
            else:
                # For non-square matrices, use truncated orthogonal
                max_dim = max(self.input_dim, self.output_dim)
                temp = tf.initializers.Orthogonal()(shape=[max_dim, max_dim])
                init_value = temp[:self.input_dim, :self.output_dim]
            init_value = init_value * 0.5
        
        elif initialization == "xavier":
            # Xavier/Glorot initialization
            init_value = tf.initializers.GlorotUniform()(shape=[self.input_dim, self.output_dim])
        
        else:  # random
            # Random normal initialization
            init_value = tf.random.normal([self.input_dim, self.output_dim], stddev=0.1)
        
        return tf.Variable(
            init_value,
            name=f"{self.name}_matrix",
            trainable=True
        )
    
    def transform(self, x: tf.Tensor) -> tf.Tensor:
        """Apply trainable transformation."""
        output = tf.matmul(x, self.matrix)
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim
    
    @property
    def trainable_variables(self) -> list:
        """Return trainable variables."""
        variables = [self.matrix]
        if self.bias is not None:
            variables.append(self.bias)
        return variables
    
    def compute_regularization(self, reg_type: str = "orthogonal") -> tf.Tensor:
        """
        Compute regularization loss for the transformation.
        
        Args:
            reg_type: Type of regularization ('orthogonal', 'l2', 'condition')
            
        Returns:
            Regularization loss
        """
        if reg_type == "orthogonal":
            # Encourage orthogonality
            if self.input_dim <= self.output_dim:
                gram = tf.matmul(tf.transpose(self.matrix), self.matrix)
                target = tf.eye(self.input_dim)
            else:
                gram = tf.matmul(self.matrix, tf.transpose(self.matrix))
                target = tf.eye(self.output_dim)
            return tf.reduce_mean(tf.square(gram - target))
        
        elif reg_type == "l2":
            # L2 regularization
            return tf.reduce_mean(tf.square(self.matrix))
        
        elif reg_type == "condition":
            # Encourage good conditioning
            s = tf.linalg.svd(self.matrix, compute_uv=False)
            condition_number = s[0] / (s[-1] + 1e-6)
            return tf.square(condition_number - 1.0)
        
        else:
            return tf.constant(0.0)


class TransformationPair:
    """
    Manages a pair of transformations (e.g., encoder and decoder).
    
    Useful for generator architectures that need both T and A^(-1) transformations.
    """
    
    def __init__(self, 
                 encoder_dim: Tuple[int, int],
                 decoder_dim: Tuple[int, int],
                 trainable: bool = True,
                 name_prefix: str = "transform"):
        """
        Initialize transformation pair.
        
        Args:
            encoder_dim: (input_dim, output_dim) for encoder
            decoder_dim: (input_dim, output_dim) for decoder
            trainable: Whether transformations are trainable
            name_prefix: Prefix for transformation names
        """
        self.trainable = trainable
        
        if trainable:
            self.encoder = TrainableTransformationMatrix(
                encoder_dim[0], encoder_dim[1],
                name=f"{name_prefix}_encoder"
            )
            self.decoder = TrainableTransformationMatrix(
                decoder_dim[0], decoder_dim[1],
                name=f"{name_prefix}_decoder"
            )
        else:
            self.encoder = StaticTransformationMatrix(
                encoder_dim[0], encoder_dim[1],
                name=f"{name_prefix}_encoder"
            )
            self.decoder = StaticTransformationMatrix(
                decoder_dim[0], decoder_dim[1],
                name=f"{name_prefix}_decoder"
            )
        
        logger.info(f"Transformation pair created: {encoder_dim} → {decoder_dim} (trainable={trainable})")
    
    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Apply encoder transformation."""
        return self.encoder.transform(x)
    
    def decode(self, x: tf.Tensor) -> tf.Tensor:
        """Apply decoder transformation."""
        return self.decoder.transform(x)
    
    @property
    def trainable_variables(self) -> list:
        """Return all trainable variables."""
        return self.encoder.trainable_variables + self.decoder.trainable_variables
    
    def compute_regularization(self, reg_type: str = "orthogonal") -> tf.Tensor:
        """Compute regularization for both transformations."""
        reg_loss = tf.constant(0.0)
        
        if self.trainable:
            reg_loss += self.encoder.compute_regularization(reg_type)
            reg_loss += self.decoder.compute_regularization(reg_type)
        
        return reg_loss


class AdaptiveTransformationMatrix(TransformationMatrixBase):
    """
    Adaptive transformation that can switch between static and trainable modes.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 start_static: bool = True,
                 name: str = "adaptive_transform"):
        """
        Initialize adaptive transformation matrix.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            start_static: Whether to start in static mode
            name: Name for the transformation
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.is_static = start_static
        
        # Create both static and trainable versions
        self.static_transform = StaticTransformationMatrix(input_dim, output_dim, f"{name}_static")
        self.trainable_transform = TrainableTransformationMatrix(input_dim, output_dim, name=f"{name}_trainable")
        
        logger.info(f"Adaptive transformation matrix '{name}': {input_dim} → {output_dim} (start_static={start_static})")
    
    def set_mode(self, static: bool):
        """Switch between static and trainable modes."""
        self.is_static = static
        logger.info(f"Adaptive transformation '{self.name}' mode: {'static' if static else 'trainable'}")
    
    def transform(self, x: tf.Tensor) -> tf.Tensor:
        """Apply transformation based on current mode."""
        if self.is_static:
            return self.static_transform.transform(x)
        else:
            return self.trainable_transform.transform(x)
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim
    
    @property
    def trainable_variables(self) -> list:
        """Return trainable variables based on current mode."""
        if self.is_static:
            return []
        else:
            return self.trainable_transform.trainable_variables
