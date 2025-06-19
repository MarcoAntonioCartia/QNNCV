"""
Utility functions for tensor operations.

This module provides helper functions for working with tensors,
particularly for handling compatibility issues between TensorFlow
and NumPy arrays.
"""

import tensorflow as tf
import numpy as np
from typing import Any, List, Tuple, Union

def safe_tensor_indexing(tensor: tf.Tensor, indices: Any) -> tf.Tensor:
    """
    Safely index a TensorFlow tensor with various index types.
    
    This function handles the case where indices might be a NumPy array,
    which is not directly supported for TensorFlow tensor indexing.
    
    Args:
        tensor: TensorFlow tensor to index
        indices: Indices to use (int, slice, list, numpy array, or tensor)
        
    Returns:
        Indexed tensor
    """
    if isinstance(indices, np.ndarray):
        # Convert numpy array to TensorFlow tensor
        indices_tensor = tf.constant(indices, dtype=tf.int32)
        return tf.gather(tensor, indices_tensor)
    elif isinstance(indices, list):
        # Convert list to TensorFlow tensor
        indices_tensor = tf.constant(indices, dtype=tf.int32)
        return tf.gather(tensor, indices_tensor)
    elif isinstance(indices, tf.Tensor):
        # Use tf.gather for tensor indices
        return tf.gather(tensor, indices)
    else:
        # Use standard indexing for other types (int, slice)
        return tensor[indices]

def batch_gather(params: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """
    Gather slices from params according to indices with batch support.
    
    This is similar to tf.gather but works with batched inputs.
    
    Args:
        params: Tensor from which to gather values
        indices: Tensor of indices
        
    Returns:
        Gathered tensor
    """
    return tf.gather(params, indices, batch_dims=1)

def ensure_tensor(value: Any, dtype=None) -> tf.Tensor:
    """
    Ensure that a value is a TensorFlow tensor.
    
    Args:
        value: Value to convert (tensor, numpy array, list, scalar)
        dtype: Optional dtype for the tensor
        
    Returns:
        TensorFlow tensor
    """
    if isinstance(value, tf.Tensor):
        if dtype is not None and value.dtype != dtype:
            return tf.cast(value, dtype)
        return value
    else:
        return tf.convert_to_tensor(value, dtype=dtype)

def safe_reduce_mean(tensor: tf.Tensor, axis=None) -> tf.Tensor:
    """
    Safely compute mean of tensor, handling empty tensors.
    
    Args:
        tensor: Input tensor
        axis: Axis along which to compute mean
        
    Returns:
        Mean tensor
    """
    # Check if tensor is empty
    is_empty = tf.equal(tf.size(tensor), 0)
    
    # If empty, return 0, otherwise compute mean
    return tf.cond(
        is_empty,
        lambda: tf.constant(0.0, dtype=tensor.dtype),
        lambda: tf.reduce_mean(tensor, axis=axis)
    )

def safe_random_normal(shape: List[int], mean=0.0, stddev=1.0, dtype=tf.float32) -> tf.Tensor:
    """
    Safely generate random normal values, handling empty shapes.
    
    Args:
        shape: Shape of the output tensor
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Data type of the output
        
    Returns:
        Random normal tensor
    """
    # Convert shape to tensor
    shape_tensor = tf.constant(shape, dtype=tf.int32)
    
    # Check if any dimension is zero
    is_empty = tf.reduce_any(tf.equal(shape_tensor, 0))
    
    # If empty shape, return empty tensor, otherwise generate random values
    return tf.cond(
        is_empty,
        lambda: tf.zeros(shape, dtype=dtype),
        lambda: tf.random.normal(shape, mean=mean, stddev=stddev, dtype=dtype)
    )
