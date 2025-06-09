"""
TensorFlow Compatibility Module

This module provides compatibility fixes for TensorFlow version issues,
particularly related to complex tensor operations and EagerTensor compatibility.
"""

import tensorflow as tf
import numpy as np
import sys

def _patch_tensorflow_complex_ops():
    """
    Patch TensorFlow complex operations for compatibility with different versions.
    
    This addresses issues with complex conjugate operations and EagerTensor compatibility
    that can occur with certain TensorFlow and Strawberry Fields version combinations.
    """
    try:
        tf_version = tf.__version__
        print(f"TensorFlow version detected: {tf_version}")
    except:
        print("TensorFlow version unknown")
    
    print("Applying TensorFlow complex operations compatibility patches...")
    
    # Enable numpy-like behavior if available
    try:
        if hasattr(tf, 'experimental') and hasattr(tf.experimental, 'numpy'):
            if hasattr(tf.experimental.numpy, 'experimental_enable_numpy_behavior'):
                tf.experimental.numpy.experimental_enable_numpy_behavior()
                print("Enabled TensorFlow numpy-like behavior")
    except Exception as e:
        print(f"Could not enable numpy behavior: {e}")
    
    # Patch EagerTensor to include missing methods if missing
    try:
        import tensorflow.python.framework.ops as tf_ops
        if hasattr(tf_ops, 'EagerTensor'):
            if not hasattr(tf_ops.EagerTensor, 'conj'):
                def eager_conj_method(self):
                    """Add conj method to EagerTensor if missing."""
                    return tf.math.conj(self)
                
                tf_ops.EagerTensor.conj = eager_conj_method
                print("Added conj method to EagerTensor")
            
            if not hasattr(tf_ops.EagerTensor, 'T'):
                @property
                def eager_T_property(self):
                    """Add .T property to EagerTensor if missing."""
                    return tf.transpose(self)
                
                tf_ops.EagerTensor.T = eager_T_property
                print("Added .T property to EagerTensor")
    except ImportError:
        print("Could not patch EagerTensor directly")
    
    # Also patch base Tensor class
    if hasattr(tf, 'Tensor'):
        if not hasattr(tf.Tensor, 'conj'):
            def conj_method(self):
                """Add conj method to Tensor if missing."""
                return tf.math.conj(self)
            
            tf.Tensor.conj = conj_method
            print("Added conj method to tf.Tensor")
        
        # Add transpose attribute .T if missing
        if not hasattr(tf.Tensor, 'T'):
            @property
            def T_property(self):
                """Add .T property to Tensor if missing."""
                return tf.transpose(self)
            
            tf.Tensor.T = T_property
            print("Added .T property to tf.Tensor")
    
    # Test complex conjugate operation
    try:
        test_complex = tf.constant(1 + 2j, dtype=tf.complex64)
        if hasattr(test_complex, 'conj'):
            result = test_complex.conj()
            print("TensorFlow complex conjugate: OK (method)")
        else:
            result = tf.math.conj(test_complex)
            print("TensorFlow complex conjugate: OK (function)")
    except Exception as e:
        print(f"TensorFlow complex conjugate issue: {e}")
        return False
    
    return True

def configure_tensorflow_for_quantum():
    """
    Configure TensorFlow settings optimized for quantum computing simulations.
    """
    # Disable GPU memory growth warnings
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("No GPUs detected, using CPU")
    except Exception as e:
        print(f"GPU configuration warning: {e}")
    
    # Set precision policy for better numerical stability
    try:
        tf.keras.mixed_precision.set_global_policy('float64')
        print("TensorFlow precision policy set to float64")
    except Exception as e:
        print(f"Precision policy warning: {e}")

def ensure_tensor_compatibility(tensor_like):
    """
    Ensure tensor-like objects are compatible with current TensorFlow version.
    
    Args:
        tensor_like: Input that should be converted to a compatible tensor
        
    Returns:
        tf.Tensor: Compatible tensor
    """
    if isinstance(tensor_like, (int, float, complex)):
        return tf.constant(tensor_like, dtype=tf.float64 if isinstance(tensor_like, float) else tf.complex128 if isinstance(tensor_like, complex) else tf.int32)
    elif isinstance(tensor_like, np.ndarray):
        if tensor_like.dtype in [np.float32, np.float64]:
            return tf.constant(tensor_like, dtype=tf.float64)
        elif tensor_like.dtype in [np.complex64, np.complex128]:
            return tf.constant(tensor_like, dtype=tf.complex128)
        else:
            return tf.constant(tensor_like)
    elif tf.is_tensor(tensor_like):
        return tensor_like
    else:
        return tf.constant(tensor_like)

# Apply patches when module is imported
_patch_tensorflow_complex_ops() 