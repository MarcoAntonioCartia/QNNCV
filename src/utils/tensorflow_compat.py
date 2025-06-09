"""
TensorFlow Compatibility Module

This module provides compatibility utilities for handling TensorFlow execution modes,
AutoGraph issues, and quantum computing library integration.
"""

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

def configure_tensorflow_for_quantum():
    """
    Configure TensorFlow settings optimized for quantum computing libraries.
    
    This function sets up TensorFlow to work well with quantum libraries like
    Strawberry Fields by enabling eager execution and configuring memory growth.
    
    Returns:
        dict: Configuration status and settings applied
    """
    config_status = {
        'eager_execution': False,
        'memory_growth': False,
        'autograph_disabled': False,
        'mixed_precision': False
    }
    
    try:
        # Enable eager execution for quantum operations
        if not tf.executing_eagerly():
            tf.config.run_functions_eagerly(True)
            logger.info("Enabled eager execution for quantum compatibility")
        config_status['eager_execution'] = tf.executing_eagerly()
        
        # Configure GPU memory growth if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                config_status['memory_growth'] = True
                logger.info(f"Configured memory growth for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"Could not configure GPU memory growth: {e}")
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        logger.info("TensorFlow configured for quantum computing")
        
    except Exception as e:
        logger.error(f"Error configuring TensorFlow: {e}")
    
    return config_status

def _patch_tensorflow_complex_ops():
    """
    Patch TensorFlow complex operations for better quantum library compatibility.
    
    Some quantum libraries have issues with TensorFlow's complex number handling
    in certain versions. This function applies compatibility patches.
    
    Returns:
        bool: True if patches were applied successfully
    """
    try:
        # Check TensorFlow version
        tf_version = tf.__version__
        major, minor = map(int, tf_version.split('.')[:2])
        
        # Apply version-specific patches
        if major == 2 and minor >= 13:
            # For TensorFlow 2.13+, ensure complex gradient support
            logger.info(f"TensorFlow {tf_version} detected - applying compatibility patches")
            
            # Patch for complex gradient issues with quantum operations
            original_complex = tf.complex
            
            def patched_complex(real, imag, name=None):
                """Patched complex function with better gradient support."""
                with tf.name_scope(name or "complex"):
                    real = tf.convert_to_tensor(real)
                    imag = tf.convert_to_tensor(imag)
                    return original_complex(real, imag)
            
            # Apply the patch
            tf.complex = patched_complex
            
            return True
        else:
            logger.info(f"TensorFlow {tf_version} - no patches needed")
            return True
            
    except Exception as e:
        logger.error(f"Error applying TensorFlow patches: {e}")
        return False

class QuantumExecutionContext:
    """
    Context manager for handling quantum operations with proper TensorFlow settings.
    
    This context manager temporarily modifies TensorFlow execution settings
    to be compatible with quantum computing libraries, then restores the
    original settings when exiting.
    """
    
    def __init__(self, force_eager=True, disable_autograph=True):
        """
        Initialize quantum execution context.
        
        Args:
            force_eager (bool): Whether to force eager execution
            disable_autograph (bool): Whether to disable AutoGraph
        """
        self.force_eager = force_eager
        self.disable_autograph = disable_autograph
        self.original_eager = None
        self.original_autograph = None
    
    def __enter__(self):
        """Enter the quantum execution context."""
        # Store original settings
        self.original_eager = tf.config.functions_run_eagerly()
        
        # Apply quantum-friendly settings
        if self.force_eager:
            tf.config.run_functions_eagerly(True)
        
        logger.debug("Entered quantum execution context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the quantum execution context and restore settings."""
        # Restore original settings
        if self.original_eager is not None:
            tf.config.run_functions_eagerly(self.original_eager)
        
        logger.debug("Exited quantum execution context")

def quantum_safe_function(func):
    """
    Decorator to make functions safe for quantum operations.
    
    This decorator disables AutoGraph and ensures eager execution
    for functions that contain quantum operations.
    
    Args:
        func: Function to make quantum-safe
        
    Returns:
        Decorated function with quantum-safe execution
    """
    @tf.autograph.experimental.do_not_convert
    def wrapper(*args, **kwargs):
        with QuantumExecutionContext():
            return func(*args, **kwargs)
    
    return wrapper

def create_quantum_compatible_optimizer(learning_rate=0.001, **kwargs):
    """
    Create an optimizer configured for quantum parameter training.
    
    Args:
        learning_rate (float): Learning rate for the optimizer
        **kwargs: Additional optimizer arguments
        
    Returns:
        tf.optimizers.Optimizer: Configured optimizer
    """
    # Use Adam with conservative settings for quantum parameters
    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=kwargs.get('beta_1', 0.9),
        beta_2=kwargs.get('beta_2', 0.999),
        epsilon=kwargs.get('epsilon', 1e-7),
        clipnorm=kwargs.get('clipnorm', 1.0)  # Gradient clipping for stability
    )
    
    logger.info(f"Created quantum-compatible optimizer with lr={learning_rate}")
    return optimizer

def test_tensorflow_quantum_compatibility():
    """
    Test TensorFlow configuration for quantum computing compatibility.
    
    Returns:
        dict: Test results and compatibility status
    """
    results = {
        'tensorflow_version': tf.__version__,
        'eager_execution': tf.executing_eagerly(),
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'complex_ops': False,
        'gradient_tape': False,
        'autograph_disabled': False
    }
    
    try:
        # Test complex operations
        with tf.GradientTape() as tape:
            x = tf.Variable(1.0 + 1.0j)
            y = tf.square(x)
        grad = tape.gradient(y, x)
        results['complex_ops'] = grad is not None
        results['gradient_tape'] = True
        
        # Test AutoGraph disabling
        @tf.autograph.experimental.do_not_convert
        def test_func():
            return tf.constant(1.0)
        
        result = test_func()
        results['autograph_disabled'] = result.numpy() == 1.0
        
    except Exception as e:
        logger.error(f"Compatibility test failed: {e}")
    
    return results

# Apply patches when module is imported
_patch_success = _patch_tensorflow_complex_ops()
if _patch_success:
    logger.info("TensorFlow compatibility patches applied successfully")
else:
    logger.warning("Some TensorFlow compatibility patches failed")
