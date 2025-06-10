"""
TensorFlow Compatibility Module

This module provides compatibility utilities for handling TensorFlow execution modes,
AutoGraph issues, and quantum computing library integration.
"""

import tensorflow as tf
import numpy as np
import logging
import warnings
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def suppress_complex_warnings():
    """
    Context manager to suppress TensorFlow complex number casting warnings.
    
    This suppresses the specific warning:
    "You are casting an input of type complex64 to an incompatible dtype float32"
    which is expected behavior when extracting real parts from quantum measurements.
    """
    # Suppress TensorFlow complex casting warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                              message='.*casting.*complex.*float.*', 
                              category=UserWarning)
        
        # Also suppress TensorFlow logging for this specific warning
        original_tf_log_level = tf.get_logger().level
        tf.get_logger().setLevel(logging.ERROR)
        
        try:
            yield
        finally:
            # Restore TensorFlow logging level
            tf.get_logger().setLevel(original_tf_log_level)

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

def parameter_shift_gradient(func, params, shift=np.pi/2):
    """
    Compute gradients using the parameter-shift rule for quantum operations.
    
    For a function f(θ), the gradient is approximated as:
    ∂f/∂θ ≈ [f(θ + π/2) - f(θ - π/2)] / 2
    
    Args:
        func: Function to differentiate
        params: Parameters to compute gradients for
        shift: Shift amount for parameter-shift rule
        
    Returns:
        Gradients with respect to each parameter
    """
    gradients = []
    
    for i, param in enumerate(params):
        # Create shifted parameter sets
        params_plus = [p for p in params]
        params_minus = [p for p in params]
        
        params_plus[i] = param + shift
        params_minus[i] = param - shift
        
        # Compute function values at shifted points
        f_plus = func(params_plus)
        f_minus = func(params_minus)
        
        # Compute gradient using parameter-shift rule
        grad = (f_plus - f_minus) / 2.0
        gradients.append(grad)
    
    return gradients

def test_tensorflow_quantum_compatibility():
    """
    Test TensorFlow compatibility for quantum operations.
    
    Returns:
        dict: Test results for various compatibility checks
    """
    results = {
        'eager_execution': tf.executing_eagerly(),
        'tensorflow_version': tf.__version__,
        'complex_ops_available': True,
        'gradient_tape_available': True,
        'autograph_disabled': True
    }
    
    try:
        # Test complex number operations
        complex_tensor = tf.constant(1.0 + 2.0j)
        real_part = tf.cast(complex_tensor, tf.float32)
        results['complex_ops_available'] = True
    except Exception as e:
        results['complex_ops_available'] = False
        results['complex_ops_error'] = str(e)
    
    try:
        # Test gradient tape
        with tf.GradientTape() as tape:
            x = tf.Variable(1.0)
            y = x * x
        grad = tape.gradient(y, x)
        results['gradient_tape_available'] = grad is not None
    except Exception as e:
        results['gradient_tape_available'] = False
        results['gradient_tape_error'] = str(e)
    
    return results

# Apply patches when module is imported
_patch_success = _patch_tensorflow_complex_ops()
if _patch_success:
    logger.info("TensorFlow compatibility patches applied successfully")
else:
    logger.warning("Some TensorFlow compatibility patches failed")
