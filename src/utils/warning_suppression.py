"""
Warning suppression utility for quantum machine learning training.

This module provides utilities to suppress common warnings that occur during
quantum ML training, particularly TensorFlow warnings about complex number
casting and other verbose outputs that can bury important training information.
"""

import warnings
import logging
import os
import sys
from contextlib import contextmanager
from typing import List, Optional

# Configure logging to suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning messages

def suppress_tensorflow_warnings():
    """
    Suppress common TensorFlow warnings that occur during quantum training.
    
    This function suppresses:
    - Complex64 to float32 casting warnings
    - Deprecated function warnings
    - oneDNN optimization messages
    - Keras deprecation warnings
    """
    import tensorflow as tf
    
    # Suppress TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    
    # Suppress specific warning categories
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
    
    # Suppress specific warning messages
    warnings.filterwarnings('ignore', message='.*complex64.*incompatible.*float32.*')
    warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
    warnings.filterwarnings('ignore', message='.*sparse_softmax_cross_entropy.*deprecated.*')
    warnings.filterwarnings('ignore', message='.*executing_eagerly_outside_functions.*deprecated.*')

def suppress_strawberry_fields_warnings():
    """
    Suppress common Strawberry Fields warnings during quantum circuit execution.
    """
    # Suppress SF-specific warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='strawberryfields')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='strawberryfields')
    
    # Suppress numerical precision warnings
    warnings.filterwarnings('ignore', message='.*numerical precision.*')
    warnings.filterwarnings('ignore', message='.*cutoff dimension.*')

def suppress_numpy_warnings():
    """
    Suppress common NumPy warnings during scientific computing.
    """
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
    warnings.filterwarnings('ignore', message='.*divide by zero encountered.*')

def suppress_all_quantum_warnings():
    """
    Suppress all common warnings that occur during quantum ML training.
    
    This is a convenience function that applies all warning suppressions.
    """
    suppress_tensorflow_warnings()
    suppress_strawberry_fields_warnings()
    suppress_numpy_warnings()
    
    # General warning suppression
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

@contextmanager
def clean_training_output(suppress_warnings: bool = True, 
                         show_progress: bool = True,
                         custom_suppressions: Optional[List[str]] = None):
    """
    Context manager for clean training output.
    
    Args:
        suppress_warnings (bool): Whether to suppress warnings
        show_progress (bool): Whether to show training progress
        custom_suppressions (List[str]): Additional warning patterns to suppress
        
    Usage:
        with clean_training_output():
            # Your training code here
            trainer.train(data, epochs=100)
    """
    if suppress_warnings:
        # Store original warning filters
        original_filters = warnings.filters.copy()
        
        try:
            # Apply suppressions
            suppress_all_quantum_warnings()
            
            # Apply custom suppressions if provided
            if custom_suppressions:
                for pattern in custom_suppressions:
                    warnings.filterwarnings('ignore', message=pattern)
            
            # Redirect stderr temporarily if needed
            if not show_progress:
                original_stderr = sys.stderr
                sys.stderr = open(os.devnull, 'w')
            
            yield
            
        finally:
            # Restore original warning filters
            warnings.filters = original_filters
            
            # Restore stderr
            if not show_progress:
                sys.stderr.close()
                sys.stderr = original_stderr
    else:
        yield

class QuantumTrainingLogger:
    """
    Custom logger for quantum training that filters out noise.
    """
    
    def __init__(self, name: str = "quantum_training", level: int = logging.INFO):
        """
        Initialize the quantum training logger.
        
        Args:
            name (str): Logger name
            level (int): Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create clean formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

def setup_clean_environment():
    """
    Set up a clean environment for quantum training.
    
    This function should be called at the beginning of training scripts
    to ensure clean output.
    """
    # Suppress all quantum warnings
    suppress_all_quantum_warnings()
    
    # Set environment variables for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Configure matplotlib to be non-interactive
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass
    
    print("Clean training environment initialized")
    print("Warnings suppressed, ready for quantum training")

# Convenience function for quick setup
def enable_clean_training():
    """
    Quick setup for clean quantum training output.
    
    Call this at the start of your training script:
    
    from src.utils.warning_suppression import enable_clean_training
    enable_clean_training()
    """
    setup_clean_environment()

def test_warning_suppression():
    """Test the warning suppression functionality."""
    print("\n" + "="*60)
    print("TESTING WARNING SUPPRESSION UTILITY")
    print("="*60)
    
    print("Testing warning suppression...")
    
    # Test without suppression
    print("\n1. Without suppression (should show warnings):")
    import tensorflow as tf
    
    # This should generate warnings
    x = tf.constant([1.0 + 2.0j], dtype=tf.complex64)
    y = tf.cast(x, tf.float32)  # Should warn about discarding imaginary part
    
    print("\n2. With suppression (should be clean):")
    with clean_training_output():
        # Same operation, but warnings should be suppressed
        x = tf.constant([1.0 + 2.0j], dtype=tf.complex64)
        y = tf.cast(x, tf.float32)
        print("Complex casting completed without warnings")
    
    print("\n3. Testing custom logger:")
    logger = QuantumTrainingLogger("test_logger")
    logger.info("This is a clean info message")
    logger.warning("This is a clean warning message")
    
    print("\nWarning suppression test completed")

if __name__ == "__main__":
    test_warning_suppression()
