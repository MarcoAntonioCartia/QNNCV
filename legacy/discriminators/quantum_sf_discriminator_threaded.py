"""
Threaded Quantum SF Discriminator with Universal Threading Support

This implementation extends the quantum_sf_discriminator.py with the universal
threading capabilities from sf_threading_fixed.py for significant performance improvements.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Optional, Dict, Any

# Initialize logger first
logger = logging.getLogger(__name__)

# Import threading utilities
try:
    from ...utils.sf_threading_fixed import UniversalSFThreadingMixin
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, '..', '..')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from utils.sf_threading_fixed import UniversalSFThreadingMixin
    except ImportError as e:
        logger.error(f"Failed to import threading utilities: {e}")
        raise

# Import the base discriminator
try:
    from .quantum_sf_discriminator import QuantumSFDiscriminator
except ImportError:
    from quantum_sf_discriminator import QuantumSFDiscriminator

class ThreadedQuantumSFDiscriminator(UniversalSFThreadingMixin, QuantumSFDiscriminator):
    """
    Threaded Quantum SF Discriminator with up to 8.9x performance improvement.
    
    This class combines the quantum encoding capabilities of QuantumSFDiscriminator
    with the universal threading system for optimized batch processing.
    """
    
    def __init__(self, n_modes=2, input_dim=2, layers=3, cutoff_dim=6,
                 encoding_strategy='coherent_state', config=None,
                 feature_extraction='multi_mode', enable_batch_processing=True,
                 enable_threading=True, max_threads=None):
        """
        Initialize threaded quantum discriminator.
        
        Args:
            n_modes (int): Number of quantum modes
            input_dim (int): Dimension of input data
            layers (int): Number of quantum neural network layers
            cutoff_dim (int): Fock space cutoff for simulation
            encoding_strategy (str): Quantum encoding strategy to use
            config (QuantumGANConfig): Optional configuration object
            feature_extraction (str): Feature extraction strategy
            enable_batch_processing (bool): Enable batch processing optimizations
            enable_threading (bool): Enable threading optimizations
            max_threads (int): Maximum number of threads (auto-detect if None)
        """
        # Initialize with threading support
        super().__init__(
            n_modes=n_modes,
            input_dim=input_dim,
            layers=layers,
            cutoff_dim=cutoff_dim,
            encoding_strategy=encoding_strategy,
            config=config,
            feature_extraction=feature_extraction,
            enable_batch_processing=enable_batch_processing,
            enable_threading=enable_threading,
            max_threads=max_threads
        )
        
        logger.info(f"ThreadedQuantumSFDiscriminator initialized with threading: {enable_threading}")
        if enable_threading and self.threading_manager:
            logger.info(f"  - Max threads: {self.threading_manager.max_threads}")
            logger.info(f"  - GPU available: {self.threading_manager.has_gpu}")
    
    def discriminate(self, x):
        """
        Discriminate samples with optional threading optimization.
        
        Args:
            x (tensor): Input samples [batch_size, input_dim]
            
        Returns:
            probabilities (tensor): Probability of being real [batch_size, 1]
        """
        # If threading is enabled and batch size is suitable, use threaded execution
        if self.enable_threading and self.threading_manager is not None:
            batch_size = tf.shape(x)[0]
            
            # Choose strategy based on batch size
            if batch_size >= 4:  # Threading beneficial for batch_size >= 4
                logger.debug(f"Using threaded discrimination for batch_size={batch_size}")
                return self.execute_threaded(x, method_name='discriminate', strategy='auto')
        
        # Otherwise, use the parent class implementation
        return super().discriminate(x)
    
    def discriminate_batch_optimized(self, x, strategy='auto'):
        """
        Discriminate samples with explicit threading strategy selection.
        
        Args:
            x (tensor): Input samples [batch_size, input_dim]
            strategy (str): Threading strategy ('sequential', 'cpu_batch', 'threading', 'gpu_batch', 'auto')
            
        Returns:
            probabilities (tensor): Probability of being real [batch_size, 1]
        """
        if not self.enable_threading or self.threading_manager is None:
            logger.warning("Threading not enabled, falling back to standard discrimination")
            return super().discriminate(x)
        
        return self.execute_threaded(x, method_name='discriminate', strategy=strategy)
    
    def benchmark_discrimination(self, test_batch_sizes=None):
        """
        Benchmark discrimination performance with different threading strategies.
        
        Args:
            test_batch_sizes (list): Batch sizes to test (default: [1, 4, 8, 16, 32])
            
        Returns:
            dict: Performance metrics for each batch size and strategy
        """
        if not self.enable_threading:
            return {"error": "Threading not enabled"}
        
        # Generate test inputs
        test_batch_sizes = test_batch_sizes or [1, 4, 8, 16, 32]
        max_batch = max(test_batch_sizes)
        test_inputs = tf.random.normal([max_batch, self.input_dim])
        
        return self.benchmark_threading_performance(
            test_inputs=test_inputs,
            method_name='discriminate',
            test_batch_sizes=test_batch_sizes
        )
    
    def get_threading_stats(self):
        """Get current threading performance statistics."""
        if self.threading_manager:
            return self.threading_manager.get_performance_report()
        return {"error": "Threading not enabled"}


def create_threaded_discriminator(**kwargs):
    """
    Factory function to create a threaded quantum discriminator.
    
    Args:
        **kwargs: Arguments for ThreadedQuantumSFDiscriminator
        
    Returns:
        ThreadedQuantumSFDiscriminator instance
    """
    return ThreadedQuantumSFDiscriminator(**kwargs)


def test_threaded_discriminator():
    """Test the threaded quantum discriminator."""
    print("\n" + "="*60)
    print("TESTING THREADED QUANTUM SF DISCRIMINATOR")
    print("="*60)
    
    try:
        # Create threaded discriminator
        discriminator = ThreadedQuantumSFDiscriminator(
            n_modes=2,
            input_dim=2,
            layers=2,
            cutoff_dim=6,
            enable_threading=True
        )
        
        print(f"Threaded discriminator created successfully")
        print(f"Threading enabled: {discriminator.enable_threading}")
        
        # Test standard discrimination
        x_test = tf.random.normal([16, 2])
        probabilities = discriminator.discriminate(x_test)
        print(f"\nStandard discrimination test: {probabilities.shape}")
        print(f"Probability range: [{tf.reduce_min(probabilities):.3f}, {tf.reduce_max(probabilities):.3f}]")
        
        # Test different strategies
        print("\nTesting different threading strategies:")
        strategies = ['sequential', 'cpu_batch', 'threading', 'auto']
        
        for strategy in strategies:
            try:
                probs = discriminator.discriminate_batch_optimized(x_test, strategy=strategy)
                print(f"  {strategy}: Success - shape {probs.shape}")
            except Exception as e:
                print(f"  {strategy}: Failed - {e}")
        
        # Benchmark performance
        print("\nRunning performance benchmark...")
        benchmark_results = discriminator.benchmark_discrimination(test_batch_sizes=[1, 4, 8, 16])
        
        # Display threading stats
        print("\nThreading statistics:")
        stats = discriminator.get_threading_stats()
        for key, value in stats.items():
            if key != 'strategy_usage':
                print(f"  {key}: {value}")
        
        return discriminator
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the threaded discriminator
    test_threaded_discriminator()
