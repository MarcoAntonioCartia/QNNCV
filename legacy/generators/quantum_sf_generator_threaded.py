"""
Threaded Quantum SF Generator with Universal Threading Support

This implementation extends the quantum_sf_generator.py with the universal
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

# Import the base generator
try:
    from .quantum_sf_generator import QuantumSFGenerator
except ImportError:
    from quantum_sf_generator import QuantumSFGenerator

class ThreadedQuantumSFGenerator(UniversalSFThreadingMixin, QuantumSFGenerator):
    """
    Threaded Quantum SF Generator with up to 8.9x performance improvement.
    
    This class combines the quantum encoding capabilities of QuantumSFGenerator
    with the universal threading system for optimized batch processing.
    """
    
    def __init__(self, n_modes=2, latent_dim=4, layers=2, cutoff_dim=8, 
                 encoding_strategy='coherent_state', config=None, 
                 enable_batch_processing=True, enable_threading=True,
                 max_threads=None):
        """
        Initialize threaded quantum generator.
        
        Args:
            n_modes (int): Number of quantum modes (output dimension)
            latent_dim (int): Dimension of classical latent input
            layers (int): Number of quantum neural network layers
            cutoff_dim (int): Fock space cutoff for simulation
            encoding_strategy (str): Quantum encoding strategy to use
            config (QuantumGANConfig): Optional configuration object
            enable_batch_processing (bool): Enable batch processing optimizations
            enable_threading (bool): Enable threading optimizations
            max_threads (int): Maximum number of threads (auto-detect if None)
        """
        # Initialize with threading support
        super().__init__(
            n_modes=n_modes,
            latent_dim=latent_dim,
            layers=layers,
            cutoff_dim=cutoff_dim,
            encoding_strategy=encoding_strategy,
            config=config,
            enable_batch_processing=enable_batch_processing,
            enable_threading=enable_threading,
            max_threads=max_threads
        )
        
        logger.info(f"ThreadedQuantumSFGenerator initialized with threading: {enable_threading}")
        if enable_threading and self.threading_manager:
            logger.info(f"  - Max threads: {self.threading_manager.max_threads}")
            logger.info(f"  - GPU available: {self.threading_manager.has_gpu}")
    
    def generate(self, z):
        """
        Generate samples with optional threading optimization.
        
        Args:
            z (tensor): Latent noise vector [batch_size, latent_dim]
            
        Returns:
            samples (tensor): Generated samples [batch_size, n_modes]
        """
        # If threading is enabled and batch size is suitable, use threaded execution
        if self.enable_threading and self.threading_manager is not None:
            batch_size = tf.shape(z)[0]
            
            # Choose strategy based on batch size
            if batch_size >= 4:  # Threading beneficial for batch_size >= 4
                logger.debug(f"Using threaded generation for batch_size={batch_size}")
                return self.execute_threaded(z, method_name='generate', strategy='auto')
        
        # Otherwise, use the parent class implementation
        return super().generate(z)
    
    def generate_batch_optimized(self, z, strategy='auto'):
        """
        Generate samples with explicit threading strategy selection.
        
        Args:
            z (tensor): Latent noise vector [batch_size, latent_dim]
            strategy (str): Threading strategy ('sequential', 'cpu_batch', 'threading', 'gpu_batch', 'auto')
            
        Returns:
            samples (tensor): Generated samples [batch_size, n_modes]
        """
        if not self.enable_threading or self.threading_manager is None:
            logger.warning("Threading not enabled, falling back to standard generation")
            return super().generate(z)
        
        return self.execute_threaded(z, method_name='generate', strategy=strategy)
    
    def benchmark_generation(self, test_batch_sizes=None):
        """
        Benchmark generation performance with different threading strategies.
        
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
        test_inputs = tf.random.normal([max_batch, self.latent_dim])
        
        return self.benchmark_threading_performance(
            test_inputs=test_inputs,
            method_name='generate',
            test_batch_sizes=test_batch_sizes
        )
    
    def get_threading_stats(self):
        """Get current threading performance statistics."""
        if self.threading_manager:
            return self.threading_manager.get_performance_report()
        return {"error": "Threading not enabled"}


def create_threaded_generator(**kwargs):
    """
    Factory function to create a threaded quantum generator.
    
    Args:
        **kwargs: Arguments for ThreadedQuantumSFGenerator
        
    Returns:
        ThreadedQuantumSFGenerator instance
    """
    return ThreadedQuantumSFGenerator(**kwargs)


def test_threaded_generator():
    """Test the threaded quantum generator."""
    print("\n" + "="*60)
    print("TESTING THREADED QUANTUM SF GENERATOR")
    print("="*60)
    
    try:
        # Create threaded generator
        generator = ThreadedQuantumSFGenerator(
            n_modes=2,
            latent_dim=4,
            layers=2,
            cutoff_dim=6,
            enable_threading=True
        )
        
        print(f"Threaded generator created successfully")
        print(f"Threading enabled: {generator.enable_threading}")
        
        # Test standard generation
        z_test = tf.random.normal([16, 4])
        samples = generator.generate(z_test)
        print(f"\nStandard generation test: {samples.shape}")
        print(f"Sample range: [{tf.reduce_min(samples):.3f}, {tf.reduce_max(samples):.3f}]")
        
        # Test different strategies
        print("\nTesting different threading strategies:")
        strategies = ['sequential', 'cpu_batch', 'threading', 'auto']
        
        for strategy in strategies:
            try:
                samples = generator.generate_batch_optimized(z_test, strategy=strategy)
                print(f"  {strategy}: Success - shape {samples.shape}")
            except Exception as e:
                print(f"  {strategy}: Failed - {e}")
        
        # Benchmark performance
        print("\nRunning performance benchmark...")
        benchmark_results = generator.benchmark_generation(test_batch_sizes=[1, 4, 8, 16])
        
        # Display threading stats
        print("\nThreading statistics:")
        stats = generator.get_threading_stats()
        for key, value in stats.items():
            if key != 'strategy_usage':
                print(f"  {key}: {value}")
        
        return generator
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the threaded generator
    test_threaded_generator()
