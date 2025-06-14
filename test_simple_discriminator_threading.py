"""
Simple discriminator threading test
Building up from the working quantum_threading.py approach
"""

import tensorflow as tf
import numpy as np
import time
from src.models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
from src.utils.quantum_threading import create_threaded_quantum_generator, QuantumGeneratorThreadingMixin

# Create a simple threaded discriminator using the working approach
class QuantumDiscriminatorThreadingMixin:
    """
    Simple threading mixin for quantum discriminators based on working generator approach.
    """
    
    def __init__(self, *args, enable_threading: bool = True, 
                 max_threads: int = None, **kwargs):
        """Initialize threading capabilities."""
        super().__init__(*args, **kwargs)
        
        self.enable_threading = enable_threading
        
        if enable_threading:
            from src.utils.quantum_threading import QuantumThreadingManager, QuantumBatchExecutor
            
            self.threading_manager = QuantumThreadingManager(
                n_modes=self.n_modes,
                cutoff_dim=self.cutoff_dim,
                max_threads=max_threads
            )
            self.batch_executor = QuantumBatchExecutor(self.threading_manager)
            print(f"Threading enabled for {self.__class__.__name__}")
        else:
            self.threading_manager = None
            self.batch_executor = None
            print(f"Threading disabled for {self.__class__.__name__}")
    
    def discriminate_threaded(self, x: tf.Tensor, strategy: str = "auto") -> tf.Tensor:
        """
        Discriminate samples using optimized threading.
        
        Args:
            x: Input samples [batch_size, n_modes]
            strategy: "auto", "gpu_batch", "cpu_batch", "threading", "sequential"
            
        Returns:
            Predictions [batch_size, 1]
        """
        if not self.enable_threading:
            return self.discriminate(x)  # Fallback to original method
        
        batch_size = tf.shape(x)[0]
        start_time = time.time()
        
        if strategy == "auto":
            strategy = self.threading_manager.choose_strategy(batch_size)
        
        print(f"Discriminating {batch_size} samples using {strategy}")
        
        # Execute with chosen strategy
        if strategy == "gpu_batch" and self.threading_manager.has_gpu:
            results = self._discriminate_gpu_batch(x)
        elif strategy == "cpu_batch":
            results = self._discriminate_cpu_batch(x)
        elif strategy == "threading":
            results = self._discriminate_threaded_batch(x)
        else:
            results = self.discriminate(x)  # Sequential fallback
        
        # Update performance tracking
        execution_time = time.time() - start_time
        self.threading_manager.update_performance_stats(
            "discriminate", strategy, execution_time, batch_size
        )
        
        return results
    
    def _discriminate_cpu_batch(self, x: tf.Tensor) -> tf.Tensor:
        """Execute discriminate using CPU with optimized batching."""
        cpu_batch_size = 8  # CPU-optimized batch size
        batch_size = tf.shape(x)[0]
        results = []
        
        for i in range(0, batch_size, cpu_batch_size):
            end_idx = tf.minimum(i + cpu_batch_size, batch_size)
            batch_inputs = x[i:end_idx]
            
            # Execute discriminate on CPU batch
            batch_results = self.discriminate(batch_inputs)
            results.append(batch_results)
        
        return tf.concat(results, axis=0)
    
    def _discriminate_threaded_batch(self, x: tf.Tensor) -> tf.Tensor:
        """Execute discriminate using thread pool."""
        import concurrent.futures
        
        def discriminate_single_sample(sample_data):
            """Discriminate single sample in thread."""
            sample_input, sample_idx = sample_data
            
            try:
                # Add batch dimension for single sample
                batched_input = tf.expand_dims(sample_input, 0)
                
                # Execute discriminate
                result = self.discriminate(batched_input)
                
                # Remove batch dimension
                if len(result.shape) > 0:
                    result = tf.squeeze(result, 0)
                
                return sample_idx, result
            except Exception as e:
                print(f"Thread execution failed for sample {sample_idx}: {e}")
                return sample_idx, None
        
        # Prepare data for threading
        batch_size = tf.shape(x)[0]
        sample_data = [(x[i], i) for i in range(batch_size)]
        
        # Use ThreadPoolExecutor for parallel execution
        results = [None] * batch_size
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threading_manager.max_threads) as executor:
            # Submit all tasks
            futures = [executor.submit(discriminate_single_sample, data) for data in sample_data]
            
            # Collect results in order
            for future in concurrent.futures.as_completed(futures):
                try:
                    sample_idx, result = future.result()
                    results[sample_idx] = result
                except Exception as e:
                    print(f"Thread result collection failed: {e}")
        
        # Handle any failed executions
        fallback_result = None
        
        # Find a valid result to use as template
        for result in results:
            if result is not None:
                fallback_result = result
                break
        
        # If no valid results, execute one sample to get the shape
        if fallback_result is None:
            try:
                sample_input = tf.expand_dims(x[0], 0)
                sample_result = self.discriminate(sample_input)
                fallback_result = tf.squeeze(sample_result, 0)
            except Exception:
                # Ultimate fallback
                fallback_result = tf.constant([0.5], dtype=tf.float32)
        
        # Replace None results with fallback
        final_results = []
        for i, result in enumerate(results):
            if result is not None:
                final_results.append(result)
            else:
                fallback = tf.zeros_like(fallback_result)
                final_results.append(fallback)
        
        # Stack all results
        return tf.stack(final_results, axis=0)

def create_threaded_discriminator(base_discriminator_class, *args, **kwargs):
    """
    Factory function to create a threaded discriminator.
    """
    
    class ThreadedDiscriminator(QuantumDiscriminatorThreadingMixin, base_discriminator_class):
        """Dynamically created threaded discriminator."""
        pass
    
    return ThreadedDiscriminator(*args, **kwargs)

def test_simple_discriminator_threading():
    """Test simple discriminator threading."""
    print("=" * 60)
    print("SIMPLE DISCRIMINATOR THREADING TEST")
    print("=" * 60)
    
    # Create threaded discriminator
    print("Creating threaded discriminator...")
    threaded_discriminator = create_threaded_discriminator(
        QuantumSFDiscriminator,
        n_modes=2,
        input_dim=2,
        enable_threading=True
    )
    
    # Test data
    batch_size = 16
    test_samples = tf.random.normal([batch_size, 2])
    
    print(f"\nTesting with batch size: {batch_size}")
    print(f"Test samples shape: {test_samples.shape}")
    
    # Test different strategies
    strategies = ["sequential", "cpu_batch", "threading", "auto"]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        try:
            start_time = time.time()
            predictions = threaded_discriminator.discriminate_threaded(test_samples, strategy=strategy)
            end_time = time.time()
            
            execution_time = end_time - start_time
            samples_per_second = batch_size / execution_time if execution_time > 0 else 0
            
            print(f"  ✅ Success!")
            print(f"  Time: {execution_time:.3f}s")
            print(f"  Speed: {samples_per_second:.2f} samples/s")
            print(f"  Output shape: {predictions.shape}")
            print(f"  Predictions range: [{tf.reduce_min(predictions):.3f}, {tf.reduce_max(predictions):.3f}]")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Performance report
    if threaded_discriminator.threading_manager:
        print(f"\nPerformance Report:")
        report = threaded_discriminator.threading_manager.get_performance_report()
        print(f"  Total executions: {report['total_executions']}")
        print(f"  Average speed: {report['avg_samples_per_second']:.2f} samples/s")
        print(f"  CPU utilization: {report['cpu_utilization_estimate']:.1f}%")
        print(f"  Strategy usage: {report['strategy_usage']}")
    
    print("\n" + "=" * 60)
    print("SIMPLE DISCRIMINATOR THREADING TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_simple_discriminator_threading()
