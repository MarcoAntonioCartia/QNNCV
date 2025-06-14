"""
Test the existing working generator threading from quantum_threading.py
"""

import tensorflow as tf
import numpy as np
import time
from src.models.generators.quantum_sf_generator import QuantumSFGenerator
from src.utils.quantum_threading import create_threaded_quantum_generator

def test_working_generator_threading():
    """Test the existing working generator threading."""
    print("=" * 60)
    print("TESTING WORKING GENERATOR THREADING")
    print("=" * 60)
    
    # Create threaded generator using the working approach
    print("Creating threaded generator using quantum_threading.py...")
    try:
        threaded_generator = create_threaded_quantum_generator(
            QuantumSFGenerator,
            n_modes=2,
            latent_dim=4,
            enable_threading=True
        )
        print("✅ Threaded generator created successfully!")
    except Exception as e:
        print(f"❌ Failed to create threaded generator: {e}")
        return
    
    # Test data
    batch_size = 16
    z = tf.random.normal([batch_size, 4])
    
    print(f"\nTesting with batch size: {batch_size}")
    print(f"Latent input shape: {z.shape}")
    
    # Test different strategies
    strategies = ["sequential", "cpu_batch", "threading", "auto"]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        try:
            start_time = time.time()
            samples = threaded_generator.generate_threaded(z, strategy=strategy)
            end_time = time.time()
            
            execution_time = end_time - start_time
            samples_per_second = batch_size / execution_time if execution_time > 0 else 0
            
            print(f"  ✅ Success!")
            print(f"  Time: {execution_time:.3f}s")
            print(f"  Speed: {samples_per_second:.2f} samples/s")
            print(f"  Output shape: {samples.shape}")
            print(f"  Output range: [{tf.reduce_min(samples):.3f}, {tf.reduce_max(samples):.3f}]")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Performance report
    if hasattr(threaded_generator, 'threading_manager') and threaded_generator.threading_manager:
        print(f"\nPerformance Report:")
        report = threaded_generator.threading_manager.get_performance_report()
        print(f"  Total executions: {report['total_executions']}")
        print(f"  Average speed: {report['avg_samples_per_second']:.2f} samples/s")
        print(f"  CPU utilization: {report['cpu_utilization_estimate']:.1f}%")
        print(f"  Strategy usage: {report['strategy_usage']}")
    
    print("\n" + "=" * 60)
    print("WORKING GENERATOR THREADING TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_working_generator_threading()
