"""
Simple diagnostic to understand why threading isn't providing speedup.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
import time
import threading
import concurrent.futures
from multiprocessing import cpu_count
import os

# Add src to path
import sys
sys.path.insert(0, 'src')

def test_sf_global_lock():
    """Test if SF has a global lock preventing parallelism."""
    print("\n1. Testing for SF Global Lock")
    print("=" * 60)
    
    # Create multiple engines
    engines = []
    programs = []
    for i in range(4):
        eng = sf.Engine("tf", backend_options={"cutoff_dim": 4})
        prog = sf.Program(2)
        engines.append(eng)
        programs.append(prog)
    
    # Function to run a simple circuit
    def run_circuit(engine_id):
        eng = engines[engine_id]
        prog = programs[engine_id]
        
        # Simple circuit
        with prog.context as q:
            sf.ops.Dgate(0.5) | q[0]
        
        start = time.time()
        result = eng.run(prog)
        end = time.time()
        return end - start
    
    # Test sequential execution
    print("Sequential execution (4 circuits):")
    seq_start = time.time()
    seq_times = []
    for i in range(4):
        t = run_circuit(i)
        seq_times.append(t)
        print(f"  Circuit {i}: {t*1000:.2f}ms")
    seq_total = time.time() - seq_start
    print(f"Total sequential time: {seq_total*1000:.2f}ms")
    
    # Test parallel execution
    print("\nParallel execution (4 circuits):")
    par_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_circuit, i) for i in range(4)]
        par_times = [f.result() for f in futures]
    par_total = time.time() - par_start
    
    for i, t in enumerate(par_times):
        print(f"  Circuit {i}: {t*1000:.2f}ms")
    print(f"Total parallel time: {par_total*1000:.2f}ms")
    
    print(f"\nSpeedup: {seq_total/par_total:.2f}x")
    
    if par_total >= seq_total * 0.9:  # Less than 10% improvement
        print("⚠️ WARNING: No significant speedup from threading!")
        print("This suggests SF has internal locks preventing parallelism.")

def test_tensorflow_in_sf():
    """Test if TensorFlow operations within SF can be parallelized."""
    print("\n2. Testing TensorFlow Operations in SF Context")
    print("=" * 60)
    
    # Test pure TF operations
    def tf_compute(size=1000):
        x = tf.random.normal([size, size])
        for _ in range(10):
            x = tf.matmul(x, x)
        return x
    
    # Sequential
    seq_start = time.time()
    for _ in range(4):
        _ = tf_compute()
    seq_time = time.time() - seq_start
    
    # Parallel
    par_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(tf_compute) for _ in range(4)]
        for f in futures:
            _ = f.result()
    par_time = time.time() - par_start
    
    print(f"Sequential TF operations: {seq_time*1000:.2f}ms")
    print(f"Parallel TF operations: {par_time*1000:.2f}ms")
    print(f"TF Speedup: {seq_time/par_time:.2f}x")

def analyze_batch_size_effect():
    """Analyze why batch size 1 shows different behavior."""
    print("\n3. Analyzing Batch Size Effect")
    print("=" * 60)
    
    from models.generators.quantum_sf_generator_threaded import ThreadedQuantumSFGenerator
    
    # Create generator
    gen = ThreadedQuantumSFGenerator(
        n_modes=2,
        latent_dim=4,
        layers=1,
        cutoff_dim=4,
        enable_threading=True
    )
    
    # Test initialization overhead
    print("Testing first run vs subsequent runs...")
    
    for run in range(3):
        print(f"\nRun {run + 1}:")
        for batch_size in [1, 4]:
            z = tf.random.normal([batch_size, 4])
            
            # Time sequential
            start = time.time()
            _ = gen.generate_batch_optimized(z, strategy='sequential')
            seq_time = time.time() - start
            
            # Time cpu_batch
            start = time.time()
            _ = gen.generate_batch_optimized(z, strategy='cpu_batch')
            batch_time = time.time() - start
            
            print(f"  Batch {batch_size}: seq={seq_time*1000:.2f}ms, batch={batch_time*1000:.2f}ms, speedup={seq_time/batch_time:.2f}x")

def test_thread_pool_overhead():
    """Test thread pool creation and management overhead."""
    print("\n4. Testing Thread Pool Overhead")
    print("=" * 60)
    
    # Simple task
    def simple_task(x):
        return x * 2
    
    # Test different batch sizes
    for n_tasks in [1, 4, 8, 16]:
        # Sequential
        seq_start = time.time()
        results = [simple_task(i) for i in range(n_tasks)]
        seq_time = time.time() - seq_start
        
        # Threaded
        thread_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simple_task, i) for i in range(n_tasks)]
            results = [f.result() for f in futures]
        thread_time = time.time() - thread_start
        
        overhead = thread_time - seq_time
        print(f"Tasks: {n_tasks}, Sequential: {seq_time*1000:.3f}ms, Threaded: {thread_time*1000:.3f}ms, Overhead: {overhead*1000:.3f}ms")

def main():
    """Run all diagnostics."""
    print("SIMPLIFIED THREADING DIAGNOSTICS")
    print("=" * 60)
    print(f"System: {cpu_count()} CPU cores")
    print(f"TensorFlow threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
    
    test_sf_global_lock()
    test_tensorflow_in_sf()
    analyze_batch_size_effect()
    test_thread_pool_overhead()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("\nKey Findings:")
    print("1. Check if SF operations show speedup with threading")
    print("2. Check if batch_size=1 has initialization overhead")
    print("3. Check thread pool overhead for small tasks")

if __name__ == "__main__":
    main()
