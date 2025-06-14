"""
Diagnose threading performance issues in quantum circuit execution.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
import time
import threading
import concurrent.futures
from multiprocessing import cpu_count
import psutil
import os

# Add src to path
import sys
sys.path.insert(0, 'src')

from models.generators.quantum_sf_generator_threaded import ThreadedQuantumSFGenerator
from utils.sf_threading_fixed import SFThreadingManager

def monitor_cpu_usage(duration=5, interval=0.1):
    """Monitor CPU usage during execution."""
    cpu_percentages = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
        cpu_percentages.append({
            'time': time.time() - start_time,
            'total': sum(cpu_percent) / len(cpu_percent),
            'per_core': cpu_percent
        })
    
    return cpu_percentages

def test_engine_creation_overhead():
    """Test the overhead of creating SF engines."""
    print("\n1. Testing SF Engine Creation Overhead")
    print("=" * 60)
    
    # Time single engine creation
    start = time.time()
    engine = sf.Engine("tf", backend_options={"cutoff_dim": 6})
    single_time = time.time() - start
    print(f"Single engine creation: {single_time*1000:.2f}ms")
    
    # Time multiple engine creation
    start = time.time()
    engines = []
    for i in range(8):
        eng = sf.Engine("tf", backend_options={"cutoff_dim": 6})
        engines.append(eng)
    multi_time = time.time() - start
    print(f"8 engines creation: {multi_time*1000:.2f}ms ({multi_time/8*1000:.2f}ms per engine)")
    
    return single_time, multi_time

def test_engine_reset_overhead():
    """Test the overhead of resetting SF engines."""
    print("\n2. Testing SF Engine Reset Overhead")
    print("=" * 60)
    
    engine = sf.Engine("tf", backend_options={"cutoff_dim": 6})
    prog = sf.Program(2)
    
    # Run once to initialize
    with prog.context as q:
        sf.ops.Dgate(0.5) | q[0]
    
    engine.run(prog)
    
    # Time reset operations
    reset_times = []
    for i in range(10):
        start = time.time()
        engine.reset()
        reset_time = time.time() - start
        reset_times.append(reset_time)
    
    avg_reset = np.mean(reset_times)
    print(f"Average engine reset time: {avg_reset*1000:.2f}ms")
    print(f"Min/Max reset time: {min(reset_times)*1000:.2f}ms / {max(reset_times)*1000:.2f}ms")
    
    return avg_reset

def test_quantum_circuit_execution_time():
    """Test basic quantum circuit execution time."""
    print("\n3. Testing Quantum Circuit Execution Time")
    print("=" * 60)
    
    engine = sf.Engine("tf", backend_options={"cutoff_dim": 6})
    
    for n_modes in [1, 2, 4]:
        prog = sf.Program(n_modes)
        
        # Create a simple circuit
        with prog.context as q:
            for i in range(n_modes):
                sf.ops.Dgate(0.5, 0.0) | q[i]
                sf.ops.Sgate(0.1) | q[i]
        
        # Time execution
        times = []
        for _ in range(10):
            engine.reset()
            start = time.time()
            result = engine.run(prog)
            exec_time = time.time() - start
            times.append(exec_time)
        
        avg_time = np.mean(times)
        print(f"{n_modes}-mode circuit: {avg_time*1000:.2f}ms average")

def test_threading_contention():
    """Test if there's contention when using multiple engines."""
    print("\n4. Testing Threading Contention")
    print("=" * 60)
    
    n_threads = 4
    n_executions = 10
    
    # Create separate engines for each thread
    engines = []
    progs = []
    for i in range(n_threads):
        engine = sf.Engine("tf", backend_options={"cutoff_dim": 6})
        prog = sf.Program(2)
        with prog.context as q:
            sf.ops.Dgate(0.5, 0.0) | q[0]
            sf.ops.Sgate(0.1) | q[1]
        engines.append(engine)
        progs.append(prog)
    
    # Sequential execution
    start = time.time()
    for i in range(n_threads * n_executions):
        engine_idx = i % n_threads
        engines[engine_idx].reset()
        engines[engine_idx].run(progs[engine_idx])
    sequential_time = time.time() - start
    
    # Threaded execution
    def execute_on_engine(engine_idx, n_times):
        engine = engines[engine_idx]
        prog = progs[engine_idx]
        for _ in range(n_times):
            engine.reset()
            engine.run(prog)
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            future = executor.submit(execute_on_engine, i, n_executions)
            futures.append(future)
        
        # Wait for all to complete
        for future in futures:
            future.result()
    
    threaded_time = time.time() - start
    
    print(f"Sequential execution ({n_threads*n_executions} circuits): {sequential_time:.3f}s")
    print(f"Threaded execution ({n_threads} threads): {threaded_time:.3f}s")
    print(f"Speedup: {sequential_time/threaded_time:.2f}x")
    
    if threaded_time >= sequential_time:
        print("⚠️ WARNING: Threading is SLOWER than sequential!")
        print("This suggests contention or serialization in SF backend.")

def test_tensorflow_threading():
    """Test if TensorFlow operations can be threaded effectively."""
    print("\n5. Testing TensorFlow Threading")
    print("=" * 60)
    
    # Create some TF operations
    def tf_operation(n_ops=1000):
        x = tf.random.normal([100, 100])
        for _ in range(n_ops):
            x = tf.nn.tanh(x)
            x = tf.matmul(x, x)
        return x
    
    # Sequential
    start = time.time()
    for _ in range(4):
        _ = tf_operation()
    seq_time = time.time() - start
    
    # Threaded
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(tf_operation) for _ in range(4)]
        for future in futures:
            _ = future.result()
    thread_time = time.time() - start
    
    print(f"Sequential TF ops: {seq_time:.3f}s")
    print(f"Threaded TF ops: {thread_time:.3f}s")
    print(f"Speedup: {seq_time/thread_time:.2f}x")

def test_generator_threading_breakdown():
    """Detailed breakdown of generator threading performance."""
    print("\n6. Testing Generator Threading Breakdown")
    print("=" * 60)
    
    # Create generator
    gen = ThreadedQuantumSFGenerator(
        n_modes=2,
        latent_dim=4,
        layers=1,
        cutoff_dim=6,
        enable_threading=True
    )
    
    # Test different batch sizes
    for batch_size in [1, 4, 8]:
        print(f"\nBatch size: {batch_size}")
        z = tf.random.normal([batch_size, 4])
        
        # Measure different strategies with CPU monitoring
        strategies = ['sequential', 'cpu_batch', 'threading']
        
        for strategy in strategies:
            # Start CPU monitoring in background
            cpu_monitor = threading.Thread(
                target=lambda: monitor_cpu_usage(duration=2),
                daemon=True
            )
            
            start = time.time()
            _ = gen.generate_batch_optimized(z, strategy=strategy)
            exec_time = time.time() - start
            
            print(f"  {strategy}: {exec_time:.3f}s ({batch_size/exec_time:.2f} samples/s)")

def analyze_threading_manager():
    """Analyze the threading manager configuration."""
    print("\n7. Analyzing Threading Manager Configuration")
    print("=" * 60)
    
    tm = SFThreadingManager(n_modes=2, cutoff_dim=6)
    
    print(f"CPU cores available: {cpu_count()}")
    print(f"Max threads configured: {tm.max_threads}")
    print(f"Engine pool size: {len(tm._engine_pool)}")
    
    # Test strategy selection
    print("\nStrategy selection by batch size:")
    for bs in [1, 2, 4, 8, 16, 32]:
        strategy = tm.choose_strategy(bs)
        print(f"  Batch size {bs}: {strategy}")

def main():
    """Run all diagnostics."""
    print("QUANTUM THREADING PERFORMANCE DIAGNOSTICS")
    print("=" * 60)
    print(f"System: {os.cpu_count()} CPU cores")
    print(f"Process affinity: {len(psutil.Process().cpu_affinity())} cores")
    
    # Run diagnostics
    test_engine_creation_overhead()
    test_engine_reset_overhead()
    test_quantum_circuit_execution_time()
    test_threading_contention()
    test_tensorflow_threading()
    test_generator_threading_breakdown()
    analyze_threading_manager()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
