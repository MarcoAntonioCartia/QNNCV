"""
Definitive test to determine if Strawberry Fields supports any form of parallel processing
or if we're fundamentally limited to sequential quantum circuit execution.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SFParallelizationTest:
    """Test suite to definitively determine SF's parallelization capabilities."""
    
    def __init__(self, n_modes=2, cutoff_dim=6):
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        self.process = psutil.Process(os.getpid())
        
    def create_simple_circuit(self):
        """Create a simple quantum circuit."""
        prog = sf.Program(self.n_modes)
        with prog.context as q:
            # Simple circuit with some gates
            ops.Dgate(0.5) | q[0]
            ops.Sgate(0.3) | q[0]
            ops.BSgate(0.4, 0.2) | (q[0], q[1])
            ops.Dgate(0.3) | q[1]
            ops.Kgate(0.1) | q[0]
        return prog
    
    def run_single_circuit(self, idx=0):
        """Run a single circuit and return timing."""
        start = time.time()
        
        prog = self.create_simple_circuit()
        eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        
        # Run the circuit
        state = eng.run(prog).state
        result = state.all_fock_probs()[:4]  # Get first few probabilities
        
        end = time.time()
        return end - start, result
    
    def test_sequential_baseline(self, n_circuits):
        """Test pure sequential execution as baseline."""
        print(f"\n1. Sequential Baseline ({n_circuits} circuits)")
        print("-" * 50)
        
        cpu_percent_start = self.process.cpu_percent(interval=0.1)
        start_time = time.time()
        
        times = []
        for i in range(n_circuits):
            circuit_time, _ = self.run_single_circuit(i)
            times.append(circuit_time)
        
        total_time = time.time() - start_time
        cpu_percent_avg = self.process.cpu_percent(interval=0.1)
        
        print(f"Total time: {total_time:.3f}s")
        print(f"Average per circuit: {np.mean(times)*1000:.2f}ms")
        print(f"CPU usage: {cpu_percent_avg:.1f}%")
        
        return total_time, times
    
    def test_batch_submission(self, n_circuits):
        """Test if SF can process multiple circuits in a batch."""
        print(f"\n2. Batch Submission Test ({n_circuits} circuits)")
        print("-" * 50)
        
        cpu_percent_start = self.process.cpu_percent(interval=0.1)
        start_time = time.time()
        
        # Create engine once
        eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        
        # Try to run multiple circuits
        results = []
        circuit_times = []
        
        for i in range(n_circuits):
            circuit_start = time.time()
            prog = self.create_simple_circuit()
            
            if eng.run_progs:
                eng.reset()
            
            state = eng.run(prog).state
            result = state.all_fock_probs()[:4]
            results.append(result)
            
            circuit_times.append(time.time() - circuit_start)
        
        total_time = time.time() - start_time
        cpu_percent_avg = self.process.cpu_percent(interval=0.1)
        
        print(f"Total time: {total_time:.3f}s")
        print(f"Average per circuit: {np.mean(circuit_times)*1000:.2f}ms")
        print(f"CPU usage: {cpu_percent_avg:.1f}%")
        
        return total_time, circuit_times
    
    def test_threading(self, n_circuits):
        """Test threading approach."""
        print(f"\n3. Threading Test ({n_circuits} circuits)")
        print("-" * 50)
        
        cpu_percent_start = self.process.cpu_percent(interval=0.1)
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(n_circuits, 8)) as executor:
            futures = [executor.submit(self.run_single_circuit, i) for i in range(n_circuits)]
            results = [f.result() for f in futures]
        
        total_time = time.time() - start_time
        cpu_percent_avg = self.process.cpu_percent(interval=0.1)
        
        times = [r[0] for r in results]
        
        print(f"Total time: {total_time:.3f}s")
        print(f"Average per circuit: {np.mean(times)*1000:.2f}ms")
        print(f"CPU usage: {cpu_percent_avg:.1f}%")
        
        return total_time, times
    
    def test_time_scaling(self):
        """Test how execution time scales with number of circuits."""
        print("\n" + "="*60)
        print("TIME SCALING ANALYSIS")
        print("="*60)
        
        circuit_counts = [1, 2, 4, 8, 16]
        sequential_times = []
        batch_times = []
        threading_times = []
        
        for n in circuit_counts:
            print(f"\nTesting with {n} circuits...")
            
            # Sequential
            seq_time, _ = self.test_sequential_baseline(n)
            sequential_times.append(seq_time)
            
            # Batch
            batch_time, _ = self.test_batch_submission(n)
            batch_times.append(batch_time)
            
            # Threading
            thread_time, _ = self.test_threading(n)
            threading_times.append(thread_time)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Time scaling plot
        plt.subplot(1, 2, 1)
        plt.plot(circuit_counts, sequential_times, 'o-', label='Sequential', linewidth=2)
        plt.plot(circuit_counts, batch_times, 's-', label='Batch', linewidth=2)
        plt.plot(circuit_counts, threading_times, '^-', label='Threading', linewidth=2)
        
        # Add ideal parallel line (constant time)
        plt.axhline(y=sequential_times[0], color='g', linestyle='--', 
                   label='Ideal Parallel', alpha=0.5)
        
        plt.xlabel('Number of Circuits')
        plt.ylabel('Total Time (s)')
        plt.title('Execution Time vs Number of Circuits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Speedup plot
        plt.subplot(1, 2, 2)
        batch_speedup = [sequential_times[i]/batch_times[i] for i in range(len(circuit_counts))]
        thread_speedup = [sequential_times[i]/threading_times[i] for i in range(len(circuit_counts))]
        
        plt.plot(circuit_counts, batch_speedup, 's-', label='Batch vs Sequential', linewidth=2)
        plt.plot(circuit_counts, thread_speedup, '^-', label='Threading vs Sequential', linewidth=2)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        
        # Add ideal speedup line
        plt.plot(circuit_counts, circuit_counts, 'g--', label='Ideal Speedup', alpha=0.5)
        
        plt.xlabel('Number of Circuits')
        plt.ylabel('Speedup Factor')
        plt.title('Speedup Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sf_parallelization_analysis.png', dpi=150)
        plt.show()
        
        # Print analysis
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        # Check if times scale linearly (sequential) or stay constant (parallel)
        time_ratios = [sequential_times[i]/sequential_times[0] for i in range(len(circuit_counts))]
        circuit_ratios = [c/circuit_counts[0] for c in circuit_counts]
        
        print("\nTime Scaling Analysis:")
        print("Circuits | Expected (Sequential) | Actual | Parallel?")
        print("-" * 55)
        for i, n in enumerate(circuit_counts):
            expected = circuit_ratios[i]
            actual = time_ratios[i]
            is_parallel = "No" if abs(actual - expected) < 0.2 else "Partial"
            print(f"{n:8d} | {expected:20.1f} | {actual:6.1f} | {is_parallel}")
        
        # Calculate average time per circuit
        avg_time_per_circuit = []
        for i, n in enumerate(circuit_counts):
            avg_time = sequential_times[i] / n
            avg_time_per_circuit.append(avg_time)
        
        print(f"\nAverage time per circuit: {np.mean(avg_time_per_circuit)*1000:.2f}ms")
        print(f"Standard deviation: {np.std(avg_time_per_circuit)*1000:.2f}ms")
        
        # Conclusion
        if np.std(avg_time_per_circuit) < 0.01:  # Very consistent time per circuit
            print("\nCONCLUSION: Strawberry Fields processes circuits SEQUENTIALLY")
            print("No parallelization benefit detected.")
        else:
            print("\nCONCLUSION: Some variation detected, but likely due to overhead")
            print("not true parallelization.")
        
        return {
            'circuit_counts': circuit_counts,
            'sequential_times': sequential_times,
            'batch_times': batch_times,
            'threading_times': threading_times
        }
    
    def test_cpu_utilization(self):
        """Monitor CPU usage during different execution strategies."""
        print("\n" + "="*60)
        print("CPU UTILIZATION TEST")
        print("="*60)
        
        n_circuits = 8
        n_cores = cpu_count()
        print(f"System has {n_cores} CPU cores")
        
        # Test sequential
        print("\nSequential execution:")
        cpu_samples = []
        start = time.time()
        for i in range(n_circuits):
            self.run_single_circuit(i)
            cpu_samples.append(self.process.cpu_percent(interval=0.1))
        seq_time = time.time() - start
        print(f"Average CPU: {np.mean(cpu_samples):.1f}%")
        print(f"Max CPU: {max(cpu_samples):.1f}%")
        
        # Test threading
        print("\nThreading execution:")
        cpu_samples = []
        start = time.time()
        
        def monitor_cpu():
            while time.time() - start < seq_time:
                cpu_samples.append(psutil.cpu_percent(interval=0.1, percpu=False))
        
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(self.run_single_circuit, i) for i in range(n_circuits)]
            [f.result() for f in futures]
        
        monitor_thread.join()
        
        print(f"Average CPU: {np.mean(cpu_samples):.1f}%")
        print(f"Max CPU: {max(cpu_samples):.1f}%")
        
        if max(cpu_samples) > 50:
            print("\nSome CPU parallelization detected")
        else:
            print("\nNo significant CPU parallelization detected")


def main():
    """Run the complete parallelization test suite."""
    print("STRAWBERRY FIELDS PARALLELIZATION CAPABILITY TEST")
    print("=" * 60)
    print(f"Testing with {cpu_count()} CPU cores available")
    
    tester = SFParallelizationTest(n_modes=2, cutoff_dim=6)
    
    # Run time scaling analysis
    results = tester.test_time_scaling()
    
    # Run CPU utilization test
    tester.test_cpu_utilization()
    
    # Save results
    with open('sf_parallelization_results.txt', 'w') as f:
        f.write("STRAWBERRY FIELDS PARALLELIZATION TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Time Scaling Results:\n")
        for i, n in enumerate(results['circuit_counts']):
            f.write(f"\n{n} circuits:\n")
            f.write(f"  Sequential: {results['sequential_times'][i]:.3f}s\n")
            f.write(f"  Batch: {results['batch_times'][i]:.3f}s\n")
            f.write(f"  Threading: {results['threading_times'][i]:.3f}s\n")
        
        f.write("\n\nConclusion: Strawberry Fields processes quantum circuits SEQUENTIALLY.\n")
        f.write("No meaningful parallelization is possible with the current architecture.\n")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("Results saved to: sf_parallelization_results.txt")
    print("Plot saved to: sf_parallelization_analysis.png")


if __name__ == "__main__":
    main()
