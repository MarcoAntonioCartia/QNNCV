"""
Strawberry Fields Threading and Parallel Execution Implementation
Based on research findings about SF's batch processing and TensorFlow integration
"""

import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
import numpy as np
import concurrent.futures
import threading
from multiprocessing import Pool, cpu_count
import time
from typing import List, Callable, Any
import logging

logger = logging.getLogger(__name__)

class SFThreadingManager:
    """
    Advanced threading manager for Strawberry Fields quantum simulations
    
    Implements multiple parallelization strategies:
    1. Batch processing via TensorFlow backend
    2. Multi-threading for independent circuits 
    3. GPU acceleration integration
    4. Hybrid CPU/GPU workload distribution
    """
    
    def __init__(self, n_modes=2, cutoff_dim=10, n_threads=None):
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        self.n_threads = n_threads or min(cpu_count(), 8)  # Reasonable thread limit
        
        # Initialize different engines for different strategies
        self._init_engines()
        
        logger.info(f"Threading Manager initialized:")
        logger.info(f"  - Modes: {n_modes}")
        logger.info(f"  - Cutoff: {cutoff_dim}")
        logger.info(f"  - Threads: {self.n_threads}")
        
    def _init_engines(self):
        """Initialize multiple engines for different threading strategies"""
        
        # 1. Batch processing engine (TensorFlow backend)
        self.batch_engine = sf.Engine("tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "batch_size": None  # Dynamic batch sizing
        })
        
        # 2. Single-threaded engines for independent execution
        self.thread_engines = []
        for i in range(self.n_threads):
            engine = sf.Engine("tf", backend_options={
                "cutoff_dim": self.cutoff_dim,
                "batch_size": None
            })
            self.thread_engines.append(engine)
            
        # 3. GPU-optimized engine if available
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                self.gpu_engine = sf.Engine("tf", backend_options={
                    "cutoff_dim": self.cutoff_dim,
                    "batch_size": 32  # Larger batches for GPU
                })
                logger.info(f"✅ GPU acceleration available: {len(gpus)} GPU(s)")
            else:
                self.gpu_engine = None
                logger.info("❌ No GPU acceleration available")
        except Exception as e:
            self.gpu_engine = None
            logger.warning(f"GPU initialization failed: {e}")

class BatchQuantumExecutor:
    """
    Implements SF's native batch processing for maximum performance
    """
    
    def __init__(self, threading_manager: SFThreadingManager):
        self.tm = threading_manager
        
    def execute_batch_circuits(self, programs: List[sf.Program], 
                             parameter_sets: List[dict], 
                             batch_size: int = 16):
        """
        Execute multiple quantum circuits using SF's batch processing
        
        Args:
            programs: List of SF programs to execute
            parameter_sets: List of parameter dictionaries for each program
            batch_size: Batch size for parallel execution
            
        Returns:
            List of quantum states from each execution
        """
        results = []
        
        # Process in batches
        for i in range(0, len(programs), batch_size):
            batch_programs = programs[i:i+batch_size]
            batch_params = parameter_sets[i:i+batch_size]
            
            # Use TensorFlow backend with batch processing
            batch_results = self._execute_batch(batch_programs, batch_params)
            results.extend(batch_results)
            
        return results
    
    def _execute_batch(self, programs: List[sf.Program], 
                      parameter_sets: List[dict]):
        """Execute a single batch using TensorFlow backend"""
        
        # Configure engine for this batch size
        current_batch_size = len(programs)
        engine = sf.Engine("tf", backend_options={
            "cutoff_dim": self.tm.cutoff_dim,
            "batch_size": current_batch_size
        })
        
        results = []
        
        # SF's batch processing works by vectorizing parameters
        for prog, params in zip(programs, parameter_sets):
            try:
                # Convert parameters to TensorFlow tensors for vectorization
                tf_params = {k: tf.constant(v) for k, v in params.items()}
                
                # Execute with batch-enabled engine
                state = engine.run(prog, args=tf_params).state
                results.append(state)
                
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                results.append(None)
        
        return results

class ThreadPoolQuantumExecutor:
    """
    Multi-threaded executor for independent quantum circuits
    """
    
    def __init__(self, threading_manager: SFThreadingManager):
        self.tm = threading_manager
        self._thread_local = threading.local()
        
    def execute_parallel_circuits(self, circuit_generator: Callable,
                                parameter_list: List[Any],
                                max_workers: int = None):
        """
        Execute multiple independent circuits in parallel threads
        
        Args:
            circuit_generator: Function that creates SF program from parameters
            parameter_list: List of parameters for each circuit
            max_workers: Maximum number of worker threads
            
        Returns:
            List of results from each circuit execution
        """
        max_workers = max_workers or self.tm.n_threads
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._execute_single_circuit, circuit_generator, params): params
                for params in parameter_list
            }
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Thread execution failed: {e}")
                    results.append(None)
        
        return results
    
    def _execute_single_circuit(self, circuit_generator: Callable, params: Any):
        """Execute single circuit in thread-local context"""
        
        # Get thread-local engine
        if not hasattr(self._thread_local, 'engine'):
            self._thread_local.engine = sf.Engine("tf", backend_options={
                "cutoff_dim": self.tm.cutoff_dim
            })
        
        # Generate and execute circuit
        prog = circuit_generator(params)
        state = self._thread_local.engine.run(prog).state
        
        return state

class GPUAcceleratedExecutor:
    """
    GPU-accelerated quantum circuit execution
    """
    
    def __init__(self, threading_manager: SFThreadingManager):
        self.tm = threading_manager
        
    def execute_gpu_batch(self, programs: List[sf.Program],
                         parameter_sets: List[dict],
                         gpu_batch_size: int = 64):
        """
        Execute large batches on GPU for maximum throughput
        
        Args:
            programs: List of SF programs
            parameter_sets: List of parameter dictionaries  
            gpu_batch_size: Large batch size optimized for GPU
            
        Returns:
            List of quantum states
        """
        if self.tm.gpu_engine is None:
            logger.warning("No GPU available, falling back to CPU")
            return BatchQuantumExecutor(self.tm).execute_batch_circuits(
                programs, parameter_sets, batch_size=16
            )
        
        # Use GPU context
        with tf.device('/GPU:0'):
            results = []
            
            # Process in large GPU-optimized batches
            for i in range(0, len(programs), gpu_batch_size):
                batch_programs = programs[i:i+gpu_batch_size]
                batch_params = parameter_sets[i:i+gpu_batch_size]
                
                batch_results = self._execute_gpu_batch(batch_programs, batch_params)
                results.extend(batch_results)
                
        return results
    
    def _execute_gpu_batch(self, programs: List[sf.Program], 
                          parameter_sets: List[dict]):
        """Execute batch on GPU with memory optimization"""
        
        results = []
        
        # Vectorize parameters for GPU execution
        for prog, params in zip(programs, parameter_sets):
            # Convert to GPU tensors
            gpu_params = {}
            for k, v in params.items():
                gpu_params[k] = tf.constant(v, dtype=tf.complex64)
            
            # Execute on GPU
            state = self.tm.gpu_engine.run(prog, args=gpu_params).state
            results.append(state)
            
        return results

class HybridQuantumExecutor:
    """
    Hybrid executor that intelligently distributes workload across CPU/GPU
    """
    
    def __init__(self, threading_manager: SFThreadingManager):
        self.tm = threading_manager
        self.batch_executor = BatchQuantumExecutor(threading_manager)
        self.thread_executor = ThreadPoolQuantumExecutor(threading_manager)
        self.gpu_executor = GPUAcceleratedExecutor(threading_manager)
        
    def auto_execute(self, programs: List[sf.Program],
                    parameter_sets: List[dict],
                    optimization_strategy: str = "auto"):
        """
        Automatically choose optimal execution strategy
        
        Args:
            programs: List of SF programs
            parameter_sets: List of parameter dictionaries
            optimization_strategy: "auto", "cpu_batch", "gpu_batch", "threading"
            
        Returns:
            List of quantum states and performance metrics
        """
        start_time = time.time()
        n_circuits = len(programs)
        
        # Choose strategy based on workload and hardware
        if optimization_strategy == "auto":
            strategy = self._choose_optimal_strategy(n_circuits)
        else:
            strategy = optimization_strategy
            
        logger.info(f"Executing {n_circuits} circuits using strategy: {strategy}")
        
        # Execute with chosen strategy
        if strategy == "gpu_batch" and self.tm.gpu_engine is not None:
            results = self.gpu_executor.execute_gpu_batch(programs, parameter_sets)
        elif strategy == "cpu_batch":
            results = self.batch_executor.execute_batch_circuits(programs, parameter_sets)
        elif strategy == "threading":
            # Convert to circuit generator function
            def circuit_gen(i):
                return programs[i], parameter_sets[i]
            results = self.thread_executor.execute_parallel_circuits(
                lambda i: circuit_gen(i)[0], range(n_circuits)
            )
        else:
            # Fallback to batch processing
            results = self.batch_executor.execute_batch_circuits(programs, parameter_sets)
        
        # Performance metrics
        execution_time = time.time() - start_time
        successful_results = sum(1 for r in results if r is not None)
        
        metrics = {
            "strategy": strategy,
            "total_circuits": n_circuits,
            "successful_executions": successful_results,
            "execution_time": execution_time,
            "circuits_per_second": successful_results / execution_time if execution_time > 0 else 0
        }
        
        logger.info(f"Execution completed: {metrics}")
        
        return results, metrics
    
    def _choose_optimal_strategy(self, n_circuits: int) -> str:
        """Choose optimal execution strategy based on workload"""
        
        # Strategy selection heuristics
        if self.tm.gpu_engine is not None and n_circuits > 32:
            return "gpu_batch"
        elif n_circuits > 16:
            return "cpu_batch"  
        elif n_circuits > 4:
            return "threading"
        else:
            return "cpu_batch"

# ============================================================================
# USAGE EXAMPLES AND INTEGRATION WITH YOUR QUANTUM GENERATOR
# ============================================================================

def create_test_quantum_circuit(params):
    """Example circuit generator for testing"""
    prog = sf.Program(2)
    
    with prog.context as q:
        # Squeeze gates with parameters
        Sgate(params.get('r1', 0.1)) | q[0]
        Sgate(params.get('r2', 0.1)) | q[1]
        
        # Beamsplitter
        BSgate(params.get('theta', 0.5), params.get('phi', 0.0)) | (q[0], q[1])
        
        # Displacement
        Dgate(params.get('alpha', 0.2)) | q[0]
        
    return prog

def integrate_with_quantum_generator():
    """Example integration with your QuantumSFGenerator"""
    
    # Initialize threading manager
    tm = SFThreadingManager(n_modes=2, cutoff_dim=8, n_threads=4)
    executor = HybridQuantumExecutor(tm)
    
    # Create multiple parameter sets (simulating different latent inputs)
    parameter_sets = []
    programs = []
    
    for i in range(50):  # Generate 50 different quantum states
        params = {
            'r1': np.random.normal(0, 0.1),
            'r2': np.random.normal(0, 0.1), 
            'theta': np.random.uniform(0, np.pi),
            'phi': np.random.uniform(0, 2*np.pi),
            'alpha': np.random.normal(0, 0.2)
        }
        parameter_sets.append(params)
        programs.append(create_test_quantum_circuit(params))
    
    # Execute with automatic optimization
    results, metrics = executor.auto_execute(programs, parameter_sets)
    
    print(f"Generated {len(results)} quantum states")
    print(f"Performance: {metrics['circuits_per_second']:.2f} circuits/second")
    print(f"Strategy used: {metrics['strategy']}")
    
    return results, metrics

def benchmark_threading_strategies():
    """Benchmark different threading strategies"""
    
    tm = SFThreadingManager(n_modes=2, cutoff_dim=6)
    
    # Test data
    n_circuits = 32
    programs = [create_test_quantum_circuit({'r1': 0.1, 'r2': 0.1}) for _ in range(n_circuits)]
    parameter_sets = [{'r1': 0.1, 'r2': 0.1} for _ in range(n_circuits)]
    
    strategies = ["cpu_batch", "threading", "gpu_batch"]
    
    print("🚀 THREADING BENCHMARK RESULTS:")
    print("=" * 50)
    
    for strategy in strategies:
        try:
            executor = HybridQuantumExecutor(tm)
            results, metrics = executor.auto_execute(
                programs, parameter_sets, optimization_strategy=strategy
            )
            
            print(f"Strategy: {strategy}")
            print(f"  Time: {metrics['execution_time']:.3f}s")
            print(f"  Speed: {metrics['circuits_per_second']:.2f} circuits/s")
            print(f"  Success: {metrics['successful_executions']}/{metrics['total_circuits']}")
            print()
            
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")

if __name__ == "__main__":
    # Run integration example
    print("Testing SF Threading Integration...")
    integrate_with_quantum_generator()
    
    print("\nRunning benchmark...")
    benchmark_threading_strategies()
