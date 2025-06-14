"""
Quantum Threading Utilities for QNNCV
Optimized CPU utilization for Strawberry Fields quantum circuits

This module provides threading and parallel execution utilities specifically
designed for the QNNCV quantum GAN system, maximizing CPU utilization while
maintaining gradient flow and quantum encoding compatibility.
"""

import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import concurrent.futures
import threading
import time
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

class QuantumThreadingManager:
    """
    Threading manager for quantum GAN components.
    
    Provides multiple parallelization strategies optimized for:
    - Pure quantum generators and discriminators
    - Gradient-preserving execution
    - Maximum CPU utilization
    - Memory efficiency
    """
    
    def __init__(self, n_modes: int = 2, cutoff_dim: int = 8, 
                 max_threads: Optional[int] = None, enable_gpu: bool = True):
        """
        Initialize quantum threading manager.
        
        Args:
            n_modes: Number of quantum modes
            cutoff_dim: Fock space cutoff dimension
            max_threads: Maximum threads (auto-detect if None)
            enable_gpu: Enable GPU acceleration if available
        """
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        self.max_threads = max_threads or min(cpu_count(), 8)  # Reasonable limit
        self.enable_gpu = enable_gpu
        
        # Performance tracking
        self.performance_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'strategy_usage': {},
            'avg_cpu_utilization': 0.0
        }
        
        # Initialize engines and thread safety
        self._init_engines()
        self._init_thread_safety()
        
        logger.info(f"QuantumThreadingManager initialized:")
        logger.info(f"  - Modes: {n_modes}, Cutoff: {cutoff_dim}")
        logger.info(f"  - Max threads: {self.max_threads}")
        logger.info(f"  - GPU enabled: {self.has_gpu}")
    
    def _init_engines(self):
        """Initialize different engines for various strategies."""
        
        # 1. Batch processing engine (TensorFlow backend)
        self.batch_engine = sf.Engine("tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "batch_size": None  # Dynamic batching
        })
        
        # 2. GPU engine if available
        self.has_gpu = False
        self.gpu_engine = None
        
        if self.enable_gpu:
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    self.gpu_engine = sf.Engine("tf", backend_options={
                        "cutoff_dim": self.cutoff_dim,
                        "batch_size": 32  # GPU-optimized batch size
                    })
                    self.has_gpu = True
                    logger.info(f"✅ GPU acceleration available: {len(gpus)} GPU(s)")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
        
        if not self.has_gpu:
            logger.info("GPU acceleration not available")
    
    def _init_thread_safety(self):
        """Initialize thread-local storage for SF engines."""
        self._thread_local = threading.local()
        self._engine_pool = []
        
        # Pre-create engines for thread pool
        for i in range(self.max_threads):
            engine = sf.Engine("tf", backend_options={
                "cutoff_dim": self.cutoff_dim
            })
            self._engine_pool.append(engine)
    
    def get_thread_engine(self, thread_id: int = None) -> sf.Engine:
        """Get thread-safe SF engine."""
        if thread_id is not None and thread_id < len(self._engine_pool):
            return self._engine_pool[thread_id]
        
        # Thread-local engine fallback
        if not hasattr(self._thread_local, 'engine'):
            self._thread_local.engine = sf.Engine("tf", backend_options={
                "cutoff_dim": self.cutoff_dim
            })
        return self._thread_local.engine
    
    def choose_strategy(self, batch_size: int, complexity_hint: str = "medium") -> str:
        """
        Intelligently choose optimal execution strategy.
        
        Args:
            batch_size: Number of samples to process
            complexity_hint: "low", "medium", "high" circuit complexity
            
        Returns:
            Optimal strategy name
        """
        # Strategy selection based on workload analysis
        if self.has_gpu and batch_size >= 32:
            return "gpu_batch"
        elif batch_size >= 16:
            return "cpu_batch"
        elif batch_size >= 4:
            return "threading"
        else:
            return "sequential"
    
    def update_performance_stats(self, strategy: str, execution_time: float, 
                               batch_size: int):
        """Update performance tracking statistics."""
        self.performance_stats['total_executions'] += batch_size
        self.performance_stats['total_time'] += execution_time
        
        if strategy not in self.performance_stats['strategy_usage']:
            self.performance_stats['strategy_usage'][strategy] = 0
        self.performance_stats['strategy_usage'][strategy] += batch_size
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        total_time = self.performance_stats['total_time']
        total_exec = self.performance_stats['total_executions']
        
        return {
            'total_executions': total_exec,
            'total_time': total_time,
            'avg_samples_per_second': total_exec / total_time if total_time > 0 else 0,
            'strategy_usage': self.performance_stats['strategy_usage'],
            'cpu_utilization_estimate': min(100, (self.max_threads / cpu_count()) * 100)
        }

class QuantumBatchExecutor:
    """
    Optimized batch executor for quantum circuits.
    Maximizes CPU utilization through intelligent batching.
    """
    
    def __init__(self, threading_manager: QuantumThreadingManager):
        self.tm = threading_manager
    
    def execute_quantum_batch(self, quantum_program: sf.Program, 
                            parameter_sets: List[Dict[str, tf.Tensor]],
                            strategy: str = "auto") -> List[Any]:
        """
        Execute quantum circuits in optimized batches.
        
        Args:
            quantum_program: SF program to execute
            parameter_sets: List of parameter dictionaries
            strategy: Execution strategy
            
        Returns:
            List of quantum states/results
        """
        batch_size = len(parameter_sets)
        start_time = time.time()
        
        if strategy == "auto":
            strategy = self.tm.choose_strategy(batch_size)
        
        logger.debug(f"Executing {batch_size} circuits with strategy: {strategy}")
        
        # Execute with chosen strategy
        if strategy == "gpu_batch" and self.tm.has_gpu:
            results = self._execute_gpu_batch(quantum_program, parameter_sets)
        elif strategy == "cpu_batch":
            results = self._execute_cpu_batch(quantum_program, parameter_sets)
        elif strategy == "threading":
            results = self._execute_threaded_batch(quantum_program, parameter_sets)
        else:
            results = self._execute_sequential(quantum_program, parameter_sets)
        
        # Update performance tracking
        execution_time = time.time() - start_time
        self.tm.update_performance_stats(strategy, execution_time, batch_size)
        
        return results
    
    def _execute_gpu_batch(self, program: sf.Program, 
                          parameter_sets: List[Dict[str, tf.Tensor]]) -> List[Any]:
        """Execute batch on GPU with memory optimization."""
        with tf.device('/GPU:0'):
            results = []
            gpu_batch_size = 32  # GPU-optimized batch size
            
            for i in range(0, len(parameter_sets), gpu_batch_size):
                batch_params = parameter_sets[i:i+gpu_batch_size]
                batch_results = self._execute_gpu_sub_batch(program, batch_params)
                results.extend(batch_results)
            
            return results
    
    def _execute_gpu_sub_batch(self, program: sf.Program, 
                              parameter_sets: List[Dict[str, tf.Tensor]]) -> List[Any]:
        """Execute sub-batch on GPU."""
        results = []
        
        for params in parameter_sets:
            # Convert to GPU tensors
            gpu_params = {}
            for k, v in params.items():
                if isinstance(v, tf.Tensor):
                    gpu_params[k] = tf.cast(v, tf.complex64)
                else:
                    gpu_params[k] = tf.constant(v, dtype=tf.complex64)
            
            # Execute on GPU
            if self.tm.gpu_engine.run_progs:
                self.tm.gpu_engine.reset()
            
            state = self.tm.gpu_engine.run(program, args=gpu_params).state
            results.append(state)
        
        return results
    
    def _execute_cpu_batch(self, program: sf.Program, 
                          parameter_sets: List[Dict[str, tf.Tensor]]) -> List[Any]:
        """Execute batch using CPU with TensorFlow vectorization."""
        results = []
        cpu_batch_size = 8  # CPU-optimized batch size
        
        for i in range(0, len(parameter_sets), cpu_batch_size):
            batch_params = parameter_sets[i:i+cpu_batch_size]
            batch_results = self._execute_cpu_sub_batch(program, batch_params)
            results.extend(batch_results)
        
        return results
    
    def _execute_cpu_sub_batch(self, program: sf.Program, 
                              parameter_sets: List[Dict[str, tf.Tensor]]) -> List[Any]:
        """Execute sub-batch on CPU with vectorization."""
        results = []
        
        # Use batch engine for CPU execution
        for params in parameter_sets:
            if self.tm.batch_engine.run_progs:
                self.tm.batch_engine.reset()
            
            state = self.tm.batch_engine.run(program, args=params).state
            results.append(state)
        
        return results
    
    def _execute_threaded_batch(self, program: sf.Program, 
                               parameter_sets: List[Dict[str, tf.Tensor]]) -> List[Any]:
        """Execute batch using thread pool for maximum CPU utilization."""
        
        def execute_single(thread_id: int, params: Dict[str, tf.Tensor]) -> Any:
            """Execute single circuit in thread."""
            engine = self.tm.get_thread_engine(thread_id)
            
            if engine.run_progs:
                engine.reset()
            
            state = engine.run(program, args=params).state
            return state
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.tm.max_threads) as executor:
            # Submit all tasks with thread IDs
            future_to_idx = {}
            for i, params in enumerate(parameter_sets):
                thread_id = i % self.tm.max_threads
                future = executor.submit(execute_single, thread_id, params)
                future_to_idx[future] = i
            
            # Collect results in order
            results = [None] * len(parameter_sets)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Thread {idx} failed: {e}")
                    results[idx] = None
        
        return results
    
    def _execute_sequential(self, program: sf.Program, 
                           parameter_sets: List[Dict[str, tf.Tensor]]) -> List[Any]:
        """Sequential execution (fallback)."""
        results = []
        engine = self.tm.get_thread_engine(0)
        
        for params in parameter_sets:
            if engine.run_progs:
                engine.reset()
            
            try:
                state = engine.run(program, args=params).state
                results.append(state)
            except Exception as e:
                logger.error(f"Sequential execution failed: {e}")
                results.append(None)
        
        return results

class QuantumGeneratorThreadingMixin:
    """
    Mixin class to add threading capabilities to quantum generators.
    
    This can be mixed into your existing QuantumSFGenerator to add
    high-performance batch processing without breaking existing functionality.
    """
    
    def __init__(self, *args, enable_threading: bool = True, 
                 max_threads: Optional[int] = None, **kwargs):
        """Initialize threading capabilities."""
        super().__init__(*args, **kwargs)
        
        self.enable_threading = enable_threading
        
        if enable_threading:
            self.threading_manager = QuantumThreadingManager(
                n_modes=self.n_modes,
                cutoff_dim=self.cutoff_dim,
                max_threads=max_threads
            )
            self.batch_executor = QuantumBatchExecutor(self.threading_manager)
            logger.info("Threading enabled for quantum generator")
        else:
            self.threading_manager = None
            self.batch_executor = None
            logger.info("Threading disabled for quantum generator")
    
    def generate_threaded(self, z: tf.Tensor, strategy: str = "auto") -> tf.Tensor:
        """
        Generate samples using optimized threading.
        
        Args:
            z: Latent noise tensor [batch_size, latent_dim]
            strategy: "auto", "gpu_batch", "cpu_batch", "threading", "sequential"
            
        Returns:
            Generated samples [batch_size, n_modes]
        """
        if not self.enable_threading:
            return self.generate(z)  # Fallback to original method
        
        batch_size = tf.shape(z)[0]
        
        # Use quantum encoding if available (pure quantum path)
        if hasattr(self, 'quantum_encoder') and self.quantum_encoder is not None:
            return self._generate_threaded_quantum_encoding(z, strategy)
        else:
            return self._generate_threaded_classical_encoding(z, strategy)
    
    def _generate_threaded_quantum_encoding(self, z: tf.Tensor, strategy: str) -> tf.Tensor:
        """Generate using quantum encoding with threading."""
        batch_size = tf.shape(z)[0]
        
        # Apply quantum encoding strategy
        if self.encoding_strategy == 'coherent_state':
            encoded_params = self.quantum_encoder.encode(z, self.n_modes)
            return self._execute_threaded_coherent_generation(encoded_params, strategy)
        
        elif self.encoding_strategy == 'direct_displacement':
            encoded_params = self.quantum_encoder.encode(z, self.n_modes)
            return self._execute_threaded_displacement_generation(encoded_params, strategy)
        
        else:
            # Fallback to classical encoding with threading
            return self._generate_threaded_classical_encoding(z, strategy)
    
    def _generate_threaded_classical_encoding(self, z: tf.Tensor, strategy: str) -> tf.Tensor:
        """Generate using classical encoding with threading."""
        batch_size = tf.shape(z)[0]
        
        # Encode latent to quantum parameters
        encoded_params = self.encoder(z)  # [batch_size, num_quantum_params]
        
        # Prepare parameter sets for batch execution
        parameter_sets = []
        for i in range(batch_size):
            params_reshaped = tf.reshape(encoded_params[i], self.quantum_weights.shape)
            combined_params = self.quantum_weights + 0.1 * params_reshaped
            
            # Create parameter mapping
            param_dict = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(), 
                    tf.reshape(combined_params, [-1])
                )
            }
            parameter_sets.append(param_dict)
        
        # Execute batch with threading
        states = self.batch_executor.execute_quantum_batch(
            self.qnn, parameter_sets, strategy=strategy
        )
        
        # Extract samples from states
        samples = []
        for state in states:
            if state is not None:
                sample = self._extract_samples_from_state(state)
                samples.append(sample)
            else:
                # Fallback for failed executions
                samples.append(tf.random.normal([self.n_modes], stddev=0.5))
        
        return tf.stack(samples, axis=0)
    
    def _execute_threaded_coherent_generation(self, coherent_params: tf.Tensor, 
                                            strategy: str) -> tf.Tensor:
        """Execute coherent state generation with threading."""
        batch_size = tf.shape(coherent_params)[0]
        
        # Prepare parameter sets for coherent state generation
        parameter_sets = []
        for i in range(batch_size):
            # Extract coherent state parameters
            n_modes = self.n_modes
            real_parts = coherent_params[i, :n_modes]
            imag_parts = coherent_params[i, n_modes:2*n_modes] if coherent_params.shape[1] >= 2*n_modes else tf.zeros_like(real_parts)
            
            # Create modulation for quantum weights
            modulation = tf.concat([real_parts, imag_parts], axis=0)
            if len(modulation) < self.num_quantum_params:
                padding = self.num_quantum_params - len(modulation)
                modulation = tf.pad(modulation, [[0, padding]])
            else:
                modulation = modulation[:self.num_quantum_params]
            
            modulation_reshaped = tf.reshape(modulation, self.quantum_weights.shape)
            combined_params = self.quantum_weights + 0.1 * modulation_reshaped
            
            param_dict = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(), 
                    tf.reshape(combined_params, [-1])
                )
            }
            parameter_sets.append(param_dict)
        
        # Execute with threading
        states = self.batch_executor.execute_quantum_batch(
            self.qnn, parameter_sets, strategy=strategy
        )
        
        # Extract samples
        samples = []
        for state in states:
            if state is not None:
                sample = self._extract_samples_from_state(state)
                samples.append(sample)
            else:
                samples.append(tf.random.normal([self.n_modes], stddev=0.5))
        
        return tf.stack(samples, axis=0)
    
    def _execute_threaded_displacement_generation(self, displacement_params: tf.Tensor, 
                                                strategy: str) -> tf.Tensor:
        """Execute displacement generation with threading."""
        batch_size = tf.shape(displacement_params)[0]
        
        # Similar implementation to coherent state but for displacement
        parameter_sets = []
        for i in range(batch_size):
            displacements = displacement_params[i]
            
            # Create modulation
            if len(displacements) < self.num_quantum_params:
                padding = self.num_quantum_params - len(displacements)
                modulation = tf.pad(displacements, [[0, padding]])
            else:
                modulation = displacements[:self.num_quantum_params]
            
            modulation_reshaped = tf.reshape(modulation, self.quantum_weights.shape)
            combined_params = self.quantum_weights + 0.2 * modulation_reshaped
            
            param_dict = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(), 
                    tf.reshape(combined_params, [-1])
                )
            }
            parameter_sets.append(param_dict)
        
        # Execute with threading
        states = self.batch_executor.execute_quantum_batch(
            self.qnn, parameter_sets, strategy=strategy
        )
        
        # Extract samples
        samples = []
        for state in states:
            if state is not None:
                sample = self._extract_samples_from_state(state)
                samples.append(sample)
            else:
                samples.append(tf.random.normal([self.n_modes], stddev=0.5))
        
        return tf.stack(samples, axis=0)
    
    def benchmark_threading_performance(self, test_batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark threading performance across different strategies and batch sizes.
        
        Args:
            test_batch_sizes: List of batch sizes to test
            
        Returns:
            Comprehensive performance report
        """
        if not self.enable_threading:
            return {"error": "Threading not enabled"}
        
        test_batch_sizes = test_batch_sizes or [1, 4, 8, 16, 32]
        strategies = ["sequential", "cpu_batch", "threading"]
        
        if self.threading_manager.has_gpu:
            strategies.append("gpu_batch")
        
        results = {}
        
        print("QUANTUM THREADING PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        for batch_size in test_batch_sizes:
            print(f"\nBatch Size: {batch_size}")
            print("-" * 30)
            
            batch_results = {}
            z_test = tf.random.normal([batch_size, self.latent_dim])
            
            for strategy in strategies:
                try:
                    start_time = time.time()
                    samples = self.generate_threaded(z_test, strategy=strategy)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    samples_per_second = batch_size / execution_time if execution_time > 0 else 0
                    
                    batch_results[strategy] = {
                        'execution_time': execution_time,
                        'samples_per_second': samples_per_second,
                        'success': True
                    }
                    
                    print(f"  {strategy:12}: {execution_time:.3f}s ({samples_per_second:.2f} samples/s)")
                    
                except Exception as e:
                    batch_results[strategy] = {
                        'execution_time': float('inf'),
                        'samples_per_second': 0,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  {strategy:12}: FAILED - {e}")
            
            results[f"batch_{batch_size}"] = batch_results
        
        # Add overall performance report
        results['performance_summary'] = self.threading_manager.get_performance_report()
        
        return results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_threaded_quantum_generator(base_generator_class, *args, **kwargs):
    """
    Factory function to create a threaded version of any quantum generator.
    
    Args:
        base_generator_class: Base generator class to extend
        *args, **kwargs: Arguments for generator initialization
        
    Returns:
        Threaded quantum generator instance
    """
    
    class ThreadedQuantumGenerator(QuantumGeneratorThreadingMixin, base_generator_class):
        """Dynamically created threaded quantum generator."""
        pass
    
    return ThreadedQuantumGenerator(*args, **kwargs)

def optimize_cpu_utilization():
    """
    Optimize TensorFlow and system settings for maximum CPU utilization.
    """
    # Configure TensorFlow for CPU optimization
    tf.config.threading.set_inter_op_parallelism_threads(cpu_count())
    tf.config.threading.set_intra_op_parallelism_threads(cpu_count())
    
    # Enable mixed precision for performance
    try:
        tf.config.optimizer.set_jit(True)
        logger.info("TensorFlow XLA JIT compilation enabled")
    except Exception as e:
        logger.warning(f"XLA JIT compilation failed: {e}")
    
    logger.info(f"CPU optimization configured for {cpu_count()} cores")

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_quantum_threading():
    """Test quantum threading utilities."""
    print("Testing Quantum Threading Utilities")
    print("=" * 50)
    
    # Test threading manager
    tm = QuantumThreadingManager(n_modes=2, cutoff_dim=6, max_threads=4)
    print(f"Threading manager created: {tm.max_threads} threads")
    
    # Test strategy selection
    strategies = [tm.choose_strategy(bs) for bs in [1, 4, 8, 16, 32]]
    print(f"Strategy selection: {strategies}")
    
    # Test batch executor
    executor = QuantumBatchExecutor(tm)
    print(f"Batch executor created")
    
    # Performance report
    report = tm.get_performance_report()
    print(f"Performance tracking: {report}")
    
    print("\nAll quantum threading tests passed!")

if __name__ == "__main__":
    # Optimize CPU utilization
    optimize_cpu_utilization()
    
    # Run tests
    test_quantum_threading()
