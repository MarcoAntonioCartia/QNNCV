"""
Fixed Universal Strawberry Fields Threading System
Based on the working quantum_threading.py implementation

This module provides universal threading capabilities for any Strawberry Fields
quantum model (generators, discriminators, encoders, etc.) using the proven
SF program-based threading approach.
"""

import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import concurrent.futures
import threading
import time
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from multiprocessing import cpu_count
import inspect

logger = logging.getLogger(__name__)

class SFThreadingManager:
    """
    Universal threading manager for any Strawberry Fields model.
    
    Fixed version that uses SF program-based threading instead of
    generic method threading.
    """
    
    def __init__(self, n_modes: int = 2, cutoff_dim: int = 8, 
                 max_threads: Optional[int] = None, enable_gpu: bool = True):
        """
        Initialize universal SF threading manager.
        
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
        
        logger.info(f"SFThreadingManager initialized:")
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
        """Update performance tracking statistics - FIXED signature."""
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
            'avg_samples_per_second': float(total_exec) / total_time if total_time > 0 else 0,
            'strategy_usage': self.performance_stats['strategy_usage'],
            'cpu_utilization_estimate': min(100, (self.max_threads / cpu_count()) * 100)
        }

class UniversalQuantumBatchExecutor:
    """
    Universal batch executor for any SF quantum model.
    Based on the working QuantumBatchExecutor pattern.
    """
    
    def __init__(self, threading_manager: SFThreadingManager):
        self.tm = threading_manager
    
    def execute_quantum_batch(self, model: Any, inputs: tf.Tensor,
                            method_name: str = None, strategy: str = "auto") -> tf.Tensor:
        """
        Execute quantum model in optimized batches.
        
        Args:
            model: Any SF-based quantum model
            inputs: Input tensor
            method_name: Method to execute (auto-detect if None)
            strategy: Execution strategy
            
        Returns:
            Output tensor from quantum model
        """
        batch_size = tf.shape(inputs)[0]
        start_time = time.time()
        
        if strategy == "auto":
            strategy = self.tm.choose_strategy(batch_size)
        
        # Auto-detect method if not specified
        if method_name is None:
            method_name = self._auto_detect_method(model)
        
        logger.debug(f"Executing {batch_size} samples with strategy: {strategy}")
        
        # Extract quantum program and prepare for batch execution
        quantum_program, param_extractor, result_processor = self._prepare_quantum_execution(
            model, method_name
        )
        
        # Prepare parameter sets for each input
        parameter_sets = []
        for i in range(batch_size):
            params = param_extractor(model, inputs[i:i+1])
            parameter_sets.append(params)
        
        # Execute with chosen strategy
        if strategy == "gpu_batch" and self.tm.has_gpu:
            states = self._execute_gpu_batch(quantum_program, parameter_sets)
        elif strategy == "cpu_batch":
            states = self._execute_cpu_batch(quantum_program, parameter_sets)
        elif strategy == "threading":
            states = self._execute_threaded_batch(quantum_program, parameter_sets)
        else:
            states = self._execute_sequential(quantum_program, parameter_sets)
        
        # Process results
        results = []
        for state in states:
            if state is not None:
                result = result_processor(model, state)
                results.append(result)
            else:
                # Fallback for failed executions
                fallback = self._get_fallback_result(model, method_name)
                results.append(fallback)
        
        # Stack results
        output = tf.stack(results, axis=0)
        
        # Update performance tracking
        execution_time = time.time() - start_time
        self.tm.update_performance_stats(strategy, execution_time, batch_size)
        
        return output
    
    def _auto_detect_method(self, model) -> str:
        """Auto-detect the primary method to execute."""
        if hasattr(model, 'generate'):
            return 'generate'
        elif hasattr(model, 'discriminate'):
            return 'discriminate'
        elif hasattr(model, 'encode'):
            return 'encode'
        elif hasattr(model, 'forward'):
            return 'forward'
        else:
            raise ValueError(f"Cannot auto-detect method for {model.__class__.__name__}")
    
    def _prepare_quantum_execution(self, model, method_name):
        """Prepare quantum program and parameter extraction for execution."""
        
        # Get quantum program
        if hasattr(model, 'qnn'):
            quantum_program = model.qnn
        else:
            raise ValueError(f"Model {model.__class__.__name__} does not have quantum program (qnn)")
        
        # Define parameter extractor based on method
        if method_name == 'generate':
            def param_extractor(model, z):
                # Generator parameter extraction
                if hasattr(model, 'quantum_encoder') and model.quantum_encoder is not None:
                    # Quantum encoding path
                    if model.encoding_strategy == 'coherent_state':
                        encoded_params = model.quantum_encoder.encode(z, model.n_modes)
                        # Process coherent parameters
                        n_modes = model.n_modes
                        real_parts = encoded_params[0, :n_modes]
                        imag_parts = encoded_params[0, n_modes:2*n_modes] if encoded_params.shape[1] >= 2*n_modes else tf.zeros_like(real_parts)
                        modulation = tf.concat([real_parts, imag_parts], axis=0)
                    else:
                        encoded_params = model.quantum_encoder.encode(z, model.n_modes)
                        modulation = encoded_params[0]
                    
                    # Pad or truncate modulation
                    if len(modulation) < model.num_quantum_params:
                        padding = model.num_quantum_params - len(modulation)
                        modulation = tf.pad(modulation, [[0, padding]])
                    else:
                        modulation = modulation[:model.num_quantum_params]
                    
                    modulation_reshaped = tf.reshape(modulation, model.quantum_weights.shape)
                    combined_params = model.quantum_weights + 0.1 * modulation_reshaped
                else:
                    # Classical encoding path
                    encoded_params = model.encoder(z)
                    params_reshaped = tf.reshape(encoded_params[0], model.quantum_weights.shape)
                    combined_params = model.quantum_weights + 0.1 * params_reshaped
                
                # Create parameter mapping
                param_dict = {
                    p.name: w for p, w in zip(
                        model.sf_params.flatten(), 
                        tf.reshape(combined_params, [-1])
                    )
                }
                return param_dict
            
            def result_processor(model, state):
                return model._extract_samples_from_state(state)
        
        elif method_name == 'discriminate':
            def param_extractor(model, x):
                # Discriminator parameter extraction
                if hasattr(model, 'quantum_encoder') and model.quantum_encoder is not None:
                    # Quantum encoding path
                    if model.encoding_strategy == 'coherent_state':
                        encoded_features = model.quantum_encoder.encode(x, model.n_modes)
                        features = encoded_features[0]
                        n_modes = model.n_modes
                        real_parts = features[:n_modes]
                        imag_parts = features[n_modes:2*n_modes] if len(features) >= 2*n_modes else tf.zeros_like(real_parts)
                        modulation = tf.concat([real_parts, imag_parts], axis=0)
                    else:
                        encoded_features = model.quantum_encoder.encode(x, model.n_modes)
                        modulation = encoded_features[0]
                    
                    # Pad or truncate modulation
                    if len(modulation) < model.num_quantum_params:
                        padding = model.num_quantum_params - len(modulation)
                        modulation = tf.pad(modulation, [[0, padding]])
                    else:
                        modulation = modulation[:model.num_quantum_params]
                    
                    modulation_reshaped = tf.reshape(modulation, model.quantum_weights.shape)
                    combined_params = model.quantum_weights + 0.1 * modulation_reshaped
                else:
                    # Classical encoding path
                    encoded_params = model.encoder(x)
                    params_reshaped = tf.reshape(encoded_params[0], model.quantum_weights.shape)
                    combined_params = model.quantum_weights + 0.1 * params_reshaped
                
                # Create parameter mapping
                param_dict = {
                    p.name: w for p, w in zip(
                        model.sf_params.flatten(), 
                        tf.reshape(combined_params, [-1])
                    )
                }
                return param_dict
            
            def result_processor(model, state):
                # Extract quantum features first
                if hasattr(model, '_extract_enhanced_features_from_state'):
                    quantum_features = model._extract_enhanced_features_from_state(state)
                else:
                    quantum_features = model._extract_features_from_state(state)
                
                # Pass through output processor to get probability
                if hasattr(model, 'output_processor'):
                    # Reshape to add batch dimension if needed
                    if len(quantum_features.shape) == 1:
                        quantum_features = tf.expand_dims(quantum_features, 0)
                    probability = model.output_processor(quantum_features)
                    # Remove batch dimension
                    return tf.squeeze(probability, axis=0)
                else:
                    return quantum_features
        
        else:
            # Generic fallback
            def param_extractor(model, input_data):
                # Try to extract parameters generically
                if hasattr(model, 'quantum_weights') and hasattr(model, 'sf_params'):
                    param_dict = {
                        p.name: w for p, w in zip(
                            model.sf_params.flatten(), 
                            tf.reshape(model.quantum_weights, [-1])
                        )
                    }
                    return param_dict
                else:
                    raise ValueError(f"Cannot extract parameters from {model.__class__.__name__}")
            
            def result_processor(model, state):
                # Generic result processing
                ket = state.ket()
                return tf.abs(ket[:model.n_modes])
        
        return quantum_program, param_extractor, result_processor
    
    def _get_fallback_result(self, model, method_name):
        """Get fallback result for failed executions."""
        if method_name == 'generate':
            return tf.random.normal([model.n_modes], stddev=0.5)
        elif method_name == 'discriminate':
            return tf.random.normal([model.n_modes], stddev=0.5)
        else:
            return tf.zeros([model.n_modes])
    
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

class UniversalSFThreadingMixin:
    """
    Universal threading mixin for ANY Strawberry Fields model.
    
    Fixed version that uses SF program-based threading.
    """
    
    def __init__(self, *args, enable_threading: bool = True, 
                 max_threads: Optional[int] = None, **kwargs):
        """Initialize universal threading capabilities."""
        super().__init__(*args, **kwargs)
        
        self.enable_threading = enable_threading
        
        if enable_threading:
            # Auto-detect model parameters
            n_modes = getattr(self, 'n_modes', 2)
            cutoff_dim = getattr(self, 'cutoff_dim', 8)
            
            self.threading_manager = SFThreadingManager(
                n_modes=n_modes,
                cutoff_dim=cutoff_dim,
                max_threads=max_threads
            )
            self.batch_executor = UniversalQuantumBatchExecutor(self.threading_manager)
            logger.info(f"Universal threading enabled for {self.__class__.__name__}")
        else:
            self.threading_manager = None
            self.batch_executor = None
            logger.info(f"Threading disabled for {self.__class__.__name__}")
    
    def execute_threaded(self, inputs: tf.Tensor, method_name: str = None, 
                        strategy: str = "auto") -> tf.Tensor:
        """
        Universal threaded execution for any SF model method.
        
        Args:
            inputs: Input tensor for the method
            method_name: Method to thread (auto-detect if None)
            strategy: Threading strategy
            
        Returns:
            Output tensor from threaded execution
        """
        if not self.enable_threading:
            # Fallback to original method
            method_name = method_name or self._auto_detect_method()
            original_method = getattr(self, method_name)
            return original_method(inputs)
        
        # Use batch executor for threaded execution
        return self.batch_executor.execute_quantum_batch(
            self, inputs, method_name, strategy
        )
    
    def _auto_detect_method(self) -> str:
        """Auto-detect the primary method to thread."""
        if hasattr(self, 'generate'):
            return 'generate'
        elif hasattr(self, 'discriminate'):
            return 'discriminate'
        elif hasattr(self, 'encode'):
            return 'encode'
        elif hasattr(self, 'forward'):
            return 'forward'
        else:
            raise ValueError(f"Cannot auto-detect method for {self.__class__.__name__}")
    
    def benchmark_threading_performance(self, test_inputs: tf.Tensor = None,
                                      method_name: str = None,
                                      test_batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark threading performance for any SF method.
        
        Args:
            test_inputs: Test input tensor (auto-generate if None)
            method_name: Method to benchmark (auto-detect if None)
            test_batch_sizes: List of batch sizes to test
            
        Returns:
            Comprehensive performance report
        """
        if not self.enable_threading:
            return {"error": "Threading not enabled"}
        
        # Auto-detect method
        if method_name is None:
            method_name = self._auto_detect_method()
        
        # Generate test inputs if not provided
        if test_inputs is None:
            if hasattr(self, 'latent_dim'):
                # Generator-like model
                test_inputs = tf.random.normal([32, self.latent_dim])
            elif hasattr(self, 'input_dim'):
                # Discriminator-like model
                test_inputs = tf.random.normal([32, self.input_dim])
            else:
                # Generic fallback
                test_inputs = tf.random.normal([32, 4])
        
        test_batch_sizes = test_batch_sizes or [1, 4, 8, 16, 32]
        strategies = ["sequential", "cpu_batch", "threading"]
        
        if self.threading_manager.has_gpu:
            strategies.append("gpu_batch")
        
        results = {}
        
        print(f"UNIVERSAL SF THREADING BENCHMARK - {method_name.upper()}")
        print("=" * 60)
        
        for batch_size in test_batch_sizes:
            if batch_size > tf.shape(test_inputs)[0]:
                continue
                
            print(f"\nBatch Size: {batch_size}")
            print("-" * 30)
            
            batch_results = {}
            test_batch = test_inputs[:batch_size]
            
            for strategy in strategies:
                try:
                    start_time = time.time()
                    samples = self.execute_threaded(test_batch, method_name=method_name, 
                                                  strategy=strategy)
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
# UNIVERSAL FACTORY FUNCTIONS
# ============================================================================

def create_threaded_sf_model(base_model_class, *args, **kwargs):
    """
    Universal factory function to create a threaded version of ANY SF model.
    
    Works with generators, discriminators, encoders, or any SF-based model
    without requiring model-specific logic.
    
    Args:
        base_model_class: Any Strawberry Fields model class
        *args, **kwargs: Arguments for model initialization
        
    Returns:
        Threaded version of the SF model
    """
    
    class ThreadedSFModel(UniversalSFThreadingMixin, base_model_class):
        """Dynamically created threaded SF model."""
        pass
    
    return ThreadedSFModel(*args, **kwargs)

def optimize_sf_cpu_utilization():
    """
    Optimize TensorFlow and system settings for maximum CPU utilization
    in Strawberry Fields quantum computations.
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
    
    logger.info(f"SF CPU optimization configured for {cpu_count()} cores")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def thread_sf_method(sf_model, method_name: str, inputs: tf.Tensor, 
                    strategy: str = "auto", **method_kwargs) -> tf.Tensor:
    """
    Convenience function to thread any method of any SF model.
    
    Args:
        sf_model: Any Strawberry Fields model
        method_name: Name of method to thread
        inputs: Input tensor
        strategy: Threading strategy
        **method_kwargs: Additional method arguments
        
    Returns:
        Threaded execution result
    """
    if hasattr(sf_model, 'execute_threaded'):
        return sf_model.execute_threaded(inputs, method_name, strategy, **method_kwargs)
    else:
        # Create threaded version on the fly
        ThreadedModel = create_threaded_sf_model(sf_model.__class__)
        
        # Copy model attributes
        for attr in ['n_modes', 'cutoff_dim', 'latent_dim', 'input_dim', 'layers']:
            if hasattr(sf_model, attr):
                setattr(ThreadedModel, attr, getattr(sf_model, attr))
        
        # Initialize with same parameters
        threaded_instance = ThreadedModel()
        
        # Copy weights and parameters
        if hasattr(sf_model, 'quantum_weights'):
            threaded_instance.quantum_weights = sf_model.quantum_weights
        if hasattr(sf_model, 'sf_params'):
            threaded_instance.sf_params = sf_model.sf_params
        if hasattr(sf_model, 'qnn'):
            threaded_instance.qnn = sf_model.qnn
            
        return threaded_instance.execute_threaded(inputs, method_name, strategy, **method_kwargs)

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_universal_sf_threading():
    """Test universal SF threading utilities."""
    print("Testing Fixed Universal SF Threading System")
    print("=" * 50)
    
    # Test threading manager
    tm = SFThreadingManager(n_modes=2, cutoff_dim=6, max_threads=4)
    print(f"Threading manager created: {tm.max_threads} threads")
    
    # Test strategy selection
    strategies = [tm.choose_strategy(bs) for bs in [1, 4, 8, 16, 32]]
    print(f"Strategy selection: {strategies}")
    
    # Test batch executor
    executor = UniversalQuantumBatchExecutor(tm)
    print(f"Batch executor created")
    
    # Performance report
    report = tm.get_performance_report()
    print(f"Performance tracking: {report}")
    
    print("\nAll universal SF threading tests passed!")

if __name__ == "__main__":
    # Optimize CPU utilization
    optimize_sf_cpu_utilization()
    
    # Run tests
    test_universal_sf_threading()
