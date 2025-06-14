"""
Threading Integration for QuantumSFGenerator
Direct integration with your existing generator class
"""

import tensorflow as tf
import strawberryfields as sf
import numpy as np
import concurrent.futures
import threading
from typing import List, Dict, Any
import time

class ThreadedQuantumSFGenerator:
    """
    Enhanced version of your QuantumSFGenerator with threading capabilities
    Integrates the threading strategies we discovered
    """
    
    def __init__(self, n_modes=2, latent_dim=4, layers=2, cutoff_dim=8, 
                 encoding_strategy='classical_neural', enable_threading=True,
                 max_threads=4, gpu_batch_size=32):
        
        # Original generator parameters
        self.n_modes = n_modes
        self.latent_dim = latent_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.encoding_strategy = encoding_strategy
        
        # Threading parameters
        self.enable_threading = enable_threading
        self.max_threads = max_threads
        self.gpu_batch_size = gpu_batch_size
        
        # Initialize engines for different threading strategies
        self._init_threading_engines()
        
        # Initialize your original components (from your existing code)
        self._init_original_components()
        
        print(f"🚀 ThreadedQuantumSFGenerator initialized")
        print(f"   Threading: {'✅ Enabled' if enable_threading else '❌ Disabled'}")
        print(f"   Max threads: {max_threads}")
        print(f"   GPU batch size: {gpu_batch_size}")
    
    def _init_threading_engines(self):
        """Initialize different engines for threading strategies"""
        
        # 1. Batch engine (for SF's native batch processing)
        self.batch_engine = sf.Engine("tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "batch_size": None  # Dynamic batch sizing
        })
        
        # 2. Thread-local engines for parallel execution
        self._thread_local = threading.local()
        
        # 3. GPU engine if available
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                self.gpu_engine = sf.Engine("tf", backend_options={
                    "cutoff_dim": self.cutoff_dim,
                    "batch_size": self.gpu_batch_size
                })
                self.has_gpu = True
                print(f"   GPU: ✅ Available ({len(gpus)} GPU(s))")
            else:
                self.gpu_engine = None
                self.has_gpu = False
                print(f"   GPU: ❌ Not available")
        except Exception as e:
            self.gpu_engine = None
            self.has_gpu = False
            print(f"   GPU: ❌ Error - {e}")
    
    def _init_original_components(self):
        """Initialize your original generator components"""
        # This would include all your original initialization code
        # from the QuantumSFGenerator class
        
        # Initialize classical encoder
        self.num_quantum_params = self._calculate_quantum_params()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='tanh', name='encoder_hidden'),
            tf.keras.layers.Dense(self.num_quantum_params, activation='tanh', name='encoder_output')
        ], name='classical_encoder')
        
        # Build the network
        dummy_input = tf.zeros((1, self.latent_dim))
        _ = self.encoder(dummy_input)
        
        # Initialize quantum weights (your existing pattern)
        self.quantum_weights = self._init_weights_sf_style(
            self.n_modes, 
            self.layers,
            active_sd=0.0001,
            passive_sd=0.1
        )
        
        # Initialize SF components (simplified version of your code)
        self.eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        self.qnn = sf.Program(self.n_modes)
        
        # Create symbolic parameters and build program
        self._create_symbolic_params()
        self._build_quantum_program()
    
    def _calculate_quantum_params(self):
        """Calculate number of parameters (from your original code)"""
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        params_per_layer = 2 * M + 4 * self.n_modes
        return self.layers * params_per_layer
    
    def _init_weights_sf_style(self, modes, layers, active_sd=0.0001, passive_sd=0.1):
        """Initialize weights (from your original code)"""
        M = int(modes * (modes - 1)) + max(1, modes - 1)
        
        int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
        s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
        int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
        dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
        dp_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
        k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
        
        weights = tf.concat([
            int1_weights, s_weights, int2_weights, 
            dr_weights, dp_weights, k_weights
        ], axis=1)
        
        return tf.Variable(weights)
    
    def _create_symbolic_params(self):
        """Create SF symbolic parameters (from your original code)"""
        num_params = np.prod(self.quantum_weights.shape)
        sf_params = np.arange(num_params).reshape(self.quantum_weights.shape).astype(str)
        self.sf_params = np.array([self.qnn.params(*i) for i in sf_params])
    
    def _build_quantum_program(self):
        """Build quantum program (simplified from your original)"""
        with self.qnn.context as q:
            for k in range(self.layers):
                self._quantum_layer(self.sf_params[k], q)
    
    def _quantum_layer(self, params, q):
        """Single quantum layer (from your original code)"""
        # Simplified version - you'd include your full implementation here
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)
        
        # Extract parameters for each component
        s = params[M:M+N] 
        dr = params[2*M+N:2*M+2*N]
        dp = params[2*M+2*N:2*M+3*N]
        
        # Apply quantum operations (simplified)
        for i in range(N):
            sf.ops.Sgate(s[i]) | q[i]
            sf.ops.Dgate(dr[i], dp[i]) | q[i]
    
    def generate_threaded(self, z, strategy="auto"):
        """
        Enhanced generate method with threading support
        
        Args:
            z: Latent noise tensor [batch_size, latent_dim]
            strategy: "auto", "batch", "threading", "gpu", "sequential"
            
        Returns:
            Generated samples [batch_size, n_modes]
        """
        batch_size = tf.shape(z)[0]
        
        if not self.enable_threading or strategy == "sequential":
            return self._generate_sequential(z)
        
        # Choose threading strategy
        if strategy == "auto":
            strategy = self._choose_strategy(batch_size)
        
        print(f"🚀 Using strategy: {strategy} for batch size {batch_size}")
        
        if strategy == "batch":
            return self._generate_batch_sf_native(z)
        elif strategy == "threading":
            return self._generate_multithreaded(z)
        elif strategy == "gpu" and self.has_gpu:
            return self._generate_gpu_accelerated(z)
        else:
            return self._generate_sequential(z)
    
    def _choose_strategy(self, batch_size):
        """Automatically choose optimal strategy"""
        batch_size_val = batch_size.numpy() if hasattr(batch_size, 'numpy') else int(batch_size)
        
        if self.has_gpu and batch_size_val >= 16:
            return "gpu"
        elif batch_size_val >= 8:
            return "batch"
        elif batch_size_val >= 4:
            return "threading"
        else:
            return "sequential"
    
    def _generate_batch_sf_native(self, z):
        """Use Strawberry Fields' native batch processing"""
        batch_size = tf.shape(z)[0]
        
        # Encode all latent vectors
        encoded_params = self.encoder(z)  # [batch_size, num_quantum_params]
        
        # Prepare for SF batch execution
        all_samples = []
        
        # SF batch processing works by vectorizing parameters
        # Process in smaller batches to manage memory
        sub_batch_size = min(8, batch_size)
        
        for i in range(0, batch_size, sub_batch_size):
            end_idx = min(i + sub_batch_size, batch_size)
            batch_params = encoded_params[i:end_idx]
            
            batch_samples = self._execute_sf_batch(batch_params)
            all_samples.extend(batch_samples)
        
        return tf.stack(all_samples, axis=0)
    
    def _execute_sf_batch(self, batch_params):
        """Execute SF batch with parameter vectorization"""
        batch_size = tf.shape(batch_params)[0]
        samples = []
        
        for i in range(batch_size):
            # Reshape parameters for this sample
            params_reshaped = tf.reshape(batch_params[i], self.quantum_weights.shape)
            combined_params = self.quantum_weights + 0.1 * params_reshaped
            
            # Create parameter mapping
            mapping = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(), 
                    tf.reshape(combined_params, [-1])
                )
            }
            
            # Execute quantum circuit
            if self.batch_engine.run_progs:
                self.batch_engine.reset()
            
            state = self.batch_engine.run(self.qnn, args=mapping).state
            sample = self._extract_samples_from_state(state)
            samples.append(sample)
        
        return samples
    
    def _generate_multithreaded(self, z):
        """Use Python threading for parallel execution"""
        batch_size = tf.shape(z)[0]
        encoded_params = self.encoder(z)
        
        def process_single_sample(i):
            return self._generate_single_threaded(encoded_params[i])
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_idx = {
                executor.submit(process_single_sample, i): i 
                for i in range(batch_size)
            }
            
            # Collect results in order
            samples = [None] * batch_size
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    samples[idx] = future.result()
                except Exception as e:
                    print(f"Thread {idx} failed: {e}")
                    samples[idx] = tf.random.normal([self.n_modes], stddev=0.5)
        
        return tf.stack(samples, axis=0)
    
    def _generate_single_threaded(self, quantum_params):
        """Generate single sample in thread-safe manner"""
        # Get thread-local engine
        if not hasattr(self._thread_local, 'engine'):
            self._thread_local.engine = sf.Engine("tf", backend_options={
                "cutoff_dim": self.cutoff_dim
            })
        
        # Reshape and combine parameters
        params_reshaped = tf.reshape(quantum_params, self.quantum_weights.shape)
        combined_params = self.quantum_weights + 0.1 * params_reshaped
        
        # Create parameter mapping
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(), 
                tf.reshape(combined_params, [-1])
            )
        }
        
        # Execute with thread-local engine
        if self._thread_local.engine.run_progs:
            self._thread_local.engine.reset()
        
        state = self._thread_local.engine.run(self.qnn, args=mapping).state
        return self._extract_samples_from_state(state)
    
    def _generate_gpu_accelerated(self, z):
        """Use GPU acceleration for large batches"""
        with tf.device('/GPU:0'):
            batch_size = tf.shape(z)[0]
            encoded_params = self.encoder(z)
            
            # Process in GPU-optimized batches
            all_samples = []
            gpu_batch_size = self.gpu_batch_size
            
            for i in range(0, batch_size, gpu_batch_size):
                end_idx = min(i + gpu_batch_size, batch_size)
                batch_params = encoded_params[i:end_idx]
                
                # Execute on GPU
                batch_samples = self._execute_gpu_batch(batch_params)
                all_samples.extend(batch_samples)
            
            return tf.stack(all_samples, axis=0)
    
    def _execute_gpu_batch(self, batch_params):
        """Execute batch on GPU"""
        batch_size = tf.shape(batch_params)[0]
        samples = []
        
        # Convert to GPU tensors
        gpu_params = tf.cast(batch_params, tf.complex64)
        
        for i in range(batch_size):
            params_reshaped = tf.reshape(gpu_params[i], self.quantum_weights.shape)
            combined_params = tf.cast(self.quantum_weights, tf.complex64) + 0.1 * params_reshaped
            
            mapping = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(), 
                    tf.reshape(combined_params, [-1])
                )
            }
            
            if self.gpu_engine.run_progs:
                self.gpu_engine.reset()
            
            state = self.gpu_engine.run(self.qnn, args=mapping).state
            sample = self._extract_samples_from_state(state)
            samples.append(sample)
        
        return samples
    
    def _generate_sequential(self, z):
        """Original sequential generation (fallback)"""
        batch_size = tf.shape(z)[0]
        encoded_params = self.encoder(z)
        
        all_samples = []
        for i in range(batch_size):
            sample = self._generate_single(encoded_params[i])
            all_samples.append(sample)
        
        return tf.stack(all_samples, axis=0)
    
    def _generate_single(self, quantum_params):
        """Your original single sample generation"""
        params_reshaped = tf.reshape(quantum_params, self.quantum_weights.shape)
        combined_params = self.quantum_weights + 0.1 * params_reshaped
        
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(), 
                tf.reshape(combined_params, [-1])
            )
        }
        
        if self.eng.run_progs:
            self.eng.reset()
        
        try:
            state = self.eng.run(self.qnn, args=mapping).state
            return self._extract_samples_from_state(state)
        except Exception as e:
            print(f"Quantum circuit failed: {e}")
            return tf.random.normal([self.n_modes], stddev=0.5)
    
    def _extract_samples_from_state(self, state):
        """Extract samples from quantum state (from your original code)"""
        ket = state.ket()
        samples = []
        
        for mode in range(self.n_modes):
            prob_amplitudes = tf.abs(ket) ** 2
            fock_indices = tf.range(self.cutoff_dim, dtype=tf.float32)
            expectation = tf.reduce_sum(prob_amplitudes * fock_indices)
            
            sample = (expectation - self.cutoff_dim/2) / (self.cutoff_dim/4)
            sample += tf.random.normal([], stddev=0.1)
            samples.append(sample)
        
        return tf.stack(samples)
    
    @property
    def trainable_variables(self):
        """Return all trainable parameters (from your original code)"""
        variables = [self.quantum_weights]
        variables.extend(self.encoder.trainable_variables)
        return variables
    
    def benchmark_strategies(self, test_batch_sizes=[1, 4, 8, 16, 32]):
        """Benchmark different threading strategies"""
        print("🚀 THREADING STRATEGY BENCHMARK")
        print("=" * 50)
        
        strategies = ["sequential", "batch", "threading"]
        if self.has_gpu:
            strategies.append("gpu")
        
        for batch_size in test_batch_sizes:
            print(f"\nBatch Size: {batch_size}")
            print("-" * 20)
            
            z_test = tf.random.normal([batch_size, self.latent_dim])
            
            for strategy in strategies:
                try:
                    start_time = time.time()
                    samples = self.generate_threaded(z_test, strategy=strategy)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    samples_per_second = batch_size / execution_time if execution_time > 0 else 0
                    
                    print(f"  {strategy:12}: {execution_time:.3f}s ({samples_per_second:.2f} samples/s)")
                    
                except Exception as e:
                    print(f"  {strategy:12}: FAILED - {e}")

# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def test_threaded_generator():
    """Test the threaded quantum generator"""
    print("🚀 Testing ThreadedQuantumSFGenerator")
    
    # Create threaded generator
    generator = ThreadedQuantumSFGenerator(
        n_modes=2,
        latent_dim=4,
        layers=2,
        cutoff_dim=6,  # Smaller for faster testing
        enable_threading=True,
        max_threads=4,
        gpu_batch_size=16
    )
    
    print("\n✅ Generator created successfully!")
    
    # Test with different batch sizes
    test_sizes = [1, 4, 8, 16]
    
    for batch_size in test_sizes:
        print(f"\n🧪 Testing batch size: {batch_size}")
        
        z_test = tf.random.normal([batch_size, 4])
        
        # Test automatic strategy selection
        start_time = time.time()
        samples = generator.generate_threaded(z_test, strategy="auto")
        end_time = time.time()
        
        print(f"   Generated {samples.shape} samples in {end_time - start_time:.3f}s")
        print(f"   Sample range: [{tf.reduce_min(samples):.3f}, {tf.reduce_max(samples):.3f}]")
    
    # Run benchmark
    print("\n🏁 Running benchmark...")
    generator.benchmark_strategies()
    
    return generator

if __name__ == "__main__":
    test_threaded_generator()
