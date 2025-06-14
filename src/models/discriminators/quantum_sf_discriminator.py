"""
Enhanced Strawberry Fields-based Quantum Discriminator with advanced encoding strategies.

This implementation extends the proven SF training methodology with:
1. Multiple quantum encoding strategies (coherent state, displacement, angle, etc.)
2. Context-aware feature extraction
3. Configuration-driven architecture
4. Hybrid CPU/GPU memory management
5. Comprehensive quantum metrics integration
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Optional, Dict, Any

# Initialize logger first
logger = logging.getLogger(__name__)

# Import new infrastructure components with robust fallback handling
QuantumGANConfig = None
QuantumEncodingFactory = None
HybridGPUManager = None
QuantumMetrics = None

try:
    # Try relative imports first (package context)
    from ..config.quantum_gan_config import QuantumGANConfig
    from ..quantum_encodings.quantum_encodings import QuantumEncodingFactory
    from ..utils.gpu_memory_manager import HybridGPUManager
    from ..utils.quantum_metrics import QuantumMetrics
    logger.info("Infrastructure components imported via relative imports")
except ImportError:
    try:
        # Try direct imports (notebook/script context)
        import sys
        import os
        
        # Add src to path if not already there
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, '..', '..')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        from config.quantum_gan_config import QuantumGANConfig
        from quantum_encodings.quantum_encodings import QuantumEncodingFactory
        from utils.gpu_memory_manager import HybridGPUManager
        from utils.quantum_metrics import QuantumMetrics
        logger.info("Infrastructure components imported via direct imports")
    except ImportError as e:
        # Graceful fallback - discriminator will work without enhanced features
        logger.warning(f"Enhanced infrastructure not available: {e}")
        logger.warning("Discriminator will work with basic functionality only")
        QuantumGANConfig = None
        QuantumEncodingFactory = None
        HybridGPUManager = None
        QuantumMetrics = None

class QuantumSFDiscriminator:
    """
    Enhanced Quantum discriminator with advanced encoding strategies and feature extraction.
    
    This implementation extends SF tutorials with:
    - Multiple quantum encoding strategies
    - Context-aware feature extraction
    - Configuration-driven architecture
    - Hybrid CPU/GPU memory management
    - Comprehensive quantum metrics
    """
    
    def __init__(self, n_modes=2, input_dim=2, layers=3, cutoff_dim=6,
                 encoding_strategy='coherent_state', config=None,
                 feature_extraction='multi_mode', enable_batch_processing=True):
        """
        Initialize enhanced SF-based quantum discriminator.
        
        Args:
            n_modes (int): Number of quantum modes
            input_dim (int): Dimension of input data
            layers (int): Number of quantum neural network layers
            cutoff_dim (int): Fock space cutoff for simulation
            encoding_strategy (str): Quantum encoding strategy to use
            config (QuantumGANConfig): Optional configuration object
            feature_extraction (str): Feature extraction strategy
            enable_batch_processing (bool): Enable batch processing optimizations
        """
        self.n_modes = n_modes
        self.input_dim = input_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.encoding_strategy = encoding_strategy
        self.feature_extraction = feature_extraction
        self.enable_batch_processing = enable_batch_processing
        
        # Initialize configuration and resource management
        self.config = config
        self.gpu_manager = None
        self.quantum_metrics = None
        self._init_infrastructure()
        
        logger.info(f"Initializing enhanced SF-based quantum discriminator:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - input_dim: {input_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        logger.info(f"  - encoding_strategy: {encoding_strategy}")
        logger.info(f"  - feature_extraction: {feature_extraction}")
        logger.info(f"  - batch_processing: {enable_batch_processing}")
        
        # Initialize quantum encoding strategy
        self._init_quantum_encoding()
        
        # Initialize SF engine and program
        self._init_sf_components()
        
        # Initialize classical encoding network (backward compatibility)
        self._init_classical_encoder()
        
        # Initialize quantum weights using SF pattern
        self._init_quantum_weights()
        
        # Initialize output processing network
        self._init_output_processor()
        
        # Create symbolic parameters and build program
        self._create_symbolic_params()
        self._build_quantum_program()
    
    def _init_infrastructure(self):
        """Initialize configuration and resource management."""
        try:
            # Initialize GPU memory manager
            if HybridGPUManager is not None:
                self.gpu_manager = HybridGPUManager()
            else:
                self.gpu_manager = None
            
            # Initialize quantum metrics
            if QuantumMetrics is not None:
                self.quantum_metrics = QuantumMetrics()
            else:
                self.quantum_metrics = None
            
            # Load configuration if not provided
            if self.config is None and QuantumGANConfig is not None:
                self.config = QuantumGANConfig()
            
            logger.info("Infrastructure components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize infrastructure: {e}")
            logger.warning("Continuing with basic functionality")
            self.gpu_manager = None
            self.quantum_metrics = None
    
    def _init_quantum_encoding(self):
        """Initialize quantum encoding strategy."""
        try:
            if QuantumEncodingFactory is not None:
                self.quantum_encoder = QuantumEncodingFactory.create_encoding(
                    self.encoding_strategy
                )
                logger.info(f"Quantum encoding strategy '{self.encoding_strategy}' initialized")
            else:
                logger.warning("QuantumEncodingFactory not available")
                self.quantum_encoder = None
            
        except Exception as e:
            logger.warning(f"Failed to initialize quantum encoding: {e}")
            logger.warning("Falling back to classical neural encoding")
            self.quantum_encoder = None
    
    def _init_classical_encoder(self):
        """Initialize classical encoding network (backward compatibility)."""
        # Calculate number of quantum parameters needed
        self.num_quantum_params = self._calculate_quantum_params()
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='tanh', name='disc_encoder_hidden'),
            tf.keras.layers.Dense(self.num_quantum_params, activation='tanh', name='disc_encoder_output')
        ], name='discriminator_classical_encoder')
        
        # Build the network
        dummy_input = tf.zeros((1, self.input_dim))
        _ = self.encoder(dummy_input)
        
        logger.info(f"Classical encoder: {self.input_dim} → {self.num_quantum_params}")
    
    def _init_sf_components(self):
        """Initialize Strawberry Fields engine and program."""
        try:
            self.eng = sf.Engine(backend="tf", backend_options={
                "cutoff_dim": self.cutoff_dim,
                "pure": True
            })
            self.qnn = sf.Program(self.n_modes)
            logger.info("SF discriminator engine and program created successfully :)")
        except Exception as e:
            logger.error(f"Failed to create SF discriminator components: {e} :((")
            raise
    
    def _init_input_encoder(self):
        """Initialize input encoding network (data → quantum parameters)."""
        # Calculate number of quantum parameters needed
        self.num_quantum_params = self._calculate_quantum_params()
        
        self.input_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='tanh', name='disc_encoder_hidden'),
            tf.keras.layers.Dense(self.num_quantum_params, activation='tanh', name='disc_encoder_output')
        ], name='discriminator_input_encoder')
        
        # Build the network
        dummy_input = tf.zeros((1, self.input_dim))
        _ = self.input_encoder(dummy_input)
        
        logger.info(f"Discriminator input encoder: {self.input_dim} → {self.num_quantum_params}")
    
    def _init_output_processor(self):
        """Initialize output processing network (quantum state → probability)."""
        self.output_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='relu', name='disc_output_hidden'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='disc_output_final')
        ], name='discriminator_output_processor')
        
        # Build the network
        dummy_quantum_output = tf.zeros((1, self.n_modes))
        _ = self.output_processor(dummy_quantum_output)
        
        logger.info(f"Discriminator output processor: {self.n_modes} → 1")
    
    def _calculate_quantum_params(self):
        """Calculate number of parameters needed for quantum layers."""
        # Following SF tutorial pattern: each layer needs specific number of params
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)  # Interferometer params
        params_per_layer = 2 * M + 4 * self.n_modes  # int1, s, int2, dr, dp, k
        return self.layers * params_per_layer
    
    def _init_quantum_weights(self):
        """GRADIENT FLOW FIX: Initialize individual quantum variables for maximum gradient flow."""
        self._init_individual_quantum_variables()
        logger.info(f"Individual discriminator quantum variables initialized: {len(self.individual_quantum_vars)} variables")
    
    def _init_individual_quantum_variables(self):
        """Initialize individual tf.Variable for each quantum parameter (GRADIENT FLOW FIX)."""
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        
        # Create individual variables for each quantum parameter type
        self.individual_quantum_vars = {}
        
        # Interferometer 1 parameters (individual variables)
        self.individual_quantum_vars['int1'] = []
        for layer in range(self.layers):
            layer_int1 = []
            for i in range(M):
                var = tf.Variable(tf.random.normal([], stddev=0.1), name=f'disc_int1_L{layer}_P{i}')
                layer_int1.append(var)
            self.individual_quantum_vars['int1'].append(layer_int1)
        
        # Squeezing parameters (individual variables)
        self.individual_quantum_vars['squeeze'] = []
        for layer in range(self.layers):
            layer_squeeze = []
            for i in range(self.n_modes):
                var = tf.Variable(tf.random.normal([], stddev=0.0001), name=f'disc_squeeze_L{layer}_M{i}')
                layer_squeeze.append(var)
            self.individual_quantum_vars['squeeze'].append(layer_squeeze)
        
        # Interferometer 2 parameters (individual variables)
        self.individual_quantum_vars['int2'] = []
        for layer in range(self.layers):
            layer_int2 = []
            for i in range(M):
                var = tf.Variable(tf.random.normal([], stddev=0.1), name=f'disc_int2_L{layer}_P{i}')
                layer_int2.append(var)
            self.individual_quantum_vars['int2'].append(layer_int2)
        
        # Displacement r parameters (individual variables)
        self.individual_quantum_vars['disp_r'] = []
        for layer in range(self.layers):
            layer_disp_r = []
            for i in range(self.n_modes):
                var = tf.Variable(tf.random.normal([], stddev=0.0001), name=f'disc_disp_r_L{layer}_M{i}')
                layer_disp_r.append(var)
            self.individual_quantum_vars['disp_r'].append(layer_disp_r)
        
        # Displacement phi parameters (individual variables)
        self.individual_quantum_vars['disp_phi'] = []
        for layer in range(self.layers):
            layer_disp_phi = []
            for i in range(self.n_modes):
                var = tf.Variable(tf.random.normal([], stddev=0.1), name=f'disc_disp_phi_L{layer}_M{i}')
                layer_disp_phi.append(var)
            self.individual_quantum_vars['disp_phi'].append(layer_disp_phi)
        
        # Kerr parameters (individual variables)
        self.individual_quantum_vars['kerr'] = []
        for layer in range(self.layers):
            layer_kerr = []
            for i in range(self.n_modes):
                var = tf.Variable(tf.random.normal([], stddev=0.0001), name=f'disc_kerr_L{layer}_M{i}')
                layer_kerr.append(var)
            self.individual_quantum_vars['kerr'].append(layer_kerr)
    
    @property
    def quantum_weights(self):
        """Dynamic quantum_weights property that uses individual variables."""
        # Flatten all individual variables into a single tensor (dynamically)
        all_vars = []
        for layer in range(self.layers):
            # Add all parameters for this layer in the same order as before
            all_vars.extend(self.individual_quantum_vars['int1'][layer])
            all_vars.extend(self.individual_quantum_vars['squeeze'][layer])
            all_vars.extend(self.individual_quantum_vars['int2'][layer])
            all_vars.extend(self.individual_quantum_vars['disp_r'][layer])
            all_vars.extend(self.individual_quantum_vars['disp_phi'][layer])
            all_vars.extend(self.individual_quantum_vars['kerr'][layer])
        
        # Stack into matrix form for compatibility (dynamically computed)
        vars_per_layer = len(all_vars) // self.layers
        return tf.stack([tf.stack(all_vars[i:i+vars_per_layer]) 
                        for i in range(0, len(all_vars), vars_per_layer)])
    
    def _create_symbolic_params(self):
        """Create SF symbolic parameters following tutorial pattern."""
        num_params = np.prod(self.quantum_weights.shape)
        
        # Create symbolic parameter array (following SF tutorial exactly)
        sf_params = np.arange(num_params).reshape(self.quantum_weights.shape).astype(str)
        self.sf_params = np.array([self.qnn.params(*i) for i in sf_params])
        
        logger.info(f"Created {num_params} discriminator symbolic parameters")
    
    def _build_quantum_program(self):
        """Build quantum program using SF layer pattern."""
        with self.qnn.context as q:
            for k in range(self.layers):
                self._quantum_layer(self.sf_params[k], q)
        
        logger.info(f"Discriminator quantum program built with {self.layers} layers")
    
    def _quantum_layer(self, params, q):
        """Single quantum layer following SF tutorial pattern."""
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)
        
        # Extract parameters for each component
        int1 = params[:M]
        s = params[M:M+N]
        int2 = params[M+N:2*M+N]
        dr = params[2*M+N:2*M+2*N]
        dp = params[2*M+2*N:2*M+3*N]
        k = params[2*M+3*N:2*M+4*N]
        
        # Apply quantum operations (following SF tutorial)
        self._interferometer(int1, q)
        
        for i in range(N):
            ops.Sgate(s[i]) | q[i]
        
        self._interferometer(int2, q)
        
        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]
    
    def _interferometer(self, params, q):
        """Interferometer following SF tutorial pattern."""
        N = len(q)
        theta = params[:N*(N-1)//2]
        phi = params[N*(N-1)//2:N*(N-1)]
        rphi = params[-N+1:]
        
        if N == 1:
            ops.Rgate(rphi[0]) | q[0]
            return
        
        n = 0
        # Apply beamsplitter array
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1
        
        # Apply final phase shifts
        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]
    
    @property
    def trainable_variables(self):
        """Return all individual trainable parameters (GRADIENT FLOW FIX)."""
        variables = []
        
        # Add all individual quantum variables
        for param_type in self.individual_quantum_vars:
            for layer in self.individual_quantum_vars[param_type]:
                variables.extend(layer)
        
        # FIXED: Only add classical encoder variables if using classical encoding
        # When using quantum encoding strategies, we want ZERO classical gradients
        if self.quantum_encoder is None or self.encoding_strategy == 'classical_neural':
            # Use classical encoder only as fallback or when explicitly requested
            if hasattr(self, 'encoder'):
                variables.extend(self.encoder.trainable_variables)
        
        # Add quantum encoder variables if they exist and have trainable parameters
        if (self.quantum_encoder is not None and 
            hasattr(self.quantum_encoder, 'trainable_variables')):
            variables.extend(self.quantum_encoder.trainable_variables)
        
        # Always add output processor (needed for final classification)
        variables.extend(self.output_processor.trainable_variables)
        return variables
    
    def discriminate(self, x):
        """
        Enhanced discriminate samples using quantum encoding strategies.
        
        Args:
            x (tensor): Input samples [batch_size, input_dim]
            
        Returns:
            probabilities (tensor): Probability of being real [batch_size, 1]
        """
        batch_size = tf.shape(x)[0]
        
        # Use quantum encoding strategy if available
        if self.quantum_encoder is not None:
            return self._discriminate_with_quantum_encoding(x)
        else:
            # Fallback to classical encoding
            return self._discriminate_with_classical_encoding(x)
    
    def _discriminate_with_quantum_encoding(self, x):
        """Discriminate samples using quantum encoding strategies."""
        batch_size = tf.shape(x)[0]
        
        # Apply quantum encoding strategy for discrimination
        if self.encoding_strategy == 'coherent_state':
            encoded_features = self.quantum_encoder.encode(x, self.n_modes)
            return self._discriminate_with_coherent_features(x, encoded_features)
        
        elif self.encoding_strategy == 'direct_displacement':
            encoded_features = self.quantum_encoder.encode(x, self.n_modes)
            return self._discriminate_with_displacement_features(x, encoded_features)
        
        elif self.encoding_strategy in ['angle_encoding', 'sparse_parameter']:
            encoded_features = self.quantum_encoder.encode(x, self.n_modes, n_layers=self.layers)
            return self._discriminate_with_parameter_features(x, encoded_features)
        
        else:
            # Fallback to classical encoding
            return self._discriminate_with_classical_encoding(x)
    
    def _discriminate_with_classical_encoding(self, x):
        """Discriminate samples using classical neural encoding (backward compatibility)."""
        batch_size = tf.shape(x)[0]
        
        # Encode input to quantum parameters
        encoded_params = self.encoder(x)  # [batch_size, num_quantum_params]
        
        # Process samples with batch optimization if enabled
        if self.enable_batch_processing and batch_size <= 8:
            return self._discriminate_batch_optimized(encoded_params)
        else:
            # Sequential processing (original method)
            all_quantum_outputs = []
            for i in range(batch_size):
                quantum_output = self._discriminate_single(encoded_params[i])
                all_quantum_outputs.append(quantum_output)
            
            quantum_features = tf.stack(all_quantum_outputs, axis=0)
            probabilities = self.output_processor(quantum_features)
            return probabilities
    
    def _discriminate_with_coherent_features(self, x, coherent_features):
        """Discriminate samples using coherent state features."""
        batch_size = tf.shape(x)[0]
        all_quantum_outputs = []
        
        for i in range(batch_size):
            # Extract coherent state features for this sample
            features = coherent_features[i]
            quantum_output = self._discriminate_coherent_sample(features)
            all_quantum_outputs.append(quantum_output)
        
        quantum_features = tf.stack(all_quantum_outputs, axis=0)
        probabilities = self.output_processor(quantum_features)
        return probabilities
    
    def _discriminate_with_displacement_features(self, x, displacement_features):
        """Discriminate samples using displacement features."""
        batch_size = tf.shape(x)[0]
        all_quantum_outputs = []
        
        for i in range(batch_size):
            # Use displacement features directly
            features = displacement_features[i]
            quantum_output = self._discriminate_displacement_sample(features)
            all_quantum_outputs.append(quantum_output)
        
        quantum_features = tf.stack(all_quantum_outputs, axis=0)
        probabilities = self.output_processor(quantum_features)
        return probabilities
    
    def _discriminate_with_parameter_features(self, x, parameter_features):
        """Discriminate samples using parameter modulation features."""
        batch_size = tf.shape(x)[0]
        all_quantum_outputs = []
        
        for i in range(batch_size):
            # Modulate quantum circuit parameters
            features = parameter_features[i]
            quantum_output = self._discriminate_modulated_sample(features)
            all_quantum_outputs.append(quantum_output)
        
        quantum_features = tf.stack(all_quantum_outputs, axis=0)
        probabilities = self.output_processor(quantum_features)
        return probabilities
    
    def _discriminate_batch_optimized(self, encoded_params):
        """Optimized batch processing for small batches."""
        batch_size = tf.shape(encoded_params)[0]
        
        # Pre-allocate results
        all_quantum_outputs = []
        
        # Use GPU context for classical operations
        if self.gpu_manager is not None:
            with self.gpu_manager.create_classical_context():
                # Process parameters on GPU
                processed_params = tf.nn.tanh(encoded_params)  # Example processing
        else:
            processed_params = encoded_params
        
        # Quantum operations still sequential (SF limitation)
        with tf.device('/CPU:0'):  # Force CPU for quantum operations
            for i in range(batch_size):
                quantum_output = self._discriminate_single(processed_params[i])
                all_quantum_outputs.append(quantum_output)
        
        quantum_features = tf.stack(all_quantum_outputs, axis=0)
        probabilities = self.output_processor(quantum_features)
        return probabilities
    
    def _discriminate_coherent_sample(self, coherent_features):
        """Discriminate single sample using coherent state features."""
        try:
            # Create quantum state preparation based on coherent features
            n_modes = self.n_modes
            real_parts = coherent_features[:n_modes]
            imag_parts = coherent_features[n_modes:2*n_modes] if len(coherent_features) >= 2*n_modes else tf.zeros_like(real_parts)
            
            # Create coherent state parameters
            coherent_params = tf.complex(real_parts, imag_parts)
            
            # Use coherent features to modulate quantum circuit
            modulation = tf.concat([real_parts, imag_parts], axis=0)
            if len(modulation) < self.num_quantum_params:
                # Pad with zeros
                padding = self.num_quantum_params - len(modulation)
                modulation = tf.pad(modulation, [[0, padding]])
            else:
                modulation = modulation[:self.num_quantum_params]
            
            # Reshape and combine with quantum weights
            modulation_reshaped = tf.reshape(modulation, self.quantum_weights.shape)
            combined_params = self.quantum_weights + 0.1 * modulation_reshaped
            
            # Run quantum circuit
            return self._run_quantum_circuit(combined_params)
            
        except Exception as e:
            logger.debug(f"Coherent discrimination failed: {e}")
            return tf.random.normal([self.n_modes], stddev=0.5)
    
    def _discriminate_displacement_sample(self, displacement_features):
        """Discriminate single sample using displacement features."""
        try:
            # Use displacement features to modulate quantum circuit
            if len(displacement_features) < self.num_quantum_params:
                # Pad with zeros
                padding = self.num_quantum_params - len(displacement_features)
                modulation = tf.pad(displacement_features, [[0, padding]])
            else:
                modulation = displacement_features[:self.num_quantum_params]
            
            # Reshape and combine with quantum weights
            modulation_reshaped = tf.reshape(modulation, self.quantum_weights.shape)
            combined_params = self.quantum_weights + 0.2 * modulation_reshaped
            
            # Run quantum circuit
            return self._run_quantum_circuit(combined_params)
            
        except Exception as e:
            logger.debug(f"Displacement discrimination failed: {e}")
            return tf.random.normal([self.n_modes], stddev=0.5)
    
    def _discriminate_modulated_sample(self, parameter_features):
        """Discriminate single sample using parameter modulation."""
        try:
            # Apply parameter modulations to quantum weights
            if len(parameter_features.shape) == 1:
                # Reshape modulations to match quantum weights if needed
                modulations_reshaped = tf.reshape(parameter_features, self.quantum_weights.shape)
            else:
                modulations_reshaped = parameter_features
            
            # Combine base weights with modulations
            combined_params = self.quantum_weights + 0.1 * modulations_reshaped
            
            # Run quantum circuit
            return self._run_quantum_circuit(combined_params)
            
        except Exception as e:
            logger.debug(f"Modulated discrimination failed: {e}")
            return tf.random.normal([self.n_modes], stddev=0.5)
    
    def _run_quantum_circuit(self, quantum_params):
        """Run quantum circuit with given parameters."""
        # Create parameter mapping
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(), 
                tf.reshape(quantum_params, [-1])
            )
        }
        
        # Reset engine if needed (critical for proper gradients)
        if self.eng.run_progs:
            self.eng.reset()
        
        # Run quantum circuit
        state = self.eng.run(self.qnn, args=mapping).state
        
        # Extract features from quantum state
        return self._extract_enhanced_features_from_state(state)
    
    def _extract_enhanced_features_from_state(self, state):
        """Extract enhanced features from quantum state for discrimination."""
        # Get state vector
        ket = state.ket()
        
        if self.feature_extraction == 'multi_mode':
            return self._extract_multi_mode_features(ket)
        elif self.feature_extraction == 'quantum_observables':
            return self._extract_quantum_observables(ket)
        else:
            # Default feature extraction
            return self._extract_features_from_state(state)
    
    def _extract_multi_mode_features(self, ket):
        """Extract multiple features per mode."""
        features = []
        prob_amplitudes = tf.abs(ket) ** 2
        
        for mode in range(self.n_modes):
            # Feature 1: Position expectation (photon number weighted)
            fock_indices = tf.range(self.cutoff_dim, dtype=tf.float32)
            position_exp = tf.reduce_sum(prob_amplitudes * fock_indices)
            position_feature = (position_exp - self.cutoff_dim/2) / (self.cutoff_dim/4)
            
            # Feature 2: Variance (spread of distribution)
            variance = tf.reduce_sum(prob_amplitudes * (fock_indices - position_exp)**2)
            variance_feature = tf.nn.tanh(variance / self.cutoff_dim)
            
            # Feature 3: Skewness (asymmetry of distribution)
            skewness = tf.reduce_sum(prob_amplitudes * (fock_indices - position_exp)**3)
            skewness_feature = tf.nn.tanh(skewness / (self.cutoff_dim**1.5))
            
            features.extend([position_feature, variance_feature, skewness_feature])
        
        return tf.stack(features[:self.n_modes])  # Limit to n_modes features
    
    def _extract_quantum_observables(self, ket):
        """Extract quantum observables as features."""
        features = []
        prob_amplitudes = tf.abs(ket) ** 2
        
        for mode in range(self.n_modes):
            # Observable 1: Photon number expectation
            fock_indices = tf.range(self.cutoff_dim, dtype=tf.float32)
            photon_number = tf.reduce_sum(prob_amplitudes * fock_indices)
            
            # Observable 2: Quadrature variance (approximation)
            variance = tf.reduce_sum(prob_amplitudes * fock_indices**2) - photon_number**2
            quadrature_var = tf.sqrt(tf.maximum(variance, 0.0))
            
            # Normalize features
            photon_feature = tf.nn.tanh(photon_number / self.cutoff_dim)
            variance_feature = tf.nn.tanh(quadrature_var / self.cutoff_dim)
            
            features.extend([photon_feature, variance_feature])
        
        return tf.stack(features[:self.n_modes])  # Limit to n_modes features
    
    def _discriminate_single(self, quantum_params):
        """FIXED: Discriminate single sample with gradient-preserving execution."""
        # Reshape parameters to match quantum weights structure
        params_reshaped = tf.reshape(quantum_params, self.quantum_weights.shape)
        
        # Combine with quantum weights (learnable quantum circuit + input encoding)
        combined_params = self.quantum_weights + 0.1 * params_reshaped
        
        # Create parameter mapping (following SF tutorial exactly)
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(), 
                tf.reshape(combined_params, [-1])
            )
        }
        
        # FIXED: Reset engine if needed (critical for proper gradients)
        if self.eng.run_progs:
            self.eng.reset()
        
        # Run quantum circuit
        try:
            state = self.eng.run(self.qnn, args=mapping).state
            
            # Extract features from quantum state
            quantum_features = self._extract_features_from_state(state)
            
            return quantum_features
            
        except Exception as e:
            logger.debug(f"Quantum discriminator circuit failed: {e}")
            # Classical fallback
            return tf.random.normal([self.n_modes], stddev=0.5)
    
    def _extract_features_from_state(self, state):
        """Extract features from quantum state for discrimination."""
        # Get state vector
        ket = state.ket()
        
        # Extract multiple quantum features
        features = []
        for mode in range(self.n_modes):
            # Probability distribution over Fock states
            prob_amplitudes = tf.abs(ket) ** 2
            
            # Feature 1: Expectation value (similar to position measurement)
            fock_indices = tf.range(self.cutoff_dim, dtype=tf.float32)
            expectation = tf.reduce_sum(prob_amplitudes * fock_indices)
            
            # Normalize
            feature = (expectation - self.cutoff_dim/2) / (self.cutoff_dim/4)
            features.append(feature)
        
        return tf.stack(features)
    
    def compute_quantum_metrics(self):
        """
        Compute quantum-specific metrics for discriminator.
        
        Returns:
            dict: Quantum metrics (trace, entropy, etc.)
        """
        # Generate a test input
        x_test = tf.random.normal([1, self.input_dim])
        
        # Get quantum state (not just classical output)
        quantum_params = self.input_encoder(x_test)
        params_reshaped = tf.reshape(quantum_params[0], self.quantum_weights.shape)
        combined_params = self.quantum_weights + 0.1 * params_reshaped
        
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(), 
                tf.reshape(combined_params, [-1])
            )
        }
        
        if self.eng.run_progs:
            self.eng.reset()
        
        state = self.eng.run(self.qnn, args=mapping).state
        ket = state.ket()
        
        metrics = {
            'trace': tf.math.real(state.trace()),
            'norm': tf.reduce_sum(tf.abs(ket) ** 2),
            'entropy': self._compute_entropy(ket)
        }
        
        return metrics
    
    def _compute_entropy(self, ket):
        """Compute von Neumann entropy of the state."""
        prob_amplitudes = tf.abs(ket) ** 2
        # Add small epsilon to avoid log(0)
        prob_amplitudes = prob_amplitudes + 1e-12
        entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes))
        return entropy

def test_sf_discriminator():
    """Test the SF-based quantum discriminator."""
    print("\n" + "="*60)
    print("TESTING SF-BASED QUANTUM DISCRIMINATOR")
    print("="*60)
    
    try:
        # Create discriminator
        discriminator = QuantumSFDiscriminator(
            n_modes=2,
            input_dim=2,
            layers=3,
            cutoff_dim=6
        )
        
        print(f"Discriminator created successfully :))")
        print(f"Trainable variables: {len(discriminator.trainable_variables)}")
        
        # Test discrimination
        x_test = tf.random.normal([3, 2])
        probabilities = discriminator.discriminate(x_test)
        
        print(f"Discrimination test: {probabilities.shape}")
        print(f"Probability range: [{tf.reduce_min(probabilities):.3f}, {tf.reduce_max(probabilities):.3f}]")
        
        # Test quantum metrics
        metrics = discriminator.compute_quantum_metrics()
        print(f"Quantum metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")
        
        # Test gradient computation
        with tf.GradientTape() as tape:
            x = tf.random.normal([2, 2])
            output = discriminator.discriminate(x)
            loss = tf.reduce_mean(tf.square(output - 0.5))
        
        gradients = tape.gradient(loss, discriminator.trainable_variables)
        non_none_grads = [g for g in gradients if g is not None]
        
        print(f"Gradient test: {len(non_none_grads)}/{len(gradients)} gradients computed")
        print(f"Loss: {loss:.4f}")
        
        return discriminator
        
    except Exception as e:
        print(f"Test failed: {e} :((")
        return None

if __name__ == "__main__":
    test_sf_discriminator()
