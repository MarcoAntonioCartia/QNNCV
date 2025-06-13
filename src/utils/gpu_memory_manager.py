"""
GPU memory management for hybrid quantum-classical processing.

This module provides GPU memory management for quantum GANs where:
- Quantum operations run on CPU (Strawberry Fields limitation)
- Classical operations run on GPU when available
- Automatic fallback to CPU when GPU is unavailable
- Memory allocation for Colab constraints (must test on Colab)
"""

import tensorflow as tf
import numpy as np
import psutil
import os
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class HybridGPUManager:
    """
    Hybrid GPU/CPU memory manager for quantum-classical processing.
    
    Features:
    - Automatic GPU detection and configuration
    - Smart memory allocation for quantum simulations
    - Colab-optimized resource management
    - Automatic fallback to CPU-only mode
    - Memory usage monitoring and optimization
    """
    
    def __init__(self, enable_memory_growth: bool = True, 
                 gpu_memory_limit: Optional[int] = None):
        """
        Initialize hybrid GPU manager.
        
        Args:
            enable_memory_growth: Enable GPU memory growth (recommended for Colab)
            gpu_memory_limit: GPU memory limit in MB (None for auto-detection)
        """
        self.enable_memory_growth = enable_memory_growth
        self.gpu_memory_limit = gpu_memory_limit
        
        # Hardware detection
        self.hardware_info = self._detect_hardware()
        self.gpu_available = self.hardware_info['gpu_available']
        self.gpu_count = self.hardware_info['gpu_count']
        
        # Device configuration
        self.quantum_device = 'cpu'  # SF limitation
        self.classical_device = 'gpu' if self.gpu_available else 'cpu'
        
        # Configure GPU if available
        if self.gpu_available:
            self._configure_gpu()
        
        # Memory tracking
        self.memory_usage = {
            'quantum_allocated': 0.0,
            'classical_allocated': 0.0,
            'peak_usage': 0.0
        }
        
        logger.info(f"HybridGPUManager initialized:")
        logger.info(f"  GPU Available: {self.gpu_available}")
        logger.info(f"  GPU Count: {self.gpu_count}")
        logger.info(f"  Quantum Device: {self.quantum_device}")
        logger.info(f"  Classical Device: {self.classical_device}")
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware resources."""
        hardware_info = {
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'gpu_count': len(tf.config.list_physical_devices('GPU')),
            'cpu_count': os.cpu_count(),
            'system_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        # Get GPU details if available
        if hardware_info['gpu_available']:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                gpu_details = []
                for gpu in gpus:
                    details = tf.config.experimental.get_device_details(gpu)
                    gpu_details.append({
                        'name': details.get('device_name', 'Unknown GPU'),
                        'compute_capability': details.get('compute_capability', 'Unknown')
                    })
                hardware_info['gpu_details'] = gpu_details
            except Exception as e:
                logger.debug(f"Could not get GPU details: {e}")
                hardware_info['gpu_details'] = []
        
        return hardware_info
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                for gpu in gpus:
                    # Enable memory growth (important for Colab)
                    if self.enable_memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for {gpu}")
                    
                    # Set memory limit if specified
                    if self.gpu_memory_limit is not None:
                        tf.config.experimental.set_memory_limit(
                            gpu, self.gpu_memory_limit
                        )
                        logger.info(f"Set memory limit to {self.gpu_memory_limit}MB for {gpu}")
                
                # Set visible devices
                tf.config.experimental.set_visible_devices(gpus, 'GPU')
                
                # Disable mixed precision by default (can cause issues with quantum ops)
                tf.keras.mixed_precision.set_global_policy('float32')
                
                logger.info("GPU configuration completed successfully")
                
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}")
            self.gpu_available = False
            self.classical_device = 'cpu'
    
    def get_device_context(self, operation_type: str = 'classical'):
        """
        Get appropriate device context for operation.
        
        Args:
            operation_type: 'quantum' or 'classical'
            
        Returns:
            TensorFlow device context
        """
        if operation_type == 'quantum':
            return tf.device('/CPU:0')
        elif operation_type == 'classical' and self.gpu_available:
            return tf.device('/GPU:0')
        else:
            return tf.device('/CPU:0')
    
    def allocate_quantum_memory(self, n_modes: int, cutoff_dim: int, 
                              batch_size: int) -> Dict[str, float]:
        """
        Estimate and allocate memory for quantum operations.
        
        Args:
            n_modes: Number of quantum modes
            cutoff_dim: Cutoff dimension
            batch_size: Batch size
            
        Returns:
            Memory allocation info
        """
        # Estimate quantum state memory requirements
        state_size = cutoff_dim ** n_modes
        complex_state_memory = state_size * 16 * batch_size  # bytes (complex128)
        memory_gb = complex_state_memory / (1024**3)
        
        # Check if allocation is feasible
        available_memory = self.hardware_info['available_memory_gb']
        
        allocation_info = {
            'requested_memory_gb': memory_gb,
            'available_memory_gb': available_memory,
            'allocation_feasible': memory_gb < available_memory * 0.8,  # 80% safety margin
            'recommended_cutoff': cutoff_dim,
            'recommended_batch_size': batch_size
        }
        
        # Suggest optimizations if needed
        if not allocation_info['allocation_feasible']:
            # Reduce cutoff dimension
            max_cutoff = int((available_memory * 0.8 * (1024**3) / (16 * batch_size)) ** (1/n_modes))
            allocation_info['recommended_cutoff'] = max(4, max_cutoff)
            
            # Or reduce batch size
            max_batch = int(available_memory * 0.8 * (1024**3) / (16 * state_size))
            allocation_info['recommended_batch_size'] = max(1, max_batch)
            
            logger.warning(f"Quantum memory allocation may exceed available memory")
            logger.warning(f"Recommended: cutoff_dim={allocation_info['recommended_cutoff']}, "
                         f"batch_size={allocation_info['recommended_batch_size']}")
        
        # Track allocation
        self.memory_usage['quantum_allocated'] = memory_gb
        
        return allocation_info
    
    def allocate_classical_memory(self, model_parameters: int, 
                                batch_size: int) -> Dict[str, float]:
        """
        Estimate and allocate memory for classical operations.
        
        Args:
            model_parameters: Number of model parameters
            batch_size: Batch size
            
        Returns:
            Memory allocation info
        """
        # Estimate classical model memory
        param_memory = model_parameters * 4 * 3  # float32 * (weights + gradients + optimizer state)
        batch_memory = batch_size * 1000 * 4  # Rough estimate for activations
        total_memory = (param_memory + batch_memory) / (1024**3)  # GB
        
        allocation_info = {
            'requested_memory_gb': total_memory,
            'device': self.classical_device,
            'gpu_available': self.gpu_available
        }
        
        # Track allocation
        self.memory_usage['classical_allocated'] = total_memory
        
        return allocation_info
    
    def optimize_batch_size(self, base_batch_size: int, n_modes: int, 
                          cutoff_dim: int) -> int:
        """
        Optimize batch size based on available memory.
        
        Args:
            base_batch_size: Desired batch size
            n_modes: Number of quantum modes
            cutoff_dim: Cutoff dimension
            
        Returns:
            Optimized batch size
        """
        # Check if base batch size is feasible
        allocation_info = self.allocate_quantum_memory(n_modes, cutoff_dim, base_batch_size)
        
        if allocation_info['allocation_feasible']:
            return base_batch_size
        
        # Find maximum feasible batch size
        available_memory = self.hardware_info['available_memory_gb']
        state_size = cutoff_dim ** n_modes
        max_batch = int(available_memory * 0.8 * (1024**3) / (16 * state_size))
        
        optimized_batch_size = max(1, min(base_batch_size, max_batch))
        
        logger.info(f"Optimized batch size: {base_batch_size} → {optimized_batch_size}")
        
        return optimized_batch_size
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        # System memory
        system_memory = psutil.virtual_memory()
        
        memory_stats = {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_percent': system_memory.percent,
            'quantum_allocated_gb': self.memory_usage['quantum_allocated'],
            'classical_allocated_gb': self.memory_usage['classical_allocated']
        }
        
        # GPU memory if available
        if self.gpu_available:
            try:
                # Get GPU memory info (TensorFlow 2.x method)
                gpu_details = tf.config.experimental.get_memory_info('/GPU:0')
                memory_stats.update({
                    'gpu_current_mb': gpu_details['current'] / (1024**2),
                    'gpu_peak_mb': gpu_details['peak'] / (1024**2)
                })
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")
        
        return memory_stats
    
    def clear_memory(self):
        """Clear GPU memory and reset tracking."""
        if self.gpu_available:
            try:
                # Clear TensorFlow GPU memory
                tf.keras.backend.clear_session()
                logger.info("GPU memory cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")
        
        # Reset tracking
        self.memory_usage = {
            'quantum_allocated': 0.0,
            'classical_allocated': 0.0,
            'peak_usage': 0.0
        }
    
    def create_quantum_context(self):
        """Create context manager for quantum operations."""
        return self.get_device_context('quantum')
    
    def create_classical_context(self):
        """Create context manager for classical operations."""
        return self.get_device_context('classical')
    
    def get_optimal_configuration(self, data_size: int, 
                                model_complexity: str = 'medium') -> Dict[str, Any]:
        """
        Get optimal configuration based on available resources.
        
        Args:
            data_size: Size of training dataset
            model_complexity: 'minimal', 'medium', 'high'
            
        Returns:
            Optimal configuration parameters
        """
        available_memory = self.hardware_info['available_memory_gb']
        
        # Base configurations
        configs = {
            'minimal': {
                'n_modes': 1,
                'layers': 1,
                'cutoff_dim': 6,
                'batch_size': 8
            },
            'medium': {
                'n_modes': 2,
                'layers': 2,
                'cutoff_dim': 8,
                'batch_size': 8
            },
            'high': {
                'n_modes': 3,
                'layers': 3,
                'cutoff_dim': 10,
                'batch_size': 4
            }
        }
        
        base_config = configs.get(model_complexity, configs['medium'])
        
        # Optimize based on available memory
        if available_memory < 4:  # Low memory system
            base_config['cutoff_dim'] = min(base_config['cutoff_dim'], 6)
            base_config['batch_size'] = min(base_config['batch_size'], 4)
        elif available_memory > 16:  # High memory system
            base_config['cutoff_dim'] = min(base_config['cutoff_dim'] + 2, 12)
        
        # Optimize batch size
        optimized_batch_size = self.optimize_batch_size(
            base_config['batch_size'],
            base_config['n_modes'],
            base_config['cutoff_dim']
        )
        base_config['batch_size'] = optimized_batch_size
        
        # Add device information
        base_config.update({
            'quantum_device': self.quantum_device,
            'classical_device': self.classical_device,
            'gpu_available': self.gpu_available,
            'memory_optimized': True
        })
        
        return base_config
    
    def monitor_training_memory(self, epoch: int, log_interval: int = 10):
        """Monitor memory usage during training."""
        if epoch % log_interval == 0:
            memory_stats = self.get_memory_usage()
            
            logger.info(f"Epoch {epoch} Memory Usage:")
            logger.info(f"  System: {memory_stats['system_used_percent']:.1f}% "
                       f"({memory_stats['system_available_gb']:.1f}GB available)")
            
            if 'gpu_current_mb' in memory_stats:
                logger.info(f"  GPU: {memory_stats['gpu_current_mb']:.1f}MB current, "
                           f"{memory_stats['gpu_peak_mb']:.1f}MB peak")
            
            # Warning if memory usage is high
            if memory_stats['system_used_percent'] > 90:
                logger.warning("High system memory usage detected!")
            
            # Update peak usage
            total_allocated = (memory_stats['quantum_allocated_gb'] + 
                             memory_stats['classical_allocated_gb'])
            self.memory_usage['peak_usage'] = max(
                self.memory_usage['peak_usage'], total_allocated
            )

def test_gpu_memory_manager():
    """Test GPU memory manager functionality."""
    print("Testing GPU Memory Manager...")
    print("=" * 50)
    
    # Initialize manager
    manager = HybridGPUManager()
    
    print(f"Hardware Detection:")
    print(f"  GPU Available: {manager.gpu_available}")
    print(f"  GPU Count: {manager.gpu_count}")
    print(f"  System Memory: {manager.hardware_info['system_memory_gb']:.1f}GB")
    print(f"  Available Memory: {manager.hardware_info['available_memory_gb']:.1f}GB")
    
    # Test device contexts
    print(f"\nDevice Contexts:")
    with manager.create_quantum_context():
        print(f"  Quantum operations: CPU")
    
    with manager.create_classical_context():
        print(f"  Classical operations: {manager.classical_device.upper()}")
    
    # Test memory allocation
    print(f"\nMemory Allocation Tests:")
    
    # Test quantum memory allocation
    quantum_alloc = manager.allocate_quantum_memory(
        n_modes=2, cutoff_dim=8, batch_size=8
    )
    print(f"  Quantum allocation:")
    print(f"    Requested: {quantum_alloc['requested_memory_gb']:.3f}GB")
    print(f"    Feasible: {quantum_alloc['allocation_feasible']}")
    
    # Test classical memory allocation
    classical_alloc = manager.allocate_classical_memory(
        model_parameters=10000, batch_size=8
    )
    print(f"  Classical allocation:")
    print(f"    Requested: {classical_alloc['requested_memory_gb']:.3f}GB")
    print(f"    Device: {classical_alloc['device'].upper()}")
    
    # Test batch size optimization
    print(f"\nBatch Size Optimization:")
    original_batch = 16
    optimized_batch = manager.optimize_batch_size(
        base_batch_size=original_batch,
        n_modes=2,
        cutoff_dim=8
    )
    print(f"  Original: {original_batch} → Optimized: {optimized_batch}")
    
    # Test optimal configuration
    print(f"\nOptimal Configuration:")
    config = manager.get_optimal_configuration(
        data_size=1000,
        model_complexity='medium'
    )
    print(f"  Configuration: {config}")
    
    # Test memory monitoring
    print(f"\nMemory Usage:")
    memory_stats = manager.get_memory_usage()
    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✓ GPU Memory Manager testing completed!")
    
    return manager

if __name__ == "__main__":
    test_gpu_memory_manager()
