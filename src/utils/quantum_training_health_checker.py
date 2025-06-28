"""
Quantum Training Health Checker

This module provides comprehensive health checking for quantum GAN training to prevent
system crashes and resource exhaustion. It includes:
- Memory safety analysis for quantum states and classical models
- Processing load assessment and optimization
- Training time estimation with hardware benchmarking
- Real-time resource monitoring and intervention
- Parameter optimization for safe training
"""

import psutil
import time
import numpy as np
import tensorflow as tf
import logging
import os
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

# Import existing utilities
try:
    from .gpu_memory_manager import HybridGPUManager
    from .quantum_metrics import QuantumMetrics
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils.gpu_memory_manager import HybridGPUManager
    from src.utils.quantum_metrics import QuantumMetrics

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of pre-training health check."""
    safe_to_proceed: bool
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    warnings: List[str]
    recommendations: List[str]
    optimized_config: Dict[str, Any]
    estimated_memory_gb: float
    estimated_time_hours: float
    confidence_score: float


@dataclass
class TimeEstimate:
    """Training time estimation result."""
    total_hours: float
    per_epoch_minutes: float
    confidence_interval: Tuple[float, float]  # (min_hours, max_hours)
    bottleneck_analysis: Dict[str, float]
    hardware_score: float


@dataclass
class HealthStatus:
    """Runtime health monitoring status."""
    status: str  # 'healthy', 'warning', 'critical', 'emergency'
    memory_usage_percent: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    intervention_needed: bool
    recommended_actions: List[str]
    time_remaining_estimate: Optional[float]


class QuantumTrainingHealthChecker:
    """
    Comprehensive health checker for quantum GAN training.
    
    Features:
    - Pre-training safety analysis
    - Memory and processing load estimation
    - Training time prediction with benchmarking
    - Real-time monitoring and intervention
    - Parameter optimization for safe training
    """
    
    def __init__(self, hardware_manager: Optional[HybridGPUManager] = None):
        """
        Initialize health checker.
        
        Args:
            hardware_manager: Optional hardware manager (will create if None)
        """
        self.hardware_manager = hardware_manager or HybridGPUManager()
        self.quantum_metrics = QuantumMetrics()
        
        # System information
        self.system_info = self._gather_system_info()
        
        # Benchmarking data
        self.benchmark_data = {}
        self.training_history = []
        
        # Safety thresholds
        self.safety_thresholds = {
            'memory_usage_percent': 85.0,  # Max memory usage
            'cpu_usage_percent': 90.0,     # Max CPU usage
            'gpu_memory_percent': 90.0,    # Max GPU memory
            'quantum_state_memory_gb': None,  # Will be calculated
            'min_available_memory_gb': 2.0,   # Minimum free memory
        }
        
        # Performance benchmarks (will be populated)
        self.performance_benchmarks: Dict[str, Optional[float]] = {
            'quantum_circuit_eval_ms': None,
            'classical_forward_pass_ms': None,
            'gradient_computation_ms': None,
            'memory_allocation_speed': None
        }
        
        logger.info("QuantumTrainingHealthChecker initialized")
        logger.info(f"System: {self.system_info['cpu_count']} CPUs, "
                   f"{self.system_info['memory_gb']:.1f}GB RAM, "
                   f"GPU: {self.system_info['gpu_available']}")
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information."""
        memory = psutil.virtual_memory()
        
        system_info = {
            'cpu_count': os.cpu_count(),
            'memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_usage_percent': memory.percent,
            'gpu_available': self.hardware_manager.gpu_available,
            'gpu_count': self.hardware_manager.gpu_count,
            'platform': os.name,
            'python_version': f"{tf.__version__}",
            'tensorflow_version': tf.__version__
        }
        
        # Add CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                system_info['cpu_freq_mhz'] = cpu_freq.current
        except:
            pass
        
        # Add GPU details if available
        if system_info['gpu_available']:
            gpu_info = self.hardware_manager.hardware_info
            if isinstance(gpu_info, dict):
                system_info.update(gpu_info)
        
        return system_info
    
    def estimate_quantum_memory_requirements(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate memory requirements for quantum operations.
        
        Args:
            config: Training configuration
            
        Returns:
            Memory requirement analysis
        """
        n_modes = config.get('n_modes', 2)
        cutoff_dim = config.get('cutoff_dim', 8)
        batch_size = config.get('batch_size', 16)
        layers = config.get('layers', 1)
        
        # Quantum state memory calculation
        # Each quantum state: cutoff_dim^n_modes complex numbers
        state_size = cutoff_dim ** n_modes
        complex_state_memory = state_size * 16  # bytes (complex128)
        
        # Memory per batch
        batch_memory = complex_state_memory * batch_size
        
        # Additional memory for intermediate calculations
        # - Gradient computation: ~2x state memory
        # - Circuit evaluation: ~1.5x state memory
        # - Optimization buffers: ~1x state memory
        overhead_factor = 4.5
        
        total_quantum_memory = batch_memory * overhead_factor * layers
        memory_gb = total_quantum_memory / (1024**3)
        
        # Classical model memory estimation
        estimated_params = self._estimate_model_parameters(config)
        classical_memory_gb = (estimated_params * 4 * 3) / (1024**3)  # weights + grads + optimizer
        
        return {
            'quantum_state_memory_gb': memory_gb,
            'classical_model_memory_gb': classical_memory_gb,
            'total_estimated_memory_gb': memory_gb + classical_memory_gb,
            'state_size': state_size,
            'batch_memory_mb': batch_memory / (1024**2),
            'overhead_factor': overhead_factor
        }
    
    def _estimate_model_parameters(self, config: Dict[str, Any]) -> int:
        """Estimate total number of model parameters."""
        n_modes = config.get('n_modes', 2)
        cutoff_dim = config.get('cutoff_dim', 8)
        layers = config.get('layers', 1)
        latent_dim = config.get('latent_dim', 2)
        output_dim = config.get('output_dim', 2)
        
        # Rough estimation based on typical quantum GAN architectures
        # Generator parameters
        gen_params = n_modes * layers * 10  # Gate parameters
        gen_params += latent_dim * output_dim * 50  # Classical layers
        
        # Discriminator parameters
        disc_params = n_modes * layers * 10  # Gate parameters
        disc_params += output_dim * 50  # Classical layers
        
        return gen_params + disc_params
    
    def benchmark_hardware_performance(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Benchmark hardware performance for quantum operations.
        
        Args:
            config: Training configuration
            
        Returns:
            Performance benchmarks
        """
        logger.info("Running hardware performance benchmarks...")
        
        benchmarks = {}
        
        try:
            # Benchmark quantum circuit evaluation
            benchmarks['quantum_circuit_eval_ms'] = self._benchmark_quantum_circuit(config)
            
            # Benchmark classical operations
            benchmarks['classical_forward_pass_ms'] = self._benchmark_classical_ops(config)
            
            # Benchmark memory allocation
            benchmarks['memory_allocation_speed'] = self._benchmark_memory_allocation(config)
            
            # Overall hardware score (higher is better)
            benchmarks['hardware_score'] = self._calculate_hardware_score(benchmarks)
            
        except Exception as e:
            logger.warning(f"Benchmarking failed: {e}")
            # Use conservative estimates
            benchmarks = {
                'quantum_circuit_eval_ms': 1000.0,  # 1 second per evaluation
                'classical_forward_pass_ms': 10.0,
                'memory_allocation_speed': 1.0,
                'hardware_score': 0.5
            }
        
        for key, value in benchmarks.items():
            self.performance_benchmarks[key] = value
        logger.info(f"Hardware benchmarks completed: {benchmarks}")
        
        return benchmarks
    
    def _benchmark_quantum_circuit(self, config: Dict[str, Any]) -> float:
        """Benchmark quantum circuit evaluation time."""
        try:
            # Create a minimal quantum circuit for benchmarking
            import strawberryfields as sf
            from strawberryfields.ops import Sgate, BSgate, MZgate
            
            n_modes = min(config.get('n_modes', 2), 2)  # Limit for benchmarking
            cutoff_dim = min(config.get('cutoff_dim', 8), 6)  # Limit for benchmarking
            
            # Create circuit
            prog = sf.Program(n_modes)
            with prog.context as q:
                for i in range(n_modes):
                    Sgate(0.1) | q[i]
                if n_modes > 1:
                    BSgate(0.1, 0.1) | (q[0], q[1])
            
            # Benchmark execution
            engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
            
            start_time = time.time()
            n_runs = 5
            
            for _ in range(n_runs):
                result = engine.run(prog)
            
            end_time = time.time()
            avg_time_ms = ((end_time - start_time) / n_runs) * 1000
            
            # Scale estimate based on actual configuration
            scaling_factor = (config.get('cutoff_dim', 8) / cutoff_dim) ** config.get('n_modes', 2)
            estimated_time_ms = avg_time_ms * scaling_factor
            
            return estimated_time_ms
            
        except Exception as e:
            logger.warning(f"Quantum circuit benchmarking failed: {e}")
            return 1000.0  # Conservative estimate
    
    def _benchmark_classical_ops(self, config: Dict[str, Any]) -> float:
        """Benchmark classical operations."""
        try:
            batch_size = config.get('batch_size', 16)
            output_dim = config.get('output_dim', 2)
            
            # Create simple neural network operations
            device_context = self.hardware_manager.create_classical_context()
            with device_context:
                x = tf.random.normal([batch_size, output_dim])
                w = tf.random.normal([output_dim, 64])
                
                start_time = time.time()
                n_runs = 10
                
                for _ in range(n_runs):
                    y = tf.matmul(x, w)
                    y = tf.nn.relu(y)
                    loss = tf.reduce_mean(tf.square(y))
                
                end_time = time.time()
                avg_time_ms = ((end_time - start_time) / n_runs) * 1000
                
                return avg_time_ms
                
        except Exception as e:
            logger.warning(f"Classical operations benchmarking failed: {e}")
            return 10.0  # Conservative estimate
    
    def _benchmark_memory_allocation(self, config: Dict[str, Any]) -> float:
        """Benchmark memory allocation speed."""
        try:
            memory_requirements = self.estimate_quantum_memory_requirements(config)
            test_size_mb = min(100, memory_requirements['batch_memory_mb'])  # Test with smaller size
            
            start_time = time.time()
            
            # Allocate and deallocate memory
            test_array = np.random.random((int(test_size_mb * 1024 * 1024 / 8),))  # 8 bytes per float64
            del test_array
            
            end_time = time.time()
            allocation_time_ms = (end_time - start_time) * 1000
            
            # Speed score (MB/ms)
            speed_score = test_size_mb / max(allocation_time_ms, 0.1)
            
            return speed_score
            
        except Exception as e:
            logger.warning(f"Memory allocation benchmarking failed: {e}")
            return 1.0  # Conservative estimate
    
    def _calculate_hardware_score(self, benchmarks: Dict[str, float]) -> float:
        """Calculate overall hardware performance score."""
        try:
            # Normalize benchmarks (lower times are better, higher speeds are better)
            quantum_score = max(0.1, 1000.0 / benchmarks.get('quantum_circuit_eval_ms', 1000.0))
            classical_score = max(0.1, 100.0 / benchmarks.get('classical_forward_pass_ms', 100.0))
            memory_score = benchmarks.get('memory_allocation_speed', 1.0)
            
            # Weighted average
            overall_score = (quantum_score * 0.5 + classical_score * 0.3 + memory_score * 0.2)
            
            # Normalize to 0-1 range
            return min(1.0, overall_score / 10.0)
            
        except Exception as e:
            logger.warning(f"Hardware score calculation failed: {e}")
            return 0.5
    
    def estimate_training_time(self, config: Dict[str, Any]) -> TimeEstimate:
        """
        Estimate total training time based on configuration and hardware.
        
        Args:
            config: Training configuration
            
        Returns:
            Time estimation with confidence intervals
        """
        epochs = config.get('epochs', 10)
        steps_per_epoch = config.get('steps_per_epoch', 5)
        
        # Get or run benchmarks
        if not self.performance_benchmarks.get('quantum_circuit_eval_ms'):
            self.benchmark_hardware_performance(config)
        
        # Time per training step estimation
        quantum_time_ms = self.performance_benchmarks['quantum_circuit_eval_ms'] or 1000.0
        classical_time_ms = self.performance_benchmarks['classical_forward_pass_ms'] or 10.0
        
        # Account for discriminator training (3 steps) + generator training (1 step)
        time_per_step_ms = quantum_time_ms * 4 + classical_time_ms * 8  # Conservative estimate
        
        # Add overhead for monitoring, visualization, etc.
        overhead_factor = 1.5
        time_per_step_ms *= overhead_factor
        
        # Calculate total time
        total_steps = epochs * steps_per_epoch
        total_time_ms = total_steps * time_per_step_ms
        total_hours = total_time_ms / (1000 * 60 * 60)
        
        per_epoch_minutes = (steps_per_epoch * time_per_step_ms) / (1000 * 60)
        
        # Confidence intervals based on hardware score
        hardware_score = self.performance_benchmarks.get('hardware_score', 0.5) or 0.5
        confidence_factor = 0.5 + hardware_score * 0.5  # 0.5 to 1.0
        
        min_hours = total_hours * 0.7 * confidence_factor
        max_hours = total_hours * 1.8 / confidence_factor
        
        # Bottleneck analysis
        bottleneck_analysis = {
            'quantum_operations_percent': (quantum_time_ms * 4) / time_per_step_ms * 100,
            'classical_operations_percent': (classical_time_ms * 8) / time_per_step_ms * 100,
            'overhead_percent': ((overhead_factor - 1) / overhead_factor) * 100
        }
        
        return TimeEstimate(
            total_hours=total_hours,
            per_epoch_minutes=per_epoch_minutes,
            confidence_interval=(min_hours, max_hours),
            bottleneck_analysis=bottleneck_analysis,
            hardware_score=hardware_score
        )
    
    def pre_training_health_check(self, config: Dict[str, Any]) -> HealthCheckResult:
        """
        Comprehensive pre-training health check.
        
        Args:
            config: Training configuration
            
        Returns:
            Health check result with safety assessment
        """
        logger.info("Running pre-training health check...")
        
        warnings = []
        recommendations = []
        optimized_config = config.copy()
        risk_level = 'low'
        
        # 1. Memory Analysis
        memory_analysis = self.estimate_quantum_memory_requirements(config)
        estimated_memory_gb = memory_analysis['total_estimated_memory_gb']
        available_memory_gb = self.system_info['available_memory_gb']
        
        logger.info(f"Memory analysis: {estimated_memory_gb:.2f}GB required, "
                   f"{available_memory_gb:.2f}GB available")
        
        if estimated_memory_gb > available_memory_gb * 0.8:
            risk_level = 'high'
            warnings.append(f"High memory usage: {estimated_memory_gb:.1f}GB required, "
                          f"only {available_memory_gb:.1f}GB available")
            
            # Optimize configuration
            optimized_config = self._optimize_memory_configuration(config, available_memory_gb)
            recommendations.append("Reduced batch size and/or cutoff dimension for memory safety")
        
        elif estimated_memory_gb > available_memory_gb * 0.6:
            risk_level = max(risk_level, 'medium')
            warnings.append("Moderate memory usage detected")
            recommendations.append("Consider reducing batch size if system becomes unresponsive")
        
        # 2. Processing Load Analysis
        time_estimate = self.estimate_training_time(optimized_config)
        
        if time_estimate.total_hours > 24:
            risk_level = max(risk_level, 'medium')
            warnings.append(f"Long training time estimated: {time_estimate.total_hours:.1f} hours")
            recommendations.append("Consider reducing epochs or using faster hardware")
        
        # 3. Hardware Compatibility Check
        hardware_score = time_estimate.hardware_score
        if hardware_score < 0.3:
            risk_level = max(risk_level, 'medium')
            warnings.append("Low hardware performance detected")
            recommendations.append("Training may be very slow on this hardware")
        
        # 4. Configuration Validation
        if config.get('cutoff_dim', 8) > 10:
            warnings.append("High cutoff dimension may cause exponential memory growth")
            recommendations.append("Consider cutoff_dim <= 10 for stability")
        
        if config.get('n_modes', 2) > 3:
            warnings.append("High number of modes increases computational complexity")
            recommendations.append("Consider n_modes <= 3 for reasonable performance")
        
        # 5. System Resource Check
        current_memory_usage = psutil.virtual_memory().percent
        if current_memory_usage > 70:
            risk_level = max(risk_level, 'medium')
            warnings.append(f"High current memory usage: {current_memory_usage:.1f}%")
            recommendations.append("Close other applications before training")
        
        # Determine if safe to proceed
        safe_to_proceed = risk_level != 'critical' and estimated_memory_gb < available_memory_gb * 0.9
        
        # Calculate confidence score
        confidence_factors = [
            1.0 - min(1.0, estimated_memory_gb / available_memory_gb),
            hardware_score,
            1.0 - min(1.0, time_estimate.total_hours / 48.0),  # Penalize very long training
            1.0 - min(1.0, current_memory_usage / 100.0)
        ]
        confidence_score = float(np.mean(confidence_factors))
        
        result = HealthCheckResult(
            safe_to_proceed=safe_to_proceed,
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations,
            optimized_config=optimized_config,
            estimated_memory_gb=estimated_memory_gb,
            estimated_time_hours=time_estimate.total_hours,
            confidence_score=confidence_score
        )
        
        logger.info(f"Health check complete: Risk={risk_level}, Safe={safe_to_proceed}, "
                   f"Confidence={confidence_score:.2f}")
        
        return result
    
    def _optimize_memory_configuration(self, config: Dict[str, Any], 
                                     available_memory_gb: float) -> Dict[str, Any]:
        """Optimize configuration for available memory."""
        optimized = config.copy()
        target_memory_gb = available_memory_gb * 0.7  # 70% of available memory
        
        # Try reducing batch size first
        for batch_size in [16, 8, 4, 2, 1]:
            test_config = optimized.copy()
            test_config['batch_size'] = batch_size
            
            memory_req = self.estimate_quantum_memory_requirements(test_config)
            if memory_req['total_estimated_memory_gb'] <= target_memory_gb:
                optimized['batch_size'] = batch_size
                break
        
        # If still too much memory, reduce cutoff dimension
        memory_req = self.estimate_quantum_memory_requirements(optimized)
        if memory_req['total_estimated_memory_gb'] > target_memory_gb:
            for cutoff_dim in [8, 6, 4]:
                test_config = optimized.copy()
                test_config['cutoff_dim'] = cutoff_dim
                
                memory_req = self.estimate_quantum_memory_requirements(test_config)
                if memory_req['total_estimated_memory_gb'] <= target_memory_gb:
                    optimized['cutoff_dim'] = cutoff_dim
                    break
        
        return optimized
    
    def monitor_training_health(self, epoch: int, step: int, 
                              metrics: Optional[Dict[str, Any]] = None) -> HealthStatus:
        """
        Monitor training health in real-time.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Optional training metrics
            
        Returns:
            Current health status
        """
        # Get current system status
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # GPU usage if available
        gpu_usage_percent = None
        if self.hardware_manager.gpu_available:
            try:
                gpu_stats = self.hardware_manager.get_memory_usage()
                if 'gpu_current_mb' in gpu_stats:
                    # Rough GPU usage estimation
                    gpu_usage_percent = min(100.0, gpu_stats['gpu_current_mb'] / 1000.0)
            except:
                pass
        
        # Determine status
        status = 'healthy'
        intervention_needed = False
        recommended_actions = []
        
        # Memory checks
        if memory.percent > self.safety_thresholds['memory_usage_percent']:
            status = 'critical'
            intervention_needed = True
            recommended_actions.append("Reduce batch size immediately")
            recommended_actions.append("Clear unnecessary variables")
        elif memory.percent > 75:
            status = 'warning'
            recommended_actions.append("Monitor memory usage closely")
        
        # CPU checks
        if cpu_percent > self.safety_thresholds['cpu_usage_percent']:
            status = max(status, 'warning', key=['healthy', 'warning', 'critical'].index)
            recommended_actions.append("High CPU usage detected")
        
        # GPU checks
        if gpu_usage_percent and gpu_usage_percent > self.safety_thresholds['gpu_memory_percent']:
            status = max(status, 'warning', key=['healthy', 'warning', 'critical'].index)
            recommended_actions.append("High GPU memory usage")
        
        # Time remaining estimate
        time_remaining_estimate = None
        if hasattr(self, '_training_start_time') and epoch > 0:
            elapsed_time = time.time() - self._training_start_time
            time_per_epoch = elapsed_time / epoch
            remaining_epochs = getattr(self, '_total_epochs', 10) - epoch
            time_remaining_estimate = time_per_epoch * remaining_epochs / 3600  # hours
        
        return HealthStatus(
            status=status,
            memory_usage_percent=memory.percent,
            cpu_usage_percent=cpu_percent,
            gpu_usage_percent=gpu_usage_percent,
            intervention_needed=intervention_needed,
            recommended_actions=recommended_actions,
            time_remaining_estimate=time_remaining_estimate
        )
    
    def start_training_monitoring(self, total_epochs: int):
        """Start training monitoring session."""
        self._training_start_time = time.time()
        self._total_epochs = total_epochs
        logger.info(f"Started training monitoring for {total_epochs} epochs")
    
    def generate_health_report(self, save_path: str):
        """Generate comprehensive health report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'performance_benchmarks': self.performance_benchmarks,
            'safety_thresholds': self.safety_thresholds,
            'training_history': self.training_history
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Health report saved to {save_path}")


def test_health_checker():
    """Test the health checker functionality."""
    print("Testing Quantum Training Health Checker...")
    print("=" * 60)
    
    # Initialize health checker
    health_checker = QuantumTrainingHealthChecker()
    
    # Test configuration
    test_config = {
        'n_modes': 2,
        'cutoff_dim': 8,
        'batch_size': 16,
        'layers': 1,
        'epochs': 5,
        'steps_per_epoch': 5,
        'latent_dim': 2,
        'output_dim': 2
    }
    
    print("Test Configuration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    
    # Run health check
    print("\nRunning pre-training health check...")
    health_result = health_checker.pre_training_health_check(test_config)
    
    print(f"\nHealth Check Results:")
    print(f"  Safe to proceed: {health_result.safe_to_proceed}")
    print(f"  Risk level: {health_result.risk_level}")
    print(f"  Estimated memory: {health_result.estimated_memory_gb:.2f}GB")
    print(f"  Estimated time: {health_result.estimated_time_hours:.2f} hours")
    print(f"  Confidence score: {health_result.confidence_score:.2f}")
    
    if health_result.warnings:
        print(f"\nWarnings:")
        for warning in health_result.warnings:
            print(f"  ⚠️ {warning}")
    
    if health_result.recommendations:
        print(f"\nRecommendations:")
        for rec in health_result.recommendations:
            print(f"  💡 {rec}")
    
    # Test runtime monitoring
    print(f"\nTesting runtime monitoring...")
    health_checker.start_training_monitoring(test_config['epochs'])
    
    for epoch in range(1, 3):
        health_status = health_checker.monitor_training_health(epoch, 0)
        print(f"  Epoch {epoch}: {health_status.status} "
              f"(Memory: {health_status.memory_usage_percent:.1f}%, "
              f"CPU: {health_status.cpu_usage_percent:.1f}%)")
    
    print(f"\n✅ Health checker testing completed!")
    
    return health_checker


if __name__ == "__main__":
    test_health_checker()
