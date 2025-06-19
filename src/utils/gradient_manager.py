"""
C++-Inspired Gradient Manager for Quantum Neural Networks

This module provides RAII-style gradient management with comprehensive
monitoring, safety checks, and debugging capabilities.
"""

import tensorflow as tf
import numpy as np
from contextlib import contextmanager
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class GradientStats:
    """Statistics for gradient computation."""
    step: int
    computation_time: float
    total_norm: float
    max_norm: float
    min_norm: float
    nan_count: int
    inf_count: int
    zero_count: int
    variable_norms: List[float]
    variable_names: List[str]


class QuantumGradientManager:
    """
    C++-Inspired RAII-style gradient manager for quantum neural networks.
    
    Features:
    - Automatic resource management (context manager)
    - NaN/Inf detection and handling
    - Parameter bounds enforcement
    - Gradient norm monitoring
    - Verbose debugging output
    - Performance tracking
    """
    
    def __init__(self, 
                 verbose: bool = False,
                 max_gradient_norm: float = 10.0,
                 parameter_bounds: Optional[Tuple[float, float]] = None,
                 clip_gradients: bool = True):
        """
        Initialize gradient manager.
        
        Args:
            verbose: Enable detailed logging
            max_gradient_norm: Maximum allowed gradient norm (for clipping)
            parameter_bounds: Optional (min, max) bounds for parameters
            clip_gradients: Whether to clip gradients automatically
        """
        self.verbose = verbose
        self.max_gradient_norm = max_gradient_norm
        self.parameter_bounds = parameter_bounds
        self.clip_gradients = clip_gradients
        
        # Statistics tracking
        self.step_counter = 0
        self.gradient_history: List[GradientStats] = []
        self.parameter_history: List[Dict[str, float]] = []
        
        # Safety counters
        self.nan_detections = 0
        self.inf_detections = 0
        self.gradient_clips = 0
        self.parameter_corrections = 0
        
        if self.verbose:
            logger.info("QuantumGradientManager initialized")
            logger.info(f"  Max gradient norm: {max_gradient_norm}")
            logger.info(f"  Parameter bounds: {parameter_bounds}")
            logger.info(f"  Gradient clipping: {clip_gradients}")
    
    @contextmanager
    def managed_computation(self, variables: List[tf.Variable]):
        """
        RAII-style context manager for gradient computation.
        
        Args:
            variables: List of variables to monitor
            
        Yields:
            GradientTape for computation
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"🎬 Step {self.step_counter}: Starting gradient computation")
            logger.info(f"   Monitoring {len(variables)} variables")
        
        # Pre-computation parameter check
        self._log_parameter_state(variables, "PRE")
        
        try:
            with tf.GradientTape() as tape:
                # Ensure all variables are watched
                for var in variables:
                    tape.watch(var)
                
                yield tape
                
                # Post-computation will be handled in safe_gradient
                
        except Exception as e:
            logger.error(f"❌ Exception in gradient computation: {e}")
            raise
        finally:
            computation_time = time.time() - start_time
            if self.verbose:
                logger.info(f"🎬 Step {self.step_counter}: Computation completed in {computation_time:.3f}s")
    
    def safe_gradient(self, 
                     tape: tf.GradientTape, 
                     loss: tf.Tensor, 
                     variables: List[tf.Variable]) -> List[tf.Tensor]:
        """
        Compute gradients with comprehensive safety checks.
        
        Args:
            tape: GradientTape from managed_computation
            loss: Loss tensor
            variables: Variables to compute gradients for
            
        Returns:
            Safe, processed gradients
        """
        start_time = time.time()
        
        # Check loss for numerical issues
        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            logger.error(f"🚨 Invalid loss detected: {loss}")
            return [tf.zeros_like(var) for var in variables]
        
        # Compute raw gradients
        raw_gradients = tape.gradient(loss, variables)
        
        # Process gradients with safety checks
        safe_gradients = []
        gradient_norms = []
        variable_names = []
        
        nan_count = 0
        inf_count = 0
        zero_count = 0
        
        for i, (grad, var) in enumerate(zip(raw_gradients, variables)):
            var_name = var.name.split(':')[0]
            variable_names.append(var_name)
            
            if grad is None:
                if self.verbose:
                    logger.warning(f"⚠️  None gradient for {var_name}")
                safe_grad = tf.zeros_like(var)
                zero_count += 1
            
            elif tf.reduce_any(tf.math.is_nan(grad)):
                if self.verbose:
                    logger.warning(f"🚨 NaN gradient detected for {var_name}")
                safe_grad = tf.zeros_like(var)
                nan_count += 1
                self.nan_detections += 1
            
            elif tf.reduce_any(tf.math.is_inf(grad)):
                if self.verbose:
                    logger.warning(f"🚨 Inf gradient detected for {var_name}")
                safe_grad = tf.zeros_like(var)
                inf_count += 1
                self.inf_detections += 1
            
            else:
                safe_grad = grad
            
            # Gradient clipping
            grad_norm = tf.norm(safe_grad)
            if self.clip_gradients and grad_norm > self.max_gradient_norm:
                if self.verbose:
                    logger.info(f"✂️  Clipping gradient for {var_name}: {grad_norm:.3f} → {self.max_gradient_norm}")
                safe_grad = safe_grad * (self.max_gradient_norm / grad_norm)
                self.gradient_clips += 1
            
            gradient_norms.append(float(grad_norm))
            safe_gradients.append(safe_grad)
        
        # Compute statistics
        total_norm = float(tf.norm([tf.norm(g) for g in safe_gradients]))
        max_norm = max(gradient_norms) if gradient_norms else 0.0
        min_norm = min(gradient_norms) if gradient_norms else 0.0
        
        # Store statistics
        computation_time = time.time() - start_time
        stats = GradientStats(
            step=self.step_counter,
            computation_time=computation_time,
            total_norm=total_norm,
            max_norm=max_norm,
            min_norm=min_norm,
            nan_count=nan_count,
            inf_count=inf_count,
            zero_count=zero_count,
            variable_norms=gradient_norms,
            variable_names=variable_names
        )
        
        self.gradient_history.append(stats)
        
        if self.verbose:
            logger.info(f"📊 Gradient stats: total_norm={total_norm:.3f}, max={max_norm:.3f}, min={min_norm:.3f}")
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"🚨 Issues: {nan_count} NaN, {inf_count} Inf, {zero_count} Zero")
        
        self.step_counter += 1
        return safe_gradients
    
    def apply_gradients_safely(self, 
                              optimizer: tf.optimizers.Optimizer,
                              gradients: List[tf.Tensor],
                              variables: List[tf.Variable]) -> bool:
        """
        Apply gradients with parameter bounds checking.
        
        Args:
            optimizer: TensorFlow optimizer
            gradients: Processed gradients
            variables: Variables to update
            
        Returns:
            True if successful, False if corrections were needed
        """
        # Store pre-update state
        pre_update_params = {var.name: float(tf.reduce_mean(var)) for var in variables}
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, variables))
        
        # Check parameter bounds and correct if needed
        corrections_made = False
        if self.parameter_bounds is not None:
            min_bound, max_bound = self.parameter_bounds
            
            for var in variables:
                # Check for violations
                if tf.reduce_any(var < min_bound) or tf.reduce_any(var > max_bound):
                    if self.verbose:
                        logger.info(f"🔧 Correcting parameter bounds for {var.name}")
                    
                    # Clip to bounds
                    var.assign(tf.clip_by_value(var, min_bound, max_bound))
                    corrections_made = True
                    self.parameter_corrections += 1
        
        # Log parameter state
        if self.verbose:
            self._log_parameter_state(variables, "POST")
        
        # Store post-update state
        post_update_params = {var.name: float(tf.reduce_mean(var)) for var in variables}
        self.parameter_history.append(post_update_params)
        
        return not corrections_made
    
    def _log_parameter_state(self, variables: List[tf.Variable], stage: str):
        """Log current parameter state."""
        if not self.verbose:
            return
        
        logger.info(f"📋 {stage}-UPDATE Parameter State:")
        for var in variables[:3]:  # Show first 3 for brevity
            mean_val = float(tf.reduce_mean(var))
            std_val = float(tf.math.reduce_std(var))
            min_val = float(tf.reduce_min(var))
            max_val = float(tf.reduce_max(var))
            logger.info(f"   {var.name}: μ={mean_val:.3f}, σ={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of gradient manager state."""
        if not self.gradient_history:
            return {"status": "No gradients computed yet"}
        
        recent_stats = self.gradient_history[-1]
        
        return {
            "total_steps": self.step_counter,
            "nan_detections": self.nan_detections,
            "inf_detections": self.inf_detections,
            "gradient_clips": self.gradient_clips,
            "parameter_corrections": self.parameter_corrections,
            "recent_gradient_norm": recent_stats.total_norm,
            "average_computation_time": np.mean([s.computation_time for s in self.gradient_history]),
            "status": "healthy" if recent_stats.nan_count == 0 and recent_stats.inf_count == 0 else "issues_detected"
        }
    
    def reset_stats(self):
        """Reset all statistics and counters."""
        self.step_counter = 0
        self.gradient_history.clear()
        self.parameter_history.clear()
        self.nan_detections = 0
        self.inf_detections = 0
        self.gradient_clips = 0
        self.parameter_corrections = 0
        
        if self.verbose:
            logger.info("🔄 Gradient manager statistics reset")


def create_quantum_gradient_manager(verbose: bool = False) -> QuantumGradientManager:
    """
    Factory function to create a quantum-optimized gradient manager.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        Configured QuantumGradientManager
    """
    return QuantumGradientManager(
        verbose=verbose,
        max_gradient_norm=1.0,  # Conservative for quantum circuits
        parameter_bounds=(-5.0, 5.0),  # Reasonable bounds for quantum parameters
        clip_gradients=True
    )
