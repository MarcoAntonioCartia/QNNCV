"""
X-Quadrature Regularization Losses

Prevents X-quadrature measurements from collapsing to vacuum state (zero).
Maintains diversity while preserving pure X-quadrature decoder architecture.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple


class XQuadratureDiversityLoss:
    """
    Encourages diversity in X-quadrature measurements to prevent mode collapse.
    
    Prevents vacuum state convergence while maintaining pure quantum processing.
    """
    
    def __init__(self, 
                 diversity_weight: float = 0.1,
                 min_variance: float = 0.01,
                 min_mean_magnitude: float = 0.05):
        """
        Initialize X-quadrature diversity loss.
        
        Args:
            diversity_weight: Weight for diversity loss component
            min_variance: Minimum variance required across X-quadrature measurements
            min_mean_magnitude: Minimum mean magnitude to prevent zero convergence
        """
        self.diversity_weight = diversity_weight
        self.min_variance = min_variance
        self.min_mean_magnitude = min_mean_magnitude
        
        print(f"🎯 XQuadratureDiversityLoss initialized:")
        print(f"   Diversity weight: {diversity_weight}")
        print(f"   Min variance: {min_variance}")
        print(f"   Min mean magnitude: {min_mean_magnitude}")
    
    def compute_diversity_loss(self, x_quadrature_batch: tf.Tensor) -> tf.Tensor:
        """
        Compute diversity loss for X-quadrature measurements.
        
        Args:
            x_quadrature_batch: X-quadrature measurements [batch_size, n_modes]
            
        Returns:
            Diversity loss scalar
        """
        # Variance loss: penalize when variance is too low
        batch_variance = tf.math.reduce_variance(x_quadrature_batch)
        variance_loss = tf.maximum(0.0, self.min_variance - batch_variance)
        
        # Mean magnitude loss: penalize when mean magnitude is too small
        mean_magnitude = tf.reduce_mean(tf.abs(x_quadrature_batch))
        magnitude_loss = tf.maximum(0.0, self.min_mean_magnitude - mean_magnitude)
        
        # Mode-wise variance: ensure each mode has some variance
        mode_variances = tf.math.reduce_variance(x_quadrature_batch, axis=0)  # [n_modes]
        mode_variance_loss = tf.reduce_mean(
            tf.maximum(0.0, self.min_variance * 0.5 - mode_variances)
        )
        
        total_diversity_loss = self.diversity_weight * (
            variance_loss + magnitude_loss + mode_variance_loss
        )
        
        return total_diversity_loss
    
    def get_diversity_metrics(self, x_quadrature_batch: tf.Tensor) -> Dict[str, float]:
        """Get diversity metrics for monitoring."""
        return {
            'batch_variance': float(tf.math.reduce_variance(x_quadrature_batch)),
            'mean_magnitude': float(tf.reduce_mean(tf.abs(x_quadrature_batch))),
            'mode_variances': tf.math.reduce_variance(x_quadrature_batch, axis=0).numpy().tolist(),
            'diversity_score': float(tf.math.reduce_variance(x_quadrature_batch) * tf.reduce_mean(tf.abs(x_quadrature_batch)))
        }


class AntiVacuumRegularization:
    """
    Prevents X-quadrature measurements from converging to vacuum state (zero).
    
    Uses exponential penalty to strongly discourage zero measurements.
    """
    
    def __init__(self, strength: float = 0.05, sharpness: float = 2.0):
        """
        Initialize anti-vacuum regularization.
        
        Args:
            strength: Strength of vacuum penalty
            sharpness: Sharpness of exponential penalty (higher = sharper)
        """
        self.strength = strength
        self.sharpness = sharpness
        
        print(f"🚫 AntiVacuumRegularization initialized:")
        print(f"   Strength: {strength}")
        print(f"   Sharpness: {sharpness}")
    
    def compute_vacuum_penalty(self, x_quadrature_batch: tf.Tensor) -> tf.Tensor:
        """
        Compute penalty for measurements close to vacuum (zero).
        
        Args:
            x_quadrature_batch: X-quadrature measurements [batch_size, n_modes]
            
        Returns:
            Vacuum penalty scalar
        """
        # Exponential penalty for measurements close to zero
        # Higher penalty as measurements approach zero
        squared_measurements = tf.square(x_quadrature_batch)
        vacuum_penalties = tf.exp(-self.sharpness * squared_measurements)
        
        # Average penalty across batch and modes
        mean_vacuum_penalty = tf.reduce_mean(vacuum_penalties)
        
        return self.strength * mean_vacuum_penalty
    
    def get_vacuum_metrics(self, x_quadrature_batch: tf.Tensor) -> Dict[str, float]:
        """Get vacuum state metrics for monitoring."""
        abs_measurements = tf.abs(x_quadrature_batch)
        
        return {
            'min_abs_measurement': float(tf.reduce_min(abs_measurements)),
            'max_abs_measurement': float(tf.reduce_max(abs_measurements)),
            'mean_abs_measurement': float(tf.reduce_mean(abs_measurements)),
            'vacuum_proximity': float(tf.reduce_mean(tf.exp(-tf.square(x_quadrature_batch)))),
            'measurements_near_zero': float(tf.reduce_sum(tf.cast(abs_measurements < 0.01, tf.float32)))
        }


class ModeSeparationLoss:
    """
    Encourages different modes to have distinct X-quadrature signatures.
    
    Prevents all modes from converging to the same values.
    """
    
    def __init__(self, 
                 separation_weight: float = 0.02,
                 target_separation: float = 0.1):
        """
        Initialize mode separation loss.
        
        Args:
            separation_weight: Weight for separation loss
            target_separation: Target minimum separation between modes
        """
        self.separation_weight = separation_weight
        self.target_separation = target_separation
        
        print(f"🎭 ModeSeparationLoss initialized:")
        print(f"   Separation weight: {separation_weight}")
        print(f"   Target separation: {target_separation}")
    
    def compute_separation_loss(self, x_quadrature_batch: tf.Tensor) -> tf.Tensor:
        """
        Compute loss to encourage mode separation.
        
        Args:
            x_quadrature_batch: X-quadrature measurements [batch_size, n_modes]
            
        Returns:
            Separation loss scalar
        """
        # Compute pairwise distances between modes
        n_modes = tf.shape(x_quadrature_batch)[1]
        
        # Mean measurement per mode across batch
        mode_means = tf.reduce_mean(x_quadrature_batch, axis=0)  # [n_modes]
        
        # Compute pairwise differences
        mode_differences = tf.abs(mode_means[:, None] - mode_means[None, :])  # [n_modes, n_modes]
        
        # Mask diagonal (mode with itself)
        mask = 1.0 - tf.eye(n_modes)
        masked_differences = mode_differences * mask
        
        # Penalty for modes that are too close
        separation_penalties = tf.maximum(0.0, self.target_separation - masked_differences)
        
        # Average penalty (excluding diagonal)
        mean_separation_penalty = tf.reduce_sum(separation_penalties) / tf.reduce_sum(mask)
        
        return self.separation_weight * mean_separation_penalty
    
    def get_separation_metrics(self, x_quadrature_batch: tf.Tensor) -> Dict[str, any]:
        """Get mode separation metrics for monitoring."""
        mode_means = tf.reduce_mean(x_quadrature_batch, axis=0)
        n_modes = tf.shape(x_quadrature_batch)[1]
        
        # Compute pairwise distances
        mode_differences = tf.abs(mode_means[:, None] - mode_means[None, :])
        mask = 1.0 - tf.eye(n_modes)
        masked_differences = mode_differences * mask
        
        # Get minimum and maximum separations
        non_zero_differences = tf.boolean_mask(masked_differences, mask > 0)
        
        return {
            'mode_means': mode_means.numpy().tolist(),
            'min_separation': float(tf.reduce_min(non_zero_differences)) if tf.size(non_zero_differences) > 0 else 0.0,
            'max_separation': float(tf.reduce_max(non_zero_differences)) if tf.size(non_zero_differences) > 0 else 0.0,
            'mean_separation': float(tf.reduce_mean(non_zero_differences)) if tf.size(non_zero_differences) > 0 else 0.0
        }


class XQuadratureRegularizer:
    """
    Combined X-quadrature regularization system.
    
    Integrates diversity, anti-vacuum, and separation losses for comprehensive regularization.
    """
    
    def __init__(self,
                 diversity_weight: float = 0.1,
                 vacuum_strength: float = 0.05,
                 separation_weight: float = 0.02,
                 enable_diversity: bool = True,
                 enable_vacuum: bool = True,
                 enable_separation: bool = True):
        """
        Initialize combined regularizer.
        
        Args:
            diversity_weight: Weight for diversity loss
            vacuum_strength: Strength for anti-vacuum regularization
            separation_weight: Weight for mode separation loss
            enable_diversity: Enable diversity loss
            enable_vacuum: Enable anti-vacuum regularization
            enable_separation: Enable mode separation loss
        """
        self.enable_diversity = enable_diversity
        self.enable_vacuum = enable_vacuum
        self.enable_separation = enable_separation
        
        # Initialize component losses
        if enable_diversity:
            self.diversity_loss = XQuadratureDiversityLoss(diversity_weight=diversity_weight)
        
        if enable_vacuum:
            self.vacuum_regularization = AntiVacuumRegularization(strength=vacuum_strength)
        
        if enable_separation:
            self.separation_loss = ModeSeparationLoss(separation_weight=separation_weight)
        
        print(f"🎯 XQuadratureRegularizer initialized:")
        print(f"   Diversity: {'✅' if enable_diversity else '❌'}")
        print(f"   Anti-vacuum: {'✅' if enable_vacuum else '❌'}")
        print(f"   Mode separation: {'✅' if enable_separation else '❌'}")
    
    def compute_regularization_loss(self, x_quadrature_batch: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute combined regularization loss.
        
        Args:
            x_quadrature_batch: X-quadrature measurements [batch_size, n_modes]
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        total_loss = tf.constant(0.0)
        loss_components = {}
        
        # Diversity loss
        if self.enable_diversity:
            diversity_loss = self.diversity_loss.compute_diversity_loss(x_quadrature_batch)
            total_loss += diversity_loss
            loss_components['diversity'] = diversity_loss
        
        # Anti-vacuum regularization
        if self.enable_vacuum:
            vacuum_penalty = self.vacuum_regularization.compute_vacuum_penalty(x_quadrature_batch)
            total_loss += vacuum_penalty
            loss_components['vacuum'] = vacuum_penalty
        
        # Mode separation loss
        if self.enable_separation:
            separation_loss = self.separation_loss.compute_separation_loss(x_quadrature_batch)
            total_loss += separation_loss
            loss_components['separation'] = separation_loss
        
        return total_loss, loss_components
    
    def get_comprehensive_metrics(self, x_quadrature_batch: tf.Tensor) -> Dict[str, any]:
        """Get comprehensive metrics for all regularization components."""
        metrics = {}
        
        if self.enable_diversity:
            metrics.update({f"diversity_{k}": v for k, v in 
                          self.diversity_loss.get_diversity_metrics(x_quadrature_batch).items()})
        
        if self.enable_vacuum:
            metrics.update({f"vacuum_{k}": v for k, v in 
                          self.vacuum_regularization.get_vacuum_metrics(x_quadrature_batch).items()})
        
        if self.enable_separation:
            metrics.update({f"separation_{k}": v for k, v in 
                          self.separation_loss.get_separation_metrics(x_quadrature_batch).items()})
        
        return metrics


def create_x_quadrature_regularizer(config: Dict[str, any] = None) -> XQuadratureRegularizer:
    """
    Factory function to create X-quadrature regularizer with sensible defaults.
    
    Args:
        config: Configuration dictionary for regularization parameters
        
    Returns:
        Configured XQuadratureRegularizer
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        'diversity_weight': 0.1,
        'vacuum_strength': 0.05,
        'separation_weight': 0.02,
        'enable_diversity': True,
        'enable_vacuum': True,
        'enable_separation': True
    }
    
    # Update with user config
    final_config = {**default_config, **config}
    
    return XQuadratureRegularizer(**final_config)
