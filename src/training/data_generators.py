"""
Data generators for Quantum GAN training.

This module provides synthetic data generators for testing and training
quantum GANs with various distributions.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BimodalDataGenerator:
    """
    Generator for bimodal synthetic data.
    
    Creates data with two distinct modes for testing quantum GAN
    mode coverage and diversity preservation.
    """
    
    def __init__(self, 
                 batch_size: int = 16,
                 n_features: int = 2,
                 mode1_center: Tuple[float, float] = (-2.0, -2.0),
                 mode2_center: Tuple[float, float] = (2.0, 2.0),
                 mode_std: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize bimodal data generator.
        
        Args:
            batch_size: Number of samples per batch
            n_features: Number of features (dimensions)
            mode1_center: Center of first mode
            mode2_center: Center of second mode  
            mode_std: Standard deviation of each mode
            seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.n_features = n_features
        self.mode1_center = np.array(mode1_center)
        self.mode2_center = np.array(mode2_center)
        self.mode_std = mode_std
        
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        logger.info(f"Bimodal data generator initialized:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Mode 1 center: {mode1_center}")
        logger.info(f"  Mode 2 center: {mode2_center}")
        logger.info(f"  Mode std: {mode_std}")
    
    def __call__(self) -> tf.Tensor:
        """Generate a batch of bimodal data."""
        return self.generate_batch()
    
    def generate_batch(self) -> tf.Tensor:
        """Generate a single batch of bimodal data."""
        # Split batch between two modes
        n_mode1 = self.batch_size // 2
        n_mode2 = self.batch_size - n_mode1
        
        # Generate mode 1 samples
        mode1_samples = np.random.normal(
            self.mode1_center, 
            self.mode_std, 
            (n_mode1, self.n_features)
        )
        
        # Generate mode 2 samples
        mode2_samples = np.random.normal(
            self.mode2_center,
            self.mode_std,
            (n_mode2, self.n_features)
        )
        
        # Combine and shuffle
        batch = np.vstack([mode1_samples, mode2_samples])
        np.random.shuffle(batch)
        
        return tf.constant(batch, dtype=tf.float32)
    
    def generate_dataset(self, n_batches: int) -> tf.Tensor:
        """Generate multiple batches as a single tensor."""
        batches = []
        for _ in range(n_batches):
            batches.append(self.generate_batch())
        
        return tf.concat(batches, axis=0)
    
    def visualize_distribution(self, n_samples: int = 1000, save_path: Optional[str] = None):
        """Visualize the bimodal distribution."""
        if self.n_features != 2:
            logger.warning("Visualization only supported for 2D data")
            return
        
        # Generate samples
        n_batches = n_samples // self.batch_size
        data = self.generate_dataset(n_batches)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=30)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Bimodal Distribution')
        plt.grid(True, alpha=0.3)
        
        # Mark mode centers
        plt.scatter(*self.mode1_center, color='red', s=100, marker='x', label='Mode 1 Center')
        plt.scatter(*self.mode2_center, color='red', s=100, marker='x', label='Mode 2 Center')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Distribution visualization saved to {save_path}")
        
        plt.show()


class SyntheticDataGenerator:
    """
    Generator for various synthetic distributions.
    
    Supports multiple distribution types for comprehensive
    quantum GAN testing.
    """
    
    def __init__(self,
                 distribution_type: str = 'bimodal',
                 batch_size: int = 16,
                 n_features: int = 2,
                 **kwargs):
        """
        Initialize synthetic data generator.
        
        Args:
            distribution_type: Type of distribution ('bimodal', 'circular', 'swiss_roll')
            batch_size: Number of samples per batch
            n_features: Number of features
            **kwargs: Additional parameters for specific distributions
        """
        self.distribution_type = distribution_type
        self.batch_size = batch_size
        self.n_features = n_features
        self.kwargs = kwargs
        
        # Initialize specific generator
        if distribution_type == 'bimodal':
            self._generator = self._create_bimodal_generator()
        elif distribution_type == 'circular':
            self._generator = self._create_circular_generator()
        elif distribution_type == 'swiss_roll':
            self._generator = self._create_swiss_roll_generator()
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        logger.info(f"Synthetic data generator initialized: {distribution_type}")
    
    def __call__(self) -> tf.Tensor:
        """Generate a batch of synthetic data."""
        return self._generator()
    
    def _create_bimodal_generator(self) -> Callable[[], tf.Tensor]:
        """Create bimodal distribution generator."""
        bimodal_gen = BimodalDataGenerator(
            batch_size=self.batch_size,
            n_features=self.n_features,
            **self.kwargs
        )
        return bimodal_gen.generate_batch
    
    def _create_circular_generator(self) -> Callable[[], tf.Tensor]:
        """Create circular distribution generator."""
        radius = self.kwargs.get('radius', 2.0)
        noise_std = self.kwargs.get('noise_std', 0.1)
        
        def generate_circular_batch():
            # Generate angles
            angles = np.random.uniform(0, 2*np.pi, self.batch_size)
            
            # Add noise to radius
            radii = radius + np.random.normal(0, noise_std, self.batch_size)
            
            # Convert to cartesian
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            
            if self.n_features == 2:
                batch = np.column_stack([x, y])
            else:
                # Extend to higher dimensions with noise
                batch = np.column_stack([x, y])
                for i in range(2, self.n_features):
                    extra_dim = np.random.normal(0, noise_std, self.batch_size)
                    batch = np.column_stack([batch, extra_dim])
            
            return tf.constant(batch, dtype=tf.float32)
        
        return generate_circular_batch
    
    def _create_swiss_roll_generator(self) -> Callable[[], tf.Tensor]:
        """Create Swiss roll distribution generator."""
        noise_std = self.kwargs.get('noise_std', 0.1)
        
        def generate_swiss_roll_batch():
            # Generate Swiss roll parameters
            t = np.random.uniform(1.5*np.pi, 4.5*np.pi, self.batch_size)
            height = np.random.uniform(0, 10, self.batch_size)
            
            # Swiss roll coordinates
            x = t * np.cos(t)
            y = height
            z = t * np.sin(t)
            
            # Add noise
            x += np.random.normal(0, noise_std, self.batch_size)
            y += np.random.normal(0, noise_std, self.batch_size)
            z += np.random.normal(0, noise_std, self.batch_size)
            
            if self.n_features == 2:
                batch = np.column_stack([x, z])  # Project to 2D
            elif self.n_features == 3:
                batch = np.column_stack([x, y, z])
            else:
                # Extend or reduce dimensions
                base_batch = np.column_stack([x, y, z])
                if self.n_features > 3:
                    for i in range(3, self.n_features):
                        extra_dim = np.random.normal(0, noise_std, self.batch_size)
                        base_batch = np.column_stack([base_batch, extra_dim])
                batch = base_batch[:, :self.n_features]
            
            return tf.constant(batch, dtype=tf.float32)
        
        return generate_swiss_roll_batch


def create_data_generator(generator_type: str = 'bimodal', **kwargs) -> Callable[[], tf.Tensor]:
    """
    Factory function to create data generators.
    
    Args:
        generator_type: Type of generator ('bimodal', 'synthetic')
        **kwargs: Additional arguments for generator
        
    Returns:
        Data generator function
    """
    if generator_type == 'bimodal':
        generator = BimodalDataGenerator(**kwargs)
        return generator.generate_batch
    elif generator_type == 'synthetic':
        generator = SyntheticDataGenerator(**kwargs)
        return generator
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def visualize_generator(generator: Callable[[], tf.Tensor], 
                       n_samples: int = 1000,
                       title: str = "Generated Data",
                       save_path: Optional[str] = None):
    """
    Visualize data from any generator.
    
    Args:
        generator: Data generator function
        n_samples: Number of samples to generate
        title: Plot title
        save_path: Optional path to save plot
    """
    # Generate samples
    samples = []
    batch_size = generator().shape[0]
    n_batches = n_samples // batch_size
    
    for _ in range(n_batches):
        batch = generator()
        samples.append(batch)
    
    data = tf.concat(samples, axis=0)
    
    # Create visualization
    if data.shape[-1] == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=30)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    else:
        logger.warning("Visualization only supported for 2D data")


# Example usage and testing
if __name__ == "__main__":
    # Test bimodal generator
    print("Testing BimodalDataGenerator...")
    bimodal_gen = BimodalDataGenerator(batch_size=8, n_features=2)
    batch = bimodal_gen.generate_batch()
    print(f"Generated batch shape: {batch.shape}")
    print(f"Sample values:\n{batch[:3]}")
    
    # Test synthetic generator
    print("\nTesting SyntheticDataGenerator...")
    synthetic_gen = SyntheticDataGenerator('circular', batch_size=8, radius=3.0)
    batch = synthetic_gen()
    print(f"Generated batch shape: {batch.shape}")
    print(f"Sample values:\n{batch[:3]}")
    
    print("\n✅ Data generators working correctly!")
