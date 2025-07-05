"""
Comprehensive Generator Mode Collapse Comparison Test

This script tests all available quantum generators with standardized parameters
to identify mode collapse issues and compare their bimodal learning capabilities.

Test Configuration:
- 10 epochs, 5 steps each
- 4 modes, 3 layers, cutoff_dim 5
- Classical discriminator for speed
- Bimodal target data (two distinct clusters)
- Evolution visualization through training
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all generators to test
from src.models.generators.clusterized_quantum_generator import ClusterizedQuantumGenerator
from src.models.generators.constellation_sf_generator import ConstellationSFGenerator
from src.models.generators.pure_sf_generator import PureSFGenerator
from src.models.generators.sf_tutorial_generator import SFTutorialGenerator

# Import utilities
from src.training.data_generators import BimodalDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class SimpleClassicalDiscriminator:
    """Simple classical discriminator for fast training."""
    
    def __init__(self, input_dim: int = 2):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def discriminate(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)
    
    @property
    def trainable_variables(self):
        return self.model.trainable_variables


class ModeCollapseAnalyzer:
    """Analyzes mode collapse in generated samples."""
    
    @staticmethod
    def analyze_bimodal_quality(samples: np.ndarray, target_centers: np.ndarray) -> Dict[str, float]:
        """
        Analyze quality of bimodal generation.
        
        Args:
            samples: Generated samples [n_samples, 2]
            target_centers: Target cluster centers [2, 2]
            
        Returns:
            Dictionary of quality metrics
        """
        if len(samples) == 0:
            return {
                'mode_balance': 0.0,
                'separation_distance': 0.0,
                'cluster_variance': 0.0,
                'mode_collapse_score': 1.0  # 1.0 = complete collapse
            }
        
        # Calculate distances to each target center
        center1, center2 = target_centers[0], target_centers[1]
        
        dist_to_center1 = np.linalg.norm(samples - center1, axis=1)
        dist_to_center2 = np.linalg.norm(samples - center2, axis=1)
        
        # Assign samples to closest center
        assignments = (dist_to_center1 < dist_to_center2).astype(int)
        
        # Mode balance (0.5 = perfect balance)
        mode1_ratio = np.mean(assignments == 0)
        mode2_ratio = np.mean(assignments == 1)
        mode_balance = min(mode1_ratio, mode2_ratio) / max(mode1_ratio, mode2_ratio)
        
        # Separation distance between generated cluster centers
        if np.sum(assignments == 0) > 0 and np.sum(assignments == 1) > 0:
            gen_center1 = np.mean(samples[assignments == 0], axis=0)
            gen_center2 = np.mean(samples[assignments == 1], axis=0)
            separation_distance = np.linalg.norm(gen_center1 - gen_center2)
        else:
            separation_distance = 0.0
        
        # Cluster variance (diversity within clusters)
        cluster_variances = []
        for mode in [0, 1]:
            mode_samples = samples[assignments == mode]
            if len(mode_samples) > 1:
                cluster_variances.append(np.var(mode_samples))
            else:
                cluster_variances.append(0.0)
        cluster_variance = np.mean(cluster_variances)
        
        # Mode collapse score (0.0 = no collapse, 1.0 = complete collapse)
        target_separation = np.linalg.norm(center1 - center2)
        if target_separation > 0:
            collapse_score = 1.0 - (separation_distance / target_separation)
            collapse_score = max(0.0, min(1.0, collapse_score))
        else:
            collapse_score = 1.0
        
        return {
            'mode_balance': mode_balance,
            'separation_distance': separation_distance,
            'cluster_variance': cluster_variance,
            'mode_collapse_score': collapse_score,
            'mode1_ratio': mode1_ratio,
            'mode2_ratio': mode2_ratio
        }


class GeneratorTester:
    """Tests individual generators for mode collapse."""
    
    def __init__(self, 
                 epochs: int = 10,
                 steps_per_epoch: int = 5,
                 n_modes: int = 4,
                 layers: int = 3,
                 cutoff_dim: int = 5,
                 latent_dim: int = 6,
                 output_dim: int = 2):
        
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Create bimodal target data
        self.data_generator = BimodalDataGenerator(
            batch_size=200,
            n_features=2,
            mode1_center=(-1.5, -1.5),
            mode2_center=(1.5, 1.5),
            mode_std=0.3
        )
        
        self.target_data = self.data_generator.generate_batch()
        self.target_centers = np.array([[-1.5, -1.5], [1.5, 1.5]])
        
        # Create discriminator
        self.discriminator = SimpleClassicalDiscriminator(input_dim=output_dim)
        
        # Analysis tools
        self.analyzer = ModeCollapseAnalyzer()
        
        logger.info(f"Generator tester initialized:")
        logger.info(f"  Training: {epochs} epochs × {steps_per_epoch} steps")
        logger.info(f"  Quantum: {n_modes} modes, {layers} layers, cutoff {cutoff_dim}")
        logger.info(f"  Target data: {len(self.target_data)} bimodal samples")
    
    def test_generator(self, generator_class, generator_name: str) -> Dict[str, Any]:
        """
        Test a single generator for mode collapse.
        
        Args:
            generator_class: Generator class to instantiate
            generator_name: Name for logging and results
            
        Returns:
            Test results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {generator_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Create generator with standardized parameters
            if generator_name == "ClusterizedQuantumGenerator":
                generator = generator_class(
                    latent_dim=self.latent_dim,
                    output_dim=self.output_dim,
                    n_modes=self.n_modes,
                    layers=self.layers,
                    cutoff_dim=self.cutoff_dim
                )
                # Analyze target data for clusterized generator
                generator.analyze_target_data(self.target_data)
            else:
                generator = generator_class(
                    latent_dim=self.latent_dim,
                    output_dim=self.output_dim,
                    n_modes=self.n_modes,
                    layers=self.layers,
                    cutoff_dim=self.cutoff_dim
                )
            
            logger.info(f"Generator created with {len(generator.trainable_variables)} trainable variables")
            
            # Training history
            training_history = {
                'epochs': [],
                'generator_loss': [],
                'discriminator_loss': [],
                'mode_collapse_score': [],
                'mode_balance': [],
                'separation_distance': [],
                'generated_samples': []  # Store samples for visualization
            }
            
            # Training loop
            for epoch in range(self.epochs):
                epoch_start = time.time()
                
                epoch_g_losses = []
                epoch_d_losses = []
                
                for step in range(self.steps_per_epoch):
                    # Generate batch
                    z = tf.random.normal([16, self.latent_dim])
                    
                    # Train discriminator
                    with tf.GradientTape() as d_tape:
                        fake_samples = generator.generate(z)
                        real_batch = tf.constant(
                            self.target_data[np.random.choice(len(self.target_data), 16)],
                            dtype=tf.float32
                        )
                        
                        d_real = self.discriminator.discriminate(real_batch)
                        d_fake = self.discriminator.discriminate(fake_samples)
                        
                        # Simple binary cross-entropy loss
                        d_loss = -tf.reduce_mean(tf.math.log(d_real + 1e-8) + 
                                               tf.math.log(1 - d_fake + 1e-8))
                    
                    d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
                    self.discriminator.optimizer.apply_gradients(
                        zip(d_gradients, self.discriminator.trainable_variables)
                    )
                    
                    # Train generator
                    with tf.GradientTape() as g_tape:
                        fake_samples = generator.generate(z)
                        d_fake = self.discriminator.discriminate(fake_samples)
                        
                        # Generator wants to fool discriminator
                        g_loss = -tf.reduce_mean(tf.math.log(d_fake + 1e-8))
                    
                    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
                    
                    # Check for gradient issues
                    valid_gradients = [g for g in g_gradients if g is not None]
                    if len(valid_gradients) == 0:
                        logger.warning(f"No valid gradients at epoch {epoch}, step {step}")
                        continue
                    
                    # Apply gradients with simple Adam optimizer
                    if not hasattr(generator, 'optimizer'):
                        generator.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    
                    generator.optimizer.apply_gradients(
                        zip(g_gradients, generator.trainable_variables)
                    )
                    
                    epoch_g_losses.append(float(g_loss))
                    epoch_d_losses.append(float(d_loss))
                
                # Analyze generation quality
                test_z = tf.random.normal([100, self.latent_dim])
                test_samples = generator.generate(test_z).numpy()
                
                quality_metrics = self.analyzer.analyze_bimodal_quality(
                    test_samples, self.target_centers
                )
                
                # Record history
                training_history['epochs'].append(epoch)
                training_history['generator_loss'].append(np.mean(epoch_g_losses))
                training_history['discriminator_loss'].append(np.mean(epoch_d_losses))
                training_history['mode_collapse_score'].append(quality_metrics['mode_collapse_score'])
                training_history['mode_balance'].append(quality_metrics['mode_balance'])
                training_history['separation_distance'].append(quality_metrics['separation_distance'])
                training_history['generated_samples'].append(test_samples.copy())
                
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Epoch {epoch+1:2d}: "
                          f"G_loss={np.mean(epoch_g_losses):.4f}, "
                          f"D_loss={np.mean(epoch_d_losses):.4f}, "
                          f"Collapse={quality_metrics['mode_collapse_score']:.3f}, "
                          f"Balance={quality_metrics['mode_balance']:.3f}, "
                          f"Sep={quality_metrics['separation_distance']:.3f} "
                          f"({epoch_time:.1f}s)")
            
            total_time = time.time() - start_time
            
            # Final analysis
            final_samples = training_history['generated_samples'][-1]
            final_metrics = self.analyzer.analyze_bimodal_quality(
                final_samples, self.target_centers
            )
            
            results = {
                'generator_name': generator_name,
                'success': True,
                'training_time': total_time,
                'final_metrics': final_metrics,
                'training_history': training_history,
                'parameter_count': len(generator.trainable_variables),
                'target_centers': self.target_centers.tolist(),
                'config': {
                    'epochs': self.epochs,
                    'steps_per_epoch': self.steps_per_epoch,
                    'n_modes': self.n_modes,
                    'layers': self.layers,
                    'cutoff_dim': self.cutoff_dim
                }
            }
            
            logger.info(f"✅ {generator_name} completed successfully!")
            logger.info(f"   Final collapse score: {final_metrics['mode_collapse_score']:.3f}")
            logger.info(f"   Final mode balance: {final_metrics['mode_balance']:.3f}")
            logger.info(f"   Training time: {total_time:.1f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {generator_name} failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'generator_name': generator_name,
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }


def create_evolution_visualization(results: List[Dict[str, Any]], save_path: str):
    """Create comprehensive evolution visualization."""
    
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        logger.error("No successful results to visualize")
        return
    
    n_generators = len(successful_results)
    fig, axes = plt.subplots(2, n_generators, figsize=(5*n_generators, 10))
    
    if n_generators == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Generator Mode Collapse Comparison - Bimodal Evolution', 
                 fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, result in enumerate(successful_results):
        generator_name = result['generator_name']
        history = result['training_history']
        target_centers = np.array(result['target_centers'])
        
        # Plot 1: Final generation comparison
        ax1 = axes[0, i]
        
        # Plot target data
        target_data = BimodalDataGenerator(
            batch_size=100,
            n_features=2,
            mode1_center=(-1.5, -1.5),
            mode2_center=(1.5, 1.5),
            mode_std=0.3
        ).generate_batch()
        
        ax1.scatter(target_data[:, 0], target_data[:, 1], 
                   alpha=0.6, c='blue', s=30, label='Target')
        
        # Plot final generated samples
        final_samples = history['generated_samples'][-1]
        ax1.scatter(final_samples[:, 0], final_samples[:, 1], 
                   alpha=0.6, c='red', s=30, label='Generated')
        
        # Plot target centers
        ax1.scatter(target_centers[:, 0], target_centers[:, 1], 
                   c='black', s=100, marker='x', linewidth=3, label='Target Centers')
        
        ax1.set_title(f'{generator_name}\nCollapse: {result["final_metrics"]["mode_collapse_score"]:.3f}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # Plot 2: Training metrics evolution
        ax2 = axes[1, i]
        
        epochs = history['epochs']
        ax2.plot(epochs, history['mode_collapse_score'], 'r-', label='Mode Collapse', linewidth=2)
        ax2.plot(epochs, history['mode_balance'], 'b-', label='Mode Balance', linewidth=2)
        
        ax2.set_title(f'{generator_name} - Training Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Evolution visualization saved to: {save_path}")
    plt.show()


def main():
    """Run comprehensive generator comparison test."""
    
    print("🧪 Starting Comprehensive Generator Mode Collapse Test")
    print("="*70)
    
    # Test configuration
    config = {
        'epochs': 10,
        'steps_per_epoch': 5,
        'n_modes': 4,
        'layers': 3,
        'cutoff_dim': 5,
        'latent_dim': 6,
        'output_dim': 2
    }
    
    print(f"Test Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Generators to test
    generators_to_test = [
        (ClusterizedQuantumGenerator, "ClusterizedQuantumGenerator"),
        (ConstellationSFGenerator, "ConstellationSFGenerator"),
        (PureSFGenerator, "PureSFGenerator"),
        (SFTutorialGenerator, "SFTutorialGenerator")
    ]
    
    # Create tester
    tester = GeneratorTester(**config)
    
    # Run tests
    all_results = []
    
    for generator_class, generator_name in generators_to_test:
        try:
            result = tester.test_generator(generator_class, generator_name)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to test {generator_name}: {e}")
            all_results.append({
                'generator_name': generator_name,
                'success': False,
                'error': str(e)
            })
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/generator_comparison_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(results_dir, "comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create visualization
    viz_file = os.path.join(results_dir, "generator_comparison_evolution.png")
    create_evolution_visualization(all_results, viz_file)
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATOR COMPARISON SUMMARY")
    print("="*70)
    
    successful_tests = [r for r in all_results if r['success']]
    failed_tests = [r for r in all_results if not r['success']]
    
    print(f"Successful tests: {len(successful_tests)}/{len(all_results)}")
    
    if successful_tests:
        print("\nMode Collapse Rankings (lower is better):")
        successful_tests.sort(key=lambda x: x['final_metrics']['mode_collapse_score'])
        
        for i, result in enumerate(successful_tests):
            metrics = result['final_metrics']
            print(f"  {i+1}. {result['generator_name']}")
            print(f"     Collapse Score: {metrics['mode_collapse_score']:.3f}")
            print(f"     Mode Balance: {metrics['mode_balance']:.3f}")
            print(f"     Separation: {metrics['separation_distance']:.3f}")
            print(f"     Training Time: {result['training_time']:.1f}s")
    
    if failed_tests:
        print(f"\nFailed tests: {len(failed_tests)}")
        for result in failed_tests:
            print(f"  ❌ {result['generator_name']}: {result.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
