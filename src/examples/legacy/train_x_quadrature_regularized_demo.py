"""
X-Quadrature Regularized Demo

Enhanced X-quadrature decoder with regularization to prevent vacuum collapse.
Includes comprehensive 2D visualization to see what the decoder actually outputs.
"""

import numpy as np
import tensorflow as tf
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.generators.clusterized_quantum_generator import ClusterizedQuantumGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from src.training.data_generators import BimodalDataGenerator
from src.quantum.measurements.enhanced_measurement_extractor import create_enhanced_measurement_system
from src.losses.x_quadrature_regularization import create_x_quadrature_regularizer
from src.utils.bimodal_visualization_tracker import create_bimodal_tracker


class RegularizedXQuadratureGenerator:
    """
    X-quadrature generator with anti-vacuum regularization.
    
    Prevents X-quadrature measurements from collapsing to zero while maintaining
    pure quantum decoder architecture.
    """
    
    def __init__(self, 
                 latent_dim: int,
                 output_dim: int,
                 n_modes: int,
                 layers: int,
                 cutoff_dim: int,
                 coordinate_names: List[str] = ['X', 'Y'],
                 regularization_config: Dict = None):
        """Initialize regularized X-quadrature generator."""
        
        # Base generator
        self.base_generator = ClusterizedQuantumGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            n_modes=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim,
            clustering_method='kmeans',
            coordinate_names=coordinate_names
        )
        
        # Enhanced measurement system
        self.measurement_separator, self.distribution_analyzer, self.visualizer = \
            create_enhanced_measurement_system(n_modes, cutoff_dim)
        
        # X-quadrature regularizer
        self.regularizer = create_x_quadrature_regularizer(regularization_config)
        
        # X-quadrature decoder (pure quantum)
        self.x_quadrature_decoder = tf.Variable(
            tf.random.normal([n_modes, output_dim], stddev=0.1, seed=42),
            name="x_quadrature_decoder"
        )
        
        self.n_modes = n_modes
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Store latest measurements for regularization
        self.latest_x_quadrature = None
        self.latest_all_measurements = None
        
        print(f"🔬 RegularizedXQuadratureGenerator initialized:")
        print(f"   Architecture: {latent_dim}D → {n_modes} modes → X-quad only → {output_dim}D")
        print(f"   Regularization: Anti-vacuum + diversity + mode separation")
        print(f"   Total parameters: {len(self.trainable_variables)}")
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        variables = []
        variables.extend(self.base_generator.trainable_variables)
        variables.append(self.x_quadrature_decoder)
        return variables
    
    def analyze_target_data(self, target_data: np.ndarray):
        """Analyze target data for cluster assignment."""
        self.base_generator.analyze_target_data(target_data)
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate samples using regularized X-quadrature decoder.
        
        Args:
            z: Latent input tensor
            
        Returns:
            Generated output tensor
        """
        batch_size = tf.shape(z)[0]
        
        # Process through quantum circuit (individual samples)
        quantum_measurements_list = []
        x_quadrature_list = []
        
        for i in range(batch_size):
            sample_z = z[i:i+1]
            
            # Generate through base quantum circuit
            quantum_params = self.base_generator.input_encoder(sample_z)
            
            # Execute quantum circuit
            quantum_state = self.base_generator.quantum_circuit.execute(input_encoding=quantum_params[0])
            
            # Extract ALL measurements
            measurements = self.measurement_separator.extract_all_measurements(quantum_state)
            
            # Store for analysis
            quantum_measurements_list.append(measurements)
            x_quadrature_list.append(measurements['x_quadrature'])
        
        # Stack X-quadrature measurements
        batch_x_quadrature = tf.stack([tf.constant(x, dtype=tf.float32) for x in x_quadrature_list], axis=0)
        
        # Store for regularization
        self.latest_x_quadrature = batch_x_quadrature
        self.latest_all_measurements = quantum_measurements_list
        
        # Decode using X-quadrature ONLY
        output = tf.matmul(batch_x_quadrature, self.x_quadrature_decoder)
        
        # Update measurement history for visualization
        for measurements in quantum_measurements_list:
            self.distribution_analyzer.update_measurement_history(measurements)
        
        return output
    
    def compute_regularization_loss(self) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute regularization loss for latest X-quadrature measurements.
        
        Returns:
            Tuple of (total_regularization_loss, loss_components)
        """
        if self.latest_x_quadrature is None:
            return tf.constant(0.0), {}
        
        return self.regularizer.compute_regularization_loss(self.latest_x_quadrature)
    
    def get_regularization_metrics(self) -> Dict[str, any]:
        """Get comprehensive regularization metrics."""
        if self.latest_x_quadrature is None:
            return {}
        
        return self.regularizer.get_comprehensive_metrics(self.latest_x_quadrature)
    
    def get_latest_measurements(self) -> Dict[str, np.ndarray]:
        """Get latest quantum measurements for analysis."""
        return self.distribution_analyzer.get_distribution_statistics()
    
    def visualize_probability_distributions(self, save_path: Optional[str] = None):
        """Create probability distribution plots."""
        distributions = self.distribution_analyzer.compute_probability_distributions()
        return self.visualizer.plot_probability_distributions(distributions, save_path)


class RegularizedXQuadratureTrainer:
    """
    Enhanced trainer with regularization and 2D visualization.
    """
    
    def __init__(self):
        """Initialize regularized trainer."""
        # Optimized configuration
        self.config = {
            'batch_size': 8,            # Larger batches for better regularization
            'latent_dim': 4,            
            'n_modes': 4,               # Optimized for ~7s/epoch
            'layers': 4,                
            'cutoff_dim': 6,            
            'epochs': 8,                # More epochs to see evolution
            'steps_per_epoch': 5,       
            'learning_rate': 1e-3
        }
        
        # Regularization configuration
        self.regularization_config = {
            'diversity_weight': 0.15,    # Strong diversity encouragement
            'vacuum_strength': 0.08,     # Strong anti-vacuum penalty
            'separation_weight': 0.03,   # Moderate mode separation
            'enable_diversity': True,
            'enable_vacuum': True,
            'enable_separation': True
        }
        
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.data_generator = None
        self.bimodal_tracker = None
        
        # Training history
        self.training_history = {
            'epochs': [],
            'epoch_times': [],
            'g_losses': [],
            'd_losses': [],
            'regularization_losses': [],
            'x_quadrature_history': [],
            'regularization_metrics_history': []
        }
        
        print(f"🎯 RegularizedXQuadratureTrainer initialized:")
        print(f"   Configuration: {self.config}")
        print(f"   Regularization: {self.regularization_config}")
    
    def setup_models_and_data(self, target_data: np.ndarray):
        """Setup models, data, and visualization tracker."""
        print(f"\n🔧 Setting up regularized models...")
        
        # Create regularized generator
        self.generator = RegularizedXQuadratureGenerator(
            latent_dim=self.config['latent_dim'],
            output_dim=2,
            n_modes=self.config['n_modes'],
            layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim'],
            coordinate_names=['X', 'Y'],
            regularization_config=self.regularization_config
        )
        
        # Create discriminator
        self.discriminator = PureSFDiscriminator(
            input_dim=2,
            n_modes=self.config['n_modes'],
            layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim']
        )
        
        # Analyze target data
        self.generator.analyze_target_data(target_data)
        
        # Create data generator
        if hasattr(self.generator.base_generator, 'cluster_centers') and \
           self.generator.base_generator.cluster_centers is not None and \
           len(self.generator.base_generator.cluster_centers) >= 2:
            mode1_center = tuple(self.generator.base_generator.cluster_centers[0])
            mode2_center = tuple(self.generator.base_generator.cluster_centers[1])
        else:
            mode1_center = (-1.5, -1.5)
            mode2_center = (1.5, 1.5)
        
        self.data_generator = BimodalDataGenerator(
            batch_size=self.config['batch_size'],
            n_features=2,
            mode1_center=mode1_center,
            mode2_center=mode2_center,
            mode_std=0.3
        )
        
        # Create bimodal visualization tracker
        output_dir = "results/x_quadrature_regularized"
        self.bimodal_tracker = create_bimodal_tracker(
            save_dir=output_dir,
            target_centers=[mode1_center, mode2_center]
        )
        
        # Create optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], 
            beta_1=0.5
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], 
            beta_1=0.5
        )
        
        print(f"   ✅ Models and tracker created:")
        print(f"      Generator: {len(self.generator.trainable_variables)} parameters")
        print(f"      Discriminator: {len(self.discriminator.trainable_variables)} parameters")
        print(f"      Target centers: {[mode1_center, mode2_center]}")
    
    def train_single_step(self, real_batch: tf.Tensor) -> Dict[str, float]:
        """Execute single training step with regularization."""
        batch_size = tf.shape(real_batch)[0]
        z = tf.random.normal([batch_size, self.config['latent_dim']])
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            fake_batch = self.generator.generate(z)
            
            real_output = self.discriminator.discriminate(real_batch)
            fake_output = self.discriminator.discriminate(fake_batch)
            
            w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
            d_loss = -w_distance
        
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator with regularization
        with tf.GradientTape() as g_tape:
            fake_batch = self.generator.generate(z)
            fake_output = self.discriminator.discriminate(fake_batch)
            
            # Standard GAN loss
            g_loss = -tf.reduce_mean(fake_output)
            
            # Add regularization losses
            reg_loss, reg_components = self.generator.compute_regularization_loss()
            
            # Total generator loss
            total_g_loss = g_loss + reg_loss
        
        g_gradients = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Compute gradient flow
        g_grad_count = sum(1 for g in g_gradients if g is not None)
        d_grad_count = sum(1 for g in d_gradients if g is not None)
        g_grad_flow = g_grad_count / len(self.generator.trainable_variables)
        d_grad_flow = d_grad_count / len(self.discriminator.trainable_variables)
        
        return {
            'g_loss': float(g_loss),
            'd_loss': float(d_loss),
            'reg_loss': float(reg_loss),
            'total_g_loss': float(total_g_loss),
            'w_distance': float(w_distance),
            'g_gradient_flow': g_grad_flow,
            'd_gradient_flow': d_grad_flow,
            'real_batch': real_batch.numpy(),
            'fake_batch': fake_batch.numpy(),
            'x_quadrature': self.generator.latest_x_quadrature.numpy()
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with comprehensive tracking."""
        print(f"\n📈 Epoch {epoch + 1}/{self.config['epochs']}")
        epoch_start = time.time()
        
        step_results = []
        
        for step in range(self.config['steps_per_epoch']):
            print(f"   Step {step + 1}/{self.config['steps_per_epoch']}... ", end='', flush=True)
            
            real_batch = self.data_generator.generate_batch()
            step_result = self.train_single_step(real_batch)
            step_results.append(step_result)
            
            print(f"✅")
        
        epoch_time = time.time() - epoch_start
        
        # Aggregate epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'g_loss': np.mean([s['g_loss'] for s in step_results]),
            'd_loss': np.mean([s['d_loss'] for s in step_results]),
            'reg_loss': np.mean([s['reg_loss'] for s in step_results]),
            'total_g_loss': np.mean([s['total_g_loss'] for s in step_results]),
            'w_distance': np.mean([s['w_distance'] for s in step_results]),
            'g_gradient_flow': np.mean([s['g_gradient_flow'] for s in step_results]),
            'd_gradient_flow': np.mean([s['d_gradient_flow'] for s in step_results])
        }
        
        # Get current measurements and regularization metrics
        current_measurements = self.generator.get_latest_measurements()
        reg_metrics = self.generator.get_regularization_metrics()
        
        x_quad_means = current_measurements['x_quadrature']['mean']
        
        # Update bimodal visualization tracker
        # Use data from last step of epoch
        last_step = step_results[-1]
        self.bimodal_tracker.update(
            epoch=epoch,
            real_batch=last_step['real_batch'],
            fake_batch=last_step['fake_batch'],
            x_quadrature_batch=last_step['x_quadrature'],
            decoder_output=last_step['fake_batch']  # Final output is decoder output
        )
        
        # Store in history
        self.training_history['epochs'].append(epoch)
        self.training_history['epoch_times'].append(epoch_time)
        self.training_history['g_losses'].append(epoch_metrics['g_loss'])
        self.training_history['d_losses'].append(epoch_metrics['d_loss'])
        self.training_history['regularization_losses'].append(epoch_metrics['reg_loss'])
        self.training_history['x_quadrature_history'].append(x_quad_means)
        self.training_history['regularization_metrics_history'].append(reg_metrics)
        
        # Print results
        print(f"   📊 Results:")
        print(f"      Time: {epoch_time:.2f}s ({'✅ FAST' if epoch_time <= 7.0 else '⚠️ SLOW'})")
        print(f"      Losses: G={epoch_metrics['g_loss']:.4f}, D={epoch_metrics['d_loss']:.4f}, Reg={epoch_metrics['reg_loss']:.4f}")
        print(f"      Gradient flow: G={epoch_metrics['g_gradient_flow']:.1%}, D={epoch_metrics['d_gradient_flow']:.1%}")
        print(f"      X-quadrature means: {[f'{x:.3f}' for x in x_quad_means]}")
        
        # Print key regularization metrics
        if 'vacuum_mean_abs_measurement' in reg_metrics:
            print(f"      Anti-vacuum: {reg_metrics['vacuum_mean_abs_measurement']:.3f} (>0 is good)")
        if 'diversity_batch_variance' in reg_metrics:
            print(f"      Diversity: {reg_metrics['diversity_batch_variance']:.3f} (>0 is good)")
        
        return epoch_metrics
    
    def train(self, target_data: np.ndarray) -> Dict:
        """Execute complete training with visualization."""
        print(f"🚀 Starting Regularized X-Quadrature Training")
        print(f"=" * 65)
        
        # Setup
        self.setup_models_and_data(target_data)
        
        # Training loop
        epoch_results = []
        for epoch in range(self.config['epochs']):
            epoch_result = self.train_epoch(epoch)
            epoch_results.append(epoch_result)
        
        total_time = sum(self.training_history['epoch_times'])
        avg_epoch_time = np.mean(self.training_history['epoch_times'])
        
        print(f"\n" + "=" * 65)
        print(f"🎉 REGULARIZED TRAINING COMPLETED!")
        print(f"=" * 65)
        print(f"Performance:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average epoch time: {avg_epoch_time:.2f}s")
        print(f"  Target: ≤7s/epoch ({'✅ ACHIEVED' if avg_epoch_time <= 7.0 else '❌ MISSED'})")
        
        # Generate all visualizations
        print(f"\n📊 Generating comprehensive visualizations...")
        
        # 1. Standard quantum measurement visualizations
        output_dir = "results/x_quadrature_regularized"
        os.makedirs(output_dir, exist_ok=True)
        
        prob_dist_path = os.path.join(output_dir, "probability_distributions.png")
        self.generator.visualize_probability_distributions(prob_dist_path)
        
        evolution_path = os.path.join(output_dir, "x_quadrature_evolution.png")
        self.generator.visualizer.plot_decoder_input_evolution(
            self.training_history['x_quadrature_history'], 
            evolution_path
        )
        
        # 2. Training summary with regularization
        summary_path = os.path.join(output_dir, "training_summary_regularized.png")
        self._plot_regularized_training_summary(summary_path)
        
        # 3. NEW: Comprehensive 2D bimodal visualizations
        bimodal_files = self.bimodal_tracker.generate_all_visualizations()
        
        print(f"📁 All visualizations saved to: {output_dir}")
        print(f"   Standard visualizations:")
        print(f"   • {prob_dist_path} - P(x), P(p), P(n) distributions")
        print(f"   • {evolution_path} - X-quadrature evolution")
        print(f"   • {summary_path} - Training summary with regularization")
        print(f"   NEW 2D bimodal visualizations:")
        for viz_type, path in bimodal_files.items():
            if path:
                print(f"   • {path} - {viz_type}")
        
        return {
            'config': self.config,
            'regularization_config': self.regularization_config,
            'epoch_results': epoch_results,
            'training_history': self.training_history,
            'avg_epoch_time': avg_epoch_time,
            'target_achieved': avg_epoch_time <= 7.0,
            'output_directory': output_dir,
            'bimodal_files': bimodal_files
        }
    
    def _plot_regularized_training_summary(self, save_path: str):
        """Plot training summary with regularization metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Regularized X-Quadrature Decoder Training Summary', fontsize=16, fontweight='bold')
        
        epochs = self.training_history['epochs']
        
        # Epoch times
        axes[0, 0].plot(epochs, self.training_history['epoch_times'], 'o-', linewidth=2)
        axes[0, 0].axhline(y=7.0, color='red', linestyle='--', label='Target (7s)')
        axes[0, 0].set_title('Epoch Times')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training losses with regularization
        axes[0, 1].plot(epochs, self.training_history['g_losses'], 'b-', label='Generator')
        axes[0, 1].plot(epochs, self.training_history['d_losses'], 'r-', label='Discriminator')
        axes[0, 1].plot(epochs, self.training_history['regularization_losses'], 'g-', label='Regularization')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # X-quadrature evolution
        for mode in range(min(self.config['n_modes'], 3)):
            mode_values = [x_quad[mode] if len(x_quad) > mode else 0 
                          for x_quad in self.training_history['x_quadrature_history']]
            axes[0, 2].plot(epochs, mode_values, 'o-', label=f'X_{mode} (→ Decoder)')
        axes[0, 2].set_title('X-Quadrature Evolution (Decoder Inputs)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('X-Quadrature Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Regularization metrics evolution
        if self.training_history['regularization_metrics_history']:
            # Extract vacuum metrics
            vacuum_metrics = []
            diversity_metrics = []
            for metrics in self.training_history['regularization_metrics_history']:
                vacuum_metrics.append(metrics.get('vacuum_mean_abs_measurement', 0))
                diversity_metrics.append(metrics.get('diversity_batch_variance', 0))
            
            axes[1, 0].plot(epochs, vacuum_metrics, 'purple', linewidth=2, label='Anti-vacuum')
            axes[1, 0].set_title('Anti-Vacuum Regularization')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Mean Abs Measurement')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(epochs, diversity_metrics, 'orange', linewidth=2, label='Diversity')
            axes[1, 1].set_title('Diversity Regularization')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Batch Variance')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary
        avg_time = np.mean(self.training_history['epoch_times'])
        target_achieved = avg_time <= 7.0
        
        axes[1, 2].text(0.1, 0.8, f"Average Epoch Time:", fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.7, f"{avg_time:.2f} seconds", fontsize=14)
        axes[1, 2].text(0.1, 0.5, f"Target Achievement:", fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.4, f"{'✅ SUCCESS' if target_achieved else '❌ MISSED'}", 
                        fontsize=14, color='green' if target_achieved else 'red')
        axes[1, 2].text(0.1, 0.25, f"Decoder Input:", fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.15, f"X-quadrature ONLY", fontsize=14, color='blue')
        axes[1, 2].text(0.1, 0.05, f"Regularization: ENABLED", fontsize=14, color='green')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_target_data(n_samples: int = 100) -> np.ndarray:
    """Create bimodal target data."""
    np.random.seed(42)
    
    cluster1 = np.random.normal([-1.5, -1.5], 0.3, (n_samples // 2, 2))
    cluster2 = np.random.normal([1.5, 1.5], 0.3, (n_samples // 2, 2))
    
    data = np.vstack([cluster1, cluster2])
    np.random.shuffle(data)
    
    return data


def main():
    """Main regularized X-quadrature decoder demonstration."""
    print(f"🔬 REGULARIZED X-QUADRATURE DECODER DEMONSTRATION")
    print(f"=" * 80)
    print(f"Features:")
    print(f"  • Decoder uses ONLY X-quadrature measurements (position-like)")
    print(f"  • Anti-vacuum regularization prevents zero collapse")
    print(f"  • Diversity regularization maintains mode separation")
    print(f"  • 2D bimodal visualization shows actual decoder output")
    print(f"  • Animated GIF shows evolution over training")
    print(f"=" * 80)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Create target data
        print(f"\n📊 Creating target data...")
        target_data = create_target_data(n_samples=100)
        print(f"   Target data shape: {target_data.shape}")
        
        # Create trainer
        print(f"\n🔧 Initializing regularized trainer...")
        trainer = RegularizedXQuadratureTrainer()
        
        # Execute training
        print(f"\n🚀 Starting regularized training...")
        results = trainer.train(target_data)
        
        # Final summary
        print(f"\n" + "=" * 80)
        print(f"🎯 REGULARIZED X-QUADRATURE DEMONSTRATION COMPLETE!")
        print(f"=" * 80)
        
        print(f"Results:")
        print(f"  Average epoch time: {results['avg_epoch_time']:.2f}s")
        print(f"  Target achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
        print(f"  Decoder architecture: X-quadrature ONLY → coordinates")
        print(f"  Regularization: Anti-vacuum + diversity + mode separation")
        print(f"  2D visualization: Comprehensive bimodal analysis")
        
        print(f"\nFiles created in: {results['output_directory']}")
        print(f"  Standard quantum visualizations:")
        print(f"  • probability_distributions.png - P(x), P(p), P(n) for each mode")
        print(f"  • x_quadrature_evolution.png - Decoder input evolution")
        print(f"  • training_summary_regularized.png - Training with regularization")
        print(f"  NEW 2D bimodal visualizations:")
        print(f"  • bimodal_evolution.gif - Animated real vs fake evolution")
        print(f"  • decoder_analysis.png - X-quadrature → output mapping")
        print(f"  • evolution_summary.png - Mode coverage/separation/balance")
        print(f"  • bimodal_metrics.json - Complete metrics data")
        
        print(f"\n🎉 Demonstration shows regularized quantum decoder with X-quadrature only!")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
