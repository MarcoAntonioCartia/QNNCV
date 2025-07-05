"""
X-Quadrature Decoder Demo

Demonstrates pure quantum decoder that:
- Uses ONLY X-quadrature measurements for decoding (position-like)
- Extracts ALL measurements (X, P, photon) for visualization
- Shows probability distributions P(x), P(p), P(n) for quantum insight
- Optimized for fast training (~7s/epoch target)
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


class XQuadratureQuantumGenerator:
    """
    Enhanced Quantum Generator that uses ONLY X-quadrature for decoding.
    
    Architecture:
    Latent → [Static Encoding] → Quantum Circuit → [X-quadrature ONLY] → [Static Decoder] → Output
    
    All other measurements (P-quadrature, photon numbers) are extracted for visualization.
    """
    
    def __init__(self, 
                 latent_dim: int,
                 output_dim: int,
                 n_modes: int,
                 layers: int,
                 cutoff_dim: int,
                 coordinate_names: List[str] = ['X', 'Y']):
        """Initialize X-quadrature decoder generator."""
        
        # Base generator (will be modified to use only X-quadrature)
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
        
        # Modified decoder: X-quadrature only (n_modes → output_dim)
        self.x_quadrature_decoder = tf.Variable(
            tf.random.normal([n_modes, output_dim], stddev=0.1, seed=42),
            name="x_quadrature_decoder"
        )
        
        self.n_modes = n_modes
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        print(f"🔬 XQuadratureQuantumGenerator initialized:")
        print(f"   Architecture: {latent_dim}D → {n_modes} modes → X-quad only → {output_dim}D")
        print(f"   Decoder input: X-quadrature measurements only ({n_modes} values)")
        print(f"   Visualization: All measurements (X, P, photon)")
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
        Generate samples using ONLY X-quadrature measurements.
        
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
            
            # Generate through base quantum circuit using input encoder
            quantum_params = self.base_generator.input_encoder(sample_z)  # [1, n_modes * 2]
            
            # Execute quantum circuit to get state
            quantum_state = self.base_generator.quantum_circuit.execute(input_encoding=quantum_params[0])
            
            # Extract ALL measurements using enhanced system
            measurements = self.measurement_separator.extract_all_measurements(quantum_state)
            
            # Store for analysis
            quantum_measurements_list.append(measurements)
            
            # Use ONLY X-quadrature for decoder
            x_quadrature = measurements['x_quadrature']  # Shape: (n_modes,)
            x_quadrature_list.append(x_quadrature)
        
        # Stack X-quadrature measurements
        batch_x_quadrature = tf.stack([tf.constant(x, dtype=tf.float32) for x in x_quadrature_list], axis=0)
        
        # Decode using X-quadrature ONLY
        output = tf.matmul(batch_x_quadrature, self.x_quadrature_decoder)
        
        # Update measurement history for visualization
        for measurements in quantum_measurements_list:
            self.distribution_analyzer.update_measurement_history(measurements)
        
        return output
    
    def get_latest_measurements(self) -> Dict[str, np.ndarray]:
        """Get latest quantum measurements for analysis."""
        # Generate a test sample to get measurements
        z_test = tf.random.normal([1, self.latent_dim])
        _ = self.generate(z_test)  # This updates the measurement history
        
        return self.distribution_analyzer.get_distribution_statistics()
    
    def visualize_probability_distributions(self, save_path: Optional[str] = None):
        """Create probability distribution plots."""
        distributions = self.distribution_analyzer.compute_probability_distributions()
        return self.visualizer.plot_probability_distributions(distributions, save_path)


class OptimizedXQuadratureTrainer:
    """
    Optimized trainer for X-quadrature decoder demo.
    
    Configuration optimized for ~7s/epoch target based on scaling analysis.
    """
    
    def __init__(self):
        """Initialize optimized trainer."""
        # Optimized configuration (based on scaling analysis)
        self.config = {
            'batch_size': 4,            # Keep efficient batch size
            'latent_dim': 4,            # Reasonable latent space
            'n_modes': 4,               # REDUCED from 4 to hit 7s target
            'layers': 3,                # Keep depth for quality
            'cutoff_dim': 6,            # CRITICAL: Keep at 4
            'epochs': 10,                # Quick demo
            'steps_per_epoch': 10,       # Quick demo
            'learning_rate': 1e-3
        }
        
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.data_generator = None
        
        # Training history
        self.training_history = {
            'epochs': [],
            'epoch_times': [],
            'g_losses': [],
            'd_losses': [],
            'x_quadrature_history': []
        }
        
        print(f"🎯 OptimizedXQuadratureTrainer initialized:")
        print(f"   Target epoch time: ~7s (3 modes vs 4 modes)")
        print(f"   Configuration: {self.config}")
    
    def setup_models_and_data(self, target_data: np.ndarray):
        """Setup models and data generators."""
        print(f"\n🔧 Setting up optimized models...")
        
        # Create X-quadrature generator
        self.generator = XQuadratureQuantumGenerator(
            latent_dim=self.config['latent_dim'],
            output_dim=2,
            n_modes=self.config['n_modes'],
            layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim'],
            coordinate_names=['X', 'Y']
        )
        
        # Create discriminator (also optimized)
        self.discriminator = PureSFDiscriminator(
            input_dim=2,
            n_modes=self.config['n_modes'],  # Reduced to 3
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
        
        # Create optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], 
            beta_1=0.5
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], 
            beta_1=0.5
        )
        
        print(f"   ✅ Models created:")
        print(f"      Generator: {len(self.generator.trainable_variables)} parameters")
        print(f"      Discriminator: {len(self.discriminator.trainable_variables)} parameters")
    
    def train_single_step(self, real_batch: tf.Tensor) -> Dict[str, float]:
        """Execute single training step."""
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
        
        # Train generator
        with tf.GradientTape() as g_tape:
            fake_batch = self.generator.generate(z)
            fake_output = self.discriminator.discriminate(fake_batch)
            g_loss = -tf.reduce_mean(fake_output)
        
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Compute gradient flow
        g_grad_count = sum(1 for g in g_gradients if g is not None)
        d_grad_count = sum(1 for g in d_gradients if g is not None)
        g_grad_flow = g_grad_count / len(self.generator.trainable_variables)
        d_grad_flow = d_grad_count / len(self.discriminator.trainable_variables)
        
        return {
            'g_loss': float(g_loss),
            'd_loss': float(d_loss),
            'w_distance': float(w_distance),
            'g_gradient_flow': g_grad_flow,
            'd_gradient_flow': d_grad_flow,
            'generated_samples': fake_batch.numpy()
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
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
        
        # Aggregate metrics
        epoch_metrics = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'g_loss': np.mean([s['g_loss'] for s in step_results]),
            'd_loss': np.mean([s['d_loss'] for s in step_results]),
            'w_distance': np.mean([s['w_distance'] for s in step_results]),
            'g_gradient_flow': np.mean([s['g_gradient_flow'] for s in step_results]),
            'd_gradient_flow': np.mean([s['d_gradient_flow'] for s in step_results])
        }
        
        # Get current X-quadrature measurements
        current_measurements = self.generator.get_latest_measurements()
        x_quad_means = current_measurements['x_quadrature']['mean']
        
        # Store in history
        self.training_history['epochs'].append(epoch)
        self.training_history['epoch_times'].append(epoch_time)
        self.training_history['g_losses'].append(epoch_metrics['g_loss'])
        self.training_history['d_losses'].append(epoch_metrics['d_loss'])
        self.training_history['x_quadrature_history'].append(x_quad_means)
        
        # Print results
        print(f"   📊 Results:")
        print(f"      Time: {epoch_time:.2f}s ({'✅ FAST' if epoch_time <= 7.0 else '⚠️ SLOW'})")
        print(f"      Losses: G={epoch_metrics['g_loss']:.4f}, D={epoch_metrics['d_loss']:.4f}")
        print(f"      Gradient flow: G={epoch_metrics['g_gradient_flow']:.1%}, D={epoch_metrics['d_gradient_flow']:.1%}")
        print(f"      X-quadrature means: {[f'{x:.3f}' for x in x_quad_means]}")
        
        return epoch_metrics
    
    def train(self, target_data: np.ndarray) -> Dict:
        """Execute complete training with visualization."""
        print(f"🚀 Starting X-Quadrature Decoder Training")
        print(f"=" * 60)
        
        # Setup
        self.setup_models_and_data(target_data)
        
        # Training loop
        epoch_results = []
        for epoch in range(self.config['epochs']):
            epoch_result = self.train_epoch(epoch)
            epoch_results.append(epoch_result)
        
        total_time = sum(self.training_history['epoch_times'])
        avg_epoch_time = np.mean(self.training_history['epoch_times'])
        
        print(f"\n" + "=" * 60)
        print(f"🎉 TRAINING COMPLETED!")
        print(f"=" * 60)
        print(f"Performance:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average epoch time: {avg_epoch_time:.2f}s")
        print(f"  Target: ≤7s/epoch ({'✅ ACHIEVED' if avg_epoch_time <= 7.0 else '❌ MISSED'})")
        
        # Generate visualizations
        print(f"\n📊 Generating quantum measurement visualizations...")
        
        # Create output directory
        output_dir = "results/x_quadrature_demo"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Probability distributions
        prob_dist_path = os.path.join(output_dir, "probability_distributions.png")
        self.generator.visualize_probability_distributions(prob_dist_path)
        
        # 2. X-quadrature evolution
        evolution_path = os.path.join(output_dir, "x_quadrature_evolution.png")
        self.generator.visualizer.plot_decoder_input_evolution(
            self.training_history['x_quadrature_history'], 
            evolution_path
        )
        
        # 3. Training summary
        summary_path = os.path.join(output_dir, "training_summary.png")
        self._plot_training_summary(summary_path)
        
        print(f"📁 Visualizations saved to: {output_dir}")
        
        return {
            'config': self.config,
            'epoch_results': epoch_results,
            'training_history': self.training_history,
            'avg_epoch_time': avg_epoch_time,
            'target_achieved': avg_epoch_time <= 7.0,
            'output_directory': output_dir
        }
    
    def _plot_training_summary(self, save_path: str):
        """Plot training summary."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('X-Quadrature Decoder Training Summary', fontsize=16, fontweight='bold')
        
        epochs = self.training_history['epochs']
        
        # Epoch times
        axes[0, 0].plot(epochs, self.training_history['epoch_times'], 'o-', linewidth=2)
        axes[0, 0].axhline(y=7.0, color='red', linestyle='--', label='Target (7s)')
        axes[0, 0].set_title('Epoch Times')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training losses
        axes[0, 1].plot(epochs, self.training_history['g_losses'], 'b-', label='Generator')
        axes[0, 1].plot(epochs, self.training_history['d_losses'], 'r-', label='Discriminator')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # X-quadrature evolution
        for mode in range(min(self.config['n_modes'], 3)):
            mode_values = [x_quad[mode] if len(x_quad) > mode else 0 
                          for x_quad in self.training_history['x_quadrature_history']]
            axes[1, 0].plot(epochs, mode_values, 'o-', label=f'X_{mode} (→ Decoder)')
        axes[1, 0].set_title('X-Quadrature Evolution (Decoder Inputs)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('X-Quadrature Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary
        avg_time = np.mean(self.training_history['epoch_times'])
        target_achieved = avg_time <= 7.0
        
        axes[1, 1].text(0.1, 0.8, f"Average Epoch Time:", fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.7, f"{avg_time:.2f} seconds", fontsize=14)
        axes[1, 1].text(0.1, 0.5, f"Target Achievement:", fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.4, f"{'✅ SUCCESS' if target_achieved else '❌ MISSED'}", 
                        fontsize=14, color='green' if target_achieved else 'red')
        axes[1, 1].text(0.1, 0.2, f"Decoder Input:", fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.1, f"X-quadrature ONLY", fontsize=14, color='blue')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
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
    """Main X-quadrature decoder demonstration."""
    print(f"🔬 X-QUADRATURE DECODER DEMONSTRATION")
    print(f"=" * 80)
    print(f"Features:")
    print(f"  • Decoder uses ONLY X-quadrature measurements (position-like)")
    print(f"  • Extracts ALL measurements (X, P, photon) for visualization")
    print(f"  • Shows probability distributions P(x), P(p), P(n)")
    print(f"  • Optimized for ~7s/epoch (3 modes instead of 4)")
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
        print(f"\n🔧 Initializing optimized trainer...")
        trainer = OptimizedXQuadratureTrainer()
        
        # Execute training
        print(f"\n🚀 Starting training...")
        results = trainer.train(target_data)
        
        # Final summary
        print(f"\n" + "=" * 80)
        print(f"🎯 X-QUADRATURE DECODER DEMONSTRATION COMPLETE!")
        print(f"=" * 80)
        
        print(f"Results:")
        print(f"  Average epoch time: {results['avg_epoch_time']:.2f}s")
        print(f"  Target achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
        print(f"  Decoder architecture: X-quadrature ONLY → coordinates")
        print(f"  Quantum measurements: Comprehensive P(x), P(p), P(n) distributions")
        
        print(f"\nFiles created in: {results['output_directory']}")
        print(f"  • probability_distributions.png - P(x), P(p), P(n) for each mode")
        print(f"  • x_quadrature_evolution.png - Decoder input evolution")
        print(f"  • training_summary.png - Complete training analysis")
        
        print(f"\n🎉 Demonstration shows pure quantum decoder with X-quadrature only!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
