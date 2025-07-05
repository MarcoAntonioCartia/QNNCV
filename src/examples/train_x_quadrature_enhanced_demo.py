"""
Enhanced X-Quadrature Decoder Demo

Implements aggressive strategies to prevent vacuum collapse:
1. Aggressive Regularization (10-100x stronger)
3. Learnable Decoder Bias (bias toward target modes)
5. Modified Loss Function (penalize origin, reward X-quadrature magnitude)
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


class BiasedXQuadratureDecoder:
    """
    X-quadrature decoder with learnable bias to escape vacuum state.
    
    The bias pulls outputs toward target cluster centers.
    """
    
    def __init__(self, 
                 n_modes: int, 
                 output_dim: int, 
                 target_centers: List[Tuple[float, float]]):
        """
        Initialize biased decoder.
        
        Args:
            n_modes: Number of quantum modes
            output_dim: Output dimensionality (2 for 2D coordinates)
            target_centers: Target cluster centers to bias toward
        """
        self.n_modes = n_modes
        self.output_dim = output_dim
        self.target_centers = target_centers
        
        # Standard decoder matrix
        self.decoder_matrix = tf.Variable(
            tf.random.normal([n_modes, output_dim], stddev=0.1, seed=42),
            name="biased_decoder_matrix"
        )
        
        # Learnable bias that pulls toward target modes
        # Initialize bias to encourage outputs toward target centers
        center_bias = np.mean(target_centers, axis=0)  # Average of target centers
        self.bias = tf.Variable(
            tf.constant(center_bias, dtype=tf.float32),
            name="decoder_bias"
        )
        
        # Mode selection weights (learnable)
        self.mode_weights = tf.Variable(
            tf.random.normal([n_modes], stddev=0.5, seed=43),
            name="mode_selection_weights"
        )
        
        print(f"🎯 BiasedXQuadratureDecoder initialized:")
        print(f"   Decoder matrix: {self.decoder_matrix.shape}")
        print(f"   Bias: {self.bias.numpy()}")
        print(f"   Target centers: {target_centers}")
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        return [self.decoder_matrix, self.bias, self.mode_weights]
    
    def decode(self, x_quadrature: tf.Tensor) -> tf.Tensor:
        """
        Decode X-quadrature with bias and mode selection.
        
        Args:
            x_quadrature: X-quadrature measurements [batch_size, n_modes]
            
        Returns:
            Decoded output [batch_size, output_dim]
        """
        # Weight the X-quadrature values by mode importance
        weighted_x_quad = x_quadrature * tf.nn.softmax(self.mode_weights)
        
        # Standard matrix multiplication
        decoded = tf.matmul(weighted_x_quad, self.decoder_matrix)
        
        # Add learnable bias to pull away from origin
        biased_output = decoded + self.bias
        
        return biased_output


class EnhancedXQuadratureGenerator:
    """
    Enhanced X-quadrature generator with aggressive anti-vacuum strategies.
    """
    
    def __init__(self, 
                 latent_dim: int,
                 output_dim: int,
                 n_modes: int,
                 layers: int,
                 cutoff_dim: int,
                 target_centers: List[Tuple[float, float]],
                 coordinate_names: List[str] = ['X', 'Y'],
                 regularization_config: Dict = None):
        """Initialize enhanced X-quadrature generator."""
        
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
        
        # Aggressive X-quadrature regularizer
        self.regularizer = create_x_quadrature_regularizer(regularization_config)
        
        # Biased X-quadrature decoder
        self.x_quadrature_decoder = BiasedXQuadratureDecoder(
            n_modes=n_modes,
            output_dim=output_dim,
            target_centers=target_centers
        )
        
        self.n_modes = n_modes
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.target_centers = target_centers
        
        # Store latest measurements for regularization
        self.latest_x_quadrature = None
        self.latest_all_measurements = None
        self.latest_output = None
        
        print(f"🚀 EnhancedXQuadratureGenerator initialized:")
        print(f"   Architecture: {latent_dim}D → {n_modes} modes → X-quad only → {output_dim}D")
        print(f"   Aggressive regularization + bias + anti-origin loss")
        print(f"   Total parameters: {len(self.trainable_variables)}")
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        variables = []
        variables.extend(self.base_generator.trainable_variables)
        variables.extend(self.x_quadrature_decoder.trainable_variables)
        return variables
    
    def analyze_target_data(self, target_data: np.ndarray):
        """Analyze target data for cluster assignment."""
        self.base_generator.analyze_target_data(target_data)
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """Generate samples with enhanced anti-vacuum strategies."""
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
        
        # Decode using enhanced biased X-quadrature decoder
        output = self.x_quadrature_decoder.decode(batch_x_quadrature)
        self.latest_output = output
        
        # Update measurement history for visualization
        for measurements in quantum_measurements_list:
            self.distribution_analyzer.update_measurement_history(measurements)
        
        return output
    
    def compute_regularization_loss(self) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Compute enhanced regularization loss."""
        if self.latest_x_quadrature is None:
            return tf.constant(0.0), {}
        
        return self.regularizer.compute_regularization_loss(self.latest_x_quadrature)
    
    def compute_enhanced_generator_loss(self, fake_output: tf.Tensor, discriminator_output: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute enhanced generator loss with anti-origin and X-quadrature rewards.
        
        Args:
            fake_output: Generated samples
            discriminator_output: Discriminator's output on generated samples
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        loss_components = {}
        
        # Standard GAN loss
        gan_loss = -tf.reduce_mean(discriminator_output)
        loss_components['gan'] = gan_loss
        
        # Anti-origin penalty: strongly penalize outputs near (0,0)
        distances_from_origin = tf.norm(fake_output, axis=1)
        origin_penalty = tf.reduce_mean(tf.exp(-distances_from_origin * 3.0)) * 2.0
        loss_components['origin_penalty'] = origin_penalty
        
        # X-quadrature magnitude reward: encourage non-zero X-quadrature
        if self.latest_x_quadrature is not None:
            x_quad_magnitudes = tf.reduce_mean(tf.abs(self.latest_x_quadrature), axis=1)
            x_quad_reward = -tf.reduce_mean(x_quad_magnitudes) * 1.0
            loss_components['x_quad_reward'] = x_quad_reward
        else:
            x_quad_reward = tf.constant(0.0)
            loss_components['x_quad_reward'] = x_quad_reward
        
        # Mode separation reward: encourage outputs toward target centers
        mode_separation_reward = 0.0
        for center in self.target_centers:
            center_tensor = tf.constant(center, dtype=tf.float32)
            distances_to_center = tf.norm(fake_output - center_tensor, axis=1)
            # Reward when at least some samples are close to each target
            min_distance_to_center = tf.reduce_min(distances_to_center)
            mode_separation_reward += tf.exp(-min_distance_to_center * 2.0)
        
        mode_separation_reward = mode_separation_reward * 0.5
        loss_components['mode_separation_reward'] = mode_separation_reward
        
        # Regularization losses
        reg_loss, reg_components = self.compute_regularization_loss()
        loss_components.update(reg_components)
        
        # Total enhanced loss
        total_loss = (gan_loss + 
                     origin_penalty + 
                     x_quad_reward + 
                     mode_separation_reward + 
                     reg_loss)
        
        return total_loss, loss_components
    
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


class EnhancedXQuadratureTrainer:
    """
    Enhanced trainer with aggressive anti-vacuum strategies.
    """
    
    def __init__(self):
        """Initialize enhanced trainer."""
        # Configuration with faster performance for testing
        self.config = {
            'batch_size': 8,            
            'latent_dim': 4,            
            'n_modes': 3,               
            'layers': 2,                
            'cutoff_dim': 4,            
            'epochs': 10,               # More epochs to see evolution
            'steps_per_epoch': 3,       # Fewer steps for faster testing
            'learning_rate': 2e-3       # Higher learning rate
        }
        
        # AGGRESSIVE regularization configuration
        self.regularization_config = {
            'diversity_weight': 2.0,     # 13x stronger than before
            'vacuum_strength': 1.0,      # 12x stronger than before
            'separation_weight': 0.5,    # 16x stronger than before
            'enable_diversity': True,
            'enable_vacuum': True,
            'enable_separation': True
        }
        
        # Target centers for biasing
        self.target_centers = [(-1.5, -1.5), (1.5, 1.5)]
        
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
            'enhanced_loss_components': [],
            'x_quadrature_history': [],
            'regularization_metrics_history': []
        }
        
        print(f"🚀 EnhancedXQuadratureTrainer initialized:")
        print(f"   Configuration: {self.config}")
        print(f"   AGGRESSIVE Regularization: {self.regularization_config}")
        print(f"   Target centers: {self.target_centers}")
    
    def setup_models_and_data(self, target_data: np.ndarray):
        """Setup enhanced models and data."""
        print(f"\n🔧 Setting up enhanced models...")
        
        # Create enhanced generator
        self.generator = EnhancedXQuadratureGenerator(
            latent_dim=self.config['latent_dim'],
            output_dim=2,
            n_modes=self.config['n_modes'],
            layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim'],
            target_centers=self.target_centers,
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
        self.data_generator = BimodalDataGenerator(
            batch_size=self.config['batch_size'],
            n_features=2,
            mode1_center=self.target_centers[0],
            mode2_center=self.target_centers[1],
            mode_std=0.3
        )
        
        # Create bimodal visualization tracker
        output_dir = "results/x_quadrature_enhanced"
        self.bimodal_tracker = create_bimodal_tracker(
            save_dir=output_dir,
            target_centers=self.target_centers
        )
        
        # Create optimizers with higher learning rate
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], 
            beta_1=0.5
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], 
            beta_1=0.5
        )
        
        print(f"   ✅ Enhanced models created:")
        print(f"      Generator: {len(self.generator.trainable_variables)} parameters")
        print(f"      Discriminator: {len(self.discriminator.trainable_variables)} parameters")
        print(f"      Biased decoder: {len(self.generator.x_quadrature_decoder.trainable_variables)} parameters")
    
    def train_single_step(self, real_batch: tf.Tensor) -> Dict[str, float]:
        """Execute single training step with enhanced losses."""
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
        
        # Train generator with ENHANCED loss
        with tf.GradientTape() as g_tape:
            fake_batch = self.generator.generate(z)
            fake_output = self.discriminator.discriminate(fake_batch)
            
            # Enhanced generator loss with all improvements
            total_g_loss, loss_components = self.generator.compute_enhanced_generator_loss(
                fake_batch, fake_output
            )
        
        g_gradients = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Compute gradient flow
        g_grad_count = sum(1 for g in g_gradients if g is not None)
        d_grad_count = sum(1 for g in d_gradients if g is not None)
        g_grad_flow = g_grad_count / len(self.generator.trainable_variables)
        d_grad_flow = d_grad_count / len(self.discriminator.trainable_variables)
        
        # Convert loss components to float
        loss_components_float = {k: float(v) for k, v in loss_components.items()}
        
        return {
            'total_g_loss': float(total_g_loss),
            'd_loss': float(d_loss),
            'w_distance': float(w_distance),
            'g_gradient_flow': g_grad_flow,
            'd_gradient_flow': d_grad_flow,
            'loss_components': loss_components_float,
            'real_batch': real_batch.numpy(),
            'fake_batch': fake_batch.numpy(),
            'x_quadrature': self.generator.latest_x_quadrature.numpy()
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with enhanced tracking."""
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
            'total_g_loss': np.mean([s['total_g_loss'] for s in step_results]),
            'd_loss': np.mean([s['d_loss'] for s in step_results]),
            'w_distance': np.mean([s['w_distance'] for s in step_results]),
            'g_gradient_flow': np.mean([s['g_gradient_flow'] for s in step_results]),
            'd_gradient_flow': np.mean([s['d_gradient_flow'] for s in step_results])
        }
        
        # Average loss components
        loss_components_avg = {}
        for key in step_results[0]['loss_components'].keys():
            loss_components_avg[key] = np.mean([s['loss_components'][key] for s in step_results])
        
        # Get current measurements and regularization metrics
        current_measurements = self.generator.get_latest_measurements()
        reg_metrics = self.generator.get_regularization_metrics()
        
        x_quad_means = current_measurements['x_quadrature']['mean']
        
        # Update bimodal visualization tracker
        last_step = step_results[-1]
        self.bimodal_tracker.update(
            epoch=epoch,
            real_batch=last_step['real_batch'],
            fake_batch=last_step['fake_batch'],
            x_quadrature_batch=last_step['x_quadrature'],
            decoder_output=last_step['fake_batch']
        )
        
        # Store in history
        self.training_history['epochs'].append(epoch)
        self.training_history['epoch_times'].append(epoch_time)
        self.training_history['g_losses'].append(epoch_metrics['total_g_loss'])
        self.training_history['d_losses'].append(epoch_metrics['d_loss'])
        self.training_history['enhanced_loss_components'].append(loss_components_avg)
        self.training_history['x_quadrature_history'].append(x_quad_means)
        self.training_history['regularization_metrics_history'].append(reg_metrics)
        
        # Print results with enhanced metrics
        print(f"   📊 Results:")
        print(f"      Time: {epoch_time:.2f}s ({'✅ FAST' if epoch_time <= 7.0 else '⚠️ SLOW'})")
        print(f"      Losses: G={epoch_metrics['total_g_loss']:.4f}, D={epoch_metrics['d_loss']:.4f}")
        print(f"      Enhanced: Origin={loss_components_avg.get('origin_penalty', 0):.4f}, X-quad={loss_components_avg.get('x_quad_reward', 0):.4f}")
        print(f"      Gradient flow: G={epoch_metrics['g_gradient_flow']:.1%}, D={epoch_metrics['d_gradient_flow']:.1%}")
        print(f"      X-quadrature means: {[f'{x:.3f}' for x in x_quad_means]}")
        
        # Print key regularization metrics
        if 'vacuum_mean_abs_measurement' in reg_metrics:
            print(f"      Anti-vacuum: {reg_metrics['vacuum_mean_abs_measurement']:.3f} (>0 is good)")
        if 'diversity_batch_variance' in reg_metrics:
            print(f"      Diversity: {reg_metrics['diversity_batch_variance']:.3f} (>0 is good)")
        
        return epoch_metrics
    
    def train(self, target_data: np.ndarray) -> Dict:
        """Execute complete enhanced training."""
        print(f"🚀 Starting Enhanced X-Quadrature Training")
        print(f"=" * 70)
        print(f"Enhancements:")
        print(f"  • Aggressive regularization (10-100x stronger)")
        print(f"  • Learnable decoder bias toward target modes")
        print(f"  • Anti-origin penalty and X-quadrature rewards")
        print(f"=" * 70)
        
        # Setup
        self.setup_models_and_data(target_data)
        
        # Training loop
        epoch_results = []
        for epoch in range(self.config['epochs']):
            epoch_result = self.train_epoch(epoch)
            epoch_results.append(epoch_result)
        
        total_time = sum(self.training_history['epoch_times'])
        avg_epoch_time = np.mean(self.training_history['epoch_times'])
        
        print(f"\n" + "=" * 70)
        print(f"🎉 ENHANCED TRAINING COMPLETED!")
        print(f"=" * 70)
        print(f"Performance:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average epoch time: {avg_epoch_time:.2f}s")
        print(f"  Target: ≤7s/epoch ({'✅ ACHIEVED' if avg_epoch_time <= 7.0 else '❌ MISSED'})")
        
        # Generate all visualizations
        print(f"\n📊 Generating enhanced visualizations...")
        
        output_dir = "results/x_quadrature_enhanced"
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard quantum measurement visualizations
        prob_dist_path = os.path.join(output_dir, "probability_distributions.png")
        self.generator.visualize_probability_distributions(prob_dist_path)
        
        evolution_path = os.path.join(output_dir, "x_quadrature_evolution.png")
        self.generator.visualizer.plot_decoder_input_evolution(
            self.training_history['x_quadrature_history'], 
            evolution_path
        )
        
        # Enhanced training summary
        summary_path = os.path.join(output_dir, "training_summary_enhanced.png")
        self._plot_enhanced_training_summary(summary_path)
        
        # Comprehensive 2D bimodal visualizations
        bimodal_files = self.bimodal_tracker.generate_all_visualizations()
        
        print(f"📁 All enhanced visualizations saved to: {output_dir}")
        print(f"   Enhanced visualizations:")
        print(f"   • {prob_dist_path} - P(x), P(p), P(n) distributions")
        print(f"   • {evolution_path} - X-quadrature evolution")
        print(f"   • {summary_path} - Enhanced training summary")
        print(f"   Bimodal visualizations:")
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
    
    def _plot_enhanced_training_summary(self, save_path: str):
        """Plot enhanced training summary with all loss components."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('Enhanced X-Quadrature Decoder Training Summary', fontsize=16, fontweight='bold')
        
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
        
        # Enhanced loss components
        if self.training_history['enhanced_loss_components']:
            origin_penalties = [comp.get('origin_penalty', 0) for comp in self.training_history['enhanced_loss_components']]
            x_quad_rewards = [comp.get('x_quad_reward', 0) for comp in self.training_history['enhanced_loss_components']]
            
            axes[0, 2].plot(epochs, origin_penalties, 'purple', label='Origin Penalty')
            axes[0, 2].plot(epochs, x_quad_rewards, 'orange', label='X-Quad Reward')
            axes[0, 2].set_title('Enhanced Loss Components')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss Component')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
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
        
        # Regularization metrics evolution
        if self.training_history['regularization_metrics_history']:
            vacuum_metrics = []
            diversity_metrics = []
            for metrics in self.training_history['regularization_metrics_history']:
                vacuum_metrics.append(metrics.get('vacuum_mean_abs_measurement', 0))
                diversity_metrics.append(metrics.get('diversity_batch_variance', 0))
            
            axes[1, 1].plot(epochs, vacuum_metrics, 'purple', linewidth=2, label='Anti-vacuum')
            axes[1, 1].set_title('Anti-Vacuum Regularization')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Mean Abs Measurement')
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(epochs, diversity_metrics, 'orange', linewidth=2, label='Diversity')
            axes[1, 2].set_title('Diversity Regularization')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Batch Variance')
            axes[1, 2].grid(True, alpha=0.3)
        
        # Decoder bias evolution
        if self.training_history['enhanced_loss_components']:
            bias_values_x = []
            bias_values_y = []
            for i, epoch in enumerate(epochs):
                if hasattr(self.generator, 'x_quadrature_decoder'):
                    bias = self.generator.x_quadrature_decoder.bias.numpy()
                    bias_values_x.append(bias[0])
                    bias_values_y.append(bias[1])
                else:
                    bias_values_x.append(0)
                    bias_values_y.append(0)
            
            axes[2, 0].plot(epochs, bias_values_x, 'red', linewidth=2, label='X Bias')
            axes[2, 0].plot(epochs, bias_values_y, 'blue', linewidth=2, label='Y Bias')
            axes[2, 0].set_title('Learnable Decoder Bias Evolution')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Bias Value')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # Performance summary
        avg_time = np.mean(self.training_history['epoch_times'])
        target_achieved = avg_time <= 7.0
        
        axes[2, 1].text(0.1, 0.8, f"Average Epoch Time:", fontsize=12, fontweight='bold')
        axes[2, 1].text(0.1, 0.7, f"{avg_time:.2f} seconds", fontsize=14)
        axes[2, 1].text(0.1, 0.5, f"Target Achievement:", fontsize=12, fontweight='bold')
        axes[2, 1].text(0.1, 0.4, f"{'✅ SUCCESS' if target_achieved else '❌ MISSED'}", 
                        fontsize=14, color='green' if target_achieved else 'red')
        axes[2, 1].text(0.1, 0.25, f"Enhancements:", fontsize=12, fontweight='bold')
        axes[2, 1].text(0.1, 0.15, f"Aggressive Regularization", fontsize=12, color='green')
        axes[2, 1].text(0.1, 0.05, f"Learnable Bias + Anti-Origin", fontsize=12, color='green')
        axes[2, 1].set_xlim(0, 1)
        axes[2, 1].set_ylim(0, 1)
        axes[2, 1].axis('off')
        
        # Final metrics summary
        if self.training_history['x_quadrature_history']:
            final_x_quad = self.training_history['x_quadrature_history'][-1]
            final_reg_metrics = self.training_history['regularization_metrics_history'][-1] if self.training_history['regularization_metrics_history'] else {}
            
            axes[2, 2].text(0.1, 0.8, f"Final X-Quadrature:", fontsize=12, fontweight='bold')
            axes[2, 2].text(0.1, 0.7, f"{[f'{x:.3f}' for x in final_x_quad]}", fontsize=10)
            axes[2, 2].text(0.1, 0.5, f"Anti-Vacuum:", fontsize=12, fontweight='bold')
            axes[2, 2].text(0.1, 0.4, f"{final_reg_metrics.get('vacuum_mean_abs_measurement', 0):.3f}", fontsize=12)
            axes[2, 2].text(0.1, 0.25, f"Diversity:", fontsize=12, fontweight='bold')
            axes[2, 2].text(0.1, 0.15, f"{final_reg_metrics.get('diversity_batch_variance', 0):.3f}", fontsize=12)
            axes[2, 2].set_xlim(0, 1)
            axes[2, 2].set_ylim(0, 1)
            axes[2, 2].axis('off')
        
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
    """Main enhanced X-quadrature decoder demonstration."""
    print(f"🚀 ENHANCED X-QUADRATURE DECODER DEMONSTRATION")
    print(f"=" * 80)
    print(f"Enhanced Features:")
    print(f"  • Aggressive regularization (10-100x stronger)")
    print(f"  • Learnable decoder bias toward target modes")
    print(f"  • Anti-origin penalty (punish outputs near zero)")
    print(f"  • X-quadrature magnitude rewards")
    print(f"  • Mode separation rewards")
    print(f"  • 2D bimodal visualization with evolution tracking")
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
        print(f"\n🔧 Initializing enhanced trainer...")
        trainer = EnhancedXQuadratureTrainer()
        
        # Execute training
        print(f"\n🚀 Starting enhanced training...")
        results = trainer.train(target_data)
        
        # Final summary
        print(f"\n" + "=" * 80)
        print(f"🎯 ENHANCED X-QUADRATURE DEMONSTRATION COMPLETE!")
        print(f"=" * 80)
        
        print(f"Results:")
        print(f"  Average epoch time: {results['avg_epoch_time']:.2f}s")
        print(f"  Target achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
        print(f"  Decoder architecture: X-quadrature ONLY → coordinates")
        print(f"  Enhanced features: ALL implemented")
        print(f"  2D visualization: Comprehensive bimodal analysis")
        
        print(f"\nFiles created in: {results['output_directory']}")
        print(f"  Enhanced quantum visualizations:")
        print(f"  • probability_distributions.png - P(x), P(p), P(n) for each mode")
        print(f"  • x_quadrature_evolution.png - Decoder input evolution")
        print(f"  • training_summary_enhanced.png - Enhanced training analysis")
        print(f"  Enhanced 2D bimodal visualizations:")
        print(f"  • bimodal_evolution.gif - Animated real vs fake evolution")
        print(f"  • decoder_analysis.png - X-quadrature → output mapping")
        print(f"  • evolution_summary.png - Mode coverage/separation/balance")
        print(f"  • bimodal_metrics.json - Complete enhanced metrics")
        
        print(f"\n🎉 Enhanced demonstration with aggressive anti-vacuum strategies!")
        
        # Analysis of final results
        if results['training_history']['x_quadrature_history']:
            final_x_quad = results['training_history']['x_quadrature_history'][-1]
            print(f"\n📊 Final Analysis:")
            print(f"   Final X-quadrature values: {[f'{x:.3f}' for x in final_x_quad]}")
            print(f"   {'✅ SUCCESS' if max(final_x_quad) > 0.1 else '❌ STILL COLLAPSED'} - Vacuum escape")
        
    except Exception as e:
        print(f"❌ Enhanced demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
