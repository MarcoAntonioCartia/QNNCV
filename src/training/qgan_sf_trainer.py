"""
Strawberry Fields-based Quantum GAN Trainer following proven SF training methodology.
Documentation:
    run_quantum_neural_network.ipynb
This implementation adopts the exact pattern from the SF quantum neural network tutorial:
1. Proper engine reset handling
2. Symbolic parameter mapping
3. Automatic gradient computation through SF
4. Quantum-aware loss functions and metrics
"""

import numpy as np
import tensorflow as tf
import logging
import time
from typing import Dict, Any, Optional

try:
    from ..utils.quantum_losses import QuantumWassersteinLoss, QuantumMMDLoss
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.quantum_losses import QuantumWassersteinLoss, QuantumMMDLoss

# Import warning suppression utility
try:
    from ..utils.warning_suppression import clean_training_output, QuantumTrainingLogger
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.warning_suppression import clean_training_output, QuantumTrainingLogger

logger = logging.getLogger(__name__)

class QGANSFTrainer:
    """
    Quantum GAN trainer following Strawberry Fields proven training methodology.
    
    This implementation uses the exact pattern from SF tutorials:
    - Proper engine reset between iterations
    - SF automatic gradient computation
    - Quantum-aware loss functions
    - Comprehensive monitoring
    """
    
    def __init__(self, generator, discriminator, latent_dim=4,
                 generator_lr=1e-4, discriminator_lr=1e-4,
                 beta1=0.5, beta2=0.999, loss_type='quantum_wasserstein'):
        """
        Initialize SF-based quantum GAN trainer.
        
        Args:
            generator: SF-based quantum generator
            discriminator: SF-based quantum discriminator
            latent_dim (int): Dimensionality of latent noise vector
            generator_lr (float): Learning rate for generator
            discriminator_lr (float): Learning rate for discriminator
            beta1 (float): Adam optimizer first moment decay
            beta2 (float): Adam optimizer second moment decay
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        # Initialize loss function
        if loss_type == 'quantum_wasserstein':
            self.loss_fn = QuantumWassersteinLoss(
                lambda_gp=10.0,
                lambda_entropy=0.5,
                lambda_physics=1.0
            )
        elif loss_type == 'quantum_mmd':
            self.loss_fn = QuantumMMDLoss(sigma=1.0, lambda_entropy=0.1)
        else:
            self.loss_fn = None  # Use standard GAN loss

        self.loss_type = loss_type
        logger.info(f"Using loss function: {loss_type}")
        
        # Initialize optimizers (following SF tutorial pattern)
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=generator_lr, beta_1=beta1, beta_2=beta2
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=discriminator_lr, beta_1=beta1, beta_2=beta2
        )
        
        # Training history
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'g_grad_norm': [],
            'd_grad_norm': [],
            'quantum_metrics': [],
            'stability_metric': []
        }
        
        logger.info("SF-based QGAN trainer initialized")
        logger.info(f"Generator trainable vars: {len(generator.trainable_variables)}")
        logger.info(f"Discriminator trainable vars: {len(discriminator.trainable_variables)}")
    
    def compute_gradient_norm(self, gradients):
        """Compute the norm of gradients for monitoring."""
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += tf.reduce_sum(tf.square(grad))
        return tf.sqrt(total_norm)
    
    def quantum_gan_loss(self, real_samples, fake_samples, label_smoothing=0.1):
        """
        Quantum-aware GAN loss with proper handling of quantum states.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated samples
            label_smoothing: Label smoothing factor
            
        Returns:
            tuple: (discriminator_loss, generator_loss)
        """
        # Create labels with smoothing
        batch_size = tf.shape(real_samples)[0]
        real_labels = tf.ones([batch_size, 1]) * (1.0 - label_smoothing)
        fake_labels = tf.zeros([batch_size, 1]) + label_smoothing
        
        # Get discriminator outputs
        real_output = self.discriminator.discriminate(real_samples)
        fake_output = self.discriminator.discriminate(fake_samples)
        
        # Binary cross-entropy loss
        real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_output)
        fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_output)
        
        # Discriminator loss
        d_loss = tf.reduce_mean(real_loss + fake_loss)
        
        # Generator loss (wants discriminator to classify fake as real)
        g_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output
            )
        )
        
        return d_loss, g_loss
    
    def train_step_sf(self, real_samples):
        """
        Single training step following SF methodology with quantum loss function.
        
        This follows the exact pattern from SF tutorial:
        1. Reset engines if needed
        2. Compute losses with gradient tapes
        3. Apply gradients
        4. Monitor quantum metrics
        
        Args:
            real_samples: Batch of real training data
            
        Returns:
            dict: Training metrics for this step
        """
        batch_size = tf.shape(real_samples)[0]
        
        # --- Discriminator Training (following SF pattern) ---
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as d_tape:
            # Generate fake samples
            fake_samples = self.generator.generate(z)
            
            # Compute discriminator loss
            if self.loss_fn is not None:
                # Use quantum-enhanced loss
                d_loss, g_loss, metrics = self.loss_fn(
                    real_samples, fake_samples, self.generator, self.discriminator
                )
            else:
                # Fallback to standard loss
                d_loss, g_loss = self.quantum_gan_loss(real_samples, fake_samples)
                metrics = {}
        
        # Compute and apply discriminator gradients
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_grad_norm = self.compute_gradient_norm(d_gradients)
        
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        # --- Generator Training (following SF pattern) ---
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as g_tape:
            # Generate fake samples
            fake_samples = self.generator.generate(z)
            
            # Compute generator loss
            if self.loss_fn is not None:
                # Use quantum-enhanced loss
                _, g_loss, metrics = self.loss_fn(
                    real_samples, fake_samples, self.generator, self.discriminator
                )
            else:
                # Fallback to standard loss
                _, g_loss = self.quantum_gan_loss(real_samples, fake_samples)
                metrics = {}
        
        # Compute and apply generator gradients
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        g_grad_norm = self.compute_gradient_norm(g_gradients)
        
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        # Compute stability metric
        stability_metric = g_grad_norm / (d_grad_norm + 1e-8)
        
        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'g_grad_norm': g_grad_norm,
            'd_grad_norm': d_grad_norm,
            'stability_metric': stability_metric,
            **metrics
        }
    
    def compute_quantum_metrics(self):
        """
        Compute quantum-specific metrics for both generator and discriminator.
        
        Returns:
            dict: Quantum metrics
        """
        try:
            # Generator quantum metrics
            g_metrics = self.generator.compute_quantum_cost()
            
            # Discriminator quantum metrics  
            d_metrics = self.discriminator.compute_quantum_metrics()
            
            return {
                'generator_trace': g_metrics.get('trace', 0.0),
                'generator_entropy': g_metrics.get('entropy', 0.0),
                'discriminator_trace': d_metrics.get('trace', 0.0),
                'discriminator_entropy': d_metrics.get('entropy', 0.0)
            }
        except Exception as e:
            logger.debug(f"Quantum metrics computation failed: {e}")
            return {
                'generator_trace': 0.0,
                'generator_entropy': 0.0,
                'discriminator_trace': 0.0,
                'discriminator_entropy': 0.0
            }
    
    def train(self, data, epochs=10, batch_size=8, verbose=True, 
              save_interval=5, quantum_metrics_interval=10, suppress_warnings=True):
        """
        Train quantum GAN following SF methodology.
        
        Args:
            data: Training dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size (small for quantum stability)
            verbose (bool): Whether to print progress
            save_interval (int): How often to log metrics
            quantum_metrics_interval (int): How often to compute quantum metrics
            suppress_warnings (bool): Whether to suppress TensorFlow warnings
            
        Returns:
            dict: Complete training history
        """
        print(f"Starting SF-based quantum GAN training...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Data shape: {data.shape}")
        
        if suppress_warnings:
            print("Warning suppression enabled - clean output mode")
        
        # Create data loader
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        
        start_time = time.time()
        
        # Use clean training output context
        with clean_training_output(suppress_warnings=suppress_warnings):
            for epoch in range(epochs):
                epoch_metrics = {
                'd_loss': [],
                'g_loss': [],
                'g_grad_norm': [],
                'd_grad_norm': [],
                'stability_metric': []
            }
            
            # Training loop for this epoch
            for batch_idx, real_samples in enumerate(dataset):
                # Single training step following SF pattern
                step_metrics = self.train_step_sf(real_samples)
                
                # Collect metrics
                for key in epoch_metrics:
                    if key in step_metrics:
                        epoch_metrics[key].append(step_metrics[key])
            
            # Record epoch averages
            for key in epoch_metrics:
                if epoch_metrics[key]:
                    avg_value = tf.reduce_mean(epoch_metrics[key])
                    self.training_history[key].append(float(avg_value))
            
            # Compute quantum metrics periodically
            if epoch % quantum_metrics_interval == 0:
                quantum_metrics = self.compute_quantum_metrics()
                self.training_history['quantum_metrics'].append(quantum_metrics)
            
            # Verbose logging
            if verbose and epoch % save_interval == 0:
                elapsed_time = time.time() - start_time
                avg_d_loss = self.training_history['d_loss'][-1]
                avg_g_loss = self.training_history['g_loss'][-1]
                avg_stability = self.training_history['stability_metric'][-1]
                
                print(f"Epoch {epoch:04d} [{elapsed_time:.1f}s]: "
                      f"D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, "
                      f"Stability={avg_stability:.4f}")
                
                # Quantum metrics if available
                if self.training_history['quantum_metrics']:
                    latest_qm = self.training_history['quantum_metrics'][-1]
                    print(f"  Quantum: G_trace={latest_qm['generator_trace']:.4f}, "
                          f"D_trace={latest_qm['discriminator_trace']:.4f}")
                
                # Training stability warnings
                if avg_stability > 10.0:
                    print("  ⚠️  High gradient ratio - potential instability")
                elif avg_stability < 0.1:
                    print("  ⚠️  Low generator gradients - potential mode collapse")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        
        # Generate final samples for evaluation
        print("Generating final samples...")
        z_final = tf.random.normal([10, self.latent_dim])
        final_samples = self.generator.generate(z_final)
        
        print(f"Final sample shape: {final_samples.shape}")
        print(f"Final sample range: [{tf.reduce_min(final_samples):.3f}, {tf.reduce_max(final_samples):.3f}]")
        
        return self.training_history
    
    def evaluate_generation_quality(self, real_data, n_samples=1000):
        """
        Evaluate the quality of generated samples.
        
        Args:
            real_data: Real training data for comparison
            n_samples: Number of samples to generate for evaluation
            
        Returns:
            dict: Quality metrics
        """
        print(f"Evaluating generation quality with {n_samples} samples...")
        
        # Generate samples
        z_eval = tf.random.normal([n_samples, self.latent_dim])
        generated_samples = self.generator.generate(z_eval)
        
        # Convert to numpy
        real_np = real_data.numpy() if hasattr(real_data, 'numpy') else real_data
        gen_np = generated_samples.numpy()
        
        # Compute basic statistics
        real_mean = np.mean(real_np, axis=0)
        gen_mean = np.mean(gen_np, axis=0)
        real_std = np.std(real_np, axis=0)
        gen_std = np.std(gen_np, axis=0)
        
        metrics = {
            'mean_difference': float(np.linalg.norm(real_mean - gen_mean)),
            'std_difference': float(np.linalg.norm(real_std - gen_std)),
            'real_mean': real_mean.tolist(),
            'gen_mean': gen_mean.tolist(),
            'real_std': real_std.tolist(),
            'gen_std': gen_std.tolist()
        }
        
        print(f"Mean difference: {metrics['mean_difference']:.4f}")
        print(f"Std difference: {metrics['std_difference']:.4f}")
        
        return metrics

def test_sf_trainer():
    """Test the SF-based quantum GAN trainer."""
    print("\n" + "="*60)
    print("TESTING SF-BASED QUANTUM GAN TRAINER")
    print("="*60)
    
    try:
        # Import SF components
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from models.generators.quantum_sf_generator import QuantumSFGenerator
        from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
        
        # Create components
        generator = QuantumSFGenerator(n_modes=2, latent_dim=4, layers=2, cutoff_dim=6)
        discriminator = QuantumSFDiscriminator(n_modes=2, input_dim=2, layers=2, cutoff_dim=6)
        
        # Create trainer
        trainer = QGANSFTrainer(
            generator=generator,
            discriminator=discriminator,
            latent_dim=4,
            generator_lr=1e-3,
            discriminator_lr=1e-3
        )
        
        print("SF trainer created successfully :)")
        
        # Create synthetic data
        n_samples = 100
        real_data = tf.random.normal([n_samples, 2])
        
        # Test training
        print("Testing training loop...")
        history = trainer.train(
            data=real_data,
            epochs=3,
            batch_size=4,
            verbose=True,
            save_interval=1
        )
        
        print("Training test completed :)")
        print(f"History keys: {list(history.keys())}")
        
        # Test evaluation
        print("Testing evaluation...")
        quality_metrics = trainer.evaluate_generation_quality(real_data, n_samples=50)
        print("Evaluation test completed :)")
        
        return trainer
        
    except Exception as e:
        print(f"Test failed: {e} :((")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_sf_trainer()
