import tensorflow as tf
import numpy as np
import sys
import os

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import load_dataset
from utils.visualization import plot_results
from utils.tensorflow_compat import QuantumExecutionContext, configure_tensorflow_for_quantum, suppress_complex_warnings

class QGAN:
    """
    Quantum Generative Adversarial Network implementation for adversarial training.
    
    This class implements the adversarial training framework for quantum GANs,
    supporting both classical and quantum generators and discriminators. The
    training process alternates between optimizing the generator to produce
    realistic samples and the discriminator to distinguish real from generated data.
    
    Mathematical Framework:
    The adversarial objective follows the minimax formulation:
    min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
    
    Quantum-specific considerations:
    - Gradient clipping prevents parameter divergence in quantum circuits
    - Learning rate scheduling accommodates quantum parameter optimization
    - Stability monitoring detects training instabilities specific to quantum systems
    """
    
    def __init__(self, generator, discriminator, latent_dim=10, 
                 generator_lr=0.0001, discriminator_lr=0.0001,
                 beta1=0.5, beta2=0.999, gradient_clip_norm=1.0):
        """
        Initialize quantum GAN training framework.
        
        This method sets up the adversarial training components including
        optimizers, learning rate schedules, and metric tracking systems
        for monitoring the training dynamics.
        
        Args:
            generator: Generator network instance (classical or quantum)
            discriminator: Discriminator network instance (classical or quantum)
            latent_dim (int): Dimensionality of latent noise vector z
            generator_lr (float): Learning rate for generator optimization
            discriminator_lr (float): Learning rate for discriminator optimization
            beta1 (float): Adam optimizer first moment decay rate
            beta2 (float): Adam optimizer second moment decay rate
            gradient_clip_norm (float): Maximum gradient norm for clipping
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gradient_clip_norm = gradient_clip_norm
        
        # Initialize Adam optimizers for adversarial training
        self.g_optimizer = tf.optimizers.Adam(
            learning_rate=generator_lr, beta_1=beta1, beta_2=beta2
        )
        self.d_optimizer = tf.optimizers.Adam(
            learning_rate=discriminator_lr, beta_1=beta1, beta_2=beta2
        )
        
        # Exponential decay schedules for learning rate adaptation
        self.g_scheduler = tf.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=generator_lr,
            decay_steps=100,
            decay_rate=0.98
        )
        self.d_scheduler = tf.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=discriminator_lr,
            decay_steps=100,
            decay_rate=0.98
        )
        
        # Training history for loss and metric tracking
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'g_grad_norm': [],
            'd_grad_norm': [],
            'stability_metric': []
        }
    
    def wasserstein_loss(self, real_samples, fake_samples, lambda_gp=10.0):
        """
        Wasserstein loss with gradient penalty for Lipschitz constraint enforcement.
        
        This loss function implements the Wasserstein-1 distance with gradient penalty
        (WGAN-GP) which provides more stable training dynamics compared to the original
        GAN objective. The gradient penalty enforces the Lipschitz constraint required
        for the Kantorovich-Rubinstein duality.
        
        Mathematical formulation:
        L_D = E[D(x_fake)] - E[D(x_real)] + λ * E[(||∇D(x_hat)||_2 - 1)^2]
        L_G = -E[D(G(z))]
        
        Args:
            real_samples: Real data samples from training distribution
            fake_samples: Generated samples from generator network
            lambda_gp: Gradient penalty coefficient (typically 10)
            
        Returns:
            tuple: (discriminator_loss, generator_loss, gradient_penalty)
        """
        batch_size = tf.shape(real_samples)[0]
        
        # Compute discriminator outputs for real and fake samples
        real_output = self.discriminator.discriminate(real_samples)
        fake_output = self.discriminator.discriminate(fake_samples)
        
        # Wasserstein distance estimation
        wasserstein_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        # Gradient penalty computation for Lipschitz constraint
        alpha = tf.random.uniform([batch_size, 1], minval=0, maxval=1)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_output = self.discriminator.discriminate(interpolated)
        
        gradients = gp_tape.gradient(interpolated_output, interpolated)
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = lambda_gp * tf.reduce_mean(tf.square(gradient_norm - 1.0))
        
        # Final loss computations
        d_loss = -wasserstein_distance + gradient_penalty
        g_loss = -tf.reduce_mean(fake_output)
        
        return d_loss, g_loss, gradient_penalty
    
    def traditional_gan_loss(self, real_samples, fake_samples, label_smoothing=0.1):
        """
        Traditional GAN loss with label smoothing for training stabilization.
        
        This implements the original GAN objective with binary cross-entropy loss
        and label smoothing to reduce the discriminator's overconfidence and
        improve training stability, particularly important for quantum components.
        
        Mathematical formulation:
        L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
        L_G = -E[log D(G(z))]
        
        Args:
            real_samples: Real data samples from training distribution
            fake_samples: Generated samples from generator network
            label_smoothing: Amount of label smoothing (0.0 to 0.5)
            
        Returns:
            tuple: (discriminator_loss, generator_loss)
        """
        # Apply label smoothing to reduce discriminator overconfidence
        real_labels = tf.ones_like(real_samples[:, 0:1]) * (1.0 - label_smoothing)
        fake_labels = tf.zeros_like(fake_samples[:, 0:1]) + label_smoothing
        
        # Compute discriminator outputs
        real_output = self.discriminator.discriminate(real_samples)
        fake_output = self.discriminator.discriminate(fake_samples)
        
        # Binary cross-entropy loss computation
        real_loss = tf.losses.binary_crossentropy(real_labels, real_output)
        fake_loss = tf.losses.binary_crossentropy(fake_labels, fake_output)
        
        # Discriminator loss combines real and fake sample losses
        d_loss = tf.reduce_mean(real_loss + fake_loss)
        
        # Generator loss encourages discriminator to classify fake as real
        g_loss = tf.reduce_mean(tf.losses.binary_crossentropy(
            tf.ones_like(fake_output), fake_output
        ))
        
        return d_loss, g_loss
    
    def compute_gradient_norm(self, gradients):
        """Compute the norm of gradients for monitoring."""
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += tf.reduce_sum(tf.square(grad))
        return tf.sqrt(total_norm)
    
    def clip_gradients(self, gradients):
        """Clip gradients for quantum parameter stability."""
        clipped_gradients = []
        for grad in gradients:
            if grad is not None:
                clipped_grad = tf.clip_by_norm(grad, self.gradient_clip_norm)
                clipped_gradients.append(clipped_grad)
            else:
                clipped_gradients.append(grad)
        return clipped_gradients
    
    def _quantum_safe_generate(self, z):
        """Generate samples using quantum-safe execution context with warning suppression."""
        # Check if generator has quantum components
        generator_type = type(self.generator).__name__
        if 'Quantum' in generator_type:
            # Use quantum execution context with warning suppression for quantum generators
            with suppress_complex_warnings():
                # REMOVED QuantumExecutionContext to fix gradient flow
                return self.generator.generate(z)
        else:
            # Use normal execution for classical generators
            return self.generator.generate(z)
    
    def train_step_eager(self, real_samples, use_wasserstein=False):
        """Training step optimized for quantum operations (eager execution).
        
        This version removes @tf.function to avoid AutoGraph issues with quantum circuits.
        
        Args:
            real_samples: Batch of real training data
            use_wasserstein: Whether to use Wasserstein loss
            
        Returns:
            Dictionary of losses and metrics
        """
        batch_size = tf.shape(real_samples)[0]
        
        # --- Discriminator Training ---
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as d_tape:
            # Use quantum-safe generation
            fake_samples = self._quantum_safe_generate(z)
            
            if use_wasserstein:
                d_loss, _, gp = self.wasserstein_loss(real_samples, fake_samples)
            else:
                d_loss, _ = self.traditional_gan_loss(real_samples, fake_samples)
                gp = tf.constant(0.0)
        
        # Discriminator gradients with clipping
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_grad_norm = self.compute_gradient_norm(d_gradients)
        d_gradients = self.clip_gradients(d_gradients)
        
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        # --- Generator Training ---
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as g_tape:
            # Use quantum-safe generation
            fake_samples = self._quantum_safe_generate(z)
            
            if use_wasserstein:
                _, g_loss, _ = self.wasserstein_loss(real_samples, fake_samples)
            else:
                _, g_loss = self.traditional_gan_loss(real_samples, fake_samples)
        
        # Generator gradients with clipping
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        g_grad_norm = self.compute_gradient_norm(g_gradients)
        g_gradients = self.clip_gradients(g_gradients)
        
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        # Stability metric (ratio of gradient norms)
        stability_metric = g_grad_norm / (d_grad_norm + 1e-8)
        
        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'gradient_penalty': gp,
            'g_grad_norm': g_grad_norm,
            'd_grad_norm': d_grad_norm,
            'stability_metric': stability_metric
        }
    
    # Alias for backward compatibility
    def train_step(self, real_samples, use_wasserstein=False):
        """Training step with automatic quantum/classical handling."""
        return self.train_step_eager(real_samples, use_wasserstein)
    
    def train(self, data, epochs=100, batch_size=32, use_wasserstein=False, 
              verbose=True, save_interval=10):
        """Enhanced training loop with quantum-aware monitoring.
        
        Args:
            data (tensor): Training dataset (real samples).
            epochs (int): Number of training iterations.
            batch_size (int): Samples per training batch.
            use_wasserstein (bool): Whether to use Wasserstein loss.
            verbose (bool): Whether to print training progress.
            save_interval (int): How often to log detailed metrics.
        """
        print(f"Starting QGAN training for {epochs} epochs...")
        print(f"Using {'Wasserstein' if use_wasserstein else 'Traditional'} GAN loss")
        print(f"Gradient clipping norm: {self.gradient_clip_norm}")
        
        # Create data loader
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        
        for epoch in range(epochs):
            epoch_metrics = {
                'd_loss': [],
                'g_loss': [],
                'g_grad_norm': [],
                'd_grad_norm': [],
                'stability_metric': []
            }
            
            for batch_idx, real_samples in enumerate(dataset):
                # Training step
                metrics = self.train_step(real_samples, use_wasserstein)
                
                # Collect metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
            
            # Record epoch averages
            for key in epoch_metrics:
                if epoch_metrics[key]:
                    avg_value = tf.reduce_mean(epoch_metrics[key])
                    self.training_history[key].append(float(avg_value))
            
            # Verbose logging
            if verbose and epoch % save_interval == 0:
                avg_d_loss = self.training_history['d_loss'][-1]
                avg_g_loss = self.training_history['g_loss'][-1]
                avg_stability = self.training_history['stability_metric'][-1]
                
                print(f"Epoch {epoch:04d}: D_loss={avg_d_loss:.4f}, "
                      f"G_loss={avg_g_loss:.4f}, Stability={avg_stability:.4f}")
                
                # Check for training instability
                if avg_stability > 10.0:
                    print("Warning: Training may be unstable (high gradient ratio)")
                elif avg_stability < 0.1:
                    print("Warning: Generator gradients very small (potential mode collapse)")
        
        print("Training completed!")
        
        # Generate final samples for evaluation
        z_test = tf.random.normal([5, self.latent_dim])
        final_samples = self.generator.generate(z_test)
        
        # Plot final results if data is 2D
        if data.shape[1] <= 4:
            plot_results(data[:100], final_samples, epoch=epochs-1)
        
        return self.training_history

if __name__ == "__main__":
    # Example usage can be found in the tests directory
    print("QGAN training framework loaded successfully.")
    print("See tests/integration/ for usage examples.")
