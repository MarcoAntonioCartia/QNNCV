import tensorflow as tf
import numpy as np
from utils import load_qm9_data, plot_results

class QGAN:
    """Enhanced QGAN with quantum-aware training stability features.
    
    Architecture:
    - Generator (G): Creates fake samples from noise (quantum/classical).
    - Discriminator (D): Classifies real vs. fake samples (quantum/classical).
    
    Quantum-specific enhancements:
    - Gradient clipping for quantum parameter stability
    - Learning rate scheduling for quantum optimization
    - Quantum-aware loss functions
    - Training monitoring and stability checks
    """
    
    def __init__(self, generator, discriminator, latent_dim=10, 
                 generator_lr=0.0001, discriminator_lr=0.0001,
                 beta1=0.5, beta2=0.999, gradient_clip_norm=1.0):
        """Initialize QGAN with enhanced quantum-aware training.
        
        Args:
            generator: Instance of a generator (e.g., QuantumContinuousGenerator).
            discriminator: Instance of a discriminator.
            latent_dim (int): Dimension of latent noise vector.
            generator_lr (float): Generator learning rate (lower for quantum stability).
            discriminator_lr (float): Discriminator learning rate.
            beta1 (float): Adam optimizer beta1 parameter.
            beta2 (float): Adam optimizer beta2 parameter.
            gradient_clip_norm (float): Gradient clipping norm for stability.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gradient_clip_norm = gradient_clip_norm
        
        # Enhanced optimizers with quantum-friendly settings
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=generator_lr, beta_1=beta1, beta_2=beta2
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=discriminator_lr, beta_1=beta1, beta_2=beta2
        )
        
        # Learning rate schedulers for quantum parameter optimization
        self.g_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=generator_lr,
            decay_steps=100,
            decay_rate=0.98
        )
        self.d_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=discriminator_lr,
            decay_steps=100,
            decay_rate=0.98
        )
        
        # Training metrics for monitoring
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'g_grad_norm': [],
            'd_grad_norm': [],
            'stability_metric': []
        }
    
    def wasserstein_loss(self, real_samples, fake_samples, lambda_gp=10.0):
        """Wasserstein loss with gradient penalty for improved quantum training stability.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated samples
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            d_loss: Discriminator loss
            g_loss: Generator loss
            gp: Gradient penalty value
        """
        batch_size = tf.shape(real_samples)[0]
        
        # Discriminator predictions
        real_output = self.discriminator.discriminate(real_samples)
        fake_output = self.discriminator.discriminate(fake_samples)
        
        # Wasserstein distance
        wasserstein_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        # Gradient penalty for Lipschitz constraint
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_output = self.discriminator.discriminate(interpolated)
        
        gradients = gp_tape.gradient(interpolated_output, interpolated)
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = lambda_gp * tf.reduce_mean(tf.square(gradient_norm - 1.0))
        
        # Losses
        d_loss = -wasserstein_distance + gradient_penalty
        g_loss = -tf.reduce_mean(fake_output)
        
        return d_loss, g_loss, gradient_penalty
    
    def traditional_gan_loss(self, real_samples, fake_samples, label_smoothing=0.1):
        """Traditional GAN loss with label smoothing for quantum stability.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated samples
            label_smoothing: Amount of label smoothing for stability
            
        Returns:
            d_loss: Discriminator loss
            g_loss: Generator loss
        """
        # Label smoothing for training stability
        real_labels = tf.ones_like(real_samples[:, :1]) * (1.0 - label_smoothing)
        fake_labels = tf.zeros_like(fake_samples[:, :1]) + label_smoothing
        
        real_output = self.discriminator.discriminate(real_samples)
        fake_output = self.discriminator.discriminate(fake_samples)
        
        # Binary crossentropy loss with smoothing
        real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_output)
        fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_output)
        
        d_loss = tf.reduce_mean(real_loss + fake_loss)
        g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
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
    
    @tf.function
    def train_step(self, real_samples, use_wasserstein=False):
        """Enhanced training step with quantum-aware stability features.
        
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
            fake_samples = self.generator.generate(z)
            
            if use_wasserstein:
                d_loss, _, gp = self.wasserstein_loss(real_samples, fake_samples)
            else:
                d_loss, _ = self.traditional_gan_loss(real_samples, fake_samples)
                gp = 0.0
        
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
            fake_samples = self.generator.generate(z)
            
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

# Example Usage (for a classical baseline)
if __name__ == "__main__":
    from NN.classical_generator import ClassicalGenerator
    from NN.classical_discriminator import ClassicalDiscriminator
    
    # Load preprocessed molecular descriptors
    data = load_qm9_data()
    
    # Initialize QGAN with classical components
    qgan = QGAN(
        generator=ClassicalGenerator(latent_dim=10, output_dim=30),
        discriminator=ClassicalDiscriminator(input_dim=30)
    )
    
    # Train and benchmark
    qgan.train(data, epochs=50)
