import tensorflow as tf
from utils import load_qm9_data, plot_results

class QGAN:
    """Master class for training a GAN with interchangeable components.
    
    Architecture:
    - Generator (G): Creates fake samples from noise (quantum/classical).
    - Discriminator (D): Classifies real vs. fake samples (quantum/classical).
    """
    
    def __init__(self, generator, discriminator, latent_dim=10):
        """Initialize QGAN with modular components.
        
        Args:
            generator: Instance of a generator (e.g., QuantumContinuousGenerator).
            discriminator: Instance of a discriminator.
            latent_dim (int): Dimension of latent noise vector.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
        # Optimizers (adjust learning rates based on component type)
        self.g_optimizer = tf.keras.optimizers.Adam(0.001)
        self.d_optimizer = tf.keras.optimizers.Adam(0.001)

    def train(self, data, epochs=100, batch_size=32):
        """Adversarial training loop.
        
        Args:
            data (tensor): Training dataset (real samples).
            epochs (int): Number of training iterations.
            batch_size (int): Samples per training batch.
        """
        for epoch in range(epochs):
            for real_samples in tf.data.Dataset.from_tensor_slices(data).batch(batch_size):
                # --- Discriminator Training ---
                # Generate noise vectors for this batch
                z = tf.random.normal([batch_size, self.latent_dim])
                
                # Track gradients for discriminator parameters
                with tf.GradientTape() as d_tape:
                    # Generate fake samples using the generator
                    fake_samples = self.generator.generate(z)
                    
                    # Discriminator's predictions on real and fake data
                    real_output = self.discriminator.discriminate(real_samples)
                    fake_output = self.discriminator.discriminate(fake_samples)
                    
                    # Discriminator loss (maximize log(D(real)) + log(1 - D(fake)))
                    d_loss = -tf.reduce_mean(
                        tf.math.log(real_output + 1e-8) +  # Avoid log(0)
                        tf.math.log(1 - fake_output + 1e-8)
                    )
                
                # Compute gradients and update discriminator
                d_grads = d_tape.gradient(
                    d_loss, 
                    self.discriminator.trainable_variables  # Quantum/classical params
                )
                self.d_optimizer.apply_gradients(
                    zip(d_grads, self.discriminator.trainable_variables)
                )

                # --- Generator Training ---
                z = tf.random.normal([batch_size, self.latent_dim])
                
                # Track gradients for generator parameters
                with tf.GradientTape() as g_tape:
                    fake_samples = self.generator.generate(z)
                    fake_output = self.discriminator.discriminate(fake_samples)
                    
                    # Generator loss (maximize log(D(fake)))
                    g_loss = -tf.reduce_mean(tf.math.log(fake_output + 1e-8))
                
                # Compute gradients and update generator
                g_grads = g_tape.gradient(
                    g_loss, 
                    self.generator.trainable_variables  # Quantum/classical params
                )
                self.g_optimizer.apply_gradients(
                    zip(g_grads, self.generator.trainable_variables)
                )

            # Logging (customize based on framework)
            print(f"Epoch {epoch}: D_loss={d_loss.numpy():.4f}, G_loss={g_loss.numpy():.4f}")

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