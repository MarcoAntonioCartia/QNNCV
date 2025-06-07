import tensorflow as tf

class ClassicalGenerator:
    """Classical Generator (Dense Neural Network).
    
    Architecture:
    - Input: Latent noise vector (z)
    - Output: Fake sample matching QM9 data dimensions
    """
    
    def __init__(self, latent_dim=10, output_dim=30):
        """Initialize layers.
        
        Args:
            latent_dim (int): Dimension of input noise vector.
            output_dim (int): Output dimension (matches molecular descriptor size).
        """
        self.model = tf.keras.Sequential([
            # Hidden layer with ReLU activation
            tf.keras.layers.Dense(64, activation='relu', name="dense1"),
            # Output layer with linear activation (real-valued features)
            tf.keras.layers.Dense(output_dim, activation='linear', name="output")
        ])
        
        # Build the model by calling it once
        dummy_input = tf.zeros((1, latent_dim))
        _ = self.model(dummy_input)

    def generate(self, z):
        """Forward pass: Generate samples from noise.
        
        Args:
            z (tensor): Input noise of shape [batch_size, latent_dim].
        
        Returns:
            tensor: Generated samples of shape [batch_size, output_dim].
        """
        return self.model(z)
    
    @property
    def trainable_variables(self):
        """Return trainable variables for optimization."""
        return self.model.trainable_variables