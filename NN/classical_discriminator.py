import tensorflow as tf

class ClassicalDiscriminator:
    """Classical Discriminator (Binary Classifier).
    
    Architecture:
    - Input: 30-D molecular descriptor (real or fake)
    - Output: Probability of being real (0-1)
    """
    
    def __init__(self, input_dim=30):
        """Initialize layers.
        
        Args:
            input_dim (int): Dimension of input data.
        """
        super().__init__()
        self.model = tf.keras.Sequential([
            # Hidden layer with ReLU
            tf.keras.layers.Dense(64, activation='relu', name="dense1"),
            # Output layer with sigmoid (probability)
            tf.keras.layers.Dense(1, activation='sigmoid', name="output")
        ])

    def discriminate(self, x):
        """Forward pass: Classify samples as real/fake.
        
        Args:
            x (tensor): Input data of shape [batch_size, input_dim].
        
        Returns:
            tensor: Probability scores of shape [batch_size, 1].
        """
        return self.model(x)