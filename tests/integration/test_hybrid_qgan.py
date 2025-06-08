import tensorflow as tf
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from training.qgan_trainer import QGAN
from models.generators.quantum_continuous_generator import QuantumContinuousGenerator
from models.discriminators.classical_discriminator import ClassicalDiscriminator

def test_cv_qgan():
    """
    Test continuous variable quantum generator with classical discriminator.
    
    This test validates the integration between quantum continuous variable
    generators and classical discriminators in a hybrid QGAN architecture.
    """
    # Initialize hybrid components
    generator = QuantumContinuousGenerator(n_qumodes=30)
    discriminator = ClassicalDiscriminator(input_dim=30)
    qgan = QGAN(generator, discriminator)

    # Test forward pass through generator
    z = tf.random.normal([5, 10])  # Batch of 5 latent vectors
    fake_samples = generator.generate(z)
    assert fake_samples.shape == (5, 30), "Generator output shape mismatch!"

    # Test discriminator classification
    probs = discriminator.discriminate(fake_samples)
    assert tf.reduce_all(probs >= 0.0) and tf.reduce_all(probs <= 1.0), "Invalid probabilities!"

def test_training_step():
    """
    Test single training step execution.
    
    This test validates that the training loop can execute one complete
    iteration without errors, including gradient computation and parameter updates.
    """
    data = tf.random.normal([100, 30])  # Synthetic training data
    qgan = QGAN(QuantumContinuousGenerator(), ClassicalDiscriminator())
    qgan.train(data, epochs=1, batch_size=10)
    print("Training step completed without errors!")

if __name__ == "__main__":
    test_cv_qgan()
    test_training_step()
