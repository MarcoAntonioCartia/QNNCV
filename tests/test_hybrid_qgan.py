import tensorflow as tf  
from main_qgan import QGAN  
from NN.quantum_continuous_generator import QuantumContinuousGenerator  
from NN.classical_discriminator import ClassicalDiscriminator  

def test_cv_qgan():  
    """Test CV Quantum Generator + Classical Discriminator."""  
    # Initialize components  
    generator = QuantumContinuousGenerator(n_qumodes=30)  
    discriminator = ClassicalDiscriminator(input_dim=30)  
    qgan = QGAN(generator, discriminator)  

    # Test forward pass  
    z = tf.random.normal([5, 10])  # Batch of 5 noise vectors  
    fake_samples = generator.generate(z)  
    assert fake_samples.shape == (5, 30), "Generator output shape mismatch!"  

    # Test discriminator  
    probs = discriminator.discriminate(fake_samples)  
    assert tf.reduce_all(probs >= 0.0) and tf.reduce_all(probs <= 1.0), "Invalid probabilities!"  

def test_training_step():  
    """Test single training step."""  
    data = tf.random.normal([100, 30])  # Mock QM9 data  
    qgan = QGAN(QuantumContinuousGenerator(), ClassicalDiscriminator())  
    qgan.train(data, epochs=1, batch_size=10)  
    print("Training step completed without errors!")  

if __name__ == "__main__":  
    test_cv_qgan()  
    test_training_step()  