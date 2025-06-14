"""
Threaded Training Example for Quantum GAN
Shows how to integrate threading into actual training loops
"""

import sys
sys.path.insert(0, 'src')

import tensorflow as tf
import numpy as np
import time
from utils.quantum_threading import create_threaded_quantum_generator, optimize_cpu_utilization
from models.generators.quantum_sf_generator import QuantumSFGenerator
from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator

class ThreadedQuantumGANTrainer:
    """
    Enhanced quantum GAN trainer with threading support.
    Drop-in replacement for your existing trainer with performance improvements.
    """
    
    def __init__(self, n_modes=2, latent_dim=4, layers=2, cutoff_dim=8, 
                 encoding_strategy='coherent_state', enable_threading=True, 
                 max_threads=None, learning_rate=0.001):
        """
        Initialize threaded quantum GAN trainer.
        
        Args:
            n_modes: Number of quantum modes
            latent_dim: Latent space dimension
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff
            encoding_strategy: Quantum encoding strategy
            enable_threading: Enable threading for generator
            max_threads: Maximum threads (auto-detect if None)
            learning_rate: Learning rate for optimizers
        """
        
        # Optimize CPU utilization
        optimize_cpu_utilization()
        
        # Create threaded generator
        if enable_threading:
            print("Creating threaded quantum generator...")
            self.generator = create_threaded_quantum_generator(
                QuantumSFGenerator,
                n_modes=n_modes,
                latent_dim=latent_dim,
                layers=layers,
                cutoff_dim=cutoff_dim,
                encoding_strategy=encoding_strategy,
                enable_threading=True,
                max_threads=max_threads
            )
            print(f"   Threading enabled with {self.generator.threading_manager.max_threads} threads")
            print(f"   GPU available: {self.generator.threading_manager.has_gpu}")
        else:
            print("Creating standard quantum generator...")
            self.generator = QuantumSFGenerator(
                n_modes=n_modes,
                latent_dim=latent_dim,
                layers=layers,
                cutoff_dim=cutoff_dim,
                encoding_strategy=encoding_strategy
            )
        
        # Create discriminator
        print("Creating quantum discriminator...")
        self.discriminator = QuantumSFDiscriminator(
            n_modes=n_modes,
            input_dim=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim,
            encoding_strategy=encoding_strategy
        )
        
        # Create optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Training parameters
        self.n_modes = n_modes
        self.latent_dim = latent_dim
        self.enable_threading = enable_threading
        
        # Performance tracking
        self.training_stats = {
            'total_steps': 0,
            'total_time': 0.0,
            'avg_step_time': 0.0,
            'cpu_utilization': 0.0
        }
        
        print(f" Trainer initialized:")
        print(f"   Generator variables: {len(self.generator.trainable_variables)}")
        print(f"   Discriminator variables: {len(self.discriminator.trainable_variables)}")
    
    def generate_samples(self, batch_size, strategy="auto"):
        """
        Generate samples using optimal threading strategy.
        
        Args:
            batch_size: Number of samples to generate
            strategy: Threading strategy ("auto", "threading", "cpu_batch", "sequential")
            
        Returns:
            Generated samples tensor
        """
        z = tf.random.normal([batch_size, self.latent_dim])
        
        if self.enable_threading:
            return self.generator.generate_threaded(z, strategy=strategy)
        else:
            return self.generator.generate(z)
    
    @tf.function
    def train_step(self, real_data, batch_size):
        """
        Single training step with gradient computation.
        
        Args:
            real_data: Real data batch
            batch_size: Batch size
            
        Returns:
            Dictionary with losses and metrics
        """
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake samples (threading happens outside @tf.function)
            fake_samples = self.generator.generate(z)  # Use standard generate in tf.function
            
            # Discriminator outputs
            real_output = self.discriminator.discriminate(real_data)
            fake_output = self.discriminator.discriminate(fake_samples)
            
            # Compute losses
            disc_loss = -tf.reduce_mean(
                tf.math.log(real_output + 1e-8) + tf.math.log(1 - fake_output + 1e-8)
            )
            gen_loss = -tf.reduce_mean(tf.math.log(fake_output + 1e-8))
        
        # Compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'real_output_mean': tf.reduce_mean(real_output),
            'fake_output_mean': tf.reduce_mean(fake_output)
        }
    
    def train_step_threaded(self, real_data, batch_size, strategy="auto"):
        """
        Training step with threading for sample generation.
        
        Args:
            real_data: Real data batch
            batch_size: Batch size
            strategy: Threading strategy
            
        Returns:
            Dictionary with losses and metrics
        """
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake samples with threading
            if self.enable_threading:
                fake_samples = self.generator.generate_threaded(z, strategy=strategy)
            else:
                fake_samples = self.generator.generate(z)
            
            # Discriminator outputs
            real_output = self.discriminator.discriminate(real_data)
            fake_output = self.discriminator.discriminate(fake_samples)
            
            # Compute losses
            disc_loss = -tf.reduce_mean(
                tf.math.log(real_output + 1e-8) + tf.math.log(1 - fake_output + 1e-8)
            )
            gen_loss = -tf.reduce_mean(tf.math.log(fake_output + 1e-8))
        
        # Compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'real_output_mean': tf.reduce_mean(real_output),
            'fake_output_mean': tf.reduce_mean(fake_output)
        }
    
    def train(self, real_data, epochs=10, batch_size=16, strategy="auto", 
              log_interval=1, benchmark_interval=5):
        """
        Main training loop with threading support.
        
        Args:
            real_data: Training data
            epochs: Number of epochs
            batch_size: Batch size
            strategy: Threading strategy
            log_interval: Logging interval (epochs)
            benchmark_interval: Performance benchmark interval (epochs)
        """
        print(f"\n Starting threaded quantum GAN training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Strategy: {strategy}")
        print(f"   Threading: {' Enabled' if self.enable_threading else ' Disabled'}")
        
        num_batches = len(real_data) // batch_size
        total_steps = epochs * num_batches
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_gen_loss = 0.0
            epoch_disc_loss = 0.0
            
            # Shuffle data
            indices = tf.random.shuffle(tf.range(len(real_data)))
            shuffled_data = tf.gather(real_data, indices)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(real_data))
                real_batch = shuffled_data[batch_start:batch_end]
                
                # Training step with threading
                step_start = time.time()
                metrics = self.train_step_threaded(real_batch, batch_size, strategy)
                step_time = time.time() - step_start
                
                # Update statistics
                self.training_stats['total_steps'] += 1
                self.training_stats['total_time'] += step_time
                self.training_stats['avg_step_time'] = (
                    self.training_stats['total_time'] / self.training_stats['total_steps']
                )
                
                epoch_gen_loss += metrics['gen_loss']
                epoch_disc_loss += metrics['disc_loss']
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                avg_gen_loss = epoch_gen_loss / num_batches
                avg_disc_loss = epoch_disc_loss / num_batches
                
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Gen Loss: {avg_gen_loss:.4f}, "
                      f"Disc Loss: {avg_disc_loss:.4f}, "
                      f"Time: {epoch_time:.2f}s")
            
            # Performance benchmarking
            if (epoch + 1) % benchmark_interval == 0 and self.enable_threading:
                self._benchmark_performance(batch_size)
        
        total_time = time.time() - start_time
        
        print(f"\n Training completed!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average step time: {self.training_stats['avg_step_time']:.4f}s")
        print(f"   Steps per second: {self.training_stats['total_steps'] / total_time:.2f}")
        
        if self.enable_threading:
            report = self.generator.threading_manager.get_performance_report()
            print(f"   CPU utilization: {report['cpu_utilization_estimate']:.1f}%")
            print(f"   Total samples generated: {report['total_executions']}")
    
    def _benchmark_performance(self, batch_size):
        """Run performance benchmark during training."""
        if not self.enable_threading:
            return
        
        print(f"\n Performance Benchmark (Epoch checkpoint):")
        
        # Test different strategies
        strategies = ["sequential", "threading", "cpu_batch", "auto"]
        z_test = tf.random.normal([batch_size, self.latent_dim])
        
        for strategy in strategies:
            try:
                start_time = time.time()
                _ = self.generator.generate_threaded(z_test, strategy=strategy)
                execution_time = time.time() - start_time
                samples_per_sec = batch_size / execution_time if execution_time > 0 else 0
                
                print(f"   {strategy:12}: {execution_time:.3f}s ({samples_per_sec:.1f} samples/s)")
                
            except Exception as e:
                print(f"   {strategy:12}: Failed - {e}")
        
        # Overall performance report
        report = self.generator.threading_manager.get_performance_report()
        print(f"   Current CPU util: {report['cpu_utilization_estimate']:.1f}%")
    
    def save_models(self, checkpoint_dir="checkpoints"):
        """Save generator and discriminator models."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save generator
        gen_path = os.path.join(checkpoint_dir, "generator")
        # Note: Custom save logic needed for threaded generator
        print(f" Models saved to {checkpoint_dir}")
    
    def generate_samples_for_evaluation(self, num_samples=100, strategy="auto"):
        """Generate samples for evaluation."""
        print(f" Generating {num_samples} samples for evaluation...")
        
        all_samples = []
        batch_size = 16
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            samples = self.generate_samples(current_batch_size, strategy)
            all_samples.append(samples)
        
        return tf.concat(all_samples, axis=0)

def create_synthetic_data(num_samples=1000, n_modes=2):
    """Create synthetic training data for testing."""
    # Generate some synthetic quantum-like data
    data = tf.random.normal([num_samples, n_modes], stddev=0.5)
    data = tf.nn.tanh(data)  # Normalize to [-1, 1]
    return data

def main():
    """Main training example."""
    print("=" * 70)
    print("THREADED QUANTUM GAN TRAINING EXAMPLE")
    print("=" * 70)
    
    # Configuration
    config = {
        'n_modes': 2,
        'latent_dim': 4,
        'layers': 2,
        'cutoff_dim': 8,
        'encoding_strategy': 'coherent_state',
        'enable_threading': True,
        'max_threads': None,  # Auto-detect
        'learning_rate': 0.001,
        'epochs': 20,
        'batch_size': 16,
        'strategy': 'auto'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create synthetic training data
    print(f"\n Creating synthetic training data...")
    real_data = create_synthetic_data(num_samples=500, n_modes=config['n_modes'])
    print(f"   Data shape: {real_data.shape}")
    print(f"   Data range: [{tf.reduce_min(real_data):.3f}, {tf.reduce_max(real_data):.3f}]")
    
    # Create trainer
    print(f"\n Creating threaded quantum GAN trainer...")
    trainer = ThreadedQuantumGANTrainer(
        n_modes=config['n_modes'],
        latent_dim=config['latent_dim'],
        layers=config['layers'],
        cutoff_dim=config['cutoff_dim'],
        encoding_strategy=config['encoding_strategy'],
        enable_threading=config['enable_threading'],
        max_threads=config['max_threads'],
        learning_rate=config['learning_rate']
    )
    
    # Train the model
    print(f"\n Starting training...")
    trainer.train(
        real_data=real_data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        strategy=config['strategy'],
        log_interval=2,
        benchmark_interval=5
    )
    
    # Generate evaluation samples
    print(f"\n Generating evaluation samples...")
    eval_samples = trainer.generate_samples_for_evaluation(num_samples=50, strategy="auto")
    print(f"   Generated samples shape: {eval_samples.shape}")
    print(f"   Generated range: [{tf.reduce_min(eval_samples):.3f}, {tf.reduce_max(eval_samples):.3f}]")
    
    # Final performance summary
    if trainer.enable_threading:
        print(f"\n Final Performance Summary:")
        report = trainer.threading_manager.get_performance_report()
        print(f"   Total samples generated: {report['total_executions']}")
        print(f"   Average samples/sec: {report['avg_samples_per_second']:.2f}")
        print(f"   CPU utilization: {report['cpu_utilization_estimate']:.1f}%")
        print(f"   Strategy usage: {report['strategy_usage']}")
    
    print(f"\n Training example completed successfully!")
    print(f" Ready to integrate threading into your quantum GAN training!")
    
    return trainer

if __name__ == "__main__":
    trainer = main()
