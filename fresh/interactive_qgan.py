"""
Interactive CV Quantum GAN Development Script
==============================================

This script is designed for interactive development and experimentation.
Run sections individually in Jupyter/Colab or as a complete script.

Usage in Jupyter/Colab:
    # Run each cell marked with # %% 

Usage as script:
    python interactive_qgan.py
"""

# %%
# =============================================================================
# SETUP AND IMPORTS
# =============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TensorFlow version:", tf.__version__)

# Check if Strawberry Fields is available
try:
    import strawberryfields as sf
    from strawberryfields import ops
    print("Strawberry Fields version:", sf.__version__)
    SF_AVAILABLE = True
except ImportError:
    print("Strawberry Fields not available - install with: pip install strawberryfields")
    SF_AVAILABLE = False

# %%
# =============================================================================
# QUICK GRADIENT FLOW CHECK
# =============================================================================

if SF_AVAILABLE:
    print("\n--- Quick Gradient Flow Check ---")
    
    # Minimal circuit
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 5})
    prog = sf.Program(1)
    
    weights = tf.Variable(tf.random.normal([2], stddev=0.1))
    sf_params = [prog.params("r"), prog.params("phi")]
    
    with prog.context as q:
        ops.Sgate(sf_params[0]) | q[0]
        ops.Rgate(sf_params[1]) | q[0]
    
    with tf.GradientTape() as tape:
        mapping = {"r": weights[0], "phi": weights[1]}
        state = eng.run(prog, args=mapping).state
        x_quad = state.quad_expectation(0, 0)
        loss = tf.square(x_quad)
    
    grads = tape.gradient(loss, weights)
    
    if grads is not None:
        print(f"✓ Gradient flow working! Gradient: {grads.numpy()}")
    else:
        print("✗ No gradients - check SF installation")
    
    eng.reset()

# %%
# =============================================================================
# DATA GENERATORS
# =============================================================================

def make_bimodal_data(n_samples=1000, centers=((-2, -2), (2, 2)), std=0.3):
    """Generate bimodal 2D data."""
    n_half = n_samples // 2
    
    c1 = np.array(centers[0])
    c2 = np.array(centers[1])
    
    data1 = np.random.randn(n_half, 2) * std + c1
    data2 = np.random.randn(n_samples - n_half, 2) * std + c2
    
    data = np.vstack([data1, data2])
    np.random.shuffle(data)
    
    return tf.cast(data, tf.float32)

def make_ring_data(n_samples=1000, radius=2.0, noise=0.1):
    """Generate ring-shaped 2D data."""
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = radius + np.random.randn(n_samples) * noise
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    data = np.column_stack([x, y])
    return tf.cast(data, tf.float32)

# %%
# Visualize data
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

bimodal_data = make_bimodal_data(500)
axes[0].scatter(bimodal_data[:, 0], bimodal_data[:, 1], alpha=0.5, s=10)
axes[0].set_title("Bimodal Data")
axes[0].set_aspect('equal')
axes[0].grid(True)

ring_data = make_ring_data(500)
axes[1].scatter(ring_data[:, 0], ring_data[:, 1], alpha=0.5, s=10)
axes[1].set_title("Ring Data")
axes[1].set_aspect('equal')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# SIMPLE QUANTUM GENERATOR (For Understanding)
# =============================================================================

if SF_AVAILABLE:
    
    class SimpleQuantumGenerator:
        """
        Simplified quantum generator for learning/debugging.
        
        This is a minimal implementation to understand the concepts.
        For production, use pure_sf_qgan.py.
        """
        
        def __init__(self, n_modes=2, n_layers=2, output_dim=2, cutoff=6):
            self.n_modes = n_modes
            self.n_layers = n_layers
            self.output_dim = output_dim
            
            # SF components
            self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
            self.prog = sf.Program(n_modes)
            
            # Count parameters per layer
            # For each mode: squeezing + rotation = 2 params
            # Between modes: beamsplitter = 1 param per pair
            params_per_layer = n_modes * 2 + (n_modes - 1)
            total_params = params_per_layer * n_layers
            
            # Unified weight matrix
            self.weights = tf.Variable(
                tf.random.normal([total_params], stddev=0.1),
                name="generator_weights"
            )
            
            # Create symbolic parameters
            self.sf_params = [self.prog.params(f"p{i}") for i in range(total_params)]
            
            # Build program
            idx = 0
            with self.prog.context as q:
                for layer in range(n_layers):
                    # Squeezing
                    for m in range(n_modes):
                        ops.Sgate(self.sf_params[idx]) | q[m]
                        idx += 1
                    
                    # Beamsplitters
                    for m in range(n_modes - 1):
                        ops.BSgate(self.sf_params[idx], 0) | (q[m], q[m+1])
                        idx += 1
                    
                    # Rotations
                    for m in range(n_modes):
                        ops.Rgate(self.sf_params[idx]) | q[m]
                        idx += 1
            
            # Decoder
            self.decoder = tf.Variable(
                tf.random.normal([n_modes, output_dim], stddev=0.5),
                name="decoder"
            )
            
            print(f"Generator: {total_params} quantum params + {n_modes * output_dim} decoder params")
        
        def generate(self, z):
            """Generate samples. z is just used for batch size currently."""
            batch_size = tf.shape(z)[0]
            
            # Execute circuit once per batch item
            outputs = []
            for i in range(batch_size):
                if self.eng.run_progs:
                    self.eng.reset()
                
                # Map weights to SF params
                mapping = {
                    self.sf_params[j].name: self.weights[j] 
                    for j in range(len(self.sf_params))
                }
                
                state = self.eng.run(self.prog, args=mapping).state
                
                # Extract measurements
                measurements = [state.quad_expectation(m, 0) for m in range(self.n_modes)]
                outputs.append(tf.stack(measurements))
            
            # Stack and decode
            batch_quantum = tf.stack(outputs, axis=0)  # [batch, n_modes]
            output = tf.matmul(batch_quantum, self.decoder)  # [batch, output_dim]
            
            return output
        
        @property
        def trainable_variables(self):
            return [self.weights, self.decoder]
    
    # Test it
    print("\n--- Testing Simple Generator ---")
    gen = SimpleQuantumGenerator(n_modes=2, n_layers=1, output_dim=2)
    
    z = tf.random.normal([4, 2])  # Batch of 4
    
    with tf.GradientTape() as tape:
        output = gen.generate(z)
        loss = tf.reduce_mean(tf.square(output))
    
    grads = tape.gradient(loss, gen.trainable_variables)
    
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.numpy()}")
    print(f"Gradients present: {[g is not None for g in grads]}")

# %%
# =============================================================================
# SIMPLE TRAINING LOOP
# =============================================================================

if SF_AVAILABLE:
    print("\n--- Simple Training Test ---")
    
    gen = SimpleQuantumGenerator(n_modes=2, n_layers=2, output_dim=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Target: generate samples with mean (1, 1)
    target = tf.constant([[1.0, 1.0]])
    
    losses = []
    
    for step in range(20):
        with tf.GradientTape() as tape:
            z = tf.random.normal([4, 2])
            output = gen.generate(z)
            
            # Loss: distance from target
            loss = tf.reduce_mean(tf.square(output - target))
        
        grads = tape.gradient(loss, gen.trainable_variables)
        
        # Filter None gradients
        valid_grads_vars = [(g, v) for g, v in zip(grads, gen.trainable_variables) if g is not None]
        
        if valid_grads_vars:
            optimizer.apply_gradients(valid_grads_vars)
            losses.append(float(loss))
            
            if (step + 1) % 5 == 0:
                print(f"Step {step+1}: Loss = {loss.numpy():.4f}")
                print(f"  Output mean: {tf.reduce_mean(output, axis=0).numpy()}")
    
    # Plot loss
    if losses:
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Simple Training Progress")
        plt.grid(True)
        plt.show()

# %%
# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_generated_vs_real(generator, real_data, n_samples=200):
    """
    Compare generated and real samples visually.
    """
    z = tf.random.normal([n_samples, 2])
    generated = generator.generate(z).numpy()
    real = real_data[:n_samples].numpy() if len(real_data) > n_samples else real_data.numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Real data
    axes[0].scatter(real[:, 0], real[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_title("Real Data")
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].grid(True)
    axes[0].set_aspect('equal')
    
    # Generated data
    axes[1].scatter(generated[:, 0], generated[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_title("Generated Data")
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    
    # Overlay
    axes[2].scatter(real[:, 0], real[:, 1], alpha=0.3, s=10, c='blue', label='Real')
    axes[2].scatter(generated[:, 0], generated[:, 1], alpha=0.3, s=10, c='red', label='Generated')
    axes[2].set_title("Overlay")
    axes[2].set_xlim(-5, 5)
    axes[2].set_ylim(-5, 5)
    axes[2].grid(True)
    axes[2].legend()
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def plot_training_progress(history):
    """
    Plot training metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generator loss
    if 'g_loss' in history:
        axes[0, 0].plot(history['g_loss'])
        axes[0, 0].set_title("Generator Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True)
    
    # Discriminator loss
    if 'd_loss' in history:
        axes[0, 1].plot(history['d_loss'])
        axes[0, 1].set_title("Discriminator Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True)
    
    # Gradient flow
    if 'g_gradient_flow' in history:
        axes[1, 0].plot(history['g_gradient_flow'], label='Generator')
        axes[1, 0].plot(history['d_gradient_flow'], label='Discriminator')
        axes[1, 0].set_title("Gradient Flow")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Wasserstein distance (if available)
    if 'w_distance' in history:
        axes[1, 1].plot(history['w_distance'])
        axes[1, 1].set_title("Wasserstein Distance Estimate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# FULL GAN TRAINING (Import from pure_sf_qgan)
# =============================================================================

print("\n" + "="*60)
print("To run the full Quantum GAN, use:")
print("  python pure_sf_qgan.py")
print("Or import:")
print("  from pure_sf_qgan import QuantumGAN, bimodal_data_generator")
print("="*60)

# %%
# =============================================================================
# NOTES FOR THESIS
# =============================================================================

"""
THESIS COMPARISON POINTS
========================

1. PARAMETER EFFICIENCY
   - CV Quantum: ~50-100 parameters for simple tasks
   - Classical MLP: typically 1000s of parameters
   - Measure: performance vs parameter count

2. TRAINING DYNAMICS
   - CV Quantum: slower per step (Fock simulation)
   - Classical: faster per step (GPU optimized)
   - Measure: wall-clock time to convergence

3. EXPRESSIVITY
   - CV Quantum: continuous naturally, quantum correlations
   - Classical: arbitrary functions with enough capacity
   - Measure: distribution matching quality (Wasserstein distance)

4. MODE COVERAGE
   - Both can suffer from mode collapse
   - Measure: sample diversity, mode coverage percentage

5. GRADIENT STABILITY
   - CV Quantum: can have gradient issues (solved in this implementation)
   - Classical: well-understood optimization
   - Measure: gradient norm over training, loss variance

6. QUANTUM ADVANTAGE
   - Theoretical: potential exponential advantage
   - Practical: currently limited by simulation costs
   - Future: real quantum hardware could change this

RECOMMENDED EXPERIMENTS
=======================

1. Train on same bimodal data with CV, DV, Classical
2. Compare learning curves (epochs to convergence)
3. Compare final distribution matching quality
4. Compare parameter efficiency
5. Analyze failure modes (mode collapse, etc.)
"""

print("See NOTES FOR THESIS section for comparison points")
