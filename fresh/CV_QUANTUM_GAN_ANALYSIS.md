# CV Quantum GAN: Understanding Gradient Flow in Strawberry Fields

## Executive Summary

This document explains the **critical differences** between implementations that achieve gradient flow and those that don't in Strawberry Fields CV quantum computing. Understanding these concepts is essential for your thesis comparing CV Quantum, DV Quantum, and Classical GANs.

---

## The Core Problem: Gradient Disconnection

In your original implementation and many attempts, gradients were getting **disconnected** from the quantum circuit. This manifests as:
- `None` gradients during backpropagation
- NaN losses
- Parameters that don't update
- Mode collapse (all outputs identical)

---

## Understanding Strawberry Fields Architecture

### The SF Program-Engine Model

Strawberry Fields uses a **symbolic programming model**:

```
1. Build Program (symbolic operations with placeholder parameters)
          ↓
2. Create Parameter Mapping (placeholder → TensorFlow values)
          ↓  
3. Execute Engine (runs circuit with actual values)
          ↓
4. Extract State (returns TensorFlow-compatible state)
          ↓
5. Compute Loss (from state measurements)
          ↓
6. Backpropagate (gradients flow back through the mapping)
```

The key insight is that **gradients flow through the parameter mapping**, not through the program definition.

---

## The SF Tutorial Pattern (100% Gradient Flow)

### What Works ✓

```python
# 1. UNIFIED WEIGHT MATRIX (single tf.Variable)
weights = tf.Variable(tf.concat([int1, sq, int2, dr, dp, k], axis=1))

# 2. SYMBOLIC PARAMETERS (created once)
sf_params = np.array([prog.params(*i) for i in param_indices])

# 3. DIRECT MAPPING (preserves gradient path)
mapping = {
    p.name: w 
    for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))
}

# 4. EXECUTE WITH MAPPING
state = eng.run(prog, args=mapping).state

# 5. EXTRACT MEASUREMENTS (NO tf.constant()!)
x_quad = state.quad_expectation(mode, 0)  # Keep as TF tensor!
measurements.append(x_quad)  # NOT tf.constant(x_quad)
```

### Why This Works

1. **Single tf.Variable**: TensorFlow can track gradients through one variable
2. **Direct indexing**: `tf.reshape(weights, [-1])` maintains gradient connection
3. **No tf.constant()**: Keeps measurements in the computation graph

---

## What Breaks Gradients ✗

### Problem 1: Individual tf.Variables

```python
# ❌ BREAKS GRADIENTS
self.params = {}
for name in param_names:
    self.params[name] = tf.Variable(tf.random.normal([1]))
```

**Why it fails**: SF's parameter mapping system was designed for unified arrays. Individual variables create complex mapping that can confuse the autodiff.

### Problem 2: tf.constant() in Measurements

```python
# ❌ BREAKS GRADIENTS  
x_quad = state.quad_expectation(mode, 0)
measurements.append(tf.constant(x_quad.numpy()))  # KILLS gradient path!
```

**Why it fails**: `tf.constant()` creates a new tensor disconnected from the computation graph.

### Problem 3: Complex Parameter Transformations

```python
# ❌ CAN BREAK GRADIENTS
modified_param = some_complex_function(self.params[name])
mapping[sf_param.name] = modified_param
```

**Why it fails**: Complex transformations may not have well-defined gradients in the SF context.

### Problem 4: Batch Averaging Before Measurement

```python
# ❌ CAN CAUSE MODE COLLAPSE
batch_state = process_all_samples_together()
mean_measurement = tf.reduce_mean(measurements)  # Destroys diversity!
```

**Why it fails**: Averaging eliminates the quantum diversity you're trying to preserve.

---

## CV Quantum Layer Structure

The SF tutorial defines a complete layer as:

```
Layer = Φ ∘ D ∘ U₂ ∘ S ∘ U₁
```

Where:
- **U₁, U₂**: Interferometers (beamsplitters + phase rotations)
- **S**: Squeezing gates (create quantum correlations)
- **D**: Displacement gates (encode classical information)
- **Φ**: Non-Gaussian gates (Kerr gate provides nonlinearity)

### Parameters per Layer

For N modes:
```
M = N(N-1) + max(1, N-1)  # Interferometer params

Per layer parameters:
- int1: M (first interferometer)
- s: N (squeezing magnitudes)
- int2: M (second interferometer)  
- dr: N (displacement magnitudes)
- dp: N (displacement phases)
- k: N (Kerr magnitudes)

Total per layer = 2M + 4N
```

### Example: 3 modes, 2 layers

```
M = 3*2 + max(1,2) = 8
Per layer = 2*8 + 4*3 = 28
Total = 28 * 2 = 56 parameters
```

---

## The GAN Architecture

### Generator

```
Latent z [batch, latent_dim]
          ↓
    Quantum Circuit
          ↓
X-quadrature measurements [batch, n_modes]
          ↓
    Linear Decoder
          ↓
Generated samples [batch, output_dim]
```

The quantum circuit provides:
- **Nonlinearity** (via Kerr gates)
- **Entanglement** (via beamsplitters)
- **Quantum correlations** (via squeezing)

### Discriminator

```
Input x [batch, input_dim]
          ↓
    Linear Encoder
          ↓
Mode displacements [batch, n_modes]
          ↓
    Quantum Circuit
          ↓
X-quadrature measurements [batch, n_modes]
          ↓
    Linear Classifier
          ↓
Score [batch, 1]
```

---

## Comparing CV vs DV vs Classical

### Continuous Variable (CV) Quantum

**Advantages:**
- Natural fit for continuous data
- Gaussian operations are efficient to simulate
- Non-Gaussian gates provide universal quantum computing

**Challenges:**
- Infinite-dimensional Hilbert space (requires truncation)
- Fock cutoff limits expressivity
- Slower simulation than discrete systems

### Discrete Variable (DV) Quantum

**Advantages:**
- Finite Hilbert space (exact simulation possible)
- Many established frameworks (Qiskit, Cirq, PennyLane)
- Closer to actual quantum hardware

**Challenges:**
- Requires encoding continuous data into discrete qubits
- Measurement gives discrete outcomes

### Classical Neural Networks

**Advantages:**
- Fast, GPU-accelerated
- Well-understood optimization
- Large capacity with many parameters

**Challenges:**
- No quantum speedup
- May miss quantum correlations in data

---

## Key Metrics for Your Thesis

### 1. Gradient Flow

```python
# Test gradient flow
with tf.GradientTape() as tape:
    output = model(input)
    loss = some_loss(output)
    
gradients = tape.gradient(loss, model.trainable_variables)
flow = sum(1 for g in gradients if g is not None) / len(gradients)
```

### 2. Sample Diversity

```python
# Measure variance across generated samples
samples = generator.generate(latent_noise)
variance = tf.math.reduce_variance(samples, axis=0)
diversity = tf.reduce_mean(variance)
```

### 3. Distribution Matching

```python
# Wasserstein distance approximation
from scipy.stats import wasserstein_distance
w_dist = wasserstein_distance(real_samples.flatten(), 
                                generated_samples.flatten())
```

### 4. Training Stability

```python
# Monitor loss oscillations
loss_std = np.std(history['g_loss'][-20:])  # Last 20 epochs
is_stable = loss_std < threshold
```

---

## Recommended Experiments for Thesis

### Experiment 1: Gradient Flow Comparison

Compare gradient flow across frameworks:
- SF Tutorial Pattern (CV)
- Your DV implementation
- Classical MLP

### Experiment 2: Learning Bimodal Distributions

Test ability to learn multi-modal data:
- Simple bimodal (2 clusters)
- Complex (ring, Swiss roll)

### Experiment 3: Parameter Efficiency

Compare number of parameters needed to achieve similar performance:
- CV Quantum: ~50-100 params
- DV Quantum: varies by encoding
- Classical: usually 1000s of params

### Experiment 4: Training Speed

Measure wall-clock time per epoch:
- CV Quantum: slowest (Fock space simulation)
- DV Quantum: medium
- Classical: fastest

---

## Troubleshooting Guide

### Problem: Zero gradient flow
**Solution**: Use SF tutorial pattern exactly. Check for tf.constant() calls.

### Problem: NaN losses
**Solution**: 
- Reduce learning rate
- Use gradient clipping
- Check parameter initialization

### Problem: Mode collapse
**Solution**:
- Process samples individually (not batch)
- Add diversity regularization
- Use WGAN-GP loss

### Problem: Vacuum collapse (all outputs ≈ 0)
**Solution**:
- Check initial displacements
- Verify squeezing parameters aren't too large
- Ensure Kerr parameters are small

---

## Summary

The key to successful CV Quantum GANs is **following the SF tutorial pattern exactly**:

1. ✓ Unified weight matrix (single tf.Variable)
2. ✓ Symbolic program built once
3. ✓ Direct parameter mapping
4. ✓ No tf.constant() in measurements
5. ✓ Individual sample processing for diversity

Your thesis can demonstrate that CV quantum systems provide a viable alternative to classical and DV quantum approaches, with unique characteristics in terms of parameter efficiency and natural handling of continuous data.

---

## References

1. Killoran et al., "Continuous-variable quantum neural networks", Phys. Rev. Research 1, 033063 (2019)
2. Strawberry Fields Documentation: https://strawberryfields.ai/
3. QML State Learning Tutorial: https://strawberryfields.ai/photonics/demos/run_state_learner.html
4. QML Neural Network Tutorial: https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html
