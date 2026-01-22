# Fresh CV Quantum GAN Implementation

This folder contains a clean, from-scratch implementation of a Continuous Variable (CV) Quantum GAN using Strawberry Fields. It follows the exact patterns from the official SF tutorials to ensure 100% gradient flow.

## Files

### `pure_sf_qgan.py`
**Main implementation** - Complete Quantum GAN with:
- `QuantumGenerator` - Pure quantum generator following SF tutorial pattern
- `QuantumDiscriminator` - Pure quantum discriminator
- `QuantumGAN` - Complete GAN with WGAN-GP training
- Data generators for testing (bimodal, ring)
- Visualization utilities

**Usage:**
```python
from pure_sf_qgan import QuantumGAN, bimodal_data_generator

qgan = QuantumGAN(latent_dim=4, data_dim=2, g_modes=3, d_modes=2)
history = qgan.train(bimodal_data_generator, epochs=30)
```

### `test_gradient_flow.py`
**Verification script** - Run this FIRST to verify your SF installation works correctly.

**Usage:**
```bash
python test_gradient_flow.py
```

Expected output: All tests should pass with "GRADIENT FLOW TEST PASSED"

### `interactive_qgan.py`
**Development script** - For interactive experimentation in Jupyter/Colab.
Contains simplified examples for understanding the concepts.

### `CV_QUANTUM_GAN_ANALYSIS.md`
**Documentation** - Comprehensive explanation of:
- Why gradients flow or don't flow in SF
- The SF tutorial pattern explained
- CV vs DV vs Classical comparison points
- Recommended experiments for your thesis

## Quick Start

1. **Verify gradient flow works:**
   ```bash
   cd src/fresh
   python test_gradient_flow.py
   ```

2. **Run full GAN training:**
   ```bash
   python pure_sf_qgan.py
   ```

3. **For interactive development:**
   Open `interactive_qgan.py` in Jupyter or run cells individually.

## Key Concepts

### The SF Tutorial Pattern (100% Gradient Flow)

```python
# 1. UNIFIED WEIGHT MATRIX
weights = tf.Variable(tf.concat([all_params], axis=1))

# 2. SYMBOLIC PARAMETERS
sf_params = np.array([prog.params(*i) for i in param_indices])

# 3. DIRECT MAPPING
mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}

# 4. EXECUTE
state = eng.run(prog, args=mapping).state

# 5. MEASURE (NO tf.constant()!)
x_quad = state.quad_expectation(mode, 0)  # Keep as tensor!
```

### What Breaks Gradients

- ❌ Individual `tf.Variable` for each parameter
- ❌ Using `tf.constant()` on measurement values
- ❌ Complex parameter transformations
- ❌ Batch averaging before measurement

## Thesis Comparison Points

1. **Parameter Efficiency**: CV Quantum typically needs fewer parameters
2. **Training Speed**: Classical is faster per step (GPU), but quantum may converge in fewer steps
3. **Distribution Matching**: Compare Wasserstein distances
4. **Mode Coverage**: Both can suffer mode collapse - measure diversity
5. **Gradient Stability**: Track gradient norms over training

## Legacy Code

The previous implementations are in `src/legacy/`. They're kept for reference but contain architectural issues that break gradient flow.
