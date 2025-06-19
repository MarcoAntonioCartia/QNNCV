# Quantum GAN Mode Collapse Debugging Plan

## Current Situation
- **Problem**: Persistent mode collapse in quantum GANs despite various fixes
- **Observation**: Both generators and discriminators with only quantum variables still collapse
- **Key Finding**: Generator produces only Mode 1 samples (100/0 split)

## Root Cause Analysis

### 1. **Gradient Flow Issues**
- Quantum circuits may have vanishing/exploding gradients
- Parameter initialization might be suboptimal
- The quantum state measurement process might lose gradient information

### 2. **Discriminator Dominance**
- Discriminator might be too powerful, preventing generator learning
- The pure quantum discriminator might not provide useful gradients

### 3. **Mode Selection Mechanism**
- The latent-to-mode mapping might be ineffective
- Mode parameters might not be differentiating properly

### 4. **Loss Function Issues**
- The bimodal loss might not be properly balanced
- Quantum-specific losses might be needed

## Debugging Steps

### Step 1: Gradient Analysis
```python
# Check gradient magnitudes for all components
def analyze_gradients(generator, discriminator, real_data):
    with tf.GradientTape(persistent=True) as tape:
        z = tf.random.normal([8, 6])
        fake = generator.generate(z)
        
        # Generator loss
        d_fake = discriminator.discriminate(fake)
        g_loss = tf.reduce_mean(-tf.math.log(d_fake + 1e-8))
        
        # Discriminator loss
        d_real = discriminator.discriminate(real_data[:8])
        d_loss = tf.reduce_mean(-tf.math.log(d_real + 1e-8) - tf.math.log(1 - d_fake + 1e-8))
    
    g_grads = tape.gradient(g_loss, generator.trainable_variables)
    d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
    
    return g_grads, d_grads
```

### Step 2: Mode Parameter Investigation
```python
# Check if mode parameters are actually changing
def check_mode_parameters(generator):
    mode1_params = generator.mode1_params.numpy()
    mode2_params = generator.mode2_params.numpy()
    
    print(f"Mode 1 params: {mode1_params}")
    print(f"Mode 2 params: {mode2_params}")
    print(f"Difference: {np.linalg.norm(mode1_params - mode2_params)}")
```

### Step 3: Simplified Test
Create a minimal test case:
1. Fixed discriminator (not trainable)
2. Generator with explicit mode control
3. Direct mode supervision

### Step 4: Alternative Approaches

#### A. Explicit Mode Control
```python
class ExplicitModeGenerator:
    def generate(self, z, mode_labels):
        # Use explicit mode labels instead of inferring from latent
        pass
```

#### B. Wasserstein Loss
Replace binary cross-entropy with Wasserstein distance for better gradients

#### C. Progressive Training
1. Train generator to produce Mode 1 only
2. Train generator to produce Mode 2 only  
3. Train generator to produce both modes

#### D. Quantum Circuit Modifications
- Add more entanglement between modes
- Use different measurement strategies
- Implement quantum dropout

### Step 5: Measurement Strategy Analysis
The current measurement extraction might be the issue:
```python
def _extract_bimodal_measurement(self, state, mode_weight):
    # Current: Maps photon statistics to mode centers
    # Problem: Might always favor one mode
    
    # Alternative 1: Use phase information
    # Alternative 2: Use entanglement measures
    # Alternative 3: Use multiple measurement bases
```

## Proposed Solutions

### Solution 1: Hybrid Classical-Quantum Approach
Keep quantum circuits but add minimal classical components for stable mode selection

### Solution 2: Modified Training Procedure
1. Pre-train modes separately
2. Use curriculum learning
3. Implement mode-specific discriminators

### Solution 3: Quantum Circuit Redesign
- Use variational quantum eigensolver (VQE) inspired architectures
- Implement quantum feature maps
- Add measurement error mitigation

### Solution 4: Loss Function Redesign
```python
class ImprovedBimodalLoss:
    def __call__(self, real, fake, generator):
        # Mode assignment loss
        mode_loss = self.compute_mode_assignment_loss(fake)
        
        # Distribution matching loss
        dist_loss = self.compute_distribution_loss(real, fake)
        
        # Quantum state regularization
        quantum_loss = self.compute_quantum_regularization(generator)
        
        return mode_loss + dist_loss + quantum_loss
```

## Implementation Priority

1. **Immediate**: Gradient analysis and debugging
2. **Short-term**: Simplified test cases
3. **Medium-term**: Alternative training procedures
4. **Long-term**: Quantum circuit redesign

## Success Metrics

- [ ] Both modes generated (count > 10 each)
- [ ] Balance score > 0.3
- [ ] Separation accuracy > 0.7
- [ ] Stable training (no oscillations)
- [ ] Reproducible results

## Next Steps

1. Run gradient analysis to identify where gradients vanish
2. Test with fixed discriminator to isolate generator issues
3. Implement explicit mode control for debugging
4. Try Wasserstein loss formulation
5. Consider hybrid approach if pure quantum fails
