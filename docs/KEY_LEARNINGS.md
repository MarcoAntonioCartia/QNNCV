# Key Learnings - QNNCV Development

Critical insights gained through trial and error. **Read this before making changes.**

---

## 1. The Kerr Gate is Essential

**Problem:** CV-QNNs with only Gaussian gates (Displacement, Squeeze, Rotation) can only produce Gaussian distributions.

**Solution:** Add the Kerr gate `K(κ) = exp(iκn²)` as a nonlinear activation.

**Proof:** Without Kerr, 3-modal target → single Gaussian output. With Kerr, 3-modal achieved W₁ = 0.064.

**Code:**
```python
if use_kerr:
    Kgate(self.weights[idx + 6])  # After displacement
```

---

## 2. Cutoff Dimension Limits Expressivity

**Problem:** Low cutoff (e.g., 5) truncates Fock space, limiting interference patterns.

**Rule of thumb:**
- 1-2 peaks: cutoff ≥ 7
- 3-4 peaks: cutoff ≥ 10
- 5+ peaks: cutoff ≥ 12

**Trade-off:** Higher cutoff = more memory, slower training.

---

## 3. GAN Training is Noisy (Expected!)

**Problem:** Wasserstein distance oscillates wildly during training.

**Why it's normal:**
1. Adversarial dynamics (G and D compete)
2. Stochastic batching
3. Non-convex quantum parameter landscape
4. Mode switching between solutions

**Solution:**
- Use moving average for monitoring
- Track best checkpoint, not final
- Don't stop early based on single W₁ spike

---

## 4. Gradient Flow with Strawberry Fields

**Problem:** SF's TensorFlow backend has gradient discontinuities.

**Fixes applied:**
- `scipy.integrate.simps` → `scipy.integrate.simpson` (deprecation fix)
- Use `tf.function` carefully (some SF ops don't trace)
- Avoid in-place tensor modifications

---

## 5. Weight Initialization Matters

**Problem:** Random initialization can put circuit in degenerate state.

**Solution:** Initialize squeeze/displacement magnitudes small (±0.1), Kerr near zero.

```python
self.weights = tf.Variable(
    tf.random.uniform([n_params], -0.1, 0.1)
)
```

---

## 6. Learning Rate Asymmetry

**Problem:** Equal G/D learning rates cause oscillation.

**Best settings found:**
```python
g_lr = 0.005  # Generator: slightly higher
d_lr = 0.001  # Discriminator: more conservative
```

---

## 7. Distribution-Based Loss > Sample-Based

**Problem:** Sample-based training has high variance (different samples each batch).

**Solution:** Compare probability distributions directly using:
- Wasserstein distance (W₁)
- Histogram comparison (100 bins)

This gives smoother gradients and faster convergence.

---

## 8. The Wigner Function Shows Everything

**Problem:** Hard to debug what the quantum circuit is doing.

**Solution:** Plot Wigner function at key epochs:
- Gaussian = single blob
- Non-Gaussian = interference fringes, multiple blobs
- Cat state = two blobs with negative region between

If Wigner looks Gaussian but you want multi-modal → Kerr is not working.

---

## 9. Peak Detection Threshold

**Problem:** Noise in distribution creates false peaks.

**Solution:** Only count peaks above 10% of max:
```python
peaks, _ = find_peaks(prob, height=0.1 * np.max(prob))
```

---

## 10. Layer Count vs Cutoff Trade-off

For fixed compute budget:
- More layers = better gate expressivity
- Higher cutoff = richer Fock space

**Empirically found:**
| Target | Layers | Cutoff | Why |
|--------|--------|--------|-----|
| 1-modal | 4-6 | 7-8 | Simple, low resource |
| 2-3 modal | 6-8 | 8-10 | Balanced |
| 4-5 modal | 8-12 | 10-15 | Need both |

---

## 11. Don't Trust Final Epoch

**Problem:** GAN oscillation means final weights aren't necessarily best.

**Solution:** Always track and use `best_weights`:
```python
if wasserstein < best_wasserstein:
    best_weights = generator.get_weights()
    best_wasserstein = wasserstein
```

---

## 12. Hermite Polynomials for Differentiable Measurement

**Problem:** Actual homodyne measurement breaks gradient flow.

**Solution:** Use Hermite polynomial expansion to compute position distribution from Fock state:
```python
# ψ_n(x) = H_n(x) * exp(-x²/2) / sqrt(2^n * n! * sqrt(π))
# P(x) = |Σ c_n * ψ_n(x)|²
```

This is implemented in `killoran_init.py`.

---

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Only Gaussian output | No Kerr gate | Add `--use-kerr` |
| W₁ stuck at ~1.0 | Learning rate too low | Increase `g_lr` |
| W₁ explodes | Learning rate too high | Decrease both LRs |
| Training very slow | Cutoff too high | Reduce cutoff |
| "NaN in gradient" | Squeeze too large | Clip squeeze parameter |
| Peak count oscillates | Normal GAN behavior | Use smoothed metrics |

---

## File Reference

| Need | File |
|------|------|
| Killoran generator | `src/models/generators/killoran_cvqnn.py` |
| Full training loop | `src/training/killoran_trainer.py` |
| Hermite functions | `src/utils/killoran_init.py` |
| Compatibility fixes | `src/utils/compatibility.py` |
| Visualization | `src/utils/visualization.py` |
