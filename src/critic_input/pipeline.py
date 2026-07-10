"""
Critic-input pipeline
=====================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).

Invariants (load-bearing, golden-tested):
- to_critic_input peak-normalizes EVERY critic input (real, fake, GP
  interpolants, and the generator's differentiable adversarial pass).
- Blur is applied at ENTRY POINTS only, never inside to_critic_input
  (which runs twice under instance noise) — at most once per tensor path.
"""

import numpy as np
import tensorflow as tf


def to_critic_input(x):
    """
    Rescale densities to peak = 1 for the critic (per sample if batched).

    Sum-normalized 40x40 densities have pixel values ~0.01 and L2 distances
    ~0.1 between entirely different shapes, so a Lipschitz-constrained
    (WGAN-GP) critic can never separate real from fake by more than ~0.1 --
    observed as D(r) = D(f) throughout training. Peak-normalization restores
    O(1) dynamic range. Differentiable (also used inside the generator's
    tape), accepts (gx, gy) or (B, gx, gy).
    """
    peak = tf.reduce_max(x, axis=[-2, -1], keepdims=True)
    return x / (peak + 1e-10)


def build_blur_kernel(sigma):
    """Build the fixed Gaussian kernel for critic_blur (None if sigma <= 0).

    sigma is in grid cells; kernel radius 3*sigma (odd size), normalized
    to sum 1 so density mass is preserved (up to border clipping).
    """
    if sigma is None or sigma <= 0:
        return None
    radius = max(1, int(np.ceil(3 * sigma)))
    coords = np.arange(-radius, radius + 1)
    g = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel2d = np.outer(g, g)
    kernel2d /= kernel2d.sum()
    return tf.constant(kernel2d[:, :, None, None], dtype=tf.float32)


def critic_blur(x, kernel):
    """Differentiable Gaussian blur of densities, (gx, gy) or (B, gx, gy).

    Removes the high-frequency Hermite/Fock interference ripples that only
    generated densities carry, so the critic must judge on coarse shape.
    Applied exactly once per critic input, BEFORE peak-normalization; since
    blur is linear, GP interpolants of blurred inputs equal blurred
    interpolants. No-op when kernel is None (--critic-blur 0).
    """
    if kernel is None:
        return x
    single = (len(x.shape) == 2)
    if single:
        x = tf.expand_dims(x, 0)
    x = tf.nn.conv2d(x[..., None], kernel, strides=1, padding='SAME')[..., 0]
    if single:
        x = x[0]
    return x


def compute_gradient_penalty(discriminator, real_batch, fake_batch):
    """
    Compute gradient penalty for WGAN-GP over a batch.

    GP = E_b[(||∇D(x̂_b)||_2 - 1)²]
    where x̂_b = ε_b*real_b + (1-ε_b)*fake_b, one ε per batch element.

    Inputs are (B, gx, gy) and should be the SAME tensors the critic is
    trained on (peak-normalized, noisy if instance noise is active).
    """
    # Random interpolation coefficient per batch element (eager: shape is concrete)
    b = real_batch.shape[0]
    epsilon = tf.random.uniform([b, 1, 1], 0.0, 1.0)

    # Interpolate between real and fake
    interpolated = epsilon * real_batch + (1 - epsilon) * fake_batch

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    # Compute gradients w.r.t. interpolated input, per element
    grads = gp_tape.gradient(pred, interpolated)  # (B, gx, gy)

    # Per-element L2 norm, then mean penalty over the batch
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-8)
    return tf.reduce_mean(tf.square(grad_norm - 1.0))
