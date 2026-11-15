"""
2D CV-QGAN training loop
========================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor), plus
SEAM 2: the training objective is a class boundary. The trainers own exactly
the per-epoch discriminator phase and generator phase; train_2d_qgan keeps
everything else (setup, dataset, history, validation, checkpointing, outputs)
verbatim.

  WGANGPTrainer        — current GAN-mode behavior (WGAN-GP critic), 1:1.
  PureSupervisedTrainer— current Path A behavior (D disabled, fixed canonical
                         target, KL objective), 1:1.

Future trainers (e.g. an MMD or energy-distance objective) plug in as
additional trainer classes implementing discriminator_epoch/generator_epoch —
declared here as the seam, NO such trainer is implemented yet.

Mode selection reproduces the existing auto-detection exactly:
pure_supervised = (supervised_weight >= 1.0 and supervised_warmup >= epochs).

RNG-order invariant (golden-tested): per epoch, GAN mode draws
n_critic x (np.randint batch idxs -> tf z -> [2x tf noise if current_noise>0]
-> tf GP epsilon) -> G np.randint -> G tf z -> [validation draws] -> [plot
draws]. Path A draws NO D-phase RNG and NO G-phase np.randint.
"""

import os
import random
import secrets
import sys
import time as _time
from datetime import datetime

import numpy as np
import tensorflow as tf
import strawberryfields as sf
import matplotlib.pyplot as plt

from src.quantum.hermite import recommend_cutoff
from src.quantum.circuit import CVQGANGenerator, EncodingSpec
from src.quantum import weights_io
from src.models.discriminators.qgan2d_discriminator import Discriminator2D
from src.families.registry import get_family
from src.data.dataset import generate_dataset
from src.metrics.wasserstein import compute_wasserstein_2d
from src.metrics.energy import build_energy_distance_context
from src.metrics.validation import validate
from src.critic_input.pipeline import (to_critic_input, build_blur_kernel,
                                       critic_blur, compute_gradient_penalty)
from src.viz.plots import plot_comparison, plot_training_history


def resolve_seed(seed):
    """Return a concrete integer seed.

    If ``seed`` is None, draw an entropy-based 32-bit seed so unseeded runs still
    vary — but the resolved integer is always returned so it can be recorded.
    Idempotent: passing an int returns it unchanged.
    """
    return int(seed) if seed is not None else secrets.randbits(32)


def seed_everything(seed):
    """Seed all sources of randomness used by this module.

    Covers Python stdlib ``random``, NumPy global RNG (dataset generation, batch
    index draws), and TensorFlow global RNG (weight init, latent z, instance
    noise, GP interpolation). Strawberry Fields' TF backend draws from the TF
    global state, so it is covered here too; it exposes no dedicated seed knob.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =============================================================================
# SEAM 2: trainer classes (objective boundary; current behavior only)
# =============================================================================

class _BaseTrainer2D:
    """Shared per-epoch machinery. Subclasses fix the objective."""

    STOP = object()   # sentinel returned by generator_epoch == old `break`

    def __init__(self, *, generator, discriminator, g_optimizer, d_optimizer,
                 train_set, fixed_real_tf, blur_kernel, xvec, yvec,
                 batch_size, n_critic, gp_weight, gp_warmup,
                 instance_noise, noise_anneal, noise_floor,
                 supervised_weight, supervised_warmup,
                 ket_penalty_weight, g_grad_clip, norm_floor=0.99):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.train_set = train_set
        self.fixed_real_tf = fixed_real_tf
        self.blur_kernel = blur_kernel
        self.xvec = xvec
        self.yvec = yvec
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.gp_warmup = gp_warmup
        self.instance_noise = instance_noise
        self.noise_anneal = noise_anneal
        self.noise_floor = noise_floor
        self.supervised_weight = supervised_weight
        self.supervised_warmup = supervised_warmup
        self.ket_penalty_weight = ket_penalty_weight
        self.g_grad_clip = g_grad_clip
        self.norm_floor = norm_floor
        self.degenerate_count = 0

    def schedules(self, epoch):
        """Per-epoch GP-warmup and instance-noise schedules (both modes)."""
        # Compute current GP weight (warmup)
        current_gp_weight = self.gp_weight * min(1.0, epoch / max(1, self.gp_warmup))

        # Instance noise: decays to 0 over noise_anneal epochs
        # This blurs real/fake boundary, preventing D from trivially winning
        # Decays from instance_noise to noise_floor over noise_anneal epochs.
        # A nonzero floor keeps the critic from overfitting high-frequency
        # generator ripples after the anneal ends. floor=0 reduces exactly
        # to the legacy schedule.
        current_noise = self.noise_floor + (self.instance_noise - self.noise_floor) * max(
            0.0, 1.0 - epoch / max(1, self.noise_anneal))

        return current_gp_weight, current_noise

    # --- objective hooks -----------------------------------------------------

    def _adversarial_loss(self, gen_prob):
        """Generator-side objective term, recorded as 'adversarial_loss'.
        Default: -E[D(fake)] on the blurred, peak-normalised densities."""
        fake_score = self.discriminator(
            to_critic_input(critic_blur(gen_prob, self.blur_kernel)), training=False)
        # Adversarial loss: G wants to maximize fake_score
        return -tf.reduce_mean(fake_score)

    def _g_real_batch(self):
        """Select the real target(s) for the generator step."""
        raise NotImplementedError

    def _combine_losses(self, epoch, adversarial_loss, supervised_loss,
                        ket_penalty):
        """Return (g_loss, current_sw) for this epoch."""
        raise NotImplementedError

    # --- generator phase (shared body, 1:1 with the monolith) ----------------

    def generator_epoch(self, epoch):
        """One generator update. Returns a stats dict, None (== old
        `continue`: degenerate output, skip epoch bookkeeping) or STOP
        (== old `break`: too many degenerates)."""
        # === GENERATOR TRAINING ===
        # Select the real target(s).
        #   Path A: always the SAME fixed canonical target (representability test).
        #   GAN mode: random family members (adversarial training over the family).
        real_tf = self._g_real_batch()
        if hasattr(self.discriminator, 'new_points'):
            self.discriminator.new_points()   # point-wise critics: one subset per step

        # Sample new latent vectors
        z = tf.random.normal([self.batch_size, self.generator.latent_dim])

        with tf.GradientTape() as tape:
            gen_prob, ket_norm_tf = self.generator.generate_batch(
                z, self.xvec, self.yvec, return_ket_norm=True
            )

            if not tf.reduce_all(tf.math.is_finite(gen_prob)):
                print(f"  Warning: Degenerate at epoch {epoch}")
                self.degenerate_count += 1
                if self.degenerate_count > 30:
                    print("Stopping: Too many degenerate outputs")
                    return self.STOP
                return None
            else:
                self.degenerate_count = 0

            # Objective term (SEAM 2): the critic score for WGAN-GP, a fixed
            # kernel two-sample statistic for MMD. Same ops as before for the
            # default objective.
            adversarial_loss = self._adversarial_loss(gen_prob)

            # Supervised loss: KL(target || generated).
            # Scale-free (dimensionless, independent of grid resolution) and
            # order-1 in magnitude, unlike mean-squared-error over a normalized
            # density which is ~1e-6 here and produces vanishing gradients.
            # Both tensors are already normalized to sum to 1 in
            # generate_distribution_2d. KL heavily penalizes placing ~zero mass
            # where the target has mass -- exactly the ring-hole failure mode.
            eps = 1e-10
            supervised_loss = tf.reduce_mean(tf.reduce_sum(
                real_tf * (tf.math.log(real_tf + eps) - tf.math.log(gen_prob + eps)),
                axis=[-2, -1]
            ))

            # Ket-norm penalty: keep the truncated state representable in the Fock
            # cutoff. Without it, the optimizer drifts gate parameters past safe
            # thresholds (squeeze magnitudes > 0.5 are catastrophic at cutoff=10)
            # and W1 numbers become truncation artifacts.
            ket_penalty = tf.reduce_mean(tf.square(1.0 - ket_norm_tf))

            # Combined loss.
            #   Path A: pure supervised (KL) + ket penalty, no adversarial term.
            #   GAN mode: warmup-decayed blend of adversarial + supervised.
            g_loss, current_sw = self._combine_losses(
                epoch, adversarial_loss, supervised_loss, ket_penalty)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        # Filter (grad, var) PAIRS so a None grad never misaligns the zip
        # (matters once the generator exposes more than one variable).
        pairs = [(g, v) for g, v in
                 zip(g_grads, self.generator.trainable_variables)
                 if g is not None]
        if pairs:
            if self.g_grad_clip > 0:
                gs = [tf.clip_by_norm(g, self.g_grad_clip) for g, _ in pairs]
            else:
                gs = [g for g, _ in pairs]
            self.g_optimizer.apply_gradients(zip(gs, [v for _, v in pairs]))
            g_grad_norm = float(tf.linalg.global_norm(gs))
        else:
            g_grad_norm = 0.0

        # Per-sample ket-norm monitoring: z-dependent squeezing makes the
        # Fock-truncation error z-dependent, so a batch MEAN can hide one bad
        # sample. Diagnostic only — the penalty above stays batch-mean.
        kn_np = ket_norm_tf.numpy().reshape(-1)
        worst = int(np.argmin(kn_np))
        below = int(np.sum(kn_np < self.norm_floor))
        if below:
            print(f"WARNING epoch {epoch}: ket_norm < {self.norm_floor} for "
                  f"{below}/{kn_np.size} samples (min {kn_np.min():.4f}); "
                  f"worst z = {np.round(z.numpy()[worst], 3).tolist()}",
                  file=sys.stderr)

        return {
            'g_loss': g_loss,
            'supervised_loss': supervised_loss,
            'adversarial_loss': adversarial_loss,
            'current_sw': current_sw,
            'gen_prob': gen_prob,
            'ket_norm_tf': ket_norm_tf,
            'g_grad_norm': g_grad_norm,
            'ket_norm_min': float(kn_np.min()),
            'ket_norm_max': float(kn_np.max()),
            'ket_norm_warn': below,
            'worst_z': z.numpy()[worst].copy(),
        }


class WGANGPTrainer(_BaseTrainer2D):
    """GAN-mode objective: WGAN-GP critic + warmup-decayed supervised blend.

    1:1 extraction of the monolith's adversarial branch.
    """

    def discriminator_epoch(self, epoch):
        """n_critic WGAN-GP critic updates (one epoch's D phase)."""
        # === DISCRIMINATOR TRAINING ===
        current_gp_weight, current_noise = self.schedules(epoch)

        d_losses = []
        d_grad_norms = []
        gp_values = []
        d_real_scores = []
        d_fake_scores = []

        for _ in range(self.n_critic):
            if hasattr(self.discriminator, 'new_points'):
                self.discriminator.new_points()   # same subset for real, fake and GP
            # Pick a RANDOM batch of real samples from train_set
            idxs = np.random.randint(len(self.train_set), size=self.batch_size)
            real_tf = tf.constant(
                np.stack([self.train_set[i][0] for i in idxs]), dtype=tf.float32)

            # Generate with random z, OUTSIDE the critic's tape: the fake
            # is a constant for the D update, and recording the SF circuit
            # on D's tape costs memory/time for gradients never taken.
            z = tf.random.normal([self.batch_size, self.generator.latent_dim])
            gen_prob = self.generator.generate_batch(z, self.xvec, self.yvec)
            if not tf.reduce_all(tf.math.is_finite(gen_prob)):
                continue
            gen_prob = tf.stop_gradient(gen_prob)

            # Blur (optional), then peak-normalize what the critic sees.
            # Single blur per pipeline: the noise re-normalization below
            # must NOT blur again.
            real_in = to_critic_input(critic_blur(real_tf, self.blur_kernel))
            fake_in = to_critic_input(critic_blur(gen_prob, self.blur_kernel))

            with tf.GradientTape() as tape:
                # Add instance noise to D inputs (handicap D); noise std is
                # now relative to peak=1, so 0.1 = 10% of the brightest pixel
                if current_noise > 0:
                    real_in_n = to_critic_input(tf.nn.relu(
                        real_in + current_noise * tf.random.normal(tf.shape(real_in))))
                    fake_in_n = to_critic_input(tf.nn.relu(
                        fake_in + current_noise * tf.random.normal(tf.shape(fake_in))))
                else:
                    real_in_n = real_in
                    fake_in_n = fake_in

                # Discriminator scores (inputs already carry the batch dim)
                real_score = self.discriminator(real_in_n, training=True)
                fake_score = self.discriminator(fake_in_n, training=True)

                # WGAN loss: D wants real_score > fake_score
                d_loss_wgan = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

                # Gradient penalty (WGAN-GP), on the same inputs D sees
                gp = compute_gradient_penalty(self.discriminator, real_in_n, fake_in_n)

                # Total D loss = WGAN loss + λ * GP
                d_loss = d_loss_wgan + current_gp_weight * gp

            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            if d_grads[0] is not None:
                self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
                d_grad_norms.append(float(tf.linalg.global_norm(d_grads)))

            d_losses.append(float(d_loss_wgan))  # Track WGAN loss (not total)
            gp_values.append(float(gp))
            d_real_scores.append(float(tf.reduce_mean(real_score)))
            d_fake_scores.append(float(tf.reduce_mean(fake_score)))

        return {
            'd_loss_avg': np.mean(d_losses) if d_losses else 0.0,
            'd_grad_avg': np.mean(d_grad_norms) if d_grad_norms else 0.0,
            'gp_avg': np.mean(gp_values) if gp_values else 0.0,
            'd_real_scores': d_real_scores,
            'd_fake_scores': d_fake_scores,
            'current_gp_weight': current_gp_weight,
            'current_noise': current_noise,
        }

    def _g_real_batch(self):
        idxs = np.random.randint(len(self.train_set), size=self.batch_size)
        return tf.constant(
            np.stack([self.train_set[i][0] for i in idxs]), dtype=tf.float32)

    def _combine_losses(self, epoch, adversarial_loss, supervised_loss,
                        ket_penalty):
        current_sw = self.supervised_weight * max(0.0, 1.0 - epoch / max(1, self.supervised_warmup))
        g_loss = (
            (1 - current_sw) * adversarial_loss
            + current_sw * supervised_loss
            + self.ket_penalty_weight * ket_penalty
        )
        return g_loss, current_sw


class PureSupervisedTrainer(_BaseTrainer2D):
    """Path A objective: discriminator disabled, target fixed to the
    canonical family member, scale-free KL(target || generated) loss.

    1:1 extraction of the monolith's pure_supervised branch. Draws NO RNG in
    the D phase and NO numpy RNG in the G-phase target selection (the RNG
    stream must match the monolith exactly; golden-tested).
    """

    def discriminator_epoch(self, epoch):
        # Path A: no critic. Skip discriminator training entirely.
        current_gp_weight, current_noise = self.schedules(epoch)
        return {
            'd_loss_avg': 0.0,
            'd_grad_avg': 0.0,
            'gp_avg': 0.0,
            'd_real_scores': [],
            'd_fake_scores': [],
            'current_gp_weight': current_gp_weight,
            'current_noise': current_noise,
        }

    def _g_real_batch(self):
        return tf.expand_dims(self.fixed_real_tf, 0)   # broadcasts over the batch

    def _combine_losses(self, epoch, adversarial_loss, supervised_loss,
                        ket_penalty):
        current_sw = 1.0
        g_loss = supervised_loss + self.ket_penalty_weight * ket_penalty
        return g_loss, current_sw


class MMDTrainer(_BaseTrainer2D):
    """Born-machine objective: no critic; the generator minimises a fixed-
    kernel maximum mean discrepancy (MMD^2) between its output and the data.
    Same generators, same data, same validation and checkpointing as the
    GAN, so the arm isolates the effect of the training objective.

    mmd_kernel='density' (default): the family-level statistic. Each density
    is one point; k(p, q) = sum_b exp(-||p - q||^2 / (b^2 * m)) on flattened,
    sum-normalised densities, m = median pairwise SQUARED distance over the
    training set (fixed at construction, so the kernel is pre-registered and
    data-only), b in mmd_bandwidths. At b = 1 this is rescore.py's evaluation
    kernel exp(-d^2 / m); as a Gaussian length scale, h = b * sqrt(m / 2). The unbiased MMD^2 between the batch of
    B generated densities and a fresh batch of mmd_real_batch training
    members is minimised (generative moment matching, Li et al. 2015;
    Dziugaite et al. 2015). Penalises collapse: a generator emitting one
    density cannot match the spread of the family.

    mmd_kernel='sample': the sample-space MMD used for Born machines (Liu &
    Wang 2018 use a mixture of Gaussian kernels with FIXED absolute
    bandwidths; their kernel divides by 2*sigma, not 2*sigma^2, so their
    sigma = 0.25, 10, 1000 are length scales 0.5, 3.2, 31.6 on a 1024-point
    support: even their widest is far below the support size in length terms
    [exponent read from extracted text, check the PDF]. Ours are length
    scales l = b * 0.5 in x units, i.e. 0.25-1 for the default b, below the
    ring's diameter ~2.3).
    Each generated or real density is a distribution over grid points; the
    unbiased U-statistic over the B generated and R real densities is
        sum_{b!=b'} p_b^T K p_b' / (B(B-1)) + sum_{r!=r'} q_r^T K q_r' / (R(R-1))
        - 2 sum_{b,r} p_b^T K q_r / (B R),
    computed exactly (densities are known, no sampling). Its expectation is
    the MMD^2 between the generated and real MIXTURES, so a generator that
    collapses onto the family's mean density scores 0, like a perfect one:
    it cannot see collapse (the earlier V-statistic (p_bar - q_bar)^T K
    (p_bar - q_bar) was worse: its (1/B) tr(K Cov_z p) bias rewarded
    collapse). Kept as the literal Born-machine baseline, not the main arm.

    Both kernels are a plain SUM over mmd_bandwidths, not a mean: changing
    the number of bandwidths rescales the loss and hence the effective
    learning rate, so runs with different bandwidth counts are not
    comparable at the same lr.

    mmd_shots=N > 0 (finite measurement budget): inside the loss, each
    generated density is replaced by the histogram of N outcomes drawn from
    it (multinomial on the grid), as a photonic device sees homodyne samples,
    not densities. Kolarovszki et al. (2024) train CV Born machines this way
    with ~10^3 homodyne shots per iteration for both loss and gradients;
    Ngo et al. (2026) study MMD training under a finite shot budget (there
    residual HARDWARE noise, not shot noise, acts as the regulariser). Ours
    is a cheaper approximation: straight-through estimator, the loss VALUE
    carries the shot noise, the gradient flows through the exact density.
    Describe it as a sensitivity test of the loss to measurement noise, with
    exact gradients, not as hardware-like training. With the 'sample' kernel
    the noisy loss is an unbiased estimate of the exact one (linear in each
    histogram, and the U-statistic never pairs a histogram with itself);
    with the 'density' kernel the histogram sits inside an exponential, so
    it is biased. On hardware the gradient would also cost shots: two
    shifted circuits per parameter, 186 x 2 x 8 x 1000 ~ 3.0e6 shots per
    step at 1000 shots, against 8000 for the loss value. And only some
    parameters have a shift rule: Kerr gates have no known exact one, and a
    non-Gaussian gate also breaks the rule for every Gaussian gate BEFORE it
    (Kolarovszki et al. 2024 train only gates after all non-Gaussian ones),
    so with a Kerr gate ending every Killoran layer only the last layer's
    Gaussian gates would qualify. Exactness of the VC Gaussian-gate rules
    for a kernel loss (as opposed to quadrature expectation values) is not
    verified here; do not claim it.
    The real batch stays exact. N = 0 (default) = exact densities.
    Meaningful with the 'sample' kernel (it compares grid points, exactly how
    shot-trained Born machines work). With the 'density' kernel only when N
    is well above the number of grid cells: a 100-shot histogram on a 32x32
    grid is so spiky that it sits far from every smooth training density,
    the kernel goes flat and the loss stops discriminating (seen in the
    smoke check: MMD^2 1.80 exact -> 0.61 at 100 shots, a symptom, not a fit).

    The ket-norm penalty and the supervised blend are combined exactly as in
    WGANGPTrainer. RNG: no critic-phase draws; the generator phase draws the
    supervised batch and, inside the objective, the MMD real batch (then the
    shot samples when mmd_shots > 0), all from the NumPy global stream.
    """

    SAMPLE_LENGTH = 0.5   # x units; ring width ~0.25, radius ~1.15

    def __init__(self, *, mmd_kernel='density', mmd_bandwidths=(0.5, 1.0, 2.0),
                 mmd_real_batch=32, mmd_shots=0, **kw):
        super().__init__(**kw)
        self.mmd_shots = int(mmd_shots)
        if self.mmd_shots < 0:
            raise ValueError(f"mmd_shots must be >= 0, got {mmd_shots}")
        if mmd_kernel not in ('density', 'sample'):
            raise ValueError(f"mmd_kernel must be 'density' or 'sample', got {mmd_kernel!r}")
        if not mmd_bandwidths or any(float(b) <= 0 for b in mmd_bandwidths):
            raise ValueError(f"mmd_bandwidths must be a non-empty list of positive numbers, "
                             f"got {mmd_bandwidths!r}")
        if int(self.batch_size) < 2 or int(mmd_real_batch) < 2:
            raise ValueError("the unbiased MMD needs at least 2 generated and 2 real "
                             f"densities per step (batch_size={self.batch_size}, "
                             f"mmd_real_batch={mmd_real_batch})")
        self.mmd_kernel = mmd_kernel
        self.mmd_bandwidths = tuple(float(b) for b in mmd_bandwidths)
        self.mmd_real_batch = int(mmd_real_batch)
        train = np.stack([m for m, _ in self.train_set]).astype(np.float64)
        self._train_flat = (train.reshape(len(train), -1)
                            / train.reshape(len(train), -1).sum(1, keepdims=True)).astype(np.float32)
        if mmd_kernel == 'density':
            F = self._train_flat.astype(np.float64)
            sq = (F ** 2).sum(1)
            D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * F @ F.T, 0.0)
            self.mmd_median = float(np.median(D2[np.triu_indices(len(F), 1)]))
            self._scales = tf.constant([b * b * self.mmd_median for b in self.mmd_bandwidths],
                                       tf.float32)
        else:
            X, Y = np.meshgrid(np.asarray(self.xvec, np.float64), np.asarray(self.yvec, np.float64))
            P = np.stack([X.ravel(), Y.ravel()], 1)
            d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            K = sum(np.exp(-d2 / (2.0 * (b * self.SAMPLE_LENGTH) ** 2)) for b in self.mmd_bandwidths)
            self._Ks = tf.constant(K, tf.float32)

    def discriminator_epoch(self, epoch):
        current_gp_weight, current_noise = self.schedules(epoch)
        return {'d_loss_avg': 0.0, 'd_grad_avg': 0.0, 'gp_avg': 0.0,
                'd_real_scores': [], 'd_fake_scores': [],
                'current_gp_weight': current_gp_weight, 'current_noise': current_noise}

    def _g_real_batch(self):
        idxs = np.random.randint(len(self.train_set), size=self.batch_size)
        return tf.constant(
            np.stack([self.train_set[i][0] for i in idxs]), dtype=tf.float32)

    def _kernel(self, A, B):
        sa = tf.reduce_sum(A * A, 1)[:, None]
        sb = tf.reduce_sum(B * B, 1)[None, :]
        d2 = tf.maximum(sa + sb - 2.0 * tf.matmul(A, B, transpose_b=True), 0.0)
        return tf.add_n([tf.exp(-d2 / s) for s in tf.unstack(self._scales)])

    def _adversarial_loss(self, gen_prob):
        B = int(gen_prob.shape[0])
        idx = np.random.randint(len(self.train_set), size=self.mmd_real_batch)
        real = tf.constant(self._train_flat[idx])
        fake = tf.reshape(gen_prob, [B, -1])
        fake = fake / (tf.reduce_sum(fake, 1, keepdims=True) + 1e-12)
        if self.mmd_shots > 0:
            p = np.clip(fake.numpy().astype(np.float64), 0.0, None)
            p = p / p.sum(1, keepdims=True)
            hist = np.stack([np.random.multinomial(self.mmd_shots, row) for row in p])
            hist = tf.constant(hist / float(self.mmd_shots), tf.float32)
            fake = fake + tf.stop_gradient(hist - fake)   # straight-through
        R = self.mmd_real_batch
        if self.mmd_kernel == 'sample':
            FK = tf.matmul(fake, self._Ks)                  # (B, G^2)
            Kff = tf.matmul(FK, fake, transpose_b=True)     # p_b^T K p_b'
            Krr = tf.matmul(tf.matmul(real, self._Ks), real, transpose_b=True)
            Kfr = tf.matmul(FK, real, transpose_b=True)
        else:
            Kff, Krr, Kfr = self._kernel(fake, fake), self._kernel(real, real), self._kernel(fake, real)
        return ((tf.reduce_sum(Kff) - tf.linalg.trace(Kff)) / (B * (B - 1))
                + (tf.reduce_sum(Krr) - tf.linalg.trace(Krr)) / (R * (R - 1))
                - 2.0 * tf.reduce_mean(Kfr))

    def _combine_losses(self, epoch, adversarial_loss, supervised_loss, ket_penalty):
        current_sw = self.supervised_weight * max(0.0, 1.0 - epoch / max(1, self.supervised_warmup))
        g_loss = ((1 - current_sw) * adversarial_loss + current_sw * supervised_loss
                  + self.ket_penalty_weight * ket_penalty)
        return g_loss, current_sw


# =============================================================================
# Training Loop
# =============================================================================

def train_2d_qgan(
    # Keyword-only. Params without a default here are the ones whose old
    # signature default diverged from the CLI default: the argparse defaults
    # in build_parser() are the single source of truth, so callers must be
    # explicit (the CLI's main() always passes them).
    *,
    family_name,            # required (CLI --family is required too)
    n_train=400,
    n_val=100,
    n_total_modes=2,        # Total qumodes (2 output + ancilla)
    n_layers=6,
    cutoff_dim,             # required (old sig default 8 diverged from CLI 12)
    use_kerr=True,
    epochs=500,
    g_lr=0.005,
    d_lr,                   # required (old 0.0002 diverged from CLI 0.005)
    n_critic,               # required (old 1 diverged from CLI 5)
    batch_size,             # required (old 1 diverged from CLI 8); SF TF batching, 1 = legacy sequential
    supervised_weight=0.0,  # Start with 30% supervised to give G a head start
    supervised_warmup=200,  # Slow decay from supervised to adversarial
    gp_weight=5.0,          # Gradient penalty weight (lambda)
    gp_warmup=50,           # Epochs to warm up GP
    instance_noise=0.1,     # Initial noise std added to D inputs (capacity handicap)
    noise_anneal=200,       # Epochs to anneal instance noise to the floor
    noise_floor=0.0,        # Instance noise floor after anneal (0 = legacy, anneal to zero)
    critic_blur_sigma=0.0,  # Gaussian blur (grid cells) on critic inputs (0 = off)
    d_dropout,              # required (old 0.3 diverged from CLI 0.0)
    latent_scale,           # required (old 1.0 diverged from CLI 0.3)
    ket_penalty_weight=20.0,  # Weight on (1 - ket_norm)^2 penalty in G's loss
    g_grad_clip=5.0,          # Max norm for G gradient clipping (<=0 disables)
    grid_size=40,
    x_range=3.0,
    log_dir=None,
    plot_every=20,
    val_every=20,
    seed=None,              # Resolved to a concrete int and recorded; None => entropy-based
    deterministic=False,    # Opt into stronger (slower) same-machine determinism
    encoding='input_displacement',  # SEAM 1: 'input_displacement' | 'interleaved_squeeze'
    inj_layers=None,        # interleaved only: layers to inject ("0,2" or sequence; None = all)
    r_max=None,             # interleaved only: hard clip |r| <= r_max on squeeze magnitudes
    init_weights=None,      # Warm start: path to a best_weights.npy (v1 or v2 format)
    norm_floor=0.99,        # Per-sample ket-norm warning threshold
    squeeze_scale=0.0,      # interleaved only: z-squeeze gain w_s (0 = channel off)
    train_squeeze_scale=False,  # interleaved only: make w_s trainable (init at squeeze_scale)
    inj_latent='shared',    # interleaved only: 'shared' (z[2m]) | 'dedicated' (extra z components)
    generator='cv',         # benchmark backend: 'cv' | 'classical' | 'dv' (src/backends)
    critic='dense_2d',      # 'dense_2d' | 'coord_2d' | 'cv_qnn_2d' | 'dv_qnn_2d'
    critic_points=64,       # point-wise critics: grid points sampled per critic call (0 = all)
    critic_layers=6,        # point-wise quantum critics: circuit layers
    gen_hidden='9,9',       # classical generator: hidden sizes of the coordinate net
    gen_activation='tanh',  # classical generator: 'tanh' | 'sine' (SIREN)
    gen_fourier=0,          # classical generator: random Fourier features of the point (0 = off)
    dv_qubits=10,           # dv generator: qubits (2k -> 2^k x 2^k grid)
    dv_layers=9,            # dv generator: layers
    dv_reupload=False,      # dv generator: re-encode z before every layer (data re-uploading)
    dv_encoding='ry',       # dv generator: 'ry' | 'ryrz' angle encoding of z
    objective='wgan_gp',    # 'wgan_gp' (critic) | 'mmd' (Born machine: fixed-kernel MMD, no critic)
    mmd_kernel='density',   # mmd only: 'density' (family-level) | 'sample' (Liu-Wang, grid points)
    mmd_bandwidths='0.5,1,2',  # mmd only: bandwidth multipliers of the pre-registered kernel
    mmd_real_batch=32,      # mmd only: real family members per generator step
    mmd_shots=0,            # mmd only: homodyne outcomes per generated density in the loss (0 = exact)
):
    """
    Train 2D CV-QGAN with pre-generated dataset.

    Key features:
    1. Pre-generate 500 distributions (400 train, 100 validation)
    2. Optional true batching via the SF TF backend (batch_size > 1)
    3. TRUE QGAN with latent vector input
    4. Validation on held-out set
    5. Show ACTUAL training samples in visualizations
    """

    # Setup
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./logs/qgan_2d_{family_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Seed EVERYTHING before any dataset generation, weight init, or z sampling.
    seed = resolve_seed(seed)
    _t_start = _time.time()   # wall clock for backend_info.json (no RNG use)
    seed_everything(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()  # TF 2.8+
        except Exception:
            pass
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    # Determinism provenance block (printed on every entry path, right after seeding).
    print("-" * 70)
    print("Determinism / reproducibility")
    print(f"  Seed: {seed}")
    print(f"  TF: {tf.__version__} | NumPy: {np.__version__} | SF: {sf.__version__}")
    print(f"  --deterministic: {deterministic} | "
          f"TF_DETERMINISTIC_OPS={os.environ.get('TF_DETERMINISTIC_OPS')} | "
          f"TF_ENABLE_ONEDNN_OPTS={os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
    print("  Note: seeding gives same-machine reproducibility; cross-machine "
          "(CPU ISA / BLAS / oneDNN / lib versions) can still drift.")
    print("-" * 70)

    n_ancilla = n_total_modes - 2

    # Memory warning for large Fock spaces
    fock_states = cutoff_dim ** n_total_modes
    if fock_states > 5000:
        rec = recommend_cutoff(n_total_modes, cutoff_dim)
        print(f"WARNING: cutoff={cutoff_dim} with {n_total_modes} modes = "
              f"{fock_states} Fock states. Recommended cutoff: {rec}")

    print("=" * 70)
    print("2D CV-QGAN with Pre-Generated Dataset")
    print("=" * 70)
    print(f"Family: {family_name}")
    print(f"Dataset: {n_train} train, {n_val} validation")
    print(f"Modes: {n_total_modes} total (2 output + {n_ancilla} ancilla)")
    print(f"Layers: {n_layers}, Cutoff: {cutoff_dim}, Kerr: {use_kerr}")
    print(f"Fock space: {fock_states} states")
    print(f"Grid: {grid_size}x{grid_size}, Range: [-{x_range}, {x_range}]")
    print(f"Learning rates - G: {g_lr}, D: {d_lr}")
    print(f"D steps per G step: {n_critic}")
    print(f"Batch size: {batch_size}")
    print(f"Discriminator hidden dims: [16, 8], dropout={d_dropout}")
    print(f"Instance noise: std={instance_noise}, anneal over {noise_anneal} "
          f"epochs to floor {noise_floor}")
    print(f"Critic blur sigma: {critic_blur_sigma} cells")
    print(f"Supervised weight: {supervised_weight} (warmup decay over {supervised_warmup} epochs)")
    print(f"Gradient penalty: weight={gp_weight}, warmup={gp_warmup} epochs")
    print(f"Ket-norm penalty weight: {ket_penalty_weight}")
    print(f"Latent scale: {latent_scale}")
    print(f"G gradient clip: {g_grad_clip if g_grad_clip > 0 else 'disabled'}")
    print(f"Encoding: {encoding}"
          + (f" (inj_layers={inj_layers}, r_max={r_max}, "
             f"squeeze_scale={squeeze_scale}, "
             f"train_squeeze_scale={train_squeeze_scale}, "
             f"inj_latent={inj_latent})"
             if encoding == 'interleaved_squeeze' else ""))
    if (encoding == 'interleaved_squeeze' and not train_squeeze_scale
            and float(squeeze_scale) == 0.0):
        print("NOTE: squeeze_scale=0 with a fixed gain: the z-squeeze channel "
              "is OFF — this run is numerically the input_displacement "
              "baseline (control arm). Use --squeeze-scale S for a fixed "
              "gain or --train-squeeze-scale for a learned one.")
    print(f"Seed: {seed}")
    print(f"Output: {log_dir}")
    print("=" * 70)

    # Get distribution family
    family = get_family(family_name, grid_size, x_range)
    xvec = family.xvec
    yvec = family.yvec
    X, Y = family.X, family.Y

    # PRE-GENERATE DATASET (KEY CHANGE)
    print("\n" + "=" * 70)
    full_dataset = generate_dataset(family, n_train + n_val)
    train_set = full_dataset[:n_train]
    val_set = full_dataset[n_train:]
    print(f"Split: {len(train_set)} train, {len(val_set)} validation")
    # Fingerprint of the validation set (recorded in backend_info.json so
    # scripts/rescore.py can assert it rebuilt the same split).
    import hashlib as _hashlib
    _val_hash = _hashlib.sha1(np.concatenate(
        [np.asarray(m, np.float32).ravel() for m, _ in val_set]).tobytes()).hexdigest()[:16]

    # Reference diversity of the data itself: the generator's 'diversity'
    # metric should approach this value, not zero.
    _idx = np.random.choice(len(val_set), size=(50, 2))
    _pairs = [(i, j) for i, j in _idx if i != j]
    data_diversity = np.mean([
        compute_wasserstein_2d(val_set[i][0], val_set[j][0])
        for i, j in _pairs]) if _pairs else 0.0
    print(f"Validation-set self-diversity (mean pairwise W1): {data_diversity:.4f}")
    print("=" * 70)

    # Get canonical target for reference
    canonical_target, _ = family.get_canonical()

    # Full-2D metric (energy distance): precompute the grid distance matrix
    # and per-val-member products so nearest-member search is one matvec.
    print("Precomputing energy-distance context...")
    ed_ctx = build_energy_distance_context(X, Y, val_set, canonical_target)

    # Initialize generator
    print("\nInitializing generator...")
    # SEAM 1: build the encoding strategy. Passing an explicit
    # EncodingSpec('input_displacement') is identical to the generator's
    # None default, so the legacy path is untouched.
    if isinstance(inj_layers, str):
        inj_layers = tuple(int(s) for s in inj_layers.split(',') if s.strip())
    if encoding == 'input_displacement':
        enc_spec = EncodingSpec(encoding)
    else:
        enc_spec = EncodingSpec(encoding, inj_layers=inj_layers, r_max=r_max,
                                squeeze_scale=squeeze_scale,
                                train_squeeze_scale=train_squeeze_scale,
                                inj_latent=inj_latent)
    if generator == 'cv':
        generator = CVQGANGenerator(
            n_modes=n_total_modes,
            n_output_modes=2,
            n_layers=n_layers,
            cutoff_dim=cutoff_dim,
            use_kerr=use_kerr,
            latent_scale=latent_scale,
            batch_size=batch_size,
            encoding=enc_spec,
        )
    else:
        # Benchmark backend (src/backends) behind the CV-shaped shim; the
        # same latent dimension as the CV generator of this config.
        from src.backends.trainer_shim import GeneratorShim, build_generator
        # latent_dim follows the CV generator OF THIS CONFIG (2*n_modes, or
        # 3*n_modes under --inj-latent dedicated): matched controls draw the
        # same number of normals per sample.
        generator = GeneratorShim(build_generator({
            'generator': generator, 'n_modes': n_total_modes,
            'latent_dim': enc_spec.latent_dim(n_total_modes),
            'gen_hidden': gen_hidden, 'gen_activation': gen_activation,
            'gen_fourier': gen_fourier, 'seed': seed, 'dv_qubits': dv_qubits,
            'dv_layers': dv_layers, 'dv_reupload': dv_reupload,
            'dv_encoding': dv_encoding, 'latent_scale': latent_scale}))
        print(f"Generator backend: {generator.describe()}")
        if generator.batch_size is None and hasattr(generator.backend, 'grid_sizes') \
                and grid_size not in generator.backend.grid_sizes:
            raise ValueError(f"--generator {generator.backend.name} produces grids "
                             f"{generator.backend.grid_sizes}; pass --grid-size accordingly")

    # Warm start (both modes; a legacy v1 file into an interleaved model
    # leaves w_s at zero = frozen-baseline start).
    if init_weights:
        print(f"Warm start from {init_weights}")
        if isinstance(generator, CVQGANGenerator):
            weights_io.apply_weights(generator, weights_io.load_weights(init_weights),
                                     source=init_weights)
        else:
            generator.load(init_weights)

    # Initialize discriminator
    print("\nInitializing discriminator...")
    if critic == 'dense_2d':
        discriminator = Discriminator2D(hidden_dims=[16, 8], dropout_rate=d_dropout)
    else:
        from src.backends.trainer_shim import build_critic
        discriminator = build_critic({'critic': critic, 'critic_points': critic_points,
                                      'critic_layers': critic_layers,
                                      'critic_cutoff': cutoff_dim},
                                     xvec, yvec, seed=seed if seed is not None else 0)
        print(f"Critic backend: {discriminator.describe()}")

    # Fixed Gaussian kernel for critic-input blur (None = off)
    blur_kernel = build_blur_kernel(critic_blur_sigma)

    # Optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.5, beta_2=0.9)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.5, beta_2=0.9)

    # History tracking
    history = {
        'g_loss': [], 'd_loss': [], 'wasserstein': [], 'val_wasserstein': [],
        'val_nearest_w1': [], 'val_diversity': [],  # family-aware validation (vs canonical-only)
        'val_canonical_ed': [], 'val_nearest_ed': [],  # full-2D energy distance (x/y units)
        'g_grad_norm': [], 'd_grad_norm': [],
        'supervised_loss': [], 'adversarial_loss': [],
        'gp_value': [], 'gp_weight_current': [],  # Gradient penalty tracking
        'd_real_score': [], 'd_fake_score': [],    # Discriminator score tracking
        'ket_norm': [],                             # Ket normalization (should be ~1.0)
        'supervised_weight_current': [],            # Current supervised weight after decay
        'instance_noise_current': [],               # Current instance noise std
    }

    best_val_metric = float('inf')
    best_weights = None
    best_w_s = None

    # Initial validation
    print("\nComputing initial validation...")
    init_val = validate(generator, canonical_target, val_set, xvec, yvec, ed_ctx)
    print(f"Initial validation: canonW1 {init_val['canonical_w1']:.4f} | "
          f"nearW1 {init_val['nearest_w1']:.4f} | div {init_val['diversity']:.4f} | "
          f"nearED {init_val['nearest_ed']:.4f}")

    # Plot initial state
    print("\nGenerating initial distribution...")
    try:
        z_init = tf.zeros([generator.latent_dim])
        init_prob = generator.generate_distribution_2d(z_init, xvec, yvec).numpy()
        plot_comparison(canonical_target, init_prob, X, Y,
                       f"{log_dir}/epoch_000.png", "Epoch 0 (Initial)")
        print("Initial plot saved.")
    except Exception as e:
        print(f"Warning: Initial generation failed: {e}")

    # Training loop
    print("\n" + "-" * 70)
    print("Training...")
    print("-" * 70)

    # =========================================================================
    # PATH A DETECTION (pure supervised representability test)
    # -------------------------------------------------------------------------
    # When the caller sets supervised_weight >= 1.0 AND supervised_warmup >= epochs
    # (i.e. the supervised weight never decays below 1.0), we interpret the run as
    # a PURE SUPERVISED representability experiment:
    #   * the discriminator is not trained at all (no adversarial machinery),
    #   * the generator fits ONE fixed target (the canonical family member),
    #   * the generator objective is a scale-free KL divergence.
    # This cleanly answers "can a CV-QNN at this cutoff represent this target?"
    # without any GAN dynamics confounding the result.
    # =========================================================================
    pure_supervised = (supervised_weight >= 1.0 and supervised_warmup >= epochs)
    fixed_real_tf = tf.constant(canonical_target, dtype=tf.float32)
    if pure_supervised:
        print("\n" + "*" * 70)
        print("PATH A MODE: pure supervised representability test")
        print("  - discriminator DISABLED (no adversarial training)")
        print("  - target FIXED to canonical family member")
        print("  - generator objective: KL(target || generated)")
        print("*" * 70)

    # SEAM 2: pick the trainer for the detected objective (current behavior
    # only; future MMD/energy-distance trainers plug in here).
    if objective not in ('wgan_gp', 'mmd'):
        raise ValueError(f"objective must be 'wgan_gp' or 'mmd', got {objective!r}")
    if objective == 'mmd' and pure_supervised:
        raise ValueError("--objective mmd conflicts with the pure-supervised settings "
                         "(supervised_weight >= 1 and supervised_warmup >= epochs): "
                         "the MMD objective would be silently ignored")
    trainer_cls = (PureSupervisedTrainer if pure_supervised
                   else MMDTrainer if objective == 'mmd' else WGANGPTrainer)
    mmd_kw = {}
    if trainer_cls is MMDTrainer:
        bws = (tuple(float(s) for s in mmd_bandwidths.split(',') if s.strip())
               if isinstance(mmd_bandwidths, str) else tuple(mmd_bandwidths))
        mmd_kw = dict(mmd_kernel=mmd_kernel, mmd_bandwidths=bws, mmd_real_batch=mmd_real_batch,
                      mmd_shots=mmd_shots)
        print(f"Objective: MMD ({mmd_kernel} kernel, bandwidths {bws}, "
              f"{mmd_real_batch} real members per step, "
              f"{'exact densities' if not mmd_shots else f'{mmd_shots} shots per density'}); "
              f"the critic is not trained")
    trainer = trainer_cls(
        **mmd_kw,
        generator=generator, discriminator=discriminator,
        g_optimizer=g_optimizer, d_optimizer=d_optimizer,
        train_set=train_set, fixed_real_tf=fixed_real_tf,
        blur_kernel=blur_kernel, xvec=xvec, yvec=yvec,
        batch_size=batch_size, n_critic=n_critic,
        gp_weight=gp_weight, gp_warmup=gp_warmup,
        instance_noise=instance_noise, noise_anneal=noise_anneal,
        noise_floor=noise_floor,
        supervised_weight=supervised_weight, supervised_warmup=supervised_warmup,
        ket_penalty_weight=ket_penalty_weight, g_grad_clip=g_grad_clip,
        norm_floor=norm_floor,
    )

    # Per-sample ket-norm monitor — separate file next to history.npz
    # (history.npz keys are frozen by the E2E goldens' strict key match).
    norm_mon = {'ket_norm_min': [], 'ket_norm_mean': [], 'ket_norm_max': [],
                'warn_count': [], 'worst_z': []}

    for epoch in range(1, epochs + 1):

        d_stats = trainer.discriminator_epoch(epoch)
        step = trainer.generator_epoch(epoch)
        if step is trainer.STOP:
            break
        if step is None:
            continue

        # === METRICS ===
        gen_np = step['gen_prob'].numpy()[0]  # first batch element for the cheap per-epoch metric
        w_dist = compute_wasserstein_2d(canonical_target, gen_np)

        history['g_loss'].append(float(step['g_loss']))
        history['d_loss'].append(d_stats['d_loss_avg'])
        history['wasserstein'].append(w_dist)
        history['g_grad_norm'].append(step['g_grad_norm'])
        history['d_grad_norm'].append(d_stats['d_grad_avg'])
        history['supervised_loss'].append(float(step['supervised_loss']))
        history['adversarial_loss'].append(float(step['adversarial_loss']))
        history['gp_value'].append(d_stats['gp_avg'])
        history['gp_weight_current'].append(d_stats['current_gp_weight'])
        history['d_real_score'].append(
            np.mean(d_stats['d_real_scores']) if d_stats['d_real_scores'] else 0.0)
        history['d_fake_score'].append(
            np.mean(d_stats['d_fake_scores']) if d_stats['d_fake_scores'] else 0.0)
        history['supervised_weight_current'].append(step['current_sw'])
        history['instance_noise_current'].append(d_stats['current_noise'])

        # Ket norm diagnostic — recorded every epoch (free: same tensor used in G loss).
        # Print a warning at val_every cadence so logs don't get spammy.
        ket_norm = float(tf.reduce_mean(step['ket_norm_tf']))
        history['ket_norm'].append(ket_norm)
        norm_mon['ket_norm_min'].append(step['ket_norm_min'])
        norm_mon['ket_norm_mean'].append(ket_norm)
        norm_mon['ket_norm_max'].append(step['ket_norm_max'])
        norm_mon['warn_count'].append(step['ket_norm_warn'])
        norm_mon['worst_z'].append(step['worst_z'])
        if epoch % val_every == 0:
            flag = "  <-- LOW (raise cutoff_dim)" if ket_norm < 0.9 else ""
            print(f"  ket_norm = {ket_norm:.3f}{flag}")

        # === VALIDATION ===
        if epoch % val_every == 0:
            val_metrics = validate(generator, canonical_target, val_set,
                                   xvec, yvec, ed_ctx)
            history['val_wasserstein'].append(val_metrics['canonical_w1'])
            history['val_nearest_w1'].append(val_metrics['nearest_w1'])
            history['val_diversity'].append(val_metrics['diversity'])
            history['val_canonical_ed'].append(val_metrics['canonical_ed'])
            history['val_nearest_ed'].append(val_metrics['nearest_ed'])

            # Checkpoint selection:
            #   GAN mode: nearest-member W1 (doesn't reward collapse-to-canonical)
            #   Path A:   canonical W1 (the objective IS the canonical member;
            #             keeps runs comparable with validated Path A results)
            select_metric = (val_metrics['canonical_w1'] if pure_supervised
                             else val_metrics['nearest_w1'])
            if select_metric < best_val_metric:
                best_val_metric = select_metric
                # Snapshot weights AND w_s at the same epoch — never pair
                # best-epoch weights with w_s from another epoch.
                if isinstance(generator, CVQGANGenerator):
                    best_weights = generator.weights.numpy().copy()
                    best_w_s = (generator.w_s.numpy().copy()
                                if generator.w_s is not None else None)
                else:
                    best_weights = generator.state_dict()   # backend state
                    best_w_s = None

            print(f"Epoch {epoch:4d} | G: {float(step['g_loss']):.4f} | "
                  f"D: {d_stats['d_loss_avg']:+.4f} | "
                  f"nearW1: {val_metrics['nearest_w1']:.4f} | "
                  f"canonW1: {val_metrics['canonical_w1']:.4f} | "
                  f"div: {val_metrics['diversity']:.4f} | "
                  f"nearED: {val_metrics['nearest_ed']:.4f} | Best: {best_val_metric:.4f}")
        else:
            # Regular logging
            if epoch % 10 == 0 or epoch == 1:
                d_real_avg = (np.mean(d_stats['d_real_scores'])
                              if d_stats['d_real_scores'] else 0.0)
                d_fake_avg = (np.mean(d_stats['d_fake_scores'])
                              if d_stats['d_fake_scores'] else 0.0)
                print(f"Epoch {epoch:4d} | G: {float(step['g_loss']):.4f} | "
                      f"D: {d_stats['d_loss_avg']:+.4f} | "
                      f"W1: {w_dist:.4f} | D(r):{d_real_avg:+.2f} D(f):{d_fake_avg:+.2f} | "
                      f"SW:{step['current_sw']:.2f} noise:{d_stats['current_noise']:.3f}")

        # === VISUALIZATION ===
        if epoch % plot_every == 0:
            # KEY FIX: Show ACTUAL training sample, not canonical
            sample_idx = np.random.randint(len(train_set))
            sample_real, _ = train_set[sample_idx]

            z_sample = tf.random.normal([generator.latent_dim])
            sample_fake = generator.generate_distribution_2d(z_sample, xvec, yvec).numpy()

            plot_comparison(
                sample_real, sample_fake, X, Y,
                f"{log_dir}/epoch_{epoch:04d}_sample_{sample_idx}.png",
                f"Epoch {epoch} - Train Sample #{sample_idx} (W1={w_dist:.4f})"
            )

    # Final results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    select_name = 'canonical W1' if pure_supervised else 'nearest-member W1'
    print(f"Best validation ({select_name}): {best_val_metric:.4f}")

    # Restore best weights
    if best_weights is not None and isinstance(generator, CVQGANGenerator):
        print("Restoring best weights from validation...")
        generator.weights.assign(best_weights)
        if best_w_s is not None:
            generator.w_s.assign(best_w_s)
        # Persist for post-hoc re-scoring (e.g. with metrics added later);
        # history.npz alone cannot reconstruct the generator. v1 raw array
        # for input_displacement (byte-identical to the legacy np.save),
        # v2 dict with w_s for interleaved_squeeze.
        weights_io.save_weights(
            f"{log_dir}/best_weights.npy", best_weights, w_s=best_w_s,
            meta=({'squeeze_scale': float(squeeze_scale),
                   'train_squeeze_scale': bool(train_squeeze_scale),
                   'inj_latent': inj_latent}
                  if best_w_s is not None else None))
    elif best_weights is not None:
        print("Restoring best backend state from validation...")
        from src.backends.trainer_shim import save_backend_state
        generator.load_state_dict(best_weights)
        save_backend_state(f"{log_dir}/best_weights.npy", generator.backend,
                           state=best_weights)

    # Additive provenance for the benchmark: what was trained, parameter and
    # Hilbert-space sizes, wall time (backend_info.json next to run_config).
    try:
        from src.backends.trainer_shim import write_backend_info
        from src.backends.cv import CVGeneratorBackend
        from src.backends.classical_critic import DenseCriticBackend
        g_desc = (CVGeneratorBackend(generator=generator).describe()
                  if isinstance(generator, CVQGANGenerator) else generator.describe())
        c_desc = (DenseCriticBackend(model=discriminator).describe()
                  if isinstance(discriminator, Discriminator2D) else discriminator.describe())
        write_backend_info(log_dir, g_desc, c_desc, _time.time() - _t_start,
                           extra={'epochs_run': int(len(history['g_loss'])),
                                  'val_set_sha1_16': _val_hash})
    except Exception as e:  # noqa: BLE001 - provenance must never fail a run
        print(f"  (backend_info.json not written: {e})")

    # Final validation
    final_val = validate(generator, canonical_target, val_set, xvec, yvec, ed_ctx)
    print(f"Final validation: canonW1 {final_val['canonical_w1']:.4f} | "
          f"nearW1 {final_val['nearest_w1']:.4f} | div {final_val['diversity']:.4f} | "
          f"nearED {final_val['nearest_ed']:.4f}")

    # Generate final samples with different z values
    print("\nGenerating final samples with different latent vectors...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # First row: target
    im = axes[0, 0].contourf(X, Y, canonical_target, levels=30, cmap='viridis')
    axes[0, 0].set_title('Target (Canonical)')
    plt.colorbar(im, ax=axes[0, 0])

    # Rest: generated with different z
    for i in range(7):
        row = (i + 1) // 4
        col = (i + 1) % 4

        z = tf.random.normal([generator.latent_dim])
        gen_prob = generator.generate_distribution_2d(z, xvec, yvec).numpy()

        im = axes[row, col].contourf(X, Y, gen_prob, levels=30, cmap='viridis')
        axes[row, col].set_title(f'Generated (z #{i+1})')
        plt.colorbar(im, ax=axes[row, col])

    plt.tight_layout()
    plt.savefig(f"{log_dir}/final_samples.png", dpi=150)
    plt.close()

    # Save training history
    plot_training_history(history, f"{log_dir}/training_history.png")
    np.savez(f"{log_dir}/history.npz", **history, seed=np.int64(seed))
    np.savez(f"{log_dir}/norm_monitor.npz",
             ket_norm_min=np.array(norm_mon['ket_norm_min']),
             ket_norm_mean=np.array(norm_mon['ket_norm_mean']),
             ket_norm_max=np.array(norm_mon['ket_norm_max']),
             warn_count=np.array(norm_mon['warn_count'], dtype=np.int64),
             worst_z=np.array(norm_mon['worst_z']),
             norm_floor=np.float64(norm_floor))
    # CV quantum critic: its own Fock-truncation monitor (separate file; the
    # critic score is not renormalised, so this is the evidence of whether a
    # trained CV critic still evaluates inside its cutoff).
    if getattr(discriminator, 'ket_norm_log', None):
        kl = np.array(discriminator.ket_norm_log, dtype=np.float64)
        np.savez(f"{log_dir}/critic_norm_monitor.npz", ket_norm_min=kl[:, 0], ket_norm_mean=kl[:, 1])

    print(f"\nResults saved to: {log_dir}")

    return generator, history
