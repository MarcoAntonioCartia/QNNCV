#!/usr/bin/env python
"""
Unified 2D CV-QGAN with Pre-Generated Dataset
==============================================

TRUE QGAN Implementation with these key improvements:
1. Pre-generate 500 distributions (400 train, 100 validation)
2. Generator receives RANDOM latent vectors z as input
3. z is encoded into the circuit via displacement gates
4. Optional true batching via the SF TF backend (--batch-size; 1 = sequential)
5. Visualizations show ACTUAL training samples, not just canonical
6. Validation on held-out set for generalization metrics

Architecture (per Killoran et al.):
    ENCODING: Dgate(z0, z1)|q[0], Dgate(z2, z3)|q[1], ...  # Latent -> quantum state
    Then for each layer:
        U1(interferometer) -> S(squeeze) -> U2(interferometer) -> D(displacement) -> K(Kerr)

Supports N >= 2 total qumodes. Only the first 2 modes produce the output P(x,y).
Extra modes (ancilla) provide additional entanglement and expressivity via
beamsplitters, then are traced out (partial trace) at measurement time.

Author: QNNCV Project
Date: January 2026
"""

# =============================================================================
# Imports & Compatibility
# =============================================================================

# Compatibility patches
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import os
import sys
import argparse
import json
import random
import secrets

# Refactor strangler: the training code is being extracted into src/ (see
# tests/golden/). Hard `from src...` imports below need the repo root on
# sys.path even when this file is importlib-loaded from elsewhere (e.g.
# scripts/verify_batching.py runs with scripts/ as sys.path[0]).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Determinism env vars are read at TensorFlow *import* time, but argparse runs
# in main() long after the import below. Scan sys.argv here so that opt-in
# --deterministic runs take effect. Default (flag absent) leaves env untouched.
if '--deterministic' in sys.argv:
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import wasserstein_distance
from scipy.special import hermite
from math import factorial, sqrt, pi

# Warning suppression (if available)
try:
    from src.utils.warning_suppression import enable_clean_training
    enable_clean_training()
except ImportError:
    pass

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Hermite Basis for Fock -> Position transformation
# =============================================================================

from src.quantum.hermite import compute_hermite_basis, recommend_cutoff


# =============================================================================
# CV-QGAN Generator with Latent Input
# =============================================================================

class CVQGANGenerator:
    """
    CV-QNN Generator that takes RANDOM LATENT VECTORS as input.

    This is the TRUE GAN setup where:
    - Input: Random z vector (the "noise" in GAN terminology)
    - Output: 2D probability distribution
    - Training: Optimize circuit parameters to map z -> target distribution

    Architecture:
        1. ENCODING LAYER: Displace vacuum by latent vector z
           Dgate(z[0], z[1]) | q[0]  # Mode 0
           Dgate(z[2], z[3]) | q[1]  # Mode 1
           ... (for all n_modes modes)

        2. QNN LAYERS (Killoran architecture): For each layer l:
           - U1: Interferometer (beamsplitters + rotations)
           - S: Squeeze gates on all modes
           - U2: Interferometer
           - D: Displacement gates on all modes
           - K: Kerr gates on all modes (non-Gaussian activation)

        3. MEASUREMENT: Partial trace over ancilla modes, then P(x, y)
           For N > 2 modes, ancilla modes are traced out:
           P(x,y) = SUM_k |SUM_{n,m} c_{n,m,k} * phi_n(x) * phi_m(y)|^2

    The latent dimension is 2*n_modes (magnitude + phase for each mode).
    n_output_modes controls how many modes are measured (always 2 for 2D).
    Extra modes (n_modes - n_output_modes) serve as ancilla for entanglement.
    """

    def __init__(
        self,
        n_modes: int = 2,
        n_output_modes: int = 2,    # Modes to measure (always 2 for 2D output)
        n_layers: int = 6,
        cutoff_dim: int = 10,
        use_kerr: bool = True,
        latent_scale: float = 1.0,  # Scale for latent vector encoding
        active_sd: float = 0.1,     # Init std for active params (squeeze, displacement, kerr)
        passive_sd: float = 0.1,    # Init std for passive params (interferometer)
        batch_size=None,            # SF TF backend batching (None/1 = unbatched)
    ):
        self.n_modes = n_modes
        self.batch_size = batch_size if (batch_size and batch_size > 1) else None
        self.n_output_modes = n_output_modes
        self.n_ancilla = n_modes - n_output_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.use_kerr = use_kerr
        self.latent_scale = latent_scale

        # Latent dimension: magnitude + phase for each mode
        self.latent_dim = 2 * n_modes

        # Calculate parameters per layer
        self.interferometer_params = self._calc_interferometer_params(n_modes)

        # Per layer:
        # - 2 interferometers
        # - n_modes squeeze gates (2 params each: r, phi)
        # - n_modes displacement gates (2 params each: r, phi)
        # - n_modes Kerr gates (1 param each) if use_kerr
        self.params_per_layer = (
            2 * self.interferometer_params +  # U1 and U2
            2 * n_modes +                      # Squeeze (r, phi)
            2 * n_modes +                      # Displacement (r, phi)
            (n_modes if use_kerr else 0)       # Kerr
        )

        self.total_params = self.params_per_layer * n_layers

        # Initialize weights
        self.weights = self._init_weights(active_sd, passive_sd)

        # Create SF program and engine
        self.prog = sf.Program(n_modes)
        backend_options = {'cutoff_dim': cutoff_dim}
        if self.batch_size:
            # SF TF backend evaluates the whole batch in one vectorized run;
            # gradients flow through exactly as in the unbatched case.
            backend_options['batch_size'] = self.batch_size
        self.eng = sf.Engine('tf', backend_options=backend_options)

        # Create symbolic parameters for the circuit
        self._build_symbolic_circuit()

        print(f"CVQGANGenerator initialized:")
        print(f"  Total modes: {n_modes} ({n_output_modes} output + {self.n_ancilla} ancilla)")
        print(f"  Layers: {n_layers}")
        print(f"  Cutoff: {cutoff_dim}")
        print(f"  Fock space: {cutoff_dim ** n_modes} states")
        print(f"  Use Kerr: {use_kerr}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Total trainable params: {self.total_params}")
        print(f"  Batch size: {self.batch_size or 1}")

    def _calc_interferometer_params(self, n_modes):
        """Number of parameters for an N-mode interferometer."""
        if n_modes == 1:
            return 1  # Just a rotation
        # N(N-1)/2 BS angles + N(N-1)/2 BS phases + (N-1) rotations
        return n_modes * (n_modes - 1) + n_modes - 1

    def _init_weights(self, active_sd, passive_sd):
        """Initialize QNN weights with appropriate scales."""
        M = self.interferometer_params
        N = self.n_modes
        L = self.n_layers

        # Per layer: [int1 | squeeze | int2 | disp_r | disp_phi | kerr]
        all_weights = []

        for _ in range(L):
            # Interferometer 1 (passive)
            all_weights.append(tf.random.normal([M], stddev=passive_sd))
            # Squeeze magnitudes (active - start small)
            all_weights.append(tf.random.normal([N], stddev=active_sd))
            # Squeeze phases (passive)
            all_weights.append(tf.random.normal([N], stddev=passive_sd))
            # Interferometer 2 (passive)
            all_weights.append(tf.random.normal([M], stddev=passive_sd))
            # Displacement r (active)
            all_weights.append(tf.random.normal([N], stddev=active_sd))
            # Displacement phi (passive)
            all_weights.append(tf.random.normal([N], stddev=passive_sd))
            # Kerr (active - start very small)
            if self.use_kerr:
                all_weights.append(tf.random.normal([N], stddev=active_sd * 0.5))

        weights = tf.concat(all_weights, axis=0)
        return tf.Variable(weights, trainable=True, name='qnn_weights')

    def _build_symbolic_circuit(self):
        """Build the symbolic SF circuit with parameters."""
        self.prog = sf.Program(self.n_modes)

        # Create symbolic parameters for latent input
        self.latent_params = [self.prog.params(f"z_{i}") for i in range(self.latent_dim)]

        # Create symbolic parameters for QNN weights
        self.weight_params = [self.prog.params(f"w_{i}") for i in range(self.total_params)]

        with self.prog.context as q:
            # ENCODING LAYER: Apply latent vector as displacement
            for mode in range(self.n_modes):
                r_idx = 2 * mode
                phi_idx = 2 * mode + 1
                ops.Dgate(
                    self.latent_params[r_idx] * self.latent_scale,
                    self.latent_params[phi_idx]
                ) | q[mode]

            # QNN LAYERS
            w_idx = 0
            for layer in range(self.n_layers):
                # Interferometer 1
                w_idx = self._apply_interferometer(q, w_idx)

                # Squeeze gates (magnitude + phase)
                for mode in range(self.n_modes):
                    ops.Sgate(self.weight_params[w_idx], self.weight_params[w_idx + 1]) | q[mode]
                    w_idx += 2

                # Interferometer 2
                w_idx = self._apply_interferometer(q, w_idx)

                # Displacement gates
                for mode in range(self.n_modes):
                    ops.Dgate(
                        self.weight_params[w_idx],
                        self.weight_params[w_idx + 1]
                    ) | q[mode]
                    w_idx += 2

                # Kerr gates
                if self.use_kerr:
                    for mode in range(self.n_modes):
                        ops.Kgate(self.weight_params[w_idx]) | q[mode]
                        w_idx += 1

    def _apply_interferometer(self, q, start_idx):
        """Apply parameterized interferometer to modes."""
        N = self.n_modes

        if N == 1:
            # Single mode: just a rotation
            ops.Rgate(self.weight_params[start_idx]) | q[0]
            return start_idx + 1

        # Multi-mode: rectangular beamsplitter array
        n_bs = N * (N - 1) // 2

        # BS angles
        theta_params = self.weight_params[start_idx:start_idx + n_bs]
        # BS phases
        phi_params = self.weight_params[start_idx + n_bs:start_idx + 2*n_bs]
        # Final rotations
        rphi_params = self.weight_params[start_idx + 2*n_bs:start_idx + 2*n_bs + N - 1]

        n = 0
        for layer in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (layer + k) % 2 != 1:
                    ops.BSgate(theta_params[n], phi_params[n]) | (q1, q2)
                    n += 1

        for i in range(N - 1):
            ops.Rgate(rphi_params[i]) | q[i]

        return start_idx + self.interferometer_params

    def _run_circuit(self, z):
        """Run the circuit for one latent vector (latent_dim,) or, in
        batched-engine mode, a batch (batch_size, latent_dim).

        Scalar parameters broadcast across the batch, so single-z calls
        also work on a batched engine (all batch elements identical)."""
        if self.eng.run_progs:
            self.eng.reset()

        # Build parameter mapping
        mapping = {}
        batched_z = (len(z.shape) == 2)

        # Map latent vector(s): one (batch_size,) tensor per parameter when batched
        for i, param in enumerate(self.latent_params):
            mapping[param.name] = z[:, i] if batched_z else z[i]

        # Map QNN weights (scalars; broadcast over the batch)
        for i, param in enumerate(self.weight_params):
            mapping[param.name] = self.weights[i]

        # Run circuit
        result = self.eng.run(self.prog, args=mapping)
        return result.state

    def generate_distribution_2d(self, z, xvec, yvec, return_ket_norm=False):
        """
        Generate 2D distribution for a single latent vector.

        For n_modes == 2 (no ancilla):
            P(x,y) = |Σ_nm c_nm φ_n(x) φ_m(y)|²

        For n_modes > 2 (with ancilla, partial trace):
            P(x,y) = Σ_k |Σ_nm c_{nm,k} φ_n(x) φ_m(y)|²
            where k indexes all ancilla Fock states (traced out).

        Args:
            z: Latent vector [latent_dim]
            xvec: x-axis grid points
            yvec: y-axis grid points
            return_ket_norm: if True, also return the differentiable ket norm

        Returns:
            prob: (len(xvec), len(yvec)) joint probability distribution
            ket_norm (optional): scalar tf.Tensor, sum |ket|^2 inside the Fock cutoff
        """
        # Get quantum state
        state = self._run_circuit(z)
        ket = state.ket()  # shape: (cutoff,) * n_modes
        if self.batch_size:
            # A batched engine broadcasts the single z to identical batch
            # elements; keep one.
            ket = ket[0]

        # Compute Hermite basis for the 2 output modes
        hermite_x = compute_hermite_basis(xvec, self.cutoff_dim)
        hermite_y = compute_hermite_basis(yvec, self.cutoff_dim)

        ket_c = tf.cast(ket, tf.complex64)
        hx_c = tf.cast(hermite_x, tf.complex64)
        hy_c = tf.cast(hermite_y, tf.complex64)

        # Differentiable ket norm: |ket|^2 summed over all Fock indices.
        # = 1.0 if cutoff captures the full state; < 1.0 means truncation.
        ket_norm = tf.reduce_sum(tf.abs(ket_c) ** 2)

        if self.n_ancilla == 0:
            # Original 2-mode path (no ancilla) -- no partial trace needed
            # ψ(x,y) = Σ_nm c_nm φ_n(x) φ_m(y)
            psi = tf.einsum('nm,xn,ym->xy', ket_c, hx_c, hy_c)
            prob = tf.abs(psi) ** 2
        else:
            # Partial trace over ancilla modes
            # ket has shape (c, c, c, ...) with n_modes dimensions
            # Reshape to (c, c, K) where K = c^n_ancilla
            c = self.cutoff_dim
            K = c ** self.n_ancilla
            ket_reshaped = tf.reshape(ket_c, [c, c, K])

            # For each ancilla config k, compute ψ_k(x,y):
            #   psi_all[x, y, k] = Σ_nm c_{nm,k} * φ_n(x) * φ_m(y)
            psi_all = tf.einsum('nmk,xn,ym->xyk', ket_reshaped, hx_c, hy_c)

            # Partial trace: P(x,y) = Σ_k |ψ_k(x,y)|²
            prob = tf.reduce_sum(tf.abs(psi_all) ** 2, axis=-1)

        # Normalize
        prob = prob / (tf.reduce_sum(prob) + 1e-10)

        if return_ket_norm:
            return prob, ket_norm
        return prob

    def generate_batch(self, z_batch, xvec, yvec, return_ket_norm=False):
        """
        Generate distributions for a batch of latent vectors.

        Args:
            z_batch: (B, latent_dim) latent vectors. When the engine is
                batched, B must equal self.batch_size (one vectorized SF
                run); otherwise circuits run sequentially and are stacked.
            xvec, yvec: grid points
            return_ket_norm: if True, also return (B,) ket norms

        Returns:
            probs: (B, len(xvec), len(yvec)); optionally ket_norms (B,)
        """
        n = int(z_batch.shape[0])
        if self.batch_size and n == self.batch_size:
            return self._generate_batched(z_batch, xvec, yvec, return_ket_norm)

        # Sequential fallback (also the batch_size=1 legacy path)
        probs, norms = [], []
        for b in range(n):
            p, kn = self.generate_distribution_2d(
                z_batch[b], xvec, yvec, return_ket_norm=True)
            probs.append(p)
            norms.append(kn)
        probs = tf.stack(probs)
        norms = tf.stack(norms)
        if return_ket_norm:
            return probs, norms
        return probs

    def _generate_batched(self, z_batch, xvec, yvec, return_ket_norm=False):
        """Batched-engine path: one SF run for the whole batch."""
        state = self._run_circuit(z_batch)
        ket = state.ket()  # (batch_size,) + (cutoff,) * n_modes

        hermite_x = compute_hermite_basis(xvec, self.cutoff_dim)
        hermite_y = compute_hermite_basis(yvec, self.cutoff_dim)

        ket_c = tf.cast(ket, tf.complex64)
        hx_c = tf.cast(hermite_x, tf.complex64)
        hy_c = tf.cast(hermite_y, tf.complex64)

        # Per-element ket norm: sum |ket|^2 over all Fock indices -> (B,)
        ket_norm = tf.reduce_sum(
            tf.abs(ket_c) ** 2, axis=list(range(1, self.n_modes + 1)))

        if self.n_ancilla == 0:
            psi = tf.einsum('bnm,xn,ym->bxy', ket_c, hx_c, hy_c)
            prob = tf.abs(psi) ** 2
        else:
            # Partial trace over ancilla modes, batched
            c = self.cutoff_dim
            K = c ** self.n_ancilla
            ket_reshaped = tf.reshape(ket_c, [self.batch_size, c, c, K])
            psi_all = tf.einsum('bnmk,xn,ym->bxyk', ket_reshaped, hx_c, hy_c)
            prob = tf.reduce_sum(tf.abs(psi_all) ** 2, axis=-1)

        # Normalize each batch element separately
        prob = prob / (tf.reduce_sum(prob, axis=[1, 2], keepdims=True) + 1e-10)

        if return_ket_norm:
            return prob, ket_norm
        return prob

    def get_ket_norm(self, z):
        """Compute the norm of the ket state (should be ~1.0 if cutoff is sufficient)."""
        state = self._run_circuit(z)
        ket = state.ket()
        if self.batch_size:
            ket = ket[0]
        return float(tf.reduce_sum(tf.abs(tf.cast(ket, tf.complex64)) ** 2))

    @property
    def trainable_variables(self):
        return [self.weights]

    def get_config(self):
        return {
            'n_modes': self.n_modes,
            'n_output_modes': self.n_output_modes,
            'n_ancilla': self.n_ancilla,
            'n_layers': self.n_layers,
            'cutoff_dim': self.cutoff_dim,
            'use_kerr': self.use_kerr,
            'latent_dim': self.latent_dim,
            'total_params': self.total_params,
        }


# =============================================================================
# Discriminator
# =============================================================================

class Discriminator2D(tf.keras.Model):
    """
    Discriminator for 2D probability distributions.

    Takes a 2D distribution and outputs a score (real vs fake).

    IMPORTANT: Must be kept deliberately weak to avoid overwhelming
    the quantum generator (~96 params). Even [16, 8] has ~25k params
    due to the 1600-dim input (40x40 grid).
    """

    def __init__(self, hidden_dims=[16, 8], init_scale=0.05, dropout_rate=0.3):
        super().__init__()

        initializer = tf.keras.initializers.RandomNormal(stddev=init_scale)

        self.flatten = tf.keras.layers.Flatten()

        layers = []
        for dim in hidden_dims:
            layers.append(tf.keras.layers.Dense(dim, kernel_initializer=initializer))
            layers.append(tf.keras.layers.LayerNormalization())
            layers.append(tf.keras.layers.LeakyReLU(0.2))
            if dropout_rate > 0:
                layers.append(tf.keras.layers.Dropout(dropout_rate))

        layers.append(tf.keras.layers.Dense(1, kernel_initializer=initializer))

        self.net = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.flatten(x)
        return self.net(x, training=training)


# =============================================================================
# Distribution Families (for training data)
# =============================================================================

class DistributionFamily:
    """Base class for distribution families."""

    def __init__(self, grid_size, x_range):
        self.grid_size = grid_size
        self.x_range = x_range
        self.xvec = np.linspace(-x_range, x_range, grid_size)
        self.yvec = np.linspace(-x_range, x_range, grid_size)
        self.X, self.Y = np.meshgrid(self.xvec, self.yvec)

    def sample(self):
        """Sample a distribution from the family."""
        raise NotImplementedError

    def get_canonical(self):
        """Get the canonical member."""
        raise NotImplementedError


class GaussianFamily(DistributionFamily):
    """Independent 2D Gaussians with varying parameters."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=1.0, sigma_range=(0.3, 0.8)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.sigma_range = sigma_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        sigma = np.random.uniform(*self.sigma_range)

        dist = self._make_gaussian(cx, cy, sigma)
        params = {'cx': cx, 'cy': cy, 'sigma': sigma}
        return dist, params

    def get_canonical(self):
        sigma = (self.sigma_range[0] + self.sigma_range[1]) / 2
        dist = self._make_gaussian(0, 0, sigma)
        return dist, {'cx': 0, 'cy': 0, 'sigma': sigma}

    def _make_gaussian(self, cx, cy, sigma):
        dist = np.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2 * sigma**2))
        return dist / (dist.sum() + 1e-10)


class RingFamily(DistributionFamily):
    """Ring distributions with varying parameters."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=0.5, radius_range=(0.8, 1.5), width_range=(0.15, 0.35)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.radius_range = radius_range
        self.width_range = width_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        radius = np.random.uniform(*self.radius_range)
        width = np.random.uniform(*self.width_range)

        dist = self._make_ring(cx, cy, radius, width)
        params = {'cx': cx, 'cy': cy, 'radius': radius, 'width': width}
        return dist, params

    def get_canonical(self):
        radius = (self.radius_range[0] + self.radius_range[1]) / 2
        width = (self.width_range[0] + self.width_range[1]) / 2
        dist = self._make_ring(0, 0, radius, width)
        return dist, {'cx': 0, 'cy': 0, 'radius': radius, 'width': width}

    def _make_ring(self, cx, cy, radius, width):
        r = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
        dist = np.exp(-0.5 * ((r - radius) / width)**2)
        return dist / (dist.sum() + 1e-10)


class CorrelatedGaussianFamily(DistributionFamily):
    """Correlated 2D Gaussians."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=0.5, rho_range=(0.4, 0.85), scale_range=(0.6, 1.2)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.rho_range = rho_range
        self.scale_range = scale_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        rho = np.random.uniform(*self.rho_range)
        scale = np.random.uniform(*self.scale_range)

        dist = self._make_correlated(cx, cy, rho, scale)
        params = {'cx': cx, 'cy': cy, 'rho': rho, 'scale': scale}
        return dist, params

    def get_canonical(self):
        rho = (self.rho_range[0] + self.rho_range[1]) / 2
        scale = (self.scale_range[0] + self.scale_range[1]) / 2
        dist = self._make_correlated(0, 0, rho, scale)
        return dist, {'cx': 0, 'cy': 0, 'rho': rho, 'scale': scale}

    def _make_correlated(self, cx, cy, rho, scale):
        cov = np.array([[1, rho], [rho, 1]]) * scale**2
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)

        dx = self.X - cx
        dy = self.Y - cy

        mahal = (cov_inv[0,0] * dx**2 +
                 2 * cov_inv[0,1] * dx * dy +
                 cov_inv[1,1] * dy**2)

        dist = np.exp(-0.5 * mahal) / (2 * np.pi * np.sqrt(det))
        return dist / (dist.sum() + 1e-10)


class FourGaussiansFamily(DistributionFamily):
    """Four Gaussians at corners of a square. Tests multi-modal 2D learning."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=0.3, spread_range=(1.0, 1.8), sigma_range=(0.2, 0.4)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.spread_range = spread_range
        self.sigma_range = sigma_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        spread = np.random.uniform(*self.spread_range)
        sigma = np.random.uniform(*self.sigma_range)

        dist = self._make_four_gaussians(cx, cy, spread, sigma)
        params = {'cx': cx, 'cy': cy, 'spread': spread, 'sigma': sigma}
        return dist, params

    def get_canonical(self):
        spread = (self.spread_range[0] + self.spread_range[1]) / 2
        sigma = (self.sigma_range[0] + self.sigma_range[1]) / 2
        dist = self._make_four_gaussians(0, 0, spread, sigma)
        return dist, {'cx': 0, 'cy': 0, 'spread': spread, 'sigma': sigma}

    def _make_four_gaussians(self, cx, cy, spread, sigma):
        """Four Gaussians at corners of a square centered at (cx, cy)."""
        corners = [
            (cx - spread, cy - spread),
            (cx - spread, cy + spread),
            (cx + spread, cy - spread),
            (cx + spread, cy + spread),
        ]

        dist = np.zeros_like(self.X)
        for gx, gy in corners:
            dist += np.exp(-((self.X - gx)**2 + (self.Y - gy)**2) / (2 * sigma**2))

        return dist / (dist.sum() + 1e-10)


class VibronicFamily(DistributionFamily):
    """2-mode position-quadrature marginal of a molecular vibronic Doktorov state.

    The Doktorov operator U = D(α)·Interferometer(U2)·Squeeze(r)·Interferometer(U1)
    is a sequence of Gaussian gates, so the resulting N-mode state is Gaussian and
    its 2-mode position marginal P(x_i, x_j) is an analytical bivariate Gaussian
    obtainable from the position sub-block of the full state covariance.

    Each family member is one (i, j) pair of normal modes from the molecule's
    Doktorov state. Canonical = the pair with largest combined |μ| (the
    spectroscopically "loudest" mode pair).

    Default molecule: Formic (14 modes, T=0). Also supports Water (3 modes) and
    Pyrrole (24 modes) -- but the time-resolved Water/Pyrrole datasets need
    additional handling not yet implemented here.
    """

    def __init__(self, grid_size=40, x_range=3.0,
                 molecule='formic', canonical_pair=None,
                 standardize=True, target_std=0.5):
        super().__init__(grid_size, x_range)

        # Lazy SF imports so non-vibronic runs don't pay the import cost
        from strawberryfields.apps.data import Formic
        from strawberryfields.apps.qchem.vibronic import gbs_params, VibronicTransition
        import strawberryfields as sf

        mol_classes = {'formic': Formic}
        # Water and Pyrrole are time-resolved; deferred for now.
        if molecule not in mol_classes:
            raise ValueError(
                f"Unknown molecule '{molecule}'. Currently supported: "
                f"{list(mol_classes.keys())}")
        mol = mol_classes[molecule]()
        self.molecule = molecule

        # Build Doktorov parameters from molecule spectroscopic data.
        # At T=0, the two-mode squeezing vector t is all zeros and unused here.
        # mol.modes can include thermal-pair modes (= 2 * normal modes); the
        # Doktorov state itself lives on the n normal modes that U1 acts on.
        _t, U1, r, U2, alpha = gbs_params(mol.w, mol.wp, mol.Ud, mol.delta, T=mol.T)
        self.n_modes = U1.shape[0]

        # Run the Gaussian program ONCE to get full (cov, means) of the
        # Doktorov state. At T=0, gbs_params returns t=0 so no two-mode squeezing.
        prog = sf.Program(self.n_modes)
        with prog.context as q:
            VibronicTransition(U1, r, U2, alpha) | q
        eng = sf.Engine('gaussian')
        state = eng.run(prog).state
        self.cov_full = state.cov()        # shape (2N, 2N), xxpp ordering
        self.means_full = state.means()    # shape (2N,),     xxpp ordering

        # Position-only sub-block: indices [0..N-1] of means and cov give the
        # x-quadrature marginal (xxpp ordering means first N are positions).
        self.means_x = self.means_full[:self.n_modes]
        self.cov_xx = self.cov_full[:self.n_modes, :self.n_modes]

        # Enumerate pairs, precompute marginals, rank by displacement
        self.pairs = []
        for i in range(self.n_modes):
            for j in range(i + 1, self.n_modes):
                mu = np.array([self.means_x[i], self.means_x[j]])
                V = np.array([[self.cov_xx[i, i], self.cov_xx[i, j]],
                              [self.cov_xx[j, i], self.cov_xx[j, j]]])
                self.pairs.append({'pair': (i, j), 'mu': mu, 'V': V,
                                   'norm_mu': float(np.linalg.norm(mu))})
        self.pairs.sort(key=lambda p: -p['norm_mu'])

        # Canonical = most-displaced pair, unless caller pinned one
        if canonical_pair is None:
            self.canonical_pair_idx = 0
        else:
            target = tuple(sorted(canonical_pair))
            for k, p in enumerate(self.pairs):
                if p['pair'] == target:
                    self.canonical_pair_idx = k
                    break
            else:
                raise ValueError(f"Pair {canonical_pair} not in family")

        # Standardization so the canonical bivariate Gaussian fits comfortably
        # within [-x_range, x_range]: shift its mean to origin, rescale so the
        # larger marginal std equals target_std.
        if standardize:
            mu_c = self.pairs[self.canonical_pair_idx]['mu']
            V_c = self.pairs[self.canonical_pair_idx]['V']
            self.shift = mu_c.copy()
            max_std = float(max(np.sqrt(V_c[0, 0]), np.sqrt(V_c[1, 1])))
            self.scale = max_std / target_std if max_std > 0 else 1.0
        else:
            self.shift = np.zeros(2)
            self.scale = 1.0

    def _render(self, mu, V):
        """Render a bivariate Gaussian (mu, V) on the 40x40 grid, standardized."""
        mu_std = (mu - self.shift) / self.scale
        V_std = V / (self.scale ** 2)
        det = np.linalg.det(V_std)
        inv = np.linalg.inv(V_std)
        dx = self.X - mu_std[0]
        dy = self.Y - mu_std[1]
        mahal = inv[0, 0] * dx**2 + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy**2
        dist = np.exp(-0.5 * mahal) / (2 * np.pi * np.sqrt(max(det, 1e-12)))
        return dist / (dist.sum() + 1e-10)

    def sample(self):
        k = np.random.randint(len(self.pairs))
        p = self.pairs[k]
        dist = self._render(p['mu'], p['V'])
        return dist, {'pair': p['pair'], 'mu': p['mu'].tolist()}

    def get_canonical(self):
        p = self.pairs[self.canonical_pair_idx]
        dist = self._render(p['mu'], p['V'])
        return dist, {'pair': p['pair'], 'mu': p['mu'].tolist()}


def get_family(name, grid_size=40, x_range=3.0):
    """Get distribution family by name."""
    families = {
        'gaussian': GaussianFamily,
        'ring': RingFamily,
        'correlated': CorrelatedGaussianFamily,
        'four_gaussians': FourGaussiansFamily,
        'vibronic': VibronicFamily,
    }
    if name not in families:
        raise ValueError(f"Unknown family: {name}. Available: {list(families.keys())}")
    return families[name](grid_size=grid_size, x_range=x_range)


# =============================================================================
# Metrics
# =============================================================================

def compute_wasserstein_2d(p, q):
    """Approximate 2D Wasserstein distance via marginals."""
    p = np.asarray(p)
    q = np.asarray(q)

    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return float('inf')

    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum <= 0 or q_sum <= 0:
        return float('inf')

    p = p / p_sum
    q = q / q_sum

    # Marginals
    p_x = np.clip(p.sum(axis=0), 1e-10, None)
    p_y = np.clip(p.sum(axis=1), 1e-10, None)
    q_x = np.clip(q.sum(axis=0), 1e-10, None)
    q_y = np.clip(q.sum(axis=1), 1e-10, None)

    p_x = p_x / p_x.sum()
    p_y = p_y / p_y.sum()
    q_x = q_x / q_x.sum()
    q_y = q_y / q_y.sum()

    try:
        w_x = wasserstein_distance(range(len(p_x)), range(len(q_x)), p_x, q_x)
        w_y = wasserstein_distance(range(len(p_y)), range(len(q_y)), p_y, q_y)
        return (w_x + w_y) / 2
    except:
        return float('inf')


def build_energy_distance_context(X, Y, val_set, canonical_target):
    """
    Precompute what's needed to evaluate 2D energy distances cheaply.

    Energy distance between distributions p, q supported on the same grid:
        ED(p, q) = 2 p^T D q  -  p^T D p  -  q^T D q
    with D[i, j] = Euclidean distance between grid points i and j (in x/y
    units, NOT grid cells). It is a proper metric on distributions and is
    sensitive to the full 2D geometry -- unlike compute_wasserstein_2d,
    which only compares marginals and cannot distinguish e.g. a ring from
    a 4-lobe pattern with matching marginals.

    Precomputing D @ q for every validation member turns nearest-member
    search into one matrix-vector product per generated sample.
    """
    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))  # (G^2, G^2), ~10 MB at 40x40

    def _flat(p):
        p = np.asarray(p, dtype=np.float32).ravel()
        return p / (p.sum() + 1e-10)

    val_flat = np.stack([_flat(m) for m, _ in val_set], axis=1)  # (G^2, n_val)
    val_Dq = D @ val_flat                                        # (G^2, n_val)
    val_self = np.einsum('ij,ij->j', val_flat, val_Dq)           # (n_val,)

    canon_flat = _flat(canonical_target)
    canon_Dq = D @ canon_flat
    canon_self = float(canon_flat @ canon_Dq)

    return {'D': D, 'val_Dq': val_Dq, 'val_self': val_self,
            'canon_Dq': canon_Dq, 'canon_self': canon_self}


def energy_distances(p, ed_ctx):
    """Return (ED to canonical, ED to nearest validation member) for density p."""
    p = np.asarray(p, dtype=np.float32).ravel()
    p = p / (p.sum() + 1e-10)
    Dp = ed_ctx['D'] @ p
    p_self = float(p @ Dp)

    ed_canon = 2.0 * float(p @ ed_ctx['canon_Dq']) - p_self - ed_ctx['canon_self']
    ed_val = 2.0 * (p @ ed_ctx['val_Dq']) - p_self - ed_ctx['val_self']
    return ed_canon, float(ed_val.min())


# =============================================================================
# Dataset Generation (NEW)
# =============================================================================

def generate_dataset(family, n_samples):
    """
    Pre-generate dataset from distribution family.

    Args:
        family: DistributionFamily instance
        n_samples: Number of distributions to generate

    Returns:
        dataset: List of (distribution, params) tuples
    """
    print(f"Generating {n_samples} distributions from {family.__class__.__name__}...")
    dataset = []

    for i in range(n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Generated {i + 1}/{n_samples}...")
        dist, params = family.sample()
        dataset.append((dist, params))

    print(f"Dataset generation complete: {len(dataset)} distributions")
    return dataset


# =============================================================================
# Validation Function (NEW)
# =============================================================================

def validate(generator, canonical_target, val_set, xvec, yvec, ed_ctx=None,
             n_samples=10):
    """
    Validate generator over n_samples random z draws.

    Metrics (all marginal-W1 based, grid-cell units):
      canonical_w1: mean W1 to the canonical family member. Legacy metric,
          kept for continuity with earlier runs (rewards collapse-to-canonical).
      nearest_w1:  mean over samples of the W1 to the CLOSEST validation-set
          member. Does not penalize a generator that learned family variety.
      diversity:   mean pairwise W1 between the generated samples themselves.
          ~0 means z-ignoring mode collapse. Compare against the validation
          set's own self-diversity printed at startup.

    Caveat: nearest_w1 alone cannot see collapse onto a single val member --
    always read it together with diversity.

    If ed_ctx (from build_energy_distance_context) is given, also reports
    full-2D energy distances (x/y units, NOT grid cells like the W1s):
      canonical_ed / nearest_ed: analogous to the W1 metrics but sensitive
          to the full 2D geometry, not just the marginals.

    Returns:
        dict with keys 'canonical_w1', 'nearest_w1', 'diversity',
        'canonical_ed', 'nearest_ed' (EDs are NaN when ed_ctx is None).
    """
    samples = []
    B = getattr(generator, 'batch_size', None)
    if B:
        # Batched engine: fill the sample budget a batch at a time
        attempts = 0
        while len(samples) < n_samples and attempts < 10:
            attempts += 1
            z = tf.random.normal([B, generator.latent_dim])
            batch = generator.generate_batch(z, xvec, yvec).numpy()
            for b in range(B):
                if len(samples) < n_samples and np.isfinite(batch[b]).all():
                    samples.append(batch[b])
    else:
        for _ in range(n_samples):
            z = tf.random.normal([generator.latent_dim])
            fake = generator.generate_distribution_2d(z, xvec, yvec).numpy()
            if np.isfinite(fake).all():
                samples.append(fake)

    if not samples:
        return {'canonical_w1': float('inf'),
                'nearest_w1': float('inf'),
                'diversity': 0.0,
                'canonical_ed': float('inf'),
                'nearest_ed': float('inf')}

    canonical_w1s, nearest_w1s = [], []
    canonical_eds, nearest_eds = [], []
    for fake in samples:
        w_canon = compute_wasserstein_2d(canonical_target, fake)
        if np.isfinite(w_canon):
            canonical_w1s.append(w_canon)
        w_near = min(compute_wasserstein_2d(member, fake)
                     for member, _ in val_set)
        if np.isfinite(w_near):
            nearest_w1s.append(w_near)
        if ed_ctx is not None:
            ed_canon, ed_near = energy_distances(fake, ed_ctx)
            if np.isfinite(ed_canon):
                canonical_eds.append(ed_canon)
            if np.isfinite(ed_near):
                nearest_eds.append(ed_near)

    pair_w1s = [compute_wasserstein_2d(samples[i], samples[j])
                for i in range(len(samples))
                for j in range(i + 1, len(samples))]
    pair_w1s = [w for w in pair_w1s if np.isfinite(w)]

    return {
        'canonical_w1': np.mean(canonical_w1s) if canonical_w1s else float('inf'),
        'nearest_w1': np.mean(nearest_w1s) if nearest_w1s else float('inf'),
        'diversity': np.mean(pair_w1s) if pair_w1s else 0.0,
        'canonical_ed': np.mean(canonical_eds) if canonical_eds else float('nan'),
        'nearest_ed': np.mean(nearest_eds) if nearest_eds else float('nan'),
    }


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


# =============================================================================
# Training Loop
# =============================================================================

def train_2d_qgan(
    family_name='ring',
    n_train=400,
    n_val=100,
    n_total_modes=2,        # Total qumodes (2 output + ancilla)
    n_layers=6,
    cutoff_dim=8,
    use_kerr=True,
    epochs=500,
    g_lr=0.005,
    d_lr=0.0002,            # Critic LR; raise toward g_lr for a strong WGAN critic
    n_critic=1,             # Critic steps per G step (WGAN-GP standard: 5)
    batch_size=1,           # Samples per step via SF TF batching (1 = legacy sequential)
    supervised_weight=0.0,  # Start with 30% supervised to give G a head start
    supervised_warmup=200,  # Slow decay from supervised to adversarial
    gp_weight=5.0,          # Gradient penalty weight (lambda)
    gp_warmup=50,           # Epochs to warm up GP
    instance_noise=0.1,     # Initial noise std added to D inputs (capacity handicap)
    noise_anneal=200,       # Epochs to anneal instance noise to the floor
    noise_floor=0.0,        # Instance noise floor after anneal (0 = legacy, anneal to zero)
    critic_blur_sigma=0.0,  # Gaussian blur (grid cells) on critic inputs (0 = off)
    d_dropout=0.3,          # Dropout rate in discriminator
    latent_scale=1.0,
    ket_penalty_weight=20.0,  # Weight on (1 - ket_norm)^2 penalty in G's loss
    g_grad_clip=5.0,          # Max norm for G gradient clipping (<=0 disables)
    grid_size=40,
    x_range=3.0,
    log_dir=None,
    plot_every=20,
    val_every=20,
    seed=None,              # Resolved to a concrete int and recorded; None => entropy-based
    deterministic=False,    # Opt into stronger (slower) same-machine determinism
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
    generator = CVQGANGenerator(
        n_modes=n_total_modes,
        n_output_modes=2,
        n_layers=n_layers,
        cutoff_dim=cutoff_dim,
        use_kerr=use_kerr,
        latent_scale=latent_scale,
        batch_size=batch_size,
    )

    # Initialize discriminator
    print("\nInitializing discriminator...")
    discriminator = Discriminator2D(hidden_dims=[16, 8], dropout_rate=d_dropout)

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
    degenerate_count = 0

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

    for epoch in range(1, epochs + 1):

        # === DISCRIMINATOR TRAINING ===
        # Compute current GP weight (warmup)
        current_gp_weight = gp_weight * min(1.0, epoch / max(1, gp_warmup))

        # Instance noise: decays to 0 over noise_anneal epochs
        # This blurs real/fake boundary, preventing D from trivially winning
        # Decays from instance_noise to noise_floor over noise_anneal epochs.
        # A nonzero floor keeps the critic from overfitting high-frequency
        # generator ripples after the anneal ends. floor=0 reduces exactly
        # to the legacy schedule.
        current_noise = noise_floor + (instance_noise - noise_floor) * max(
            0.0, 1.0 - epoch / max(1, noise_anneal))

        if pure_supervised:
            # Path A: no critic. Skip discriminator training entirely.
            d_loss_avg = 0.0
            d_grad_avg = 0.0
            gp_avg = 0.0
            d_real_scores = []
            d_fake_scores = []
        else:
            d_losses = []
            d_grad_norms = []
            gp_values = []
            d_real_scores = []
            d_fake_scores = []

            for _ in range(n_critic):
                # Pick a RANDOM batch of real samples from train_set
                idxs = np.random.randint(len(train_set), size=batch_size)
                real_tf = tf.constant(
                    np.stack([train_set[i][0] for i in idxs]), dtype=tf.float32)

                # Generate with random z, OUTSIDE the critic's tape: the fake
                # is a constant for the D update, and recording the SF circuit
                # on D's tape costs memory/time for gradients never taken.
                z = tf.random.normal([batch_size, generator.latent_dim])
                gen_prob = generator.generate_batch(z, xvec, yvec)
                if not tf.reduce_all(tf.math.is_finite(gen_prob)):
                    continue
                gen_prob = tf.stop_gradient(gen_prob)

                # Blur (optional), then peak-normalize what the critic sees.
                # Single blur per pipeline: the noise re-normalization below
                # must NOT blur again.
                real_in = to_critic_input(critic_blur(real_tf, blur_kernel))
                fake_in = to_critic_input(critic_blur(gen_prob, blur_kernel))

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
                    real_score = discriminator(real_in_n, training=True)
                    fake_score = discriminator(fake_in_n, training=True)

                    # WGAN loss: D wants real_score > fake_score
                    d_loss_wgan = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

                    # Gradient penalty (WGAN-GP), on the same inputs D sees
                    gp = compute_gradient_penalty(discriminator, real_in_n, fake_in_n)

                    # Total D loss = WGAN loss + λ * GP
                    d_loss = d_loss_wgan + current_gp_weight * gp

                d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
                if d_grads[0] is not None:
                    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
                    d_grad_norms.append(float(tf.linalg.global_norm(d_grads)))

                d_losses.append(float(d_loss_wgan))  # Track WGAN loss (not total)
                gp_values.append(float(gp))
                d_real_scores.append(float(tf.reduce_mean(real_score)))
                d_fake_scores.append(float(tf.reduce_mean(fake_score)))

            d_loss_avg = np.mean(d_losses) if d_losses else 0.0
            d_grad_avg = np.mean(d_grad_norms) if d_grad_norms else 0.0
            gp_avg = np.mean(gp_values) if gp_values else 0.0

        # === GENERATOR TRAINING ===
        # Select the real target(s).
        #   Path A: always the SAME fixed canonical target (representability test).
        #   GAN mode: random family members (adversarial training over the family).
        if pure_supervised:
            real_tf = tf.expand_dims(fixed_real_tf, 0)   # broadcasts over the batch
        else:
            idxs = np.random.randint(len(train_set), size=batch_size)
            real_tf = tf.constant(
                np.stack([train_set[i][0] for i in idxs]), dtype=tf.float32)

        # Sample new latent vectors
        z = tf.random.normal([batch_size, generator.latent_dim])

        with tf.GradientTape() as tape:
            gen_prob, ket_norm_tf = generator.generate_batch(
                z, xvec, yvec, return_ket_norm=True
            )

            if not tf.reduce_all(tf.math.is_finite(gen_prob)):
                print(f"  Warning: Degenerate at epoch {epoch}")
                degenerate_count += 1
                if degenerate_count > 30:
                    print("Stopping: Too many degenerate outputs")
                    break
                continue
            else:
                degenerate_count = 0

            fake_score = discriminator(
                to_critic_input(critic_blur(gen_prob, blur_kernel)), training=False)

            # Adversarial loss: G wants to maximize fake_score
            adversarial_loss = -tf.reduce_mean(fake_score)

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
            if pure_supervised:
                current_sw = 1.0
                g_loss = supervised_loss + ket_penalty_weight * ket_penalty
            else:
                current_sw = supervised_weight * max(0.0, 1.0 - epoch / max(1, supervised_warmup))
                g_loss = (
                    (1 - current_sw) * adversarial_loss
                    + current_sw * supervised_loss
                    + ket_penalty_weight * ket_penalty
                )

        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        if g_grads[0] is not None:
            if g_grad_clip > 0:
                g_grads = [tf.clip_by_norm(g, g_grad_clip) for g in g_grads if g is not None]
            else:
                g_grads = [g for g in g_grads if g is not None]
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
            g_grad_norm = float(tf.linalg.global_norm(g_grads))
        else:
            g_grad_norm = 0.0

        # === METRICS ===
        gen_np = gen_prob.numpy()[0]  # first batch element for the cheap per-epoch metric
        w_dist = compute_wasserstein_2d(canonical_target, gen_np)

        history['g_loss'].append(float(g_loss))
        history['d_loss'].append(d_loss_avg)
        history['wasserstein'].append(w_dist)
        history['g_grad_norm'].append(g_grad_norm)
        history['d_grad_norm'].append(d_grad_avg)
        history['supervised_loss'].append(float(supervised_loss))
        history['adversarial_loss'].append(float(adversarial_loss))
        history['gp_value'].append(gp_avg)
        history['gp_weight_current'].append(current_gp_weight)
        history['d_real_score'].append(np.mean(d_real_scores) if d_real_scores else 0.0)
        history['d_fake_score'].append(np.mean(d_fake_scores) if d_fake_scores else 0.0)
        history['supervised_weight_current'].append(current_sw)
        history['instance_noise_current'].append(current_noise)

        # Ket norm diagnostic — recorded every epoch (free: same tensor used in G loss).
        # Print a warning at val_every cadence so logs don't get spammy.
        ket_norm = float(tf.reduce_mean(ket_norm_tf))
        history['ket_norm'].append(ket_norm)
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
                best_weights = generator.weights.numpy().copy()

            print(f"Epoch {epoch:4d} | G: {float(g_loss):.4f} | D: {d_loss_avg:+.4f} | "
                  f"nearW1: {val_metrics['nearest_w1']:.4f} | "
                  f"canonW1: {val_metrics['canonical_w1']:.4f} | "
                  f"div: {val_metrics['diversity']:.4f} | "
                  f"nearED: {val_metrics['nearest_ed']:.4f} | Best: {best_val_metric:.4f}")
        else:
            # Regular logging
            if epoch % 10 == 0 or epoch == 1:
                d_real_avg = np.mean(d_real_scores) if d_real_scores else 0.0
                d_fake_avg = np.mean(d_fake_scores) if d_fake_scores else 0.0
                print(f"Epoch {epoch:4d} | G: {float(g_loss):.4f} | D: {d_loss_avg:+.4f} | "
                      f"W1: {w_dist:.4f} | D(r):{d_real_avg:+.2f} D(f):{d_fake_avg:+.2f} | "
                      f"SW:{current_sw:.2f} noise:{current_noise:.3f}")

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
    if best_weights is not None:
        print("Restoring best weights from validation...")
        generator.weights.assign(best_weights)
        # Persist for post-hoc re-scoring (e.g. with metrics added later);
        # history.npz alone cannot reconstruct the generator.
        np.save(f"{log_dir}/best_weights.npy", best_weights)

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

    print(f"\nResults saved to: {log_dir}")

    return generator, history


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(target, generated, X, Y, save_path, title=""):
    """Plot target vs generated."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].contourf(X, Y, target, levels=30, cmap='viridis')
    axes[0].set_title('Target')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, generated, levels=30, cmap='viridis')
    axes[1].set_title('Generated')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    diff = np.abs(target - generated)
    im2 = axes[2].contourf(X, Y, diff, levels=30, cmap='Reds')
    axes[2].set_title(f'|Difference|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    epochs = range(1, len(history['g_loss']) + 1)

    # Losses
    axes[0, 0].plot(epochs, history['g_loss'], label='G Total', alpha=0.7)
    axes[0, 0].plot(epochs, history['d_loss'], label='D Loss', alpha=0.7)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GAN Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Wasserstein
    w = history['wasserstein']
    valid_w = [x for x in w if x < float('inf')]
    if valid_w:
        axes[0, 1].plot(epochs[:len(valid_w)], valid_w, 'g-', alpha=0.5, label='Train')

    # Validation Wasserstein
    if history['val_wasserstein']:
        val_w = history['val_wasserstein']
        valid_val_w = [x for x in val_w if x < float('inf')]
        if valid_val_w:
            axes[0, 1].plot(range(1, len(valid_val_w) + 1), valid_val_w, 'b-',
                           linewidth=2, marker='o', label='Val (canonical)')
            axes[0, 1].axhline(y=min(valid_val_w), color='r', linestyle='--',
                              label=f'Best Val: {min(valid_val_w):.4f}')

    val_near = [x for x in history.get('val_nearest_w1', []) if x < float('inf')]
    if val_near:
        axes[0, 1].plot(range(1, len(val_near) + 1), val_near, 'm-',
                       linewidth=2, marker='s', label='Val (nearest member)')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Wasserstein Distance')
    axes[0, 1].set_title('Distribution Quality')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gradient norms
    axes[0, 2].plot(epochs, history['g_grad_norm'], label='Generator', alpha=0.7)
    axes[0, 2].plot(epochs, history['d_grad_norm'], label='Discriminator', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Gradient Norm')
    axes[0, 2].set_title('Gradient Norms')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # GP Weight Warmup (or Loss Components if no GP)
    if 'gp_weight_current' in history and history['gp_weight_current']:
        axes[1, 0].plot(epochs, history['gp_weight_current'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GP Weight')
        axes[1, 0].set_title('Gradient Penalty Weight (warmup)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].plot(epochs, history['adversarial_loss'], label='Adversarial', alpha=0.7)
        axes[1, 0].plot(epochs, history['supervised_loss'], label='Supervised', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Smoothed Wasserstein
    if valid_w:
        window = min(20, len(valid_w) // 5)
        if window > 1:
            smoothed = np.convolve(valid_w, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window, len(valid_w) + 1), smoothed, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Wasserstein (Smoothed)')
    axes[1, 1].set_title('Convergence Trend')
    axes[1, 1].grid(True, alpha=0.3)

    # GP Value (or G/D balance if no GP)
    if 'gp_value' in history and history['gp_value']:
        axes[1, 2].plot(epochs, history['gp_value'], 'orange', alpha=0.7)
        axes[1, 2].axhline(y=1, color='gray', linestyle='--', label='Target (1.0)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Gradient Penalty')
        axes[1, 2].set_title('GP Value (should stabilize ~1)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        g_power = np.abs(history['g_grad_norm'])
        d_power = np.abs(history['d_grad_norm'])
        ratio = np.array(g_power) / (np.array(d_power) + 1e-8)
        axes[1, 2].plot(epochs, ratio, 'orange', alpha=0.7)
        axes[1, 2].axhline(y=1, color='gray', linestyle='--')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('G/D Gradient Ratio')
        axes[1, 2].set_title('Training Balance')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

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


def _format_val(v):
    """Format a value for directory naming (strip trailing zeros from floats)."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_hp_suffix(args, parser):
    """Build a suffix string from non-default hyperparameters.

    Only parameters that differ from argparse defaults are included,
    using short abbreviations. Cosmetic params are excluded.
    """
    # (argparse dest, abbreviation) in display order
    HP_MAP = [
        ('n_modes',           'm'),
        ('n_layers',          'nl'),
        ('cutoff_dim',        'c'),
        ('no_kerr',           'nokerr'),
        ('epochs',            'ep'),
        ('g_lr',              'glr'),
        ('d_lr',              'dlr'),
        ('n_critic',          'nc'),
        ('batch_size',        'bs'),
        ('supervised_weight', 'sw'),
        ('supervised_warmup', 'swup'),
        ('gp_weight',         'gp'),
        ('gp_warmup',         'gpwup'),
        ('instance_noise',    'noise'),
        ('noise_anneal',      'nann'),
        ('noise_floor',       'nfloor'),
        ('critic_blur',       'blur'),
        ('d_dropout',         'drop'),
        ('latent_scale',      'ls'),
        ('ket_penalty_weight', 'kpen'),
        ('g_grad_clip',       'gclip'),
        ('grid_size',         'gs'),
        ('seed',              'seed'),
    ]

    parts = []
    for dest, abbrev in HP_MAP:
        default = parser.get_default(dest)
        actual = getattr(args, dest, default)
        if actual != default:
            if isinstance(actual, bool):
                if actual:          # flag was set (e.g. --no-kerr)
                    parts.append(abbrev)
            else:
                parts.append(f"{abbrev}{_format_val(actual)}")
    return ('_' + '_'.join(parts)) if parts else ''


def main():
    parser = argparse.ArgumentParser(description='Train 2D CV-QGAN with Pre-Generated Dataset')
    parser.add_argument('--family', type=str, required=True,
                       choices=['gaussian', 'ring', 'correlated', 'four_gaussians', 'vibronic'],
                       help='Distribution family to learn (REQUIRED)')
    parser.add_argument('--n-train', type=int, default=400,
                       help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=100,
                       help='Number of validation samples')
    parser.add_argument('--n-modes', type=int, default=2,
                       help='Total qumodes (2=no ancilla, 3=1 ancilla, etc.)')
    parser.add_argument('--n-ancilla', type=int, default=None,
                       help='Number of ancilla modes (alternative to --n-modes)')
    parser.add_argument('--n-layers', type=int, default=6,
                       help='Number of CV-QNN layers')
    parser.add_argument('--cutoff-dim', type=int, default=12,
                       help='Fock space cutoff dimension')
    parser.add_argument('--no-kerr', action='store_true',
                       help='Disable Kerr gates (not recommended)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--g-lr', type=float, default=0.005,
                       help='Generator learning rate')
    parser.add_argument('--d-lr', type=float, default=0.005,
                       help='Critic learning rate (WGAN-GP wants a well-trained '
                            'critic; consider raising toward --g-lr)')
    parser.add_argument('--n-critic', type=int, default=5,
                       help='Critic steps per generator step (WGAN-GP standard: 5)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Samples per training step via SF TF backend batching '
                            '(1 = legacy sequential; try 8)')
    parser.add_argument('--supervised-weight', type=float, default=0.0,
                       help='Initial supervised loss weight (0=pure GAN, 1=pure supervised)')
    parser.add_argument('--supervised-warmup', type=int, default=200,
                       help='Epochs over which supervised weight decays to 0')
    parser.add_argument('--gp-weight', type=float, default=5.0,
                       help='Gradient penalty weight (lambda)')
    parser.add_argument('--gp-warmup', type=int, default=50,
                       help='Epochs to warm up gradient penalty')
    parser.add_argument('--instance-noise', type=float, default=0.1,
                       help='Initial instance noise std added to D inputs')
    parser.add_argument('--noise-anneal', type=int, default=200,
                       help='Epochs to anneal instance noise to the floor')
    parser.add_argument('--noise-floor', type=float, default=0.0,
                       help='Instance-noise floor after anneal (0 = anneal to zero, legacy)')
    parser.add_argument('--critic-blur', type=float, default=0.0,
                       help='Gaussian blur sigma (grid cells) applied identically to real '
                            'and fake critic inputs; 0 = off (legacy). Intended range 0.5-1.0')
    parser.add_argument('--d-dropout', type=float, default=0.0,
                       help='Dropout rate in discriminator')
    parser.add_argument('--latent-scale', type=float, default=0.3,
                       help='Scale for latent vector encoding')
    parser.add_argument('--ket-penalty-weight', type=float, default=20.0,
                       help='Weight on (1 - ket_norm)^2 penalty in G loss to prevent Fock truncation')
    parser.add_argument('--g-grad-clip', type=float, default=5.0,
                       help='Max norm for G gradient clipping (<=0 disables)')
    parser.add_argument('--grid-size', type=int, default=40,
                       help='Grid resolution')
    parser.add_argument('--plot-every', type=int, default=20,
                       help='Plot frequency')
    parser.add_argument('--val-every', type=int, default=20,
                       help='Validation frequency')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility. Default: an '
                            'entropy-based seed is drawn and recorded, so '
                            'unseeded runs still vary but stay reproducible.')
    parser.add_argument('--deterministic', action='store_true',
                       help='Opt into stronger (slower) same-machine determinism: '
                            'TF op determinism + single-threaded intra/inter-op + '
                            'TF_DETERMINISTIC_OPS/oneDNN env (off by default).')

    args = parser.parse_args()

    # Resolve total modes from --n-modes or --n-ancilla
    if args.n_ancilla is not None:
        n_total_modes = 2 + args.n_ancilla
    else:
        n_total_modes = args.n_modes

    if n_total_modes < 2:
        parser.error("--n-modes must be at least 2 (need 2 output modes)")

    # Resolve the seed up front so it is recorded everywhere (folder tag,
    # run_config.json, console, history.npz) — never leave it unknown.
    args.seed = resolve_seed(args.seed)

    # Build log directory name with non-default hyperparameters
    # If --n-ancilla was used, reflect it in args.n_modes for suffix building
    args.n_modes = n_total_modes
    hp_suffix = build_hp_suffix(args, parser)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/qgan_2d_{args.family}_{timestamp}{hp_suffix}"

    # Dump a durable run config (args + environment + seed) so any result can be
    # reproduced and each A/B table row can cite its seed.
    os.makedirs(log_dir, exist_ok=True)
    run_config = dict(vars(args))
    run_config.update({
        'timestamp': timestamp,
        'tf_version': tf.__version__,
        'numpy_version': np.__version__,
        'sf_version': sf.__version__,
        'deterministic': args.deterministic,
        'TF_DETERMINISTIC_OPS': os.environ.get('TF_DETERMINISTIC_OPS'),
        'TF_ENABLE_ONEDNN_OPTS': os.environ.get('TF_ENABLE_ONEDNN_OPTS'),
    })
    with open(f"{log_dir}/run_config.json", 'w') as f:
        json.dump(run_config, f, indent=2, sort_keys=True)

    train_2d_qgan(
        family_name=args.family,
        n_train=args.n_train,
        n_val=args.n_val,
        n_total_modes=n_total_modes,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        use_kerr=not args.no_kerr,
        epochs=args.epochs,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        batch_size=args.batch_size,
        supervised_weight=args.supervised_weight,
        supervised_warmup=args.supervised_warmup,
        gp_weight=args.gp_weight,
        gp_warmup=args.gp_warmup,
        instance_noise=args.instance_noise,
        noise_anneal=args.noise_anneal,
        noise_floor=args.noise_floor,
        critic_blur_sigma=args.critic_blur,
        d_dropout=args.d_dropout,
        latent_scale=args.latent_scale,
        ket_penalty_weight=args.ket_penalty_weight,
        g_grad_clip=args.g_grad_clip,
        grid_size=args.grid_size,
        log_dir=log_dir,
        plot_every=args.plot_every,
        val_every=args.val_every,
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
