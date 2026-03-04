#!/usr/bin/env python
"""
Unified 2D CV-QGAN with Pre-Generated Dataset
==============================================

TRUE QGAN Implementation with these key improvements:
1. Pre-generate 500 distributions (400 train, 100 validation)
2. Generator receives RANDOM latent vectors z as input
3. z is encoded into the circuit via displacement gates
4. Sequential training (no batching - circuits run one at a time)
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

def compute_hermite_basis(xvec, cutoff_dim):
    """
    Compute Hermite basis functions φ_n(x) for n = 0, 1, ..., cutoff-1.

    φ_n(x) = (1/sqrt(2^n n! sqrt(π))) H_n(x) exp(-x²/2)

    Args:
        xvec: (n_points,) array of x values
        cutoff_dim: Fock space cutoff

    Returns:
        basis: (n_points, cutoff) array of basis function values
    """
    xvec = np.asarray(xvec)
    n_points = len(xvec)
    basis = np.zeros((n_points, cutoff_dim), dtype=np.float64)

    for n in range(cutoff_dim):
        Hn = hermite(n)
        norm = 1.0 / sqrt(2**n * factorial(n) * sqrt(pi))
        basis[:, n] = norm * Hn(xvec) * np.exp(-xvec**2 / 2)

    return tf.constant(basis, dtype=tf.float32)


def recommend_cutoff(n_modes, base_cutoff=8, max_fock_states=5000):
    """
    Suggest cutoff dimension that keeps Fock space manageable.

    The Fock space has cutoff^n_modes states. For large n_modes, the default
    cutoff may use too much memory.

    Args:
        n_modes: Total number of qumodes
        base_cutoff: Desired cutoff (will be reduced if needed)
        max_fock_states: Maximum acceptable Fock space size

    Returns:
        Recommended cutoff dimension
    """
    cutoff = base_cutoff
    while cutoff ** n_modes > max_fock_states and cutoff > 4:
        cutoff -= 1
    return cutoff


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
    ):
        self.n_modes = n_modes
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
        self.eng = sf.Engine('tf', backend_options={'cutoff_dim': cutoff_dim})

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

    def _run_circuit(self, z_single):
        """Run the circuit for a single latent vector."""
        if self.eng.run_progs:
            self.eng.reset()

        # Build parameter mapping
        mapping = {}

        # Map latent vector
        for i, param in enumerate(self.latent_params):
            mapping[param.name] = z_single[i]

        # Map QNN weights
        for i, param in enumerate(self.weight_params):
            mapping[param.name] = self.weights[i]

        # Run circuit
        result = self.eng.run(self.prog, args=mapping)
        return result.state

    def generate_distribution_2d(self, z, xvec, yvec):
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

        Returns:
            prob: (len(xvec), len(yvec)) joint probability distribution
        """
        # Get quantum state
        state = self._run_circuit(z)
        ket = state.ket()  # shape: (cutoff,) * n_modes

        # Compute Hermite basis for the 2 output modes
        hermite_x = compute_hermite_basis(xvec, self.cutoff_dim)
        hermite_y = compute_hermite_basis(yvec, self.cutoff_dim)

        ket_c = tf.cast(ket, tf.complex64)
        hx_c = tf.cast(hermite_x, tf.complex64)
        hy_c = tf.cast(hermite_y, tf.complex64)

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

        return prob

    def get_ket_norm(self, z):
        """Compute the norm of the ket state (should be ~1.0 if cutoff is sufficient)."""
        state = self._run_circuit(z)
        ket = state.ket()
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


def get_family(name, grid_size=40, x_range=3.0):
    """Get distribution family by name."""
    families = {
        'gaussian': GaussianFamily,
        'ring': RingFamily,
        'correlated': CorrelatedGaussianFamily,
        'four_gaussians': FourGaussiansFamily,
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

def validate(generator, canonical_target, xvec, yvec, n_samples=10):
    """
    Validate generator by comparing generated distributions to canonical target.

    Generates n_samples distributions with different random z vectors and
    computes the average Wasserstein distance to the canonical target.
    This measures how well the generator has learned the distribution shape,
    independent of the random latent input.

    Args:
        generator: CVQGANGenerator instance
        canonical_target: The canonical (reference) distribution from the family
        xvec: x-axis grid points
        yvec: y-axis grid points
        n_samples: Number of z vectors to average over

    Returns:
        Average Wasserstein distance to canonical target.
    """
    w_distances = []

    for _ in range(n_samples):
        z = tf.random.normal([generator.latent_dim])
        fake_dist = generator.generate_distribution_2d(z, xvec, yvec).numpy()
        w = compute_wasserstein_2d(canonical_target, fake_dist)

        if np.isfinite(w):
            w_distances.append(w)

    return np.mean(w_distances) if w_distances else float('inf')


def compute_gradient_penalty(discriminator, real_dist, fake_dist):
    """
    Compute gradient penalty for WGAN-GP.

    GP = E[(||∇D(x̂)||_2 - 1)²]
    where x̂ = ε*real + (1-ε)*fake

    This enforces a 1-Lipschitz constraint on the discriminator.
    """
    # Random interpolation coefficient
    epsilon = tf.random.uniform([1], 0.0, 1.0)

    # Interpolate between real and fake
    interpolated = epsilon * real_dist + (1 - epsilon) * fake_dist
    interpolated = tf.expand_dims(interpolated, 0)  # Add batch dim

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    # Compute gradients w.r.t. interpolated input
    grads = gp_tape.gradient(pred, interpolated)

    # Compute L2 norm of gradients
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads)) + 1e-8)

    # Gradient penalty: (||grad|| - 1)²
    gp = tf.square(grad_norm - 1.0)

    return gp


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
    d_lr=0.0002,            # D learns 25x slower than G (capacity mismatch)
    n_critic=1,             # Equal updates — D is already much stronger than Q-G
    supervised_weight=0.0,  # Start with 30% supervised to give G a head start
    supervised_warmup=200,  # Slow decay from supervised to adversarial
    gp_weight=5.0,          # Gradient penalty weight (lambda)
    gp_warmup=50,           # Epochs to warm up GP
    instance_noise=0.1,     # Initial noise std added to D inputs (capacity handicap)
    noise_anneal=200,       # Epochs to anneal instance noise to 0
    d_dropout=0.3,          # Dropout rate in discriminator
    latent_scale=1.0,
    grid_size=40,
    x_range=3.0,
    log_dir=None,
    plot_every=20,
    val_every=20,
):
    """
    Train 2D CV-QGAN with pre-generated dataset.

    Key features:
    1. Pre-generate 500 distributions (400 train, 100 validation)
    2. Sequential training (no batching - circuits run one at a time)
    3. TRUE QGAN with latent vector input
    4. Validation on held-out set
    5. Show ACTUAL training samples in visualizations
    """

    # Setup
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./logs/qgan_2d_{family_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

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
    print(f"Discriminator hidden dims: [16, 8], dropout={d_dropout}")
    print(f"Instance noise: std={instance_noise}, anneal over {noise_anneal} epochs")
    print(f"Supervised weight: {supervised_weight} (warmup decay over {supervised_warmup} epochs)")
    print(f"Gradient penalty: weight={gp_weight}, warmup={gp_warmup} epochs")
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
    print("=" * 70)

    # Get canonical target for reference
    canonical_target, _ = family.get_canonical()

    # Initialize generator
    print("\nInitializing generator...")
    generator = CVQGANGenerator(
        n_modes=n_total_modes,
        n_output_modes=2,
        n_layers=n_layers,
        cutoff_dim=cutoff_dim,
        use_kerr=use_kerr,
        latent_scale=latent_scale,
    )

    # Initialize discriminator
    print("\nInitializing discriminator...")
    discriminator = Discriminator2D(hidden_dims=[16, 8], dropout_rate=d_dropout)

    # Optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.5, beta_2=0.9)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.5, beta_2=0.9)

    # History tracking
    history = {
        'g_loss': [], 'd_loss': [], 'wasserstein': [], 'val_wasserstein': [],
        'g_grad_norm': [], 'd_grad_norm': [],
        'supervised_loss': [], 'adversarial_loss': [],
        'gp_value': [], 'gp_weight_current': [],  # Gradient penalty tracking
        'd_real_score': [], 'd_fake_score': [],    # Discriminator score tracking
        'ket_norm': [],                             # Ket normalization (should be ~1.0)
        'supervised_weight_current': [],            # Current supervised weight after decay
        'instance_noise_current': [],               # Current instance noise std
    }

    best_val_wasserstein = float('inf')
    best_weights = None
    degenerate_count = 0

    # Initial validation
    print("\nComputing initial validation...")
    init_val = validate(generator, canonical_target, xvec, yvec)
    print(f"Initial validation W1: {init_val:.4f}")

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

    for epoch in range(1, epochs + 1):

        # === DISCRIMINATOR TRAINING ===
        d_losses = []
        d_grad_norms = []
        gp_values = []
        d_real_scores = []
        d_fake_scores = []

        # Compute current GP weight (warmup)
        current_gp_weight = gp_weight * min(1.0, epoch / max(1, gp_warmup))

        # Instance noise: decays to 0 over noise_anneal epochs
        # This blurs real/fake boundary, preventing D from trivially winning
        current_noise = instance_noise * max(0.0, 1.0 - epoch / max(1, noise_anneal))

        for _ in range(n_critic):
            # Pick RANDOM real sample from train_set
            idx = np.random.randint(len(train_set))
            real_dist, _ = train_set[idx]
            real_tf = tf.constant(real_dist, dtype=tf.float32)

            # Generate with random z
            z = tf.random.normal([generator.latent_dim])

            with tf.GradientTape() as tape:
                # Generate fake distribution
                gen_prob = generator.generate_distribution_2d(z, xvec, yvec)

                if not tf.reduce_all(tf.math.is_finite(gen_prob)):
                    continue

                # Add instance noise to D inputs (handicap D)
                if current_noise > 0:
                    real_noisy = real_tf + current_noise * tf.random.normal(real_tf.shape)
                    real_noisy = tf.nn.relu(real_noisy)
                    real_noisy = real_noisy / (tf.reduce_sum(real_noisy) + 1e-10)

                    fake_noisy = gen_prob + current_noise * tf.random.normal(gen_prob.shape)
                    fake_noisy = tf.nn.relu(fake_noisy)
                    fake_noisy = fake_noisy / (tf.reduce_sum(fake_noisy) + 1e-10)
                else:
                    real_noisy = real_tf
                    fake_noisy = gen_prob

                # Discriminator scores
                real_batch = tf.expand_dims(real_noisy, 0)
                fake_batch = tf.expand_dims(fake_noisy, 0)

                real_score = discriminator(real_batch, training=True)
                fake_score = discriminator(fake_batch, training=True)

                # WGAN loss: D wants real_score > fake_score
                d_loss_wgan = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

                # Gradient penalty (WGAN-GP)
                gp = compute_gradient_penalty(discriminator, real_tf, gen_prob)

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
        # Pick random real sample for G
        idx = np.random.randint(len(train_set))
        real_dist, _ = train_set[idx]
        real_tf = tf.constant(real_dist, dtype=tf.float32)

        # Sample new latent vector
        z = tf.random.normal([generator.latent_dim])

        with tf.GradientTape() as tape:
            gen_prob = generator.generate_distribution_2d(z, xvec, yvec)

            if not tf.reduce_all(tf.math.is_finite(gen_prob)):
                print(f"  Warning: Degenerate at epoch {epoch}")
                degenerate_count += 1
                if degenerate_count > 30:
                    print("Stopping: Too many degenerate outputs")
                    break
                continue
            else:
                degenerate_count = 0

            fake_batch = tf.expand_dims(gen_prob, 0)
            fake_score = discriminator(fake_batch, training=False)

            # Adversarial loss: G wants to maximize fake_score
            adversarial_loss = -tf.reduce_mean(fake_score)

            # Supervised loss: direct distribution matching
            supervised_loss = tf.reduce_mean(tf.square(gen_prob - real_tf))

            # Combined loss with supervised warmup decay
            current_sw = supervised_weight * max(0.0, 1.0 - epoch / max(1, supervised_warmup))
            g_loss = (1 - current_sw) * adversarial_loss + current_sw * supervised_loss

        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        if g_grads[0] is not None:
            # Clip gradients to prevent large parameter updates
            g_grads = [tf.clip_by_norm(g, 1.0) for g in g_grads if g is not None]
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
            g_grad_norm = float(tf.linalg.global_norm(g_grads))
        else:
            g_grad_norm = 0.0

        # === METRICS ===
        gen_np = gen_prob.numpy()
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

        # Ket norm diagnostic (periodic - expensive to compute)
        if epoch % val_every == 0:
            ket_norm = generator.get_ket_norm(z)
            history['ket_norm'].append(ket_norm)
            if ket_norm < 0.9:
                print(f"  WARNING: Ket norm = {ket_norm:.3f} (cutoff may be too low)")

        # === VALIDATION ===
        if epoch % val_every == 0:
            val_w1 = validate(generator, canonical_target, xvec, yvec)
            history['val_wasserstein'].append(val_w1)

            if val_w1 < best_val_wasserstein:
                best_val_wasserstein = val_w1
                best_weights = generator.weights.numpy().copy()

            print(f"Epoch {epoch:4d} | G: {float(g_loss):.4f} | D: {d_loss_avg:+.4f} | "
                  f"W1: {w_dist:.4f} | Val W1: {val_w1:.4f} | Best Val: {best_val_wasserstein:.4f}")
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
    print(f"Best validation W1: {best_val_wasserstein:.4f}")

    # Restore best weights
    if best_weights is not None:
        print("Restoring best weights from validation...")
        generator.weights.assign(best_weights)

    # Final validation
    final_val = validate(generator, canonical_target, xvec, yvec)
    print(f"Final validation W1: {final_val:.4f}")

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
    np.savez(f"{log_dir}/history.npz", **history)

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
                           linewidth=2, marker='o', label='Validation')
            axes[0, 1].axhline(y=min(valid_val_w), color='r', linestyle='--',
                              label=f'Best Val: {min(valid_val_w):.4f}')

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

def main():
    parser = argparse.ArgumentParser(description='Train 2D CV-QGAN with Pre-Generated Dataset')
    parser.add_argument('--family', type=str, required=True,
                       choices=['gaussian', 'ring', 'correlated', 'four_gaussians'],
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
    parser.add_argument('--cutoff-dim', type=int, default=8,
                       help='Fock space cutoff dimension')
    parser.add_argument('--no-kerr', action='store_true',
                       help='Disable Kerr gates (not recommended)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--g-lr', type=float, default=0.005,
                       help='Generator learning rate')
    parser.add_argument('--d-lr', type=float, default=0.0002,
                       help='Discriminator learning rate (keep much lower than G)')
    parser.add_argument('--n-critic', type=int, default=1,
                       help='Discriminator steps per generator step (1=equal)')
    parser.add_argument('--supervised-weight', type=float, default=0.3,
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
                       help='Epochs to anneal instance noise to 0')
    parser.add_argument('--d-dropout', type=float, default=0.3,
                       help='Dropout rate in discriminator')
    parser.add_argument('--latent-scale', type=float, default=1.0,
                       help='Scale for latent vector encoding')
    parser.add_argument('--grid-size', type=int, default=40,
                       help='Grid resolution')
    parser.add_argument('--plot-every', type=int, default=20,
                       help='Plot frequency')
    parser.add_argument('--val-every', type=int, default=20,
                       help='Validation frequency')

    args = parser.parse_args()

    # Resolve total modes from --n-modes or --n-ancilla
    if args.n_ancilla is not None:
        n_total_modes = 2 + args.n_ancilla
    else:
        n_total_modes = args.n_modes

    if n_total_modes < 2:
        parser.error("--n-modes must be at least 2 (need 2 output modes)")

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
        supervised_weight=args.supervised_weight,
        supervised_warmup=args.supervised_warmup,
        gp_weight=args.gp_weight,
        gp_warmup=args.gp_warmup,
        instance_noise=args.instance_noise,
        noise_anneal=args.noise_anneal,
        d_dropout=args.d_dropout,
        latent_scale=args.latent_scale,
        grid_size=args.grid_size,
        plot_every=args.plot_every,
        val_every=args.val_every,
    )


if __name__ == "__main__":
    main()
