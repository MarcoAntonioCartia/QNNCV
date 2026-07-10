"""
Symbolic CV-QNN circuit and generator
=====================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor), plus
SEAM 1: the latent-encoding site is an EncodingSpec strategy. Only
'input_displacement' (the original front-Dgate encoding) is implemented;
it reproduces the previous inline code gate-for-gate (identical parameter
names z_i/w_i, gate types and order -> identical sf.Program, golden-tested).
"""

# Compatibility patch: SF still calls scipy.integrate.simps, removed in
# modern SciPy. Idempotent duplicate of the root-script guard so this
# module also imports standalone.
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops

from src.quantum.hermite import compute_hermite_basis


class EncodingSpec:
    """How the latent vector z enters the circuit (SEAM 1, interface only).

    kinds:
      'input_displacement' — the current behavior: z displaces the vacuum
          at the circuit front, Dgate(z[2m] * latent_scale, z[2m+1]) per mode.
      'interleaved_squeeze' — FUTURE mid-circuit z-injection; declared but
          not implemented (raises NotImplementedError).
    """

    KINDS = ('input_displacement', 'interleaved_squeeze')

    def __init__(self, kind='input_displacement'):
        if kind not in self.KINDS:
            raise ValueError(f"Unknown encoding kind: {kind!r}. "
                             f"Available: {list(self.KINDS)}")
        if kind == 'interleaved_squeeze':
            raise NotImplementedError(
                "interleaved_squeeze (mid-circuit z-injection) is a future "
                "task — only 'input_displacement' is implemented")
        self.kind = kind

    def latent_dim(self, n_modes):
        # Latent dimension: magnitude + phase for each mode
        return 2 * n_modes

    def apply(self, q, latent_params, latent_scale, n_modes):
        """Apply the encoding at the circuit front (inside prog.context)."""
        # ENCODING LAYER: Apply latent vector as displacement
        for mode in range(n_modes):
            r_idx = 2 * mode
            phi_idx = 2 * mode + 1
            ops.Dgate(
                latent_params[r_idx] * latent_scale,
                latent_params[phi_idx]
            ) | q[mode]


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
        encoding=None,              # SEAM 1: EncodingSpec (None = input_displacement)
    ):
        self.n_modes = n_modes
        self.batch_size = batch_size if (batch_size and batch_size > 1) else None
        self.n_output_modes = n_output_modes
        self.n_ancilla = n_modes - n_output_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.use_kerr = use_kerr
        self.latent_scale = latent_scale

        # SEAM 1: latent-encoding strategy (default reproduces legacy behavior)
        self.encoding = encoding if encoding is not None else EncodingSpec('input_displacement')

        # Latent dimension: magnitude + phase for each mode
        self.latent_dim = self.encoding.latent_dim(n_modes)

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
            # ENCODING LAYER (SEAM 1): delegated to the encoding strategy;
            # 'input_displacement' reproduces the legacy inline loop verbatim.
            self.encoding.apply(q, self.latent_params, self.latent_scale,
                                self.n_modes)

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
