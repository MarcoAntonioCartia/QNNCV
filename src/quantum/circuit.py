"""
Symbolic CV-QNN circuit and generator
=====================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor), plus
SEAM 1: the latent-encoding site is an EncodingSpec strategy. Only
'input_displacement' (the original front-Dgate encoding) is implemented;
it reproduces the previous inline code gate-for-gate (same gate types and
order -> identical sf.Program up to parameter names, golden-tested; the
names carry a per-instance prefix since the sympy Symbol-cache fix).
"""

# Compatibility patch: SF still calls scipy.integrate.simps, removed in
# modern SciPy. Idempotent duplicate of the root-script guard so this
# module also imports standalone.
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import itertools

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops

from src.quantum.hermite import compute_hermite_basis

# Per-instance prefix for symbolic parameter names. strawberryfields'
# FreeParameter subclasses sympy.Symbol, and sympy caches Symbols by name, so
# two programs that both call params("z_0") share one object and the values
# bound in one program leak into the other (seen as "Parameter can be either
# a scalar or a vector of length B" once two generators with different batch
# sizes live in one process). Names never enter the numerics.
_GENERATOR_UID = itertools.count()


class EncodingSpec:
    """How the latent vector z enters the circuit (SEAM 1).

    kinds:
      'input_displacement' — the legacy behavior: z displaces the vacuum
          at the circuit front, Dgate(z[2m] * latent_scale, z[2m+1]) per mode.
      'interleaved_squeeze' — the front displacement PLUS mid-circuit
          z-injection: in each injected layer the squeeze magnitude becomes
          r = w_r[l,m] + w_s[l,m] * tanh(z[2m]), where w_s is an
          (n_layers, n_modes) gain filled with `squeeze_scale`. The gain is
          FIXED (non-trainable) unless `train_squeeze_scale` is True, in
          which case it is a trainable variable initialized at
          `squeeze_scale`. At w_s == 0 the model is exactly the
          input_displacement baseline, so the defaults (scale 0, fixed)
          make 'interleaved_squeeze' a numerically identical control arm;
          the original learned-gain design is train_squeeze_scale=True with
          squeeze_scale=0. The symbolic program is unchanged; the injection
          lives in the args-dict binding (CVQGANGenerator._run_circuit).

    inj_layers: iterable of layer indices to inject into (interleaved only;
        None = all layers). r_max: optional hard clip |r| <= r_max
        (interleaved only; None = no clip). squeeze_scale: value of the
        gain (interleaved only; default 0.0). train_squeeze_scale: whether
        the gain is trainable (interleaved only; default False).
    inj_latent (interleaved only): which latent component the squeeze
        channel of mode m reads. 'shared' (default) = z[2m], the component
        that also sets mode m's front-displacement magnitude, so ring size
        and position are driven together (latent_dim = 2*n_modes).
        'dedicated' = n_modes extra components z[2*n_modes + m] read ONLY
        by the squeeze channel, so size and position decouple
        (latent_dim = 3*n_modes). The front encoding always uses the first
        2*n_modes components; only those become symbolic circuit params.
    """

    KINDS = ('input_displacement', 'interleaved_squeeze')
    INJ_LATENTS = ('shared', 'dedicated')

    def __init__(self, kind='input_displacement', *, inj_layers=None,
                 r_max=None, squeeze_scale=0.0, train_squeeze_scale=False,
                 inj_latent='shared'):
        if kind not in self.KINDS:
            raise ValueError(f"Unknown encoding kind: {kind!r}. "
                             f"Available: {list(self.KINDS)}")
        if inj_latent not in self.INJ_LATENTS:
            raise ValueError(f"Unknown inj_latent: {inj_latent!r}. "
                             f"Available: {list(self.INJ_LATENTS)}")
        if kind == 'input_displacement':
            if inj_layers is not None or r_max is not None:
                raise ValueError(
                    "inj_layers/r_max only apply to 'interleaved_squeeze'")
            if float(squeeze_scale) != 0.0 or bool(train_squeeze_scale):
                raise ValueError(
                    "squeeze_scale/train_squeeze_scale only apply to "
                    "'interleaved_squeeze'")
            if inj_latent != 'shared':
                raise ValueError(
                    "inj_latent only applies to 'interleaved_squeeze'")
        self.kind = kind
        self.inj_layers = (None if inj_layers is None
                           else tuple(sorted(int(l) for l in inj_layers)))
        self.r_max = None if r_max is None else float(r_max)
        self.squeeze_scale = float(squeeze_scale)
        self.train_squeeze_scale = bool(train_squeeze_scale)
        self.inj_latent = inj_latent

    def resolve_inj_layers(self, n_layers):
        """Concrete tuple of injected layer indices (all layers if None)."""
        if self.inj_layers is None:
            return tuple(range(n_layers))
        if len(self.inj_layers) == 0:
            raise ValueError("inj_layers must not be empty")
        if len(set(self.inj_layers)) != len(self.inj_layers):
            raise ValueError(f"inj_layers has duplicates: {self.inj_layers}")
        bad = [l for l in self.inj_layers if not 0 <= l < n_layers]
        if bad:
            raise ValueError(
                f"inj_layers {bad} out of range [0, {n_layers})")
        return self.inj_layers

    def z_index_for_mode(self, mode, n_modes=None):
        """Latent component driving mode `mode`'s squeeze modulation.

        'shared': z[2m], the component that also drives mode m's
        front-displacement magnitude. 'dedicated': z[2*n_modes + m], an
        extra component nothing else reads (n_modes required).
        """
        if self.inj_latent == 'dedicated':
            if n_modes is None:
                raise ValueError("z_index_for_mode needs n_modes for "
                                 "inj_latent='dedicated'")
            return 2 * n_modes + mode
        return 2 * mode

    def n_front_latents(self, n_modes):
        """Latent components consumed by the front encoding (magnitude +
        phase per mode) = the symbolic z params of the circuit program."""
        return 2 * n_modes

    def latent_dim(self, n_modes):
        """Dimension of the sampled latent vector: the front components
        plus, for 'dedicated', one squeeze-channel component per mode."""
        if self.kind == 'interleaved_squeeze' and self.inj_latent == 'dedicated':
            return 3 * n_modes
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
        # Keyword-only. cutoff_dim/latent_scale have no default because their
        # old signature defaults (10 / 1.0) diverged from the CLI defaults
        # (12 / 0.3); build_parser() is the single source of truth.
        *,
        n_modes: int = 2,
        n_output_modes: int = 2,    # Modes to measure (always 2 for 2D output)
        n_layers: int = 6,
        cutoff_dim: int,
        use_kerr: bool = True,
        latent_scale: float,        # Scale for latent vector encoding
        active_sd: float = 0.1,     # Init std for active params (squeeze, displacement, kerr)
        passive_sd: float = 0.1,    # Init std for passive params (interferometer)
        batch_size=None,            # SF TF backend batching (None/1 = unbatched);
                                    # deliberately NOT the CLI --batch-size (that is
                                    # the training batch; None here means no batched engine)
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

        # interleaved_squeeze: the z-squeeze gain w_s, filled with the
        # encoding's squeeze_scale (0 => exactly the input_displacement
        # baseline). A non-trainable tf.Variable when the scale is fixed, so
        # .assign()/.numpy() (warm starts, checkpoints) work in both modes
        # while trainable_variables excludes it. tf.fill is RNG-free — the
        # random stream is identical in every mode.
        self.w_s = None
        if self.encoding.kind == 'interleaved_squeeze':
            self._inj_layers = self.encoding.resolve_inj_layers(n_layers)
            init_ws = tf.fill([n_layers, n_modes],
                              tf.constant(self.encoding.squeeze_scale,
                                          dtype=tf.float32))
            self.w_s = tf.Variable(init_ws,
                                   trainable=self.encoding.train_squeeze_scale,
                                   name='qnn_ws')

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
        print(f"  Encoding: {self.encoding.kind}")
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
        # Only the front-encoding components become symbolic params (2 per
        # mode, exactly as before); dedicated squeeze-channel components are
        # consumed in the args-dict binding and never enter the program.
        # Names carry a per-instance prefix (see _GENERATOR_UID); every
        # binding goes through param.name, so nothing else depends on them.
        uid = next(_GENERATOR_UID)
        self.latent_params = [self.prog.params(f"g{uid}_z_{i}")
                              for i in range(self.encoding.n_front_latents(self.n_modes))]

        # Create symbolic parameters for QNN weights
        self.weight_params = [self.prog.params(f"g{uid}_w_{i}")
                              for i in range(self.total_params)]

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
        width = int(z.shape[-1])
        if width != self.latent_dim:
            raise ValueError(
                f"latent vector has {width} components; this generator "
                f"expects latent_dim={self.latent_dim} (encoding "
                f"{self.encoding.kind}, inj_latent="
                f"{self.encoding.inj_latent})")

        # Map latent vector(s): one (batch_size,) tensor per parameter when batched
        for i, param in enumerate(self.latent_params):
            mapping[param.name] = z[:, i] if batched_z else z[i]

        # Map QNN weights (scalars; broadcast over the batch)
        for i, param in enumerate(self.weight_params):
            mapping[param.name] = self.weights[i]

        # interleaved_squeeze: override the squeeze-MAGNITUDE bindings with
        # r = w_r + w_s * tanh(z_component) — (B,) batched, scalar otherwise.
        # The symbolic program is untouched; only the bound tensors change.
        # At w_s == 0 this is w_r + 0.0, i.e. the exact baseline.
        if self.w_s is not None:
            M = self.interferometer_params
            for l in self._inj_layers:
                base = l * self.params_per_layer + M
                for m in range(self.n_modes):
                    # Consumption order: Sgate(w[idx], w[idx+1]) per mode
                    idx = base + 2 * m
                    zi = self.encoding.z_index_for_mode(m, self.n_modes)
                    zc = z[:, zi] if batched_z else z[zi]
                    r = self.weights[idx] + self.w_s[l, m] * tf.tanh(zc)
                    if self.encoding.r_max is not None:
                        r = tf.clip_by_value(r, -self.encoding.r_max,
                                             self.encoding.r_max)
                    mapping[self.weight_params[idx].name] = r

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
        # A fixed squeeze scale is a non-trainable Variable: excluded here,
        # so the optimizer and gradient tape never see it.
        if self.w_s is not None and self.w_s.trainable:
            return [self.weights, self.w_s]
        return [self.weights]

    def get_config(self):
        return {
            'encoding': self.encoding.kind,
            'squeeze_scale': self.encoding.squeeze_scale,
            'train_squeeze_scale': self.encoding.train_squeeze_scale,
            'inj_latent': self.encoding.inj_latent,
            'n_modes': self.n_modes,
            'n_output_modes': self.n_output_modes,
            'n_ancilla': self.n_ancilla,
            'n_layers': self.n_layers,
            'cutoff_dim': self.cutoff_dim,
            'use_kerr': self.use_kerr,
            'latent_dim': self.latent_dim,
            'total_params': self.total_params,
        }
