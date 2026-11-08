"""
Classical generator backend (Phase 1a of docs/benchmark_plan.md): a Keras
coordinate network f(x, y, z) evaluated on the grid.

Why a coordinate network and not an MLP that emits the 1,600 grid values
directly: the CV generator has 186 trainable parameters and produces a
40x40 density through physics, not through 1,600 output weights. A plain
MLP would spend 1,600 parameters per hidden unit on its last layer alone,
so no parameter-matched comparison is possible. A coordinate network
takes a grid point (x, y) and the latent z and returns one value; the
density is softplus(f) over the grid, normalised to sum to 1. Its
parameter count is independent of the grid size, so it can be matched to
the CV generator (default hidden sizes (9, 9) with latent_dim 6 -> 181
parameters, the closest size to 186) and a larger reference (`large()`,
4,801 parameters) gives the upper bound of the architecture family.

Two standard upgrades for the spectral bias of a tiny tanh coordinate net
(both parameter-neutral or nearly so, see docs/references_benchmark.md):
activation='sine' is a SIREN (Sitzmann et al. 2020: sin(w0 * Wx + b) with
the paper's uniform initialisation, w0 = 30 on the first layer); fourier=K
prepends K random Fourier features of the grid point (Tancik et al. 2020:
[sin(2 pi B p), cos(2 pi B p)], B ~ N(0, sigma^2) of shape (K, 2), fixed,
never trainable; sigma = 1 on the [-3, 3] domain, i.e. up to a few cycles
across the plane). The latent z is always fed raw.

Fairness notes recorded in describe(): hilbert_dim is None (no simulated
state space), n_params is the exact trainable count, latent_dim matches
the CV default (2 * n_modes = 6) unless overridden. z enters as plain
inputs to every evaluation (there is no "encoding" to tune); it is never
trainable, the same rule as for the quantum generators.
"""

import numpy as np
import tensorflow as tf

from src.backends.base import GeneratorBackend


class _Sine(tf.keras.layers.Layer):
    """SIREN activation sin(w0 * x); w0 is a fixed hyper-parameter."""

    def __init__(self, w0, **kw):
        super().__init__(**kw)
        self.w0 = float(w0)

    def call(self, x):
        return tf.sin(self.w0 * x)


class ClassicalCoordGenerator(GeneratorBackend):
    name = 'classical'

    FEATURES = ('xy', 'xyr')
    ACTIVATIONS = ('tanh', 'sine')
    FOURIER_SIGMA = 1.0
    SIREN_W0 = 30.0

    @classmethod
    def n_inputs(cls, latent_dim=6, features='xy', fourier=0):
        return int(latent_dim) + (3 if features == 'xyr' else 2) + 2 * int(fourier)

    def __init__(self, *, latent_dim=6, hidden_dims=(9, 9), features='xy',
                 activation='tanh', init_scale=0.5, fourier=0, seed=0):
        """features: 'xy' feeds the raw grid point; 'xyr' also feeds the
        radius sqrt(x^2 + y^2) (an explicit radial inductive bias, an
        ablation, not the default). activation: 'tanh' or 'sine' (SIREN).
        fourier: number K of random Fourier features of the point (0 = off);
        their frequency matrix is drawn once from `seed` and never trained."""
        if features not in self.FEATURES:
            raise ValueError(f"features must be one of {self.FEATURES}")
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation must be one of {self.ACTIVATIONS}")
        self._latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.features, self.activation = features, activation
        self.fourier = int(fourier)
        self.fourier_B = (np.random.RandomState(seed).normal(0.0, self.FOURIER_SIGMA,
                                                             size=(self.fourier, 2))
                          .astype(np.float32) if self.fourier else None)
        n_in = self.n_inputs(self._latent_dim, features, self.fourier)
        layers = [tf.keras.Input(shape=(n_in,))]
        if activation == 'sine':
            # Sitzmann et al. 2020, SIREN init: first layer U(-1/n, 1/n),
            # hidden layers U(-sqrt(6/n)/w0, sqrt(6/n)/w0); sin(w0 * .) after each
            fan = n_in
            for k, h in enumerate(self.hidden_dims):
                bound = (1.0 / fan) if k == 0 else (np.sqrt(6.0 / fan) / self.SIREN_W0)
                layers.append(tf.keras.layers.Dense(
                    h, kernel_initializer=tf.keras.initializers.RandomUniform(-bound, bound)))
                layers.append(_Sine(self.SIREN_W0))
                fan = h
            bound = np.sqrt(6.0 / fan) / self.SIREN_W0
            layers.append(tf.keras.layers.Dense(
                1, kernel_initializer=tf.keras.initializers.RandomUniform(-bound, bound)))
        else:
            init = tf.keras.initializers.RandomNormal(stddev=init_scale)
            for h in self.hidden_dims:
                layers.append(tf.keras.layers.Dense(h, activation=activation,
                                                    kernel_initializer=init))
            layers.append(tf.keras.layers.Dense(1, kernel_initializer=init))
        self.net = tf.keras.Sequential(layers)
        self._grid_cache = {}

    @classmethod
    def parameter_matched(cls, target=186, latent_dim=6, **kw):
        """Two-layer (h, h) network whose parameter count is closest to
        target: 181 at (9, 9) for the default latent_dim 6 against the CV
        frozen config's 186 (the next size, (10, 10), has 211); with
        fourier=8 the input widens to 24 and (6, 6) gives 199."""
        n_in = cls.n_inputs(latent_dim, kw.get('features', 'xy'), kw.get('fourier', 0))
        count = lambda h: (n_in + 1) * h + (h + 1) * h + (h + 1)  # noqa: E731
        h = min(range(1, 256), key=lambda h: abs(count(h) - target))
        return cls(latent_dim=latent_dim, hidden_dims=(h, h), **kw)

    @classmethod
    def large(cls, latent_dim=6, **kw):
        """Upper-reference size of the same architecture (4,801 params)."""
        return cls(latent_dim=latent_dim, hidden_dims=(64, 64), **kw)

    # ---- grid features ----------------------------------------------------
    def _grid_features(self, xvec, yvec):
        key = (tuple(np.round(np.asarray(xvec, float), 9)),
               tuple(np.round(np.asarray(yvec, float), 9)))
        feats = self._grid_cache.get(key)
        if feats is None:
            X, Y = np.meshgrid(np.asarray(xvec, np.float32),
                               np.asarray(yvec, np.float32), indexing='ij')
            cols = [X.ravel(), Y.ravel()]
            if self.features == 'xyr':
                cols.append(np.sqrt(X.ravel() ** 2 + Y.ravel() ** 2))
            if self.fourier:
                proj = 2.0 * np.pi * np.stack([X.ravel(), Y.ravel()], 1) @ self.fourier_B.T
                cols += [np.sin(proj[:, k]) for k in range(self.fourier)]
                cols += [np.cos(proj[:, k]) for k in range(self.fourier)]
            feats = tf.constant(np.stack(cols, axis=1).astype(np.float32))  # (G*G, n_in - L)
            self._grid_cache[key] = (feats, X.shape)
            return feats, X.shape
        return feats

    # ---- contract ---------------------------------------------------------
    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def batch_size(self):
        return None

    def generate_batch(self, z, xvec, yvec):
        z = tf.convert_to_tensor(z, tf.float32)
        if int(z.shape[-1]) != self._latent_dim:
            raise ValueError(f"latent vector has {int(z.shape[-1])} "
                             f"components; expected latent_dim="
                             f"{self._latent_dim}")
        feats, shape = self._grid_features(xvec, yvec)
        B, P = int(z.shape[0]), int(feats.shape[0])
        zt = tf.repeat(z[:, None, :], P, axis=1)                 # (B, P, L)
        ft = tf.repeat(feats[None, :, :], B, axis=0)            # (B, P, F)
        inp = tf.reshape(tf.concat([ft, zt], axis=-1), [B * P, -1])
        f = tf.reshape(self.net(inp), [B, P])
        dens = tf.nn.softplus(f) + 1e-12
        dens = dens / tf.reduce_sum(dens, axis=1, keepdims=True)
        return tf.reshape(dens, [B, shape[0], shape[1]]), None

    @property
    def trainable_variables(self):
        return self.net.trainable_variables

    def state_dict(self):
        return {'weights': [w.copy() for w in self.net.get_weights()],
                'meta': {'latent_dim': self._latent_dim,
                         'hidden_dims': self.hidden_dims,
                         'features': self.features, 'activation': self.activation,
                         'fourier': self.fourier,
                         'fourier_B': None if self.fourier_B is None else self.fourier_B.copy()}}

    def load_state_dict(self, state):
        self.net.set_weights([np.asarray(w) for w in state['weights']])
        B = state.get('meta', {}).get('fourier_B')
        if B is not None:
            if self.fourier_B is None or np.asarray(B).shape != self.fourier_B.shape:
                raise ValueError("classical state has Fourier features this instance lacks")
            self.fourier_B = np.asarray(B, np.float32)
            self._grid_cache = {}

    def describe(self):
        return {'backend': self.name,
                'n_params': self.count_params(self.trainable_variables),
                'hilbert_dim': None,
                'latent_dim': self._latent_dim,
                'hidden_dims': self.hidden_dims,
                'features': self.features, 'activation': self.activation,
                'fourier': self.fourier}
