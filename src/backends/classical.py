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

Fairness notes recorded in describe(): hilbert_dim is None (no simulated
state space), n_params is the exact trainable count, latent_dim matches
the CV default (2 * n_modes = 6) unless overridden. z enters as plain
inputs to every evaluation (there is no "encoding" to tune); it is never
trainable, the same rule as for the quantum generators.
"""

import numpy as np
import tensorflow as tf

from src.backends.base import GeneratorBackend


class ClassicalCoordGenerator(GeneratorBackend):
    name = 'classical'

    FEATURES = ('xy', 'xyr')

    def __init__(self, *, latent_dim=6, hidden_dims=(9, 9), features='xy',
                 activation='tanh', init_scale=0.5):
        """features: 'xy' feeds the raw grid point; 'xyr' also feeds the
        radius sqrt(x^2 + y^2) (an explicit radial inductive bias, an
        ablation, not the default)."""
        if features not in self.FEATURES:
            raise ValueError(f"features must be one of {self.FEATURES}")
        self._latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.features = features
        n_in = self._latent_dim + (3 if features == 'xyr' else 2)
        init = tf.keras.initializers.RandomNormal(stddev=init_scale)
        layers = [tf.keras.Input(shape=(n_in,))]
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
        frozen config's 186 (the next size, (10, 10), has 211)."""
        n_in = latent_dim + (3 if kw.get('features') == 'xyr' else 2)
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
            feats = tf.constant(np.stack(cols, axis=1))  # (G*G, 2|3)
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
                         'features': self.features}}

    def load_state_dict(self, state):
        self.net.set_weights([np.asarray(w) for w in state['weights']])

    def describe(self):
        return {'backend': self.name,
                'n_params': self.count_params(self.trainable_variables),
                'hilbert_dim': None,
                'latent_dim': self._latent_dim,
                'hidden_dims': self.hidden_dims,
                'features': self.features}
