"""
CV (Strawberry Fields) generator backend: an adapter around
src.quantum.circuit.CVQGANGenerator that satisfies GeneratorBackend.

Behaviour-preserving by construction: every call forwards to the wrapped
generator, so outputs are bit-identical to calling it directly (pinned by
tests/golden/test_backend_contract.py against test_circuit_forward's
construction). Checkpoints are the existing weights_io payloads (v1 raw
array for input_displacement, v2 dict with w_s for interleaved_squeeze), so
every best_weights.npy already on disk loads unchanged.
"""

import numpy as np
import tensorflow as tf

from src.backends.base import GeneratorBackend
from src.quantum import weights_io
from src.quantum.circuit import CVQGANGenerator, EncodingSpec


class CVGeneratorBackend(GeneratorBackend):
    name = 'cv'

    def __init__(self, *, n_modes=None, n_layers=None, cutoff_dim=None,
                 latent_scale=None, use_kerr=True, batch_size=None,
                 encoding=None, generator=None):
        """Build a CVQGANGenerator with the given config, or wrap an
        existing one (`generator=`; the other kwargs are then ignored)."""
        if generator is not None:
            self.gen = generator
        else:
            missing = [k for k, v in (('n_modes', n_modes), ('n_layers', n_layers),
                                      ('cutoff_dim', cutoff_dim),
                                      ('latent_scale', latent_scale)) if v is None]
            if missing:
                raise TypeError(f"CVGeneratorBackend needs {missing} (or generator=)")
            self.gen = CVQGANGenerator(
                n_modes=n_modes, n_output_modes=2, n_layers=n_layers,
                cutoff_dim=cutoff_dim, use_kerr=use_kerr,
                latent_scale=latent_scale, batch_size=batch_size,
                encoding=encoding if encoding is not None
                else EncodingSpec('input_displacement'))

    # ---- contract ---------------------------------------------------------
    @property
    def latent_dim(self):
        return self.gen.latent_dim

    @property
    def batch_size(self):
        return self.gen.batch_size

    def generate_batch(self, z, xvec, yvec):
        probs, ket_norm = self.gen.generate_batch(z, xvec, yvec,
                                                  return_ket_norm=True)
        return probs, ket_norm

    def generate_distribution_2d(self, z, xvec, yvec):
        return self.gen.generate_distribution_2d(z, xvec, yvec)

    @property
    def trainable_variables(self):
        return self.gen.trainable_variables

    def regularizer(self, aux):
        # The trainer's ket-norm penalty, verbatim: mean((1 - ket_norm)^2).
        return tf.reduce_mean(tf.square(1.0 - aux))

    def state_dict(self):
        w_s = (self.gen.w_s.numpy().copy() if self.gen.w_s is not None
               else None)
        enc = self.gen.encoding
        meta = ({'squeeze_scale': enc.squeeze_scale,
                 'train_squeeze_scale': enc.train_squeeze_scale,
                 'inj_latent': enc.inj_latent} if w_s is not None else {})
        return {'weights': self.gen.weights.numpy().copy(), 'w_s': w_s,
                'meta': meta}

    def load_state_dict(self, state):
        weights_io.apply_weights(self.gen, {
            'weights': np.asarray(state['weights']),
            'w_s': (None if state.get('w_s') is None
                    else np.asarray(state['w_s'])),
            'meta': dict(state.get('meta') or {})}, source='<state_dict>')

    def save(self, path):
        s = self.state_dict()
        weights_io.save_weights(path, s['weights'], w_s=s['w_s'],
                                meta=s['meta'] or None)

    def load(self, path):
        weights_io.apply_weights(self.gen, weights_io.load_weights(path),
                                 source=path)

    def describe(self):
        g = self.gen
        return {
            'backend': self.name,
            'n_params': self.count_params(self.trainable_variables),
            'hilbert_dim': int(g.cutoff_dim ** g.n_modes),
            'latent_dim': int(g.latent_dim),
            'n_modes': int(g.n_modes), 'n_layers': int(g.n_layers),
            'cutoff_dim': int(g.cutoff_dim),
            'encoding': g.encoding.kind,
            'inj_latent': g.encoding.inj_latent,
        }
