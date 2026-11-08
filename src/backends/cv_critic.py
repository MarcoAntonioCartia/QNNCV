"""
CV quantum critic (Strawberry Fields), point-wise: D(x, y) is a 2-mode
Killoran CV-QNN that receives the grid point as a displacement
(alpha_0 = point_scale * x, alpha_1 = point_scale * y, fixed, never
trainable) and reads out <n_0> - <n_1> (mean photon numbers of the two
modes, parameter-free, bounded by the cutoff). 16 trainable parameters per
layer for 2 modes (2 interferometers x 3 + squeeze 4 + displacement 4 +
Kerr 2): 96 at 6 layers, the "parity" configuration of the 12 Jul 2026
spec (vs ~25k for the classical dense critic and 186 for the generator).

The circuit machinery is CVQGANGenerator's own (same layer, same weight
layout, same SF TF-backend batching: the point set S is the SF batch), so
the critic and the generator are the same family of circuits — only the
readout differs.
"""

import tensorflow as tf

from src.backends.pointwise import PointwiseCritic
from src.quantum.circuit import CVQGANGenerator, EncodingSpec


class CVQuantumCritic(PointwiseCritic):
    name = 'cv_qnn_2d'

    def __init__(self, *, xvec, yvec, n_layers=6, cutoff_dim=10, use_kerr=True,
                 point_scale=0.5, n_points=64, seed=0, active_sd=0.1,
                 passive_sd=0.1):
        super().__init__(xvec=xvec, yvec=yvec, n_points=n_points, seed=seed)
        n_batch = len(self.sample_points())
        self.point_scale = float(point_scale)
        self.cutoff_dim = int(cutoff_dim)
        # latent_scale=1: the front Dgate gets exactly (point_scale * x, 0)
        self.circuit = CVQGANGenerator(
            n_modes=2, n_output_modes=2, n_layers=n_layers,
            cutoff_dim=cutoff_dim, use_kerr=use_kerr, latent_scale=1.0,
            batch_size=n_batch, encoding=EncodingSpec('input_displacement'),
            active_sd=active_sd, passive_sd=passive_sd)
        n = tf.range(cutoff_dim, dtype=tf.float32)
        self._n0 = n[:, None]          # photon number weights on mode 0
        self._n1 = n[None, :]          # ... and mode 1

    def evaluate(self, points):
        pts = tf.cast(points, tf.float32) * self.point_scale
        zeros = tf.zeros_like(pts[:, :1])
        # (alpha_0 magnitude, phase 0, alpha_1 magnitude, phase 0)
        z = tf.concat([pts[:, :1], zeros, pts[:, 1:2], zeros], axis=1)
        n = int(z.shape[0])
        bs = self.circuit.batch_size or n
        # SF's batched engine is sized at construction: evaluate in chunks
        # of bs, padding the last chunk by repetition and trimming after.
        outs = []
        for start in range(0, n, bs):
            chunk = z[start:start + bs]
            m = int(chunk.shape[0])
            if m < bs:
                chunk = tf.concat([chunk, tf.repeat(chunk[-1:], bs - m, axis=0)], axis=0)
            state = self.circuit._run_circuit(chunk)
            ket = state.ket()                               # (bs, c, c) complex
            prob = tf.abs(ket) ** 2
            n0 = tf.reduce_sum(prob * self._n0[None], axis=[1, 2])
            n1 = tf.reduce_sum(prob * self._n1[None], axis=[1, 2])
            outs.append((n0 - n1)[:m])
        return tf.concat(outs, axis=0)

    @property
    def trainable_variables(self):
        return self.circuit.trainable_variables

    def describe_extra(self):
        return {'hilbert_dim': int(self.cutoff_dim ** 2),
                'n_layers': int(self.circuit.n_layers),
                'cutoff_dim': self.cutoff_dim, 'point_scale': self.point_scale,
                'readout': 'n0 - n1'}
