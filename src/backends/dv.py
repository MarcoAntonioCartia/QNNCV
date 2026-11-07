"""
Discrete-variable (qubit) generator backend on PennyLane (Phase 1b of
docs/benchmark_plan.md). Requires the cloned env `qnncv-dv` (PennyLane +
pennylane-lightning); in the base `qnncv` env this module does not import
and the registry simply omits 'dv'.

Design (the June-2025 sketch, made concrete):
- grid: a 2^k x 2^k density is the Born distribution of n = 2k qubits,
  the first k qubits indexing the row (y) and the last k the column (x),
  so the output follows src/families/base.py's [y, x] convention. 32x32 =
  10 qubits (hilbert_dim 1024), 16x16 = 8. The 40x40 grid is NOT a power of
  two: `grid_sizes` lists what this backend can produce, and the benchmark
  rule is one grid for every backend (the plan's open decision).
- encoding: z (latent_dim, default 6 like the CV generator) enters as
  fixed RY angles on the qubits, cyclically (angle encoding; never
  trainable, the January-2026 rule). latent_scale multiplies z first.
- layers: n_layers x [RY(w), RZ(w) on every qubit, CNOT ring] with w of
  shape (n_layers, n_qubits, 2): 2*n*L trainable parameters (10 qubits,
  9 layers = 180, parameter-matched to the CV's 186).
- readout: qml.probs over all wires, reshaped to (G, G); sums to 1.
- differentiation: lightning.qubit with diff_method='best' (adjoint does
  not support probs), TF interface; measured 0.13 s per sample fwd+bwd at
  10 qubits / 3 layers on this PC.
"""

import numpy as np
import tensorflow as tf

import pennylane as qml

from src.backends.base import GeneratorBackend


class DVGeneratorBackend(GeneratorBackend):
    name = 'dv'

    def __init__(self, *, n_qubits=10, n_layers=9, latent_dim=6, latent_scale=1.0,
                 device='lightning.qubit', init_sd=0.1):
        if n_qubits % 2:
            raise ValueError("n_qubits must be even (k qubits per axis)")
        self.n_qubits, self.n_layers = int(n_qubits), int(n_layers)
        self.k = self.n_qubits // 2
        self.grid = 2 ** self.k
        self.grid_sizes = (self.grid,)
        self._latent_dim = int(latent_dim)
        self.latent_scale = float(latent_scale)
        self.device_name = device
        self.dev = qml.device(device, wires=self.n_qubits)
        self.w = tf.Variable(tf.random.normal([self.n_layers, self.n_qubits, 2],
                                              stddev=init_sd), name='dv_w')
        n, L = self.n_qubits, self.n_layers
        diff = 'best' if device.startswith('lightning') else 'backprop'

        @qml.qnode(self.dev, interface='tf', diff_method=diff)
        def circuit(z, w):
            for i in range(n):
                qml.RY(z[i % self._latent_dim], wires=i)
            for l in range(L):
                for i in range(n):
                    qml.RY(w[l, i, 0], wires=i)
                    qml.RZ(w[l, i, 1], wires=i)
                for i in range(n):
                    qml.CNOT(wires=[i, (i + 1) % n])
            return qml.probs(wires=range(n))
        self._qnode = circuit

    # ---- contract ---------------------------------------------------------
    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def batch_size(self):
        return None

    def _check_grid(self, xvec, yvec):
        if len(xvec) != self.grid or len(yvec) != self.grid:
            raise ValueError(f"dv backend with {self.n_qubits} qubits produces a "
                             f"{self.grid}x{self.grid} grid, got {len(yvec)}x{len(xvec)}; "
                             f"use --grid-size {self.grid} for every backend")

    def generate_batch(self, z, xvec, yvec):
        self._check_grid(xvec, yvec)
        z = tf.convert_to_tensor(z, tf.float32)
        if int(z.shape[-1]) != self._latent_dim:
            raise ValueError(f"latent vector has {int(z.shape[-1])} components; "
                             f"expected latent_dim={self._latent_dim}")
        outs = []
        for b in range(int(z.shape[0])):
            p = self._qnode(z[b] * self.latent_scale, self.w)
            outs.append(tf.cast(tf.math.real(p), tf.float32))
        probs = tf.stack(outs, axis=0)                       # (B, 2^n)
        return tf.reshape(probs, [-1, self.grid, self.grid]), None

    @property
    def trainable_variables(self):
        return [self.w]

    def state_dict(self):
        return {'weights': self.w.numpy().copy(),
                'meta': {'n_qubits': self.n_qubits, 'n_layers': self.n_layers,
                         'latent_dim': self._latent_dim, 'latent_scale': self.latent_scale}}

    def load_state_dict(self, state):
        w = np.asarray(state['weights'], np.float32)
        if tuple(w.shape) != tuple(self.w.shape):
            raise ValueError(f"dv weights shape {w.shape} != {tuple(self.w.shape)}")
        self.w.assign(w)

    def describe(self):
        return {'backend': self.name,
                'n_params': self.count_params(self.trainable_variables),
                'hilbert_dim': int(2 ** self.n_qubits),
                'latent_dim': self._latent_dim,
                'n_qubits': self.n_qubits, 'n_layers': self.n_layers,
                'grid': self.grid, 'device': self.device_name,
                'encoding': 'angle_ry_cyclic'}
