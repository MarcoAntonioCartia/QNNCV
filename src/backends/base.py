"""
Backend contracts for the benchmark framework (Phase 0 of
docs/benchmark_plan.md).

A *generator backend* turns a batch of latent vectors into probability
densities on the 2-D grid; a *critic backend* scores critic inputs. The
trainer (src/training/qgan_2d.py) is NOT wired to these yet: every backend
is built and tested against this contract on its own first, and the
trainer learns about the registry in the final plumbing phase. Until then
the live training path is byte-identical to before.

The method set mirrors exactly what the trainer already uses on the CV
generator, plus three things it needs to stop assuming Strawberry Fields:

- `regularizer(aux)`: the backend's own training penalty on the auxiliary
  output of `generate_batch` (CV: the ket-norm penalty that keeps the
  truncated state representable; classical / DV: zero).
- `state_dict()` / `load_state_dict()`: checkpoints that are not one tensor.
- `describe()`: parameter count and Hilbert-space size, reported together
  for every backend (the fairness rule of the plan).
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class GeneratorBackend(ABC):
    """Contract every generator backend satisfies (see test_backend_contract)."""

    #: short registry name ('cv', 'classical', 'dv')
    name = 'abstract'

    @property
    @abstractmethod
    def latent_dim(self):
        """Width of the latent vectors the backend consumes."""

    @property
    @abstractmethod
    def batch_size(self):
        """Batch the backend evaluates in one call (None = any size)."""

    @abstractmethod
    def generate_batch(self, z, xvec, yvec):
        """(B, latent_dim) -> (probs (B, len(xvec), len(yvec)) float32,
        aux). `aux` is backend-specific (CV: (B,) ket norms) and is what
        `regularizer` consumes; it may be None."""

    def generate_distribution_2d(self, z, xvec, yvec):
        """One latent vector (latent_dim,) -> (len(xvec), len(yvec))."""
        probs, _ = self.generate_batch(tf.reshape(z, [1, -1]), xvec, yvec)
        return probs[0]

    @property
    @abstractmethod
    def trainable_variables(self):
        """List of tf.Variable the optimizer updates."""

    def regularizer(self, aux):
        """Differentiable scalar added to the generator loss (weighted by
        the trainer). Default: no penalty."""
        return tf.constant(0.0, dtype=tf.float32)

    @abstractmethod
    def state_dict(self):
        """Checkpoint as a dict of numpy arrays / plain values."""

    @abstractmethod
    def load_state_dict(self, state):
        """Inverse of state_dict (exact round-trip)."""

    @abstractmethod
    def describe(self):
        """dict with at least: backend, n_params, hilbert_dim, latent_dim.
        hilbert_dim = size of the simulated state space (CV: cutoff^modes,
        DV: 2^qubits, classical: None)."""

    # ---- shared helpers -------------------------------------------------
    @staticmethod
    def count_params(variables):
        return int(sum(int(np.prod(v.shape)) for v in variables))


class CriticBackend(ABC):
    """Contract every critic backend satisfies."""

    name = 'abstract'

    @abstractmethod
    def __call__(self, x, training=False):
        """Critic input (B, G, G, 1) or (B, G, G) -> scores (B, 1)."""

    @property
    @abstractmethod
    def trainable_variables(self):
        """List of tf.Variable the optimizer updates."""

    @abstractmethod
    def describe(self):
        """dict with at least: backend, n_params, hilbert_dim."""
