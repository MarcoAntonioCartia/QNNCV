"""
Phase 2 plumbing: how src/training/qgan_2d.py reaches a non-CV backend.

The trainer was written against CVQGANGenerator's surface (generate_batch
with return_ket_norm, generate_distribution_2d, weights / w_s snapshots).
The default path (--generator cv --critic dense_2d) still constructs those
objects directly and is byte-identical to before. Any other backend is
wrapped in GeneratorShim, which presents the same surface:

- generate_batch(..., return_ket_norm=True) returns ket norms of 1, so the
  trainer's ket penalty is exactly zero (= GeneratorBackend.regularizer of
  the classical and DV backends). A future backend with a real
  regularizer needs the trainer to call regularizer(aux) instead.
- snapshots go through state_dict()/load_state_dict(); the checkpoint file
  keeps the name best_weights.npy (so sweep/analysis tooling that tests
  for its presence keeps working) but holds the backend-state payload
  below, which weights_io.load_weights deliberately rejects.

BACKEND_STATE_FORMAT files are read with load_backend_state(); the generic
builder build_generator(cfg) rebuilds any backend from a run_config dict
(used by rescore.py and the trainer alike).
"""

import json
import os

import numpy as np
import tensorflow as tf

BACKEND_STATE_FORMAT = 'qnncv-backend-state-v1'


class GeneratorShim:
    def __init__(self, backend):
        self.backend = backend
        self.w_s = None            # the trainer snapshots w_s when present

    @property
    def latent_dim(self):
        return self.backend.latent_dim

    @property
    def batch_size(self):
        return self.backend.batch_size

    @property
    def trainable_variables(self):
        return self.backend.trainable_variables

    def generate_batch(self, z, xvec, yvec, return_ket_norm=False):
        probs, _aux = self.backend.generate_batch(z, xvec, yvec)
        if return_ket_norm:
            return probs, tf.ones([tf.shape(probs)[0]], dtype=probs.dtype)
        return probs

    def generate_distribution_2d(self, z, xvec, yvec, return_ket_norm=False):
        probs = self.backend.generate_distribution_2d(z, xvec, yvec)
        if return_ket_norm:
            return probs, tf.constant(1.0, dtype=probs.dtype)
        return probs

    def state_dict(self):
        return self.backend.state_dict()

    def load_state_dict(self, state):
        self.backend.load_state_dict(state)

    def describe(self):
        return self.backend.describe()

    def load(self, path):
        payload = load_backend_state(path)
        if payload['backend'] != self.backend.name:
            print(f"  Note: {path} was saved by backend {payload['backend']!r}; "
                  f"this generator is {self.backend.name!r}")
        self.backend.load_state_dict(payload['state'])


def save_backend_state(path, backend, state=None):
    payload = {'format': BACKEND_STATE_FORMAT, 'backend': backend.name,
               'state': state if state is not None else backend.state_dict(),
               'describe': backend.describe()}
    np.save(path, np.array(payload, dtype=object), allow_pickle=True)


def load_backend_state(path):
    arr = np.load(path, allow_pickle=True)
    payload = arr.item() if arr.dtype == object else None
    if not (isinstance(payload, dict) and payload.get('format') == BACKEND_STATE_FORMAT):
        raise ValueError(f"{path}: not a {BACKEND_STATE_FORMAT} file (CV checkpoints "
                         f"are read with weights_io.load_weights)")
    return payload


def parse_hidden(spec):
    return tuple(int(s) for s in str(spec).split(',') if s.strip())


def build_generator(cfg):
    """Build the generator named by a run_config-like dict (keys: generator,
    gen_hidden, dv_qubits, dv_layers, dv_reupload, dv_encoding, latent_scale,
    n_modes). Returns a
    GeneratorBackend; the CV backend is built by the trainer itself."""
    from src.backends import registry
    name = cfg.get('generator', 'cv')
    if name not in registry.GENERATOR_REGISTRY:
        raise ValueError(f"generator {name!r} not available in this environment; "
                         f"registered: {sorted(registry.GENERATOR_REGISTRY)} (the DV "
                         f"backend needs the qnncv-dv env with PennyLane)")
    # match the CV latent of the same run (the trainer passes the EncodingSpec's
    # value: 2*n_modes, or 3*n_modes under --inj-latent dedicated)
    latent_dim = int(cfg.get('latent_dim') or 2 * int(cfg.get('n_modes', 3)))
    if name == 'classical':
        from src.backends.classical import ClassicalCoordGenerator
        return ClassicalCoordGenerator(latent_dim=latent_dim,
                                       hidden_dims=parse_hidden(cfg.get('gen_hidden', '9,9')))
    if name == 'dv':
        return registry.GENERATOR_REGISTRY['dv'](
            n_qubits=int(cfg.get('dv_qubits', 10)), n_layers=int(cfg.get('dv_layers', 9)),
            latent_dim=latent_dim, latent_scale=float(cfg.get('latent_scale', 1.0)),
            reupload=bool(cfg.get('dv_reupload', False)),
            encoding=str(cfg.get('dv_encoding', 'ry')))
    raise ValueError(f"build_generator: {name!r} is built by the trainer directly")


def build_critic(cfg, xvec, yvec, seed=0):
    """Build a non-default critic from a run_config-like dict (keys: critic,
    critic_points, critic_layers, cutoff_dim). dense_2d is built by the
    trainer itself."""
    from src.backends import registry
    name = cfg.get('critic', 'dense_2d')
    if name not in registry.CRITIC_REGISTRY:
        raise ValueError(f"critic {name!r} not available in this environment; "
                         f"registered: {sorted(registry.CRITIC_REGISTRY)}")
    n_points, layers = int(cfg.get('critic_points', 64)), int(cfg.get('critic_layers', 6))
    cls = registry.CRITIC_REGISTRY[name]
    if name == 'coord_2d':
        return cls(xvec=xvec, yvec=yvec, n_points=n_points, seed=seed)
    if name == 'cv_qnn_2d':
        return cls(xvec=xvec, yvec=yvec, n_layers=layers,
                   cutoff_dim=int(cfg.get('critic_cutoff', 10)), n_points=n_points, seed=seed)
    if name == 'dv_qnn_2d':
        return cls(xvec=xvec, yvec=yvec, n_layers=layers, n_points=n_points, seed=seed)
    raise ValueError(f"build_critic: {name!r} is built by the trainer directly")


def write_backend_info(log_dir, generator_desc, critic_desc, wall_seconds, extra=None):
    """Additive provenance next to run_config.json: what was trained, its
    parameter / Hilbert-space sizes and the wall time."""
    info = {'generator': generator_desc, 'critic': critic_desc,
            'wall_seconds': round(float(wall_seconds), 1)}
    if extra:
        info.update(extra)
    with open(os.path.join(log_dir, 'backend_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, indent=2, sort_keys=True, default=str)
