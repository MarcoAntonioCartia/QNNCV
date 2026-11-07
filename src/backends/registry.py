"""
Backend registries (Phase 0). Keys are the names the final plumbing phase
will expose as --generator / --critic. Nothing in the live trainer reads
these yet.

Adding a backend = one module under src/backends/ + one entry here; the
contract suite (tests/golden/test_backend_contract.py) picks it up through
`toy_factories()` below, which builds every backend at a toy size.
"""

from src.backends.cv import CVGeneratorBackend
from src.backends.classical_critic import DenseCriticBackend

GENERATOR_REGISTRY = {
    'cv': CVGeneratorBackend,
    # 'classical': ...  (Phase 1a)
    # 'dv': ...         (Phase 1b)
}

CRITIC_REGISTRY = {
    'dense_2d': DenseCriticBackend,
    # 'coord_2d', 'cv_qnn_2d', 'dv_qnn_2d'  (Phase 1c-1e)
}


def toy_factories(batch_size=4):
    """name -> zero-arg factory building that backend at a toy size, for the
    contract suite. Every registered backend MUST appear here."""
    from src.quantum.circuit import EncodingSpec
    gens = {
        'cv': lambda: CVGeneratorBackend(
            n_modes=2, n_layers=2, cutoff_dim=6, latent_scale=1.0,
            batch_size=batch_size),
        'cv/interleaved': lambda: CVGeneratorBackend(
            n_modes=2, n_layers=2, cutoff_dim=6, latent_scale=1.0,
            batch_size=batch_size,
            encoding=EncodingSpec('interleaved_squeeze', squeeze_scale=0.2,
                                  train_squeeze_scale=True)),
    }
    critics = {
        'dense_2d': lambda: DenseCriticBackend(dropout_rate=0.0),
    }
    return gens, critics
