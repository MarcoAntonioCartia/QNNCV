"""
Backend registries (Phase 0). Keys are the names the final plumbing phase
will expose as --generator / --critic. Nothing in the live trainer reads
these yet.

Adding a backend = one module under src/backends/ + one entry here; the
contract suite (tests/golden/test_backend_contract.py) picks it up through
`toy_factories()` below, which builds every backend at a toy size.
"""

from src.backends.cv import CVGeneratorBackend
from src.backends.classical import ClassicalCoordGenerator
from src.backends.classical_critic import DenseCriticBackend

GENERATOR_REGISTRY = {
    'cv': CVGeneratorBackend,
    'classical': ClassicalCoordGenerator,   # Phase 1a (coordinate network)
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
        # learned gain (--train-squeeze-scale): w_s trainable, 36 params
        'cv/interleaved-trainable': lambda: CVGeneratorBackend(
            n_modes=2, n_layers=2, cutoff_dim=6, latent_scale=1.0,
            batch_size=batch_size,
            encoding=EncodingSpec('interleaved_squeeze', squeeze_scale=0.2,
                                  train_squeeze_scale=True)),
        # CLI default since 50d80b5: FIXED gain (32 params) + the dedicated
        # latent of ef1301e (latent_dim 3*n_modes)
        'cv/interleaved-fixed-dedicated': lambda: CVGeneratorBackend(
            n_modes=2, n_layers=2, cutoff_dim=6, latent_scale=1.0,
            batch_size=batch_size,
            encoding=EncodingSpec('interleaved_squeeze', squeeze_scale=0.2,
                                  inj_latent='dedicated')),
        # parameter-matched to the CV frozen config (190 vs 186) and the
        # radial-feature ablation
        'classical': lambda: ClassicalCoordGenerator.parameter_matched(),
        'classical/xyr': lambda: ClassicalCoordGenerator(features='xyr'),
    }
    critics = {
        'dense_2d': lambda: DenseCriticBackend(dropout_rate=0.0),
    }
    return gens, critics
