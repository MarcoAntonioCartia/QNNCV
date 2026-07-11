"""
Generator validation over random z draws
========================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np
import tensorflow as tf

from src.metrics.wasserstein import compute_wasserstein_2d
from src.metrics.energy import energy_distances


def validate(generator, canonical_target, val_set, xvec, yvec, ed_ctx=None,
             n_samples=10):
    """
    Validate generator over n_samples random z draws.

    Metrics (all marginal-W1 based, grid-cell units):
      canonical_w1: mean W1 to the canonical family member. Legacy metric,
          kept for continuity with earlier runs (rewards collapse-to-canonical).
      nearest_w1:  mean over samples of the W1 to the CLOSEST validation-set
          member. Does not penalize a generator that learned family variety.
      diversity:   mean pairwise W1 between the generated samples themselves.
          ~0 means z-ignoring mode collapse. Compare against the validation
          set's own self-diversity printed at startup.

    Caveat: nearest_w1 alone cannot see collapse onto a single val member --
    always read it together with diversity.

    If ed_ctx (from build_energy_distance_context) is given, also reports
    full-2D energy distances (x/y units, NOT grid cells like the W1s):
      canonical_ed / nearest_ed: analogous to the W1 metrics but sensitive
          to the full 2D geometry, not just the marginals.

    Returns:
        dict with keys 'canonical_w1', 'nearest_w1', 'diversity',
        'canonical_ed', 'nearest_ed' (EDs are NaN when ed_ctx is None).
    """
    samples = []
    B = getattr(generator, 'batch_size', None)
    if B:
        # Batched engine: fill the sample budget a batch at a time
        attempts = 0
        while len(samples) < n_samples and attempts < 10:
            attempts += 1
            z = tf.random.normal([B, generator.latent_dim])
            batch = generator.generate_batch(z, xvec, yvec).numpy()
            for b in range(B):
                if len(samples) < n_samples and np.isfinite(batch[b]).all():
                    samples.append(batch[b])
    else:
        for _ in range(n_samples):
            z = tf.random.normal([generator.latent_dim])
            fake = generator.generate_distribution_2d(z, xvec, yvec).numpy()
            if np.isfinite(fake).all():
                samples.append(fake)

    if not samples:
        return {'canonical_w1': float('inf'),
                'nearest_w1': float('inf'),
                'diversity': 0.0,
                'canonical_ed': float('inf'),
                'nearest_ed': float('inf')}

    canonical_w1s, nearest_w1s = [], []
    canonical_eds, nearest_eds = [], []
    for fake in samples:
        w_canon = compute_wasserstein_2d(canonical_target, fake)
        if np.isfinite(w_canon):
            canonical_w1s.append(w_canon)
        w_near = min(compute_wasserstein_2d(member, fake)
                     for member, _ in val_set)
        if np.isfinite(w_near):
            nearest_w1s.append(w_near)
        if ed_ctx is not None:
            ed_canon, ed_near = energy_distances(fake, ed_ctx)
            if np.isfinite(ed_canon):
                canonical_eds.append(ed_canon)
            if np.isfinite(ed_near):
                nearest_eds.append(ed_near)

    pair_w1s = [compute_wasserstein_2d(samples[i], samples[j])
                for i in range(len(samples))
                for j in range(i + 1, len(samples))]
    pair_w1s = [w for w in pair_w1s if np.isfinite(w)]

    return {
        'canonical_w1': np.mean(canonical_w1s) if canonical_w1s else float('inf'),
        'nearest_w1': np.mean(nearest_w1s) if nearest_w1s else float('inf'),
        'diversity': np.mean(pair_w1s) if pair_w1s else 0.0,
        'canonical_ed': np.mean(canonical_eds) if canonical_eds else float('nan'),
        'nearest_ed': np.mean(nearest_eds) if nearest_eds else float('nan'),
    }
