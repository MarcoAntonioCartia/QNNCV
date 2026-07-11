"""
Dataset Generation
==================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""


def generate_dataset(family, n_samples):
    """
    Pre-generate dataset from distribution family.

    Args:
        family: DistributionFamily instance
        n_samples: Number of distributions to generate

    Returns:
        dataset: List of (distribution, params) tuples
    """
    print(f"Generating {n_samples} distributions from {family.__class__.__name__}...")
    dataset = []

    for i in range(n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Generated {i + 1}/{n_samples}...")
        dist, params = family.sample()
        dataset.append((dist, params))

    print(f"Dataset generation complete: {len(dataset)} distributions")
    return dataset
