"""
2D-density critic for the 2D CV-QGAN
====================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).

NOTE: this module is imported by its full path
(src.models.discriminators.qgan2d_discriminator); the legacy package
__init__.py is deliberately untouched.
"""

import tensorflow as tf


class Discriminator2D(tf.keras.Model):
    """
    Discriminator for 2D probability distributions.

    Takes a 2D distribution and outputs a score (real vs fake).

    IMPORTANT: Must be kept deliberately weak to avoid overwhelming
    the quantum generator (~96 params). Even [16, 8] has ~25k params
    due to the 1600-dim input (40x40 grid).
    """

    # Keyword-only; dropout_rate has no default because its old signature
    # default (0.3) diverged from the CLI --d-dropout default (0.0);
    # build_parser() is the single source of truth.
    def __init__(self, *, hidden_dims=[16, 8], init_scale=0.05, dropout_rate):
        super().__init__()

        initializer = tf.keras.initializers.RandomNormal(stddev=init_scale)

        self.flatten = tf.keras.layers.Flatten()

        layers = []
        for dim in hidden_dims:
            layers.append(tf.keras.layers.Dense(dim, kernel_initializer=initializer))
            layers.append(tf.keras.layers.LayerNormalization())
            layers.append(tf.keras.layers.LeakyReLU(0.2))
            if dropout_rate > 0:
                layers.append(tf.keras.layers.Dropout(dropout_rate))

        layers.append(tf.keras.layers.Dense(1, kernel_initializer=initializer))

        self.net = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.flatten(x)
        return self.net(x, training=training)


# Seam registry for future critic architectures (swappable in experiments).
# The training loop still constructs Discriminator2D directly — this is an
# interface-only seam, no behavior change.
CRITIC_REGISTRY = {
    'dense_2d': Discriminator2D,
}
