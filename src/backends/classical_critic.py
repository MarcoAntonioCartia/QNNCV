"""
Classical critic backend: adapter around the live Discriminator2D
(src.models.discriminators.qgan2d_discriminator, registry name 'dense_2d').
Forwards every call, so scores are identical to the direct model.
"""

from src.backends.base import CriticBackend
from src.models.discriminators.qgan2d_discriminator import Discriminator2D


class DenseCriticBackend(CriticBackend):
    name = 'dense_2d'

    def __init__(self, *, dropout_rate=None, hidden_dims=(16, 8), init_scale=0.05,
                 model=None):
        if model is None and dropout_rate is None:
            raise TypeError("DenseCriticBackend needs dropout_rate (or model=)")
        self.model = model if model is not None else Discriminator2D(
            hidden_dims=list(hidden_dims), init_scale=init_scale,
            dropout_rate=dropout_rate)

    def __call__(self, x, training=False):
        return self.model(x, training=training)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def describe(self):
        return {'backend': self.name,
                'n_params': int(sum(int(v.shape.num_elements())
                                    for v in self.model.trainable_variables)),
                'hilbert_dim': None}
