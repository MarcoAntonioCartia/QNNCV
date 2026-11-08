"""
Classical point-wise critic: D(x, y) is a small Keras MLP on the grid
point, scored with the same Monte-Carlo point sum as the quantum
point-wise critics (src/backends/pointwise.py). This is the classical
member of the point-wise critic family: parameter-matched to the 96-param
CV critic by default ((8, 8) hidden: 3*8 + 9*8 + 9 = 105 parameters).
"""

import tensorflow as tf

from src.backends.pointwise import PointwiseCritic


class CoordCritic(PointwiseCritic):
    name = 'coord_2d'

    def __init__(self, *, xvec, yvec, hidden_dims=(8, 8), activation='tanh',
                 init_scale=0.5, n_points=64, seed=0):
        super().__init__(xvec=xvec, yvec=yvec, n_points=n_points, seed=seed)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        init = tf.keras.initializers.RandomNormal(stddev=init_scale)
        layers = [tf.keras.Input(shape=(2,))]
        for h in self.hidden_dims:
            layers.append(tf.keras.layers.Dense(h, activation=activation,
                                                kernel_initializer=init))
        layers.append(tf.keras.layers.Dense(1, kernel_initializer=init))
        self.net = tf.keras.Sequential(layers)

    def evaluate(self, points):
        return tf.reshape(self.net(tf.cast(points, tf.float32)), [-1])

    @property
    def trainable_variables(self):
        return self.net.trainable_variables

    def describe_extra(self):
        return {'hidden_dims': self.hidden_dims}
