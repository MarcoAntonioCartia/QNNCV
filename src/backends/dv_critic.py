"""
Discrete-variable point-wise critic on PennyLane (Phase 1d): the qubit twin
of the CV quantum critic. D(x, y) = <Z_0> of an n-qubit circuit that
receives the grid point as fixed RY angles (point_scale * x on the even
wires, point_scale * y on the odd wires; never trainable) followed by
n_layers x [RY(w), RZ(w), CNOT ring]: 2 * n_qubits * n_layers trainable
parameters (4 qubits, 12 layers = 96, the CV critic's parity figure).
Scored with the shared Monte-Carlo point sum (src/backends/pointwise.py).

Point sets are evaluated with PennyLane parameter broadcasting (one device
call per point set) when the device supports it, else one call per point.
Requires the qnncv-dv env; the registry omits 'dv_qnn_2d' without PennyLane.
"""

import tensorflow as tf

import pennylane as qml

from src.backends.pointwise import PointwiseCritic


class DVQuantumCritic(PointwiseCritic):
    name = 'dv_qnn_2d'

    def __init__(self, *, xvec, yvec, n_qubits=4, n_layers=12, point_scale=0.5,
                 device='lightning.qubit', n_points=64, seed=0, init_sd=0.1):
        super().__init__(xvec=xvec, yvec=yvec, n_points=n_points, seed=seed)
        self.n_qubits, self.n_layers = int(n_qubits), int(n_layers)
        self.point_scale = float(point_scale)
        self.device_name = device
        self.dev = qml.device(device, wires=self.n_qubits)
        self.w = tf.Variable(tf.random.normal([self.n_layers, self.n_qubits, 2],
                                              stddev=init_sd), name='dvc_w')
        n, L = self.n_qubits, self.n_layers
        diff = 'adjoint' if device.startswith('lightning') else 'backprop'

        @qml.qnode(self.dev, interface='tf', diff_method=diff)
        def circuit(x, y, w):
            for i in range(n):
                qml.RY(x if i % 2 == 0 else y, wires=i)
            for l in range(L):
                for i in range(n):
                    qml.RY(w[l, i, 0], wires=i)
                    qml.RZ(w[l, i, 1], wires=i)
                for i in range(n):
                    qml.CNOT(wires=[i, (i + 1) % n])
            return qml.expval(qml.PauliZ(0))
        self._qnode = circuit
        self._broadcast = self._probe_broadcast()

    def _probe_broadcast(self):
        try:
            out = self._qnode(tf.constant([0.1, 0.2]), tf.constant([0.3, 0.4]), self.w)
            return tuple(out.shape) == (2,)
        except Exception:  # noqa: BLE001 - device without broadcasting
            return False

    def evaluate(self, points):
        pts = tf.cast(points, tf.float32) * self.point_scale
        xs, ys = pts[:, 0], pts[:, 1]
        if self._broadcast:
            out = self._qnode(xs, ys, self.w)
        else:
            out = tf.stack([self._qnode(xs[i], ys[i], self.w)
                            for i in range(int(pts.shape[0]))])
        return tf.cast(tf.math.real(out), tf.float32)

    @property
    def trainable_variables(self):
        return [self.w]

    def describe_extra(self):
        return {'hilbert_dim': int(2 ** self.n_qubits), 'n_qubits': self.n_qubits,
                'n_layers': self.n_layers, 'point_scale': self.point_scale,
                'device': self.device_name, 'broadcast': self._broadcast,
                'readout': '<Z0>'}
