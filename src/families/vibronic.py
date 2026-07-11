"""
Vibronic (molecular Doktorov state) distribution family
=======================================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
NOTE: the Strawberry Fields apps imports stay INSIDE __init__ (lazy) so
importing this module does not pay the SF-apps import cost.
"""

import numpy as np

from src.families.base import DistributionFamily


class VibronicFamily(DistributionFamily):
    """2-mode position-quadrature marginal of a molecular vibronic Doktorov state.

    The Doktorov operator U = D(α)·Interferometer(U2)·Squeeze(r)·Interferometer(U1)
    is a sequence of Gaussian gates, so the resulting N-mode state is Gaussian and
    its 2-mode position marginal P(x_i, x_j) is an analytical bivariate Gaussian
    obtainable from the position sub-block of the full state covariance.

    Each family member is one (i, j) pair of normal modes from the molecule's
    Doktorov state. Canonical = the pair with largest combined |μ| (the
    spectroscopically "loudest" mode pair).

    Default molecule: Formic (14 modes, T=0). Also supports Water (3 modes) and
    Pyrrole (24 modes) -- but the time-resolved Water/Pyrrole datasets need
    additional handling not yet implemented here.
    """

    def __init__(self, grid_size=40, x_range=3.0,
                 molecule='formic', canonical_pair=None,
                 standardize=True, target_std=0.5):
        super().__init__(grid_size, x_range)

        # Lazy SF imports so non-vibronic runs don't pay the import cost
        from strawberryfields.apps.data import Formic
        from strawberryfields.apps.qchem.vibronic import gbs_params, VibronicTransition
        import strawberryfields as sf

        mol_classes = {'formic': Formic}
        # Water and Pyrrole are time-resolved; deferred for now.
        if molecule not in mol_classes:
            raise ValueError(
                f"Unknown molecule '{molecule}'. Currently supported: "
                f"{list(mol_classes.keys())}")
        mol = mol_classes[molecule]()
        self.molecule = molecule

        # Build Doktorov parameters from molecule spectroscopic data.
        # At T=0, the two-mode squeezing vector t is all zeros and unused here.
        # mol.modes can include thermal-pair modes (= 2 * normal modes); the
        # Doktorov state itself lives on the n normal modes that U1 acts on.
        _t, U1, r, U2, alpha = gbs_params(mol.w, mol.wp, mol.Ud, mol.delta, T=mol.T)
        self.n_modes = U1.shape[0]

        # Run the Gaussian program ONCE to get full (cov, means) of the
        # Doktorov state. At T=0, gbs_params returns t=0 so no two-mode squeezing.
        prog = sf.Program(self.n_modes)
        with prog.context as q:
            VibronicTransition(U1, r, U2, alpha) | q
        eng = sf.Engine('gaussian')
        state = eng.run(prog).state
        self.cov_full = state.cov()        # shape (2N, 2N), xxpp ordering
        self.means_full = state.means()    # shape (2N,),     xxpp ordering

        # Position-only sub-block: indices [0..N-1] of means and cov give the
        # x-quadrature marginal (xxpp ordering means first N are positions).
        self.means_x = self.means_full[:self.n_modes]
        self.cov_xx = self.cov_full[:self.n_modes, :self.n_modes]

        # Enumerate pairs, precompute marginals, rank by displacement
        self.pairs = []
        for i in range(self.n_modes):
            for j in range(i + 1, self.n_modes):
                mu = np.array([self.means_x[i], self.means_x[j]])
                V = np.array([[self.cov_xx[i, i], self.cov_xx[i, j]],
                              [self.cov_xx[j, i], self.cov_xx[j, j]]])
                self.pairs.append({'pair': (i, j), 'mu': mu, 'V': V,
                                   'norm_mu': float(np.linalg.norm(mu))})
        self.pairs.sort(key=lambda p: -p['norm_mu'])

        # Canonical = most-displaced pair, unless caller pinned one
        if canonical_pair is None:
            self.canonical_pair_idx = 0
        else:
            target = tuple(sorted(canonical_pair))
            for k, p in enumerate(self.pairs):
                if p['pair'] == target:
                    self.canonical_pair_idx = k
                    break
            else:
                raise ValueError(f"Pair {canonical_pair} not in family")

        # Standardization so the canonical bivariate Gaussian fits comfortably
        # within [-x_range, x_range]: shift its mean to origin, rescale so the
        # larger marginal std equals target_std.
        if standardize:
            mu_c = self.pairs[self.canonical_pair_idx]['mu']
            V_c = self.pairs[self.canonical_pair_idx]['V']
            self.shift = mu_c.copy()
            max_std = float(max(np.sqrt(V_c[0, 0]), np.sqrt(V_c[1, 1])))
            self.scale = max_std / target_std if max_std > 0 else 1.0
        else:
            self.shift = np.zeros(2)
            self.scale = 1.0

    def _render(self, mu, V):
        """Render a bivariate Gaussian (mu, V) on the 40x40 grid, standardized."""
        mu_std = (mu - self.shift) / self.scale
        V_std = V / (self.scale ** 2)
        det = np.linalg.det(V_std)
        inv = np.linalg.inv(V_std)
        dx = self.X - mu_std[0]
        dy = self.Y - mu_std[1]
        mahal = inv[0, 0] * dx**2 + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy**2
        dist = np.exp(-0.5 * mahal) / (2 * np.pi * np.sqrt(max(det, 1e-12)))
        return dist / (dist.sum() + 1e-10)

    def sample(self):
        k = np.random.randint(len(self.pairs))
        p = self.pairs[k]
        dist = self._render(p['mu'], p['V'])
        return dist, {'pair': p['pair'], 'mu': p['mu'].tolist()}

    def get_canonical(self):
        p = self.pairs[self.canonical_pair_idx]
        dist = self._render(p['mu'], p['V'])
        return dist, {'pair': p['pair'], 'mu': p['mu'].tolist()}
