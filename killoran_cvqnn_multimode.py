"""
Killoran CV-QNN Multi-Mode Generator
====================================

Multi-mode extension of the Killoran et al. (2018) CV-QNN architecture.

Architecture per layer (from the paper):
    Layer L = U1(theta1,phi1) -> S(r) -> U2(theta2,phi2) -> D(alpha) -> Phi(lambda)

Where:
    - U1, U2: Interferometers (beamsplitters + rotations) that mix modes
    - S: Squeeze gates on each mode
    - D: Displacement gates on each mode  
    - Phi: Kerr (non-Gaussian) gates on each mode

For N=1 mode, interferometer = single rotation (reduces to our existing implementation)
For N>=2 modes, interferometer = rectangular BS array + final rotations

Reference:
    Killoran et al., "Continuous-variable quantum neural networks" (2018)
    https://arxiv.org/abs/1806.06871
"""

# Warning suppression (if available)
try:
    from src.utils.warning_suppression import enable_clean_training
    enable_clean_training()
except ImportError:
    pass

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops


def interferometer(params, q):
    """
    Parameterised interferometer acting on N modes.
    
    Implements a rectangular decomposition of an NxN unitary matrix.
    
    Args:
        params: list of length N(N-1) + N-1 parameters
            - First N(N-1)/2: beamsplitter angles (theta)
            - Next N(N-1)/2: beamsplitter phases (phi)
            - Final N-1: local rotations on first N-1 modes
        q: list of Strawberry Fields quantum registers
    
    For N=1: Single rotation gate
    For N=2: 1 BS + 1 rotation = 3 params
    For N=3: 3 BS + 2 rotations = 8 params
    """
    N = len(q)
    
    if N == 1:
        # Single mode: interferometer is just a rotation
        ops.Rgate(params[0]) | q[0]
        return
    
    # Multi-mode: rectangular beamsplitter array
    n_bs = N * (N - 1) // 2
    theta = params[:n_bs]           # BS angles
    phi = params[n_bs:2*n_bs]       # BS phases
    rphi = params[2*n_bs:]          # Final rotations (N-1 of them)
    
    n = 0  # Parameter counter
    
    # Apply rectangular beamsplitter array (depth N)
    for layer in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # Skip even or odd pairs depending on layer
            if (layer + k) % 2 != 1:
                ops.BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1
    
    # Apply final local phase shifts to first N-1 modes
    for i in range(N - 1):
        ops.Rgate(rphi[i]) | q[i]


def interferometer_param_count(n_modes):
    """Calculate number of parameters for an interferometer with n_modes."""
    if n_modes == 1:
        return 1  # Just a rotation
    # N(N-1)/2 BS angles + N(N-1)/2 BS phases + (N-1) rotations
    return n_modes * (n_modes - 1) + n_modes - 1


class KilloranCVQNNMultiMode:
    """
    Multi-mode Killoran CV-QNN for N-dimensional probability distributions.
    
    Layer structure (following Killoran et al.):
        U1 -> S(all modes) -> U2 -> D(all modes) -> Phi(all modes)
    
    Where U1, U2 are interferometers (mix modes via beamsplitters).
    
    Parameters per layer:
        - 2 interferometers: 2 x (N^2 - 1) for N>=2, or 2 for N=1
        - N squeeze gates: 2N (magnitude r and phase phi each)
        - N displacement gates: 2N (magnitude r and phase phi each)
        - N Kerr gates: N
    
    Total per layer:
        - N=1: 1 + 2 + 1 + 2 + 1 = 7 params (matches single-mode!)
        - N=2: 3 + 4 + 3 + 4 + 2 = 16 params
        - N=3: 8 + 6 + 8 + 6 + 3 = 31 params
    
    Args:
        n_modes: Number of qumodes (= output dimension)
        n_layers: Number of circuit layers
        cutoff_dim: Fock space cutoff dimension
        use_kerr: Whether to include Kerr gates (essential for non-Gaussian)
    """
    
    def __init__(
        self,
        n_modes: int = 2,
        n_layers: int = 6,
        cutoff_dim: int = 10,
        use_kerr: bool = True
    ):
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.use_kerr = use_kerr
        
        # Calculate parameters per layer
        self.interferometer_params = interferometer_param_count(n_modes)
        
        # Per layer:
        # - 2 interferometers
        # - N squeeze gates (2 params each: r, phi)
        # - N displacement gates (2 params each: r, phi)  
        # - N Kerr gates (1 param each) if use_kerr
        self.params_per_layer = (
            2 * self.interferometer_params +  # U1 and U2
            2 * n_modes +                      # Squeeze (r, phi) x N
            2 * n_modes +                      # Displacement (r, phi) x N
            (n_modes if use_kerr else 0)       # Kerr x N
        )
        
        self.total_params = self.params_per_layer * n_layers
        
        # Initialize weights
        self.weights = tf.Variable(
            tf.random.uniform([self.total_params], -0.1, 0.1),
            trainable=True,
            dtype=tf.float32,
            name='quantum_weights'
        )
        
        # Create SF program and engine
        self.prog = sf.Program(n_modes)
        self.eng = sf.Engine('tf', backend_options={'cutoff_dim': cutoff_dim})
        
        print(f"KilloranCVQNNMultiMode initialized:")
        print(f"  Modes: {n_modes}")
        print(f"  Layers: {n_layers}")
        print(f"  Cutoff: {cutoff_dim}")
        print(f"  Params per interferometer: {self.interferometer_params}")
        print(f"  Params per layer: {self.params_per_layer}")
        print(f"  Total params: {self.total_params}")
    
    @property
    def trainable_variables(self):
        return [self.weights]
    
    def get_config(self):
        return {
            'n_modes': self.n_modes,
            'n_layers': self.n_layers,
            'cutoff_dim': self.cutoff_dim,
            'use_kerr': self.use_kerr,
            'total_params': self.total_params,
            'params_per_layer': self.params_per_layer
        }
    
    def _build_layer(self, q, layer_weights):
        """
        Build one layer of the circuit.
        
        Structure: U1 -> S(all) -> U2 -> D(all) -> Phi(all)
        """
        idx = 0
        N = self.n_modes
        int_params = self.interferometer_params
        
        # U1: First interferometer
        interferometer(layer_weights[idx:idx + int_params], q)
        idx += int_params
        
        # S: Squeeze gates on all modes
        for mode in range(N):
            r = layer_weights[idx]
            phi = layer_weights[idx + 1]
            ops.Sgate(r, phi) | q[mode]
            idx += 2
        
        # U2: Second interferometer
        interferometer(layer_weights[idx:idx + int_params], q)
        idx += int_params
        
        # D: Displacement gates on all modes
        for mode in range(N):
            r = layer_weights[idx]
            phi = layer_weights[idx + 1]
            ops.Dgate(r, phi) | q[mode]
            idx += 2
        
        # Phi: Kerr gates on all modes (if enabled)
        if self.use_kerr:
            for mode in range(N):
                kappa = layer_weights[idx]
                ops.Kgate(kappa) | q[mode]
                idx += 1
        
        return idx
    
    def _run_circuit(self, weights):
        """Run the quantum circuit and return the output state."""
        self.prog = sf.Program(self.n_modes)
        
        with self.prog.context as q:
            # Build all layers
            for layer in range(self.n_layers):
                start_idx = layer * self.params_per_layer
                end_idx = start_idx + self.params_per_layer
                layer_weights = weights[start_idx:end_idx]
                self._build_layer(q, layer_weights)
        
        # Run and get state
        result = self.eng.run(self.prog)
        return result.state
    
    def get_fock_state(self):
        """
        Get the output Fock state amplitudes.
        
        Returns:
            For N=1: (cutoff,) complex amplitudes
            For N=2: (cutoff, cutoff) complex amplitudes
            For N=3: (cutoff, cutoff, cutoff) complex amplitudes
        """
        state = self._run_circuit(self.weights)
        return state.ket()
    
    def generate_distribution_1d(self, xvec, mode=0):
        """
        Generate 1D marginal distribution for a single mode.
        
        Args:
            xvec: (n_points,) array of x values
            mode: Which mode to measure (0 to n_modes-1)
        
        Returns:
            prob: (n_points,) marginal probability distribution
        """
        from src.utils.killoran_init import hermite_prob_distribution
        
        # Get full state
        state = self._run_circuit(self.weights)
        ket = state.ket()  # Shape: (cutoff,) * n_modes
        
        if self.n_modes == 1:
            # Single mode: use existing function
            return hermite_prob_distribution(ket, xvec)
        
        # Multi-mode: trace out other modes to get marginal
        # For simplicity, compute |psi|^2 and marginalize
        # This is an approximation - full treatment needs reduced density matrix
        
        # TODO: Implement proper reduced density matrix for marginals
        raise NotImplementedError(
            "1D marginals for multi-mode not yet implemented. "
            "Use generate_distribution_2d for 2-mode systems."
        )
    
    def generate_distribution_2d(self, xvec, yvec):
        """
        Generate 2D joint probability distribution P(x, y).
        
        Uses Hermite polynomial expansion:
            psi(x,y) = Sumnm cnm phin(x) phim(y)
            P(x,y) = |psi(x,y)|^2
        
        Args:
            xvec: (nx,) array of x values for mode 0
            yvec: (ny,) array of y values for mode 1
        
        Returns:
            prob: (nx, ny) joint probability distribution
        """
        if self.n_modes != 2:
            raise ValueError(f"generate_distribution_2d requires n_modes=2, got {self.n_modes}")
        
        # Get Fock state amplitudes: shape (cutoff, cutoff)
        state = self._run_circuit(self.weights)
        ket = state.ket()  # (cutoff, cutoff) for 2 modes
        
        # Compute Hermite basis functions
        hermite_x = self._hermite_basis(xvec)  # (nx, cutoff)
        hermite_y = self._hermite_basis(yvec)  # (ny, cutoff)
        
        # Wavefunction: psi(x,y) = Sumnm cnm phin(x) phim(y)
        # Using einsum: ket[n,m] * hermite_x[x,n] * hermite_y[y,m] -> psi[x,y]
        psi = tf.einsum('nm,xn,ym->xy', tf.cast(ket, tf.complex64), 
                       tf.cast(hermite_x, tf.complex64),
                       tf.cast(hermite_y, tf.complex64))
        
        # Probability
        prob = tf.abs(psi) ** 2
        
        # Normalize
        prob = prob / tf.reduce_sum(prob)
        
        return prob
    
    def _hermite_basis(self, xvec):
        """
        Compute Hermite basis functions phin(x) for n = 0, 1, ..., cutoff-1.
        
        phin(x) = (1/sqrt(2^n n! sqrtpi)) Hn(x) exp(-x^2/2)
        
        Args:
            xvec: (n_points,) array of x values
        
        Returns:
            basis: (n_points, cutoff) array of basis function values
        """
        from scipy.special import hermite
        from math import factorial, sqrt, pi
        
        xvec = np.asarray(xvec)
        n_points = len(xvec)
        basis = np.zeros((n_points, self.cutoff_dim), dtype=np.float64)
        
        for n in range(self.cutoff_dim):
            # Hermite polynomial coefficients
            Hn = hermite(n)
            # Normalization
            norm = 1.0 / sqrt(2**n * factorial(n) * sqrt(pi))
            # Basis function
            basis[:, n] = norm * Hn(xvec) * np.exp(-xvec**2 / 2)
        
        return tf.constant(basis, dtype=tf.float32)


# =============================================================================
# Target Distribution Functions for 2D
# =============================================================================

def gaussian_2d_independent(X, Y, mu=(0, 0), sigma=0.5):
    """
    Independent 2D Gaussian: P(x,y) = P(x) x P(y)
    
    This should be learnable WITHOUT beamsplitters.
    """
    prob = np.exp(-((X - mu[0])**2 + (Y - mu[1])**2) / (2 * sigma**2))
    return prob / prob.sum()


def gaussian_2d_correlated(X, Y, mu=(0, 0), cov=None):
    """
    Correlated bivariate Gaussian.
    
    This REQUIRES beamsplitters to learn (modes must be entangled).
    
    Args:
        cov: 2x2 covariance matrix, e.g., [[1, 0.7], [0.7, 1]] for rho=0.7
    """
    from scipy.stats import multivariate_normal
    
    if cov is None:
        cov = [[1, 0.7], [0.7, 1]]  # Correlation rho = 0.7
    
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, cov)
    prob = rv.pdf(pos)
    return prob / prob.sum()


def ring_2d(X, Y, radius=1.5, width=0.3):
    """
    Ring distribution centered at origin.
    
    Tests ability to learn non-convex shapes.
    """
    dist = np.sqrt(X**2 + Y**2)
    prob = np.exp(-(dist - radius)**2 / (2 * width**2))
    return prob / prob.sum()


def four_gaussians_2d(X, Y, spread=1.5, sigma=0.3):
    """
    Four Gaussians at corners of a square.
    
    Tests multi-modal 2D learning.
    """
    centers = [(-spread, -spread), (-spread, spread), 
               (spread, -spread), (spread, spread)]
    
    prob = np.zeros_like(X)
    for cx, cy in centers:
        prob += np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    
    return prob / prob.sum()


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing KilloranCVQNNMultiMode")
    print("=" * 60)
    
    # Test 1: Single mode (should match original)
    print("\n--- Test 1: Single mode (N=1) ---")
    gen1 = KilloranCVQNNMultiMode(n_modes=1, n_layers=4, cutoff_dim=8)
    assert gen1.params_per_layer == 7, f"Expected 7, got {gen1.params_per_layer}"
    print("OK: Single mode params correct (7 per layer)")
    
    # Test 2: Two modes
    print("\n--- Test 2: Two modes (N=2) ---")
    gen2 = KilloranCVQNNMultiMode(n_modes=2, n_layers=4, cutoff_dim=8)
    # 2 interferometers x 3 params + 4 squeeze + 4 displacement + 2 kerr = 16
    expected = 2 * 3 + 4 + 4 + 2
    assert gen2.params_per_layer == expected, f"Expected {expected}, got {gen2.params_per_layer}"
    print(f"OK: Two mode params correct ({expected} per layer)")
    
    # Test 3: Three modes
    print("\n--- Test 3: Three modes (N=3) ---")
    gen3 = KilloranCVQNNMultiMode(n_modes=3, n_layers=4, cutoff_dim=6)
    # 2 interferometers x 8 params + 6 squeeze + 6 displacement + 3 kerr = 31
    expected = 2 * 8 + 6 + 6 + 3
    assert gen3.params_per_layer == expected, f"Expected {expected}, got {gen3.params_per_layer}"
    print(f"OK: Three mode params correct ({expected} per layer)")
    
    # Test 4: Run 2-mode circuit
    print("\n--- Test 4: Run 2-mode circuit ---")
    try:
        gen2_small = KilloranCVQNNMultiMode(n_modes=2, n_layers=2, cutoff_dim=6)
        ket = gen2_small.get_fock_state()
        print(f"OK: 2-mode Fock state shape: {ket.shape}")
        print(f"  Total probability: {tf.reduce_sum(tf.abs(ket)**2).numpy():.6f}")
    except Exception as e:
        print(f"FAIL: Failed: {e}")
    
    # Test 5: Generate 2D distribution
    print("\n--- Test 5: Generate 2D distribution ---")
    try:
        xvec = np.linspace(-3, 3, 50)
        yvec = np.linspace(-3, 3, 50)
        
        prob_2d = gen2_small.generate_distribution_2d(xvec, yvec)
        print(f"OK: 2D distribution shape: {prob_2d.shape}")
        print(f"  Sum (should be ~1): {tf.reduce_sum(prob_2d).numpy():.6f}")
    except Exception as e:
        print(f"FAIL: Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Gradient flow
    print("\n--- Test 6: Gradient flow ---")
    try:
        gen_test = KilloranCVQNNMultiMode(n_modes=2, n_layers=2, cutoff_dim=6)
        xvec = np.linspace(-3, 3, 30)
        yvec = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(xvec, yvec)
        target = gaussian_2d_independent(X, Y)
        target_tf = tf.constant(target, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            gen_prob = gen_test.generate_distribution_2d(xvec, yvec)
            loss = tf.reduce_mean(tf.square(gen_prob - target_tf))
        
        grads = tape.gradient(loss, gen_test.trainable_variables)
        
        if grads[0] is not None:
            grad_norm = tf.norm(grads[0]).numpy()
            print(f"OK: Gradients computed!")
            print(f"  Loss: {loss.numpy():.6f}")
            print(f"  Gradient norm: {grad_norm:.6f}")
        else:
            print("FAIL: Gradients are None!")
    except Exception as e:
        print(f"FAIL: Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)
