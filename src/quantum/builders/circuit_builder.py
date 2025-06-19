"""
Circuit Builder for Quantum Circuits

This module builds quantum circuit layers within an existing SF program,
ensuring gradient flow is maintained.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class CircuitBuilder:
    """
    Builds quantum circuit components within an existing SF program.
    
    This class adds gates to an existing program rather than creating
    new programs, which is essential for gradient preservation.
    """
    
    def __init__(self, prog: sf.Program):
        """
        Initialize circuit builder.
        
        Args:
            prog: Existing SF program to build into
        """
        self.prog = prog
        logger.info("Circuit builder initialized")
    
    def build_layer(self, q: Any, layer_params: Dict[str, List[Any]], layer_idx: int) -> None:
        """
        Build a complete quantum layer.
        
        Args:
            q: Quantum registers from the program context
            layer_params: Dictionary of symbolic parameters for this layer
            layer_idx: Layer index for logging
        """
        logger.debug(f"Building layer {layer_idx}")
        
        # First interferometer
        self.build_interferometer(
            q,
            layer_params.get('bs1_theta', []),
            layer_params.get('bs1_phi', []),
            layer_params.get('rot1_phi', [])
        )
        
        # Squeezing gates
        self.build_squeezing(q, layer_params.get('squeeze_r', []))
        
        # Second interferometer
        self.build_interferometer(
            q,
            layer_params.get('bs2_theta', []),
            layer_params.get('bs2_phi', []),
            layer_params.get('rot2_phi', [])
        )
        
        # Displacement gates
        self.build_displacement(
            q,
            layer_params.get('disp_r', []),
            layer_params.get('disp_phi', [])
        )
        
        # Kerr nonlinearity
        self.build_kerr(q, layer_params.get('kerr_kappa', []))
    
    def build_interferometer(self, q: Any, theta_params: List[Any], 
                           phi_params: List[Any], rot_params: List[Any]) -> None:
        """
        Build an interferometer with beamsplitters and rotations.
        
        Args:
            q: Quantum registers
            theta_params: Beamsplitter theta parameters
            phi_params: Beamsplitter phi parameters
            rot_params: Rotation parameters
        """
        N = len(q)
        
        if N == 1:
            # Single mode: just rotation
            if rot_params:
                ops.Rgate(rot_params[0]) | q[0]
            return
        
        # Apply beamsplitter array
        param_idx = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    if param_idx < len(theta_params) and param_idx < len(phi_params):
                        ops.BSgate(theta_params[param_idx], phi_params[param_idx]) | (q1, q2)
                        param_idx += 1
        
        # Apply final rotations
        for i in range(min(len(rot_params), len(q))):
            ops.Rgate(rot_params[i]) | q[i]
    
    def build_squeezing(self, q: Any, squeeze_params: List[Any]) -> None:
        """
        Build squeezing gates.
        
        Args:
            q: Quantum registers
            squeeze_params: Squeezing parameters
        """
        for i in range(min(len(squeeze_params), len(q))):
            ops.Sgate(squeeze_params[i]) | q[i]
    
    def build_displacement(self, q: Any, r_params: List[Any], phi_params: List[Any]) -> None:
        """
        Build displacement gates.
        
        Args:
            q: Quantum registers
            r_params: Displacement magnitude parameters
            phi_params: Displacement phase parameters
        """
        for i in range(min(len(r_params), len(q))):
            if i < len(phi_params):
                ops.Dgate(r_params[i], phi_params[i]) | q[i]
    
    def build_kerr(self, q: Any, kerr_params: List[Any]) -> None:
        """
        Build Kerr nonlinearity gates.
        
        Args:
            q: Quantum registers
            kerr_params: Kerr parameters
        """
        for i in range(min(len(kerr_params), len(q))):
            ops.Kgate(kerr_params[i]) | q[i]


class LayerBuilder:
    """
    Builds specific types of quantum layers.
    
    Provides templates for common quantum circuit patterns.
    """
    
    @staticmethod
    def build_entangling_layer(q: Any, params: Dict[str, Any]) -> None:
        """
        Build an entangling layer with two-mode gates.
        
        Args:
            q: Quantum registers
            params: Layer parameters
        """
        N = len(q)
        
        # CZ gates for entanglement
        for i in range(N - 1):
            if 'cz_params' in params and i < len(params['cz_params']):
                ops.CZgate(params['cz_params'][i]) | (q[i], q[i + 1])
        
        # Single-mode rotations
        if 'rotation_params' in params:
            for i in range(min(N, len(params['rotation_params']))):
                ops.Rgate(params['rotation_params'][i]) | q[i]
    
    @staticmethod
    def build_gaussian_layer(q: Any, params: Dict[str, Any]) -> None:
        """
        Build a Gaussian transformation layer.
        
        Args:
            q: Quantum registers
            params: Layer parameters
        """
        # Displacement
        if 'disp_r' in params and 'disp_phi' in params:
            for i in range(min(len(q), len(params['disp_r']))):
                if i < len(params['disp_phi']):
                    ops.Dgate(params['disp_r'][i], params['disp_phi'][i]) | q[i]
        
        # Squeezing
        if 'squeeze_r' in params:
            for i in range(min(len(q), len(params['squeeze_r']))):
                ops.Sgate(params['squeeze_r'][i]) | q[i]
        
        # Rotation
        if 'rotation' in params:
            for i in range(min(len(q), len(params['rotation']))):
                ops.Rgate(params['rotation'][i]) | q[i]
    
    @staticmethod
    def build_nonlinear_layer(q: Any, params: Dict[str, Any]) -> None:
        """
        Build a nonlinear transformation layer.
        
        Args:
            q: Quantum registers
            params: Layer parameters
        """
        # Kerr nonlinearity
        if 'kerr_kappa' in params:
            for i in range(min(len(q), len(params['kerr_kappa']))):
                ops.Kgate(params['kerr_kappa'][i]) | q[i]
        
        # Cross-Kerr interaction
        if 'cross_kerr' in params:
            N = len(q)
            param_idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if param_idx < len(params['cross_kerr']):
                        ops.CKgate(params['cross_kerr'][param_idx]) | (q[i], q[j])
                        param_idx += 1


class InterferometerBuilder:
    """
    Specialized builder for different interferometer architectures.
    """
    
    @staticmethod
    def build_rectangular(q: Any, theta_params: List[Any], phi_params: List[Any]) -> None:
        """
        Build a rectangular (Reck) interferometer.
        
        Args:
            q: Quantum registers
            theta_params: Beamsplitter theta parameters
            phi_params: Beamsplitter phi parameters
        """
        N = len(q)
        param_idx = 0
        
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    if param_idx < len(theta_params) and param_idx < len(phi_params):
                        ops.BSgate(theta_params[param_idx], phi_params[param_idx]) | (q1, q2)
                        param_idx += 1
    
    @staticmethod
    def build_triangular(q: Any, theta_params: List[Any], phi_params: List[Any]) -> None:
        """
        Build a triangular interferometer.
        
        Args:
            q: Quantum registers
            theta_params: Beamsplitter theta parameters
            phi_params: Beamsplitter phi parameters
        """
        N = len(q)
        param_idx = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                if param_idx < len(theta_params) and param_idx < len(phi_params):
                    ops.BSgate(theta_params[param_idx], phi_params[param_idx]) | (q[i], q[j])
                    param_idx += 1
