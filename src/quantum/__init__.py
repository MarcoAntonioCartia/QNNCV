"""
Quantum Module for Pure Quantum Neural Networks

This module provides gradient-safe quantum circuit implementations
with modular components for building quantum neural networks.
"""

from .core.quantum_circuit import QuantumCircuitBase, PureQuantumCircuit

__all__ = ['QuantumCircuitBase', 'PureQuantumCircuit']
