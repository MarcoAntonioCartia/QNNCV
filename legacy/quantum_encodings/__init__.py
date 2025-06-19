"""
Quantum encoding strategies for continuous variable quantum GANs.

This package provides various encoding schemes to map classical data into quantum
parameters for continuous variable quantum circuits.
"""

from .quantum_encodings import (
    QuantumEncodingStrategy,
    CoherentStateEncoding,
    DirectDisplacementEncoding,
    AngleEncoding,
    SparseParameterEncoding,
    ClassicalNeuralEncoding,
    QuantumEncodingFactory
)

__all__ = [
    'QuantumEncodingStrategy',
    'CoherentStateEncoding',
    'DirectDisplacementEncoding',
    'AngleEncoding',
    'SparseParameterEncoding',
    'ClassicalNeuralEncoding',
    'QuantumEncodingFactory'
]
