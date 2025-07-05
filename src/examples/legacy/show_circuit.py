"""
Simple script to visualize a quantum circuit.
"""

import sys
sys.path.append('.')

from quantum.core.quantum_circuit import PureQuantumCircuit
from utils.quantum_circuit_visualizer import visualize_circuit

# Create a circuit
circuit = PureQuantumCircuit(n_modes=3, layers=2, cutoff_dim=4)

# Print basic info
print(f"\nCircuit created with {len(circuit.trainable_variables)} parameters")
print(f"Modes: {circuit.n_modes}, Layers: {circuit.layers}")

# Show the circuit visualization
print("\n=== COMPACT CIRCUIT VISUALIZATION ===")
visualize_circuit(circuit, style='compact')

print("\n=== PARAMETER LIST ===")
visualize_circuit(circuit, style='list', show_values=True)

print("\n=== FULL CIRCUIT DIAGRAM ===")
visualize_circuit(circuit, style='full', show_values=True)
