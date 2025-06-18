"""
Quantum Circuit ASCII Visualizer

Creates detailed ASCII visualizations of quantum circuits showing:
- Circuit topology with gates
- Individual parameter names for each gate
- Parameter counts and statistics
- Layer structure and mode connections
- Similar to Qiskit's circuit visualization but for Strawberry Fields CV circuits
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any
import textwrap


class QuantumCircuitVisualizer:
    """ASCII visualizer for quantum circuits with individual gate parameters."""
    
    def __init__(self, quantum_circuit):
        """
        Initialize visualizer with a quantum circuit.
        
        Args:
            quantum_circuit: PureQuantumCircuitCorrected instance
        """
        self.circuit = quantum_circuit
        self.n_modes = quantum_circuit.n_modes
        self.layers = quantum_circuit.layers
        self.param_names = quantum_circuit.param_names
        self.gate_params = quantum_circuit.trainable_variables
        
        # Create parameter mapping
        self._create_parameter_mapping()
    
    def _create_parameter_mapping(self):
        """Create mapping from parameter names to gate types and positions."""
        self.param_map = {}
        
        for i, param_name in enumerate(self.param_names):
            # Parse parameter name: L{layer}_{gate_type}_{param_type}_{index}
            parts = param_name.split('_')
            layer = int(parts[0][1:])  # Remove 'L' and convert to int
            gate_type = parts[1]
            param_type = parts[2] if len(parts) > 3 else parts[2]
            index = int(parts[-1]) if parts[-1].isdigit() else 0
            
            self.param_map[param_name] = {
                'layer': layer,
                'gate_type': gate_type,
                'param_type': param_type,
                'index': index,
                'tf_variable': self.gate_params[i],
                'value': float(self.gate_params[i].numpy())
            }
    
    def print_circuit_diagram(self, show_values=False, show_gradients=False):
        """
        Print ASCII circuit diagram with parameter information.
        
        Args:
            show_values (bool): Show current parameter values
            show_gradients (bool): Show gradient information if available
        """
        print("\n" + "="*80)
        print("🔬 QUANTUM CIRCUIT DIAGRAM")
        print("="*80)
        
        # Circuit header
        self._print_header()
        
        # For each layer
        for layer in range(self.layers):
            print(f"\n{'─'*80}")
            print(f"LAYER {layer}")
            print(f"{'─'*80}")
            
            # Interferometer 1
            self._print_interferometer(layer, "BS1", "ROT1", "Interferometer 1", show_values)
            
            # Squeezing
            self._print_squeezing(layer, show_values)
            
            # Interferometer 2  
            self._print_interferometer(layer, "BS2", "ROT2", "Interferometer 2", show_values)
            
            # Displacement
            self._print_displacement(layer, show_values)
            
            # Kerr nonlinearity
            self._print_kerr(layer, show_values)
        
        # Circuit footer
        self._print_footer()
        
        # Parameter summary
        self._print_parameter_summary(show_values, show_gradients)
    
    def _print_header(self):
        """Print circuit header with mode labels."""
        print(f"\nModes: {self.n_modes}, Layers: {self.layers}, Total Parameters: {len(self.gate_params)}")
        print("\nMode Labels:")
        for mode in range(self.n_modes):
            print(f"  q[{mode}]: ─────────────────────────────────────────────")
        print()
    
    def _print_interferometer(self, layer, bs_prefix, rot_prefix, title, show_values):
        """Print interferometer section."""
        print(f"\n{title}:")
        
        # Get beamsplitter parameters
        bs_params = {}
        rot_params = {}
        
        for param_name, param_info in self.param_map.items():
            if param_info['layer'] == layer:
                if param_info['gate_type'] == bs_prefix:
                    if param_info['param_type'] == 'theta':
                        bs_params[f"θ_{param_info['index']}"] = param_info
                    elif param_info['param_type'] == 'phi':
                        bs_params[f"φ_{param_info['index']}"] = param_info
                elif param_info['gate_type'] == rot_prefix:
                    rot_params[f"φ_{param_info['index']}"] = param_info
        
        # Print beamsplitter connections
        if self.n_modes > 1:
            n_bs = self.n_modes * (self.n_modes - 1) // 2
            bs_index = 0
            
            for l in range(self.n_modes):
                for k in range(self.n_modes - 1):
                    if (l + k) % 2 != 1:
                        mode1 = k
                        mode2 = k + 1
                        
                        theta_key = f"θ_{bs_index}"
                        phi_key = f"φ_{bs_index}"
                        
                        if theta_key in bs_params and phi_key in bs_params:
                            theta_param = bs_params[theta_key]
                            phi_param = bs_params[phi_key]
                            
                            # Create beamsplitter visualization
                            print(f"  q[{mode1}]: ────┤BS({theta_param['tf_variable'].name}")
                            if show_values:
                                print(f"           θ={theta_param['value']:.3f}")
                            print(f"  q[{mode2}]: ────┤  ({phi_param['tf_variable'].name}")
                            if show_values:
                                print(f"           φ={phi_param['value']:.3f}")
                            print("              ├────")
                            
                        bs_index += 1
        
        # Print rotation parameters
        if rot_params:
            print(f"  Rotations:")
            for i, (rot_key, rot_param) in enumerate(rot_params.items()):
                print(f"    q[{i}]: ──R({rot_param['tf_variable'].name})", end="")
                if show_values:
                    print(f", φ={rot_param['value']:.3f})", end="")
                print("──")
    
    def _print_squeezing(self, layer, show_values):
        """Print squeezing gates."""
        print(f"\nSqueezing Gates:")
        
        for mode in range(self.n_modes):
            param_name = f"L{layer}_SQUEEZE_r_{mode}"
            if param_name in self.param_map:
                param_info = self.param_map[param_name]
                print(f"  q[{mode}]: ──S({param_info['tf_variable'].name}", end="")
                if show_values:
                    print(f", r={param_info['value']:.3f})", end="")
                print(")──")
            else:
                print(f"  q[{mode}]: ──S(?)──")
    
    def _print_displacement(self, layer, show_values):
        """Print displacement gates."""
        print(f"\nDisplacement Gates:")
        
        for mode in range(self.n_modes):
            r_param_name = f"L{layer}_DISP_r_{mode}"
            phi_param_name = f"L{layer}_DISP_phi_{mode}"
            
            if r_param_name in self.param_map and phi_param_name in self.param_map:
                r_param = self.param_map[r_param_name]
                phi_param = self.param_map[phi_param_name]
                
                print(f"  q[{mode}]: ──D({r_param['tf_variable'].name},")
                print(f"           {phi_param['tf_variable'].name}", end="")
                if show_values:
                    print(f", r={r_param['value']:.3f}, φ={phi_param['value']:.3f})", end="")
                print(")──")
            else:
                print(f"  q[{mode}]: ──D(?,?)──")
    
    def _print_kerr(self, layer, show_values):
        """Print Kerr nonlinearity gates."""
        print(f"\nKerr Nonlinearity:")
        
        for mode in range(self.n_modes):
            param_name = f"L{layer}_KERR_kappa_{mode}"
            if param_name in self.param_map:
                param_info = self.param_map[param_name]
                print(f"  q[{mode}]: ──K({param_info['tf_variable'].name}", end="")
                if show_values:
                    print(f", κ={param_info['value']:.3f})", end="")
                print(")──")
            else:
                print(f"  q[{mode}]: ──K(?)──")
    
    def _print_footer(self):
        """Print circuit footer."""
        print(f"\nOutput Measurements:")
        for mode in range(self.n_modes):
            print(f"  q[{mode}]: ────[X,P,N]───► measurements[{mode*3}:{mode*3+3}]")
    
    def _print_parameter_summary(self, show_values, show_gradients):
        """Print detailed parameter summary."""
        print(f"\n{'='*80}")
        print("📊 PARAMETER SUMMARY")
        print(f"{'='*80}")
        
        # Group parameters by type
        param_groups = {}
        for param_name, param_info in self.param_map.items():
            gate_type = param_info['gate_type']
            if gate_type not in param_groups:
                param_groups[gate_type] = []
            param_groups[gate_type].append(param_info)
        
        # Print summary by gate type
        total_params = 0
        for gate_type, params in param_groups.items():
            print(f"\n{gate_type.upper()} Parameters ({len(params)} total):")
            
            for param in params:
                layer = param['layer']
                param_type = param['param_type']
                index = param['index']
                var_name = param['tf_variable'].name
                
                print(f"  Layer {layer}, {param_type}[{index}]: {var_name}", end="")
                
                if show_values:
                    print(f" = {param['value']:.4f}", end="")
                
                if show_gradients:
                    # This would require gradient information to be passed in
                    print(f" (grad: available)", end="")
                
                print()
            
            total_params += len(params)
        
        print(f"\n📈 TOTAL PARAMETERS: {total_params}")
        print(f"🔬 TOTAL TF VARIABLES: {len(self.gate_params)}")
        
        # Parameter distribution
        print(f"\n📊 Parameter Distribution:")
        for gate_type, params in param_groups.items():
            percentage = len(params) / total_params * 100
            print(f"  {gate_type.upper()}: {len(params)} ({percentage:.1f}%)")
    
    def print_compact_circuit(self):
        """Print compact circuit representation."""
        print("\n" + "="*60)
        print("🔬 COMPACT CIRCUIT DIAGRAM")
        print("="*60)
        
        for mode in range(self.n_modes):
            print(f"\nq[{mode}]: |0⟩", end="")
            
            for layer in range(self.layers):
                print("─INT1─S─INT2─D─K", end="")
                if layer < self.layers - 1:
                    print("─", end="")
            
            print("─[X,P,N]")
        
        print(f"\nLegend:")
        print(f"  INT1/INT2: Interferometers (beamsplitters + rotations)")
        print(f"  S: Squeezing gates")
        print(f"  D: Displacement gates") 
        print(f"  K: Kerr nonlinearity")
        print(f"  [X,P,N]: Measurements (X quadrature, P quadrature, photon Number)")
    
    def print_parameter_list(self, show_values=True):
        """Print complete list of all parameters."""
        print("\n" + "="*80)
        print("📋 COMPLETE PARAMETER LIST")
        print("="*80)
        
        print(f"Total Parameters: {len(self.gate_params)}")
        print(f"Circuit: {self.n_modes} modes, {self.layers} layers\n")
        
        for i, (param_name, tf_var) in enumerate(zip(self.param_names, self.gate_params)):
            value_str = f" = {float(tf_var.numpy()):.4f}" if show_values else ""
            print(f"{i:3d}. {param_name:<25} ({tf_var.name}){value_str}")
    
    def export_parameter_info(self):
        """Export parameter information as dictionary."""
        param_info = {
            'circuit_info': {
                'n_modes': self.n_modes,
                'layers': self.layers,
                'total_parameters': len(self.gate_params)
            },
            'parameters': []
        }
        
        for i, (param_name, tf_var) in enumerate(zip(self.param_names, self.gate_params)):
            param_info['parameters'].append({
                'index': i,
                'name': param_name,
                'tf_name': tf_var.name,
                'value': float(tf_var.numpy()),
                'shape': tf_var.shape.as_list(),
                'dtype': str(tf_var.dtype)
            })
        
        return param_info


def visualize_circuit(quantum_circuit, style='full', show_values=False, show_gradients=False):
    """
    Convenience function to visualize a quantum circuit.
    
    Args:
        quantum_circuit: PureQuantumCircuitCorrected instance
        style (str): 'full', 'compact', or 'list'
        show_values (bool): Show parameter values
        show_gradients (bool): Show gradient information
    """
    visualizer = QuantumCircuitVisualizer(quantum_circuit)
    
    if style == 'full':
        visualizer.print_circuit_diagram(show_values, show_gradients)
    elif style == 'compact':
        visualizer.print_compact_circuit()
    elif style == 'list':
        visualizer.print_parameter_list(show_values)
    else:
        print(f"Unknown style: {style}. Use 'full', 'compact', or 'list'")


def demo_circuit_visualization():
    """Demonstrate circuit visualization with a sample circuit."""
    print("🚀 QUANTUM CIRCUIT VISUALIZATION DEMO")
    print("="*50)
    
    # Import the corrected circuit class
    try:
        from pure_quantum_static_matrices import PureQuantumCircuitCorrected
        
        # Create sample circuit
        circuit = PureQuantumCircuitCorrected(n_modes=3, layers=2, cutoff_dim=6)
        
        print(f"Created circuit: {circuit.n_modes} modes, {circuit.layers} layers")
        print(f"Total parameters: {circuit.get_parameter_count()}")
        
        # Show different visualization styles
        print("\n" + "🔍 FULL CIRCUIT DIAGRAM:")
        visualize_circuit(circuit, style='full', show_values=True)
        
        print("\n" + "📏 COMPACT DIAGRAM:")
        visualize_circuit(circuit, style='compact')
        
        print("\n" + "📋 PARAMETER LIST:")
        visualize_circuit(circuit, style='list', show_values=True)
        
        # Export parameter information
        visualizer = QuantumCircuitVisualizer(circuit)
        param_info = visualizer.export_parameter_info()
        print(f"\n📤 Parameter info exported: {len(param_info['parameters'])} parameters")
        
        return circuit, visualizer
        
    except ImportError as e:
        print(f"❌ Could not import PureQuantumCircuitCorrected: {e}")
        print("Make sure pure_quantum_static_matrices.py is in the same directory")
        return None, None


if __name__ == "__main__":
    # Run demonstration
    circuit, visualizer = demo_circuit_visualization()
    
    if circuit is not None:
        print("\n🎉 Circuit visualization demo completed!")
        print("\nUsage examples:")
        print("  visualize_circuit(circuit, 'full', show_values=True)")
        print("  visualize_circuit(circuit, 'compact')")
        print("  visualize_circuit(circuit, 'list', show_values=True)")
    else:
        print("\n❌ Demo failed. Check your circuit implementation.")
