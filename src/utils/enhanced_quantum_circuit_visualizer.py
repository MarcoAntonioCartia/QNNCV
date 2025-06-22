"""
Enhanced Visual Quantum Circuit Visualizer

Creates publication-quality visual circuit diagrams with:
- Colored gate boxes with parameter values
- Accurate operation detection from SF programs
- Coordinate generator specific features
- High-quality matplotlib output
- Interactive parameter display
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Any, Tuple
import os

class EnhancedQuantumCircuitVisualizer:
    """Enhanced visual quantum circuit visualizer with publication-quality output."""
    
    def __init__(self, quantum_circuit, coordinate_generator=None):
        """
        Initialize enhanced visualizer.
        
        Args:
            quantum_circuit: PureSFQuantumCircuit instance
            coordinate_generator: Optional coordinate generator for mode assignment info
        """
        self.circuit = quantum_circuit
        self.coordinate_generator = coordinate_generator
        self.n_modes = quantum_circuit.n_modes
        self.n_layers = getattr(quantum_circuit, 'layers', getattr(quantum_circuit, 'n_layers', 2))
        
        # Gate color scheme (professional quantum circuit colors)
        self.gate_colors = {
            'squeeze': '#9B59B6',      # Purple
            'beamsplitter': '#3498DB', # Blue  
            'rotation': '#E74C3C',     # Red
            'displacement': '#2ECC71', # Green
            'measurement_x': '#F39C12', # Orange (X quadrature only)
            'constellation': '#1ABC9C', # Teal
            'kerr': '#34495E'          # Dark gray
        }
        
        # Extract circuit operations
        self.operations = self._extract_operations()
        
    def _extract_operations(self) -> List[Dict[str, Any]]:
        """Extract actual operations from the SF circuit."""
        operations = []
        
        # Parse parameter names to understand circuit structure
        for param_name in self.circuit.sf_param_names:
            parts = param_name.split('_')
            
            if 'squeeze_r' in param_name:
                layer = int(parts[2])
                mode = int(parts[3])
                value = float(self.circuit.tf_parameters[param_name].numpy())
                operations.append({
                    'type': 'squeeze',
                    'layer': layer,
                    'mode': mode,
                    'param_name': param_name,
                    'value': value,
                    'label': f'S\nr={value:.3f}'
                })
                
            elif 'bs_theta' in param_name:
                layer = int(parts[2])
                mode = int(parts[3])
                value = float(self.circuit.tf_parameters[param_name].numpy())
                operations.append({
                    'type': 'beamsplitter',
                    'layer': layer,
                    'modes': [mode, mode + 1],
                    'param_name': param_name,
                    'value': value,
                    'label': f'BS\nθ={value:.3f}'
                })
                
            elif 'rotation' in param_name:
                layer = int(parts[1])
                mode = int(parts[2])
                value = float(self.circuit.tf_parameters[param_name].numpy())
                operations.append({
                    'type': 'rotation',
                    'layer': layer,
                    'mode': mode,
                    'param_name': param_name,
                    'value': value,
                    'label': f'R\nφ={value:.3f}'
                })
                
            elif 'displacement' in param_name:
                layer = int(parts[1])
                mode = int(parts[2])
                value = float(self.circuit.tf_parameters[param_name].numpy())
                operations.append({
                    'type': 'displacement',
                    'layer': layer,
                    'mode': mode,
                    'param_name': param_name,
                    'value': value,
                    'label': f'D\nα={value:.3f}'
                })
        
        return operations
    
    def create_visual_circuit(self, save_path: Optional[str] = None, 
                            show_values: bool = True,
                            show_coordinate_info: bool = True,
                            figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
        """
        Create publication-quality visual circuit diagram.
        
        Args:
            save_path: Path to save the diagram
            show_values: Show parameter values in gates
            show_coordinate_info: Show coordinate generator mode assignments
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout parameters
        mode_spacing = 1.0
        layer_spacing = 2.0
        gate_width = 0.8
        gate_height = 0.6
        
        # Draw mode lines
        for mode in range(self.n_modes):
            y = mode * mode_spacing
            x_start = -0.5
            x_end = (self.n_layers + 1) * layer_spacing
            ax.plot([x_start, x_end], [y, y], 'k-', linewidth=2, alpha=0.7)
            
            # Mode labels with coordinate info
            mode_label = f'q[{mode}]'
            if show_coordinate_info and self.coordinate_generator:
                coord_info = self._get_coordinate_info(mode)
                if coord_info:
                    mode_label += f'\n{coord_info}'
            
            ax.text(x_start - 0.3, y, mode_label, 
                   ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Draw gates by layer
        for layer in range(self.n_layers):
            x_center = layer * layer_spacing
            
            # Group operations by layer
            layer_ops = [op for op in self.operations if op['layer'] == layer]
            
            # Draw operations in order: squeeze, beamsplitter, rotation, displacement
            for op_type in ['squeeze', 'beamsplitter', 'rotation', 'displacement']:
                ops_of_type = [op for op in layer_ops if op['type'] == op_type]
                
                for op in ops_of_type:
                    if op['type'] == 'beamsplitter':
                        self._draw_beamsplitter(ax, op, x_center, mode_spacing, 
                                              gate_width, gate_height, show_values)
                    else:
                        self._draw_single_gate(ax, op, x_center, mode_spacing,
                                             gate_width, gate_height, show_values)
        
        # Draw measurements (X quadrature only for coordinate generator)
        measurement_x = (self.n_layers) * layer_spacing
        for mode in range(self.n_modes):
            y = mode * mode_spacing
            
            # Only X quadrature measurement for coordinate generator
            if self.coordinate_generator:
                measurement_label = 'X'
                measurement_color = self.gate_colors['measurement_x']
            else:
                measurement_label = 'X,P,N'
                measurement_color = self.gate_colors['measurement_x']
            
            rect = FancyBboxPatch(
                (measurement_x - gate_width/2, y - gate_height/2),
                gate_width, gate_height,
                boxstyle="round,pad=0.05",
                facecolor=measurement_color,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            ax.text(measurement_x, y, measurement_label,
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Add title and labels
        circuit_title = f"Quantum Circuit: {self.n_modes} modes, {self.n_layers} layers"
        if self.coordinate_generator:
            circuit_title += " (Coordinate Generator)"
        ax.set_title(circuit_title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        self._add_legend(ax, show_coordinate_info)
        
        # Set axis properties
        ax.set_xlim(-1, (self.n_layers + 1) * layer_spacing + 0.5)
        ax.set_ylim(-0.5, (self.n_modes - 1) * mode_spacing + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add parameter summary
        if show_values:
            self._add_parameter_summary(fig, ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Circuit diagram saved to: {save_path}")
        
        return fig
    
    def _draw_single_gate(self, ax, operation, x_center, mode_spacing, 
                         gate_width, gate_height, show_values):
        """Draw a single-mode gate."""
        mode = operation['mode']
        y = mode * mode_spacing
        
        # Adjust x position based on gate type for better layout
        x_offset = {
            'squeeze': -0.6,
            'rotation': 0.0,
            'displacement': 0.6
        }.get(operation['type'], 0.0)
        
        x = x_center + x_offset
        
        # Create gate box
        color = self.gate_colors[operation['type']]
        rect = FancyBboxPatch(
            (x - gate_width/2, y - gate_height/2),
            gate_width, gate_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label
        if show_values:
            label = operation['label']
        else:
            label = operation['type'][0].upper()
        
        ax.text(x, y, label, ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    
    def _draw_beamsplitter(self, ax, operation, x_center, mode_spacing,
                          gate_width, gate_height, show_values):
        """Draw a two-mode beamsplitter gate."""
        modes = operation['modes']
        y1 = modes[0] * mode_spacing
        y2 = modes[1] * mode_spacing
        y_center = (y1 + y2) / 2
        
        x = x_center - 0.3  # Slightly left for beamsplitters
        
        # Create beamsplitter box spanning both modes
        color = self.gate_colors['beamsplitter']
        rect = FancyBboxPatch(
            (x - gate_width/2, y1 - gate_height/4),
            gate_width, y2 - y1 + gate_height/2,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label
        if show_values:
            label = operation['label']
        else:
            label = 'BS'
        
        ax.text(x, y_center, label, ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')
    
    def _get_coordinate_info(self, mode: int) -> Optional[str]:
        """Get coordinate assignment info for a mode."""
        if not self.coordinate_generator or not hasattr(self.coordinate_generator, 'mode_coordinate_mapping'):
            return None
        
        mapping = self.coordinate_generator.mode_coordinate_mapping
        if mapping and mode in mapping:
            coord_info = mapping[mode]
            cluster_id = coord_info['cluster_id']
            coordinate = coord_info['coordinate']
            active = "✅" if coord_info['active'] else "❌"
            return f"{active} C{cluster_id}:{coordinate}"
        
        return None
    
    def _add_legend(self, ax, show_coordinate_info):
        """Add legend explaining gate colors and symbols."""
        legend_elements = []
        
        # Gate type legend
        for gate_type, color in self.gate_colors.items():
            if gate_type == 'measurement_x':
                if self.coordinate_generator:
                    label = 'X Measurement (Coordinate Gen)'
                else:
                    label = 'Measurements'
            else:
                label = gate_type.title()
            
            legend_elements.append(
                patches.Patch(color=color, label=label)
            )
        
        # Add coordinate info legend if applicable
        if show_coordinate_info and self.coordinate_generator:
            legend_elements.append(
                patches.Patch(color='none', label='Mode Labels: ✅/❌ C{cluster}:{coord}')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1), fontsize=9)
    
    def _add_parameter_summary(self, fig, ax):
        """Add parameter summary box."""
        # Count parameters by type
        param_counts = {}
        total_params = 0
        
        for op in self.operations:
            op_type = op['type']
            param_counts[op_type] = param_counts.get(op_type, 0) + 1
            total_params += 1
        
        # Create summary text
        summary_text = f"Parameters: {total_params} total\n"
        for op_type, count in param_counts.items():
            summary_text += f"  {op_type}: {count}\n"
        
        # Add coordinate generator info
        if self.coordinate_generator:
            summary_text += f"\nCoordinate Generator:\n"
            summary_text += f"  Modes: {self.n_modes}\n"
            summary_text += f"  Coordinates: {getattr(self.coordinate_generator, 'coordinate_names', ['X', 'Y'])}\n"
            if hasattr(self.coordinate_generator, 'cluster_analyzer') and self.coordinate_generator.cluster_analyzer:
                summary_text += f"  Active clusters: {len(self.coordinate_generator.cluster_analyzer.active_clusters)}\n"
        
        # Add text box
        fig.text(0.02, 0.98, summary_text, transform=fig.transFigure,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def create_parameter_evolution_plot(self, training_history: Dict[str, List],
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Create parameter evolution plot during training."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quantum Parameter Evolution During Training', fontsize=16, fontweight='bold')
        
        epochs = range(len(training_history.get('generator_loss', [])))
        
        # Plot 1: Loss evolution
        if 'generator_loss' in training_history and 'discriminator_loss' in training_history:
            axes[0, 0].plot(epochs, training_history['generator_loss'], 'r-', label='Generator')
            axes[0, 0].plot(epochs, training_history['discriminator_loss'], 'b-', label='Discriminator')
            axes[0, 0].set_title('Loss Evolution')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter magnitudes by type
        param_magnitudes = self._calculate_parameter_magnitudes()
        param_types = list(param_magnitudes.keys())
        param_values = list(param_magnitudes.values())
        
        colors = [self.gate_colors.get(ptype, '#95A5A6') for ptype in param_types]
        axes[0, 1].bar(param_types, param_values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Current Parameter Magnitudes')
        axes[0, 1].set_ylabel('RMS Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Mode coverage (if available)
        if 'mode_coverage' in training_history:
            mode_coverage = training_history['mode_coverage']
            if mode_coverage:
                mode1_coverage = [mc.get('mode1_coverage', 0) for mc in mode_coverage]
                mode2_coverage = [mc.get('mode2_coverage', 0) for mc in mode_coverage]
                balanced_coverage = [mc.get('balanced_coverage', 0) for mc in mode_coverage]
                
                axes[1, 0].plot(epochs, mode1_coverage, 'r-', label='Mode 1')
                axes[1, 0].plot(epochs, mode2_coverage, 'b-', label='Mode 2')
                axes[1, 0].plot(epochs, balanced_coverage, 'g-', linewidth=2, label='Balanced')
                axes[1, 0].set_title('Mode Coverage Evolution')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Coverage')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Circuit complexity metrics
        complexity_metrics = self._calculate_complexity_metrics()
        metric_names = list(complexity_metrics.keys())
        metric_values = list(complexity_metrics.values())
        
        axes[1, 1].bar(metric_names, metric_values, color='purple', alpha=0.7)
        axes[1, 1].set_title('Circuit Complexity Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Parameter evolution plot saved to: {save_path}")
        
        return fig
    
    def _calculate_parameter_magnitudes(self) -> Dict[str, float]:
        """Calculate RMS parameter magnitudes by type."""
        magnitudes = {}
        
        for op in self.operations:
            op_type = op['type']
            value = abs(op['value'])
            
            if op_type not in magnitudes:
                magnitudes[op_type] = []
            magnitudes[op_type].append(value)
        
        # Calculate RMS for each type
        rms_magnitudes = {}
        for op_type, values in magnitudes.items():
            rms_magnitudes[op_type] = np.sqrt(np.mean(np.array(values)**2))
        
        return rms_magnitudes
    
    def _calculate_complexity_metrics(self) -> Dict[str, float]:
        """Calculate circuit complexity metrics."""
        return {
            'Total Gates': len(self.operations),
            'Layers': self.n_layers,
            'Modes': self.n_modes,
            'Parameters': len(self.circuit.tf_parameters),
            'Connectivity': self.n_modes * (self.n_modes - 1) / 2  # Max possible connections
        }
    
    def export_circuit_data(self, save_path: str):
        """Export circuit data as JSON for analysis."""
        circuit_data = {
            'circuit_info': {
                'n_modes': self.n_modes,
                'n_layers': self.n_layers,
                'total_parameters': len(self.circuit.tf_parameters)
            },
            'operations': self.operations,
            'parameter_magnitudes': self._calculate_parameter_magnitudes(),
            'complexity_metrics': self._calculate_complexity_metrics()
        }
        
        if self.coordinate_generator:
            circuit_data['coordinate_info'] = {
                'coordinate_names': getattr(self.coordinate_generator, 'coordinate_names', ['X', 'Y']),
                'mode_mappings': getattr(self.coordinate_generator, 'mode_coordinate_mapping', {})
            }
        
        import json
        with open(save_path, 'w') as f:
            json.dump(circuit_data, f, indent=2, default=str)
        
        print(f"✅ Circuit data exported to: {save_path}")


def create_enhanced_circuit_visualization(quantum_circuit, coordinate_generator=None,
                                        save_dir: str = "results/training/circuit_visualization",
                                        training_history: Optional[Dict] = None):
    """
    Create comprehensive circuit visualization suite.
    
    Args:
        quantum_circuit: PureSFQuantumCircuit instance
        coordinate_generator: Optional coordinate generator
        save_dir: Directory to save visualizations
        training_history: Optional training history for evolution plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create enhanced visualizer
    visualizer = EnhancedQuantumCircuitVisualizer(quantum_circuit, coordinate_generator)
    
    # Create main circuit diagram
    circuit_fig = visualizer.create_visual_circuit(
        save_path=os.path.join(save_dir, "quantum_circuit_diagram.png"),
        show_values=True,
        show_coordinate_info=True
    )
    
    # Create parameter evolution plot if training history available
    if training_history:
        evolution_fig = visualizer.create_parameter_evolution_plot(
            training_history,
            save_path=os.path.join(save_dir, "parameter_evolution.png")
        )
    
    # Export circuit data
    visualizer.export_circuit_data(
        os.path.join(save_dir, "circuit_data.json")
    )
    
    print(f"✅ Enhanced circuit visualization suite created in: {save_dir}")
    
    return visualizer


def test_enhanced_visualizer():
    """Test the enhanced visualizer."""
    print("🧪 Testing Enhanced Circuit Visualizer...")
    
    try:
        # Import circuit
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit
        
        # Create test circuit
        circuit = PureSFQuantumCircuit(n_modes=4, n_layers=2, cutoff_dim=6)
        
        # Create visualizer
        visualizer = EnhancedQuantumCircuitVisualizer(circuit)
        
        # Create visualization
        fig = visualizer.create_visual_circuit(show_values=True)
        
        print("✅ Enhanced visualizer test successful!")
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_visualizer()
