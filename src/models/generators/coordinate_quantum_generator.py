"""
Coordinate-Wise Quantum Generator

This generator implements coordinate-wise quantum generation using single
quadrature measurements per mode. It uses cluster analysis to automatically
assign quantum modes to specific coordinates and data regions.

Key Features:
- Single quadrature measurements (realistic quantum implementation)
- Coordinate-wise mode assignment (X/Y coordinates handled separately)
- Automatic cluster analysis and probability weighting
- Hard assignment without learnable attention (pure quantum approach)
- Generalizable to any data shape through preprocessing
"""

import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict, Any, List, Tuple
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit
from src.utils.cluster_analyzer import ClusterAnalyzer

logger = logging.getLogger(__name__)


class CoordinateQuantumGenerator:
    """
    Coordinate-wise quantum generator with automatic cluster analysis.
    
    This generator uses single quadrature measurements and coordinate-wise
    mode assignment to generate data that matches any input distribution
    through automatic clustering and probability weighting.
    """
    
    def __init__(self,
                 latent_dim: int = 6,
                 output_dim: int = 2,
                 n_modes: int = 4,
                 layers: int = 2,
                 cutoff_dim: int = 8,
                 clustering_method: str = 'kmeans',
                 min_cluster_density: float = 0.01,
                 coordinate_names: List[str] = None,
                 name: str = "CoordinateQuantumGenerator"):
        """
        Initialize coordinate-wise quantum generator.
        
        Args:
            latent_dim: Dimension of input latent space
            output_dim: Dimension of output data (should match coordinate_names)
            n_modes: Number of quantum modes (should be >= output_dim * n_clusters)
            layers: Number of variational quantum layers
            cutoff_dim: Fock space cutoff dimension
            clustering_method: Method for automatic clustering ('kmeans', 'gmm', 'grid')
            min_cluster_density: Minimum probability density for active clusters
            coordinate_names: Names for coordinates (e.g., ['X', 'Y'])
            name: Generator name
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.clustering_method = clustering_method
        self.min_cluster_density = min_cluster_density
        self.coordinate_names = coordinate_names or [f'coord_{i}' for i in range(output_dim)]
        self.name = name
        
        # Calculate number of clusters based on available modes
        # Each cluster needs output_dim modes (one per coordinate)
        self.n_clusters = n_modes // output_dim
        
        if self.n_clusters < 1:
            raise ValueError(f"Need at least {output_dim} modes for {output_dim}D output, got {n_modes}")
        
        # Cluster analysis (will be initialized when data is provided)
        self.cluster_analyzer = None
        self.cluster_gains = None
        self.mode_coordinate_mapping = None
        
        # Create quantum circuit for single quadrature measurements
        self.quantum_circuit = PureSFQuantumCircuit(
            n_modes=n_modes,
            n_layers=layers,
            cutoff_dim=cutoff_dim,
            circuit_type="basic"
        )
        
        # Create input encoder (latent → quantum parameters)
        self.input_encoder = self._build_input_encoder()
        
        # Coordinate decoders (will be built after cluster analysis)
        self.coordinate_decoders = {}
        
        # State tracking
        self.last_measurements = None
        self.last_cluster_assignments = None
        
        logger.info(f"🎯 {name} initialized:")
        logger.info(f"   Architecture: {latent_dim}D → {n_modes} modes → {output_dim}D")
        logger.info(f"   Clusters: {self.n_clusters} (auto-detected)")
        logger.info(f"   Coordinates: {self.coordinate_names}")
        logger.info(f"   Clustering method: {clustering_method}")
        logger.info(f"   Single quadrature measurements: ✅")
    
    def _build_input_encoder(self) -> tf.keras.Model:
        """Build input encoder: latent → quantum parameters."""
        # Simple linear encoder for quantum parameter modulation
        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_modes * 2,  # 2 parameters per mode (displacement components)
                activation='tanh',  # Bounded activation for stable quantum parameters
                name='quantum_parameter_encoder'
            )
        ], name='input_encoder')
        
        # Build with dummy input
        dummy_input = tf.zeros([1, self.latent_dim])
        encoder(dummy_input)
        
        return encoder
    
    def analyze_target_data(self, target_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze target data distribution and set up coordinate mappings.
        
        Args:
            target_data: Target data to analyze [n_samples, n_features]
            
        Returns:
            Analysis results
        """
        print(f"🔍 Analyzing target data for coordinate-wise generation...")
        
        # Create cluster analyzer
        self.cluster_analyzer = ClusterAnalyzer(
            n_clusters=self.n_clusters,
            clustering_method=self.clustering_method,
            min_cluster_density=self.min_cluster_density,
            coordinate_names=self.coordinate_names
        )
        
        # Perform cluster analysis
        analysis_results = self.cluster_analyzer.analyze_data(target_data)
        
        # Extract cluster gains and mode mappings
        self.cluster_gains = self.cluster_analyzer.get_cluster_gains()
        self.mode_coordinate_mapping = self.cluster_analyzer.get_mode_coordinate_mapping()
        
        # Build coordinate decoders based on analysis
        self._build_coordinate_decoders()
        
        print(f"✅ Target data analysis complete:")
        print(f"   Active clusters: {len(self.cluster_analyzer.active_clusters)}")
        print(f"   Cluster gains: {self.cluster_gains}")
        print(f"   Mode mappings: {len(self.mode_coordinate_mapping)} modes assigned")
        
        return analysis_results
    
    def _build_coordinate_decoders(self):
        """Build coordinate-specific decoders based on cluster analysis."""
        if self.mode_coordinate_mapping is None:
            raise ValueError("Must analyze target data first")
        
        print(f"🔧 Building coordinate-specific decoders...")
        
        # Group modes by coordinate
        coordinate_modes = {}
        for mode_idx, mapping in self.mode_coordinate_mapping.items():
            coord_name = mapping['coordinate']
            if coord_name not in coordinate_modes:
                coordinate_modes[coord_name] = []
            coordinate_modes[coord_name].append({
                'mode_idx': mode_idx,
                'cluster_id': mapping['cluster_id'],
                'gain': mapping['gain'],
                'active': mapping['active']
            })
        
        # Create decoder for each coordinate
        for coord_name, modes in coordinate_modes.items():
            active_modes = [m for m in modes if m['active']]
            n_active_modes = len(active_modes)
            
            if n_active_modes > 0:
                # Create simple linear decoder for this coordinate
                decoder = tf.keras.Sequential([
                    tf.keras.layers.Dense(
                        1,  # Single coordinate output
                        activation='linear',
                        name=f'{coord_name}_coordinate_decoder'
                    )
                ], name=f'{coord_name}_decoder')
                
                # Build with dummy input (measurements from active modes)
                dummy_input = tf.zeros([1, n_active_modes])
                decoder(dummy_input)
                
                self.coordinate_decoders[coord_name] = {
                    'decoder': decoder,
                    'modes': active_modes,
                    'n_modes': n_active_modes
                }
                
                print(f"   {coord_name}: {n_active_modes} active modes")
            else:
                print(f"   ⚠️  {coord_name}: No active modes, using zero output")
                self.coordinate_decoders[coord_name] = {
                    'decoder': None,
                    'modes': [],
                    'n_modes': 0
                }
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate samples using coordinate-wise quantum approach.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, output_dim]
        """
        if self.cluster_analyzer is None:
            raise ValueError("Must analyze target data first using analyze_target_data()")
        
        batch_size = tf.shape(z)[0]
        
        # Step 1: Encode latent input to quantum parameters
        quantum_params = self.input_encoder(z)  # [batch_size, n_modes * 2]
        
        # Step 2: Generate quantum measurements for each sample
        batch_measurements = []
        batch_cluster_assignments = []
        
        for i in range(batch_size):
            # Apply quantum parameters to circuit
            sample_params = quantum_params[i]  # [n_modes * 2]
            self._apply_parameters_to_circuit(sample_params)
            
            # Execute quantum circuit and get single quadrature measurements
            quantum_state = self.quantum_circuit.execute()
            measurements = self._extract_single_quadrature_measurements(quantum_state)
            
            batch_measurements.append(measurements)
            
            # Assign sample to cluster (for tracking)
            # For now, use probabilistic assignment based on cluster gains
            cluster_probs = tf.nn.softmax(tf.constant(self.cluster_gains))
            cluster_assignment = tf.random.categorical(
                tf.math.log(cluster_probs[None, :]), 1
            )[0, 0]
            batch_cluster_assignments.append(cluster_assignment)
        
        # Stack measurements
        batch_measurements = tf.stack(batch_measurements, axis=0)  # [batch_size, n_modes]
        batch_cluster_assignments = tf.stack(batch_cluster_assignments, axis=0)
        
        # Step 3: Decode measurements to coordinates
        coordinate_outputs = {}
        
        for coord_name, decoder_info in self.coordinate_decoders.items():
            if decoder_info['n_modes'] > 0:
                # Extract measurements for this coordinate's modes
                mode_indices = [m['mode_idx'] for m in decoder_info['modes']]
                coord_measurements = tf.gather(batch_measurements, mode_indices, axis=1)
                
                # Apply coordinate decoder
                coord_output = decoder_info['decoder'](coord_measurements)
                # Ensure it's 1D [batch_size] - squeeze all dimensions except batch
                coord_output = tf.squeeze(coord_output)
                # If still multi-dimensional, take first element
                if len(coord_output.shape) > 1:
                    coord_output = coord_output[:, 0]
                coordinate_outputs[coord_name] = coord_output
            else:
                # No active modes for this coordinate
                coordinate_outputs[coord_name] = tf.zeros([batch_size])
        
        # Step 4: Combine coordinates into final output
        output_list = []
        for coord_name in self.coordinate_names:
            if coord_name in coordinate_outputs:
                coord_output = coordinate_outputs[coord_name]
                # Ensure it's 1D [batch_size]
                while len(coord_output.shape) > 1:
                    coord_output = tf.squeeze(coord_output, axis=-1)
                output_list.append(coord_output)
            else:
                output_list.append(tf.zeros([batch_size]))
        
        generated_samples = tf.stack(output_list, axis=1)  # [batch_size, output_dim]
        
        # Store for analysis
        self.last_measurements = batch_measurements
        self.last_cluster_assignments = batch_cluster_assignments
        
        return generated_samples
    
    def _apply_parameters_to_circuit(self, params: tf.Tensor):
        """Apply quantum parameters to the circuit."""
        # params shape: [n_modes * 2]
        for mode in range(self.n_modes):
            # Extract displacement parameters for this mode
            disp_x = params[mode * 2]
            disp_y = params[mode * 2 + 1]
            
            # Apply to quantum circuit
            if f'displacement_0_{mode}' in self.quantum_circuit.tf_parameters:
                self.quantum_circuit.tf_parameters[f'displacement_0_{mode}'].assign([disp_x])
            
            # For single quadrature, we can also set a small squeezing
            if f'squeeze_r_0_{mode}' in self.quantum_circuit.tf_parameters:
                self.quantum_circuit.tf_parameters[f'squeeze_r_0_{mode}'].assign([0.1])  # Small squeezing
    
    def _extract_single_quadrature_measurements(self, quantum_state) -> tf.Tensor:
        """Extract single quadrature measurements (X quadrature only)."""
        measurements = []
        
        for mode in range(self.n_modes):
            # Measure X quadrature only (position quadrature)
            x_measurement = quantum_state.quad_expectation(mode, 0)  # 0 = X quadrature
            measurements.append(x_measurement)
        
        return tf.stack(measurements)  # [n_modes]
    
    def visualize_generation(self, z: tf.Tensor, target_data: np.ndarray, save_path: Optional[str] = None):
        """Create comprehensive visualization of coordinate-wise generation."""
        if self.cluster_analyzer is None:
            print("❌ No cluster analysis available")
            return
        
        # Generate samples
        generated_samples = self.generate(z)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎯 Coordinate-Wise Quantum Generation Analysis', fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Plot 1: Target data with clusters
        cluster_labels = self.cluster_analyzer.cluster_assignments
        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = target_data[cluster_mask]
            is_active = i in self.cluster_analyzer.active_clusters
            
            alpha = 0.7 if is_active else 0.3
            marker = 'o' if is_active else 'x'
            
            if len(cluster_data) > 0:
                axes[0, 0].scatter(cluster_data[:, 0], cluster_data[:, 1], 
                                 c=colors[i % len(colors)], alpha=alpha, 
                                 marker=marker, s=50, label=f'Cluster {i}')
        
        axes[0, 0].set_title('Target Data Clusters')
        axes[0, 0].set_xlabel(self.coordinate_names[0])
        axes[0, 0].set_ylabel(self.coordinate_names[1])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Generated data
        axes[0, 1].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                          alpha=0.6, c='green', s=50)
        axes[0, 1].set_title('Generated Data')
        axes[0, 1].set_xlabel(self.coordinate_names[0])
        axes[0, 1].set_ylabel(self.coordinate_names[1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Overlay comparison
        axes[0, 2].scatter(target_data[:, 0], target_data[:, 1], 
                          alpha=0.5, c='blue', s=30, label='Target')
        axes[0, 2].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                          alpha=0.5, c='red', s=30, label='Generated')
        axes[0, 2].set_title('Target vs Generated')
        axes[0, 2].set_xlabel(self.coordinate_names[0])
        axes[0, 2].set_ylabel(self.coordinate_names[1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Cluster probability distribution
        cluster_indices = range(self.n_clusters)
        bars = axes[1, 0].bar(cluster_indices, self.cluster_gains, 
                             color=[colors[i % len(colors)] for i in cluster_indices],
                             alpha=0.7)
        
        axes[1, 0].set_title('Cluster Probability Gains')
        axes[1, 0].set_xlabel('Cluster Index')
        axes[1, 0].set_ylabel('Probability Gain')
        axes[1, 0].set_xticks(cluster_indices)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Mode assignments
        if self.mode_coordinate_mapping:
            mode_data = []
            for mode_idx in sorted(self.mode_coordinate_mapping.keys()):
                mapping = self.mode_coordinate_mapping[mode_idx]
                mode_data.append({
                    'mode': mode_idx,
                    'cluster': mapping['cluster_id'],
                    'coordinate': mapping['coordinate'],
                    'gain': mapping['gain'],
                    'active': mapping['active']
                })
            
            # Create mode assignment visualization
            mode_indices = [d['mode'] for d in mode_data]
            gains = [d['gain'] for d in mode_data]
            coord_colors = ['red' if d['coordinate'] == self.coordinate_names[0] else 'blue' 
                           for d in mode_data]
            
            bars = axes[1, 1].bar(mode_indices, gains, color=coord_colors, alpha=0.7)
            
            # Mark inactive modes
            for i, d in enumerate(mode_data):
                if not d['active']:
                    bars[i].set_alpha(0.3)
                    bars[i].set_hatch('///')
            
            axes[1, 1].set_title('Mode Assignments')
            axes[1, 1].set_xlabel('Mode Index')
            axes[1, 1].set_ylabel('Gain')
            axes[1, 1].set_xticks(mode_indices)
            
            # Add legend for coordinates
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label=self.coordinate_names[0]),
                             Patch(facecolor='blue', label=self.coordinate_names[1])]
            axes[1, 1].legend(handles=legend_elements)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Quantum measurements distribution
        if self.last_measurements is not None:
            measurements_np = self.last_measurements.numpy()
            
            for mode in range(min(4, self.n_modes)):  # Show first 4 modes
                axes[1, 2].hist(measurements_np[:, mode], bins=20, alpha=0.6, 
                               label=f'Mode {mode}')
            
            axes[1, 2].set_title('Quantum Measurements Distribution')
            axes[1, 2].set_xlabel('Measurement Value')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🎨 Coordinate generation visualization saved to: {save_path}")
        
        plt.show()
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables."""
        variables = []
        
        # Input encoder variables
        variables.extend(self.input_encoder.trainable_variables)
        
        # Quantum circuit variables
        variables.extend(self.quantum_circuit.trainable_variables)
        
        # Coordinate decoder variables
        for decoder_info in self.coordinate_decoders.values():
            if decoder_info['decoder'] is not None:
                variables.extend(decoder_info['decoder'].trainable_variables)
        
        return variables
    
    def compute_quantum_cost(self) -> tf.Tensor:
        """
        Compute quantum cost for physics-informed loss.
        
        Returns:
            Quantum cost tensor for regularization
        """
        # Basic quantum cost based on parameter magnitudes
        total_cost = 0.0
        
        # Cost from quantum circuit parameters
        for param in self.quantum_circuit.trainable_variables:
            total_cost += tf.reduce_sum(tf.square(param))
        
        # Cost from encoder parameters (encourage small quantum parameter modulation)
        for param in self.input_encoder.trainable_variables:
            total_cost += 0.1 * tf.reduce_sum(tf.square(param))
        
        # Cost from coordinate decoders
        for decoder_info in self.coordinate_decoders.values():
            if decoder_info['decoder'] is not None:
                for param in decoder_info['decoder'].trainable_variables:
                    total_cost += 0.01 * tf.reduce_sum(tf.square(param))
        
        return total_cost
    
    def get_coordinate_mapping_summary(self) -> str:
        """Get summary of coordinate mappings."""
        if self.mode_coordinate_mapping is None:
            return "No coordinate mapping available - analyze target data first"
        
        summary = f"""
🎯 COORDINATE-WISE QUANTUM GENERATOR SUMMARY
===========================================

Architecture: {self.latent_dim}D → {self.n_modes} modes → {self.output_dim}D
Clusters: {self.n_clusters} (auto-detected)
Coordinates: {self.coordinate_names}
Clustering method: {self.clustering_method}

Mode Assignments:
"""
        
        for mode_idx in sorted(self.mode_coordinate_mapping.keys()):
            mapping = self.mode_coordinate_mapping[mode_idx]
            status = "✅ ACTIVE" if mapping['active'] else "❌ INACTIVE"
            summary += f"  Mode {mode_idx}: {status} → Cluster {mapping['cluster_id']}, {mapping['coordinate']} (gain={mapping['gain']:.3f})\n"
        
        summary += f"""
Cluster Gains: {self.cluster_gains}
Active Clusters: {len(self.cluster_analyzer.active_clusters) if self.cluster_analyzer else 0}

Features:
✅ Single quadrature measurements (realistic quantum)
✅ Coordinate-wise mode assignment
✅ Automatic cluster analysis
✅ Hard assignment (no learnable attention)
✅ Generalizable to any data shape
===========================================
"""
        return summary


def test_coordinate_quantum_generator():
    """Test the coordinate quantum generator."""
    print("🧪 Testing CoordinateQuantumGenerator...")
    
    # Create test bimodal data
    np.random.seed(42)
    cluster1 = np.random.normal([-1.5, -1.5], 0.3, (200, 2))
    cluster2 = np.random.normal([1.5, 1.5], 0.3, (300, 2))
    target_data = np.vstack([cluster1, cluster2])
    
    # Create generator
    generator = CoordinateQuantumGenerator(
        latent_dim=6,
        output_dim=2,
        n_modes=4,
        layers=2,
        cutoff_dim=6,
        clustering_method='kmeans',
        coordinate_names=['X', 'Y']
    )
    
    print(f"Created generator with {len(generator.trainable_variables)} trainable variables")
    
    # Analyze target data
    analysis_results = generator.analyze_target_data(target_data)
    
    print(generator.get_coordinate_mapping_summary())
    
    # Test generation
    test_z = tf.random.normal([100, 6])
    generated_samples = generator.generate(test_z)
    
    print(f"✅ Generation successful: {test_z.shape} → {generated_samples.shape}")
    print(f"   Generated range: X=[{generated_samples[:, 0].numpy().min():.3f}, {generated_samples[:, 0].numpy().max():.3f}], "
          f"Y=[{generated_samples[:, 1].numpy().min():.3f}, {generated_samples[:, 1].numpy().max():.3f}]")
    
    # Visualize results
    generator.visualize_generation(test_z, target_data)
    
    print("✅ CoordinateQuantumGenerator test completed!")


if __name__ == "__main__":
    test_coordinate_quantum_generator()
