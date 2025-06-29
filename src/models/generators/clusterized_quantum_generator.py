"""
Clusterized Quantum Generator - Pure Quantum Implementation

This generator implements the target data clusterization strategy without neural networks:
- Each quantum mode is assigned to a specific coordinate/cluster combination
- Simple linear transformations based on cluster statistics
- Weighted combination of mode outputs
- No classical neural network decoders

Key Features:
- Mode specialization (each mode handles specific coordinate/cluster)
- Pure quantum processing (no neural network bottlenecks)
- Direct scaling based on cluster analysis
- Maintains quantum advantages while preventing mode collapse
"""

import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict, Any, List, Tuple, Union
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


class SimpleCoordinateTransform:
    """
    Simple linear transformation for quantum measurements to coordinates.
    No neural networks - just direct scaling based on cluster statistics.
    """
    
    def __init__(self, cluster_centers: np.ndarray, cluster_ranges: np.ndarray, mode_assignments: Dict):
        """
        Initialize coordinate transformation.
        
        Args:
            cluster_centers: [n_clusters, n_coordinates] cluster centers
            cluster_ranges: [n_clusters, n_coordinates] cluster ranges
            mode_assignments: Dictionary mapping mode_idx to assignment info
        """
        self.cluster_centers = cluster_centers
        self.cluster_ranges = cluster_ranges
        self.mode_assignments = mode_assignments
        
        logger.info(f"🔧 SimpleCoordinateTransform initialized:")
        logger.info(f"   Cluster centers: {cluster_centers.shape}")
        logger.info(f"   Cluster ranges: {cluster_ranges.shape}")
        logger.info(f"   Mode assignments: {len(mode_assignments)}")
    
    def transform_mode_measurement(self, mode_idx: int, measurement: tf.Tensor) -> tf.Tensor:
        """
        Transform a single mode's measurement to coordinate space.
        
        Args:
            mode_idx: Index of the quantum mode
            measurement: Raw quantum measurement (typically in [-1, 1])
            
        Returns:
            Transformed coordinate value
        """
        if mode_idx not in self.mode_assignments:
            return tf.constant(0.0)
        
        assignment = self.mode_assignments[mode_idx]
        cluster_id = assignment['cluster_id']
        coord_idx = assignment['coordinate_index']
        
        # Get cluster-specific scaling parameters
        center = self.cluster_centers[cluster_id, coord_idx]
        range_scale = self.cluster_ranges[cluster_id, coord_idx]
        
        # Simple linear transformation: scale and shift
        # Quantum measurements are typically in [-1, 1], we want to map to cluster range
        scaled_measurement = measurement * tf.constant(range_scale, dtype=tf.float32)
        shifted_measurement = scaled_measurement + tf.constant(center, dtype=tf.float32)
        
        return shifted_measurement
    
    def get_coordinate_info(self, mode_idx: int) -> Optional[Dict[str, Any]]:
        """Get coordinate information for a specific mode."""
        if mode_idx not in self.mode_assignments:
            return None
        
        assignment = self.mode_assignments[mode_idx]
        cluster_id = assignment['cluster_id']
        coord_idx = assignment['coordinate_index']
        
        return {
            'cluster_id': cluster_id,
            'coordinate_index': coord_idx,
            'coordinate_name': assignment['coordinate'],
            'cluster_center': self.cluster_centers[cluster_id, coord_idx],
            'cluster_range': self.cluster_ranges[cluster_id, coord_idx],
            'gain': assignment['gain']
        }


class ClusterizedQuantumGenerator:
    """
    Quantum generator using target data clusterization strategy.
    
    This implementation:
    1. Analyzes target data to create clusters
    2. Assigns each quantum mode to specific coordinate/cluster
    3. Uses simple linear transformations (no neural networks)
    4. Combines mode outputs using weighted averaging
    """
    
    def __init__(self,
                 latent_dim: int = 6,
                 output_dim: int = 2,
                 n_modes: int = 4,
                 layers: int = 2,
                 cutoff_dim: int = 8,
                 clustering_method: str = 'kmeans',
                 coordinate_names: List[str] = None,
                 name: str = "ClusterizedQuantumGenerator"):
        """
        Initialize clusterized quantum generator.
        
        Args:
            latent_dim: Dimension of input latent space
            output_dim: Dimension of output data
            n_modes: Number of quantum modes
            layers: Number of variational quantum layers
            cutoff_dim: Fock space cutoff dimension
            clustering_method: Method for clustering ('kmeans', 'gmm')
            coordinate_names: Names for coordinates (e.g., ['X', 'Y'])
            name: Generator name
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.clustering_method = clustering_method
        self.coordinate_names = coordinate_names or [f'coord_{i}' for i in range(output_dim)]
        self.name = name
        
        # Calculate number of clusters based on available modes
        self.n_clusters = n_modes // output_dim
        
        if self.n_clusters < 1:
            raise ValueError(f"Need at least {output_dim} modes for {output_dim}D output, got {n_modes}")
        
        # Cluster analysis components (initialized when data is provided)
        self.cluster_analyzer = None
        self.cluster_centers = None
        self.cluster_ranges = None
        self.mode_assignments = None
        self.coordinate_transform = None
        
        # Create quantum circuit
        self.quantum_circuit = PureSFQuantumCircuit(
            n_modes=n_modes,
            n_layers=layers,
            cutoff_dim=cutoff_dim,
            circuit_type="variational"
        )
        
        # Create input encoder (latent → quantum parameters)
        self.input_encoder = self._build_input_encoder()
        
        # State tracking
        self.last_measurements = None
        self.last_cluster_assignments = None
        
        logger.info(f"🎯 {name} initialized:")
        logger.info(f"   Architecture: {latent_dim}D → {n_modes} modes → {output_dim}D")
        logger.info(f"   Clusters: {self.n_clusters} (auto-detected)")
        logger.info(f"   Coordinates: {self.coordinate_names}")
        logger.info(f"   Pure quantum processing: ✅ (no neural networks)")
    
    def _build_input_encoder(self) -> tf.keras.Model:
        """Build simple input encoder: latent → quantum parameter modulation."""
        # Simple linear encoder for quantum parameter modulation
        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_modes * 2,  # 2 parameters per mode for displacement
                activation='tanh',  # Bounded activation for stable quantum parameters
                kernel_initializer='glorot_uniform',
                name='quantum_parameter_encoder'
            )
        ], name='input_encoder')
        
        # Build with dummy input
        dummy_input = tf.zeros([1, self.latent_dim])
        encoder(dummy_input)
        
        return encoder
    
    def analyze_target_data(self, target_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze target data distribution and set up mode assignments.
        
        Args:
            target_data: Target data to analyze [n_samples, n_features]
            
        Returns:
            Analysis results
        """
        print(f"🔍 Analyzing target data for clusterization...")
        
        # Create cluster analyzer
        self.cluster_analyzer = ClusterAnalyzer(
            n_clusters=self.n_clusters,
            clustering_method=self.clustering_method,
            coordinate_names=self.coordinate_names
        )
        
        # Perform cluster analysis
        analysis_results = self.cluster_analyzer.analyze_data(target_data)
        
        # Extract cluster information
        self.cluster_centers = self.cluster_analyzer.cluster_centers
        self.mode_assignments = self.cluster_analyzer.get_mode_coordinate_mapping()
        
        # Calculate cluster ranges for scaling
        self.cluster_ranges = self._calculate_cluster_ranges(target_data)
        
        # Create coordinate transformation
        self.coordinate_transform = SimpleCoordinateTransform(
            self.cluster_centers,
            self.cluster_ranges,
            self.mode_assignments
        )
        
        print(f"✅ Target data analysis complete:")
        print(f"   Active clusters: {len(self.cluster_analyzer.active_clusters)}")
        print(f"   Cluster centers: {self.cluster_centers}")
        print(f"   Cluster ranges: {self.cluster_ranges}")
        print(f"   Mode assignments: {len(self.mode_assignments)} modes assigned")
        
        return analysis_results
    
    def _calculate_cluster_ranges(self, target_data: np.ndarray) -> np.ndarray:
        """
        Calculate cluster ranges for proper scaling.
        
        Args:
            target_data: Target data [n_samples, n_features]
            
        Returns:
            Cluster ranges [n_clusters, n_coordinates]
        """
        cluster_labels = self.cluster_analyzer.cluster_assignments
        cluster_ranges = np.zeros((self.n_clusters, self.output_dim))
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = target_data[cluster_mask]
            
            if len(cluster_data) > 0:
                for coord_idx in range(self.output_dim):
                    coord_data = cluster_data[:, coord_idx]
                    coord_std = np.std(coord_data)
                    # Use 2 * std as range (covers ~95% of data)
                    cluster_ranges[cluster_id, coord_idx] = max(2.0 * coord_std, 0.5)  # Minimum range
            else:
                # Default range for empty clusters
                cluster_ranges[cluster_id, :] = 1.0
        
        return cluster_ranges
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate samples using clusterization strategy.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, output_dim]
        """
        if self.coordinate_transform is None:
            raise ValueError("Must analyze target data first using analyze_target_data()")
        
        batch_size = tf.shape(z)[0]
        
        # Step 1: Encode latent input to quantum parameters
        quantum_params = self.input_encoder(z)  # [batch_size, n_modes * 2]
        
        # Step 2: Execute quantum circuits for each sample individually
        batch_measurements = []
        
        for i in range(batch_size):
            # Get parameters for this sample
            sample_params = quantum_params[i]  # [n_modes * 2]
            
            # Execute quantum circuit with this sample's parameters
            quantum_state = self.quantum_circuit.execute(input_encoding=sample_params)
            
            # Extract measurements from quantum state
            measurements = self.quantum_circuit.extract_measurements(quantum_state)
            batch_measurements.append(measurements)
        
        # Stack all measurements
        batch_measurements = tf.stack(batch_measurements, axis=0)  # [batch_size, n_modes]
        
        # Step 3: Transform measurements to coordinates using mode assignments
        generated_samples = []
        
        for i in range(batch_size):
            sample_measurements = batch_measurements[i]  # [n_modes]
            
            # Transform each mode's measurement to coordinate space
            mode_contributions = {}
            
            for mode_idx in range(self.n_modes):
                if mode_idx < tf.shape(sample_measurements)[0]:
                    measurement = sample_measurements[mode_idx]
                    
                    # Transform measurement using coordinate transform
                    coord_value = self.coordinate_transform.transform_mode_measurement(
                        mode_idx, measurement
                    )
                    
                    # Get coordinate info for this mode
                    coord_info = self.coordinate_transform.get_coordinate_info(mode_idx)
                    
                    if coord_info is not None:
                        coord_idx = coord_info['coordinate_index']
                        weight = coord_info['gain']
                        
                        if coord_idx not in mode_contributions:
                            mode_contributions[coord_idx] = []
                        
                        mode_contributions[coord_idx].append({
                            'value': coord_value,
                            'weight': weight
                        })
            
            # Combine contributions for each coordinate
            sample_coords = []
            
            for coord_idx in range(self.output_dim):
                if coord_idx in mode_contributions:
                    contributions = mode_contributions[coord_idx]
                    
                    # Weighted average of mode contributions
                    weighted_sum = tf.constant(0.0)
                    total_weight = tf.constant(0.0)
                    
                    for contrib in contributions:
                        weighted_sum += contrib['value'] * contrib['weight']
                        total_weight += contrib['weight']
                    
                    # Avoid division by zero
                    coord_value = tf.cond(
                        total_weight > 1e-8,
                        lambda: weighted_sum / total_weight,
                        lambda: tf.constant(0.0)
                    )
                else:
                    # No modes assigned to this coordinate
                    coord_value = tf.constant(0.0)
                
                sample_coords.append(coord_value)
            
            generated_samples.append(tf.stack(sample_coords))
        
        generated_samples = tf.stack(generated_samples, axis=0)  # [batch_size, output_dim]
        
        # Store for analysis
        self.last_measurements = batch_measurements
        
        return generated_samples
    
    def visualize_generation(self, z: tf.Tensor, target_data: np.ndarray, save_path: Optional[str] = None):
        """Create comprehensive visualization of clusterized generation."""
        if self.cluster_analyzer is None:
            print("❌ No cluster analysis available")
            return
        
        # Generate samples
        generated_samples = self.generate(z)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎯 Clusterized Quantum Generation Analysis', fontsize=16, fontweight='bold')
        
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
        
        # Plot 4: Cluster centers and ranges
        cluster_indices = range(self.n_clusters)
        cluster_gains = self.cluster_analyzer.get_cluster_gains()
        bars = axes[1, 0].bar(cluster_indices, cluster_gains, 
                             color=[colors[i % len(colors)] for i in cluster_indices],
                             alpha=0.7)
        
        axes[1, 0].set_title('Cluster Probability Gains')
        axes[1, 0].set_xlabel('Cluster Index')
        axes[1, 0].set_ylabel('Probability Gain')
        axes[1, 0].set_xticks(cluster_indices)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Mode assignments
        if self.mode_assignments:
            mode_data = []
            for mode_idx in sorted(self.mode_assignments.keys()):
                mapping = self.mode_assignments[mode_idx]
                mode_data.append({
                    'mode': mode_idx,
                    'cluster': mapping['cluster_id'],
                    'coordinate': mapping['coordinate'],
                    'gain': mapping['gain']
                })
            
            # Create mode assignment visualization
            mode_indices = [d['mode'] for d in mode_data]
            gains = [d['gain'] for d in mode_data]
            coord_colors = ['red' if d['coordinate'] == self.coordinate_names[0] else 'blue' 
                           for d in mode_data]
            
            bars = axes[1, 1].bar(mode_indices, gains, color=coord_colors, alpha=0.7)
            
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
            print(f"🎨 Clusterized generation visualization saved to: {save_path}")
        
        plt.show()
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables."""
        variables = []
        
        # Input encoder variables
        variables.extend(self.input_encoder.trainable_variables)
        
        # Quantum circuit variables
        variables.extend(self.quantum_circuit.trainable_variables)
        
        # No coordinate decoder variables (pure quantum approach)
        
        return variables
    
    def get_generation_summary(self) -> str:
        """Get summary of generation strategy."""
        if self.mode_assignments is None:
            return "No mode assignments available - analyze target data first"
        
        summary = f"""
🎯 CLUSTERIZED QUANTUM GENERATOR SUMMARY
========================================

Architecture: {self.latent_dim}D → {self.n_modes} modes → {self.output_dim}D
Clusters: {self.n_clusters} (auto-detected)
Coordinates: {self.coordinate_names}
Clustering method: {self.clustering_method}

Mode Assignments:
"""
        
        for mode_idx in sorted(self.mode_assignments.keys()):
            mapping = self.mode_assignments[mode_idx]
            summary += f"  Mode {mode_idx}: Cluster {mapping['cluster_id']}, {mapping['coordinate']} (gain={mapping['gain']:.3f})\n"
        
        summary += f"""
Cluster Centers: 
{self.cluster_centers}

Cluster Ranges:
{self.cluster_ranges}

Features:
✅ Pure quantum processing (no neural networks)
✅ Mode specialization (coordinate/cluster assignment)
✅ Simple linear transformations
✅ Weighted mode combination
✅ Direct scaling based on cluster statistics
========================================
"""
        return summary


def test_clusterized_quantum_generator():
    """Test the clusterized quantum generator."""
    print("🧪 Testing ClusterizedQuantumGenerator...")
    
    # Create test bimodal data
    np.random.seed(42)
    cluster1 = np.random.normal([-1.5, -1.5], 0.3, (200, 2))
    cluster2 = np.random.normal([1.5, 1.5], 0.3, (300, 2))
    target_data = np.vstack([cluster1, cluster2])
    
    # Create generator
    generator = ClusterizedQuantumGenerator(
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
    
    print(generator.get_generation_summary())
    
    # Test generation
    test_z = tf.random.normal([100, 6])
    generated_samples = generator.generate(test_z)
    
    print(f"✅ Generation successful: {test_z.shape} → {generated_samples.shape}")
    print(f"   Generated range: X=[{generated_samples[:, 0].numpy().min():.3f}, {generated_samples[:, 0].numpy().max():.3f}], "
          f"Y=[{generated_samples[:, 1].numpy().min():.3f}, {generated_samples[:, 1].numpy().max():.3f}]")
    
    # Visualize results
    generator.visualize_generation(test_z, target_data)
    
    print("✅ ClusterizedQuantumGenerator test completed!")


if __name__ == "__main__":
    test_clusterized_quantum_generator()
