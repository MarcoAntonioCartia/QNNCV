"""
Cluster Analyzer for Coordinate-Wise Quantum Generation

This module provides automatic data clustering and probability analysis
for coordinate-wise quantum generation. It can handle any data shape
by automatically detecting meaningful clusters and computing their
probability densities.

Key Features:
- Automatic clustering with multiple algorithms
- Probability density calculation per cluster
- Empty cluster detection and handling
- Coordinate-wise analysis for quantum mode assignment
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    Analyzes data distribution and creates cluster assignments for quantum generation.
    
    Supports multiple clustering algorithms and automatically handles
    empty clusters for coordinate-wise quantum mode assignment.
    """
    
    def __init__(self, 
                 n_clusters: int = 2,
                 clustering_method: str = 'kmeans',
                 min_cluster_density: float = 0.01,
                 coordinate_names: List[str] = None):
        """
        Initialize cluster analyzer.
        
        Args:
            n_clusters: Number of clusters to create
            clustering_method: 'kmeans', 'gmm', 'dbscan', or 'grid'
            min_cluster_density: Minimum probability density to consider cluster active
            coordinate_names: Names for coordinates (e.g., ['X', 'Y'])
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.min_cluster_density = min_cluster_density
        self.coordinate_names = coordinate_names or [f'coord_{i}' for i in range(2)]
        
        # Cluster analysis results
        self.cluster_centers = None
        self.cluster_probabilities = None
        self.cluster_assignments = None
        self.active_clusters = None
        self.coordinate_ranges = None
        
        logger.info(f"ClusterAnalyzer initialized:")
        logger.info(f"  Clusters: {n_clusters}")
        logger.info(f"  Method: {clustering_method}")
        logger.info(f"  Min density: {min_cluster_density}")
        logger.info(f"  Coordinates: {self.coordinate_names}")
    
    def analyze_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete cluster analysis on input data.
        
        Args:
            data: Input data array [n_samples, n_features]
            
        Returns:
            Dictionary containing cluster analysis results
        """
        print(f"🔍 Analyzing data distribution...")
        print(f"   Data shape: {data.shape}")
        print(f"   Data range: X=[{data[:, 0].min():.3f}, {data[:, 0].max():.3f}], "
              f"Y=[{data[:, 1].min():.3f}, {data[:, 1].max():.3f}]")
        
        # Store coordinate ranges for later use
        self.coordinate_ranges = {
            self.coordinate_names[i]: (data[:, i].min(), data[:, i].max())
            for i in range(min(len(self.coordinate_names), data.shape[1]))
        }
        
        # Perform clustering
        cluster_labels = self._perform_clustering(data)
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
        
        # Determine active clusters
        active_clusters = self._determine_active_clusters(cluster_stats)
        
        # Store results
        self.cluster_assignments = cluster_labels
        self.cluster_probabilities = cluster_stats['probabilities']
        self.cluster_centers = cluster_stats['centers']
        self.active_clusters = active_clusters
        
        # Create mode assignments and store them
        self.mode_assignments = self._create_mode_assignments()
        
        # Create comprehensive analysis report
        analysis_results = {
            'n_samples': len(data),
            'n_features': data.shape[1],
            'n_clusters': self.n_clusters,
            'cluster_labels': cluster_labels,
            'cluster_centers': self.cluster_centers,
            'cluster_probabilities': self.cluster_probabilities,
            'active_clusters': active_clusters,
            'coordinate_ranges': self.coordinate_ranges,
            'cluster_statistics': cluster_stats,
            'mode_assignments': self.mode_assignments
        }
        
        self._print_analysis_summary(analysis_results)
        
        return analysis_results
    
    def _perform_clustering(self, data: np.ndarray) -> np.ndarray:
        """Perform clustering using specified method."""
        print(f"   🎯 Performing {self.clustering_method} clustering...")
        
        if self.clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(data)
            
        elif self.clustering_method == 'gmm':
            clusterer = GaussianMixture(n_components=self.n_clusters, random_state=42)
            labels = clusterer.fit_predict(data)
            
        elif self.clustering_method == 'dbscan':
            # Estimate eps based on data scale
            distances = cdist(data, data)
            eps = np.percentile(distances[distances > 0], 5)
            clusterer = DBSCAN(eps=eps, min_samples=5)
            labels = clusterer.fit_predict(data)
            # Handle noise points by assigning to nearest cluster
            if -1 in labels:
                noise_mask = labels == -1
                valid_labels = labels[~noise_mask]
                if len(valid_labels) > 0:
                    # Assign noise points to nearest valid cluster
                    for i in range(len(labels)):
                        if labels[i] == -1:
                            labels[i] = np.random.choice(valid_labels)
            
        elif self.clustering_method == 'grid':
            labels = self._grid_clustering(data)
            
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Ensure we have the right number of clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) != self.n_clusters:
            print(f"   ⚠️  Warning: Found {len(unique_labels)} clusters, expected {self.n_clusters}")
            # Reassign to ensure correct number of clusters
            labels = self._reassign_clusters(data, labels)
        
        return labels
    
    def _grid_clustering(self, data: np.ndarray) -> np.ndarray:
        """Perform grid-based clustering."""
        # Create grid based on data range
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        
        # For bimodal case, create 2x1 or 1x2 grid
        if self.n_clusters == 2:
            # Determine if data is more spread horizontally or vertically
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if x_range > y_range:
                # Split horizontally
                x_mid = (x_min + x_max) / 2
                labels = (data[:, 0] > x_mid).astype(int)
            else:
                # Split vertically
                y_mid = (y_min + y_max) / 2
                labels = (data[:, 1] > y_mid).astype(int)
        else:
            # For more clusters, create square grid
            grid_size = int(np.ceil(np.sqrt(self.n_clusters)))
            x_bins = np.linspace(x_min, x_max, grid_size + 1)
            y_bins = np.linspace(y_min, y_max, grid_size + 1)
            
            x_indices = np.digitize(data[:, 0], x_bins) - 1
            y_indices = np.digitize(data[:, 1], y_bins) - 1
            
            # Combine indices to create cluster labels
            labels = x_indices * grid_size + y_indices
            labels = np.clip(labels, 0, self.n_clusters - 1)
        
        return labels
    
    def _reassign_clusters(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Reassign cluster labels to ensure correct number of clusters."""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < self.n_clusters:
            # Too few clusters - split largest cluster
            for target_clusters in range(len(unique_labels), self.n_clusters):
                # Find largest cluster
                cluster_sizes = [np.sum(labels == label) for label in unique_labels]
                largest_cluster = unique_labels[np.argmax(cluster_sizes)]
                
                # Split largest cluster using K-means
                cluster_mask = labels == largest_cluster
                cluster_data = data[cluster_mask]
                
                if len(cluster_data) > 1:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    sub_labels = kmeans.fit_predict(cluster_data)
                    
                    # Assign new cluster label
                    new_label = max(unique_labels) + 1
                    labels[cluster_mask] = np.where(sub_labels == 0, largest_cluster, new_label)
                    unique_labels = np.append(unique_labels, new_label)
        
        elif len(unique_labels) > self.n_clusters:
            # Too many clusters - merge smallest clusters
            while len(unique_labels) > self.n_clusters:
                cluster_sizes = [np.sum(labels == label) for label in unique_labels]
                smallest_idx = np.argmin(cluster_sizes)
                smallest_cluster = unique_labels[smallest_idx]
                
                # Find nearest cluster center
                centers = np.array([data[labels == label].mean(axis=0) for label in unique_labels])
                smallest_center = centers[smallest_idx]
                
                distances = np.linalg.norm(centers - smallest_center, axis=1)
                distances[smallest_idx] = np.inf  # Exclude self
                nearest_idx = np.argmin(distances)
                nearest_cluster = unique_labels[nearest_idx]
                
                # Merge clusters
                labels[labels == smallest_cluster] = nearest_cluster
                unique_labels = np.delete(unique_labels, smallest_idx)
        
        # Relabel to ensure consecutive numbering
        for i, label in enumerate(np.unique(labels)):
            labels[labels == label] = i
        
        return labels
    
    def _calculate_cluster_statistics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics for each cluster."""
        n_samples = len(data)
        unique_labels = np.unique(labels)
        
        centers = []
        probabilities = []
        sizes = []
        coordinate_stats = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = data[cluster_mask]
            cluster_size = len(cluster_data)
            
            # Basic statistics
            center = cluster_data.mean(axis=0)
            probability = cluster_size / n_samples
            
            centers.append(center)
            probabilities.append(probability)
            sizes.append(cluster_size)
            
            # Coordinate-wise statistics
            coord_stats = {}
            for i, coord_name in enumerate(self.coordinate_names[:data.shape[1]]):
                coord_data = cluster_data[:, i]
                coord_stats[coord_name] = {
                    'mean': float(coord_data.mean()),
                    'std': float(coord_data.std()),
                    'min': float(coord_data.min()),
                    'max': float(coord_data.max()),
                    'range': float(coord_data.max() - coord_data.min())
                }
            coordinate_stats.append(coord_stats)
        
        return {
            'centers': np.array(centers),
            'probabilities': np.array(probabilities),
            'sizes': np.array(sizes),
            'coordinate_statistics': coordinate_stats
        }
    
    def _determine_active_clusters(self, cluster_stats: Dict[str, Any]) -> List[int]:
        """Determine which clusters have sufficient data density."""
        probabilities = cluster_stats['probabilities']
        active_clusters = []
        
        for i, prob in enumerate(probabilities):
            if prob >= self.min_cluster_density:
                active_clusters.append(i)
            else:
                print(f"   ⚠️  Cluster {i} has low density ({prob:.3f} < {self.min_cluster_density}), marking as inactive")
        
        if not active_clusters:
            print(f"   ⚠️  No active clusters found, using cluster with highest density")
            active_clusters = [np.argmax(probabilities)]
        
        return active_clusters
    
    def _create_mode_assignments(self) -> Dict[str, Any]:
        """Create quantum mode assignments for coordinate-wise generation."""
        if self.cluster_probabilities is None:
            return {}
        
        # For coordinate-wise assignment:
        # Mode assignment pattern: [X_cluster0, Y_cluster0, X_cluster1, Y_cluster1, ...]
        mode_assignments = {}
        mode_idx = 0
        
        for cluster_idx in range(self.n_clusters):
            cluster_prob = self.cluster_probabilities[cluster_idx]
            is_active = cluster_idx in self.active_clusters
            
            for coord_idx, coord_name in enumerate(self.coordinate_names):
                mode_assignments[f'mode_{mode_idx}'] = {
                    'cluster_id': cluster_idx,
                    'coordinate': coord_name,
                    'coordinate_index': coord_idx,
                    'cluster_probability': float(cluster_prob),
                    'gain': float(cluster_prob) if is_active else 0.0,
                    'active': is_active
                }
                mode_idx += 1
                
                # Stop if we've assigned all available modes
                if mode_idx >= 4:  # Assuming 4 quantum modes
                    break
            
            if mode_idx >= 4:
                break
        
        return mode_assignments
    
    def _print_analysis_summary(self, results: Dict[str, Any]):
        """Print comprehensive analysis summary."""
        print(f"\n📊 CLUSTER ANALYSIS SUMMARY:")
        print(f"   Total samples: {results['n_samples']}")
        print(f"   Features: {results['n_features']}")
        print(f"   Clusters found: {results['n_clusters']}")
        print(f"   Active clusters: {len(results['active_clusters'])}")
        
        print(f"\n🎯 CLUSTER DETAILS:")
        for i in range(self.n_clusters):
            prob = self.cluster_probabilities[i]
            center = self.cluster_centers[i]
            is_active = i in self.active_clusters
            status = "✅ ACTIVE" if is_active else "❌ INACTIVE"
            
            print(f"   Cluster {i}: {status}")
            print(f"     Probability: {prob:.3f}")
            print(f"     Center: ({center[0]:.3f}, {center[1]:.3f})")
            print(f"     Samples: {np.sum(self.cluster_assignments == i)}")
        
        print(f"\n🔧 MODE ASSIGNMENTS:")
        mode_assignments = results['mode_assignments']
        for mode_name, assignment in mode_assignments.items():
            cluster_id = assignment['cluster_id']
            coordinate = assignment['coordinate']
            gain = assignment['gain']
            status = "✅" if assignment['active'] else "❌"
            
            print(f"   {mode_name}: {status} Cluster {cluster_id}, {coordinate} coord, gain={gain:.3f}")
    
    def visualize_clusters(self, data: np.ndarray, save_path: Optional[str] = None):
        """Create comprehensive cluster visualization."""
        if self.cluster_assignments is None:
            print("❌ No cluster analysis performed yet")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('🔍 Cluster Analysis Results', fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Plot 1: Data with cluster assignments
        for i in range(self.n_clusters):
            cluster_mask = self.cluster_assignments == i
            cluster_data = data[cluster_mask]
            is_active = i in self.active_clusters
            
            alpha = 0.7 if is_active else 0.3
            marker = 'o' if is_active else 'x'
            label = f'Cluster {i} (P={self.cluster_probabilities[i]:.3f})'
            
            if len(cluster_data) > 0:
                axes[0].scatter(cluster_data[:, 0], cluster_data[:, 1], 
                               c=colors[i % len(colors)], alpha=alpha, 
                               marker=marker, s=50, label=label)
        
        # Plot cluster centers
        for i, center in enumerate(self.cluster_centers):
            axes[0].scatter(center[0], center[1], c='black', s=200, 
                           marker='*', edgecolors=colors[i % len(colors)], linewidth=2)
        
        axes[0].set_title('Data Clusters')
        axes[0].set_xlabel(self.coordinate_names[0])
        axes[0].set_ylabel(self.coordinate_names[1])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cluster probability distribution
        cluster_indices = range(self.n_clusters)
        bars = axes[1].bar(cluster_indices, self.cluster_probabilities, 
                          color=[colors[i % len(colors)] for i in cluster_indices],
                          alpha=0.7)
        
        # Mark active/inactive clusters
        for i, bar in enumerate(bars):
            if i not in self.active_clusters:
                bar.set_alpha(0.3)
                bar.set_hatch('///')
        
        axes[1].axhline(y=self.min_cluster_density, color='red', linestyle='--', 
                       label=f'Min density: {self.min_cluster_density}')
        axes[1].set_title('Cluster Probability Distribution')
        axes[1].set_xlabel('Cluster Index')
        axes[1].set_ylabel('Probability')
        axes[1].set_xticks(cluster_indices)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🎨 Cluster visualization saved to: {save_path}")
        
        plt.show()
    
    def get_cluster_gains(self) -> np.ndarray:
        """Get cluster probability gains for quantum mode weighting."""
        if self.cluster_probabilities is None:
            raise ValueError("No cluster analysis performed yet")
        
        gains = self.cluster_probabilities.copy()
        
        # Set inactive cluster gains to zero
        for i in range(len(gains)):
            if i not in self.active_clusters:
                gains[i] = 0.0
        
        return gains
    
    def assign_sample_to_cluster(self, sample: np.ndarray) -> int:
        """Assign a single sample to the nearest cluster."""
        if self.cluster_centers is None:
            raise ValueError("No cluster analysis performed yet")
        
        distances = np.linalg.norm(self.cluster_centers - sample, axis=1)
        return np.argmin(distances)
    
    def get_mode_coordinate_mapping(self) -> Dict[int, Dict[str, Any]]:
        """Get mapping from mode index to coordinate and cluster."""
        if not hasattr(self, 'mode_assignments') or not self.mode_assignments:
            return {}
        
        mapping = {}
        for mode_name, assignment in self.mode_assignments.items():
            mode_idx = int(mode_name.split('_')[1])
            mapping[mode_idx] = {
                'cluster_id': assignment['cluster_id'],
                'coordinate': assignment['coordinate'],
                'coordinate_index': assignment['coordinate_index'],
                'gain': assignment['gain'],
                'active': assignment['active']
            }
        
        return mapping


def test_cluster_analyzer():
    """Test the cluster analyzer with bimodal data."""
    print("🧪 Testing ClusterAnalyzer...")
    
    # Create bimodal test data
    np.random.seed(42)
    cluster1 = np.random.normal([-1.5, -1.5], 0.3, (200, 2))
    cluster2 = np.random.normal([1.5, 1.5], 0.3, (300, 2))
    test_data = np.vstack([cluster1, cluster2])
    
    # Test different clustering methods
    methods = ['kmeans', 'gmm', 'grid']
    
    for method in methods:
        print(f"\n🔍 Testing {method} clustering...")
        
        analyzer = ClusterAnalyzer(
            n_clusters=2,
            clustering_method=method,
            coordinate_names=['X', 'Y']
        )
        
        results = analyzer.analyze_data(test_data)
        
        # Visualize results
        analyzer.visualize_clusters(test_data)
        
        # Test mode assignments
        mode_mapping = analyzer.get_mode_coordinate_mapping()
        print(f"Mode coordinate mapping: {mode_mapping}")
        
        # Test cluster gains
        gains = analyzer.get_cluster_gains()
        print(f"Cluster gains: {gains}")
    
    print("✅ ClusterAnalyzer test completed!")


if __name__ == "__main__":
    test_cluster_analyzer()
