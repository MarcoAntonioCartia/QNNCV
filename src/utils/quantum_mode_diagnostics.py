"""
Comprehensive Quantum Mode Diagnostics Framework

This module provides detailed analysis tools to understand quantum mode behavior,
spatial separation, and mode collapse issues in constellation QGANs.

Diagnostic Tools:
1. QuantumModeTracker - Parameter evolution during training
2. MeasurementSpaceAnalyzer - 8D measurement space visualization  
3. ModeContributionDecomposer - Per-mode output analysis
4. ConstellationValidator - Spatial separation verification
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class QuantumModeTracker:
    """
    Track quantum mode parameter evolution during training.
    
    Monitors displacement, squeezing, and other parameters for each mode
    to detect convergence, divergence, and specialization patterns.
    """
    
    def __init__(self, n_modes: int = 4, save_dir: str = "mode_diagnostics"):
        """
        Initialize mode parameter tracker.
        
        Args:
            n_modes: Number of quantum modes to track
            save_dir: Directory to save tracking results
        """
        self.n_modes = n_modes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for parameter evolution
        self.parameter_history = {
            'epochs': [],
            'modes': {i: {
                'displacement_r': [],
                'displacement_phi': [],
                'squeeze_r': [],
                'squeeze_phi': [],
                'base_location_x': [],
                'base_location_y': [],
                'modulated_location_x': [],
                'modulated_location_y': []
            } for i in range(n_modes)}
        }
        
        logger.info(f"QuantumModeTracker initialized for {n_modes} modes")
    
    def track_epoch(self, epoch: int, generator, test_z: tf.Tensor):
        """
        Track mode parameters for current epoch.
        
        Args:
            epoch: Current training epoch
            generator: Quantum generator with parameter access
            test_z: Test latent input for parameter extraction
        """
        print(f"📊 Tracking mode parameters for epoch {epoch}...")
        
        # Generate samples to get parameters
        _ = generator.generate(test_z[:1])  # Use first sample only
        
        if hasattr(generator, 'last_parameters') and generator.last_parameters:
            self.parameter_history['epochs'].append(epoch)
            
            # Extract parameters for each mode
            for mode in range(self.n_modes):
                param_key = f'sample_0_mode_{mode}'
                if param_key in generator.last_parameters:
                    mode_params = generator.last_parameters[param_key]
                    
                    # Store all parameter types
                    self.parameter_history['modes'][mode]['displacement_r'].append(
                        float(mode_params['displacement_r'])
                    )
                    self.parameter_history['modes'][mode]['displacement_phi'].append(
                        float(mode_params['displacement_phi'])
                    )
                    self.parameter_history['modes'][mode]['squeeze_r'].append(
                        float(mode_params['squeeze_r'])
                    )
                    self.parameter_history['modes'][mode]['squeeze_phi'].append(
                        float(mode_params['squeeze_phi'])
                    )
                    
                    base_loc = mode_params['base_location']
                    mod_loc = mode_params['modulated_location']
                    
                    self.parameter_history['modes'][mode]['base_location_x'].append(float(base_loc[0]))
                    self.parameter_history['modes'][mode]['base_location_y'].append(float(base_loc[1]))
                    self.parameter_history['modes'][mode]['modulated_location_x'].append(float(mod_loc[0]))
                    self.parameter_history['modes'][mode]['modulated_location_y'].append(float(mod_loc[1]))
        
        print(f"   ✅ Parameters tracked for {self.n_modes} modes")
    
    def analyze_parameter_evolution(self) -> Dict[str, Any]:
        """
        Analyze parameter evolution patterns.
        
        Returns:
            Analysis results including convergence, diversity metrics
        """
        if not self.parameter_history['epochs']:
            return {'error': 'No parameter history available'}
        
        analysis = {
            'epochs_tracked': len(self.parameter_history['epochs']),
            'mode_diversity': {},
            'parameter_convergence': {},
            'spatial_separation': {}
        }
        
        # Analyze mode diversity
        for param_type in ['displacement_r', 'squeeze_r']:
            final_values = []
            for mode in range(self.n_modes):
                if self.parameter_history['modes'][mode][param_type]:
                    final_values.append(self.parameter_history['modes'][mode][param_type][-1])
            
            if final_values:
                analysis['mode_diversity'][param_type] = {
                    'mean': float(np.mean(final_values)),
                    'std': float(np.std(final_values)),
                    'range': float(np.ptp(final_values)),
                    'coefficient_of_variation': float(np.std(final_values) / np.mean(final_values)) if np.mean(final_values) > 0 else 0
                }
        
        # Analyze spatial separation
        final_x_positions = []
        final_y_positions = []
        for mode in range(self.n_modes):
            if self.parameter_history['modes'][mode]['modulated_location_x']:
                final_x_positions.append(self.parameter_history['modes'][mode]['modulated_location_x'][-1])
                final_y_positions.append(self.parameter_history['modes'][mode]['modulated_location_y'][-1])
        
        if len(final_x_positions) > 1:
            # Calculate pairwise distances
            distances = []
            for i in range(len(final_x_positions)):
                for j in range(i+1, len(final_x_positions)):
                    dist = np.sqrt((final_x_positions[i] - final_x_positions[j])**2 + 
                                 (final_y_positions[i] - final_y_positions[j])**2)
                    distances.append(dist)
            
            analysis['spatial_separation'] = {
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances))
            }
        
        return analysis
    
    def plot_parameter_evolution(self, save_path: Optional[str] = None):
        """
        Create comprehensive parameter evolution plots.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.parameter_history['epochs']:
            print("❌ No parameter history to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔬 Quantum Mode Parameter Evolution Analysis', fontsize=16, fontweight='bold')
        
        epochs = self.parameter_history['epochs']
        colors = ['red', 'blue', 'green', 'orange']
        
        # Plot 1: Displacement magnitude evolution
        for mode in range(self.n_modes):
            disp_r = self.parameter_history['modes'][mode]['displacement_r']
            if disp_r:
                axes[0, 0].plot(epochs, disp_r, color=colors[mode], label=f'Mode {mode}', linewidth=2)
        
        axes[0, 0].set_title('Displacement Magnitude Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Displacement |r|')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Squeeze strength evolution
        for mode in range(self.n_modes):
            squeeze_r = self.parameter_history['modes'][mode]['squeeze_r']
            if squeeze_r:
                axes[0, 1].plot(epochs, squeeze_r, color=colors[mode], label=f'Mode {mode}', linewidth=2)
        
        axes[0, 1].set_title('Squeeze Strength Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Squeeze r')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Spatial positions evolution
        for mode in range(self.n_modes):
            x_pos = self.parameter_history['modes'][mode]['modulated_location_x']
            y_pos = self.parameter_history['modes'][mode]['modulated_location_y']
            if x_pos and y_pos:
                axes[0, 2].plot(x_pos, y_pos, color=colors[mode], label=f'Mode {mode}', 
                               linewidth=2, marker='o', markersize=4)
                # Mark start and end
                axes[0, 2].scatter(x_pos[0], y_pos[0], color=colors[mode], s=100, marker='s', alpha=0.7)
                axes[0, 2].scatter(x_pos[-1], y_pos[-1], color=colors[mode], s=100, marker='*', alpha=0.7)
        
        axes[0, 2].set_title('Spatial Position Trajectories')
        axes[0, 2].set_xlabel('X Position')
        axes[0, 2].set_ylabel('Y Position')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Parameter diversity over time
        diversity_metrics = []
        for epoch_idx in range(len(epochs)):
            epoch_disp_values = []
            for mode in range(self.n_modes):
                if len(self.parameter_history['modes'][mode]['displacement_r']) > epoch_idx:
                    epoch_disp_values.append(self.parameter_history['modes'][mode]['displacement_r'][epoch_idx])
            
            if len(epoch_disp_values) > 1:
                diversity = np.std(epoch_disp_values) / np.mean(epoch_disp_values) if np.mean(epoch_disp_values) > 0 else 0
                diversity_metrics.append(diversity)
            else:
                diversity_metrics.append(0)
        
        axes[1, 0].plot(epochs, diversity_metrics, color='purple', linewidth=2)
        axes[1, 0].set_title('Mode Diversity Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Coefficient of Variation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Pairwise mode distances
        for epoch_idx in range(len(epochs)):
            epoch_distances = []
            epoch_x_pos = []
            epoch_y_pos = []
            
            for mode in range(self.n_modes):
                if (len(self.parameter_history['modes'][mode]['modulated_location_x']) > epoch_idx and
                    len(self.parameter_history['modes'][mode]['modulated_location_y']) > epoch_idx):
                    epoch_x_pos.append(self.parameter_history['modes'][mode]['modulated_location_x'][epoch_idx])
                    epoch_y_pos.append(self.parameter_history['modes'][mode]['modulated_location_y'][epoch_idx])
            
            if len(epoch_x_pos) > 1:
                for i in range(len(epoch_x_pos)):
                    for j in range(i+1, len(epoch_x_pos)):
                        dist = np.sqrt((epoch_x_pos[i] - epoch_x_pos[j])**2 + 
                                     (epoch_y_pos[i] - epoch_y_pos[j])**2)
                        epoch_distances.append(dist)
                
                if epoch_distances:
                    axes[1, 1].scatter([epochs[epoch_idx]] * len(epoch_distances), epoch_distances, 
                                     alpha=0.6, s=30)
        
        axes[1, 1].set_title('Pairwise Mode Distances')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Final parameter distribution
        final_params = {
            'displacement_r': [],
            'squeeze_r': [],
            'x_position': [],
            'y_position': []
        }
        
        for mode in range(self.n_modes):
            if self.parameter_history['modes'][mode]['displacement_r']:
                final_params['displacement_r'].append(self.parameter_history['modes'][mode]['displacement_r'][-1])
            if self.parameter_history['modes'][mode]['squeeze_r']:
                final_params['squeeze_r'].append(self.parameter_history['modes'][mode]['squeeze_r'][-1])
            if self.parameter_history['modes'][mode]['modulated_location_x']:
                final_params['x_position'].append(self.parameter_history['modes'][mode]['modulated_location_x'][-1])
            if self.parameter_history['modes'][mode]['modulated_location_y']:
                final_params['y_position'].append(self.parameter_history['modes'][mode]['modulated_location_y'][-1])
        
        # Create bar plot of final parameters
        x_pos = np.arange(self.n_modes)
        width = 0.35
        
        if final_params['displacement_r']:
            axes[1, 2].bar(x_pos - width/2, final_params['displacement_r'], width, 
                          label='Displacement |r|', alpha=0.7)
        if final_params['squeeze_r']:
            axes[1, 2].bar(x_pos + width/2, final_params['squeeze_r'], width, 
                          label='Squeeze r', alpha=0.7)
        
        axes[1, 2].set_title('Final Parameter Values')
        axes[1, 2].set_xlabel('Mode Index')
        axes[1, 2].set_ylabel('Parameter Value')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Parameter evolution plot saved to: {save_path}")
        
        plt.show()
    
    def save_tracking_data(self, filename: str = "mode_parameter_tracking.json"):
        """Save tracking data to JSON file."""
        save_path = os.path.join(self.save_dir, filename)
        
        with open(save_path, 'w') as f:
            json.dump(self.parameter_history, f, indent=2)
        
        print(f"📊 Mode tracking data saved to: {save_path}")


class MeasurementSpaceAnalyzer:
    """
    Analyze quantum measurement space to detect mode separation.
    
    Visualizes 8D measurement space using dimensionality reduction
    and tracks how different modes contribute to measurements.
    """
    
    def __init__(self, n_modes: int = 4, save_dir: str = "measurement_diagnostics"):
        """
        Initialize measurement space analyzer.
        
        Args:
            n_modes: Number of quantum modes
            save_dir: Directory to save analysis results
        """
        self.n_modes = n_modes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for measurement data
        self.measurement_history = {
            'epochs': [],
            'measurements': [],  # [epoch][sample][measurement_dim]
            'mode_assignments': [],  # [epoch][sample] -> mode_id
            'generated_outputs': []  # [epoch][sample][output_dim]
        }
        
        logger.info(f"MeasurementSpaceAnalyzer initialized for {n_modes} modes")
    
    def analyze_epoch(self, epoch: int, generator, test_z: tf.Tensor, n_samples: int = 100):
        """
        Analyze measurement space for current epoch.
        
        Args:
            epoch: Current training epoch
            generator: Quantum generator
            test_z: Test latent inputs
            n_samples: Number of samples to analyze
        """
        print(f"🔬 Analyzing measurement space for epoch {epoch}...")
        
        # Limit samples for analysis
        analysis_z = test_z[:n_samples]
        
        # Generate samples and collect measurements
        epoch_measurements = []
        epoch_outputs = []
        epoch_mode_assignments = []
        
        for i in range(len(analysis_z)):
            # Generate single sample
            sample_z = analysis_z[i:i+1]
            generated_output = generator.generate(sample_z)
            
            # Extract measurements if available
            if hasattr(generator, 'last_measurements') and generator.last_measurements is not None:
                measurements = generator.last_measurements.numpy()
                epoch_measurements.append(measurements)
                epoch_outputs.append(generated_output[0].numpy())
                
                # Assign to closest mode based on parameters
                if hasattr(generator, 'last_parameters') and generator.last_parameters:
                    # Find mode with highest displacement (simple assignment)
                    max_displacement = -1
                    assigned_mode = 0
                    
                    for mode in range(self.n_modes):
                        param_key = f'sample_0_mode_{mode}'
                        if param_key in generator.last_parameters:
                            disp_r = float(generator.last_parameters[param_key]['displacement_r'])
                            if disp_r > max_displacement:
                                max_displacement = disp_r
                                assigned_mode = mode
                    
                    epoch_mode_assignments.append(assigned_mode)
                else:
                    epoch_mode_assignments.append(0)  # Default assignment
        
        # Store epoch data
        self.measurement_history['epochs'].append(epoch)
        self.measurement_history['measurements'].append(epoch_measurements)
        self.measurement_history['mode_assignments'].append(epoch_mode_assignments)
        self.measurement_history['generated_outputs'].append(epoch_outputs)
        
        print(f"   ✅ Analyzed {len(epoch_measurements)} samples")
    
    def visualize_measurement_space(self, epoch_idx: int = -1, save_path: Optional[str] = None):
        """
        Visualize measurement space using dimensionality reduction.
        
        Args:
            epoch_idx: Index of epoch to visualize (-1 for latest)
            save_path: Optional path to save the plot
        """
        if not self.measurement_history['measurements']:
            print("❌ No measurement data available")
            return
        
        epoch = self.measurement_history['epochs'][epoch_idx]
        measurements = np.array(self.measurement_history['measurements'][epoch_idx])
        mode_assignments = self.measurement_history['mode_assignments'][epoch_idx]
        outputs = np.array(self.measurement_history['generated_outputs'][epoch_idx])
        
        if len(measurements) == 0:
            print("❌ No measurements for selected epoch")
            return
        
        print(f"🔬 Visualizing measurement space for epoch {epoch}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'🔬 Measurement Space Analysis - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange']
        
        # Plot 1: Raw measurement distributions
        for dim in range(min(4, measurements.shape[1])):
            axes[0, 0].hist(measurements[:, dim], bins=20, alpha=0.6, label=f'Dim {dim}')
        
        axes[0, 0].set_title('Raw Measurement Distributions')
        axes[0, 0].set_xlabel('Measurement Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: PCA visualization
        if measurements.shape[1] > 2:
            pca = PCA(n_components=2)
            measurements_pca = pca.fit_transform(measurements)
            
            for mode in range(self.n_modes):
                mode_mask = np.array(mode_assignments) == mode
                if np.any(mode_mask):
                    axes[0, 1].scatter(measurements_pca[mode_mask, 0], measurements_pca[mode_mask, 1],
                                     c=colors[mode], label=f'Mode {mode}', alpha=0.7, s=50)
            
            axes[0, 1].set_title(f'PCA Measurement Space\n(Explained variance: {pca.explained_variance_ratio_.sum():.3f})')
            axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: t-SNE visualization (if enough samples)
        if len(measurements) >= 30 and measurements.shape[1] > 2:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(measurements)//4))
                measurements_tsne = tsne.fit_transform(measurements)
                
                for mode in range(self.n_modes):
                    mode_mask = np.array(mode_assignments) == mode
                    if np.any(mode_mask):
                        axes[0, 2].scatter(measurements_tsne[mode_mask, 0], measurements_tsne[mode_mask, 1],
                                         c=colors[mode], label=f'Mode {mode}', alpha=0.7, s=50)
                
                axes[0, 2].set_title('t-SNE Measurement Space')
                axes[0, 2].set_xlabel('t-SNE 1')
                axes[0, 2].set_ylabel('t-SNE 2')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            except Exception as e:
                axes[0, 2].text(0.5, 0.5, f't-SNE failed:\n{str(e)}', 
                               transform=axes[0, 2].transAxes, ha='center', va='center')
        
        # Plot 4: Mode assignment distribution
        mode_counts = [mode_assignments.count(i) for i in range(self.n_modes)]
        axes[1, 0].bar(range(self.n_modes), mode_counts, color=colors[:self.n_modes])
        axes[1, 0].set_title('Mode Assignment Distribution')
        axes[1, 0].set_xlabel('Mode Index')
        axes[1, 0].set_ylabel('Sample Count')
        axes[1, 0].set_xticks(range(self.n_modes))
        
        # Plot 5: Output space colored by mode
        if outputs.shape[1] >= 2:
            for mode in range(self.n_modes):
                mode_mask = np.array(mode_assignments) == mode
                if np.any(mode_mask):
                    axes[1, 1].scatter(outputs[mode_mask, 0], outputs[mode_mask, 1],
                                     c=colors[mode], label=f'Mode {mode}', alpha=0.7, s=50)
            
            axes[1, 1].set_title('Output Space (Colored by Mode)')
            axes[1, 1].set_xlabel('Output Dim 0')
            axes[1, 1].set_ylabel('Output Dim 1')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Measurement variance per mode
        mode_variances = []
        for mode in range(self.n_modes):
            mode_mask = np.array(mode_assignments) == mode
            if np.any(mode_mask):
                mode_measurements = measurements[mode_mask]
                mode_variance = np.mean(np.var(mode_measurements, axis=0))
                mode_variances.append(mode_variance)
            else:
                mode_variances.append(0)
        
        axes[1, 2].bar(range(self.n_modes), mode_variances, color=colors[:self.n_modes])
        axes[1, 2].set_title('Measurement Variance per Mode')
        axes[1, 2].set_xlabel('Mode Index')
        axes[1, 2].set_ylabel('Average Variance')
        axes[1, 2].set_xticks(range(self.n_modes))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🔬 Measurement space plot saved to: {save_path}")
        
        plt.show()
    
    def calculate_mode_separation_metrics(self, epoch_idx: int = -1) -> Dict[str, Any]:
        """
        Calculate quantitative metrics for mode separation in measurement space.
        
        Args:
            epoch_idx: Index of epoch to analyze (-1 for latest)
            
        Returns:
            Dictionary of separation metrics
        """
        if not self.measurement_history['measurements']:
            return {'error': 'No measurement data available'}
        
        measurements = np.array(self.measurement_history['measurements'][epoch_idx])
        mode_assignments = self.measurement_history['mode_assignments'][epoch_idx]
        
        if len(measurements) == 0:
            return {'error': 'No measurements for selected epoch'}
        
        metrics = {
            'epoch': self.measurement_history['epochs'][epoch_idx],
            'total_samples': len(measurements),
            'measurement_dimension': measurements.shape[1],
            'mode_separation': {},
            'measurement_diversity': {}
        }
        
        # Calculate inter-mode distances in measurement space
        mode_centers = {}
        for mode in range(self.n_modes):
            mode_mask = np.array(mode_assignments) == mode
            if np.any(mode_mask):
                mode_measurements = measurements[mode_mask]
                mode_centers[mode] = np.mean(mode_measurements, axis=0)
        
        # Calculate pairwise distances between mode centers
        if len(mode_centers) > 1:
            distances = []
            mode_pairs = []
            for mode1 in mode_centers:
                for mode2 in mode_centers:
                    if mode1 < mode2:
                        dist = np.linalg.norm(mode_centers[mode1] - mode_centers[mode2])
                        distances.append(dist)
                        mode_pairs.append((mode1, mode2))
            
            metrics['mode_separation'] = {
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'mode_pairs': mode_pairs,
                'distances': distances
            }
        
        # Calculate measurement diversity within each mode
        for mode in range(self.n_modes):
            mode_mask = np.array(mode_assignments) == mode
            if np.any(mode_mask):
                mode_measurements = measurements[mode_mask]
                mode_variance = np.mean(np.var(mode_measurements, axis=0))
                mode_range = np.mean(np.ptp(mode_measurements, axis=0))
                
                metrics['measurement_diversity'][f'mode_{mode}'] = {
                    'sample_count': int(np.sum(mode_mask)),
                    'variance': float(mode_variance),
                    'range': float(mode_range)
                }
        
        return metrics


class ModeContributionDecomposer:
    """
    Decompose final output into individual mode contributions.
    
    Analyzes how each quantum mode contributes to the final generated output
    and identifies which modes are responsible for which data regions.
    """
    
    def __init__(self, n_modes: int = 4, save_dir: str = "contribution_diagnostics"):
        """
        Initialize mode contribution decomposer.
        
        Args:
            n_modes: Number of quantum modes
            save_dir: Directory to save analysis results
        """
        self.n_modes = n_modes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"ModeContributionDecomposer initialized for {n_modes} modes")
    
    def analyze_mode_contributions(self, generator, test_z: tf.Tensor, target_data: tf.Tensor) -> Dict[str, Any]:
        """
        Analyze individual mode contributions to final output.
        
        Args:
            generator: Quantum generator
            test_z: Test latent inputs
            target_data: Target bimodal data for comparison
            
        Returns:
            Analysis results
        """
        print("🔍 Analyzing individual mode contributions...")
        
        # Generate baseline output
        baseline_output = generator.generate(test_z)
        
        # Analyze contribution by disabling modes one at a time
        mode_contributions = {}
        
        for disabled_mode in range(self.n_modes):
            print(f"   Testing with mode {disabled_mode} disabled...")
            
            # Generate with one mode disabled (set to zero displacement)
            modified_outputs = []
            
            for i in range(len(test_z)):
                sample_z = test_z[i:i+1]
                
                # Generate normally first to get parameters
                _ = generator.generate(sample_z)
                
                # Modify parameters to disable one mode
                if hasattr(generator, 'last_parameters') and generator.last_parameters:
                    # Store original parameters
                    original_params = {}
                    for mode in range(self.n_modes):
                        param_key = f'sample_0_mode_{mode}'
                        if param_key in generator.last_parameters:
                            original_params[mode] = generator.last_parameters[param_key].copy()
                    
                    # Disable target mode by setting displacement to zero
                    param_key = f'sample_0_mode_{disabled_mode}'
                    if param_key in generator.last_parameters:
                        generator.last_parameters[param_key]['displacement_r'] = tf.constant(0.0)
                        generator.last_parameters[param_key]['displacement_phi'] = tf.constant(0.0)
                    
                    # Generate with modified parameters
                    modified_output = generator.generate(sample_z)
                    modified_outputs.append(modified_output[0].numpy())
                    
                    # Restore original parameters
                    for mode in range(self.n_modes):
                        param_key = f'sample_0_mode_{mode}'
                        if param_key in generator.last_parameters and mode in original_params:
                            generator.last_parameters[param_key] = original_params[mode]
                else:
                    # Fallback: just use baseline output
                    modified_outputs.append(baseline_output[i].numpy())
            
            # Calculate contribution as difference from baseline
            if modified_outputs:
                modified_outputs = np.array(modified_outputs)
                baseline_np = baseline_output.numpy()
                contribution = baseline_np - modified_outputs
                
                mode_contributions[disabled_mode] = {
                    'contribution_magnitude': float(np.mean(np.linalg.norm(contribution, axis=1))),
                    'contribution_variance': float(np.var(np.linalg.norm(contribution, axis=1))),
                    'x_contribution': float(np.mean(contribution[:, 0])),
                    'y_contribution': float(np.mean(contribution[:, 1])) if contribution.shape[1] > 1 else 0.0,
                    'samples_analyzed': len(modified_outputs)
                }
        
        return {
            'baseline_output_shape': baseline_output.shape.as_list(),
            'mode_contributions': mode_contributions,
            'total_contribution': sum(mode_contributions[mode]['contribution_magnitude'] 
                                    for mode in mode_contributions),
            'analysis_summary': self._summarize_contributions(mode_contributions)
        }
    
    def _summarize_contributions(self, mode_contributions: Dict) -> Dict[str, Any]:
        """Summarize mode contribution analysis."""
        if not mode_contributions:
            return {'error': 'No mode contributions available'}
        
        magnitudes = [mode_contributions[mode]['contribution_magnitude'] 
                     for mode in mode_contributions]
        
        return {
            'most_important_mode': int(max(mode_contributions.keys(), 
                                         key=lambda m: mode_contributions[m]['contribution_magnitude'])),
            'least_important_mode': int(min(mode_contributions.keys(), 
                                          key=lambda m: mode_contributions[m]['contribution_magnitude'])),
            'contribution_balance': float(np.std(magnitudes) / np.mean(magnitudes)) if np.mean(magnitudes) > 0 else 0,
            'total_modes_analyzed': len(mode_contributions)
        }
    
    def visualize_mode_contributions(self, analysis_results: Dict, save_path: Optional[str] = None):
        """
        Visualize mode contribution analysis results.
        
        Args:
            analysis_results: Results from analyze_mode_contributions
            save_path: Optional path to save the plot
        """
        if 'mode_contributions' not in analysis_results:
            print("❌ No mode contribution data to visualize")
            return
        
        mode_contributions = analysis_results['mode_contributions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('🔍 Mode Contribution Analysis', fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange']
        modes = list(mode_contributions.keys())
        
        # Plot 1: Contribution magnitudes
        magnitudes = [mode_contributions[mode]['contribution_magnitude'] for mode in modes]
        axes[0, 0].bar(modes, magnitudes, color=[colors[i] for i in modes])
        axes[0, 0].set_title('Mode Contribution Magnitudes')
        axes[0, 0].set_xlabel('Mode Index')
        axes[0, 0].set_ylabel('Contribution Magnitude')
        axes[0, 0].set_xticks(modes)
        
        # Plot 2: X vs Y contributions
        x_contribs = [mode_contributions[mode]['x_contribution'] for mode in modes]
        y_contribs = [mode_contributions[mode]['y_contribution'] for mode in modes]
        
        for i, mode in enumerate(modes):
            axes[0, 1].scatter(x_contribs[i], y_contribs[i], 
                             c=colors[mode], s=200, label=f'Mode {mode}', alpha=0.7)
        
        axes[0, 1].set_title('X vs Y Contributions')
        axes[0, 1].set_xlabel('X Contribution')
        axes[0, 1].set_ylabel('Y Contribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Contribution variance
        variances = [mode_contributions[mode]['contribution_variance'] for mode in modes]
        axes[1, 0].bar(modes, variances, color=[colors[i] for i in modes])
        axes[1, 0].set_title('Mode Contribution Variance')
        axes[1, 0].set_xlabel('Mode Index')
        axes[1, 0].set_ylabel('Contribution Variance')
        axes[1, 0].set_xticks(modes)
        
        # Plot 4: Summary statistics
        summary = analysis_results.get('analysis_summary', {})
        if summary and 'error' not in summary:
            stats_text = f"""
Most Important Mode: {summary.get('most_important_mode', 'N/A')}
Least Important Mode: {summary.get('least_important_mode', 'N/A')}
Contribution Balance: {summary.get('contribution_balance', 0):.3f}
Total Contribution: {analysis_results.get('total_contribution', 0):.3f}
Modes Analyzed: {summary.get('total_modes_analyzed', 0)}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            axes[1, 1].set_title('Analysis Summary')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🔍 Mode contribution plot saved to: {save_path}")
        
        plt.show()


class ConstellationValidator:
    """
    Validate spatial separation and constellation structure.
    
    Verifies that the 4-quadrant constellation structure is maintained
    and that spatial separation is working as intended.
    """
    
    def __init__(self, expected_separation: float = 2.0, save_dir: str = "constellation_diagnostics"):
        """
        Initialize constellation validator.
        
        Args:
            expected_separation: Expected spatial separation scale
            save_dir: Directory to save validation results
        """
        self.expected_separation = expected_separation
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"ConstellationValidator initialized with separation scale {expected_separation}")
    
    def validate_constellation_structure(self, generator, test_z: tf.Tensor) -> Dict[str, Any]:
        """
        Comprehensive constellation structure validation.
        
        Args:
            generator: Quantum generator with constellation encoding
            test_z: Test latent inputs
            
        Returns:
            Validation results
        """
        print("🔍 Validating constellation structure...")
        
        # Generate samples to get constellation parameters
        _ = generator.generate(test_z[:10])  # Use 10 samples for validation
        
        validation_results = {
            'expected_separation': self.expected_separation,
            'quadrant_structure': {},
            'spatial_separation': {},
            'parameter_consistency': {},
            'validation_passed': False
        }
        
        if not (hasattr(generator, 'last_parameters') and generator.last_parameters):
            validation_results['error'] = 'No constellation parameters available'
            return validation_results
        
        # Analyze quadrant structure
        quadrant_locations = []
        base_locations = []
        modulated_locations = []
        
        for sample_idx in range(min(10, len(test_z))):
            sample_locations = []
            sample_base_locations = []
            sample_modulated_locations = []
            
            for mode in range(4):  # Assuming 4 modes for quadrant structure
                param_key = f'sample_{sample_idx}_mode_{mode}'
                if param_key in generator.last_parameters:
                    mode_params = generator.last_parameters[param_key]
                    base_loc = mode_params['base_location']
                    mod_loc = mode_params['modulated_location']
                    
                    sample_base_locations.append((float(base_loc[0]), float(base_loc[1])))
                    sample_modulated_locations.append((float(mod_loc[0]), float(mod_loc[1])))
            
            if sample_base_locations:
                base_locations.append(sample_base_locations)
                modulated_locations.append(sample_modulated_locations)
        
        # Validate quadrant structure
        if base_locations:
            expected_quadrants = [
                (self.expected_separation, self.expected_separation),    # Q1: (+, +)
                (-self.expected_separation, self.expected_separation),   # Q2: (-, +)
                (-self.expected_separation, -self.expected_separation),  # Q3: (-, -)
                (self.expected_separation, -self.expected_separation)    # Q4: (+, -)
            ]
            
            # Check if base locations match expected quadrants
            base_sample = base_locations[0]  # Use first sample
            quadrant_matches = []
            
            for i, expected_quad in enumerate(expected_quadrants):
                if i < len(base_sample):
                    actual_loc = base_sample[i]
                    distance = np.sqrt((actual_loc[0] - expected_quad[0])**2 + 
                                     (actual_loc[1] - expected_quad[1])**2)
                    quadrant_matches.append(distance < 0.1)  # Tolerance of 0.1
            
            validation_results['quadrant_structure'] = {
                'expected_quadrants': expected_quadrants,
                'actual_base_locations': base_sample,
                'quadrant_matches': quadrant_matches,
                'structure_valid': all(quadrant_matches)
            }
        
        # Validate spatial separation
        if modulated_locations:
            all_separations = []
            for sample_locs in modulated_locations:
                sample_separations = []
                for i in range(len(sample_locs)):
                    for j in range(i+1, len(sample_locs)):
                        loc1, loc2 = sample_locs[i], sample_locs[j]
                        separation = np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
                        sample_separations.append(separation)
                all_separations.extend(sample_separations)
            
            if all_separations:
                validation_results['spatial_separation'] = {
                    'min_separation': float(np.min(all_separations)),
                    'max_separation': float(np.max(all_separations)),
                    'mean_separation': float(np.mean(all_separations)),
                    'std_separation': float(np.std(all_separations)),
                    'adequate_separation': float(np.min(all_separations)) > 1.0
                }
        
        # Overall validation
        structure_valid = validation_results.get('quadrant_structure', {}).get('structure_valid', False)
        separation_adequate = validation_results.get('spatial_separation', {}).get('adequate_separation', False)
        validation_results['validation_passed'] = structure_valid and separation_adequate
        
        return validation_results
    
    def visualize_constellation_validation(self, validation_results: Dict, save_path: Optional[str] = None):
        """
        Visualize constellation validation results.
        
        Args:
            validation_results: Results from validate_constellation_structure
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('🔍 Constellation Structure Validation', fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange']
        
        # Plot 1: Expected vs Actual quadrant structure
        if 'quadrant_structure' in validation_results:
            quad_data = validation_results['quadrant_structure']
            expected = quad_data.get('expected_quadrants', [])
            actual = quad_data.get('actual_base_locations', [])
            
            if expected and actual:
                # Plot expected quadrants
                for i, (x, y) in enumerate(expected):
                    axes[0].scatter(x, y, c='black', s=200, marker='s', alpha=0.5, 
                                  label='Expected' if i == 0 else "")
                
                # Plot actual locations
                for i, (x, y) in enumerate(actual):
                    axes[0].scatter(x, y, c=colors[i], s=150, marker='o', alpha=0.8,
                                  label=f'Mode {i}')
                
                axes[0].set_title('Quadrant Structure Validation')
                axes[0].set_xlabel('X Position')
                axes[0].set_ylabel('Y Position')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 2: Separation statistics
        if 'spatial_separation' in validation_results:
            sep_data = validation_results['spatial_separation']
            
            stats_text = f"""
Minimum Separation: {sep_data.get('min_separation', 0):.3f}
Maximum Separation: {sep_data.get('max_separation', 0):.3f}
Mean Separation: {sep_data.get('mean_separation', 0):.3f}
Std Separation: {sep_data.get('std_separation', 0):.3f}
Adequate Separation: {sep_data.get('adequate_separation', False)}

Expected Separation: {self.expected_separation}
Validation Passed: {validation_results.get('validation_passed', False)}
            """
            
            axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="lightgreen" if validation_results.get('validation_passed', False) else "lightcoral", 
                                alpha=0.7))
            axes[1].set_title('Separation Statistics')
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🔍 Constellation validation plot saved to: {save_path}")
        
        plt.show()


def create_comprehensive_diagnostic_suite(generator, n_modes: int = 4, save_dir: str = "comprehensive_diagnostics"):
    """
    Create a complete diagnostic suite for quantum mode analysis.
    
    Args:
        generator: Quantum generator to analyze
        n_modes: Number of quantum modes
        save_dir: Base directory for saving results
        
    Returns:
        Dictionary containing all diagnostic tools
    """
    os.makedirs(save_dir, exist_ok=True)
    
    diagnostic_suite = {
        'mode_tracker': QuantumModeTracker(n_modes, os.path.join(save_dir, "mode_tracking")),
        'measurement_analyzer': MeasurementSpaceAnalyzer(n_modes, os.path.join(save_dir, "measurement_analysis")),
        'contribution_decomposer': ModeContributionDecomposer(n_modes, os.path.join(save_dir, "contribution_analysis")),
        'constellation_validator': ConstellationValidator(2.0, os.path.join(save_dir, "constellation_validation"))
    }
    
    logger.info(f"Comprehensive diagnostic suite created in {save_dir}")
    return diagnostic_suite


def run_full_diagnostic_analysis(generator, test_z: tf.Tensor, target_data: tf.Tensor, 
                                epoch: int = 0, save_dir: str = "full_diagnostics"):
    """
    Run complete diagnostic analysis on quantum generator.
    
    Args:
        generator: Quantum generator to analyze
        test_z: Test latent inputs
        target_data: Target data for comparison
        epoch: Current epoch number
        save_dir: Directory to save results
        
    Returns:
        Complete diagnostic results
    """
    print(f"🔬 Running FULL DIAGNOSTIC ANALYSIS for epoch {epoch}...")
    
    # Create diagnostic suite
    diagnostics = create_comprehensive_diagnostic_suite(generator, save_dir=save_dir)
    
    # Run all diagnostics
    results = {}
    
    # 1. Track mode parameters
    diagnostics['mode_tracker'].track_epoch(epoch, generator, test_z)
    results['parameter_evolution'] = diagnostics['mode_tracker'].analyze_parameter_evolution()
    
    # 2. Analyze measurement space
    diagnostics['measurement_analyzer'].analyze_epoch(epoch, generator, test_z)
    results['measurement_separation'] = diagnostics['measurement_analyzer'].calculate_mode_separation_metrics()
    
    # 3. Decompose mode contributions
    results['mode_contributions'] = diagnostics['contribution_decomposer'].analyze_mode_contributions(
        generator, test_z, target_data
    )
    
    # 4. Validate constellation structure
    results['constellation_validation'] = diagnostics['constellation_validator'].validate_constellation_structure(
        generator, test_z
    )
    
    # Create comprehensive visualization
    print("🎨 Creating comprehensive diagnostic visualizations...")
    
    # Save individual plots
    diagnostics['mode_tracker'].plot_parameter_evolution(
        os.path.join(save_dir, f"epoch_{epoch:03d}_parameter_evolution.png")
    )
    diagnostics['measurement_analyzer'].visualize_measurement_space(
        save_path=os.path.join(save_dir, f"epoch_{epoch:03d}_measurement_space.png")
    )
    diagnostics['contribution_decomposer'].visualize_mode_contributions(
        results['mode_contributions'],
        save_path=os.path.join(save_dir, f"epoch_{epoch:03d}_mode_contributions.png")
    )
    diagnostics['constellation_validator'].visualize_constellation_validation(
        results['constellation_validation'],
        save_path=os.path.join(save_dir, f"epoch_{epoch:03d}_constellation_validation.png")
    )
    
    # Save complete results
    results_path = os.path.join(save_dir, f"epoch_{epoch:03d}_complete_diagnostics.json")
    
    def convert_to_serializable(obj):
        """Convert numpy/tensorflow types to JSON serializable."""
        if isinstance(obj, (np.integer, np.floating, np.bool_, np.complexfloating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    json_results = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Complete diagnostic analysis saved to {results_path}")
    
    return results, diagnostics
