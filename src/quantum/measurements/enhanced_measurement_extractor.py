"""
Enhanced Quantum Measurement Extractor

Extracts comprehensive quantum measurements with separation of:
- X-quadrature measurements (for decoder input)
- P-quadrature measurements (for visualization)
- Photon number statistics (for visualization)
- Provides probability distribution analysis
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import strawberryfields as sf
from strawberryfields.ops import *


class QuantumMeasurementSeparator:
    """
    Separates quantum measurements into categories for different uses.
    
    X-quadrature: Used for decoder input (position-like measurements)
    P-quadrature: Used for visualization (momentum-like measurements) 
    Photon numbers: Used for visualization (occupation statistics)
    """
    
    def __init__(self, n_modes: int, cutoff_dim: int):
        """
        Initialize measurement separator.
        
        Args:
            n_modes: Number of quantum modes
            cutoff_dim: Photon number cutoff dimension
        """
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        
        print(f"🔬 QuantumMeasurementSeparator initialized:")
        print(f"   Modes: {n_modes}")
        print(f"   Cutoff dimension: {cutoff_dim}")
        print(f"   X-quadrature measurements: {n_modes} (for decoder)")
        print(f"   P-quadrature measurements: {n_modes} (for visualization)")
        print(f"   Photon measurements: {n_modes} (for visualization)")
    
    def extract_all_measurements(self, quantum_state) -> Dict[str, np.ndarray]:
        """
        Extract all quantum measurements from state.
        
        Args:
            quantum_state: Strawberry Fields quantum state
            
        Returns:
            Dictionary with separated measurements:
            - 'x_quadrature': X measurements (for decoder)
            - 'p_quadrature': P measurements (for visualization)
            - 'photon_numbers': Photon statistics (for visualization)
            - 'combined_decoder_input': Only X-quadrature (decoder input)
        """
        measurements = {}
        
        # Extract X-quadrature measurements (position-like)
        x_measurements = []
        for mode in range(self.n_modes):
            x_quad = quantum_state.quad_expectation(mode, 0)  # X quadrature
            # Handle complex numbers or tuples by taking real part
            if isinstance(x_quad, (tuple, list)):
                x_val = float(x_quad[0]) if len(x_quad) > 0 else 0.0
            elif hasattr(x_quad, 'real'):
                x_val = float(x_quad.real)
            else:
                x_val = float(x_quad)
            x_measurements.append(x_val)
        measurements['x_quadrature'] = np.array(x_measurements)
        
        # Extract P-quadrature measurements (momentum-like)
        p_measurements = []
        for mode in range(self.n_modes):
            p_quad = quantum_state.quad_expectation(mode, np.pi/2)  # P quadrature
            # Handle complex numbers or tuples by taking real part
            if isinstance(p_quad, (tuple, list)):
                p_val = float(p_quad[0]) if len(p_quad) > 0 else 0.0
            elif hasattr(p_quad, 'real'):
                p_val = float(p_quad.real)
            else:
                p_val = float(p_quad)
            p_measurements.append(p_val)
        measurements['p_quadrature'] = np.array(p_measurements)
        
        # Extract photon number statistics
        photon_measurements = []
        for mode in range(self.n_modes):
            photon_expectation = quantum_state.mean_photon(mode)
            # Handle complex numbers or tuples by taking real part
            if isinstance(photon_expectation, (tuple, list)):
                photon_val = float(photon_expectation[0]) if len(photon_expectation) > 0 else 0.0
            elif hasattr(photon_expectation, 'real'):
                photon_val = float(photon_expectation.real)
            else:
                photon_val = float(photon_expectation)
            photon_measurements.append(photon_val)
        measurements['photon_numbers'] = np.array(photon_measurements)
        
        # Decoder input: ONLY X-quadrature
        measurements['decoder_input'] = measurements['x_quadrature']
        
        # Store all measurements for visualization
        measurements['all_measurements'] = np.concatenate([
            measurements['x_quadrature'],
            measurements['p_quadrature'], 
            measurements['photon_numbers']
        ])
        
        return measurements
    
    def extract_batch_measurements(self, quantum_states: List) -> Dict[str, np.ndarray]:
        """
        Extract measurements from a batch of quantum states.
        
        Args:
            quantum_states: List of quantum states
            
        Returns:
            Dictionary with batched measurements
        """
        batch_measurements = {
            'x_quadrature': [],
            'p_quadrature': [],
            'photon_numbers': [],
            'decoder_input': [],
            'all_measurements': []
        }
        
        for state in quantum_states:
            measurements = self.extract_all_measurements(state)
            
            for key in batch_measurements.keys():
                batch_measurements[key].append(measurements[key])
        
        # Convert to numpy arrays
        for key in batch_measurements.keys():
            batch_measurements[key] = np.array(batch_measurements[key])
        
        return batch_measurements


class QuantumMeasurementDistributionAnalyzer:
    """
    Analyzes probability distributions of quantum measurements.
    
    Provides P(x), P(p), P(n) distributions for quantum mechanical insight.
    """
    
    def __init__(self, n_modes: int, n_bins: int = 50):
        """
        Initialize distribution analyzer.
        
        Args:
            n_modes: Number of quantum modes
            n_bins: Number of bins for histogram analysis
        """
        self.n_modes = n_modes
        self.n_bins = n_bins
        
        # Storage for measurement history
        self.measurement_history = {
            'x_quadrature': [[] for _ in range(n_modes)],
            'p_quadrature': [[] for _ in range(n_modes)],
            'photon_numbers': [[] for _ in range(n_modes)]
        }
        
        print(f"📊 QuantumMeasurementDistributionAnalyzer initialized:")
        print(f"   Modes: {n_modes}")
        print(f"   Distribution bins: {n_bins}")
    
    def update_measurement_history(self, measurements: Dict[str, np.ndarray]):
        """
        Update measurement history with new measurements.
        
        Args:
            measurements: Dictionary of measurements from QuantumMeasurementSeparator
        """
        # Update X-quadrature history
        for mode in range(self.n_modes):
            if len(measurements['x_quadrature'].shape) == 1:
                # Single sample
                self.measurement_history['x_quadrature'][mode].append(
                    measurements['x_quadrature'][mode]
                )
            else:
                # Batch of samples
                self.measurement_history['x_quadrature'][mode].extend(
                    measurements['x_quadrature'][:, mode]
                )
        
        # Update P-quadrature history
        for mode in range(self.n_modes):
            if len(measurements['p_quadrature'].shape) == 1:
                self.measurement_history['p_quadrature'][mode].append(
                    measurements['p_quadrature'][mode]
                )
            else:
                self.measurement_history['p_quadrature'][mode].extend(
                    measurements['p_quadrature'][:, mode]
                )
        
        # Update photon number history
        for mode in range(self.n_modes):
            if len(measurements['photon_numbers'].shape) == 1:
                self.measurement_history['photon_numbers'][mode].append(
                    measurements['photon_numbers'][mode]
                )
            else:
                self.measurement_history['photon_numbers'][mode].extend(
                    measurements['photon_numbers'][:, mode]
                )
    
    def compute_probability_distributions(self) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute probability distributions P(x), P(p), P(n) for each mode.
        
        Returns:
            Dictionary with distributions:
            - 'x_distributions': List of (bins, probabilities) for each mode
            - 'p_distributions': List of (bins, probabilities) for each mode  
            - 'photon_distributions': List of (bins, probabilities) for each mode
        """
        distributions = {
            'x_distributions': [],
            'p_distributions': [],
            'photon_distributions': []
        }
        
        # Compute X-quadrature distributions P(x)
        for mode in range(self.n_modes):
            if len(self.measurement_history['x_quadrature'][mode]) > 1:
                values = np.array(self.measurement_history['x_quadrature'][mode])
                counts, bins = np.histogram(values, bins=self.n_bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                distributions['x_distributions'].append((bin_centers, counts))
            else:
                # Not enough data
                distributions['x_distributions'].append((np.array([0]), np.array([1])))
        
        # Compute P-quadrature distributions P(p)
        for mode in range(self.n_modes):
            if len(self.measurement_history['p_quadrature'][mode]) > 1:
                values = np.array(self.measurement_history['p_quadrature'][mode])
                counts, bins = np.histogram(values, bins=self.n_bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                distributions['p_distributions'].append((bin_centers, counts))
            else:
                distributions['p_distributions'].append((np.array([0]), np.array([1])))
        
        # Compute photon number distributions P(n)
        for mode in range(self.n_modes):
            if len(self.measurement_history['photon_numbers'][mode]) > 1:
                values = np.array(self.measurement_history['photon_numbers'][mode])
                counts, bins = np.histogram(values, bins=self.n_bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                distributions['photon_distributions'].append((bin_centers, counts))
            else:
                distributions['photon_distributions'].append((np.array([0]), np.array([1])))
        
        return distributions
    
    def get_distribution_statistics(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get statistical summary of distributions.
        
        Returns:
            Dictionary with statistics for each measurement type and mode
        """
        stats = {
            'x_quadrature': {'mean': [], 'std': [], 'range': []},
            'p_quadrature': {'mean': [], 'std': [], 'range': []},
            'photon_numbers': {'mean': [], 'std': [], 'range': []}
        }
        
        for measurement_type in ['x_quadrature', 'p_quadrature', 'photon_numbers']:
            for mode in range(self.n_modes):
                values = self.measurement_history[measurement_type][mode]
                if len(values) > 0:
                    values_array = np.array(values)
                    stats[measurement_type]['mean'].append(float(np.mean(values_array)))
                    stats[measurement_type]['std'].append(float(np.std(values_array)))
                    stats[measurement_type]['range'].append([float(np.min(values_array)), float(np.max(values_array))])
                else:
                    stats[measurement_type]['mean'].append(0.0)
                    stats[measurement_type]['std'].append(0.0)
                    stats[measurement_type]['range'].append([0.0, 0.0])
        
        return stats


class QuantumMeasurementVisualizer:
    """
    Creates visualization plots for quantum measurement probability distributions.
    """
    
    def __init__(self, n_modes: int):
        """
        Initialize visualizer.
        
        Args:
            n_modes: Number of quantum modes
        """
        self.n_modes = n_modes
        
        print(f"📈 QuantumMeasurementVisualizer initialized for {n_modes} modes")
    
    def plot_probability_distributions(self, 
                                     distributions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot probability distributions P(x), P(p), P(n) for all modes.
        
        Args:
            distributions: Output from QuantumMeasurementDistributionAnalyzer
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig, axes = plt.subplots(3, self.n_modes, figsize=(4*self.n_modes, 12))
        if self.n_modes == 1:
            axes = axes.reshape(3, 1)
        
        fig.suptitle('Quantum Measurement Probability Distributions', fontsize=16, fontweight='bold')
        
        # Plot X-quadrature distributions P(x)
        for mode in range(self.n_modes):
            ax = axes[0, mode]
            bins, probs = distributions['x_distributions'][mode]
            ax.plot(bins, probs, 'b-', linewidth=2, label=f'P(X_{mode})')
            ax.fill_between(bins, probs, alpha=0.3, color='blue')
            ax.set_title(f'Mode {mode}: X-Quadrature\n(Used for Decoder)', fontweight='bold')
            ax.set_xlabel('X (Position)')
            ax.set_ylabel('Probability Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot P-quadrature distributions P(p)
        for mode in range(self.n_modes):
            ax = axes[1, mode]
            bins, probs = distributions['p_distributions'][mode]
            ax.plot(bins, probs, 'r-', linewidth=2, label=f'P(P_{mode})')
            ax.fill_between(bins, probs, alpha=0.3, color='red')
            ax.set_title(f'Mode {mode}: P-Quadrature\n(Visualization Only)', fontweight='bold')
            ax.set_xlabel('P (Momentum)')
            ax.set_ylabel('Probability Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot photon number distributions P(n)
        for mode in range(self.n_modes):
            ax = axes[2, mode]
            bins, probs = distributions['photon_distributions'][mode]
            ax.plot(bins, probs, 'g-', linewidth=2, label=f'P(n_{mode})')
            ax.fill_between(bins, probs, alpha=0.3, color='green')
            ax.set_title(f'Mode {mode}: Photon Number\n(Visualization Only)', fontweight='bold')
            ax.set_xlabel('Photon Number')
            ax.set_ylabel('Probability Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Probability distributions saved to: {save_path}")
        
        return fig
    
    def plot_decoder_input_evolution(self, 
                                   x_quadrature_history: List[np.ndarray],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot evolution of X-quadrature measurements (decoder inputs) over time.
        
        Args:
            x_quadrature_history: List of X-quadrature measurements over time
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('X-Quadrature Evolution (Decoder Inputs)', fontsize=16, fontweight='bold')
        
        epochs = range(len(x_quadrature_history))
        
        for mode in range(min(self.n_modes, 4)):  # Plot up to 4 modes
            ax = axes[mode // 2, mode % 2]
            
            # Extract values for this mode
            mode_values = [measurements[mode] if len(measurements) > mode else 0 
                          for measurements in x_quadrature_history]
            
            ax.plot(epochs, mode_values, 'o-', linewidth=2, markersize=6, 
                   label=f'X_{mode} (→ Decoder)')
            ax.set_title(f'Mode {mode}: X-Quadrature → Decoder Input', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('X-Quadrature Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Decoder input evolution saved to: {save_path}")
        
        return fig


def create_enhanced_measurement_system(n_modes: int, cutoff_dim: int) -> Tuple[QuantumMeasurementSeparator, QuantumMeasurementDistributionAnalyzer, QuantumMeasurementVisualizer]:
    """
    Factory function to create complete enhanced measurement system.
    
    Args:
        n_modes: Number of quantum modes
        cutoff_dim: Photon number cutoff dimension
        
    Returns:
        Tuple of (separator, analyzer, visualizer)
    """
    separator = QuantumMeasurementSeparator(n_modes, cutoff_dim)
    analyzer = QuantumMeasurementDistributionAnalyzer(n_modes)
    visualizer = QuantumMeasurementVisualizer(n_modes)
    
    print(f"🔬 Enhanced measurement system created:")
    print(f"   • Separator: Extracts X, P, photon measurements")
    print(f"   • Analyzer: Computes P(x), P(p), P(n) distributions")
    print(f"   • Visualizer: Creates probability distribution plots")
    
    return separator, analyzer, visualizer
