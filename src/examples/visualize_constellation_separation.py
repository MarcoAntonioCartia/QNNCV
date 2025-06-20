"""
Constellation Mode Separation Visualization

This script creates publication-quality visualizations of the quantum mode
constellation separation, comparing communication theory optimal spacing
vs random placement.

Creates visualizations like the Mode Separation in 2D Projection plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.quantum.core.multimode_constellation_circuit import MultimodalConstellationCircuit

def create_communication_theory_constellation(n_modes: int, radius: float = 1.5) -> list:
    """Create perfect equally-spaced constellation (communication theory)."""
    constellation_points = []
    
    for i in range(n_modes):
        if n_modes == 1:
            alpha = 0.0 + 0.0j
        else:
            # Perfect equal spacing - no randomness!
            angle = 2 * np.pi * i / n_modes
            alpha = radius * np.exp(1j * angle)
        
        constellation_points.append(alpha)
    
    return constellation_points

def create_random_constellation(n_modes: int, radius: float = 1.5) -> list:
    """Create random constellation (old method with jitter)."""
    constellation_points = []
    
    for i in range(n_modes):
        if n_modes == 1:
            alpha = 0.0 + 0.0j
        else:
            # Random spacing (old method)
            angle = 2 * np.pi * i / n_modes
            angle_jitter = np.random.uniform(-0.2, 0.2)
            radius_jitter = np.random.uniform(0.8, 1.2)
            
            actual_radius = radius * radius_jitter
            actual_angle = angle + angle_jitter
            alpha = actual_radius * np.exp(1j * actual_angle)
        
        constellation_points.append(alpha)
    
    return constellation_points

def calculate_minimum_distance(constellation_points: list) -> float:
    """Calculate minimum distance between constellation points."""
    min_dist = float('inf')
    n_points = len(constellation_points)
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = abs(constellation_points[i] - constellation_points[j])
            min_dist = min(min_dist, dist)
    
    return min_dist

def visualize_mode_separation():
    """Create comprehensive mode separation visualization."""
    
    # Configuration
    n_modes_list = [4, 8, 16]
    radius = 1.5
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum Mode Constellation Separation Analysis\nCommunication Theory vs Random Placement', 
                 fontsize=16, fontweight='bold')
    
    for idx, n_modes in enumerate(n_modes_list):
        
        # Create constellations
        comm_theory_points = create_communication_theory_constellation(n_modes, radius)
        random_points = create_random_constellation(n_modes, radius)
        
        # Calculate metrics
        comm_min_dist = calculate_minimum_distance(comm_theory_points)
        random_min_dist = calculate_minimum_distance(random_points)
        
        # Plot communication theory constellation (top row)
        ax_comm = axes[0, idx]
        
        for i, alpha in enumerate(comm_theory_points):
            x, y = np.real(alpha), np.imag(alpha)
            # Use different colors for each mode
            color = plt.cm.tab10(i % 10)
            ax_comm.scatter(x, y, s=100, c=[color], edgecolors='black', linewidth=1.5, alpha=0.8)
            ax_comm.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', ha='center')
        
        # Draw constellation circle
        circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                          color='gray', alpha=0.5)
        ax_comm.add_patch(circle)
        
        ax_comm.set_xlim(-2.5, 2.5)
        ax_comm.set_ylim(-2.5, 2.5)
        ax_comm.set_aspect('equal')
        ax_comm.grid(True, alpha=0.3)
        ax_comm.set_title(f'Communication Theory\n{n_modes} Modes - Perfect Spacing\nMin Distance: {comm_min_dist:.3f}',
                         fontweight='bold')
        ax_comm.set_xlabel('Real Part (X)')
        ax_comm.set_ylabel('Imaginary Part (P)')
        
        # Plot random constellation (bottom row)
        ax_random = axes[1, idx]
        
        for i, alpha in enumerate(random_points):
            x, y = np.real(alpha), np.imag(alpha)
            color = plt.cm.tab10(i % 10)
            ax_random.scatter(x, y, s=100, c=[color], edgecolors='black', linewidth=1.5, alpha=0.8)
            ax_random.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                             fontsize=10, fontweight='bold', ha='center')
        
        # Draw constellation circle
        circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                          color='gray', alpha=0.5)
        ax_random.add_patch(circle)
        
        ax_random.set_xlim(-2.5, 2.5)
        ax_random.set_ylim(-2.5, 2.5)
        ax_random.set_aspect('equal')
        ax_random.grid(True, alpha=0.3)
        ax_random.set_title(f'Random Placement\n{n_modes} Modes - With Jitter\nMin Distance: {random_min_dist:.3f}',
                           fontweight='bold')
        ax_random.set_xlabel('Real Part (X)')
        ax_random.set_ylabel('Imaginary Part (P)')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = "20250620_constellation_comparison"
    filename = f"constellation_mode_separation_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Constellation comparison saved: {filename}")
    
    plt.show()
    
    return filename

def analyze_constellation_optimality():
    """Analyze the optimality of different constellation configurations."""
    
    print("\n" + "🌟" * 60)
    print("CONSTELLATION OPTIMALITY ANALYSIS")
    print("🌟" * 60)
    
    results = []
    
    for n_modes in [2, 4, 6, 8, 12, 16]:
        
        # Communication theory constellation
        comm_points = create_communication_theory_constellation(n_modes, radius=1.5)
        comm_min_dist = calculate_minimum_distance(comm_points)
        
        # Average over multiple random trials
        random_min_dists = []
        for trial in range(10):
            random_points = create_random_constellation(n_modes, radius=1.5)
            random_min_dist = calculate_minimum_distance(random_points)
            random_min_dists.append(random_min_dist)
        
        avg_random_min_dist = np.mean(random_min_dists)
        improvement_factor = comm_min_dist / avg_random_min_dist
        
        results.append({
            'n_modes': n_modes,
            'comm_min_dist': comm_min_dist,
            'random_min_dist': avg_random_min_dist,
            'improvement': improvement_factor,
            'angular_separation': 360 / n_modes
        })
        
        print(f"\n{n_modes:2d} Modes:")
        print(f"   Communication Theory Min Distance: {comm_min_dist:.4f}")
        print(f"   Random Average Min Distance:      {avg_random_min_dist:.4f}")
        print(f"   Improvement Factor:                {improvement_factor:.2f}x")
        print(f"   Perfect Angular Separation:        {360/n_modes:.1f}°")
    
    print(f"\n🎯 COMMUNICATION THEORY ADVANTAGES:")
    print(f"   • Perfect equal spacing maximizes minimum distance")
    print(f"   • Optimal orthogonality between quantum modes")
    print(f"   • Predictable and reproducible separation")
    print(f"   • No random variations affecting performance")
    
    return results

def test_quantum_circuit_with_perfect_constellation():
    """Test the quantum circuit with perfect constellation."""
    
    print("\n" + "🔬" * 60)
    print("TESTING QUANTUM CIRCUIT WITH PERFECT CONSTELLATION")
    print("🔬" * 60)
    
    # Create circuit with perfect constellation
    circuit = MultimodalConstellationCircuit(
        n_modes=8,
        n_layers=3,
        cutoff_dim=6,
        constellation_radius=2.0,
        enable_correlations=True
    )
    
    # Get constellation info
    constellation_info = circuit.get_constellation_info()
    
    print(f"\n🌟 Perfect Constellation Configuration:")
    print(f"   Modes: {constellation_info['n_modes']}")
    print(f"   Radius: {constellation_info['constellation_radius']}")
    print(f"   Angular separation: {360/constellation_info['n_modes']:.1f}°")
    
    print(f"\n📍 Constellation Points:")
    for i, point in enumerate(constellation_info['constellation_points']):
        alpha = point['alpha']
        magnitude = point['magnitude']
        phase_deg = np.degrees(point['phase'])
        print(f"   Mode {i}: α = {alpha:.3f} (|α|={magnitude:.3f}, ∠α={phase_deg:.1f}°)")
    
    # Test multimode performance
    print(f"\n🧪 Testing Multimode Performance...")
    
    # Generate test batch
    test_encodings = tf.random.normal([12, 16])
    quantum_states = circuit.execute_batch(test_encodings)
    measurements = circuit.extract_batch_measurements(quantum_states)
    
    print(f"   Batch measurements shape: {measurements.shape}")
    
    # Analyze mode utilization
    measurements_per_mode = 2
    mode_variances = []
    
    for mode in range(circuit.n_modes):
        start_idx = mode * measurements_per_mode
        end_idx = start_idx + measurements_per_mode
        mode_measurements = measurements[:, start_idx:end_idx]
        mode_variance = float(tf.math.reduce_variance(mode_measurements))
        mode_variances.append(mode_variance)
    
    active_modes = sum(1 for v in mode_variances if v > 0.001)
    multimode_utilization = active_modes / circuit.n_modes
    total_variance = sum(mode_variances)
    
    print(f"\n📊 Multimode Performance Results:")
    print(f"   Active modes: {active_modes}/{circuit.n_modes}")
    print(f"   Multimode utilization: {multimode_utilization:.1%}")
    print(f"   Total variance: {total_variance:.4f}")
    print(f"   Mode variances: {[f'{v:.4f}' for v in mode_variances]}")
    
    if multimode_utilization >= 0.75:  # 75% or better
        print(f"   🎉 EXCELLENT: High multimode utilization achieved!")
    elif multimode_utilization >= 0.5:  # 50% or better
        print(f"   ✅ GOOD: Significant multimode utilization")
    else:
        print(f"   ⚠️  MODERATE: Some modes still inactive")
    
    return {
        'constellation_info': constellation_info,
        'multimode_utilization': multimode_utilization,
        'active_modes': active_modes,
        'total_variance': total_variance,
        'mode_variances': mode_variances
    }

def main():
    """Run complete constellation analysis and visualization."""
    
    print("🌟" * 80)
    print("QUANTUM CONSTELLATION COMMUNICATION THEORY ANALYSIS")
    print("🌟" * 80)
    
    # 1. Create visualization
    print("\n1️⃣ Creating Mode Separation Visualizations...")
    visualization_file = visualize_mode_separation()
    
    # 2. Analyze optimality
    print("\n2️⃣ Analyzing Constellation Optimality...")
    optimality_results = analyze_constellation_optimality()
    
    # 3. Test quantum circuit
    print("\n3️⃣ Testing Quantum Circuit Performance...")
    circuit_results = test_quantum_circuit_with_perfect_constellation()
    
    # 4. Summary
    print("\n" + "🎯" * 80)
    print("COMMUNICATION THEORY CONSTELLATION SUMMARY")
    print("🎯" * 80)
    
    print(f"\n📊 Key Results:")
    print(f"   • Perfect equal spacing achieved (no randomness)")
    print(f"   • Multimode utilization: {circuit_results['multimode_utilization']:.1%}")
    print(f"   • Active quantum modes: {circuit_results['active_modes']}/{circuit_results['constellation_info']['n_modes']}")
    print(f"   • Visualization saved: {visualization_file}")
    
    print(f"\n🏆 Communication Theory Advantages:")
    print(f"   • Maximum minimum distance between modes")
    print(f"   • Optimal orthogonality (like QPSK, 16-QAM)")
    print(f"   • Predictable and reproducible performance")
    print(f"   • No random variations affecting training")
    
    print(f"\n🚀 Ready for quantum GAN integration!")
    
    return {
        'visualization_file': visualization_file,
        'optimality_results': optimality_results,
        'circuit_results': circuit_results
    }

if __name__ == "__main__":
    results = main()
    print(f"\n✅ Analysis complete! Check '{results['visualization_file']}' for mode separation plot.")
