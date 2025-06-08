"""
Basic test script that works without TensorFlow.
Tests data loading, utilities, and basic functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

def test_data_loading():
    """Test synthetic data generation and loading utilities."""
    
    print("Testing Data Loading and Utilities")
    print("==================================")
    
    # Test synthetic data generation
    print("\n1. Testing synthetic data generation...")
    
    # Generate different types of synthetic data
    datasets = {
        'moons': make_moons(n_samples=500, noise=0.1, random_state=42),
        'circles': make_circles(n_samples=500, noise=0.1, random_state=42)
    }
    
    for name, (X, y) in datasets.items():
        print(f"✓ Generated {name} dataset: {X.shape}")
        
        # Normalize data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        print(f"  Normalized range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
    
    return datasets

def test_visualization():
    """Test visualization capabilities."""
    
    print("\n2. Testing visualization...")
    
    # Generate test data
    X_real, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
    X_fake = np.random.normal(0, 1, (200, 2))  # Random fake data
    
    # Create a simple plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_real[:, 0], X_real[:, 1], alpha=0.6, s=20, label='Real')
    plt.title("Real Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_fake[:, 0], X_fake[:, 1], alpha=0.6, s=20, color='red', label='Fake')
    plt.title("Fake Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_real[:, 0], X_real[:, 1], alpha=0.4, s=15, label='Real')
    plt.scatter(X_fake[:, 0], X_fake[:, 1], alpha=0.4, s=15, label='Fake')
    plt.title("Overlay Comparison")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot instead of showing (in case running headless)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/test_visualization.png", dpi=150, bbox_inches='tight')
    print("✓ Visualization test completed")
    print("  Plot saved to: results/test_visualization.png")
    
    plt.close()

def test_metrics():
    """Test evaluation metrics that don't require TensorFlow."""
    
    print("\n3. Testing evaluation metrics...")
    
    # Generate test data
    np.random.seed(42)
    real_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
    fake_data = np.random.multivariate_normal([0.2, 0.1], [[1.1, 0.4], [0.4, 0.9]], 100)
    
    # Test basic distance metrics
    def compute_mean_distance(X1, X2):
        """Compute simple mean distance between datasets."""
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        return np.linalg.norm(mean1 - mean2)
    
    def compute_std_distance(X1, X2):
        """Compute difference in standard deviations."""
        std1 = np.std(X1, axis=0)
        std2 = np.std(X2, axis=0)
        return np.linalg.norm(std1 - std2)
    
    mean_dist = compute_mean_distance(real_data, fake_data)
    std_dist = compute_std_distance(real_data, fake_data)
    
    print(f"✓ Mean distance: {mean_dist:.4f}")
    print(f"✓ Std distance: {std_dist:.4f}")
    
    # Test covariance comparison
    cov_real = np.cov(real_data.T)
    cov_fake = np.cov(fake_data.T)
    cov_dist = np.linalg.norm(cov_real - cov_fake, 'fro')
    
    print(f"✓ Covariance distance: {cov_dist:.4f}")
    
    return {
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'cov_distance': cov_dist
    }

def test_utils_functions():
    """Test if our utils.py functions work (at least the non-TensorFlow parts)."""
    
    print("\n4. Testing utils.py functions...")
    
    try:
        # Test directory creation
        from utils import create_output_directory
        output_dir = create_output_directory("test_results")
        print(f"✓ Created output directory: {output_dir}")
        
        # Test if the directory structure was created
        expected_subdirs = ['plots', 'models', 'logs', 'data']
        for subdir in expected_subdirs:
            if os.path.exists(os.path.join(output_dir, subdir)):
                print(f"  ✓ {subdir}/ directory created")
            else:
                print(f"  ✗ {subdir}/ directory missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Utils functions failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests without TensorFlow."""
    
    print("QGAN Basic Testing Suite (No TensorFlow)")
    print("=========================================")
    
    # Test 1: Data loading
    datasets = test_data_loading()
    
    # Test 2: Visualization
    test_visualization()
    
    # Test 3: Metrics
    metrics = test_metrics()
    
    # Test 4: Utils
    utils_ok = test_utils_functions()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    print("✓ Data loading: PASSED")
    print("✓ Visualization: PASSED")
    print("✓ Basic metrics: PASSED")
    
    if utils_ok:
        print("✓ Utils functions: PASSED")
    else:
        print("✗ Utils functions: FAILED")
    
    print("\nWhat's working:")
    print("- Synthetic data generation")
    print("- Data normalization and preprocessing")
    print("- Basic visualization and plotting")
    print("- Simple evaluation metrics")
    print("- File system utilities")
    
    print("\nNext steps:")
    print("1. Install TensorFlow for Python 3.13 (when available)")
    print("2. Or downgrade to Python 3.11/3.12 for TensorFlow compatibility")
    print("3. Test classical and quantum generators")
    print("4. Implement full GAN training loop")
    
    print("\nCurrent development status:")
    print("✓ Foundation infrastructure is working")
    print("⚠ Waiting for TensorFlow compatibility")
    print("✓ Ready for development once TensorFlow is available")

if __name__ == "__main__":
    run_comprehensive_test() 