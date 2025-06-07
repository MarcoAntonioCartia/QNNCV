"""
Environment setup script for QGAN project.
This script helps install dependencies and test the environment.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} is available")
        return True
    except ImportError:
        print(f"✗ {module_name} is not available")
        if package_name:
            print(f"  Try: pip install {package_name}")
        return False

def setup_basic_environment():
    """Set up basic Python environment for QGAN project."""
    
    print("QGAN Project Environment Setup")
    print("==============================")
    
    # Basic packages that should work on most systems
    basic_packages = [
        "numpy",
        "matplotlib", 
        "scikit-learn",
        "pyyaml"
    ]
    
    print("\n1. Installing basic packages...")
    for package in basic_packages:
        install_package(package)
    
    # Try to install TensorFlow
    print("\n2. Installing TensorFlow...")
    tf_success = install_package("tensorflow")
    
    if not tf_success:
        print("Trying TensorFlow CPU version...")
        tf_success = install_package("tensorflow-cpu")
    
    print("\n3. Testing imports...")
    test_results = {}
    test_results['numpy'] = test_import('numpy')
    test_results['matplotlib'] = test_import('matplotlib.pyplot')
    test_results['sklearn'] = test_import('sklearn', 'scikit-learn')
    test_results['yaml'] = test_import('yaml', 'pyyaml')
    test_results['tensorflow'] = test_import('tensorflow')
    
    print("\n4. Environment Status:")
    print("======================")
    
    basic_ready = all([test_results['numpy'], test_results['matplotlib'], 
                      test_results['sklearn'], test_results['yaml']])
    
    if basic_ready and test_results['tensorflow']:
        print("✓ Full environment ready for QGAN development!")
        return "full"
    elif basic_ready:
        print("⚠ Basic environment ready, but TensorFlow missing")
        print("  You can still run some tests and development")
        return "basic"
    else:
        print("✗ Environment setup incomplete")
        print("  Please install missing packages manually")
        return "incomplete"

def test_qgan_components():
    """Test QGAN components if environment is ready."""
    
    print("\n5. Testing QGAN Components...")
    print("=============================")
    
    try:
        # Test basic imports
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_moons
        print("✓ Basic scientific packages working")
        
        # Test TensorFlow if available
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow {tf.__version__} working")
            
            # Test basic TensorFlow operations
            x = tf.constant([1, 2, 3])
            y = tf.constant([4, 5, 6])
            z = x + y
            print("✓ TensorFlow operations working")
            
            # Test if we can create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            print("✓ TensorFlow Keras working")
            
            return True
            
        except ImportError:
            print("✗ TensorFlow not available")
            return False
            
    except ImportError as e:
        print(f"✗ Basic packages not working: {e}")
        return False

def create_test_data():
    """Create test data for QGAN experiments."""
    
    print("\n6. Creating test data...")
    print("========================")
    
    try:
        import numpy as np
        from sklearn.datasets import make_moons, make_circles
        
        # Create data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate and save test datasets
        datasets = {
            'moons': make_moons(n_samples=1000, noise=0.1, random_state=42),
            'circles': make_circles(n_samples=1000, noise=0.1, random_state=42)
        }
        
        for name, (X, y) in datasets.items():
            np.save(os.path.join(data_dir, f"{name}_data.npy"), X)
            print(f"✓ Created {name} dataset: {X.shape}")
        
        print(f"✓ Test data saved in {data_dir}/ directory")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create test data: {e}")
        return False

def main():
    """Main setup function."""
    
    # Setup environment
    env_status = setup_basic_environment()
    
    if env_status in ['full', 'basic']:
        # Test components
        components_ok = test_qgan_components()
        
        if components_ok:
            # Create test data
            create_test_data()
            
            print("\n" + "="*50)
            print("SETUP COMPLETE!")
            print("="*50)
            
            if env_status == 'full':
                print("✓ Full environment ready")
                print("✓ You can run all QGAN experiments")
                print("\nNext steps:")
                print("1. Run: python test_basic.py")
                print("2. Run: python enhanced_test.py")
                print("3. Start developing quantum components")
                
            else:
                print("⚠ Basic environment ready")
                print("⚠ Install TensorFlow for full functionality")
                print("\nNext steps:")
                print("1. Install TensorFlow: pip install tensorflow")
                print("2. Run: python test_basic.py")
        
        else:
            print("\n" + "="*50)
            print("SETUP INCOMPLETE")
            print("="*50)
            print("Please resolve the component issues above")
    
    else:
        print("\n" + "="*50)
        print("SETUP FAILED")
        print("="*50)
        print("Please install the required packages manually")

if __name__ == "__main__":
    main() 