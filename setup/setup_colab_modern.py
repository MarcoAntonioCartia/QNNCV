"""
Modern Google Colab Setup Script for QNNCV
==========================================

This script sets up QNNCV for Google Colab using modern package versions:
- Works with current Colab packages (NumPy 2.0+, SciPy 1.15+, TensorFlow 2.18+)
- Applies comprehensive compatibility patches
- No package downgrades or runtime restarts required
- Includes GPU configuration and optimization
- Provides robust error handling and diagnostics

Usage in Colab:
    !python setup/setup_colab_modern.py

This will apply compatibility patches and validate the installation.
"""

import subprocess
import sys
import os
import warnings
from typing import Dict, List, Tuple, Optional


def print_banner():
    """Print setup banner."""
    print("=" * 70)
    print("QNNCV MODERN GOOGLE COLAB SETUP")
    print("Quantum Neural Networks for Continuous Variables")
    print("Compatible with NumPy 2.0+, SciPy 1.15+, TensorFlow 2.18+")
    print("=" * 70)
    print()


def check_environment() -> Dict[str, any]:
    """Check the current environment and return details."""
    env_info = {}
    
    # Check if running in Colab
    try:
        import google.colab
        env_info['is_colab'] = True
        print("✓ Google Colab environment detected")
    except ImportError:
        env_info['is_colab'] = False
        print("⚠ Not running in Google Colab")
    
    # Get Python version
    env_info['python_version'] = sys.version
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Get platform info
    import platform
    env_info['platform'] = platform.platform()
    print(f"✓ Platform: {platform.platform()}")
    
    return env_info


def get_current_package_versions() -> Dict[str, str]:
    """Get versions of currently installed packages."""
    packages = ['numpy', 'scipy', 'tensorflow', 'matplotlib', 'pandas', 'scikit-learn']
    versions = {}
    
    print("Current package versions:")
    for package in packages:
        try:
            import pkg_resources
            version = pkg_resources.get_distribution(package).version
            versions[package] = version
            print(f"  {package}: {version}")
        except:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                versions[package] = version
                print(f"  {package}: {version}")
            except:
                versions[package] = 'not installed'
                print(f"  {package}: NOT INSTALLED")
    
    return versions


def install_missing_packages() -> bool:
    """Install any missing packages required for QNNCV."""
    print("\nChecking for missing packages...")
    
    # Required packages that might not be in Colab by default
    required_packages = [
        'strawberryfields',
        'pyyaml',
        'tqdm',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        try:
            for package in missing_packages:
                print(f"  Installing {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True, check=True)
                print(f"    ✓ {package} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Package installation failed: {e}")
            print(f"    stdout: {e.stdout}")
            print(f"    stderr: {e.stderr}")
            return False
    else:
        print("  ✓ All required packages already installed")
        return True


def setup_project_paths() -> bool:
    """Set up Python paths for the QNNCV project."""
    print("\nSetting up project paths...")
    
    # Get current working directory (should be QNNCV root)
    project_root = os.getcwd()
    src_path = os.path.join(project_root, 'src')
    
    print(f"  Project root: {project_root}")
    print(f"  Source path: {src_path}")
    
    # Verify project structure
    required_dirs = ['src', 'setup', 'tutorials']
    required_files = ['README.md', 'requirements.txt']
    
    missing_items = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"    ✓ {dir_name}/ directory found")
        else:
            missing_items.append(f"{dir_name}/")
            print(f"    ✗ {dir_name}/ directory missing")
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"    ✓ {file_name} found")
        else:
            missing_items.append(file_name)
            print(f"    ✗ {file_name} missing")
    
    if missing_items:
        print(f"  ⚠ Missing items: {missing_items}")
        print("  Make sure you're running from the QNNCV root directory")
        return False
    
    # Add src to Python path
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"    ✓ Added {src_path} to Python path")
    else:
        print(f"    ✓ {src_path} already in Python path")
    
    return True


def apply_compatibility_patches() -> bool:
    """Apply all compatibility patches using the compatibility module."""
    print("\nApplying compatibility patches...")
    
    try:
        # Import the compatibility module
        from utils.compatibility import apply_all_compatibility_patches
        
        # Apply all patches
        success = apply_all_compatibility_patches()
        
        if success:
            print("✓ All compatibility patches applied successfully")
            return True
        else:
            print("✗ Some compatibility patches failed")
            return False
            
    except ImportError as e:
        print(f"✗ Could not import compatibility module: {e}")
        print("  Make sure you're running from the QNNCV root directory")
        return False
    except Exception as e:
        print(f"✗ Compatibility patch application failed: {e}")
        return False


def test_qnncv_imports() -> Dict[str, bool]:
    """Test that QNNCV modules can be imported successfully."""
    print("\nTesting QNNCV module imports...")
    
    test_imports = [
        ("QuantumSFGenerator", "from models.generators.quantum_sf_generator import QuantumSFGenerator"),
        ("QuantumSFDiscriminator", "from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator"),
        ("QGANSFTrainer", "from training.qgan_sf_trainer import QGANSFTrainer"),
        ("data_utils", "from utils.data_utils import generate_synthetic_data"),
        ("visualization", "from utils.visualization import plot_training_history"),
        ("warning_suppression", "from utils.warning_suppression import enable_clean_training"),
    ]
    
    results = {}
    for component_name, import_code in test_imports:
        try:
            exec(import_code)
            print(f"    ✓ {component_name} import successful")
            results[component_name] = True
        except Exception as e:
            print(f"    ✗ {component_name} import failed: {e}")
            results[component_name] = False
    
    return results


def test_basic_functionality() -> bool:
    """Test basic QNNCV functionality."""
    print("\nTesting basic QNNCV functionality...")
    
    try:
        # Test Strawberry Fields basic operation
        print("  Testing Strawberry Fields...")
        import strawberryfields as sf
        import numpy as np
        
        prog = sf.Program(1)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        
        with prog.context as q:
            sf.ops.Dgate(0.5) | q[0]
        
        result = eng.run(prog)
        print("    ✓ Strawberry Fields basic test successful")
        
        # Test TensorFlow basic operation
        print("  Testing TensorFlow...")
        import tensorflow as tf
        
        # Simple TensorFlow operation
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        print("    ✓ TensorFlow basic test successful")
        
        # Test NumPy with compatibility patches
        print("  Testing NumPy compatibility...")
        import numpy as np
        
        # Test basic operations
        arr = np.array([1, 2, 3])
        result = np.sum(arr)
        
        # Test compatibility patches (if applied)
        if hasattr(np, 'bool'):
            test_bool = np.bool(True)  # type: ignore
            print("    ✓ NumPy compatibility patches working")
        else:
            print("    ✓ NumPy basic operations working")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Basic functionality test failed: {e}")
        return False


def configure_gpu_environment() -> Dict[str, any]:
    """Configure GPU settings and return GPU information."""
    print("\nConfiguring GPU environment...")
    
    gpu_info = {
        'available': False,
        'count': 0,
        'devices': [],
        'configured': False
    }
    
    try:
        import tensorflow as tf
        
        # Get GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info['count'] = len(gpus)
        gpu_info['devices'] = [str(gpu) for gpu in gpus]
        
        if gpus:
            gpu_info['available'] = True
            print(f"    ✓ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu}")
            
            try:
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("    ✓ GPU memory growth enabled")
                
                # Set mixed precision policy for better performance
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("    ✓ Mixed precision policy enabled")
                
                gpu_info['configured'] = True
                
            except RuntimeError as e:
                print(f"    ⚠ GPU configuration warning: {e}")
                print("    GPU detected but configuration failed")
                
        else:
            print("    ⚠ No GPU detected - using CPU")
            print("    Consider switching to GPU runtime: Runtime → Change runtime type → GPU")
        
        return gpu_info
        
    except Exception as e:
        print(f"    ✗ GPU configuration failed: {e}")
        return gpu_info


def provide_usage_instructions(gpu_info: Dict[str, any]):
    """Provide instructions for using QNNCV after setup."""
    print("\n" + "=" * 70)
    print("SETUP COMPLETE - USAGE INSTRUCTIONS")
    print("=" * 70)
    
    print("\n1. Import QNNCV components:")
    print("   from models.generators.quantum_sf_generator import QuantumSFGenerator")
    print("   from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator")
    print("   from training.qgan_sf_trainer import QGANSFTrainer")
    print("   from utils.compatibility import apply_all_compatibility_patches")
    
    print("\n2. Apply compatibility patches (if not auto-applied):")
    print("   apply_all_compatibility_patches()")
    
    print("\n3. Create quantum GAN components:")
    print("   generator = QuantumSFGenerator(n_modes=2, latent_dim=2)")
    print("   discriminator = QuantumSFDiscriminator(n_modes=1, input_dim=2)")
    print("   trainer = QGANSFTrainer(generator, discriminator, latent_dim=2)")
    
    print("\n4. Train your model:")
    print("   history = trainer.train(data, epochs=100, batch_size=16)")
    
    print("\n5. Check the tutorials:")
    print("   - tutorials/minimal_sf_qgan.ipynb")
    print("   - tutorials/extended_sf_qgan_training.ipynb")
    print("   - tutorials/complete_cv_sf_qgan_template.ipynb")
    
    if gpu_info['available']:
        print(f"\n🚀 GPU ACCELERATION ENABLED ({gpu_info['count']} GPU(s))")
        print("   Your quantum GAN training will use GPU acceleration!")
    else:
        print("\n💻 CPU MODE")
        print("   Training will use CPU. For faster training, switch to GPU runtime.")
    
    print("\n" + "=" * 70)


def main() -> bool:
    """Main setup function."""
    print_banner()
    
    # Step 1: Check environment
    print("Step 1: Checking environment...")
    print("-" * 30)
    env_info = check_environment()
    
    # Step 2: Get current package versions
    print("\nStep 2: Checking package versions...")
    print("-" * 35)
    package_versions = get_current_package_versions()
    
    # Step 3: Install missing packages
    print("\nStep 3: Installing missing packages...")
    print("-" * 35)
    if not install_missing_packages():
        print("✗ Package installation failed")
        return False
    
    # Step 4: Setup project paths
    print("\nStep 4: Setting up project paths...")
    print("-" * 35)
    if not setup_project_paths():
        print("✗ Project path setup failed")
        return False
    
    # Step 5: Apply compatibility patches
    print("\nStep 5: Applying compatibility patches...")
    print("-" * 40)
    if not apply_compatibility_patches():
        print("✗ Compatibility patch application failed")
        return False
    
    # Step 6: Configure GPU
    print("\nStep 6: Configuring GPU environment...")
    print("-" * 35)
    gpu_info = configure_gpu_environment()
    
    # Step 7: Test QNNCV imports
    print("\nStep 7: Testing QNNCV imports...")
    print("-" * 30)
    import_results = test_qnncv_imports()
    
    # Step 8: Test basic functionality
    print("\nStep 8: Testing basic functionality...")
    print("-" * 35)
    functionality_test = test_basic_functionality()
    
    # Step 9: Report results
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)
    
    # Check core functionality
    core_components = ['QuantumSFGenerator', 'QuantumSFDiscriminator', 'QGANSFTrainer']
    core_success = all(import_results.get(comp, False) for comp in core_components)
    
    if env_info['is_colab']:
        print("✓ Environment: Google Colab")
    else:
        print("⚠ Environment: Not Google Colab")
    
    if core_success:
        print("✓ QNNCV Core Components: WORKING")
    else:
        print("✗ QNNCV Core Components: FAILED")
    
    if functionality_test:
        print("✓ Basic Functionality: WORKING")
    else:
        print("✗ Basic Functionality: FAILED")
    
    if gpu_info['available'] and gpu_info['configured']:
        print(f"✓ GPU Acceleration: ENABLED ({gpu_info['count']} GPU(s))")
    elif gpu_info['available']:
        print(f"⚠ GPU Acceleration: DETECTED BUT NOT CONFIGURED ({gpu_info['count']} GPU(s))")
    else:
        print("⚠ GPU Acceleration: NOT AVAILABLE")
    
    # Overall success
    overall_success = core_success and functionality_test
    
    if overall_success:
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        provide_usage_instructions(gpu_info)
        return True
    else:
        print("\n❌ SETUP FAILED!")
        print("Check the error messages above for details.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ QNNCV is ready for quantum GAN experiments!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
