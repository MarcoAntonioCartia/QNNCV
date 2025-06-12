"""
Local Development Setup Script for QNNCV
========================================

This script sets up QNNCV for local development, matching Colab environment:
- Creates environment with Python 3.11
- Installs packages matching Colab versions
- Applies compatibility patches for NumPy 2.0+ and SciPy 1.15+
- Tests all QNNCV components locally
- Provides CPU fallback for development without GPU

Usage:
    python setup/setup_local.py

This will install dependencies and validate the local environment.
"""

import subprocess
import sys
import os
import warnings
from typing import Dict, List, Tuple, Optional


def print_banner():
    """Print setup banner."""
    print("=" * 70)
    print("QNNCV LOCAL DEVELOPMENT SETUP")
    print("Quantum Neural Networks for Continuous Variables")
    print("Matching Google Colab Environment (NumPy 2.0+, SciPy 1.15+)")
    print("=" * 70)
    print()


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print(f"  Current Python version: {python_version}")
    
    # Check if Python 3.8+
    if version_info.major == 3 and version_info.minor >= 8:
        print("  ✓ Python version is compatible")
        return True
    else:
        print("  ✗ Python 3.8+ required")
        print("  Please upgrade Python or create a new environment with Python 3.8+")
        return False


def install_packages_matching_colab() -> bool:
    """Install packages with versions matching Google Colab."""
    print("\nInstalling packages matching Google Colab...")
    
    # Package versions matching current Colab (as of the user's report)
    colab_packages = [
        "numpy==2.0.2",
        "scipy==1.15.3", 
        "tensorflow==2.18.0",
        "matplotlib==3.10.0",
        "pandas==2.2.2",
        "scikit-learn==1.6.1",
        "strawberryfields",  # Latest compatible version
        "pyyaml",
        "tqdm",
        "seaborn",
        "psutil",
    ]
    
    print("  Installing packages:")
    for package in colab_packages:
        print(f"    - {package}")
    
    try:
        # Install all packages at once
        cmd = [sys.executable, '-m', 'pip', 'install'] + colab_packages
        
        print("\n  Running pip install...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("  ✓ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Package installation failed: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def verify_package_versions() -> Dict[str, str]:
    """Verify installed package versions."""
    print("\nVerifying installed package versions...")
    
    expected_packages = {
        'numpy': '2.0.2',
        'scipy': '1.15.3',
        'tensorflow': '2.18.0',
        'matplotlib': '3.10.0',
        'pandas': '2.2.2',
        'scikit-learn': '1.6.1',
    }
    
    actual_versions = {}
    
    for package, expected_version in expected_packages.items():
        try:
            import pkg_resources
            actual_version = pkg_resources.get_distribution(package).version
            actual_versions[package] = actual_version
            
            if actual_version.startswith(expected_version.split('.')[0]):
                print(f"  ✓ {package}: {actual_version} (expected ~{expected_version})")
            else:
                print(f"  ⚠ {package}: {actual_version} (expected ~{expected_version})")
                
        except Exception as e:
            print(f"  ✗ {package}: Could not verify version - {e}")
            actual_versions[package] = 'unknown'
    
    # Check Strawberry Fields separately
    try:
        import strawberryfields as sf
        sf_version = sf.__version__
        actual_versions['strawberryfields'] = sf_version
        print(f"  ✓ strawberryfields: {sf_version}")
    except Exception as e:
        print(f"  ✗ strawberryfields: Could not verify version - {e}")
        actual_versions['strawberryfields'] = 'unknown'
    
    return actual_versions


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
        ("compatibility", "from utils.compatibility import apply_all_compatibility_patches"),
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
        
        # Test SciPy compatibility
        print("  Testing SciPy compatibility...")
        import scipy.integrate
        
        # Test simps function (should work with compatibility patch)
        x = np.linspace(0, 1, 11)
        y = x**2
        result = scipy.integrate.simps(y, x)
        print("    ✓ SciPy simps function working")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Basic functionality test failed: {e}")
        return False


def test_quantum_gan_creation() -> bool:
    """Test creating basic quantum GAN components."""
    print("\nTesting quantum GAN component creation...")
    
    try:
        # Import QNNCV components
        from models.generators.quantum_sf_generator import QuantumSFGenerator
        from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
        from training.qgan_sf_trainer import QGANSFTrainer
        
        print("  Creating quantum components...")
        
        # Create generator
        generator = QuantumSFGenerator(n_modes=2, latent_dim=2, layers=1, cutoff_dim=5)
        print("    ✓ QuantumSFGenerator created")
        
        # Create discriminator
        discriminator = QuantumSFDiscriminator(n_modes=1, input_dim=2, layers=1, cutoff_dim=5)
        print("    ✓ QuantumSFDiscriminator created")
        
        # Create trainer
        trainer = QGANSFTrainer(generator, discriminator, latent_dim=2)
        print("    ✓ QGANSFTrainer created")
        
        # Test basic generation
        import numpy as np
        noise = np.random.normal(0, 1, (4, 2))
        generated_data = generator.generate(noise)
        print(f"    ✓ Generated data shape: {generated_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Quantum GAN creation test failed: {e}")
        return False


def check_local_gpu() -> Dict[str, any]:
    """Check for local GPU availability."""
    print("\nChecking local GPU availability...")
    
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
                gpu_info['configured'] = True
                
            except RuntimeError as e:
                print(f"    ⚠ GPU configuration warning: {e}")
                
        else:
            print("    ⚠ No GPU detected - using CPU for local development")
            print("    This is normal for local development")
        
        return gpu_info
        
    except Exception as e:
        print(f"    ✗ GPU check failed: {e}")
        return gpu_info


def provide_local_usage_instructions(gpu_info: Dict[str, any]):
    """Provide instructions for local development."""
    print("\n" + "=" * 70)
    print("LOCAL DEVELOPMENT SETUP COMPLETE")
    print("=" * 70)
    
    print("\n🎯 NEXT STEPS FOR LOCAL DEVELOPMENT:")
    
    print("\n1. Test your setup:")
    print("   python -c \"from utils.compatibility import apply_all_compatibility_patches; apply_all_compatibility_patches()\"")
    
    print("\n2. Run a simple test:")
    print("   python -c \"")
    print("   from models.generators.quantum_sf_generator import QuantumSFGenerator")
    print("   gen = QuantumSFGenerator(n_modes=2, latent_dim=2)")
    print("   print('✓ Local setup working!')\"")
    
    print("\n3. Try the tutorials locally:")
    print("   - Open tutorials/minimal_sf_qgan.ipynb in Jupyter")
    print("   - Run tutorials/extended_sf_qgan_training.ipynb")
    
    print("\n4. When ready for Colab:")
    print("   - Commit your changes to git")
    print("   - Use setup/setup_colab_modern.py in Colab")
    print("   - The compatibility patches will work the same way")
    
    if gpu_info['available']:
        print(f"\n🚀 LOCAL GPU DETECTED ({gpu_info['count']} GPU(s))")
        print("   You can test GPU acceleration locally!")
    else:
        print("\n💻 CPU MODE (Local Development)")
        print("   Perfect for development and testing")
        print("   Switch to Colab GPU runtime for faster training")
    
    print("\n" + "=" * 70)


def main() -> bool:
    """Main setup function for local development."""
    print_banner()
    
    # Step 1: Check Python version
    print("Step 1: Checking Python version...")
    print("-" * 35)
    if not check_python_version():
        return False
    
    # Step 2: Install packages matching Colab
    print("\nStep 2: Installing packages (matching Colab)...")
    print("-" * 45)
    if not install_packages_matching_colab():
        return False
    
    # Step 3: Verify package versions
    print("\nStep 3: Verifying package versions...")
    print("-" * 35)
    package_versions = verify_package_versions()
    
    # Step 4: Setup project paths
    print("\nStep 4: Setting up project paths...")
    print("-" * 35)
    if not setup_project_paths():
        return False
    
    # Step 5: Apply compatibility patches
    print("\nStep 5: Applying compatibility patches...")
    print("-" * 40)
    if not apply_compatibility_patches():
        return False
    
    # Step 6: Test QNNCV imports
    print("\nStep 6: Testing QNNCV imports...")
    print("-" * 30)
    import_results = test_qnncv_imports()
    
    # Step 7: Test basic functionality
    print("\nStep 7: Testing basic functionality...")
    print("-" * 35)
    functionality_test = test_basic_functionality()
    
    # Step 8: Test quantum GAN creation
    print("\nStep 8: Testing quantum GAN creation...")
    print("-" * 35)
    qgan_test = test_quantum_gan_creation()
    
    # Step 9: Check local GPU
    print("\nStep 9: Checking local GPU...")
    print("-" * 25)
    gpu_info = check_local_gpu()
    
    # Step 10: Report results
    print("\n" + "=" * 70)
    print("LOCAL SETUP SUMMARY")
    print("=" * 70)
    
    # Check core functionality
    core_components = ['QuantumSFGenerator', 'QuantumSFDiscriminator', 'QGANSFTrainer']
    core_success = all(import_results.get(comp, False) for comp in core_components)
    
    print("✓ Environment: Local Development")
    
    if core_success:
        print("✓ QNNCV Core Components: WORKING")
    else:
        print("✗ QNNCV Core Components: FAILED")
    
    if functionality_test:
        print("✓ Basic Functionality: WORKING")
    else:
        print("✗ Basic Functionality: FAILED")
    
    if qgan_test:
        print("✓ Quantum GAN Creation: WORKING")
    else:
        print("✗ Quantum GAN Creation: FAILED")
    
    if gpu_info['available'] and gpu_info['configured']:
        print(f"✓ Local GPU: AVAILABLE ({gpu_info['count']} GPU(s))")
    elif gpu_info['available']:
        print(f"⚠ Local GPU: DETECTED BUT NOT CONFIGURED ({gpu_info['count']} GPU(s))")
    else:
        print("⚠ Local GPU: NOT AVAILABLE (CPU mode)")
    
    # Overall success
    overall_success = core_success and functionality_test and qgan_test
    
    if overall_success:
        print("\n🎉 LOCAL SETUP COMPLETED SUCCESSFULLY!")
        provide_local_usage_instructions(gpu_info)
        return True
    else:
        print("\n❌ LOCAL SETUP FAILED!")
        print("Check the error messages above for details.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ QNNCV local development environment is ready!")
            print("You can now develop and test locally before deploying to Colab.")
        else:
            print("\n❌ Local setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
