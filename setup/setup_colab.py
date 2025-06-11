"""
Google Colab Setup Script for QNNCV
===================================

This script sets up the QNNCV environment specifically for Google Colab:
- Direct pip package installation (no conda environment management)
- Compatibility fixes for SciPy and other packages
- GPU configuration for TensorFlow
- Path setup for project modules
- Warning suppression for clean output

Usage in Colab:
    !python setup/setup_colab.py

This will install all dependencies and apply compatibility fixes.
"""

import subprocess
import sys
import os
import warnings

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("QNNCV Google Colab Setup")
    print("Quantum Neural Networks for Continuous Variables")
    print("=" * 60)
    print()

def run_command(command, capture_output=True, shell=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            capture_output=capture_output, 
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_colab_environment():
    """Check if running in Google Colab."""
    try:
        import google.colab
        print("✓ Google Colab environment detected")
        return True
    except ImportError:
        print("⚠ Not running in Google Colab - this script is optimized for Colab")
        return False

def install_core_packages():
    """Install core scientific packages with immediate SciPy fix."""
    print("Installing core scientific packages...")
    
    # Step 1: Force correct NumPy version first (critical for binary compatibility)
    print("  Ensuring correct NumPy version...")
    
    # Uninstall any existing NumPy to prevent conflicts
    print("    Removing any existing NumPy...")
    run_command("pip uninstall numpy -y", capture_output=True)
    
    # Install exact NumPy version
    numpy_version = "numpy==1.24.4"
    print(f"    Installing {numpy_version}...")
    success, stdout, stderr = run_command(f"pip install --no-cache-dir '{numpy_version}'")
    
    if not success:
        print(f"    ✗ NumPy installation failed: {stderr}")
        return False
    
    # Verify NumPy version
    try:
        import numpy as np
        actual_version = np.__version__
        if not actual_version.startswith('1.24'):
            print(f"    ✗ Wrong NumPy version installed: {actual_version}")
            return False
        print(f"    ✓ NumPy {actual_version} installed correctly")
        
        # Test binary compatibility
        from numpy.random import RandomState
        print("    ✓ NumPy binary compatibility verified")
        
    except Exception as e:
        print(f"    ✗ NumPy verification failed: {e}")
        return False
    
    # Step 2: Install SciPy with correct version
    scipy_version = "scipy>=1.0.0,<1.14.0"
    print(f"  Installing {scipy_version}...")
    success, stdout, stderr = run_command(f"pip install -q '{scipy_version}'")
    
    if not success:
        print(f"    ✗ SciPy installation failed: {stderr}")
        return False
    
    print(f"    ✓ SciPy installed")
    
    # Apply SciPy compatibility fix immediately after SciPy installation
    print("  Applying SciPy compatibility fix...")
    try:
        import scipy.integrate
        if not hasattr(scipy.integrate, 'simps'):
            if hasattr(scipy.integrate, 'simpson'):
                scipy.integrate.simps = scipy.integrate.simpson
                print("    ✓ SciPy simps -> simpson compatibility applied")
            else:
                print("    ⚠ Neither simps nor simpson found in scipy.integrate")
        else:
            print("    ✓ SciPy simps already available")
    except ImportError:
        print("    ✗ Could not import scipy.integrate")
        return False
    
    # Install remaining packages
    remaining_packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0"
    ]
    
    for package in remaining_packages:
        print(f"  Installing {package}...")
        success, stdout, stderr = run_command(f"pip install -q '{package}'")
        
        if success:
            print(f"    ✓ {package.split('>=')[0]} installed")
        else:
            print(f"    ✗ {package} failed: {stderr}")
            return False
    
    return True

def install_tensorflow():
    """Install TensorFlow with GPU support."""
    print("Installing TensorFlow...")
    
    # Install TensorFlow with version constraints from requirements.txt
    success, stdout, stderr = run_command("pip install -q 'tensorflow>=2.13.0,<=2.15.0'")
    
    if success:
        print("    ✓ TensorFlow installed")
        return True
    else:
        print(f"    ✗ TensorFlow installation failed: {stderr}")
        return False

def install_quantum_packages():
    """Install quantum computing packages."""
    print("Installing quantum packages...")
    
    quantum_packages = [
        "strawberryfields",
        # Note: PennyLane not needed for current implementation
        # "pennylane"  
    ]
    
    success_count = 0
    for package in quantum_packages:
        print(f"  Installing {package}...")
        success, stdout, stderr = run_command(f"pip install -q {package}")
        
        if success:
            print(f"    ✓ {package} installed")
            success_count += 1
        else:
            print(f"    ⚠ {package} failed: {stderr}")
    
    return success_count > 0

def apply_compatibility_fixes():
    """Apply compatibility fixes for known issues."""
    print("Applying compatibility fixes...")
    
    # Fix 1: SciPy simps compatibility
    print("  Applying SciPy simps compatibility fix...")
    try:
        import scipy.integrate
        if not hasattr(scipy.integrate, 'simps'):
            if hasattr(scipy.integrate, 'simpson'):
                scipy.integrate.simps = scipy.integrate.simpson
                print("    ✓ SciPy simps -> simpson compatibility applied")
            else:
                print("    ⚠ Neither simps nor simpson found in scipy.integrate")
        else:
            print("    ✓ SciPy simps already available")
    except ImportError:
        print("    ✗ Could not import scipy.integrate")
        return False
    
    # Fix 2: TensorFlow warnings suppression
    print("  Configuring TensorFlow warnings...")
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print("    ✓ TensorFlow warnings suppressed")
    except ImportError:
        print("    ⚠ TensorFlow not available for warning configuration")
    
    # Fix 3: General warnings suppression
    print("  Configuring general warnings...")
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='strawberryfields')
    print("    ✓ General warnings suppressed")
    
    return True

def configure_gpu():
    """Configure GPU settings for TensorFlow."""
    print("Configuring GPU settings...")
    
    try:
        import tensorflow as tf
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"    ✓ {len(gpus)} GPU(s) detected")
            
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("    ✓ GPU memory growth enabled")
            
            # Set mixed precision for better performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("    ✓ Mixed precision enabled")
            
            return True
        else:
            print("    ⚠ No GPU detected - using CPU")
            print("    Consider switching to GPU runtime: Runtime → Change runtime type → GPU")
            return False
            
    except ImportError:
        print("    ✗ TensorFlow not available for GPU configuration")
        return False

def setup_project_paths():
    """Set up Python paths for the project."""
    print("Setting up project paths...")
    
    # Get current working directory (should be QNNCV root)
    project_root = os.getcwd()
    src_path = os.path.join(project_root, 'src')
    
    # Add src to Python path
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"    ✓ Added {src_path} to Python path")
    else:
        print(f"    ✓ {src_path} already in Python path")
    
    # Verify project structure
    required_dirs = ['src', 'tutorials', 'setup']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"    ✓ {dir_name}/ directory found")
        else:
            missing_dirs.append(dir_name)
            print(f"    ✗ {dir_name}/ directory missing")
    
    if missing_dirs:
        print(f"    ⚠ Missing directories: {missing_dirs}")
        print("    Make sure you're running from the QNNCV root directory")
        return False
    
    return True

def test_imports():
    """Test that all critical imports work."""
    print("Testing package imports...")
    
    test_cases = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("scipy", "import scipy; print(f'SciPy {scipy.__version__}')"),
        ("matplotlib", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"),
        ("tensorflow", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"),
        ("strawberryfields", "import strawberryfields as sf; print(f'Strawberry Fields {sf.__version__}')"),
    ]
    
    results = {}
    for package_name, test_code in test_cases:
        try:
            exec(test_code)
            print(f"    ✓ {package_name} working")
            results[package_name] = True
        except Exception as e:
            print(f"    ✗ {package_name} failed: {e}")
            results[package_name] = False
    
    return results

def test_qnncv_imports():
    """Test QNNCV-specific imports."""
    print("Testing QNNCV module imports...")
    
    qnncv_imports = [
        ("QuantumSFGenerator", "from models.generators.quantum_sf_generator import QuantumSFGenerator"),
        ("QuantumSFDiscriminator", "from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator"),
        ("QGANSFTrainer", "from training.qgan_sf_trainer import QGANSFTrainer"),
        ("warning_suppression", "from utils.warning_suppression import enable_clean_training"),
    ]
    
    results = {}
    for component_name, import_code in qnncv_imports:
        try:
            exec(import_code)
            print(f"    ✓ {component_name} import successful")
            results[component_name] = True
        except Exception as e:
            print(f"    ✗ {component_name} import failed: {e}")
            results[component_name] = False
    
    return results

def test_basic_functionality():
    """Test basic quantum functionality."""
    print("Testing basic quantum functionality...")
    
    try:
        # Test Strawberry Fields basic operation
        import strawberryfields as sf
        import tensorflow as tf
        
        prog = sf.Program(1)
        eng = sf.Engine("tf", backend_options={"cutoff_dim": 5})
        
        with prog.context as q:
            sf.ops.Dgate(0.5) | q[0]
            sf.ops.MeasureHomodyne(0) | q[0]
        
        result = eng.run(prog)
        print(f"    ✓ Strawberry Fields basic test successful")
        
        # Fix numpy formatting issue
        try:
            sample_value = float(result.samples[0])
            print(f"      Sample measurement: {sample_value:.3f}")
        except (TypeError, ValueError, IndexError):
            print(f"      Sample measurement: {result.samples}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Basic quantum test failed: {e}")
        return False

def provide_usage_instructions():
    """Provide instructions for using the setup."""
    print("\n" + "=" * 60)
    print("COLAB SETUP COMPLETE!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Import QNNCV modules:")
    print("   from models.generators.quantum_sf_generator import QuantumSFGenerator")
    print("   from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator")
    print("   from training.qgan_sf_trainer import QGANSFTrainer")
    
    print("\n2. Check the tutorials:")
    print("   - tutorials/minimal_sf_qgan.ipynb")
    print("   - tutorials/extended_sf_qgan_training.ipynb")
    
    print("\n3. Create your quantum GAN:")
    print("   generator = QuantumSFGenerator(n_modes=2, latent_dim=2)")
    print("   discriminator = QuantumSFDiscriminator(n_modes=1, input_dim=2)")
    print("   trainer = QGANSFTrainer(generator, discriminator, latent_dim=2)")
    
    print("\n4. Train your model:")
    print("   history = trainer.train(data, epochs=100, batch_size=16)")

def setup_colab_environment():
    """Main function to set up QNNCV in Google Colab."""
    print_banner()
    
    # Step 1: Check environment
    is_colab = check_colab_environment()
    
    # Step 2: Install packages
    print("\nStep 1: Installing packages...")
    print("-" * 30)
    
    if not install_core_packages():
        print("✗ Core package installation failed")
        return False
    
    if not install_tensorflow():
        print("✗ TensorFlow installation failed")
        return False
    
    if not install_quantum_packages():
        print("✗ Quantum package installation failed")
        return False
    
    # Step 3: Apply compatibility fixes
    print("\nStep 2: Applying compatibility fixes...")
    print("-" * 40)
    
    if not apply_compatibility_fixes():
        print("✗ Compatibility fixes failed")
        return False
    
    # Step 4: Configure GPU
    print("\nStep 3: Configuring GPU...")
    print("-" * 25)
    
    gpu_available = configure_gpu()
    
    # Step 5: Setup paths
    print("\nStep 4: Setting up project paths...")
    print("-" * 35)
    
    if not setup_project_paths():
        print("✗ Project path setup failed")
        return False
    
    # Step 6: Test imports
    print("\nStep 5: Testing imports...")
    print("-" * 25)
    
    import_results = test_imports()
    qnncv_results = test_qnncv_imports()
    
    # Step 7: Test functionality
    print("\nStep 6: Testing functionality...")
    print("-" * 30)
    
    functionality_test = test_basic_functionality()
    
    # Step 8: Report results
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    core_packages = ['numpy', 'scipy', 'matplotlib', 'tensorflow']
    core_success = all(import_results.get(pkg, False) for pkg in core_packages)
    
    if core_success:
        print("✓ Core packages: WORKING")
    else:
        print("✗ Core packages: FAILED")
    
    if import_results.get('strawberryfields', False):
        print("✓ Strawberry Fields: WORKING")
    else:
        print("✗ Strawberry Fields: FAILED")
    
    qnncv_components = ['QuantumSFGenerator', 'QuantumSFDiscriminator', 'QGANSFTrainer']
    qnncv_success = all(qnncv_results.get(comp, False) for comp in qnncv_components)
    
    if qnncv_success:
        print("✓ QNNCV modules: WORKING")
    else:
        print("✗ QNNCV modules: FAILED")
    
    if gpu_available:
        print("✓ GPU acceleration: AVAILABLE")
    else:
        print("⚠ GPU acceleration: NOT AVAILABLE")
    
    if functionality_test:
        print("✓ Basic quantum operations: WORKING")
    else:
        print("✗ Basic quantum operations: FAILED")
    
    # Step 9: Provide instructions
    if core_success and qnncv_success:
        provide_usage_instructions()
        return True
    else:
        print("\n✗ Setup failed - check error messages above")
        return False

if __name__ == "__main__":
    try:
        success = setup_colab_environment()
        if success:
            print("\nColab setup completed successfully! :)")
        else:
            print("\nColab setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
