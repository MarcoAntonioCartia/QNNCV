"""
Google Colab Setup Script for QNNCV - Conda-Style Approach
==========================================================

This script mimics the successful conda installation strategy using pip only:
- Install packages in logical groups (like conda does)
- Let pip resolve compatible versions naturally (like conda does)
- Avoid over-constraining versions (like conda does)
- Only apply specific fixes when needed

Usage in Colab:
    !python setup/setup_colab_conda_style.py
"""

import subprocess
import sys
import os
import warnings

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("QNNCV Google Colab Setup - Conda-Style Approach")
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

def install_core_scientific_stack():
    """Install core scientific packages together (conda-style)."""
    print("Installing core scientific stack...")
    print("  (Letting pip resolve compatible versions naturally)")
    
    # Install core packages together - let pip resolve versions
    core_packages = [
        "numpy",
        "scipy", 
        "matplotlib",
        "scikit-learn",
        "pandas",
        "seaborn",
        "tqdm",
        "pyyaml"
    ]
    
    # Install all core packages in one command (like conda does)
    package_list = " ".join(core_packages)
    print(f"  Installing: {package_list}")
    
    success, stdout, stderr = run_command(f"pip install {package_list}")
    
    if success:
        print("    ✓ Core scientific stack installed successfully")
        return True
    else:
        print(f"    ✗ Core stack installation failed: {stderr}")
        return False

def install_tensorflow():
    """Install TensorFlow separately (conda-style)."""
    print("Installing TensorFlow...")
    print("  (Letting pip choose compatible version)")
    
    success, stdout, stderr = run_command("pip install tensorflow")
    
    if success:
        print("    ✓ TensorFlow installed successfully")
        return True
    else:
        print(f"    ✗ TensorFlow installation failed: {stderr}")
        return False

def install_quantum_packages():
    """Install quantum packages (conda-style)."""
    print("Installing quantum packages...")
    print("  (Using pip for quantum-specific packages)")
    
    success, stdout, stderr = run_command("pip install strawberryfields")
    
    if success:
        print("    ✓ Strawberry Fields installed successfully")
        return True
    else:
        print(f"    ✗ Strawberry Fields installation failed: {stderr}")
        return False

def apply_minimal_compatibility_fixes():
    """Apply only essential compatibility fixes."""
    print("Applying minimal compatibility fixes...")
    
    # Only fix SciPy simps if needed
    try:
        import scipy.integrate
        if not hasattr(scipy.integrate, 'simps'):
            if hasattr(scipy.integrate, 'simpson'):
                scipy.integrate.simps = scipy.integrate.simpson
                print("    ✓ SciPy simps compatibility applied")
            else:
                print("    ⚠ SciPy simps/simpson not found")
        else:
            print("    ✓ SciPy simps already available")
    except ImportError:
        print("    ✗ Could not import scipy.integrate")
        return False
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    print("    ✓ Warnings suppressed")
    
    return True

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
    
    return True

def test_package_versions():
    """Test and display package versions."""
    print("Checking installed package versions...")
    
    test_cases = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("scipy", "import scipy; print(f'SciPy {scipy.__version__}')"),
        ("matplotlib", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"),
        ("sklearn", "import sklearn; print(f'Scikit-learn {sklearn.__version__}')"),
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

def test_basic_quantum_functionality():
    """Test basic quantum functionality."""
    print("Testing basic quantum functionality...")
    
    try:
        import strawberryfields as sf
        import numpy as np
        
        # Simple test
        prog = sf.Program(1)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        
        with prog.context as q:
            sf.ops.Dgate(0.5) | q[0]
        
        result = eng.run(prog)
        print("    ✓ Basic quantum operations working")
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

def setup_colab_conda_style():
    """Main function - conda-style setup for Colab."""
    print_banner()
    
    # Step 1: Check environment
    is_colab = check_colab_environment()
    
    # Step 2: Install packages in conda-style order
    print("\nStep 1: Installing packages (conda-style approach)...")
    print("-" * 50)
    
    # Install core scientific stack first (like conda)
    if not install_core_scientific_stack():
        print("✗ Core scientific stack installation failed")
        return False
    
    # Install TensorFlow separately (like conda)
    if not install_tensorflow():
        print("✗ TensorFlow installation failed")
        return False
    
    # Install quantum packages last (like conda)
    if not install_quantum_packages():
        print("✗ Quantum package installation failed")
        return False
    
    # Step 3: Apply minimal fixes
    print("\nStep 2: Applying compatibility fixes...")
    print("-" * 40)
    
    if not apply_minimal_compatibility_fixes():
        print("✗ Compatibility fixes failed")
        return False
    
    # Step 4: Setup paths
    print("\nStep 3: Setting up project paths...")
    print("-" * 35)
    
    if not setup_project_paths():
        print("✗ Project path setup failed")
        return False
    
    # Step 5: Test everything
    print("\nStep 4: Testing installation...")
    print("-" * 30)
    
    version_results = test_package_versions()
    qnncv_results = test_qnncv_imports()
    quantum_test = test_basic_quantum_functionality()
    
    # Step 6: Report results
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    core_packages = ['numpy', 'scipy', 'matplotlib', 'sklearn']
    core_success = all(version_results.get(pkg, False) for pkg in core_packages)
    
    if core_success:
        print("✓ Core scientific packages: WORKING")
    else:
        print("✗ Core scientific packages: FAILED")
    
    if version_results.get('tensorflow', False):
        print("✓ TensorFlow: WORKING")
    else:
        print("✗ TensorFlow: FAILED")
    
    if version_results.get('strawberryfields', False):
        print("✓ Strawberry Fields: WORKING")
    else:
        print("✗ Strawberry Fields: FAILED")
    
    qnncv_components = ['QuantumSFGenerator', 'QuantumSFDiscriminator', 'QGANSFTrainer']
    qnncv_success = all(qnncv_results.get(comp, False) for comp in qnncv_components)
    
    if qnncv_success:
        print("✓ QNNCV modules: WORKING")
    else:
        print("✗ QNNCV modules: FAILED")
    
    if quantum_test:
        print("✓ Basic quantum operations: WORKING")
    else:
        print("✗ Basic quantum operations: FAILED")
    
    # Step 7: Provide instructions
    if core_success and qnncv_success:
        provide_usage_instructions()
        return True
    else:
        print("\n✗ Setup failed - check error messages above")
        return False

if __name__ == "__main__":
    try:
        success = setup_colab_conda_style()
        if success:
            print("\nColab setup completed successfully! 🎉")
        else:
            print("\nColab setup failed! 😞")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
