"""
Enhanced environment setup script for QGAN project.
This script creates a conda environment and installs all necessary dependencies.
"""

import subprocess
import sys
import os
import json

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

def detect_conda():
    """Check if conda is installed and available."""
    print("Checking conda installation...")
    
    success, stdout, stderr = run_command("conda --version")
    if success:
        print(f"✓ Conda detected: {stdout.strip()}")
        return True
    else:
        print("✗ Conda not found in PATH")
        print("Please install Anaconda or Miniconda first:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        return False

def check_existing_environment(env_name):
    """Check if conda environment already exists."""
    print(f"Checking for existing environment '{env_name}'...")
    
    success, stdout, stderr = run_command("conda env list --json")
    if not success:
        print("✗ Could not list conda environments")
        return False
    
    try:
        env_data = json.loads(stdout)
        env_paths = env_data.get('envs', [])
        
        for env_path in env_paths:
            if env_name in os.path.basename(env_path):
                print(f"⚠ Environment '{env_name}' already exists at: {env_path}")
                return True
        
        print(f"✓ Environment '{env_name}' not found - ready to create")
        return False
        
    except json.JSONDecodeError:
        print("✗ Could not parse conda environment list")
        return False

def prompt_user_choice(env_name):
    """Prompt user about what to do with existing environment."""
    print(f"\nEnvironment '{env_name}' already exists.")
    print("Options:")
    print("1. Remove and recreate (recommended for clean setup)")
    print("2. Try to use existing environment")
    print("3. Cancel setup")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

def remove_conda_environment(env_name):
    """Remove existing conda environment."""
    print(f"Removing existing environment '{env_name}'...")
    
    success, stdout, stderr = run_command(f"conda env remove -n {env_name} -y")
    if success:
        print(f"✓ Environment '{env_name}' removed successfully")
        return True
    else:
        print(f"✗ Failed to remove environment: {stderr}")
        return False

def create_conda_environment(env_name, python_version="3.11"):
    """Create a new conda environment with specified Python version."""
    print(f"Creating conda environment '{env_name}' with Python {python_version}...")
    
    command = f"conda create -n {env_name} python={python_version} -y"
    success, stdout, stderr = run_command(command)
    
    if success:
        print(f"✓ Environment '{env_name}' created successfully")
        return True
    else:
        print(f"✗ Failed to create environment: {stderr}")
        return False

def install_conda_packages(env_name):
    """Install core packages via conda."""
    print("Installing core packages via conda...")
    
    # Core scientific packages that work well with conda
    conda_packages = [
        "numpy",
        "scipy", 
        "matplotlib",
        "scikit-learn",
        "jupyter",
        "pyyaml",
        "pip"  # Ensure pip is available in the environment
    ]
    
    for package in conda_packages:
        print(f"  Installing {package}...")
        command = f"conda install -n {env_name} {package} -y"
        success, stdout, stderr = run_command(command)
        
        if success:
            print(f"    ✓ {package} installed")
        else:
            print(f"    ✗ {package} failed: {stderr}")
            return False
    
    return True

def install_tensorflow_in_env(env_name):
    """Install TensorFlow in the conda environment."""
    print("Installing TensorFlow...")
    
    # Try conda first, then pip
    methods = [
        ("conda", f"conda install -n {env_name} tensorflow -c conda-forge -y"),
        ("pip", f"conda run -n {env_name} pip install tensorflow")
    ]
    
    for method_name, command in methods:
        print(f"  Trying {method_name}...")
        success, stdout, stderr = run_command(command)
        
        if success:
            print(f"    ✓ TensorFlow installed via {method_name}")
            return True
        else:
            print(f"    ✗ {method_name} failed: {stderr}")
    
    print("  ⚠ TensorFlow installation failed with both methods")
    return False

def install_quantum_packages(env_name):
    """Install quantum computing packages."""
    print("Installing quantum packages...")
    
    quantum_packages = [
        "strawberryfields",
        "pennylane"
    ]
    
    success_count = 0
    for package in quantum_packages:
        print(f"  Installing {package}...")
        command = f"conda run -n {env_name} pip install {package}"
        success, stdout, stderr = run_command(command)
        
        if success:
            print(f"    ✓ {package} installed")
            success_count += 1
        else:
            print(f"    ⚠ {package} failed: {stderr}")
    
    return success_count > 0  # Return True if at least one quantum package installed

def test_environment_imports(env_name):
    """Test basic imports in the new environment."""
    print("Testing package imports in new environment...")
    
    test_imports = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("matplotlib", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"),
        ("sklearn", "import sklearn; print(f'Scikit-learn {sklearn.__version__}')"),
        ("tensorflow", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"),
        ("strawberryfields", "import strawberryfields as sf; print(f'Strawberry Fields {sf.__version__}')"),
        ("pennylane", "import pennylane as qml; print(f'PennyLane {qml.__version__}')")
    ]
    
    results = {}
    for package_name, test_code in test_imports:
        print(f"  Testing {package_name}...")
        command = f'conda run -n {env_name} python -c "{test_code}"'
        success, stdout, stderr = run_command(command)
        
        if success:
            version_info = stdout.strip()
            print(f"    ✓ {version_info}")
            results[package_name] = True
        else:
            print(f"    ✗ {package_name} import failed")
            results[package_name] = False
    
    return results

def provide_usage_instructions(env_name):
    """Provide instructions for using the environment."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    print(f"\nTo activate your environment:")
    print(f"  conda activate {env_name}")
    
    print(f"\nTo deactivate:")
    print(f"  conda deactivate")
    
    print(f"\nTo test the QGAN project:")
    print(f"  conda activate {env_name}")
    print(f"  cd QNNCV")
    print(f"  python test_basic.py")
    
    print(f"\nTo remove this environment later:")
    print(f"  conda env remove -n {env_name}")

def setup_qgan_environment():
    """Main function to set up the QGAN development environment."""
    env_name = "qnncv"
    python_version = "3.11"
    
    print("QGAN Project Environment Setup")
    print("=" * 30)
    print(f"Target environment: {env_name}")
    print(f"Python version: {python_version}")
    print()
    
    # Step 1: Check conda
    if not detect_conda():
        return False
    
    # Step 2: Check existing environment
    env_exists = check_existing_environment(env_name)
    
    if env_exists:
        choice = prompt_user_choice(env_name)
        
        if choice == '1':  # Remove and recreate
            if not remove_conda_environment(env_name):
                return False
        elif choice == '2':  # Use existing
            print(f"Using existing environment '{env_name}'")
            # Skip to testing
            test_results = test_environment_imports(env_name)
            provide_usage_instructions(env_name)
            return True
        else:  # Cancel
            print("Setup cancelled by user")
            return False
    
    # Step 3: Create environment
    if not create_conda_environment(env_name, python_version):
        return False
    
    # Step 4: Install packages
    print("\nInstalling packages...")
    
    if not install_conda_packages(env_name):
        print("✗ Failed to install core packages")
        return False
    
    tf_success = install_tensorflow_in_env(env_name)
    quantum_success = install_quantum_packages(env_name)
    
    # Step 5: Test environment
    print("\nTesting environment...")
    test_results = test_environment_imports(env_name)
    
    # Step 6: Report results
    print("\n" + "="*50)
    print("INSTALLATION SUMMARY")
    print("="*50)
    
    core_packages = ['numpy', 'matplotlib', 'sklearn']
    core_success = all(test_results.get(pkg, False) for pkg in core_packages)
    
    if core_success:
        print("✓ Core scientific packages: WORKING")
    else:
        print("✗ Core scientific packages: FAILED")
    
    if test_results.get('tensorflow', False):
        print("✓ TensorFlow: WORKING")
    else:
        print("✗ TensorFlow: FAILED")
    
    quantum_packages = ['strawberryfields', 'pennylane']
    quantum_working = any(test_results.get(pkg, False) for pkg in quantum_packages)
    
    if quantum_working:
        print("✓ Quantum packages: PARTIALLY WORKING")
    else:
        print("⚠ Quantum packages: NONE WORKING")
    
    # Step 7: Provide instructions
    if core_success:
        provide_usage_instructions(env_name)
        return True
    else:
        print("\n✗ Setup failed - core packages not working")
        return False

# Legacy functions for backward compatibility
def setup_basic_environment():
    """Legacy function - redirects to conda setup."""
    print("Note: Redirecting to conda environment setup...")
    return setup_qgan_environment()

def install_package(package):
    """Legacy function for installing packages."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def test_import(module_name, package_name=None):
    """Legacy function for testing imports."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} is available")
        return True
    except ImportError:
        print(f"✗ {module_name} is not available")
        if package_name:
            print(f"  Try: pip install {package_name}")
        return False

if __name__ == "__main__":
    success = setup_qgan_environment()
    if success:
        print("\n🎉 Environment setup completed successfully!")
    else:
        print("\n❌ Environment setup failed!")
        sys.exit(1) 