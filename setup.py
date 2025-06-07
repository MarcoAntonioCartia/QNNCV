"""
QGAN Project Setup Script
========================

This script sets up the complete QGAN development environment including:
- Conda environment creation with Python 3.11
- All required dependencies (TensorFlow, quantum packages)
- Project structure verification
- Initial testing

Usage:
    python setup.py

This will create a conda environment called 'qnncv' and install all dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("🔬 QGAN Project Setup")
    print("   Quantum Generative Adversarial Networks")
    print("=" * 70)
    print()
    print("This script will:")
    print("• Create conda environment 'qnncv' with Python 3.11")
    print("• Install TensorFlow, quantum packages, and dependencies")
    print("• Verify project structure")
    print("• Run basic functionality tests")
    print()

def check_project_structure():
    """Verify we're in the right directory and have the required files."""
    print("Checking project structure...")
    
    required_files = [
        "setup_environment.py",
        "utils.py",
        "main_qgan.py",
        "config.yaml",
        "requirements.txt"
    ]
    
    required_dirs = [
        "NN"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"  ✓ {file}")
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
        else:
            print(f"  ✓ {dir_name}/")
    
    if missing_files or missing_dirs:
        print("\n✗ Missing required files/directories:")
        for item in missing_files + missing_dirs:
            print(f"    - {item}")
        print("\nPlease run this script from the QNNCV project root directory.")
        return False
    
    print("✓ Project structure verified")
    return True

def run_environment_setup():
    """Run the environment setup script."""
    print("\nRunning environment setup...")
    print("=" * 50)
    
    try:
        # Import and run the setup function
        from setup_environment import setup_qgan_environment
        
        success = setup_qgan_environment()
        return success
        
    except ImportError as e:
        print(f"✗ Could not import setup_environment: {e}")
        return False
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return False

def verify_installation():
    """Verify the installation by running basic tests."""
    print("\nVerifying installation...")
    print("=" * 30)
    
    # Test if we can activate the environment and run basic commands
    test_commands = [
        "conda activate qnncv && python -c \"import numpy; print('NumPy working')\"",
        "conda activate qnncv && python -c \"import sklearn; print('Scikit-learn working')\"",
        "conda activate qnncv && python -c \"import matplotlib; print('Matplotlib working')\"",
    ]
    
    # Optional tests (might fail but that's OK)
    optional_commands = [
        "conda activate qnncv && python -c \"import tensorflow; print('TensorFlow working')\"",
        "conda activate qnncv && python -c \"import strawberryfields; print('Strawberry Fields working')\"",
    ]
    
    print("Testing core packages...")
    core_success = True
    for cmd in test_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {result.stdout.strip()}")
            else:
                print(f"  ✗ Command failed: {cmd}")
                core_success = False
        except Exception as e:
            print(f"  ✗ Error running command: {e}")
            core_success = False
    
    print("\nTesting optional packages...")
    for cmd in optional_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {result.stdout.strip()}")
            else:
                print(f"  ⚠ Optional package not available")
        except Exception as e:
            print(f"  ⚠ Optional test failed: {e}")
    
    return core_success

def run_project_tests():
    """Run the project's built-in tests."""
    print("\nRunning project tests...")
    print("=" * 25)
    
    # Test with the no-TensorFlow version first
    print("Testing basic functionality (no TensorFlow)...")
    try:
        result = subprocess.run(
            "python test_basic_no_tf.py", 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✓ Basic tests passed")
        else:
            print("⚠ Basic tests had issues")
            print(result.stdout)
    except Exception as e:
        print(f"⚠ Could not run basic tests: {e}")
    
    # Test with TensorFlow if available
    print("\nTesting with TensorFlow environment...")
    try:
        result = subprocess.run(
            "conda run -n qnncv python test_basic_no_tf.py", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Environment tests passed")
        else:
            print("⚠ Environment tests had issues")
    except Exception as e:
        print(f"⚠ Could not run environment tests: {e}")

def provide_final_instructions():
    """Provide final usage instructions."""
    print("\n" + "=" * 70)
    print("🎉 QGAN PROJECT SETUP COMPLETE!")
    print("=" * 70)
    
    print("\n📋 NEXT STEPS:")
    print("-" * 15)
    
    print("\n1. Activate your environment:")
    print("   conda activate qnncv")
    
    print("\n2. Test the classical GAN:")
    print("   python test_basic.py")
    
    print("\n3. Test enhanced components:")
    print("   python enhanced_test.py")
    
    print("\n4. Start development:")
    print("   python main_qgan.py")
    
    print("\n📁 Project Structure:")
    print("-" * 18)
    print("   NN/              - Neural network components")
    print("   data/            - Dataset storage")
    print("   results/         - Output and visualizations")
    print("   utils.py         - Utility functions")
    print("   main_qgan.py     - Main training script")
    print("   config.yaml      - Configuration settings")
    
    print("\n🔧 Environment Management:")
    print("-" * 27)
    print("   conda activate qnncv     - Activate environment")
    print("   conda deactivate         - Deactivate environment")
    print("   conda env list           - List environments")
    print("   conda env remove -n qnncv - Remove environment")
    
    print("\n📚 Documentation:")
    print("-" * 17)
    print("   PROJECT_ROADMAP.md       - Development roadmap")
    print("   TESTING_REPORT.md        - Testing results")
    print("   README.md                - Project overview")
    
    print("\n💡 Tips:")
    print("-" * 8)
    print("   • Always activate the environment before running scripts")
    print("   • Check TESTING_REPORT.md for detailed component status")
    print("   • Use 'conda activate qnncv' in each new terminal session")
    
    print("\n" + "=" * 70)

def main():
    """Main setup orchestration function."""
    print_banner()
    
    # Step 1: Check project structure
    if not check_project_structure():
        print("\n❌ Setup aborted due to missing files")
        sys.exit(1)
    
    # Step 2: Run environment setup
    print("\n" + "=" * 70)
    print("STEP 1: Environment Setup")
    print("=" * 70)
    
    setup_success = run_environment_setup()
    
    if not setup_success:
        print("\n❌ Environment setup failed")
        print("\nYou can try:")
        print("1. Run setup_environment.py directly")
        print("2. Check conda installation")
        print("3. Review error messages above")
        sys.exit(1)
    
    # Step 3: Verify installation
    print("\n" + "=" * 70)
    print("STEP 2: Installation Verification")
    print("=" * 70)
    
    verify_success = verify_installation()
    
    # Step 4: Run tests
    print("\n" + "=" * 70)
    print("STEP 3: Project Testing")
    print("=" * 70)
    
    run_project_tests()
    
    # Step 5: Final instructions
    provide_final_instructions()
    
    # Final status
    if setup_success and verify_success:
        print("\n🎉 Setup completed successfully!")
        print("Ready to start QGAN development!")
    else:
        print("\n⚠ Setup completed with some issues")
        print("Check the messages above for details")
    
    return setup_success and verify_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 