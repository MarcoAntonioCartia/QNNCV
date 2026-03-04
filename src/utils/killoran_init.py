"""
Killoran CV-QNN Initialization Script
======================================

Comprehensive initialization and validation script for the Killoran CV-QNN
architecture. This script provides functions to:

1. Check all dependencies and imports
2. Initialize the environment with proper configurations
3. Validate the setup before training
4. Provide helpful error messages and suggestions
"""

import sys
import os
import traceback
from typing import Dict, List, Tuple, Optional, Any


def check_basic_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all basic Python dependencies are available.
    
    Returns:
        Tuple of (success: bool, missing_packages: List[str])
    """
    required_packages = [
        'numpy', 'tensorflow', 'strawberryfields', 'scipy', 
        'matplotlib', 'argparse', 'os', 'sys', 'typing', 'dataclasses'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing


def check_qnn_cv_modules() -> Tuple[bool, List[str]]:
    """
    Check if all QNNCV modules can be imported.
    
    Returns:
        Tuple of (success: bool, failed_modules: List[str])
    """
    modules_to_check = [
        'src.utils.warning_suppression',
        'src.utils.compatibility', 
        'src.utils.scipy_compat',
        'src.models.generators.quantum_sf_generator',
        'src.models.generators.quantum_distribution_generator',
        'src.models.discriminators.quantum_sf_discriminator',
        'src.models.discriminators.classical_discriminator',
        'src.models.discriminators.distribution_discriminator',
        'src.training.trainer',
        'src.training.qgan_sf_trainer',
        'src.training.distribution_trainer',
        'src.training.killoran_trainer',
        'src.models.generators.killoran_cvqnn'
    ]
    
    failed = []
    for module in modules_to_check:
        try:
            __import__(module)
        except ImportError as e:
            failed.append(f"{module}: {str(e)}")
        except Exception as e:
            failed.append(f"{module}: Unexpected error - {str(e)}")
    
    return len(failed) == 0, failed


def check_killoran_files() -> Tuple[bool, List[str]]:
    """
    Check if the specific Killoran files can be imported.
    
    Returns:
        Tuple of (success: bool, failed_files: List[str])
    """
    files_to_check = [
        'src.models.generators.killoran_cvqnn',
        'src.training.killoran_trainer',
        'train_killoran_qgan'
    ]
    
    failed = []
    for file_path in files_to_check:
        try:
            __import__(file_path)
        except ImportError as e:
            failed.append(f"{file_path}: {str(e)}")
        except Exception as e:
            failed.append(f"{file_path}: Unexpected error - {str(e)}")
    
    return len(failed) == 0, failed


def validate_killoran_environment() -> Dict[str, Any]:
    """
    Comprehensive validation of the Killoran CV-QNN environment.
    
    Returns:
        Dictionary with validation results
    """
    print("=" * 70)
    print("KILLORAN CV-QNN ENVIRONMENT VALIDATION")
    print("=" * 70)
    
    results = {
        'basic_dependencies': {'status': False, 'missing': []},
        'qnn_cv_modules': {'status': False, 'failed': []},
        'killoran_files': {'status': False, 'failed': []},
        'overall_success': False,
        'recommendations': []
    }
    
    # Check basic dependencies
    print("\n1. Checking Basic Dependencies...")
    print("-" * 40)
    success, missing = check_basic_dependencies()
    results['basic_dependencies']['status'] = success
    results['basic_dependencies']['missing'] = missing
    
    if success:
        print("✓ All basic dependencies available")
    else:
        print(f"✗ Missing packages: {', '.join(missing)}")
        results['recommendations'].append(
            f"Install missing packages: pip install {' '.join(missing)}"
        )
    
    # Check QNNCV modules
    print("\n2. Checking QNNCV Modules...")
    print("-" * 40)
    success, failed = check_qnn_cv_modules()
    results['qnn_cv_modules']['status'] = success
    results['qnn_cv_modules']['failed'] = failed
    
    if success:
        print("✓ All QNNCV modules available")
    else:
        print(f"✗ Failed modules ({len(failed)}):")
        for module in failed:
            print(f"  - {module}")
        results['recommendations'].append(
            "Check that all QNNCV files are present and properly structured"
        )
    
    # Check Killoran files
    print("\n3. Checking Killoran Files...")
    print("-" * 40)
    success, failed = check_killoran_files()
    results['killoran_files']['status'] = success
    results['killoran_files']['failed'] = failed
    
    if success:
        print("✓ All Killoran files available")
    else:
        print(f"✗ Failed files ({len(failed)}):")
        for file in failed:
            print(f"  - {file}")
        results['recommendations'].append(
            "Ensure Killoran files are in the correct location"
        )
    
    # Overall assessment
    results['overall_success'] = (
        results['basic_dependencies']['status'] and
        results['qnn_cv_modules']['status'] and
        results['killoran_files']['status']
    )
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if results['overall_success']:
        print("✓ Environment validation PASSED")
        print("✓ Killoran CV-QNN is ready for training!")
    else:
        print("✗ Environment validation FAILED")
        print("✗ Please address the issues above before training")
    
    if results['recommendations']:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("=" * 70)
    
    return results


def test_killoran_functionality() -> bool:
    """
    Test basic Killoran CV-QNN functionality.
    
    Returns:
        True if test passes, False otherwise
    """
    print("\nTesting Killoran CV-QNN Functionality...")
    print("-" * 50)
    
    try:
        # Import required modules
        import tensorflow as tf
        from src.models.generators.killoran_cvqnn import KilloranCVQNN
        from src.training.killoran_trainer import KilloranQGANTrainer, KilloranTrainerConfig
        
        print("✓ Imports successful")
        
        # Test generator creation
        generator = KilloranCVQNN(
            n_layers=2,
            cutoff_dim=10,
            use_kerr=True,
            num_bins=50
        )
        print(f"✓ Generator created: {generator.get_config()}")
        
        # Test forward pass
        z = tf.random.normal([2, generator.latent_dim])
        P_x = generator.generate(z)
        print(f"✓ Forward pass successful: output shape {P_x.shape}")
        
        # Test trainer creation
        from src.training.killoran_trainer import DistributionDiscriminator
        discriminator = DistributionDiscriminator(input_dim=50, hidden_dims=[32, 16])
        config = KilloranTrainerConfig(
            target_type='gaussian',
            n_layers=2,
            cutoff_dim=10,
            use_kerr=True,
            batch_size=4
        )
        trainer = KilloranQGANTrainer(generator, discriminator, config)
        print("✓ Trainer created successfully")
        
        # Test one training step
        z_train = tf.random.normal([4, generator.latent_dim])
        d_loss = trainer.train_discriminator_step(z_train)
        g_loss, P_fake = trainer.train_generator_step(z_train)
        print(f"✓ Training step successful: D loss {d_loss:.4f}, G loss {g_loss:.4f}")
        
        print("\n✓ All functionality tests PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test FAILED: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False


def initialize_killoran_environment() -> bool:
    """
    Complete initialization of the Killoran CV-QNN environment.
    
    Returns:
        True if initialization successful, False otherwise
    """
    print("KILLORAN CV-QNN ENVIRONMENT INITIALIZATION")
    print("=" * 70)
    
    # Step 1: Validate environment
    validation_results = validate_killoran_environment()
    
    if not validation_results['overall_success']:
        print("\nEnvironment validation failed. Cannot proceed with initialization.")
        return False
    
    # Step 2: Test functionality
    functionality_success = test_killoran_functionality()
    
    if not functionality_success:
        print("\nFunctionality test failed. Environment may not work correctly.")
        return False
    
    # Step 3: Final setup
    print("\nFinal Setup...")
    print("-" * 30)
    
    try:
        # Enable clean training
        from src.utils.warning_suppression import enable_clean_training
        enable_clean_training()
        print("✓ Warning suppression enabled")
        
        # Test import from main package
        from src import KilloranCVQNN, KilloranQGANTrainer, KilloranTrainerConfig
        print("✓ Main package imports successful")
        
        print("\n" + "=" * 70)
        print("✓ KILLORAN CV-QNN ENVIRONMENT INITIALIZED SUCCESSFULLY!")
        print("✓ Ready for training with: python train_killoran_qgan.py")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"✗ Final setup failed: {str(e)}")
        return False


def print_usage_examples():
    """Print usage examples for the Killoran CV-QNN."""
    print("\nUSAGE EXAMPLES:")
    print("=" * 50)
    print("# Train with bimodal target (WITH Kerr gate)")
    print("python train_killoran_qgan.py --epochs 300 --use-kerr --target-type bimodal")
    print()
    print("# Train with bimodal target (WITHOUT Kerr gate - should fail)")
    print("python train_killoran_qgan.py --epochs 300 --no-kerr --target-type bimodal")
    print()
    print("# Compare both architectures")
    print("python train_killoran_qgan.py --epochs 300 --use-kerr --exp-name bimodal_with_kerr")
    print("python train_killoran_qgan.py --epochs 300 --no-kerr --exp-name bimodal_no_kerr")
    print()
    print("# Test import checker")
    print("python -m src.utils.import_checker")
    print()
    print("# Initialize environment")
    print("python -c \"from src.utils.killoran_init import initialize_killoran_environment; initialize_killoran_environment()\"")
    print("=" * 50)


if __name__ == "__main__":
    # Run complete initialization
    success = initialize_killoran_environment()
    
    if success:
        print_usage_examples()
    else:
        print("\nInitialization failed. Please check the error messages above.")
        sys.exit(1)