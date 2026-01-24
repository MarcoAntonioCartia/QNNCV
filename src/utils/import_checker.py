"""
Import Checker Utility
======================

Utilities for checking and validating imports across the QNNCV package.
Provides functions to verify that all required dependencies are available
and that imports work correctly.
"""

import sys
import importlib
import traceback
from typing import List, Dict, Tuple, Optional


class ImportChecker:
    """Comprehensive import checking for QNNCV dependencies."""
    
    def __init__(self):
        self.results = {}
        self.missing_packages = []
        self.failed_imports = []
    
    def check_package_import(self, package_name: str, module_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if a package can be imported.
        
        Args:
            package_name: Name of the package to check
            module_path: Specific module path to import (optional)
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if module_path:
                importlib.import_module(module_path)
            else:
                importlib.import_module(package_name)
            return True, f"✓ {package_name} imported successfully"
        except ImportError as e:
            error_msg = f"✗ {package_name} import failed: {str(e)}"
            self.missing_packages.append(package_name)
            return False, error_msg
        except Exception as e:
            error_msg = f"✗ {package_name} unexpected error: {str(e)}"
            self.failed_imports.append(package_name)
            return False, error_msg
    
    def check_qnn_cv_dependencies(self) -> Dict[str, Tuple[bool, str]]:
        """Check all QNNCV dependencies."""
        dependencies = {
            'numpy': 'numpy',
            'tensorflow': 'tensorflow',
            'strawberryfields': 'strawberryfields',
            'scipy': 'scipy',
            'matplotlib': 'matplotlib',
            'argparse': 'argparse',
            'os': 'os',
            'sys': 'sys',
            'typing': 'typing',
            'dataclasses': 'dataclasses'
        }
        
        results = {}
        for package_name, module_path in dependencies.items():
            success, message = self.check_package_import(package_name, module_path)
            results[package_name] = (success, message)
            print(message)
        
        return results
    
    def check_qnn_cv_modules(self) -> Dict[str, Tuple[bool, str]]:
        """Check all QNNCV internal modules."""
        modules = {
            'src.utils.warning_suppression': 'src.utils.warning_suppression',
            'src.utils.compatibility': 'src.utils.compatibility',
            'src.utils.scipy_compat': 'src.utils.scipy_compat',
            'src.models.generators.quantum_sf_generator': 'src.models.generators.quantum_sf_generator',
            'src.models.generators.quantum_distribution_generator': 'src.models.generators.quantum_distribution_generator',
            'src.models.discriminators.quantum_sf_discriminator': 'src.models.discriminators.quantum_sf_discriminator',
            'src.models.discriminators.classical_discriminator': 'src.models.discriminators.classical_discriminator',
            'src.models.discriminators.distribution_discriminator': 'src.models.discriminators.distribution_discriminator',
            'src.training.trainer': 'src.training.trainer',
            'src.training.qgan_sf_trainer': 'src.training.qgan_sf_trainer',
            'src.training.distribution_trainer': 'src.training.distribution_trainer',
            'src.training.killoran_trainer': 'src.training.killoran_trainer',
            'src.models.generators.killoran_cvqnn': 'src.models.generators.killoran_cvqnn'
        }
        
        results = {}
        for module_name, module_path in modules.items():
            success, message = self.check_package_import(module_name, module_path)
            results[module_name] = (success, message)
            print(message)
        
        return results
    
    def check_killoran_files(self) -> Dict[str, Tuple[bool, str]]:
        """Check the specific Killoran files."""
        killoran_files = {
            'killoran_cvqnn': 'src.models.generators.killoran_cvqnn',
            'killoran_trainer': 'src.training.killoran_trainer',
            'train_killoran_qgan': 'train_killoran_qgan'
        }
        
        results = {}
        for file_name, module_path in killoran_files.items():
            success, message = self.check_package_import(file_name, module_path)
            results[file_name] = (success, message)
            print(message)
        
        return results
    
    def print_summary(self):
        """Print a summary of import checking results."""
        print("\n" + "="*60)
        print("IMPORT CHECK SUMMARY")
        print("="*60)
        
        if self.missing_packages:
            print(f"\nMissing packages ({len(self.missing_packages)}):")
            for pkg in self.missing_packages:
                print(f"  - {pkg}")
        
        if self.failed_imports:
            print(f"\nFailed imports ({len(self.failed_imports)}):")
            for imp in self.failed_imports:
                print(f"  - {imp}")
        
        if not self.missing_packages and not self.failed_imports:
            print("\n✓ All imports successful!")
        else:
            print(f"\n✗ {len(self.missing_packages + self.failed_imports)} issues found")
        
        print("="*60)
    
    def validate_environment(self) -> bool:
        """Validate the complete QNNCV environment."""
        print("Checking QNNCV Dependencies...")
        print("-" * 40)
        self.check_qnn_cv_dependencies()
        
        print("\nChecking QNNCV Modules...")
        print("-" * 40)
        self.check_qnn_cv_modules()
        
        print("\nChecking Killoran Files...")
        print("-" * 40)
        self.check_killoran_files()
        
        self.print_summary()
        
        return len(self.missing_packages + self.failed_imports) == 0


def check_imports() -> bool:
    """Convenience function to check all imports."""
    checker = ImportChecker()
    return checker.validate_environment()


def init_killoran_environment():
    """Initialize the Killoran CV-QNN environment with proper imports."""
    print("Initializing Killoran CV-QNN Environment...")
    print("-" * 50)
    
    # Check basic dependencies
    try:
        import numpy as np
        import tensorflow as tf
        import strawberryfields as sf
        print("✓ Basic dependencies loaded")
    except ImportError as e:
        print(f"✗ Basic dependencies failed: {e}")
        return False
    
    # Check QNNCV utilities
    try:
        from src.utils.warning_suppression import enable_clean_training
        enable_clean_training()
        print("✓ Warning suppression enabled")
    except ImportError:
        print("⚠ Warning suppression not available")
    
    # Check QNNCV modules
    try:
        from src.models.generators.killoran_cvqnn import KilloranCVQNN
        from src.training.killoran_trainer import KilloranQGANTrainer, KilloranTrainerConfig
        print("✓ Killoran modules imported successfully")
    except ImportError as e:
        print(f"✗ Killoran modules failed: {e}")
        return False
    
    # Test basic functionality
    try:
        # Create a simple generator to test
        gen = KilloranCVQNN(n_layers=2, cutoff_dim=10, use_kerr=True)
        z = tf.random.normal([1, gen.latent_dim])
        P_x = gen.generate(z)
        print(f"✓ Basic functionality test passed (output shape: {P_x.shape})")
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    print("\n✓ Killoran CV-QNN environment initialized successfully!")
    return True


if __name__ == "__main__":
    # Run import checking
    success = check_imports()
    
    if success:
        print("\nEnvironment validation successful!")
        # Try to initialize Killoran environment
        init_killoran_environment()
    else:
        print("\nEnvironment validation failed!")
        sys.exit(1)