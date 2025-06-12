"""
Compatibility Module for QNNCV
=============================

This module provides compatibility patches for modern package versions:
- NumPy 2.0+ compatibility (restored deprecated aliases)
- SciPy 1.14+ compatibility (simps -> simpson transition)
- Strawberry Fields compatibility with modern packages
- Automatic import hooks to ensure patches are applied before problematic imports

Usage:
    from utils.compatibility import apply_all_compatibility_patches
    apply_all_compatibility_patches()

Or import specific patches:
    from utils.compatibility import apply_numpy_2_compatibility, apply_scipy_compatibility

Auto-patching:
    The module automatically applies critical patches when imported to prevent
    import-time failures with packages like Strawberry Fields.
"""

import sys
import warnings
import importlib.util
from typing import Optional, Dict, Any, Set

# Global flag to track if patches have been applied
_patches_applied: Set[str] = set()


def get_package_version(package_name: str) -> Optional[str]:
    """Get the version of an installed package."""
    try:
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except:
        try:
            module = __import__(package_name)
            return getattr(module, '__version__', 'unknown')
        except:
            return None


def apply_scipy_compatibility_immediate() -> bool:
    """
    Apply SciPy compatibility patches immediately and silently.
    
    This is designed to be called before any problematic imports.
    Returns True if successful, False otherwise.
    """
    if 'scipy_immediate' in _patches_applied:
        return True
    
    try:
        import scipy.integrate
        
        # Check if simps function exists
        if not hasattr(scipy.integrate, 'simps'):
            if hasattr(scipy.integrate, 'simpson'):
                # Create simps as an alias to simpson
                def simps(y, x=None, dx=1.0, axis=-1, even='avg'):
                    """
                    Compatibility wrapper for scipy.integrate.simpson.
                    
                    This function provides backward compatibility for the deprecated
                    scipy.integrate.simps function by wrapping scipy.integrate.simpson.
                    """
                    # simpson doesn't have 'even' parameter, so we ignore it for compatibility
                    return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
                
                # Add the function to scipy.integrate
                scipy.integrate.simps = simps
                _patches_applied.add('scipy_immediate')
                return True
            else:
                return False
        else:
            _patches_applied.add('scipy_immediate')
            return True
        
    except Exception:
        return False


def install_import_hook():
    """
    Install an import hook that applies SciPy patches before Strawberry Fields imports.
    """
    if 'import_hook' in _patches_applied:
        return
    
    # Store the original __import__ function
    # __builtins__ can be either a module or a dict depending on context
    if isinstance(__builtins__, dict):
        original_import = __builtins__['__import__']
    else:
        original_import = __builtins__.__import__
    
    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # If someone is trying to import strawberryfields, apply SciPy patch first
        if name == 'strawberryfields' or (fromlist and 'strawberryfields' in str(fromlist)):
            apply_scipy_compatibility_immediate()
        
        # Call the original import
        return original_import(name, globals, locals, fromlist, level)
    
    # Replace the built-in __import__ with our patched version
    if isinstance(__builtins__, dict):
        __builtins__['__import__'] = patched_import
    else:
        __builtins__.__import__ = patched_import
    _patches_applied.add('import_hook')


def apply_numpy_2_compatibility() -> bool:
    """
    Apply NumPy 2.0+ compatibility patches.
    
    NumPy 2.0 removed deprecated aliases like np.bool, np.int, etc.
    This function restores them for backward compatibility.
    
    Returns:
        bool: True if patches were applied successfully
    """
    try:
        import numpy as np
        
        numpy_version = get_package_version('numpy')
        print(f"  NumPy version: {numpy_version}")
        
        # Check if we need to apply patches (NumPy 2.0+)
        if numpy_version and numpy_version.startswith('2.'):
            print("  Applying NumPy 2.0+ compatibility patches...")
            
            # Restore deprecated aliases that were removed in NumPy 2.0
            deprecated_aliases = {
                'bool': bool,
                'int': int,
                'float': float,
                'complex': complex,
                'object': object,
                'bytes': bytes,
                'str': str,
                'unicode': str,  # Python 3 compatibility
            }
            
            patches_applied = []
            for alias_name, alias_target in deprecated_aliases.items():
                if not hasattr(np, alias_name):
                    setattr(np, alias_name, alias_target)
                    patches_applied.append(alias_name)
            
            if patches_applied:
                print(f"    ✓ Restored NumPy aliases: {', '.join(patches_applied)}")
            else:
                print("    ✓ NumPy aliases already available")
                
            # Additional NumPy 2.0 compatibility fixes
            if not hasattr(np, 'int0'):
                np.int0 = np.intp  # type: ignore
            if not hasattr(np, 'uint0'):
                np.uint0 = np.uintp  # type: ignore
                
            return True
        else:
            print("    ✓ NumPy version < 2.0, no patches needed")
            return True
            
    except Exception as e:
        print(f"    ✗ NumPy compatibility patch failed: {e}")
        return False


def apply_scipy_compatibility() -> bool:
    """
    Apply SciPy 1.14+ compatibility patches.
    
    SciPy 1.14+ removed scipy.integrate.simps and replaced it with simpson.
    This function restores the simps function for backward compatibility.
    
    Returns:
        bool: True if patches were applied successfully
    """
    try:
        import scipy
        import scipy.integrate
        
        scipy_version = get_package_version('scipy')
        print(f"  SciPy version: {scipy_version}")
        
        # Check if simps function exists
        if not hasattr(scipy.integrate, 'simps'):
            print("  SciPy simps function not found, applying compatibility patch...")
            
            if hasattr(scipy.integrate, 'simpson'):
                # Create simps as an alias to simpson
                def simps(y, x=None, dx=1.0, axis=-1, even='avg'):
                    """
                    Compatibility wrapper for scipy.integrate.simpson.
                    
                    This function provides backward compatibility for the deprecated
                    scipy.integrate.simps function by wrapping scipy.integrate.simpson.
                    """
                    # simpson doesn't have 'even' parameter, so we ignore it for compatibility
                    return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
                
                # Add the function to scipy.integrate
                scipy.integrate.simps = simps
                print("    ✓ Added simps as alias to simpson")
                
            else:
                print("    ⚠ Neither simps nor simpson found in scipy.integrate")
                return False
        else:
            print("    ✓ SciPy simps function already available")
        
        return True
        
    except Exception as e:
        print(f"    ✗ SciPy compatibility patch failed: {e}")
        return False


def apply_strawberryfields_compatibility() -> bool:
    """
    Apply Strawberry Fields compatibility patches.
    
    Ensures Strawberry Fields works with modern NumPy and SciPy versions.
    
    Returns:
        bool: True if patches were applied successfully
    """
    try:
        # First ensure SciPy compatibility (SF depends on simps)
        if not apply_scipy_compatibility():
            return False
        
        # Import Strawberry Fields to trigger any import-time issues
        try:
            import strawberryfields as sf
            sf_version = get_package_version('strawberryfields')
            print(f"  Strawberry Fields version: {sf_version}")
            print("    ✓ Strawberry Fields imports successfully")
            
            # Test basic functionality
            try:
                prog = sf.Program(1)
                print("    ✓ Strawberry Fields Program creation works")
            except Exception as e:
                print(f"    ⚠ Strawberry Fields Program creation failed: {e}")
                
            return True
            
        except ImportError as e:
            print(f"    ✗ Strawberry Fields import failed: {e}")
            return False
            
    except Exception as e:
        print(f"    ✗ Strawberry Fields compatibility check failed: {e}")
        return False


def apply_tensorflow_compatibility() -> bool:
    """
    Apply TensorFlow compatibility patches and optimizations.
    
    Returns:
        bool: True if patches were applied successfully
    """
    try:
        import tensorflow as tf
        
        tf_version = get_package_version('tensorflow')
        print(f"  TensorFlow version: {tf_version}")
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Configure GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"    ✓ {len(gpus)} GPU(s) detected")
            try:
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("    ✓ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"    ⚠ GPU configuration warning: {e}")
        else:
            print("    ⚠ No GPU detected, using CPU")
        
        print("    ✓ TensorFlow warnings suppressed")
        return True
        
    except Exception as e:
        print(f"    ✗ TensorFlow compatibility patch failed: {e}")
        return False


def apply_general_compatibility() -> bool:
    """
    Apply general compatibility patches and warning suppressions.
    
    Returns:
        bool: True if patches were applied successfully
    """
    try:
        # Suppress common warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning, module='strawberryfields')
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        
        print("    ✓ General warnings suppressed")
        return True
        
    except Exception as e:
        print(f"    ✗ General compatibility patch failed: {e}")
        return False


def validate_compatibility_patches() -> Dict[str, bool]:
    """
    Validate that all compatibility patches are working correctly.
    
    Returns:
        Dict[str, bool]: Results of validation tests
    """
    results = {}
    
    print("Validating compatibility patches...")
    
    # Test NumPy compatibility
    try:
        import numpy as np
        # Test that deprecated aliases work (only if they exist)
        if hasattr(np, 'bool'):
            test_bool = np.bool(True)  # type: ignore
        if hasattr(np, 'int'):
            test_int = np.int(42)  # type: ignore
        # Test basic NumPy functionality
        test_array = np.array([1, 2, 3])
        results['numpy'] = True
        print("    ✓ NumPy compatibility validated")
    except Exception as e:
        results['numpy'] = False
        print(f"    ✗ NumPy compatibility validation failed: {e}")
    
    # Test SciPy compatibility
    try:
        import scipy.integrate
        # Test that simps function works
        import numpy as np
        x = np.linspace(0, 1, 11)
        y = x**2
        result = scipy.integrate.simps(y, x)
        results['scipy'] = True
        print("    ✓ SciPy compatibility validated")
    except Exception as e:
        results['scipy'] = False
        print(f"    ✗ SciPy compatibility validation failed: {e}")
    
    # Test Strawberry Fields compatibility
    try:
        import strawberryfields as sf
        prog = sf.Program(1)
        results['strawberryfields'] = True
        print("    ✓ Strawberry Fields compatibility validated")
    except Exception as e:
        results['strawberryfields'] = False
        print(f"    ✗ Strawberry Fields compatibility validation failed: {e}")
    
    # Test TensorFlow compatibility
    try:
        import tensorflow as tf
        # Simple TensorFlow operation
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        results['tensorflow'] = True
        print("    ✓ TensorFlow compatibility validated")
    except Exception as e:
        results['tensorflow'] = False
        print(f"    ✗ TensorFlow compatibility validation failed: {e}")
    
    return results


def apply_all_compatibility_patches() -> bool:
    """
    Apply all compatibility patches in the correct order.
    
    Returns:
        bool: True if all patches were applied successfully
    """
    print("=" * 60)
    print("APPLYING COMPATIBILITY PATCHES")
    print("=" * 60)
    
    success = True
    
    # Apply patches in order of dependency
    print("1. Applying NumPy 2.0+ compatibility...")
    if not apply_numpy_2_compatibility():
        success = False
    
    print("\n2. Applying SciPy 1.14+ compatibility...")
    if not apply_scipy_compatibility():
        success = False
    
    print("\n3. Applying TensorFlow compatibility...")
    if not apply_tensorflow_compatibility():
        success = False
    
    print("\n4. Applying Strawberry Fields compatibility...")
    if not apply_strawberryfields_compatibility():
        success = False
    
    print("\n5. Applying general compatibility...")
    if not apply_general_compatibility():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL COMPATIBILITY PATCHES APPLIED SUCCESSFULLY")
    else:
        print("✗ SOME COMPATIBILITY PATCHES FAILED")
    print("=" * 60)
    
    # Validate patches
    print("\nValidating patches...")
    validation_results = validate_compatibility_patches()
    
    all_valid = all(validation_results.values())
    if all_valid:
        print("\n✓ All compatibility patches validated successfully!")
    else:
        failed_patches = [name for name, result in validation_results.items() if not result]
        print(f"\n⚠ Some patches failed validation: {', '.join(failed_patches)}")
    
    return success and all_valid


def auto_apply_critical_patches():
    """
    Automatically apply critical patches that must be applied before imports.
    This is called when the module is imported.
    """
    if 'auto_critical' in _patches_applied:
        return
    
    # Apply SciPy compatibility immediately and silently
    apply_scipy_compatibility_immediate()
    
    # Install import hook for future imports
    install_import_hook()
    
    _patches_applied.add('auto_critical')


# Auto-apply critical patches when module is imported
auto_apply_critical_patches()
