"""
Compatibility Module for QNNCV

This module provides compatibility patches for different Python environments,
especially for Google Colab with NumPy 2.0+ and SciPy 1.15+.
"""

import numpy as np
import scipy.integrate
import sys
import warnings
from typing import Dict, List, Tuple, Optional


def apply_numpy_compatibility_patches() -> bool:
    """
    Apply NumPy compatibility patches for NumPy 2.0+.
    
    Returns:
        bool: True if patches were applied successfully
    """
    print("Applying NumPy compatibility patches...")
    
    try:
        # Check NumPy version
        numpy_version = np.__version__
        print(f"  NumPy version: {numpy_version}")
        
        # Patch for NumPy 2.0+ bool type
        if hasattr(np, 'bool') and not hasattr(np, 'bool_'):
            # In NumPy 2.0+, np.bool is deprecated, use np.bool_
            np.bool = np.bool_  # type: ignore[attr-defined]
            print("  ✓ Applied NumPy bool compatibility patch")
        
        # Patch for array creation functions
        if hasattr(np, 'bool') and not hasattr(np, 'bool_'):
            # Ensure bool_ is available
            if not hasattr(np, 'bool_'):
                np.bool_ = bool
                print("  ✓ Applied NumPy bool_ compatibility patch")
        
        return True
        
    except Exception as e:
        print(f"  ✗ NumPy compatibility patch failed: {e}")
        return False


def apply_scipy_compatibility_patches() -> bool:
    """
    Apply SciPy compatibility patches for SciPy 1.14+.
    
    Returns:
        bool: True if patches were applied successfully
    """
    print("Applying SciPy compatibility patches...")
    
    try:
        # Check SciPy version
        scipy_version = scipy.integrate.__version__ if hasattr(scipy.integrate, '__version__') else "unknown"
        print(f"  SciPy version: {scipy_version}")
        
        # Patch for simps function (removed in SciPy 1.14+)
        if not hasattr(scipy.integrate, 'simps'):
            if hasattr(scipy.integrate, 'simpson'):
                # Create alias from simpson to simps
                scipy.integrate.simps = scipy.integrate.simpson  # type: ignore[attr-defined]
                print("  ✓ Applied SciPy simps compatibility patch")
            else:
                print("  ⚠ SciPy simpson function not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ SciPy compatibility patch failed: {e}")
        return False


def apply_tensorflow_compatibility_patches() -> bool:
    """
    Apply TensorFlow compatibility patches.
    
    Returns:
        bool: True if patches were applied successfully
    """
    print("Applying TensorFlow compatibility patches...")
    
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        tf_version = tf.__version__
        print(f"  TensorFlow version: {tf_version}")
        
        # Suppress common warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Configure TensorFlow for better compatibility
        tf.config.experimental.enable_tensor_float_32_execution(False)
        
        print("  ✓ Applied TensorFlow compatibility patches")
        return True
        
    except ImportError:
        print("  ⚠ TensorFlow not available")
        return False
    except Exception as e:
        print(f"  ✗ TensorFlow compatibility patch failed: {e}")
        return False


def apply_all_compatibility_patches() -> bool:
    """
    Apply all compatibility patches.
    
    Returns:
        bool: True if all patches were applied successfully
    """
    print("Applying all compatibility patches...")
    
    patches = [
        ("NumPy", apply_numpy_compatibility_patches),
        ("SciPy", apply_scipy_compatibility_patches),
        ("TensorFlow", apply_tensorflow_compatibility_patches),
    ]
    
    success_count = 0
    total_count = len(patches)
    
    for name, patch_func in patches:
        try:
            if patch_func():
                success_count += 1
        except Exception as e:
            print(f"  ✗ {name} patch failed with exception: {e}")
    
    print(f"Compatibility patches: {success_count}/{total_count} successful")
    
    return success_count == total_count


def check_environment_compatibility() -> Dict[str, str]:
    """
    Check the current environment for compatibility issues.
    
    Returns:
        Dict[str, str]: Dictionary with version information
    """
    print("Checking environment compatibility...")
    
    env_info = {}
    
    # NumPy
    try:
        import numpy as np
        env_info['numpy'] = np.__version__
    except ImportError:
        env_info['numpy'] = 'Not available'
    
    # SciPy
    try:
        import scipy
        env_info['scipy'] = scipy.__version__
    except ImportError:
        env_info['scipy'] = 'Not available'
    
    # TensorFlow
    try:
        import tensorflow as tf
        env_info['tensorflow'] = tf.__version__
    except ImportError:
        env_info['tensorflow'] = 'Not available'
    
    # Strawberry Fields
    try:
        import strawberryfields as sf
        env_info['strawberryfields'] = sf.__version__
    except ImportError:
        env_info['strawberryfields'] = 'Not available'
    
    # Python
    env_info['python'] = sys.version
    
    print("Environment information:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    return env_info


def enable_clean_training() -> None:
    """
    Enable clean training by suppressing warnings and configuring environment.
    """
    print("Enabling clean training environment...")
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Configure TensorFlow
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("  ✓ TensorFlow configured for clean training")
    except ImportError:
        pass
    
    # Apply compatibility patches
    apply_all_compatibility_patches()
    
    print("Clean training environment enabled!")
