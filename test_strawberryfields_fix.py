#!/usr/bin/env python3
"""
Test script to verify Strawberry Fields import fix
=================================================

This script tests whether the SciPy compatibility patches work correctly
to allow Strawberry Fields to import without errors.

Usage:
    python test_strawberryfields_fix.py
"""

import sys
import os

def test_scipy_patch():
    """Test that SciPy patch is applied correctly."""
    print("Testing SciPy compatibility patch...")
    
    try:
        import scipy.integrate
        
        # Check if simps function exists
        if hasattr(scipy.integrate, 'simps'):
            print("  ✓ scipy.integrate.simps is available")
            
            # Test that it works
            import numpy as np
            x = np.linspace(0, 1, 11)
            y = x**2
            result = scipy.integrate.simps(y, x)
            print(f"  ✓ simps function works: result = {result:.4f}")
            return True
        else:
            print("  ✗ scipy.integrate.simps is NOT available")
            return False
            
    except Exception as e:
        print(f"  ✗ SciPy test failed: {e}")
        return False


def test_strawberryfields_import():
    """Test that Strawberry Fields can be imported."""
    print("\nTesting Strawberry Fields import...")
    
    try:
        import strawberryfields as sf
        print(f"  ✓ Strawberry Fields imported successfully (version: {sf.__version__})")
        
        # Test basic functionality
        try:
            prog = sf.Program(1)
            print("  ✓ Strawberry Fields Program creation works")
            return True
        except Exception as e:
            print(f"  ⚠ Strawberry Fields Program creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"  ✗ Strawberry Fields import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Strawberry Fields test failed: {e}")
        return False


def test_with_auto_patching():
    """Test with automatic patching enabled."""
    print("=" * 60)
    print("TESTING WITH AUTO-PATCHING")
    print("=" * 60)
    
    # Add src to path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"Added {src_path} to Python path")
    
    # Import utils to trigger auto-patching
    try:
        print("\nImporting utils package (triggers auto-patching)...")
        import utils
        print("  ✓ Utils package imported successfully")
        
        # Test SciPy patch
        scipy_success = test_scipy_patch()
        
        # Test Strawberry Fields import
        sf_success = test_strawberryfields_import()
        
        return scipy_success and sf_success
        
    except Exception as e:
        print(f"  ✗ Auto-patching test failed: {e}")
        return False


def test_manual_patching():
    """Test with manual patching."""
    print("\n" + "=" * 60)
    print("TESTING WITH MANUAL PATCHING")
    print("=" * 60)
    
    try:
        # Import compatibility module and apply patches manually
        print("\nApplying compatibility patches manually...")
        from utils.compatibility import apply_scipy_compatibility_immediate
        
        success = apply_scipy_compatibility_immediate()
        if success:
            print("  ✓ Manual SciPy patch applied successfully")
        else:
            print("  ✗ Manual SciPy patch failed")
            return False
        
        # Test SciPy patch
        scipy_success = test_scipy_patch()
        
        # Test Strawberry Fields import
        sf_success = test_strawberryfields_import()
        
        return scipy_success and sf_success
        
    except Exception as e:
        print(f"  ✗ Manual patching test failed: {e}")
        return False


def main():
    """Main test function."""
    print("STRAWBERRY FIELDS COMPATIBILITY TEST")
    print("=" * 60)
    print("This script tests whether the SciPy compatibility patches")
    print("allow Strawberry Fields to import without errors.")
    print()
    
    # Test 1: Auto-patching
    auto_success = test_with_auto_patching()
    
    # Test 2: Manual patching (in case auto-patching didn't work)
    manual_success = test_manual_patching()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if auto_success:
        print("✓ AUTO-PATCHING: WORKING")
    else:
        print("✗ AUTO-PATCHING: FAILED")
    
    if manual_success:
        print("✓ MANUAL PATCHING: WORKING")
    else:
        print("✗ MANUAL PATCHING: FAILED")
    
    overall_success = auto_success or manual_success
    
    if overall_success:
        print("\n🎉 STRAWBERRY FIELDS COMPATIBILITY FIX: SUCCESS!")
        print("The SciPy compatibility patches are working correctly.")
        print("Strawberry Fields should now import without errors.")
    else:
        print("\n❌ STRAWBERRY FIELDS COMPATIBILITY FIX: FAILED!")
        print("The patches are not working. Check the error messages above.")
    
    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
