#!/usr/bin/env python3
"""
Test script to verify Strawberry Fields TensorFlow backend GPU usage
"""

import tensorflow as tf
import strawberryfields as sf
import numpy as np
import sys
import os

def test_tf_backend_gpu():
    """Test if SF TensorFlow backend can use GPU"""
    print("=" * 60)
    print("TESTING STRAWBERRY FIELDS GPU USAGE")
    print("=" * 60)
    
    # Check TensorFlow GPU
    print("1. TensorFlow GPU Status:")
    print(f"   TF version: {tf.__version__}")
    print(f"   GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Enable device placement logging
    tf.debugging.set_log_device_placement(True)
    
    # Test SF TensorFlow backend
    print("\n2. Testing SF TensorFlow Backend:")
    
    try:
        # Create SF program with TF backend
        prog = sf.Program(2)
        
        # Create engine with TF backend and GPU options
        eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": 8,
            "pure": True
        })
        
        print("   ✓ SF TensorFlow engine created")
        
        # Build a simple quantum program
        with prog.context as q:
            sf.ops.Sgate(0.5) | q[0]
            sf.ops.Sgate(0.3) | q[1]
            sf.ops.BSgate(0.2, 0.1) | (q[0], q[1])
        
        print("   ✓ Quantum program built")
        
        # Run on GPU
        with tf.device('/GPU:0'):
            print("   Running quantum program on GPU...")
            result = eng.run(prog)
            state = result.state
            ket = state.ket()
            
            print(f"   ✓ Program executed successfully")
            print(f"   State shape: {ket.shape}")
            print(f"   State device: {ket.device}")
            print(f"   State norm: {tf.reduce_sum(tf.abs(ket)**2):.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ SF TensorFlow backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generator_gpu():
    """Test the actual generator on GPU"""
    print("\n3. Testing QuantumSFGenerator on GPU:")
    
    # Add src to path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from models.generators.quantum_sf_generator import QuantumSFGenerator
        
        # Force GPU device
        with tf.device('/GPU:0'):
            print("   Creating generator on GPU...")
            
            generator = QuantumSFGenerator(
                n_modes=2,
                latent_dim=2,
                layers=1,
                cutoff_dim=6
            )
            
            print("   ✓ Generator created")
            
            # Test generation
            z_test = tf.random.normal([2, 2])
            print(f"   Input device: {z_test.device}")
            
            samples = generator.generate(z_test)
            print(f"   ✓ Generation successful")
            print(f"   Output shape: {samples.shape}")
            print(f"   Output device: {samples.device}")
            
            # Check if any operations actually used GPU
            print("   Checking GPU memory usage...")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Generator GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu_memory():
    """Check actual GPU memory usage"""
    print("\n4. GPU Memory Check:")
    
    try:
        # Get GPU memory info
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu = gpus[0]
            memory_info = tf.config.experimental.get_memory_info(gpu.name.replace('/physical_device:', ''))
            print(f"   Current GPU memory: {memory_info['current'] / 1024**3:.2f} GB")
            print(f"   Peak GPU memory: {memory_info['peak'] / 1024**3:.2f} GB")
            
            if memory_info['current'] > 0:
                print("   ✓ GPU memory is being used!")
                return True
            else:
                print("   ⚠ GPU memory usage is 0 - operations may be on CPU")
                return False
        else:
            print("   ✗ No GPU available")
            return False
            
    except Exception as e:
        print(f"   ⚠ Could not check GPU memory: {e}")
        return False

def main():
    """Main test function"""
    print("GPU Strawberry Fields Test")
    print("This script tests if SF can actually use GPU with TensorFlow backend")
    print()
    
    # Run tests
    tf_backend_ok = test_tf_backend_gpu()
    generator_ok = test_generator_gpu()
    gpu_memory_ok = check_gpu_memory()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if tf_backend_ok:
        print("✓ SF TensorFlow Backend: WORKING")
    else:
        print("✗ SF TensorFlow Backend: FAILED")
    
    if generator_ok:
        print("✓ Generator Creation: WORKING")
    else:
        print("✗ Generator Creation: FAILED")
    
    if gpu_memory_ok:
        print("✓ GPU Memory Usage: DETECTED")
    else:
        print("⚠ GPU Memory Usage: NOT DETECTED")
    
    overall_success = tf_backend_ok and generator_ok
    
    if overall_success and gpu_memory_ok:
        print("\n🎉 SUCCESS: Strawberry Fields is using GPU!")
    elif overall_success:
        print("\n⚠ PARTIAL: SF works but may not be using GPU")
    else:
        print("\n❌ FAILED: SF TensorFlow backend issues")
    
    return overall_success

if __name__ == "__main__":
    main()
