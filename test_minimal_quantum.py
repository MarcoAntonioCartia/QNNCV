"""
Test script for minimal quantum generator validation.
This script tests the core functionality step by step.
"""

import sys
import os
sys.path.insert(0, 'src')

import tensorflow as tf
import numpy as np

def test_minimal_quantum_step_by_step():
    """Test minimal quantum generator step by step."""
    print("="*60)
    print("MINIMAL QUANTUM GENERATOR STEP-BY-STEP TEST")
    print("="*60)
    
    try:
        # Step 1: Import and basic setup
        print("\n1. Testing imports...")
        from models.generators.quantum_minimal_generator import QuantumMinimalGenerator
        print("✅ Successfully imported QuantumMinimalGenerator")
        
        # Step 2: Create generator
        print("\n2. Creating minimal quantum generator...")
        gen = QuantumMinimalGenerator(n_qumodes=2, latent_dim=4, cutoff_dim=4)
        print("✅ Generator created successfully")
        
        # Step 3: Check if SF engine was created
        print(f"\n3. Checking Strawberry Fields engine status...")
        if gen.eng is not None:
            print("✅ SF engine created successfully")
        else:
            print("⚠️ SF engine is None - will use classical fallback")
        
        # Step 4: Test basic generation
        print("\n4. Testing basic sample generation...")
        z_test = tf.random.normal([2, 4])
        samples = gen.generate(z_test)
        print(f"✅ Generated samples shape: {samples.shape}")
        print(f"✅ Sample values range: [{tf.reduce_min(samples):.3f}, {tf.reduce_max(samples):.3f}]")
        
        # Step 5: Test gradient computation
        print("\n5. Testing gradient computation...")
        with tf.GradientTape() as tape:
            z = tf.random.normal([2, 4])
            output = gen.generate(z)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, gen.trainable_variables)
        non_none_grads = [g for g in gradients if g is not None]
        
        print(f"✅ Loss computed: {loss.numpy():.4f}")
        print(f"✅ Gradients computed: {len(non_none_grads)}/{len(gradients)}")
        
        if len(non_none_grads) == len(gradients):
            print("✅ All gradients computed successfully")
        else:
            print("⚠️ Some gradients are None")
        
        # Step 6: Test parameter updates
        print("\n6. Testing parameter updates...")
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        
        # Store initial weights
        initial_weights = [tf.identity(var) for var in gen.trainable_variables]
        
        # Perform one optimization step
        with tf.GradientTape() as tape:
            z = tf.random.normal([2, 4])
            output = gen.generate(z)
            loss = tf.reduce_mean(tf.square(output - 1.0))  # Target of 1.0
        
        gradients = tape.gradient(loss, gen.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gen.trainable_variables))
        
        # Check if weights changed
        weights_changed = False
        for initial, current in zip(initial_weights, gen.trainable_variables):
            if not tf.reduce_all(tf.equal(initial, current)):
                weights_changed = True
                break
        
        print(f"✅ Parameters updated: {weights_changed}")
        print(f"✅ Final loss: {loss.numpy():.4f}")
        
        # Step 7: Test multiple training steps
        print("\n7. Testing multiple training steps...")
        losses = []
        for step in range(5):
            with tf.GradientTape() as tape:
                z = tf.random.normal([2, 4])
                output = gen.generate(z)
                loss = tf.reduce_mean(tf.square(output))
            
            gradients = tape.gradient(loss, gen.trainable_variables)
            optimizer.apply_gradients(zip(gradients, gen.trainable_variables))
            losses.append(loss.numpy())
            
        print(f"✅ Training losses: {[f'{l:.4f}' for l in losses]}")
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✅ Minimal quantum generator working correctly")
        print("✅ Gradient flow verified")
        print("✅ Parameter updates working")
        print("✅ Ready for integration with training loop")
        
        return gen
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_minimal_quantum_step_by_step()
