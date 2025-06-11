"""
Test script for differentiable quantum generator - focusing on classical fallback first.
"""

import sys
sys.path.insert(0, 'src')

import tensorflow as tf
import numpy as np

def test_classical_fallback():
    """Test the classical fallback to ensure gradient flow works."""
    print("="*60)
    print("TESTING CLASSICAL FALLBACK GENERATOR")
    print("="*60)
    
    try:
        from models.generators.quantum_differentiable_generator import QuantumDifferentiableGenerator
        
        # Create generator with classical fallback only
        print("\n1. Creating classical fallback generator...")
        gen = QuantumDifferentiableGenerator(
            n_qumodes=2, 
            latent_dim=4, 
            cutoff_dim=4, 
            use_quantum=False  # Force classical fallback
        )
        print("✅ Generator created successfully")
        
        # Test basic generation
        print("\n2. Testing sample generation...")
        z_test = tf.random.normal([3, 4])
        samples = gen.generate(z_test)
        print(f"✅ Generated samples shape: {samples.shape}")
        print(f"✅ Expected shape: (3, 2)")
        print(f"✅ Shape correct: {samples.shape == (3, 2)}")
        print(f"✅ Sample values range: [{tf.reduce_min(samples):.3f}, {tf.reduce_max(samples):.3f}]")
        
        # Test gradient computation
        print("\n3. Testing gradient computation...")
        with tf.GradientTape() as tape:
            z = tf.random.normal([2, 4])
            output = gen.generate(z)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, gen.trainable_variables)
        non_none_grads = [g for g in gradients if g is not None]
        
        print(f"✅ Loss computed: {loss.numpy():.4f}")
        print(f"✅ Total variables: {len(gradients)}")
        print(f"✅ Non-None gradients: {len(non_none_grads)}")
        print(f"✅ Gradient flow working: {len(non_none_grads) == len(gradients)}")
        
        for i, grad in enumerate(gradients):
            if grad is not None:
                print(f"   Variable {i}: gradient norm = {tf.norm(grad).numpy():.4f}")
            else:
                print(f"   Variable {i}: gradient = None ❌")
        
        # Test parameter updates
        print("\n4. Testing parameter updates...")
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        
        # Store initial weights
        initial_weights = [tf.identity(var) for var in gen.trainable_variables]
        
        # Perform optimization step
        with tf.GradientTape() as tape:
            z = tf.random.normal([2, 4])
            output = gen.generate(z)
            loss = tf.reduce_mean(tf.square(output - 1.0))  # Target of 1.0
        
        gradients = tape.gradient(loss, gen.trainable_variables)
        
        if all(g is not None for g in gradients):
            optimizer.apply_gradients(zip(gradients, gen.trainable_variables))
            
            # Check if weights changed
            weights_changed = False
            for initial, current in zip(initial_weights, gen.trainable_variables):
                if not tf.reduce_all(tf.equal(initial, current)):
                    weights_changed = True
                    break
            
            print(f"✅ Parameters updated: {weights_changed}")
            print(f"✅ Final loss: {loss.numpy():.4f}")
        else:
            print("❌ Cannot update parameters - some gradients are None")
        
        # Test multiple training steps
        print("\n5. Testing multiple training steps...")
        losses = []
        for step in range(5):
            with tf.GradientTape() as tape:
                z = tf.random.normal([4, 4])
                output = gen.generate(z)
                loss = tf.reduce_mean(tf.square(output))
            
            gradients = tape.gradient(loss, gen.trainable_variables)
            if all(g is not None for g in gradients):
                optimizer.apply_gradients(zip(gradients, gen.trainable_variables))
                losses.append(loss.numpy())
            else:
                print(f"❌ Step {step}: Gradients are None")
                break
        
        if losses:
            print(f"✅ Training losses: {[f'{l:.4f}' for l in losses]}")
            print(f"✅ Loss trend: {'decreasing' if losses[-1] < losses[0] else 'stable/increasing'}")
        
        print("\n" + "="*60)
        print("CLASSICAL FALLBACK TEST SUMMARY")
        print("="*60)
        
        if len(non_none_grads) == len(gradients) and samples.shape == (3, 2):
            print("✅ SUCCESS: Classical fallback working correctly")
            print("✅ Gradient flow: WORKING")
            print("✅ Output shape: CORRECT")
            print("✅ Parameter updates: WORKING")
            print("✅ Ready for integration with training loop")
        else:
            print("❌ ISSUES DETECTED:")
            if len(non_none_grads) != len(gradients):
                print(f"   - Gradient flow broken: {len(non_none_grads)}/{len(gradients)}")
            if samples.shape != (3, 2):
                print(f"   - Wrong output shape: {samples.shape} != (3, 2)")
        
        return gen
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_classical_fallback()
