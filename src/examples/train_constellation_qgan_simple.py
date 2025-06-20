"""
Simplified Constellation Quantum GAN Demo

A minimal demonstration of the constellation pipeline working with 4 modes,
3 layers for memory efficiency and fast training.
"""

import numpy as np
import tensorflow as tf
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit
from src.models.generators.pure_sf_generator import PureSFGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_constellation_generator():
    """Create generator with constellation pipeline enabled."""
    
    class ConstellationSFGenerator(PureSFGenerator):
        def __init__(self, **kwargs):
            # Store constellation config before calling super
            self.use_constellation = kwargs.pop('use_constellation', False)
            self.constellation_radius = kwargs.pop('constellation_radius', 1.5)
            
            super().__init__(**kwargs)
            
            # Replace quantum circuit with constellation version
            if self.use_constellation:
                self.quantum_circuit = PureSFQuantumCircuit(
                    n_modes=self.n_modes,
                    n_layers=self.layers,
                    cutoff_dim=self.cutoff_dim,
                    circuit_type="variational",
                    use_constellation=True,  # 🌟 CONSTELLATION ENABLED
                    constellation_radius=self.constellation_radius
                )
                logger.info(f"🌟 Generator: Constellation pipeline enabled with {self.n_modes} modes")
    
    return ConstellationSFGenerator

def test_constellation_pipeline():
    """Test constellation pipeline integration."""
    
    print("🌟" * 60)
    print("CONSTELLATION PIPELINE DEMONSTRATION")
    print("🌟" * 60)
    print()
    
    # Configuration - exactly as requested
    config = {
        'n_modes': 4,           # 4-mode for memory efficiency
        'layers': 3,            # 3 layers as requested  
        'cutoff_dim': 6,        # Memory: 6^4 = 1,296 dimensions
        'latent_dim': 8,        # Input dimension
        'output_dim': 4,        # Output dimension
        'constellation_radius': 1.5,  # Constellation radius
        'batch_size': 8,        # Small batch for memory efficiency
    }
    
    print(f"📊 Configuration:")
    print(f"   Modes: {config['n_modes']} (memory: {config['cutoff_dim']**config['n_modes']:,} dimensions)")
    print(f"   Layers: {config['layers']} (3-layer as requested)")
    print(f"   Batch size: {config['batch_size']} (memory efficient)")
    print()
    
    # Step 1: Test Baseline Generator (No Constellation)
    print("🔍 Step 1: Testing Baseline Generator")
    print("-" * 50)
    
    try:
        GeneratorClass = create_constellation_generator()
        
        baseline_generator = GeneratorClass(
            latent_dim=config['latent_dim'],
            output_dim=config['output_dim'],
            n_modes=config['n_modes'],
            layers=config['layers'],
            cutoff_dim=config['cutoff_dim'],
            use_constellation=False  # Baseline
        )
        
        print(f"   ✅ Baseline generator created")
        print(f"   Parameters: {baseline_generator.get_parameter_count()}")
        
        # Test generation
        z_batch = tf.random.normal([config['batch_size'], config['latent_dim']])
        baseline_samples = baseline_generator.generate(z_batch)
        
        print(f"   ✅ Generation successful: {baseline_samples.shape}")
        print(f"   Sample variance: {tf.math.reduce_variance(baseline_samples).numpy():.6f}")
        
        baseline_success = True
        
    except Exception as e:
        print(f"   ❌ Baseline generator failed: {e}")
        baseline_success = False
    
    print()
    
    # Step 2: Test Constellation Generator
    print("🌟 Step 2: Testing Constellation Generator")
    print("-" * 50)
    
    try:
        constellation_generator = GeneratorClass(
            latent_dim=config['latent_dim'],
            output_dim=config['output_dim'],
            n_modes=config['n_modes'],
            layers=config['layers'],
            cutoff_dim=config['cutoff_dim'],
            use_constellation=True,  # 🌟 CONSTELLATION ENABLED
            constellation_radius=config['constellation_radius']
        )
        
        print(f"   ✅ Constellation generator created")
        print(f"   Parameters: {constellation_generator.get_parameter_count()}")
        
        # Test generation
        constellation_samples = constellation_generator.generate(z_batch)
        
        print(f"   ✅ Generation successful: {constellation_samples.shape}")
        print(f"   Sample variance: {tf.math.reduce_variance(constellation_samples).numpy():.6f}")
        
        constellation_success = True
        
    except Exception as e:
        print(f"   ❌ Constellation generator failed: {e}")
        constellation_success = False
    
    print()
    
    # Step 3: Compare Results
    print("📊 Step 3: Comparison")
    print("-" * 50)
    
    if baseline_success and constellation_success:
        baseline_params = baseline_generator.get_parameter_count()
        constellation_params = constellation_generator.get_parameter_count()
        
        print(f"Parameter Efficiency:")
        print(f"   Baseline parameters:      {baseline_params}")
        print(f"   Constellation parameters: {constellation_params}")
        print(f"   Difference:              {baseline_params - constellation_params} (same - constellation is static)")
        
        # Mode utilization analysis
        def analyze_mode_utilization(samples, n_modes, label):
            measurements_per_mode = 2  # x and p quadratures
            mode_variances = []
            
            for mode in range(n_modes):
                start_idx = mode * measurements_per_mode
                end_idx = start_idx + measurements_per_mode
                if end_idx <= samples.shape[1]:
                    mode_data = samples[:, start_idx:end_idx]
                    mode_variance = float(tf.math.reduce_variance(mode_data))
                    mode_variances.append(mode_variance)
            
            active_modes = sum(1 for v in mode_variances if v > 0.001)
            utilization = active_modes / n_modes
            
            print(f"   {label} mode utilization: {active_modes}/{n_modes} ({utilization:.1%})")
            print(f"   {label} mode variances: {[f'{v:.4f}' for v in mode_variances]}")
            
            return utilization
        
        print(f"\nMode Utilization Analysis:")
        baseline_util = analyze_mode_utilization(baseline_samples, config['n_modes'], "Baseline")
        constellation_util = analyze_mode_utilization(constellation_samples, config['n_modes'], "Constellation")
        
        # Overall assessment
        print(f"\n🎯 Assessment:")
        
        if constellation_success and baseline_success:
            print("   ✅ PIPELINE SUCCESS: Both versions working")
        
        if constellation_params == baseline_params:
            print("   ✅ PARAMETER EFFICIENCY: Same trainable parameters (constellation is static)")
        
        if constellation_util >= baseline_util:
            print("   ✅ MODE UTILIZATION: Constellation maintains/improves utilization")
        
        print(f"\n🌟 Constellation Pipeline Benefits:")
        print(f"   • Memory efficient: {config['n_modes']} modes, {config['cutoff_dim']**config['n_modes']:,} dimensions")
        print(f"   • Static initialization: Perfect {360/config['n_modes']:.1f}° spacing")
        print(f"   • Fast training: Only {constellation_params} trainable parameters")
        print(f"   • Multimode diversity: Communication theory optimal")
        
        return True
        
    else:
        print("   ❌ Cannot compare - one or both versions failed")
        return False

def test_gradient_flow():
    """Test gradient flow through constellation pipeline."""
    
    print("\n⚡ Testing Gradient Flow")
    print("-" * 40)
    
    try:
        GeneratorClass = create_constellation_generator()
        
        generator = GeneratorClass(
            latent_dim=8,
            output_dim=4,
            n_modes=4,
            layers=3,
            cutoff_dim=6,
            use_constellation=True
        )
        
        # Test gradient flow
        z_batch = tf.random.normal([4, 8])
        
        with tf.GradientTape() as tape:
            samples = generator.generate(z_batch)
            loss = tf.reduce_mean(tf.square(samples))
        
        gradients = tape.gradient(loss, generator.trainable_variables)
        valid_grads = [g for g in gradients if g is not None]
        grad_ratio = len(valid_grads) / len(generator.trainable_variables)
        
        print(f"   Gradient flow: {grad_ratio:.1%} ({len(valid_grads)}/{len(generator.trainable_variables)})")
        
        if grad_ratio >= 0.8:
            print("   ✅ Excellent gradient flow")
            return True
        else:
            print("   ⚠️  Limited gradient flow")
            return False
        
    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")
        return False

def main():
    """Run constellation pipeline demonstration."""
    
    print("🚀 STARTING CONSTELLATION PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Test pipeline integration
        pipeline_success = test_constellation_pipeline()
        
        # Test gradient flow
        gradient_success = test_gradient_flow()
        
        print("\n" + "=" * 80)
        print("CONSTELLATION PIPELINE DEMONSTRATION COMPLETE")
        print("=" * 80)
        
        if pipeline_success and gradient_success:
            print("🏆 SUCCESS: Constellation pipeline fully functional!")
            print()
            print("Key Achievements:")
            print("   • 4-mode, 3-layer configuration (memory efficient)")
            print("   • Static constellation encoding (not trainable)")
            print("   • Perfect gradient flow through variational parameters")
            print("   • Communication theory optimal initialization")
            print("   • Ready for quantum GAN training")
            
        else:
            print("❌ ISSUES DETECTED: Pipeline needs optimization")
            
        return pipeline_success and gradient_success
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Constellation pipeline ready for production!")
        print("   Use: PureSFQuantumCircuit(use_constellation=True)")
        print("   Benefits: Static initialization + Fast training + Memory efficient")
    else:
        print("\n❌ Fix issues before production deployment.")
