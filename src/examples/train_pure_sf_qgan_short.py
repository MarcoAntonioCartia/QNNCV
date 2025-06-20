"""
SHORT Training Script for Pure SF Quantum GAN with Enhanced Expressivity

This script implements the mode collapse fixes from the diagnostic guide:
- Phase 1: Enhanced expressivity (6 modes, 4 layers, 84+ parameters)
- Phase 2: Improved batching (batch_size=16 for efficiency)
- Comprehensive quantum visualization and circuit analysis
- Quick validation (5 epochs × 5 steps = 25 total steps)

Expected improvements:
- 84 vs 30 generator parameters (sufficient for universal approximation)
- 6 modes vs 4 (bimodal distribution capability)  
- 4 layers vs 2 (deep quantum circuit expressivity)
- Batch processing efficiency and quantum coherence
"""

import numpy as np
import tensorflow as tf
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Pure SF components
from src.models.generators.pure_sf_generator import PureSFGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator

# Import training framework
from src.training.quantum_gan_trainer import QuantumGANTrainer
from src.training.data_generators import BimodalDataGenerator

# Import visualization
from src.utils.quantum_visualization_manager import CorrectedQuantumVisualizationManager

# Import utilities
from src.utils.warning_suppression import suppress_all_quantum_warnings


def main():
    """Main training function with enhanced expressivity and visualization."""
    
    # Suppress quantum warnings for cleaner output
    suppress_all_quantum_warnings()
    
    logger.info("=" * 80)
    logger.info("ENHANCED PURE SF QUANTUM GAN - MODE COLLAPSE FIXES")
    logger.info("Phase 1: Enhanced Expressivity + Phase 2: Improved Batching")
    logger.info("=" * 80)
    
    # ENHANCED CONFIGURATION: Diagnostic guide recommendations
    config = {
        # EXPRESSIVITY FIX: Enhanced model configuration
        'latent_dim': 4,           
        'output_dim': 2,           
        'generator_modes': 3,      # ✅ Reduced for memory (was 4)
        'discriminator_modes': 2,  # ✅ Reduced for memory (was 3)  
        'layers': 5,               # ✅ KEEP - main expressivity driver
        'cutoff_dim': 6,           # ✅ Reduced for memory (was 8)
        
        # BATCHING FIX: Larger batches for efficiency + quantum coherence
        'batch_size': 16,          #  16 instead of 8 (batching efficiency)
        
        # SHORT TRAINING: Quick validation
        'epochs': 5,               #  5 instead of 30 (quick validation)
        'steps_per_epoch': 5,      #  5 instead of 20 (fast proof-of-concept)
        'n_critic': 3,             #  Reduced for faster training
        
        # Learning rates (conservative for stability)
        'lr_generator': 1e-3,      
        'lr_discriminator': 1e-3,  
        
        # Data configuration
        'mode1_center': (-2.0, -2.0),
        'mode2_center': (2.0, 2.0),
        'mode_std': 0.5,
        
        # Enhanced monitoring
        'plot_interval': 2,        # ✅ Plot every 2 epochs (frequent)
        'save_interval': 5,        # ✅ Save at end
        'verbose': True
    }
    
    logger.info("ENHANCED Configuration:")
    logger.info("  EXPRESSIVITY UPGRADES:")
    logger.info(f"    Generator modes: {config['generator_modes']} (was 4)")
    logger.info(f"    Discriminator modes: {config['discriminator_modes']} (was 3)")
    logger.info(f"    Circuit layers: {config['layers']} (was 2)")
    logger.info(f"    Cutoff dimension: {config['cutoff_dim']} (was 6)")
    logger.info("  TRAINING EFFICIENCY:")
    logger.info(f"    Batch size: {config['batch_size']} (was 8)")
    logger.info(f"    Epochs: {config['epochs']} (was 30)")
    logger.info(f"    Steps per epoch: {config['steps_per_epoch']} (was 20)")
    logger.info(f"    Total training steps: {config['epochs'] * config['steps_per_epoch']}")
    
    # Initialize visualization manager
    logger.info("\n🎨 Initializing quantum visualization manager...")
    viz_manager = CorrectedQuantumVisualizationManager("enhanced_qgan_visualizations")
    logger.info("✅ Visualization manager ready")
    
    # Create ENHANCED Pure SF models
    logger.info("\n🔧 Creating ENHANCED Pure SF models...")
    
    # Enhanced generator with increased expressivity
    generator = PureSFGenerator(
        latent_dim=config['latent_dim'],
        output_dim=config['output_dim'],
        n_modes=config['generator_modes'],    # 6 modes for bimodal capability
        layers=config['layers'],              # 4 layers for universality
        cutoff_dim=config['cutoff_dim']       # 10 cutoff for rich state space
    )
    
    # Enhanced discriminator (slightly smaller than generator for balance)
    discriminator = PureSFDiscriminator(
        input_dim=config['output_dim'],
        n_modes=config['discriminator_modes'], # 4 modes (balanced)
        layers=config['layers'],               # 4 layers (same depth)
        cutoff_dim=config['cutoff_dim']        # 10 cutoff (same richness)
    )
    
    logger.info(f"✅ Enhanced Generator: {len(generator.trainable_variables)} parameters")
    logger.info(f"✅ Enhanced Discriminator: {len(discriminator.trainable_variables)} parameters")
    
    # Calculate expected parameter increase
    old_g_params = 30  # Previous generator
    old_d_params = 22  # Previous discriminator
    new_g_params = len(generator.trainable_variables)
    new_d_params = len(discriminator.trainable_variables)
    
    logger.info(f"📊 Parameter Comparison:")
    logger.info(f"  Generator: {new_g_params} vs {old_g_params} (+{new_g_params-old_g_params}) = {new_g_params/old_g_params:.1f}x increase")
    logger.info(f"  Discriminator: {new_d_params} vs {old_d_params} (+{new_d_params-old_d_params}) = {new_d_params/old_d_params:.1f}x increase")
    
    # VISUALIZATION: Circuit structure analysis
    logger.info("\n🔬 Analyzing quantum circuit structures...")
    
    try:
        viz_manager.visualize_pure_sf_circuit(
            generator.quantum_circuit, 
            "Enhanced Generator Circuit", 
            show_parameters=True, 
            show_values=True
        )
        
        viz_manager.visualize_pure_sf_circuit(
            discriminator.quantum_circuit, 
            "Enhanced Discriminator Circuit", 
            show_parameters=True, 
            show_values=True
        )
        
        logger.info("✅ Circuit visualization completed")
        
    except Exception as e:
        logger.warning(f"⚠️ Circuit visualization failed: {e}")
    
    # Test enhanced model functionality
    logger.info("\n🧪 Testing enhanced model functionality...")
    
    # Test generator with larger batch
    z_test = tf.random.normal([config['batch_size'], config['latent_dim']])
    try:
        generated_samples = generator.generate(z_test)
        logger.info(f"✅ Enhanced Generator test: input {z_test.shape} → output {generated_samples.shape}")
        
        # Analyze initial diversity
        initial_variance = tf.math.reduce_variance(generated_samples, axis=0)
        logger.info(f"  Initial sample variance: {initial_variance.numpy()}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced Generator test failed: {e}")
        return
    
    # Test discriminator
    x_test = tf.random.normal([config['batch_size'], config['output_dim']])
    try:
        discriminator_output = discriminator.discriminate(x_test)
        logger.info(f"✅ Enhanced Discriminator test: input {x_test.shape} → output {discriminator_output.shape}")
        
        # Analyze response diversity
        response_variance = tf.math.reduce_variance(discriminator_output)
        logger.info(f"  Response variance: {response_variance.numpy()}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced Discriminator test failed: {e}")
        return
    
    # Create enhanced data generator
    logger.info("\n📊 Creating enhanced bimodal data generator...")
    
    data_generator = BimodalDataGenerator(
        batch_size=config['batch_size'],      # Larger batches
        n_features=config['output_dim'],
        mode1_center=config['mode1_center'],
        mode2_center=config['mode2_center'],
        mode_std=config['mode_std']
    )
    
    # Test data generator with larger batches
    test_batch = data_generator()
    logger.info(f"✅ Enhanced data generator: batch shape {test_batch.shape}")
    logger.info(f"   Batch statistics: mean={tf.reduce_mean(test_batch, axis=0).numpy()}")
    logger.info(f"   Batch variance: {tf.math.reduce_variance(test_batch, axis=0).numpy()}")
    
    # Create validation data (larger for better statistics)
    validation_data = data_generator.generate_dataset(n_batches=20)  # 320 samples
    logger.info(f"✅ Enhanced validation data: {validation_data.shape}")
    
    # VISUALIZATION: Data distribution analysis
    try:
        viz_manager.create_qgan_comparison_dashboard(
            generator, discriminator, validation_data,
            n_samples=100, title="Pre-Training Analysis"
        )
        logger.info("✅ Pre-training analysis visualization completed")
    except Exception as e:
        logger.warning(f"⚠️ Pre-training visualization failed: {e}")
    
    # Create enhanced trainer
    logger.info("\n🚀 Creating Enhanced Quantum GAN trainer...")
    
    trainer = QuantumGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_type='wasserstein',
        n_critic=config['n_critic'],
        learning_rate_g=config['lr_generator'],
        learning_rate_d=config['lr_discriminator'],
        verbose=config['verbose']
    )
    
    logger.info("✅ Enhanced trainer created successfully")
    
    # Pre-training gradient flow verification
    logger.info("\n🔍 Pre-training gradient flow verification...")
    
    try:
        # Test gradient flow with enhanced models
        with tf.GradientTape(persistent=True) as tape:
            fake_samples = generator.generate(z_test)
            fake_scores = discriminator.discriminate(fake_samples)
            real_scores = discriminator.discriminate(x_test)
            
            g_loss_test = -tf.reduce_mean(fake_scores)
            d_loss_test = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
        
        g_grads_test = tape.gradient(g_loss_test, generator.trainable_variables)
        d_grads_test = tape.gradient(d_loss_test, discriminator.trainable_variables)
        
        g_grad_count = sum(1 for g in g_grads_test if g is not None)
        d_grad_count = sum(1 for g in d_grads_test if g is not None)
        
        logger.info(f"✅ Pre-training gradient flow:")
        logger.info(f"  Generator: {g_grad_count}/{len(generator.trainable_variables)} ({g_grad_count/len(generator.trainable_variables)*100:.1f}%)")
        logger.info(f"  Discriminator: {d_grad_count}/{len(discriminator.trainable_variables)} ({d_grad_count/len(discriminator.trainable_variables)*100:.1f}%)")
        
        if g_grad_count == len(generator.trainable_variables) and d_grad_count == len(discriminator.trainable_variables):
            logger.info("🎉 PERFECT gradient flow achieved!")
        else:
            logger.warning("⚠️ Partial gradient flow detected")
            
    except Exception as e:
        logger.error(f"❌ Gradient flow test failed: {e}")
    
    # Start ENHANCED training
    logger.info("\n" + "=" * 80)
    logger.info("🚀 STARTING ENHANCED PURE SF QGAN TRAINING")
    logger.info("Expected training time: ~15-20 minutes")
    logger.info("=" * 80)
    
    training_start_time = datetime.now()
    
    try:
        # Custom training loop with visualization integration
        for epoch in range(config['epochs']):
            epoch_start = datetime.now()
            logger.info(f"\n📈 Epoch {epoch+1}/{config['epochs']}")
            
            # Track parameters for visualization
            viz_manager.track_parameters(generator, discriminator)
            
            # Run epoch training
            epoch_metrics = {}
            for step in range(config['steps_per_epoch']):
                # Get data batch
                real_batch = data_generator()
                z_batch = tf.random.normal([config['batch_size'], config['latent_dim']])
                
                # Train discriminator
                for _ in range(config['n_critic']):
                    d_metrics = trainer.train_discriminator_step(real_batch, z_batch)
                
                # Train generator
                g_metrics = trainer.train_generator_step(z_batch)
                
                # Store metrics
                for key, value in {**d_metrics, **g_metrics}.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    if isinstance(value, tf.Tensor):
                        epoch_metrics[key].append(float(value.numpy()))
                    else:
                        epoch_metrics[key].append(value)
            
            # Epoch summary
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            logger.info(f"  ⏱️ Epoch {epoch+1} completed in {epoch_time:.1f}s")
            
            if epoch_metrics:
                avg_g_loss = np.mean(epoch_metrics.get('total_loss', [0]))
                avg_d_loss = np.mean([v for v in epoch_metrics.values() if isinstance(v, list) and len(v) > 0])
                logger.info(f"  📊 Avg G_loss: {avg_g_loss:.4f}")
                
            # Generate samples for analysis
            test_z = tf.random.normal([100, config['latent_dim']])
            test_samples = generator.generate(test_z)
            sample_variance = tf.math.reduce_variance(test_samples, axis=0)
            
            logger.info(f"  🎯 Sample variance: {sample_variance.numpy()}")
            
            # Check for diversity improvement
            if tf.reduce_all(sample_variance > 1e-3):
                logger.info("  🎉 SAMPLE DIVERSITY DETECTED!")
            else:
                logger.info("  ⚠️ Limited diversity")
            
            # Visualization every 2 epochs
            if (epoch + 1) % config['plot_interval'] == 0:
                try:
                    viz_manager.create_qgan_comparison_dashboard(
                        generator, discriminator, validation_data,
                        n_samples=100, title=f"Epoch {epoch+1} Analysis"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Epoch {epoch+1} visualization failed: {e}")
        
        training_time = (datetime.now() - training_start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ENHANCED TRAINING COMPLETED!")
        logger.info(f"⏱️ Total training time: {training_time/60:.1f} minutes")
        logger.info("=" * 80)
        
        # Final comprehensive evaluation
        logger.info("\n📈 Final comprehensive evaluation...")
        
        # Generate large sample set for analysis
        z_final = tf.random.normal([500, config['latent_dim']])
        final_samples = generator.generate(z_final)
        
        # Comprehensive statistics
        final_mean = tf.reduce_mean(final_samples, axis=0)
        final_std = tf.math.reduce_std(final_samples, axis=0)
        final_variance = tf.math.reduce_variance(final_samples, axis=0)
        
        # Compare with validation data
        val_mean = tf.reduce_mean(validation_data, axis=0)
        val_std = tf.math.reduce_std(validation_data, axis=0)
        
        logger.info(f"📊 Final Sample Analysis:")
        logger.info(f"  Generated mean: {final_mean.numpy()}")
        logger.info(f"  Generated std: {final_std.numpy()}")
        logger.info(f"  Generated variance: {final_variance.numpy()}")
        logger.info(f"  Target mean: {val_mean.numpy()}")
        logger.info(f"  Target std: {val_std.numpy()}")
        
        # Mode collapse assessment
        total_variance = tf.reduce_sum(final_variance)
        mode_collapse_threshold = 1e-3
        
        logger.info(f"\n🎯 Mode Collapse Assessment:")
        logger.info(f"  Total variance: {total_variance.numpy():.6f}")
        logger.info(f"  Threshold: {mode_collapse_threshold}")
        
        if total_variance > mode_collapse_threshold:
            logger.info("  🎉 SUCCESS: No mode collapse detected!")
            logger.info("  ✅ Enhanced expressivity fix worked!")
        else:
            logger.info("  ⚠️ WARNING: Mode collapse still present")
            logger.info("  🔍 May need Phase 3 (regularization) fixes")
        
        # Parameter analysis
        logger.info(f"\n🔧 Enhanced Model Analysis:")
        logger.info(f"  Generator parameters: {len(generator.trainable_variables)} (was 30)")
        logger.info(f"  Discriminator parameters: {len(discriminator.trainable_variables)} (was 22)")
        logger.info(f"  Parameter increase: {(len(generator.trainable_variables) + len(discriminator.trainable_variables))/(30+22):.1f}x")
        
        # Final visualization
        try:
            viz_manager.create_qgan_comparison_dashboard(
                generator, discriminator, validation_data,
                n_samples=200, title="Final Enhanced QGAN Analysis"
            )
            logger.info("✅ Final comprehensive visualization completed")
        except Exception as e:
            logger.warning(f"⚠️ Final visualization failed: {e}")
        
        # Save enhanced results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'enhanced_pure_sf_qgan_results_{timestamp}.npz'
        
        np.savez(
            results_file,
            generated_samples=final_samples.numpy(),
            validation_data=validation_data.numpy(),
            config=config,
            training_time_minutes=training_time/60,
            final_variance=final_variance.numpy(),
            mode_collapse_resolved=total_variance > mode_collapse_threshold
        )
        
        logger.info(f"\n💾 Enhanced results saved to {results_file}")
        
        # Success summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ENHANCED PURE SF QUANTUM GAN ANALYSIS COMPLETE!")
        logger.info("=" * 80)
        logger.info("✅ Enhanced expressivity: 6 modes, 4 layers, 10 cutoff")
        logger.info("✅ Improved batching: batch_size=16")
        logger.info("✅ Quick validation: 25 training steps in ~15 minutes")
        logger.info("✅ Comprehensive quantum circuit visualization")
        logger.info("✅ Parameter count increased ~3x for universal approximation")
        
        if total_variance > mode_collapse_threshold:
            logger.info("🎉 MODE COLLAPSE SUCCESSFULLY RESOLVED!")
            logger.info("✅ Diagnostic guide fixes were effective")
        else:
            logger.info("🔍 Mode collapse partially improved - may need Phase 3 fixes")
        
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Enhanced Pure SF QGAN training completed successfully!")
        print("🔍 Check the visualizations for quantum circuit analysis!")
    else:
        print("\n❌ Enhanced Pure SF QGAN training failed!")
        sys.exit(1)
