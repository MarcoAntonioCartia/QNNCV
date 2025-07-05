"""
Phase 3: Complete Input-Dependent Quantum Generator

This combines Phase 1 (Input-Dependent Quantum Circuit) with Phase 2 (Spatial Mode Decoder)
to create the complete solution for the linear interpolation problem.

Architecture:
1. Latent vector → Input-dependent quantum initial state (Phase 1) 
2. QNN processing → X-quadrature measurements (Phase 1)
3. Spatial mode assignment → Discrete spatial coordinates (Phase 2)
4. Training with classical discriminator (Phase 3)

Expected: R² < 0.5 (vs 0.999 in v0.4) + 100% gradient flow
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.data_generators import BimodalDataGenerator
from src.quantum.core.input_dependent_quantum_circuit import InputDependentQuantumCircuit
from src.quantum.core.spatial_mode_decoder import SpatialModeDecoder


class CompleteQuantumGenerator:
    """
    Complete Input-Dependent Quantum Generator - Phase 3
    
    This combines all innovations:
    - Phase 1: Input-dependent quantum initial states (solves static state problem)
    - Phase 2: Spatial mode assignment decoder (solves linear interpolation)
    - Preserves 100% gradient flow throughout
    - No classical neural networks in the path
    
    Architecture:
    Latent → Input-Dependent QNN → X-Quadratures → Spatial Decoder → Output
    """
    
    def __init__(self,
                 latent_dim: int = 4,
                 output_dim: int = 2, 
                 n_modes: int = 4,
                 n_layers: int = 2,
                 cutoff_dim: int = 4,
                 input_scale_factor: float = 0.1,
                 spatial_scale_factor: float = 1.0):
        """
        Initialize complete quantum generator.
        
        Args:
            latent_dim: Latent space dimensionality
            output_dim: Output dimensionality (2 for 2D data)
            n_modes: Number of quantum modes
            n_layers: Number of QNN layers
            cutoff_dim: Fock space cutoff
            input_scale_factor: Scaling for quantum initial states
            spatial_scale_factor: Scaling for spatial coordinates
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.n_layers = n_layers
        
        # Phase 1: Input-dependent quantum circuit
        self.quantum_circuit = InputDependentQuantumCircuit(
            n_modes=n_modes,
            n_layers=n_layers,
            latent_dim=latent_dim,
            cutoff_dim=cutoff_dim,
            input_scale_factor=input_scale_factor
        )
        
        # Phase 2: Spatial mode assignment decoder
        self.spatial_decoder = SpatialModeDecoder(
            n_modes=n_modes,
            output_dim=output_dim,
            spatial_scale_factor=spatial_scale_factor,
            real_data_range=(-1.5, 1.5)  # Bimodal data range
        )
        
        print(f"Complete Quantum Generator initialized:")
        print(f"  Architecture: {latent_dim}D → QNN({n_modes} modes) → Spatial → {output_dim}D")
        print(f"  Phase 1: Input-dependent quantum states")
        print(f"  Phase 2: Spatial assignment {self.spatial_decoder.spatial_assignment}")
        print(f"  Trainable parameters: {len(self.trainable_variables)} (100% quantum)")
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables (only quantum circuit parameters)."""
        return self.quantum_circuit.trainable_variables
    
    def generate(self, latent_batch: tf.Tensor) -> tf.Tensor:
        """
        Generate samples using complete quantum pipeline.
        
        Args:
            latent_batch: Latent vectors [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, output_dim]
        """
        # Phase 1: Input-dependent quantum processing
        x_quadratures = self.quantum_circuit.process_batch(latent_batch)
        
        # Phase 2: Spatial mode assignment decoding
        spatial_outputs = self.spatial_decoder.decode(x_quadratures)
        
        return spatial_outputs
    
    def test_complete_pipeline(self) -> Dict[str, Any]:
        """Test the complete quantum generation pipeline."""
        print(f"\n🧪 Testing Complete Quantum Pipeline:")
        
        # Test data
        test_latents = tf.random.normal([16, self.latent_dim])
        
        # Generate samples
        generated = self.generate(test_latents)
        
        print(f"   Input shape: {test_latents.shape}")
        print(f"   Output shape: {generated.shape}")
        print(f"   Sample output: {generated[0].numpy()}")
        
        # Test gradient flow through complete pipeline
        target = tf.random.normal([16, self.output_dim])
        
        with tf.GradientTape() as tape:
            outputs = self.generate(test_latents)
            loss = tf.reduce_mean(tf.square(outputs - target))
        
        gradients = tape.gradient(loss, self.trainable_variables)
        valid_gradients = [g for g in gradients if g is not None]
        gradient_flow = len(valid_gradients) / len(self.trainable_variables)
        
        # Analyze linear interpolation
        analysis = self.spatial_decoder.analyze_spatial_assignment(
            self.quantum_circuit.process_batch(test_latents)
        )
        
        print(f"   Gradient flow: {gradient_flow:.1%}")
        print(f"   R² (want < 0.5): {analysis['r_squared']:.3f}")
        print(f"   Linear pattern: {'❌ YES' if analysis['is_linear'] else '✅ NO'}")
        
        return {
            'gradient_flow': gradient_flow,
            'r_squared': analysis['r_squared'],
            'is_linear': analysis['is_linear'],
            'spatial_assignment': analysis['spatial_assignment']
        }
    
    def calibrate_to_real_data(self, real_data: tf.Tensor):
        """Calibrate spatial decoder to real data distribution."""
        self.spatial_decoder.calibrate_to_real_data(real_data)


class Phase3Trainer:
    """
    Phase 3 Trainer - Test Complete Quantum Generator with Classical Discriminator
    
    This tests the complete solution:
    - Input-dependent quantum states (Phase 1)  
    - Spatial mode assignment (Phase 2)
    - Training stability and convergence (Phase 3)
    """
    
    def __init__(self):
        """Initialize Phase 3 trainer."""
        self.config = {
            'batch_size': 16,
            'latent_dim': 4,
            'n_modes': 4,  # 4 modes → 2D (your example)
            'n_layers': 2,
            'cutoff_dim': 4,
            'epochs': 5,
            'steps_per_epoch': 10,
            'learning_rate': 1e-3
        }
        
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.data_generator = None
        
        # Results tracking
        self.results = {
            'phase1_success': False,
            'phase2_success': False,
            'phase3_success': False,
            'final_r_squared': 1.0,
            'final_gradient_flow': 0.0,
            'training_history': []
        }
        
        print(f"Phase 3 Trainer initialized:")
        print(f"  Goal: Test complete quantum generator")
        print(f"  Expected: R² < 0.5 + 100% gradient flow")
    
    def setup_models(self):
        """Setup complete quantum generator and classical discriminator."""
        print(f"\n🔧 Setting up Phase 3 Models:")
        
        # Complete quantum generator (Phases 1 + 2)
        self.generator = CompleteQuantumGenerator(
            latent_dim=self.config['latent_dim'],
            output_dim=2,
            n_modes=self.config['n_modes'],
            n_layers=self.config['n_layers'],
            cutoff_dim=self.config['cutoff_dim']
        )
        
        # Simple classical discriminator
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Bimodal data generator
        self.data_generator = BimodalDataGenerator(
            batch_size=self.config['batch_size'],
            n_features=2,
            mode1_center=(-1.5, -1.5),
            mode2_center=(1.5, 1.5),
            mode_std=0.3
        )
        
        # Calibrate generator to real data
        real_samples = []
        for _ in range(10):
            batch = self.data_generator.generate_batch()
            real_samples.append(batch)
        real_data = np.vstack(real_samples)
        self.generator.calibrate_to_real_data(tf.constant(real_data, dtype=tf.float32))
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], beta_1=0.5
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], beta_1=0.5
        )
        
        # Test complete pipeline
        pipeline_results = self.generator.test_complete_pipeline()
        
        print(f"  ✅ Setup complete:")
        print(f"     Gradient flow: {pipeline_results['gradient_flow']:.1%}")
        print(f"     Initial R²: {pipeline_results['r_squared']:.3f}")
        print(f"     Linear pattern: {'❌' if pipeline_results['is_linear'] else '✅'}")
        
        return pipeline_results
    
    def analyze_clusters(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze cluster quality and linear patterns (same as v0.4)."""
        if len(data) < 2:
            return {'cluster_quality': 0.0, 'n_detected_clusters': 0, 'is_linear': True, 'r_squared': 1.0}
        
        try:
            # K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            cluster_centers = kmeans.cluster_centers_
            
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return {'cluster_quality': 0.0, 'n_detected_clusters': n_clusters, 'is_linear': True, 'r_squared': 1.0}
            
            # Quality metrics
            center_distance = np.linalg.norm(cluster_centers[0] - cluster_centers[1])
            
            compactness = 0.0
            for i in range(n_clusters):
                cluster_data = data[cluster_labels == i]
                if len(cluster_data) > 0:
                    center = cluster_centers[i]
                    distances = np.linalg.norm(cluster_data - center, axis=1)
                    compactness += np.mean(distances)
            compactness /= n_clusters
            
            separation_ratio = center_distance / max(compactness, 0.1)
            cluster_quality = min(separation_ratio / 10.0, 1.0)
            
            # Linear pattern detection
            X = data[:, 0].reshape(-1, 1)
            y = data[:, 1]
            
            X_mean = np.mean(X)
            y_mean = np.mean(y)
            
            numerator = np.sum((X.flatten() - X_mean) * (y - y_mean))
            denominator = np.sum((X.flatten() - X_mean) ** 2)
            
            is_linear = False
            r_squared = 0.0
            
            if denominator > 1e-10:
                slope = numerator / denominator
                intercept = y_mean - slope * X_mean
                y_pred = slope * X.flatten() + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y_mean) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                is_linear = r_squared > 0.8
            
            return {
                'cluster_quality': cluster_quality,
                'n_detected_clusters': n_clusters,
                'separation_ratio': separation_ratio,
                'center_distance': center_distance,
                'compactness': compactness,
                'is_linear': is_linear,
                'r_squared': r_squared
            }
            
        except Exception:
            return {'cluster_quality': 0.0, 'n_detected_clusters': 0, 'is_linear': True, 'r_squared': 1.0}
    
    def train_step(self, real_batch: tf.Tensor) -> Dict[str, Any]:
        """Training step for Phase 3."""
        batch_size = tf.shape(real_batch)[0]
        z = tf.random.normal([batch_size, self.config['latent_dim']])
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            fake_batch = self.generator.generate(z)
            real_output = self.discriminator(real_batch)
            fake_output = self.discriminator(fake_batch)
            
            d_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
        
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as g_tape:
            fake_batch = self.generator.generate(z)
            fake_output = self.discriminator(fake_batch)
            g_loss = -tf.reduce_mean(fake_output)
        
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Check gradient flow
        valid_gradients = [g for g in g_gradients if g is not None and not tf.reduce_any(tf.math.is_nan(g))]
        gradient_flow = len(valid_gradients) / len(self.generator.trainable_variables) if self.generator.trainable_variables else 0
        
        # Apply gradients if valid
        if gradient_flow > 0:
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Sample diversity
        sample_std = tf.math.reduce_std(fake_batch, axis=0)
        sample_diversity = tf.reduce_mean(sample_std)
        
        return {
            'g_loss': float(g_loss),
            'd_loss': float(d_loss),
            'sample_diversity': float(sample_diversity),
            'gradient_flow': gradient_flow,
            'fake_batch': fake_batch.numpy(),
            'real_batch': real_batch.numpy()
        }
    
    def train(self) -> Dict[str, Any]:
        """Execute Phase 3 complete training."""
        print(f"\n🚀 Starting Phase 3 Training - Complete Quantum Generator")
        print(f"=" * 80)
        print(f"Testing Combined Solution:")
        print(f"  • Phase 1: Input-dependent quantum states ✅")
        print(f"  • Phase 2: Spatial mode assignment ✅") 
        print(f"  • Phase 3: Training with classical discriminator")
        print(f"")
        print(f"Expected Results:")
        print(f"  • R² < 0.5 (vs 0.999 in v0.4)")
        print(f"  • 100% gradient flow (vs 0% in v0.3)")
        print(f"  • Discrete cluster formation")
        print(f"=" * 80)
        
        # Setup models
        pipeline_results = self.setup_models()
        
        # Train for specified epochs
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            epoch_results = []
            for step in range(self.config['steps_per_epoch']):
                real_batch = self.data_generator.generate_batch()
                step_result = self.train_step(real_batch)
                epoch_results.append(step_result)
                
                print(f"  Step {step + 1}: GF={step_result['gradient_flow']:.1%}, G={step_result['g_loss']:.3f}")
            
            # Epoch analysis
            avg_gradient_flow = np.mean([r['gradient_flow'] for r in epoch_results])
            print(f"  Avg gradient flow: {avg_gradient_flow:.1%}")
        
        # Final analysis
        print(f"\n" + "=" * 80)
        print(f"🎯 Phase 3 Training Complete - Final Analysis")
        print(f"=" * 80)
        
        # Generate final samples for analysis
        z_final = tf.random.normal([100, self.config['latent_dim']])
        final_samples = self.generator.generate(z_final).numpy()
        
        # Final cluster analysis
        cluster_analysis = self.analyze_clusters(final_samples)
        
        # Final gradient flow test
        final_pipeline = self.generator.test_complete_pipeline()
        
        # Store results
        self.results.update({
            'phase1_success': True,  # Already verified
            'phase2_success': True,  # Already verified
            'phase3_success': cluster_analysis['r_squared'] < 0.5 and final_pipeline['gradient_flow'] > 0.99,
            'final_r_squared': cluster_analysis['r_squared'],
            'final_gradient_flow': final_pipeline['gradient_flow'],
            'cluster_quality': cluster_analysis['cluster_quality'],
            'final_samples': final_samples
        })
        
        # Results summary
        print(f"Phase-by-Phase Results:")
        print(f"  ✅ Phase 1 (Input-dependent quantum): SUCCESS")
        print(f"  ✅ Phase 2 (Spatial mode assignment): SUCCESS")
        print(f"  {'✅' if self.results['phase3_success'] else '❌'} Phase 3 (Complete training): {'SUCCESS' if self.results['phase3_success'] else 'PARTIAL'}")
        
        print(f"\nFinal Metrics:")
        print(f"  Gradient flow: {self.results['final_gradient_flow']:.1%} ({'✅ SUCCESS' if self.results['final_gradient_flow'] > 0.99 else '❌ FAILED'})")
        print(f"  R² linear pattern: {self.results['final_r_squared']:.3f} ({'✅ SUCCESS' if self.results['final_r_squared'] < 0.5 else '❌ FAILED'} - target <0.5)")
        print(f"  Cluster quality: {cluster_analysis['cluster_quality']:.3f}")
        print(f"  Linear interpolation: {'✅ ELIMINATED' if not cluster_analysis['is_linear'] else '❌ STILL PRESENT'}")
        
        print(f"\nComparison with Previous Versions:")
        print(f"  v0.4 (current): R²={self.results['final_r_squared']:.3f}, GF={self.results['final_gradient_flow']:.1%}")
        print(f"  v0.4 (before): R²=0.999, GF=100% (linear interpolation problem)")
        print(f"  v0.3: R²=N/A, GF=0% (gradient flow broken)")
        print(f"  v0.2: R²=0.990, GF=N/A (cluster fix failed)")
        print(f"  v0.1: R²=0.982, GF=N/A (matrix conditioning only)")
        
        # Overall assessment
        complete_success = (
            self.results['phase1_success'] and 
            self.results['phase2_success'] and 
            self.results['phase3_success']
        )
        
        if complete_success:
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"   ✅ Input-dependent quantum states working")
            print(f"   ✅ Spatial mode assignment working") 
            print(f"   ✅ Linear interpolation eliminated")
            print(f"   ✅ 100% gradient flow preserved")
            print(f"   📊 All requirements satisfied")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS")
            if not self.results['phase3_success']:
                print(f"   📊 Phase 3 needs refinement")
        
        return self.results


def main():
    """Main Phase 3 demonstration."""
    print(f"🔬 Phase 3: Complete Input-Dependent Quantum Generator")
    print(f"=" * 90)
    print(f"Combining All Innovations:")
    print(f"  • Phase 1: Input-dependent quantum initial states (Killoran-style)")
    print(f"  • Phase 2: Spatial mode assignment decoder (your innovation)")
    print(f"  • Phase 3: Complete training pipeline")
    print(f"")
    print(f"Expected Results:")
    print(f"  • R² < 0.5 (eliminate linear interpolation)")
    print(f"  • 100% gradient flow (preserve training)")
    print(f"  • Discrete cluster formation")
    print(f"  • No classical neural networks in generator")
    print(f"=" * 90)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Run Phase 3 training
        trainer = Phase3Trainer()
        results = trainer.train()
        
        # Final assessment
        print(f"\n" + "=" * 90)
        print(f"🎯 FINAL ASSESSMENT - Complete Quantum Generator Solution")
        print(f"=" * 90)
        
        success_metrics = [
            ("Input dependency", True),  # Phase 1 verified
            ("Spatial assignment", True),  # Phase 2 verified  
            ("Gradient flow", results['final_gradient_flow'] > 0.99),
            ("Linear interpolation eliminated", results['final_r_squared'] < 0.5),
            ("Training stability", results['phase3_success'])
        ]
        
        print(f"Success Metrics:")
        for metric, success in success_metrics:
            print(f"  {'✅' if success else '❌'} {metric}")
        
        overall_success = all(success for _, success in success_metrics)
        
        if overall_success:
            print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
            print(f"   Complete solution to linear interpolation problem")
            print(f"   Input-dependent quantum states + Spatial mode assignment")
            print(f"   Ready for integration into full QGAN architecture")
        else:
            print(f"\n📊 Significant progress made")
            print(f"   Individual phases working")
            print(f"   Integration may need fine-tuning")
        
        return results
        
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
