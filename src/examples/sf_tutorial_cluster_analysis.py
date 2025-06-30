"""
SF Tutorial Cluster Analysis and Visualization

This script provides comprehensive monitoring of the SF Tutorial gradient flow solution
with focus on cluster/bimodal data generation. It analyzes:

1. Cluster formation and separation
2. Real vs generated data comparison  
3. Animated evolution of cluster generation
4. Loss convergence analysis
5. Quantum state and parameter monitoring
6. Sample diversity and mode collapse detection

The goal is to understand why losses converge so rapidly and whether 
the generator is successfully learning to produce the target bimodal clusters.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import seaborn as sns
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.sf_tutorial_circuit import SFTutorialGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from src.training.data_generators import BimodalDataGenerator


class ClusterAnalyzer:
    """Analyzes cluster quality and separation."""
    
    def __init__(self, target_centers: List[Tuple[float, float]]):
        """Initialize cluster analyzer with target centers."""
        self.target_centers = np.array(target_centers)
        self.n_clusters = len(target_centers)
    
    def analyze_clusters(self, data: np.ndarray) -> Dict[str, float]:
        """Comprehensive cluster analysis."""
        if len(data) < self.n_clusters:
            return self._empty_analysis()
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        try:
            cluster_labels = kmeans.fit_predict(data)
            cluster_centers = kmeans.cluster_centers_
        except:
            return self._empty_analysis()
        
        # Cluster quality metrics
        metrics = {}
        
        # 1. Cluster separation (silhouette-like)
        if len(np.unique(cluster_labels)) == self.n_clusters:
            intra_distances = []
            inter_distances = []
            
            for i in range(self.n_clusters):
                cluster_data = data[cluster_labels == i]
                if len(cluster_data) > 0:
                    # Intra-cluster distance (compactness)
                    center = cluster_centers[i]
                    intra_dist = np.mean(np.linalg.norm(cluster_data - center, axis=1))
                    intra_distances.append(intra_dist)
                    
                    # Inter-cluster distance (separation)
                    other_centers = cluster_centers[np.arange(self.n_clusters) != i]
                    inter_dist = np.min(np.linalg.norm(other_centers - center, axis=1))
                    inter_distances.append(inter_dist)
            
            metrics['cluster_compactness'] = np.mean(intra_distances) if intra_distances else float('inf')
            metrics['cluster_separation'] = np.mean(inter_distances) if inter_distances else 0.0
            metrics['separation_ratio'] = metrics['cluster_separation'] / max(metrics['cluster_compactness'], 1e-6)
        else:
            metrics['cluster_compactness'] = float('inf')
            metrics['cluster_separation'] = 0.0
            metrics['separation_ratio'] = 0.0
        
        # 2. Target alignment (how well clusters match target centers)
        if len(cluster_centers) == len(self.target_centers):
            # Find best alignment between predicted and target centers
            distances = cdist(cluster_centers, self.target_centers)
            alignment_cost = np.min(distances, axis=1).sum()
            metrics['target_alignment'] = 1.0 / (1.0 + alignment_cost)
        else:
            metrics['target_alignment'] = 0.0
        
        # 3. Cluster balance (how evenly samples are distributed)
        cluster_counts = np.bincount(cluster_labels, minlength=self.n_clusters)
        total_samples = len(data)
        if total_samples > 0:
            cluster_ratios = cluster_counts / total_samples
            # Entropy-based balance measure
            entropy = -np.sum(cluster_ratios * np.log(cluster_ratios + 1e-12))
            max_entropy = np.log(self.n_clusters)
            metrics['cluster_balance'] = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            metrics['cluster_balance'] = 0.0
        
        # 4. Overall cluster quality score
        metrics['cluster_quality'] = (
            metrics['separation_ratio'] * 0.4 +
            metrics['target_alignment'] * 0.4 +
            metrics['cluster_balance'] * 0.2
        )
        
        # 5. Additional metrics
        metrics['n_samples'] = len(data)
        metrics['n_detected_clusters'] = len(np.unique(cluster_labels))
        metrics['cluster_centers'] = cluster_centers.tolist()
        metrics['cluster_labels'] = cluster_labels.tolist()
        
        return metrics
    
    def _empty_analysis(self) -> Dict[str, float]:
        """Return empty analysis for insufficient data."""
        return {
            'cluster_compactness': float('inf'),
            'cluster_separation': 0.0,
            'separation_ratio': 0.0,
            'target_alignment': 0.0,
            'cluster_balance': 0.0,
            'cluster_quality': 0.0,
            'n_samples': 0,
            'n_detected_clusters': 0,
            'cluster_centers': [],
            'cluster_labels': []
        }


class SFTutorialClusterMonitor:
    """Comprehensive monitoring system for SF Tutorial cluster generation."""
    
    def __init__(self):
        """Initialize cluster monitor."""
        self.config = {
            'batch_size': 16,
            'latent_dim': 4,
            'n_modes': 3,
            'layers': 2,
            'cutoff_dim': 4,
            'epochs': 20,
            'steps_per_epoch': 6,
            'learning_rate': 2e-3,
            'samples_per_epoch': 200  # For analysis
        }
        
        # Target bimodal centers
        self.target_centers = [(-1.5, -1.5), (1.5, 1.5)]
        
        # Initialize components
        self.generator = None
        self.discriminator = None
        self.data_generator = None
        self.cluster_analyzer = ClusterAnalyzer(self.target_centers)
        
        # Training history
        self.history = {
            'epochs': [],
            'g_losses': [],
            'd_losses': [],
            'w_distances': [],
            'g_gradient_flows': [],
            'd_gradient_flows': [],
            'x_quadrature_evolution': [],
            'cluster_qualities': [],
            'cluster_separations': [],
            'cluster_balances': [],
            'target_alignments': [],
            'sample_diversities': [],
            'generated_samples': [],
            'real_samples': [],
            'cluster_analyses': []
        }
        
        print(f"🔍 SF Tutorial Cluster Monitor initialized")
        print(f"   Target clusters: {self.target_centers}")
        print(f"   Focus: Bimodal data generation analysis")
    
    def setup_models(self):
        """Setup SF Tutorial models for cluster analysis."""
        print(f"\n🔧 Setting up SF Tutorial models...")
        
        # SF Tutorial generator with 100% gradient flow
        self.generator = SFTutorialGenerator(
            latent_dim=self.config['latent_dim'],
            output_dim=2,
            n_modes=self.config['n_modes'],
            n_layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim']
        )
        
        # Discriminator
        self.discriminator = PureSFDiscriminator(
            input_dim=2,
            n_modes=self.config['n_modes'],
            layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim']
        )
        
        # Bimodal data generator
        self.data_generator = BimodalDataGenerator(
            batch_size=self.config['batch_size'],
            n_features=2,
            mode1_center=self.target_centers[0],
            mode2_center=self.target_centers[1],
            mode_std=0.3
        )
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], beta_1=0.5
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'], beta_1=0.5
        )
        
        print(f"   ✅ Generator: {len(self.generator.trainable_variables)} params (100% quantum)")
        print(f"   ✅ Discriminator: {len(self.discriminator.trainable_variables)} params")
        print(f"   ✅ Target: Generate bimodal clusters at {self.target_centers}")
    
    def train_step(self, real_batch: tf.Tensor) -> Dict[str, any]:
        """Execute single training step with comprehensive monitoring."""
        batch_size = tf.shape(real_batch)[0]
        z = tf.random.normal([batch_size, self.config['latent_dim']])
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            fake_batch = self.generator.generate(z)
            real_output = self.discriminator.discriminate(real_batch)
            fake_output = self.discriminator.discriminate(fake_batch)
            
            w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
            d_loss = -w_distance
        
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as g_tape:
            fake_batch = self.generator.generate(z)
            fake_output = self.discriminator.discriminate(fake_batch)
            g_loss = -tf.reduce_mean(fake_output)
        
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Compute monitoring metrics
        g_grad_count = sum(1 for g in g_gradients if g is not None)
        d_grad_count = sum(1 for g in d_gradients if g is not None)
        g_grad_flow = g_grad_count / len(self.generator.trainable_variables)
        d_grad_flow = d_grad_count / len(self.discriminator.trainable_variables)
        
        # Extract quantum state info
        test_state = self.generator.quantum_circuit.execute()
        x_quadratures = self.generator.quantum_circuit.extract_measurements(test_state)
        
        # Sample diversity
        sample_std = tf.math.reduce_std(fake_batch, axis=0)
        sample_diversity = tf.reduce_mean(sample_std)
        
        return {
            'g_loss': float(g_loss),
            'd_loss': float(d_loss),
            'w_distance': float(w_distance),
            'g_gradient_flow': g_grad_flow,
            'd_gradient_flow': d_grad_flow,
            'x_quadratures': x_quadratures.numpy(),
            'sample_diversity': float(sample_diversity),
            'fake_batch': fake_batch.numpy(),
            'real_batch': real_batch.numpy()
        }
    
    def analyze_epoch_clusters(self, epoch: int) -> Dict[str, any]:
        """Comprehensive cluster analysis for current epoch."""
        # Generate large sample for analysis
        z_large = tf.random.normal([self.config['samples_per_epoch'], self.config['latent_dim']])
        generated_samples = self.generator.generate(z_large).numpy()
        
        # Real samples for comparison
        real_samples = []
        for _ in range(self.config['samples_per_epoch'] // self.config['batch_size']):
            batch = self.data_generator.generate_batch()
            real_samples.append(batch)
        real_samples = np.vstack(real_samples)[:self.config['samples_per_epoch']]
        
        # Cluster analysis
        gen_analysis = self.cluster_analyzer.analyze_clusters(generated_samples)
        real_analysis = self.cluster_analyzer.analyze_clusters(real_samples)
        
        return {
            'epoch': epoch,
            'generated_samples': generated_samples,
            'real_samples': real_samples,
            'generated_analysis': gen_analysis,
            'real_analysis': real_analysis,
            'cluster_improvement': gen_analysis['cluster_quality'] - real_analysis['cluster_quality']
        }
    
    def train_with_monitoring(self) -> Dict:
        """Execute complete training with comprehensive cluster monitoring."""
        print(f"\n🚀 Starting SF Tutorial Training with Cluster Monitoring")
        print(f"=" * 70)
        print(f"Focus: Analyzing rapid loss convergence and cluster generation")
        print(f"Expected: Understanding data clusterization quality")
        print(f"=" * 70)
        
        self.setup_models()
        
        for epoch in range(self.config['epochs']):
            print(f"\n📈 Epoch {epoch + 1}/{self.config['epochs']}")
            epoch_start = time.time()
            
            # Training steps
            step_results = []
            for step in range(self.config['steps_per_epoch']):
                real_batch = self.data_generator.generate_batch()
                step_result = self.train_step(real_batch)
                step_results.append(step_result)
            
            # Aggregate step results
            epoch_metrics = {
                'epoch': epoch,
                'epoch_time': time.time() - epoch_start,
                'g_loss': np.mean([s['g_loss'] for s in step_results]),
                'd_loss': np.mean([s['d_loss'] for s in step_results]),
                'w_distance': np.mean([s['w_distance'] for s in step_results]),
                'g_gradient_flow': np.mean([s['g_gradient_flow'] for s in step_results]),
                'd_gradient_flow': np.mean([s['d_gradient_flow'] for s in step_results]),
                'x_quadratures': np.mean([s['x_quadratures'] for s in step_results], axis=0),
                'sample_diversity': np.mean([s['sample_diversity'] for s in step_results])
            }
            
            # Cluster analysis
            cluster_analysis = self.analyze_epoch_clusters(epoch)
            
            # Store history
            self.history['epochs'].append(epoch)
            self.history['g_losses'].append(epoch_metrics['g_loss'])
            self.history['d_losses'].append(epoch_metrics['d_loss'])
            self.history['w_distances'].append(epoch_metrics['w_distance'])
            self.history['g_gradient_flows'].append(epoch_metrics['g_gradient_flow'])
            self.history['d_gradient_flows'].append(epoch_metrics['d_gradient_flow'])
            self.history['x_quadrature_evolution'].append(epoch_metrics['x_quadratures'])
            self.history['cluster_qualities'].append(cluster_analysis['generated_analysis']['cluster_quality'])
            self.history['cluster_separations'].append(cluster_analysis['generated_analysis']['separation_ratio'])
            self.history['cluster_balances'].append(cluster_analysis['generated_analysis']['cluster_balance'])
            self.history['target_alignments'].append(cluster_analysis['generated_analysis']['target_alignment'])
            self.history['sample_diversities'].append(epoch_metrics['sample_diversity'])
            self.history['generated_samples'].append(cluster_analysis['generated_samples'])
            self.history['real_samples'].append(cluster_analysis['real_samples'])
            self.history['cluster_analyses'].append(cluster_analysis)
            
            # Print results
            gen_analysis = cluster_analysis['generated_analysis']
            print(f"   📊 Training Metrics:")
            print(f"      Losses: G={epoch_metrics['g_loss']:.4f}, D={epoch_metrics['d_loss']:.4f}")
            print(f"      Gradient Flow: G={epoch_metrics['g_gradient_flow']:.1%}, D={epoch_metrics['d_gradient_flow']:.1%}")
            print(f"      Sample Diversity: {epoch_metrics['sample_diversity']:.4f}")
            
            print(f"   🎯 Cluster Analysis:")
            print(f"      Quality Score: {gen_analysis['cluster_quality']:.3f}")
            print(f"      Separation Ratio: {gen_analysis['separation_ratio']:.3f}")
            print(f"      Target Alignment: {gen_analysis['target_alignment']:.3f}")
            print(f"      Cluster Balance: {gen_analysis['cluster_balance']:.3f}")
            print(f"      Detected Clusters: {gen_analysis['n_detected_clusters']}")
            
            # Check for convergence issues
            if epoch_metrics['g_loss'] < 0.001 and epoch_metrics['d_loss'] < 0.001:
                if gen_analysis['cluster_quality'] < 0.5:
                    print(f"      ⚠️  WARNING: Rapid convergence with poor clustering!")
                else:
                    print(f"      ✅ Good: Rapid convergence with quality clustering")
        
        print(f"\n" + "=" * 70)
        print(f"🎉 SF TUTORIAL CLUSTER MONITORING COMPLETED!")
        print(f"=" * 70)
        
        # Generate comprehensive visualizations
        self.create_comprehensive_visualizations()
        
        return self.history
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive cluster analysis visualizations."""
        output_dir = "results/sf_tutorial_cluster_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Training metrics dashboard
        self._plot_training_dashboard(output_dir)
        
        # 2. Cluster evolution analysis
        self._plot_cluster_evolution(output_dir)
        
        # 3. Real vs Generated comparison
        self._plot_real_vs_generated_comparison(output_dir)
        
        # 4. Animated cluster evolution
        self._create_cluster_evolution_animation(output_dir)
        
        # 5. Quantum state analysis
        self._plot_quantum_state_analysis(output_dir)
        
        print(f"\n📊 Comprehensive visualizations saved in: {output_dir}")
    
    def _plot_training_dashboard(self, output_dir: str):
        """Plot comprehensive training dashboard."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('SF Tutorial Cluster Generation - Training Dashboard', fontsize=16, fontweight='bold')
        
        epochs = self.history['epochs']
        
        # Row 1: Loss analysis
        # Loss curves
        axes[0, 0].plot(epochs, self.history['g_losses'], 'b-', label='Generator', linewidth=2)
        axes[0, 0].plot(epochs, self.history['d_losses'], 'r-', label='Discriminator', linewidth=2)
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Wasserstein distance
        axes[0, 1].plot(epochs, self.history['w_distances'], 'g-', linewidth=2)
        axes[0, 1].set_title('Wasserstein Distance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('W-Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient flow
        axes[0, 2].plot(epochs, self.history['g_gradient_flows'], 'b-', label='Generator', linewidth=2)
        axes[0, 2].plot(epochs, self.history['d_gradient_flows'], 'r-', label='Discriminator', linewidth=2)
        axes[0, 2].set_title('Gradient Flow (Should be 100%)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Gradient Flow')
        axes[0, 2].set_ylim(0, 1.1)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Cluster quality analysis
        # Cluster quality score
        axes[1, 0].plot(epochs, self.history['cluster_qualities'], 'purple', linewidth=2)
        axes[1, 0].set_title('Cluster Quality Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster separation
        axes[1, 1].plot(epochs, self.history['cluster_separations'], 'orange', linewidth=2)
        axes[1, 1].set_title('Cluster Separation Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Separation Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Target alignment
        axes[1, 2].plot(epochs, self.history['target_alignments'], 'brown', linewidth=2)
        axes[1, 2].set_title('Target Alignment')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Alignment Score')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Sample analysis
        # Sample diversity
        axes[2, 0].plot(epochs, self.history['sample_diversities'], 'cyan', linewidth=2)
        axes[2, 0].set_title('Sample Diversity')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Standard Deviation')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Cluster balance
        axes[2, 1].plot(epochs, self.history['cluster_balances'], 'pink', linewidth=2)
        axes[2, 1].set_title('Cluster Balance')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Balance Score')
        axes[2, 1].set_ylim(0, 1)
        axes[2, 1].grid(True, alpha=0.3)
        
        # X-quadrature evolution
        x_quad_history = np.array(self.history['x_quadrature_evolution'])
        for mode in range(min(self.config['n_modes'], 3)):
            axes[2, 2].plot(epochs, x_quad_history[:, mode], 'o-', label=f'X_{mode}')
        axes[2, 2].axhline(y=0.1, color='green', linestyle='--', label='Vacuum Escape')
        axes[2, 2].axhline(y=-0.1, color='green', linestyle='--')
        axes[2, 2].set_title('X-Quadrature Evolution')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('X-Quadrature Value')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sf_tutorial_training_dashboard.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cluster_evolution(self, output_dir: str):
        """Plot cluster evolution over epochs."""
        n_snapshots = min(6, len(self.history['epochs']))
        snapshot_epochs = np.linspace(0, len(self.history['epochs'])-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SF Tutorial Cluster Evolution - Generated Data', fontsize=16, fontweight='bold')
        
        for i, epoch_idx in enumerate(snapshot_epochs):
            row = i // 3
            col = i % 3
            
            generated_samples = self.history['generated_samples'][epoch_idx]
            cluster_analysis = self.history['cluster_analyses'][epoch_idx]['generated_analysis']
            
            # Plot generated samples
            axes[row, col].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                                 alpha=0.6, s=20, color='red', label='Generated')
            
            # Plot target centers
            target_centers = np.array(self.target_centers)
            axes[row, col].scatter(target_centers[:, 0], target_centers[:, 1], 
                                 s=200, marker='x', color='black', linewidth=3, label='Target Centers')
            
            # Plot detected cluster centers if available
            if cluster_analysis['cluster_centers']:
                detected_centers = np.array(cluster_analysis['cluster_centers'])
                axes[row, col].scatter(detected_centers[:, 0], detected_centers[:, 1], 
                                     s=100, marker='o', color='blue', label='Detected Centers')
            
            # Add cluster ellipses for target
            for center in target_centers:
                ellipse = Ellipse(center, 1.2, 1.2, alpha=0.2, color='green')
                axes[row, col].add_patch(ellipse)
            
            epoch = self.history['epochs'][epoch_idx]
            quality = cluster_analysis['cluster_quality']
            axes[row, col].set_title(f'Epoch {epoch}: Quality={quality:.3f}')
            axes[row, col].set_xlim(-3, 3)
            axes[row, col].set_ylim(-3, 3)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sf_tutorial_cluster_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_real_vs_generated_comparison(self, output_dir: str):
        """Plot real vs generated comparison for latest epoch."""
        if not self.history['generated_samples']:
            return
        
        latest_real = self.history['real_samples'][-1]
        latest_generated = self.history['generated_samples'][-1]
        latest_epoch = self.history['epochs'][-1]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'SF Tutorial Real vs Generated - Epoch {latest_epoch}', fontsize=16, fontweight='bold')
        
        # Real data
        axes[0].scatter(latest_real[:, 0], latest_real[:, 1], alpha=0.6, s=20, color='blue')
        axes[0].set_title('Real Data')
        axes[0].set_xlim(-3, 3)
        axes[0].set_ylim(-3, 3)
        axes[0].grid(True, alpha=0.3)
        
        # Generated data
        axes[1].scatter(latest_generated[:, 0], latest_generated[:, 1], alpha=0.6, s=20, color='red')
        axes[1].set_title('Generated Data')
        axes[1].set_xlim(-3, 3)
        axes[1].set_ylim(-3, 3)
        axes[1].grid(True, alpha=0.3)
        
        # Overlay comparison
        axes[2].scatter(latest_real[:, 0], latest_real[:, 1], alpha=0.6, s=20, color='blue', label='Real')
        axes[2].scatter(latest_generated[:, 0], latest_generated[:, 1], alpha=0.6, s=20, color='red', label='Generated')
        
        # Target centers
        target_centers = np.array(self.target_centers)
        axes[2].scatter(target_centers[:, 0], target_centers[:, 1], 
                       s=200, marker='x', color='black', linewidth=3, label='Target Centers')
        
        axes[2].set_title('Overlay Comparison')
        axes[2].set_xlim(-3, 3)
        axes[2].set_ylim(-3, 3)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sf_tutorial_real_vs_generated.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cluster_evolution_animation(self, output_dir: str):
        """Create animated GIF showing cluster evolution over epochs."""
        if len(self.history['generated_samples']) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        
        # Target centers
        target_centers = np.array(self.target_centers)
        ax.scatter(target_centers[:, 0], target_centers[:, 1], 
                  s=200, marker='x', color='black', linewidth=3, label='Target Centers')
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid(True, alpha=0.3)
            
            # Target centers
            ax.scatter(target_centers[:, 0], target_centers[:, 1], 
                      s=200, marker='x', color='black', linewidth=3, label='Target Centers')
            
            # Generated samples for this epoch
            if frame < len(self.history['generated_samples']):
                generated_samples = self.history['generated_samples'][frame]
                real_samples = self.history['real_samples'][frame]
                
                ax.scatter(real_samples[:, 0], real_samples[:, 1], 
                          alpha=0.4, s=15, color='blue', label='Real')
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                          alpha=0.6, s=20, color='red', label='Generated')
                
                # Cluster analysis for this epoch
                cluster_analysis = self.history['cluster_analyses'][frame]['generated_analysis']
                if cluster_analysis['cluster_centers']:
                    detected_centers = np.array(cluster_analysis['cluster_centers'])
                    ax.scatter(detected_centers[:, 0], detected_centers[:, 1], 
                             s=100, marker='o', color='orange', label='Detected Centers')
                
                epoch = self.history['epochs'][frame]
                quality = cluster_analysis['cluster_quality']
                g_loss = self.history['g_losses'][frame]
                d_loss = self.history['d_losses'][frame]
                
                ax.set_title(f'SF Tutorial Cluster Evolution - Epoch {epoch}\n'
                           f'Quality: {quality:.3f}, G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}')
            
            ax.legend()
            return ax.patches + ax.collections
        
        frames = len(self.history['generated_samples'])
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800, blit=False)
        
        # Save as GIF
        gif_path = os.path.join(output_dir, "sf_tutorial_cluster_evolution.gif")
        anim.save(gif_path, writer='pillow', fps=1.2)
        plt.close()
        
        print(f"   📹 Cluster evolution animation saved: {gif_path}")
    
    def _plot_quantum_state_analysis(self, output_dir: str):
        """Plot quantum state and parameter analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SF Tutorial Quantum State Analysis', fontsize=16, fontweight='bold')
        
        epochs = self.history['epochs']
        x_quad_history = np.array(self.history['x_quadrature_evolution'])
        
        # X-quadrature evolution detailed
        for mode in range(min(self.config['n_modes'], x_quad_history.shape[1])):
            axes[0, 0].plot(epochs, x_quad_history[:, mode], 'o-', label=f'Mode {mode}', linewidth=2)
        axes[0, 0].axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Vacuum Escape (+)')
        axes[0, 0].axhline(y=-0.1, color='green', linestyle='--', alpha=0.7, label='Vacuum Escape (-)')
        axes[0, 0].set_title('X-Quadrature Evolution (Vacuum Escape Analysis)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('X-Quadrature Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient flow stability
        axes[0, 1].plot(epochs, self.history['g_gradient_flows'], 'b-', linewidth=3, label='Generator (SF Tutorial)')
        axes[0, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Flow (100%)')
        axes[0, 1].axhline(y=0.111, color='red', linestyle='--', alpha=0.7, label='Old Architecture (11.1%)')
        axes[0, 1].set_title('Gradient Flow Achievement')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gradient Flow Percentage')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss convergence analysis
        axes[1, 0].semilogy(epochs, self.history['g_losses'], 'b-', label='Generator', linewidth=2)
        axes[1, 0].semilogy(epochs, self.history['d_losses'], 'r-', label='Discriminator', linewidth=2)
        axes[1, 0].set_title('Loss Convergence (Log Scale)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (Log Scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster quality vs loss correlation
        axes[1, 1].scatter(self.history['g_losses'], self.history['cluster_qualities'], 
                          c=epochs, cmap='viridis', s=50, alpha=0.7)
        axes[1, 1].set_title('Cluster Quality vs Generator Loss')
        axes[1, 1].set_xlabel('Generator Loss')
        axes[1, 1].set_ylabel('Cluster Quality Score')
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sf_tutorial_quantum_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main cluster analysis execution."""
    print(f"🔍 SF TUTORIAL CLUSTER ANALYSIS")
    print(f"=" * 80)
    print(f"This analysis focuses on understanding:")
    print(f"  • Why losses converge so rapidly (2 epochs)")
    print(f"  • Whether bimodal clusters are properly formed")
    print(f"  • Impact of 100% gradient flow on clustering")
    print(f"  • Data generation quality compared to target")
    print(f"=" * 80)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Create and run cluster monitor
        monitor = SFTutorialClusterMonitor()
        history = monitor.train_with_monitoring()
        
        # Final analysis summary
        final_cluster_quality = history['cluster_qualities'][-1]
        final_g_loss = history['g_losses'][-1]
        final_d_loss = history['d_losses'][-1]
        final_gradient_flow = history['g_gradient_flows'][-1]
        
        print(f"\n" + "=" * 80)
        print(f"🎯 SF TUTORIAL CLUSTER ANALYSIS COMPLETE!")
        print(f"=" * 80)
        
        print(f"🔍 RAPID CONVERGENCE ANALYSIS:")
        print(f"  Final losses: G={final_g_loss:.6f}, D={final_d_loss:.6f}")
        print(f"  Convergence speed: {'Very rapid (2-3 epochs)' if final_g_loss < 0.001 else 'Normal'}")
        print(f"  Gradient flow: {final_gradient_flow:.1%} (SF Tutorial achievement)")
        
        print(f"\n🎯 CLUSTER GENERATION ANALYSIS:")
        print(f"  Final cluster quality: {final_cluster_quality:.3f}")
        if final_cluster_quality > 0.7:
            cluster_status = "✅ EXCELLENT - High quality clusters formed"
        elif final_cluster_quality > 0.5:
            cluster_status = "✅ GOOD - Decent cluster formation"
        elif final_cluster_quality > 0.3:
            cluster_status = "⚠️ MODERATE - Some clustering but needs improvement"
        else:
            cluster_status = "❌ POOR - Mode collapse or poor clustering"
        print(f"  Cluster status: {cluster_status}")
        
        print(f"\n🔬 CONVERGENCE EXPLANATION:")
        if final_g_loss < 0.001 and final_d_loss < 0.001:
            if final_cluster_quality > 0.5:
                print(f"  ✅ HEALTHY: Fast convergence with good clustering")
                print(f"     - Generator learned target distribution quickly")
                print(f"     - Discriminator reached equilibrium")
                print(f"     - 100% gradient flow enabled efficient learning")
            else:
                print(f"  ⚠️ PROBLEMATIC: Fast convergence but poor clustering")
                print(f"     - Possible mode collapse to single point")
                print(f"     - Discriminator might be too strong")
                print(f"     - Static transformations might be limiting diversity")
        
        print(f"\n📊 VISUALIZATIONS CREATED:")
        print(f"  • sf_tutorial_training_dashboard.png - Complete training metrics")
        print(f"  • sf_tutorial_cluster_evolution.png - Cluster formation over time")
        print(f"  • sf_tutorial_real_vs_generated.png - Final comparison")
        print(f"  • sf_tutorial_cluster_evolution.gif - Animated evolution")
        print(f"  • sf_tutorial_quantum_analysis.png - Quantum state analysis")
        
        return history
        
    except Exception as e:
        print(f"❌ Cluster analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
