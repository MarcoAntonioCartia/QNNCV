"""
Bimodal 2D Visualization Tracker

Tracks real vs fake data evolution during training and creates animated visualizations.
Shows what the X-quadrature decoder is actually outputting in 2D space.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json


class BimodalVisualizationTracker:
    """
    Tracks real vs fake 2D data evolution during training.
    
    Creates animated GIFs showing how generated data evolves over epochs.
    Analyzes decoder output quality and bimodal distribution learning.
    """
    
    def __init__(self, 
                 save_dir: str,
                 target_centers: List[Tuple[float, float]] = None,
                 xlim: Tuple[float, float] = (-3, 3),
                 ylim: Tuple[float, float] = (-3, 3)):
        """
        Initialize visualization tracker.
        
        Args:
            save_dir: Directory to save visualizations
            target_centers: Expected cluster centers for reference
            xlim: X-axis limits for plots
            ylim: Y-axis limits for plots
        """
        self.save_dir = save_dir
        self.target_centers = target_centers or [(-1.5, -1.5), (1.5, 1.5)]
        self.xlim = xlim
        self.ylim = ylim
        
        # Training data storage
        self.real_data_history = []
        self.fake_data_history = []
        self.epoch_history = []
        self.x_quadrature_history = []
        self.decoder_output_history = []
        
        # Analysis metrics
        self.bimodal_metrics_history = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"📊 BimodalVisualizationTracker initialized:")
        print(f"   Save directory: {save_dir}")
        print(f"   Target centers: {self.target_centers}")
        print(f"   Plot limits: X{xlim}, Y{ylim}")
    
    def update(self, 
               epoch: int,
               real_batch: np.ndarray,
               fake_batch: np.ndarray,
               x_quadrature_batch: Optional[np.ndarray] = None,
               decoder_output: Optional[np.ndarray] = None):
        """
        Update tracker with new epoch data.
        
        Args:
            epoch: Current training epoch
            real_batch: Real data samples [batch_size, 2]
            fake_batch: Generated data samples [batch_size, 2]
            x_quadrature_batch: X-quadrature measurements [batch_size, n_modes]
            decoder_output: Raw decoder output before any transformations
        """
        # Store epoch data
        self.epoch_history.append(epoch)
        self.real_data_history.append(real_batch.copy())
        self.fake_data_history.append(fake_batch.copy())
        
        if x_quadrature_batch is not None:
            self.x_quadrature_history.append(x_quadrature_batch.copy())
        
        if decoder_output is not None:
            self.decoder_output_history.append(decoder_output.copy())
        
        # Compute and store bimodal metrics
        metrics = self._compute_bimodal_metrics(real_batch, fake_batch)
        self.bimodal_metrics_history.append(metrics)
        
        print(f"📊 Epoch {epoch}: Coverage={metrics['coverage']:.3f}, "
              f"Separation={metrics['separation']:.3f}, "
              f"Balance={metrics['balance']:.3f}")
    
    def _compute_bimodal_metrics(self, real_batch: np.ndarray, fake_batch: np.ndarray) -> Dict[str, float]:
        """Compute bimodal distribution quality metrics."""
        # Mode coverage: how well fake data covers both target modes
        coverage_scores = []
        for center in self.target_centers:
            center_np = np.array(center)
            distances = np.linalg.norm(fake_batch - center_np, axis=1)
            min_distance = np.min(distances)
            coverage_scores.append(np.exp(-min_distance))  # Higher is better
        
        coverage = np.mean(coverage_scores)
        
        # Mode separation: how well separated the generated clusters are
        if len(fake_batch) > 1:
            fake_center = np.mean(fake_batch, axis=0)
            fake_distances = np.linalg.norm(fake_batch - fake_center, axis=1)
            separation = np.std(fake_distances)
        else:
            separation = 0.0
        
        # Mode balance: how balanced the distribution is between modes
        center1_distances = np.linalg.norm(fake_batch - np.array(self.target_centers[0]), axis=1)
        center2_distances = np.linalg.norm(fake_batch - np.array(self.target_centers[1]), axis=1)
        
        mode1_assignments = np.sum(center1_distances < center2_distances)
        mode2_assignments = len(fake_batch) - mode1_assignments
        
        if len(fake_batch) > 0:
            balance = 1.0 - abs(mode1_assignments - mode2_assignments) / len(fake_batch)
        else:
            balance = 0.0
        
        return {
            'coverage': coverage,
            'separation': separation,
            'balance': balance,
            'mode1_assignments': mode1_assignments,
            'mode2_assignments': mode2_assignments
        }
    
    def create_animated_gif(self, filename: str = "bimodal_evolution.gif", fps: int = 2) -> str:
        """
        Create animated GIF showing real vs fake data evolution.
        
        Args:
            filename: Output filename for GIF
            fps: Frames per second for animation
            
        Returns:
            Path to created GIF file
        """
        if len(self.real_data_history) == 0:
            print("❌ No data to visualize")
            return None
        
        print(f"🎬 Creating animated GIF with {len(self.real_data_history)} frames...")
        
        # Set up figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        # Plot target centers as reference
        for i, center in enumerate(self.target_centers):
            circle = plt.Circle(center, 0.3, color='gray', alpha=0.3, fill=False, linestyle='--')
            ax.add_patch(circle)
            ax.text(center[0], center[1] + 0.5, f'Target {i+1}', 
                   ha='center', va='center', fontsize=10, color='gray')
        
        # Initialize empty scatter plots
        real_scatter = ax.scatter([], [], c='blue', alpha=0.6, s=30, label='Real Data')
        fake_scatter = ax.scatter([], [], c='red', alpha=0.6, s=30, label='Generated Data')
        
        # Add title and legend
        title = ax.set_title('')
        ax.legend(loc='upper right')
        
        # Animation update function
        def update_frame(frame_idx):
            epoch = self.epoch_history[frame_idx]
            real_data = self.real_data_history[frame_idx]
            fake_data = self.fake_data_history[frame_idx]
            metrics = self.bimodal_metrics_history[frame_idx]
            
            # Update scatter plots
            real_scatter.set_offsets(real_data)
            fake_scatter.set_offsets(fake_data)
            
            # Update title with metrics
            title.set_text(f'Epoch {epoch} | Coverage: {metrics["coverage"]:.3f} | '
                          f'Separation: {metrics["separation"]:.3f} | '
                          f'Balance: {metrics["balance"]:.3f}')
            
            return real_scatter, fake_scatter, title
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(self.real_data_history),
            interval=1000//fps, blit=False, repeat=True
        )
        
        # Save as GIF
        gif_path = os.path.join(self.save_dir, filename)
        anim.save(gif_path, writer='pillow', fps=fps)
        plt.close(fig)
        
        print(f"✅ Animated GIF saved: {gif_path}")
        return gif_path
    
    def create_decoder_analysis_plot(self, filename: str = "decoder_analysis.png") -> str:
        """
        Create analysis plot showing X-quadrature → decoder output mapping.
        
        Args:
            filename: Output filename for plot
            
        Returns:
            Path to created plot file
        """
        if len(self.x_quadrature_history) == 0 or len(self.decoder_output_history) == 0:
            print("❌ No X-quadrature or decoder data to analyze")
            return None
        
        print(f"📊 Creating decoder analysis plot...")
        
        # Combine all X-quadrature and decoder output data
        all_x_quad = np.vstack(self.x_quadrature_history)
        all_decoder_out = np.vstack(self.decoder_output_history)
        
        n_modes = all_x_quad.shape[1]
        
        # Create subplots for analysis
        fig, axes = plt.subplots(2, n_modes, figsize=(4*n_modes, 8))
        if n_modes == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('X-Quadrature → Decoder Output Analysis', fontsize=14, fontweight='bold')
        
        # Plot X-quadrature vs X coordinate
        for mode in range(n_modes):
            # X-quadrature vs X output
            axes[0, mode].scatter(all_x_quad[:, mode], all_decoder_out[:, 0], 
                                alpha=0.5, s=10, c='blue')
            axes[0, mode].set_title(f'Mode {mode}: X-quad → X Output')
            axes[0, mode].set_xlabel(f'X-quadrature Mode {mode}')
            axes[0, mode].set_ylabel('X Output')
            axes[0, mode].grid(True, alpha=0.3)
            
            # X-quadrature vs Y output
            axes[1, mode].scatter(all_x_quad[:, mode], all_decoder_out[:, 1], 
                                alpha=0.5, s=10, c='red')
            axes[1, mode].set_title(f'Mode {mode}: X-quad → Y Output')
            axes[1, mode].set_xlabel(f'X-quadrature Mode {mode}')
            axes[1, mode].set_ylabel('Y Output')
            axes[1, mode].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Decoder analysis plot saved: {plot_path}")
        return plot_path
    
    def create_evolution_summary_plot(self, filename: str = "evolution_summary.png") -> str:
        """
        Create summary plot showing metric evolution over training.
        
        Args:
            filename: Output filename for plot
            
        Returns:
            Path to created plot file
        """
        if len(self.bimodal_metrics_history) == 0:
            print("❌ No metrics to plot")
            return None
        
        print(f"📈 Creating evolution summary plot...")
        
        # Extract metric arrays
        epochs = np.array(self.epoch_history)
        coverage = [m['coverage'] for m in self.bimodal_metrics_history]
        separation = [m['separation'] for m in self.bimodal_metrics_history]
        balance = [m['balance'] for m in self.bimodal_metrics_history]
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Bimodal Generation Quality Evolution', fontsize=14, fontweight='bold')
        
        # Coverage evolution
        axes[0, 0].plot(epochs, coverage, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_title('Mode Coverage')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Coverage Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Separation evolution
        axes[0, 1].plot(epochs, separation, 'r-', linewidth=2, marker='o', markersize=4)
        axes[0, 1].set_title('Mode Separation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Separation Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Balance evolution
        axes[1, 0].plot(epochs, balance, 'g-', linewidth=2, marker='o', markersize=4)
        axes[1, 0].set_title('Mode Balance')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Balance Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Final distribution snapshot
        if len(self.fake_data_history) > 0:
            final_fake = self.fake_data_history[-1]
            axes[1, 1].scatter(final_fake[:, 0], final_fake[:, 1], 
                             c='red', alpha=0.6, s=30, label='Final Generated')
            
            # Plot target centers
            for i, center in enumerate(self.target_centers):
                circle = plt.Circle(center, 0.3, color='gray', alpha=0.3, fill=False, linestyle='--')
                axes[1, 1].add_patch(circle)
                axes[1, 1].text(center[0], center[1] + 0.5, f'Target {i+1}', 
                               ha='center', va='center', fontsize=8, color='gray')
            
            axes[1, 1].set_title('Final Generated Distribution')
            axes[1, 1].set_xlabel('X Coordinate')
            axes[1, 1].set_ylabel('Y Coordinate')
            axes[1, 1].set_xlim(self.xlim)
            axes[1, 1].set_ylim(self.ylim)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Evolution summary plot saved: {plot_path}")
        return plot_path
    
    def save_metrics_json(self, filename: str = "bimodal_metrics.json") -> str:
        """
        Save all metrics to JSON file for further analysis.
        
        Args:
            filename: Output filename for JSON
            
        Returns:
            Path to created JSON file
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        metrics_data = {
            'epochs': convert_to_native(self.epoch_history),
            'bimodal_metrics': convert_to_native(self.bimodal_metrics_history),
            'target_centers': convert_to_native(self.target_centers),
            'summary': {
                'total_epochs': len(self.epoch_history),
                'final_coverage': convert_to_native(self.bimodal_metrics_history[-1]['coverage']) if self.bimodal_metrics_history else 0,
                'final_separation': convert_to_native(self.bimodal_metrics_history[-1]['separation']) if self.bimodal_metrics_history else 0,
                'final_balance': convert_to_native(self.bimodal_metrics_history[-1]['balance']) if self.bimodal_metrics_history else 0,
            }
        }
        
        json_path = os.path.join(self.save_dir, filename)
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"📄 Metrics JSON saved: {json_path}")
        return json_path
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate all visualization files.
        
        Returns:
            Dictionary mapping visualization type to file path
        """
        print(f"🎨 Generating all bimodal visualizations...")
        
        files = {}
        
        # Animated GIF
        files['animated_gif'] = self.create_animated_gif()
        
        # Decoder analysis (if data available)
        if len(self.x_quadrature_history) > 0:
            files['decoder_analysis'] = self.create_decoder_analysis_plot()
        
        # Evolution summary
        files['evolution_summary'] = self.create_evolution_summary_plot()
        
        # Metrics JSON
        files['metrics_json'] = self.save_metrics_json()
        
        print(f"✅ All bimodal visualizations generated!")
        return files


def create_bimodal_tracker(save_dir: str, 
                          target_centers: List[Tuple[float, float]] = None) -> BimodalVisualizationTracker:
    """
    Factory function to create bimodal visualization tracker.
    
    Args:
        save_dir: Directory to save visualizations
        target_centers: Expected cluster centers for reference
        
    Returns:
        Configured BimodalVisualizationTracker
    """
    return BimodalVisualizationTracker(
        save_dir=save_dir,
        target_centers=target_centers or [(-1.5, -1.5), (1.5, 1.5)]
    )
