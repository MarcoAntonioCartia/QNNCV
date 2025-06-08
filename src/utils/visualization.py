import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Try to import TensorFlow, but don't fail if it's not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_results(real_data, generated_data, epoch=None, save_path=None):
    """
    Create comparative visualization of real and generated data distributions.
    
    This function generates side-by-side plots comparing real and generated
    data distributions. For 2D data, scatter plots are used; for higher
    dimensional data, histogram comparisons are created.
    
    Args:
        real_data: Real data samples (numpy array or tf.Tensor)
        generated_data: Generated data samples (numpy array or tf.Tensor)
        epoch (int): Current training epoch for plot title
        save_path (str): Path to save the plot image
    """
    # Convert tensors to numpy arrays for plotting
    if TF_AVAILABLE and hasattr(real_data, 'numpy'):
        real_np = real_data.numpy()
        gen_np = generated_data.numpy()
    else:
        real_np = np.array(real_data)
        gen_np = np.array(generated_data)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Handle 2D data with scatter plots
    if real_np.shape[1] == 2:
        axes[0].scatter(real_np[:, 0], real_np[:, 1], alpha=0.6, s=20)
        axes[0].set_title("Real Data")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")
        
        # Plot generated data
        axes[1].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.6, s=20, color='red')
        axes[1].set_title("Generated Data")
        axes[1].set_xlabel("Feature 1")
        axes[1].set_ylabel("Feature 2")
        
        # Plot overlay comparison
        axes[2].scatter(real_np[:, 0], real_np[:, 1], alpha=0.4, s=15, label='Real')
        axes[2].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.4, s=15, label='Generated')
        axes[2].set_title("Overlay Comparison")
        axes[2].set_xlabel("Feature 1")
        axes[2].set_ylabel("Feature 2")
        axes[2].legend()
    else:
        # Handle high-dimensional data with histograms
        for i in range(min(3, real_np.shape[1])):
            axes[0].hist(real_np[:, i], alpha=0.7, bins=30, label=f'Feature {i+1}')
        axes[0].set_title("Real Data Distribution")
        axes[0].legend()
        
        for i in range(min(3, gen_np.shape[1])):
            axes[1].hist(gen_np[:, i], alpha=0.7, bins=30, label=f'Feature {i+1}')
        axes[1].set_title("Generated Data Distribution")
        axes[1].legend()
        
        # Comparison histogram
        for i in range(min(2, real_np.shape[1])):
            axes[2].hist(real_np[:, i], alpha=0.5, bins=30, label=f'Real F{i+1}')
            axes[2].hist(gen_np[:, i], alpha=0.5, bins=30, label=f'Gen F{i+1}')
        axes[2].set_title("Distribution Comparison")
        axes[2].legend()
    
    if epoch is not None:
        fig.suptitle(f"Epoch {epoch}")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_training_history(history, save_path=None):
    """
    Visualize training history including loss curves and metrics.
    
    This function creates plots showing the evolution of generator and
    discriminator losses, as well as other training metrics over epochs.
    
    Args:
        history (dict): Training history containing loss and metric values
        save_path (str): Path to save the plot image
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot generator and discriminator losses
    if 'g_loss' in history and 'd_loss' in history:
        epochs = range(len(history['g_loss']))
        axes[0, 0].plot(epochs, history['g_loss'], label='Generator Loss')
        axes[0, 0].plot(epochs, history['d_loss'], label='Discriminator Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot gradient norms if available
    if 'g_grad_norm' in history and 'd_grad_norm' in history:
        epochs = range(len(history['g_grad_norm']))
        axes[0, 1].plot(epochs, history['g_grad_norm'], label='Generator Grad Norm')
        axes[0, 1].plot(epochs, history['d_grad_norm'], label='Discriminator Grad Norm')
        axes[0, 1].set_title('Gradient Norms')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot stability metric if available
    if 'stability_metric' in history:
        epochs = range(len(history['stability_metric']))
        axes[1, 0].plot(epochs, history['stability_metric'])
        axes[1, 0].set_title('Training Stability Metric')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Stability Ratio')
        axes[1, 0].grid(True)
    
    # Plot additional metrics if available
    if 'gradient_penalty' in history:
        epochs = range(len(history['gradient_penalty']))
        axes[1, 1].plot(epochs, history['gradient_penalty'])
        axes[1, 1].set_title('Gradient Penalty')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Penalty Value')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()
