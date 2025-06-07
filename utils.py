import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qm9_data(data_path="data/qm9", num_samples=1000, feature_dim=30):
    """
    Load and preprocess QM9 molecular dataset.
    For now, creates synthetic molecular-like data until real QM9 is available.
    
    Args:
        data_path (str): Path to QM9 dataset
        num_samples (int): Number of samples to load
        feature_dim (int): Dimension of molecular features
        
    Returns:
        tf.Tensor: Preprocessed molecular descriptors
    """
    logger.info(f"Loading QM9 data from {data_path}")
    
    # Check if real QM9 data exists
    qm9_file = os.path.join(data_path, "qm9_features.npy")
    
    if os.path.exists(qm9_file):
        logger.info("Loading real QM9 data")
        data = np.load(qm9_file)[:num_samples]
    else:
        logger.warning("QM9 data not found. Generating synthetic molecular-like data.")
        # Create synthetic molecular descriptors
        # Features should be realistic molecular properties
        np.random.seed(42)
        
        # Generate realistic molecular features
        molecular_weight = np.random.normal(150, 50, num_samples)  # Typical drug MW
        logp = np.random.normal(2.5, 1.5, num_samples)  # LogP values
        num_atoms = np.random.poisson(15, num_samples)  # Number of atoms
        
        # Additional features
        other_features = np.random.normal(0, 1, (num_samples, feature_dim - 3))
        
        # Combine features
        data = np.column_stack([molecular_weight, logp, num_atoms, other_features])
        
        # Ensure data directory exists
        os.makedirs(data_path, exist_ok=True)
        
        # Save synthetic data for consistency
        np.save(qm9_file, data)
        logger.info(f"Saved synthetic QM9 data to {qm9_file}")
    
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    logger.info(f"Loaded {data_normalized.shape[0]} samples with {data_normalized.shape[1]} features")
    
    return tf.convert_to_tensor(data_normalized, dtype=tf.float32)

def load_synthetic_data(dataset_type="spiral", num_samples=1000):
    """
    Load synthetic 2D datasets for testing and visualization.
    
    Args:
        dataset_type (str): Type of dataset ('spiral', 'moons', 'circles', 'gaussian')
        num_samples (int): Number of samples to generate
        
    Returns:
        tf.Tensor: Generated 2D data
    """
    logger.info(f"Generating {dataset_type} synthetic data with {num_samples} samples")
    
    if dataset_type == "spiral":
        # Generate spiral data
        t = np.linspace(0, 4*np.pi, num_samples)
        x = t * np.cos(t) + np.random.normal(0, 0.1, num_samples)
        y = t * np.sin(t) + np.random.normal(0, 0.1, num_samples)
        data = np.column_stack([x, y])
        
    elif dataset_type == "moons":
        data, _ = make_moons(n_samples=num_samples, noise=0.1, random_state=42)
        
    elif dataset_type == "circles":
        data, _ = make_circles(n_samples=num_samples, noise=0.1, random_state=42)
        
    elif dataset_type == "gaussian":
        # Multi-modal Gaussian mixture
        data1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], num_samples//2)
        data2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], num_samples//2)
        data = np.vstack([data1, data2])
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    return tf.convert_to_tensor(data_normalized, dtype=tf.float32)

def plot_results(real_data, generated_data, epoch=None, save_path=None):
    """
    Plot comparison between real and generated data.
    
    Args:
        real_data (tf.Tensor): Real data samples
        generated_data (tf.Tensor): Generated data samples
        epoch (int): Current training epoch
        save_path (str): Path to save the plot
    """
    # Convert to numpy for plotting
    real_np = real_data.numpy()
    gen_np = generated_data.numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot real data
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
        
        # Plot overlay
        axes[2].scatter(real_np[:, 0], real_np[:, 1], alpha=0.4, s=15, label='Real')
        axes[2].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.4, s=15, label='Generated')
        axes[2].set_title("Overlay Comparison")
        axes[2].set_xlabel("Feature 1")
        axes[2].set_ylabel("Feature 2")
        axes[2].legend()
    else:
        # For high-dimensional data, plot histograms
        for i in range(min(3, real_np.shape[1])):
            axes[0].hist(real_np[:, i], alpha=0.7, bins=30, label=f'Feature {i+1}')
        axes[0].set_title("Real Data Distribution")
        axes[0].legend()
        
        for i in range(min(3, gen_np.shape[1])):
            axes[1].hist(gen_np[:, i], alpha=0.7, bins=30, label=f'Feature {i+1}')
        axes[1].set_title("Generated Data Distribution")
        axes[1].legend()
        
        # Comparison plot
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

def compute_wasserstein_distance(real_samples, generated_samples):
    """
    Compute approximate Wasserstein distance between real and generated samples.
    
    Args:
        real_samples (tf.Tensor): Real data samples
        generated_samples (tf.Tensor): Generated data samples
        
    Returns:
        float: Wasserstein distance estimate
    """
    # Convert to numpy
    real_np = real_samples.numpy()
    gen_np = generated_samples.numpy()
    
    # For high-dimensional data, use mean distance approximation
    real_mean = np.mean(real_np, axis=0)
    gen_mean = np.mean(gen_np, axis=0)
    
    # L2 distance between means (simplified Wasserstein)
    distance = np.linalg.norm(real_mean - gen_mean)
    
    return distance

def compute_mmd(real_samples, generated_samples, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between real and generated samples.
    
    Args:
        real_samples (tf.Tensor): Real data samples
        generated_samples (tf.Tensor): Generated data samples
        kernel (str): Kernel type ('rbf', 'linear')
        gamma (float): Kernel parameter
        
    Returns:
        float: MMD value
    """
    # Convert to numpy
    X = real_samples.numpy()
    Y = generated_samples.numpy()
    
    def kernel_func(x, y):
        if kernel == 'rbf':
            return np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        elif kernel == 'linear':
            return np.dot(x, y)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    
    # Compute kernel matrices
    n, m = len(X), len(Y)
    
    # K(X, X)
    K_XX = np.mean([kernel_func(X[i], X[j]) for i in range(n) for j in range(n)])
    
    # K(Y, Y)
    K_YY = np.mean([kernel_func(Y[i], Y[j]) for i in range(m) for j in range(m)])
    
    # K(X, Y)
    K_XY = np.mean([kernel_func(X[i], Y[j]) for i in range(n) for j in range(m)])
    
    # MMD^2 = K(X,X) + K(Y,Y) - 2*K(X,Y)
    mmd_squared = K_XX + K_YY - 2 * K_XY
    
    return np.sqrt(max(0, mmd_squared))

def save_model(model, filepath):
    """
    Save model parameters to file.
    
    Args:
        model: Model object to save
        filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if hasattr(model, 'trainable_variables'):
        # TensorFlow model
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.save(filepath)
        logger.info(f"Model saved to {filepath}")
    else:
        logger.warning("Model type not recognized for saving")

def load_model(model, filepath):
    """
    Load model parameters from file.
    
    Args:
        model: Model object to load parameters into
        filepath (str): Path to load the model from
    """
    if hasattr(model, 'trainable_variables'):
        # TensorFlow model
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(filepath)
        logger.info(f"Model loaded from {filepath}")
    else:
        logger.warning("Model type not recognized for loading")

def create_output_directory(base_path="results"):
    """
    Create output directory structure for saving results.
    
    Args:
        base_path (str): Base path for results
        
    Returns:
        str: Created directory path
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"qgan_run_{timestamp}")
    
    # Create subdirectories
    subdirs = ['plots', 'models', 'logs', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return output_dir 