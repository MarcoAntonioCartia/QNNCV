import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

# Try to import TensorFlow, but don't fail if it's not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import QuantumMetrics from quantum_metrics.py
from .quantum_metrics import QuantumMetrics

def compute_wasserstein_distance(real_samples, generated_samples):
    """
    Compute approximate Wasserstein distance between real and generated samples.
    
    This function calculates a simplified approximation of the Wasserstein
    distance by computing the L2 distance between the means of the two
    distributions. This provides a quick metric for comparing distributions.
    
    Args:
        real_samples: Real data samples (numpy array or tf.Tensor)
        generated_samples: Generated data samples (numpy array or tf.Tensor)
        
    Returns:
        float: Wasserstein distance estimate
    """
    # Convert tensors to numpy arrays
    if TF_AVAILABLE and hasattr(real_samples, 'numpy'):
        real_np = real_samples.numpy()
        gen_np = generated_samples.numpy()
    else:
        real_np = np.array(real_samples)
        gen_np = np.array(generated_samples)
    
    # Compute means of both distributions
    real_mean = np.mean(real_np, axis=0)
    gen_mean = np.mean(gen_np, axis=0)
    
    # L2 distance between means (simplified Wasserstein approximation)
    distance = np.linalg.norm(real_mean - gen_mean)
    
    return distance

def compute_mmd(real_samples, generated_samples, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between real and generated samples.
    
    MMD is a statistical test that measures the distance between two
    distributions by comparing their embeddings in a reproducing kernel
    Hilbert space. This provides a principled way to compare distributions.
    
    Args:
        real_samples: Real data samples (numpy array or tf.Tensor)
        generated_samples: Generated data samples (numpy array or tf.Tensor)
        kernel (str): Kernel type ('rbf', 'linear')
        gamma (float): Kernel parameter for RBF kernel
        
    Returns:
        float: MMD value
    """
    # Convert tensors to numpy arrays
    if TF_AVAILABLE and hasattr(real_samples, 'numpy'):
        X = real_samples.numpy()
        Y = generated_samples.numpy()
    else:
        X = np.array(real_samples)
        Y = np.array(generated_samples)
    
    def kernel_func(x, y):
        """Compute kernel function between two vectors."""
        if kernel == 'rbf':
            return np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        elif kernel == 'linear':
            return np.dot(x, y)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    
    # Compute kernel matrices
    n, m = len(X), len(Y)
    
    # K(X, X) - kernel matrix for real samples
    K_XX = np.mean([kernel_func(X[i], X[j]) for i in range(n) for j in range(n)])
    
    # K(Y, Y) - kernel matrix for generated samples
    K_YY = np.mean([kernel_func(Y[i], Y[j]) for i in range(m) for j in range(m)])
    
    # K(X, Y) - cross kernel matrix
    K_XY = np.mean([kernel_func(X[i], Y[j]) for i in range(n) for j in range(m)])
    
    # MMD^2 = K(X,X) + K(Y,Y) - 2*K(X,Y)
    mmd_squared = K_XX + K_YY - 2 * K_XY
    
    return np.sqrt(max(0, mmd_squared))

def compute_coverage_and_precision(real_samples, generated_samples, k=5):
    """
    Compute coverage and precision metrics for generated samples.
    
    Coverage measures what fraction of the real data distribution is
    covered by the generated samples. Precision measures what fraction
    of generated samples are close to real samples.
    
    Args:
        real_samples: Real data samples (numpy array or tf.Tensor)
        generated_samples: Generated data samples (numpy array or tf.Tensor)
        k (int): Number of nearest neighbors to consider
        
    Returns:
        tuple: (coverage, precision) values
    """
    # Convert tensors to numpy arrays
    if TF_AVAILABLE and hasattr(real_samples, 'numpy'):
        real_np = real_samples.numpy()
        gen_np = generated_samples.numpy()
    else:
        real_np = np.array(real_samples)
        gen_np = np.array(generated_samples)
    
    from sklearn.neighbors import NearestNeighbors
    
    # Fit nearest neighbors on real data
    nn_real = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_real.fit(real_np)
    
    # Fit nearest neighbors on generated data
    nn_gen = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_gen.fit(gen_np)
    
    # Compute coverage: fraction of real samples with nearby generated samples
    distances_real_to_gen, _ = nn_gen.kneighbors(real_np)
    coverage_threshold = np.percentile(distances_real_to_gen[:, 0], 50)
    coverage = np.mean(distances_real_to_gen[:, 0] <= coverage_threshold)
    
    # Compute precision: fraction of generated samples with nearby real samples
    distances_gen_to_real, _ = nn_real.kneighbors(gen_np)
    precision_threshold = np.percentile(distances_gen_to_real[:, 0], 50)
    precision = np.mean(distances_gen_to_real[:, 0] <= precision_threshold)
    
    return coverage, precision

def compute_fid_score(real_samples, generated_samples):
    """
    Compute Frechet Inception Distance (FID) score approximation.
    
    This function computes a simplified version of FID by comparing
    the means and covariances of the real and generated distributions
    directly in the data space.
    
    Args:
        real_samples: Real data samples (numpy array or tf.Tensor)
        generated_samples: Generated data samples (numpy array or tf.Tensor)
        
    Returns:
        float: FID score approximation
    """
    # Convert tensors to numpy arrays
    if TF_AVAILABLE and hasattr(real_samples, 'numpy'):
        real_np = real_samples.numpy()
        gen_np = generated_samples.numpy()
    else:
        real_np = np.array(real_samples)
        gen_np = np.array(generated_samples)
    
    # Compute means
    mu_real = np.mean(real_np, axis=0)
    mu_gen = np.mean(gen_np, axis=0)
    
    # Compute covariances
    sigma_real = np.cov(real_np, rowvar=False)
    sigma_gen = np.cov(gen_np, rowvar=False)
    
    # Compute FID score
    diff = mu_real - mu_gen
    mean_diff = np.sum(diff ** 2)
    
    # Compute trace of covariance difference
    cov_diff = sigma_real + sigma_gen - 2 * np.sqrt(sigma_real @ sigma_gen)
    trace_cov = np.trace(cov_diff)
    
    fid = mean_diff + trace_cov
    
    return fid

def save_model(model, filepath):
    """
    Save model parameters to file using TensorFlow checkpoints.
    
    This function creates a checkpoint of the model's trainable variables
    for later restoration. The checkpoint format is compatible with
    TensorFlow's standard checkpoint system.
    
    Args:
        model: Model object with trainable_variables attribute
        filepath (str): Path to save the model checkpoint
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if TF_AVAILABLE and hasattr(model, 'trainable_variables'):
        # Create TensorFlow checkpoint
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.save(filepath)
        logger.info(f"Model saved to {filepath}")
    else:
        logger.warning("Model saving requires TensorFlow or model type not recognized")

def load_model(model, filepath):
    """
    Load model parameters from file using TensorFlow checkpoints.
    
    This function restores a model's trainable variables from a previously
    saved checkpoint. The model structure must match the saved checkpoint.
    
    Args:
        model: Model object to load parameters into
        filepath (str): Path to load the model checkpoint from
    """
    if TF_AVAILABLE and hasattr(model, 'trainable_variables'):
        # Restore from TensorFlow checkpoint
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(filepath)
        logger.info(f"Model loaded from {filepath}")
    else:
        logger.warning("Model loading requires TensorFlow or model type not recognized")
