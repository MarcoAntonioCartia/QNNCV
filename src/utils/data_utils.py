import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import os
import logging

# Try to import TensorFlow, but don't fail if it's not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logging.info("TensorFlow is available")
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow is not available. Some functions will use fallbacks.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(data_path="data", num_samples=1000, feature_dim=30):
    """
    Load and preprocess dataset for quantum GAN training.
    
    This function provides a generic data loading interface that can handle
    various dataset formats. Currently supports synthetic data generation
    when no specific dataset is available.
    
    Args:
        data_path (str): Path to dataset directory
        num_samples (int): Number of samples to load or generate
        feature_dim (int): Dimension of feature vectors
        
    Returns:
        np.ndarray or tf.Tensor: Preprocessed feature vectors
    """
    logger.info(f"Loading dataset from {data_path}")
    
    # Check if dataset file exists
    dataset_file = os.path.join(data_path, "features.npy")
    
    if os.path.exists(dataset_file):
        logger.info("Loading existing dataset")
        data = np.load(dataset_file)[:num_samples]
    else:
        logger.warning("Dataset not found. Generating synthetic data.")
        # Create synthetic feature vectors
        np.random.seed(42)
        
        # Generate realistic feature distributions
        feature_1 = np.random.normal(0, 1, num_samples)
        feature_2 = np.random.normal(0, 1.5, num_samples)
        feature_3 = np.random.exponential(2, num_samples)
        
        # Additional features with various distributions
        other_features = np.random.normal(0, 1, (num_samples, feature_dim - 3))
        
        # Combine features
        data = np.column_stack([feature_1, feature_2, feature_3, other_features])
        
        # Ensure data directory exists
        os.makedirs(data_path, exist_ok=True)
        
        # Save synthetic data for consistency
        np.save(dataset_file, data)
        logger.info(f"Saved synthetic dataset to {dataset_file}")
    
    # Normalize data using standard scaling
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    logger.info(f"Loaded {data_normalized.shape[0]} samples with {data_normalized.shape[1]} features")
    
    # Return as TensorFlow tensor if available, otherwise numpy array
    if TF_AVAILABLE:
        return tf.convert_to_tensor(data_normalized, dtype=tf.float32)
    else:
        return data_normalized.astype(np.float32)

def load_synthetic_data(dataset_type="spiral", num_samples=1000):
    """
    Generate synthetic 2D datasets for testing and visualization.
    
    This function creates various 2D synthetic datasets commonly used
    for testing generative models and visualizing their performance.
    
    Args:
        dataset_type (str): Type of dataset ('spiral', 'moons', 'circles', 'gaussian')
        num_samples (int): Number of samples to generate
        
    Returns:
        np.ndarray or tf.Tensor: Generated 2D data points
    """
    logger.info(f"Generating {dataset_type} synthetic data with {num_samples} samples")
    
    if dataset_type == "spiral":
        # Generate spiral pattern data
        t = np.linspace(0, 4*np.pi, num_samples)
        x = t * np.cos(t) + np.random.normal(0, 0.1, num_samples)
        y = t * np.sin(t) + np.random.normal(0, 0.1, num_samples)
        data = np.column_stack([x, y])
        
    elif dataset_type == "moons":
        # Generate two interleaving half circles
        data, _ = make_moons(n_samples=num_samples, noise=0.1, random_state=42)
        
    elif dataset_type == "circles":
        # Generate two concentric circles
        data, _ = make_circles(n_samples=num_samples, noise=0.1, random_state=42)
        
    elif dataset_type == "gaussian":
        # Multi-modal Gaussian mixture
        data1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], num_samples//2)
        data2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], num_samples//2)
        data = np.vstack([data1, data2])
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Normalize data using standard scaling
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Return as TensorFlow tensor if available, otherwise numpy array
    if TF_AVAILABLE:
        return tf.convert_to_tensor(data_normalized, dtype=tf.float32)
    else:
        return data_normalized.astype(np.float32)

def create_output_directory(base_path="tests/results"):
    """
    Create timestamped output directory structure for saving experimental results.
    
    This function creates a structured directory layout for organizing
    experimental outputs including plots, models, logs, and data.
    
    Args:
        base_path (str): Base path for results directory
        
    Returns:
        str: Path to created timestamped directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"qgan_run_{timestamp}")
    
    # Create subdirectories for organized output
    subdirs = ['plots', 'models', 'logs', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return output_dir
