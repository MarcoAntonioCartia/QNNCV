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
