"""
Measurement Extractor for Quantum States

This module extracts raw measurements from quantum states,
preserving quantum information for optimization.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MeasurementExtractorBase(ABC):
    """
    Abstract base class for measurement extraction.
    
    Defines the interface for extracting measurements from quantum states.
    """
    
    @abstractmethod
    def extract_measurements(self, quantum_states: List[Any]) -> tf.Tensor:
        """
        Extract measurements from quantum states.
        
        Args:
            quantum_states: List of quantum states from SF
            
        Returns:
            Tensor of measurements [batch_size, measurement_dim]
        """
        pass
    
    @abstractmethod
    def get_measurement_dim(self) -> int:
        """Get the dimension of extracted measurements."""
        pass


class RawMeasurementExtractor(MeasurementExtractorBase):
    """
    Extracts raw measurements from quantum states.
    
    Implements multiple measurement types (X quadrature, P quadrature, 
    photon number) without statistical processing.
    """
    
    def __init__(self, n_modes: int, cutoff_dim: int, 
                 measurement_types: Optional[List[str]] = None):
        """
        Initialize raw measurement extractor.
        
        Args:
            n_modes: Number of quantum modes
            cutoff_dim: Fock space cutoff dimension
            measurement_types: List of measurement types to extract
                             Default: ['x_quad', 'p_quad', 'n_photon']
        """
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        
        if measurement_types is None:
            self.measurement_types = ['x_quad', 'p_quad', 'n_photon']
        else:
            self.measurement_types = measurement_types
        
        self.measurement_dim = n_modes * len(self.measurement_types)
        
        logger.info(f"Raw measurement extractor initialized: {self.measurement_dim} measurements")
    
    def extract_measurements(self, quantum_states: List[Any]) -> tf.Tensor:
        """
        Extract raw measurements from quantum states.
        
        Args:
            quantum_states: List of quantum states
            
        Returns:
            Raw measurements [batch_size, measurement_dim]
        """
        batch_size = len(quantum_states)
        all_measurements = []
        
        for state in quantum_states:
            mode_measurements = []
            
            for mode in range(self.n_modes):
                # Extract each measurement type
                if 'x_quad' in self.measurement_types:
                    x_quad = self._measure_x_quadrature(state, mode)
                    mode_measurements.append(x_quad)
                
                if 'p_quad' in self.measurement_types:
                    p_quad = self._measure_p_quadrature(state, mode)
                    mode_measurements.append(p_quad)
                
                if 'n_photon' in self.measurement_types:
                    n_photon = self._measure_photon_number(state, mode)
                    mode_measurements.append(n_photon)
                
                if 'parity' in self.measurement_types:
                    parity = self._measure_parity(state, mode)
                    mode_measurements.append(parity)
            
            # Stack measurements for this sample
            sample_measurements = tf.stack(mode_measurements)
            all_measurements.append(sample_measurements)
        
        # Stack all samples
        raw_measurements = tf.stack(all_measurements, axis=0)
        
        return raw_measurements
    
    def _measure_x_quadrature(self, state: Any, mode: int) -> tf.Tensor:
        """
        Measure X quadrature for a specific mode.
        
        Args:
            state: Quantum state
            mode: Mode index
            
        Returns:
            X quadrature measurement
        """
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # X quadrature expectation: <a + a†>/√2
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        
        # Simplified calculation using mean position
        x_expectation = tf.reduce_sum(prob_amplitudes * n_vals) / tf.sqrt(2.0)
        
        # Add quantum noise (shot noise)
        x_measurement = x_expectation + tf.random.normal([], stddev=0.1)
        
        return x_measurement
    
    def _measure_p_quadrature(self, state: Any, mode: int) -> tf.Tensor:
        """
        Measure P quadrature for a specific mode.
        
        Args:
            state: Quantum state
            mode: Mode index
            
        Returns:
            P quadrature measurement
        """
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # P quadrature: related to variance
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        mean_n = tf.reduce_sum(prob_amplitudes * n_vals)
        var_n = tf.reduce_sum(prob_amplitudes * (n_vals - mean_n)**2)
        
        p_expectation = tf.sqrt(var_n + 1e-6) / tf.sqrt(2.0)
        
        # Add quantum noise
        p_measurement = p_expectation + tf.random.normal([], stddev=0.1)
        
        return p_measurement
    
    def _measure_photon_number(self, state: Any, mode: int) -> tf.Tensor:
        """
        Measure photon number for a specific mode.
        
        Args:
            state: Quantum state
            mode: Mode index
            
        Returns:
            Photon number measurement
        """
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Photon number expectation
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        n_expectation = tf.reduce_sum(prob_amplitudes * n_vals)
        
        # Add shot noise (Poissonian)
        n_measurement = n_expectation + tf.random.normal([], stddev=tf.sqrt(n_expectation + 1e-6))
        
        return n_measurement
    
    def _measure_parity(self, state: Any, mode: int) -> tf.Tensor:
        """
        Measure parity operator for a specific mode.
        
        Args:
            state: Quantum state
            mode: Mode index
            
        Returns:
            Parity measurement
        """
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Parity: (-1)^n
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        parity_vals = tf.pow(-1.0, n_vals)
        parity_expectation = tf.reduce_sum(prob_amplitudes * parity_vals)
        
        # Add measurement noise
        parity_measurement = parity_expectation + tf.random.normal([], stddev=0.05)
        
        return parity_measurement
    
    def get_measurement_dim(self) -> int:
        """Get the dimension of extracted measurements."""
        return self.measurement_dim


class HolisticMeasurementExtractor(MeasurementExtractorBase):
    """
    Extracts holistic measurements that capture correlations between modes.
    """
    
    def __init__(self, n_modes: int, cutoff_dim: int):
        """
        Initialize holistic measurement extractor.
        
        Args:
            n_modes: Number of quantum modes
            cutoff_dim: Fock space cutoff dimension
        """
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        
        # Measurement dimension includes correlations
        self.measurement_dim = (
            n_modes * 3 +  # Individual mode measurements
            n_modes * (n_modes - 1) // 2  # Pairwise correlations
        )
        
        logger.info(f"Holistic measurement extractor initialized: {self.measurement_dim} measurements")
    
    def extract_measurements(self, quantum_states: List[Any]) -> tf.Tensor:
        """
        Extract holistic measurements including correlations.
        
        Args:
            quantum_states: List of quantum states
            
        Returns:
            Holistic measurements [batch_size, measurement_dim]
        """
        batch_size = len(quantum_states)
        all_measurements = []
        
        for state in quantum_states:
            measurements = []
            
            # Individual mode measurements
            for mode in range(self.n_modes):
                measurements.extend(self._extract_mode_measurements(state, mode))
            
            # Pairwise correlations
            for i in range(self.n_modes):
                for j in range(i + 1, self.n_modes):
                    correlation = self._measure_correlation(state, i, j)
                    measurements.append(correlation)
            
            all_measurements.append(tf.stack(measurements))
        
        return tf.stack(all_measurements, axis=0)
    
    def _extract_mode_measurements(self, state: Any, mode: int) -> List[tf.Tensor]:
        """Extract basic measurements for a single mode."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        
        # Mean photon number
        mean_n = tf.reduce_sum(prob_amplitudes * n_vals)
        
        # Variance
        var_n = tf.reduce_sum(prob_amplitudes * (n_vals - mean_n)**2)
        
        # Skewness (third moment)
        skew_n = tf.reduce_sum(prob_amplitudes * (n_vals - mean_n)**3) / (var_n**1.5 + 1e-6)
        
        return [mean_n, tf.sqrt(var_n), skew_n]
    
    def _measure_correlation(self, state: Any, mode1: int, mode2: int) -> tf.Tensor:
        """
        Measure correlation between two modes.
        
        Args:
            state: Quantum state
            mode1: First mode index
            mode2: Second mode index
            
        Returns:
            Correlation measurement
        """
        # Simplified correlation based on joint photon statistics
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # This is a placeholder - actual implementation would depend on
        # how multi-mode states are represented
        correlation = tf.reduce_sum(prob_amplitudes) * 0.1  # Placeholder
        
        return correlation
    
    def get_measurement_dim(self) -> int:
        """Get the dimension of extracted measurements."""
        return self.measurement_dim


class AdaptiveMeasurementExtractor(MeasurementExtractorBase):
    """
    Adaptively selects measurements based on the quantum state.
    """
    
    def __init__(self, n_modes: int, cutoff_dim: int, base_measurements: int = 3):
        """
        Initialize adaptive measurement extractor.
        
        Args:
            n_modes: Number of quantum modes
            cutoff_dim: Fock space cutoff dimension
            base_measurements: Base measurements per mode
        """
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        self.base_measurements = base_measurements
        
        # Use a learnable network to select measurements
        self.measurement_selector = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(n_modes * base_measurements, activation='sigmoid')
        ])
        
        self.measurement_dim = n_modes * base_measurements
        
        logger.info(f"Adaptive measurement extractor initialized")
    
    def extract_measurements(self, quantum_states: List[Any]) -> tf.Tensor:
        """
        Extract measurements adaptively based on state properties.
        
        Args:
            quantum_states: List of quantum states
            
        Returns:
            Adaptive measurements [batch_size, measurement_dim]
        """
        batch_size = len(quantum_states)
        all_measurements = []
        
        for state in quantum_states:
            # Get state properties
            state_features = self._extract_state_features(state)
            
            # Select measurement weights
            measurement_weights = self.measurement_selector(state_features)
            
            # Extract weighted measurements
            raw_measurements = self._extract_all_measurements(state)
            weighted_measurements = raw_measurements * measurement_weights
            
            all_measurements.append(weighted_measurements)
        
        return tf.stack(all_measurements, axis=0)
    
    def _extract_state_features(self, state: Any) -> tf.Tensor:
        """Extract features from quantum state for measurement selection."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Simple features: moments of photon number distribution
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        
        features = []
        for moment in range(1, 5):  # First 4 moments
            moment_val = tf.reduce_sum(prob_amplitudes * (n_vals ** moment))
            features.append(moment_val)
        
        return tf.stack(features)
    
    def _extract_all_measurements(self, state: Any) -> tf.Tensor:
        """Extract all possible measurements."""
        measurements = []
        
        for mode in range(self.n_modes):
            # Extract multiple measurement types
            extractor = RawMeasurementExtractor(1, self.cutoff_dim)
            mode_state = state  # In practice, would trace out other modes
            
            x_quad = extractor._measure_x_quadrature(mode_state, 0)
            p_quad = extractor._measure_p_quadrature(mode_state, 0)
            n_photon = extractor._measure_photon_number(mode_state, 0)
            
            measurements.extend([x_quad, p_quad, n_photon])
        
        return tf.stack(measurements)
    
    def get_measurement_dim(self) -> int:
        """Get the dimension of extracted measurements."""
        return self.measurement_dim
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables of the measurement selector."""
        return self.measurement_selector.trainable_variables
