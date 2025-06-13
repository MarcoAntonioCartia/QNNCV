"""
Configuration management for Quantum GANs.

This module provides flexible configuration management with YAML support,
hardware detection, and validation.
"""

import yaml
import os
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class QuantumGANConfig:
    """
    Configuration manager for Quantum GANs with flexible parameter management.
    
    Features:
    - YAML-based configuration management
    - Hardware detection and resource estimation
    - Configuration validation
    - Encoding strategy management
    - Manual architecture control for experimentation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.hardware_info = self._detect_hardware()
        
        logger.info(f"QuantumGANConfig initialized with {self.config_path}")
        logger.info(f"Hardware: {self.hardware_info}")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, '..', '..', 'config', 'config.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not available."""
        return {
            'training': {
                'epochs': 100,
                'batch_size': 8,
                'latent_dim': 4
            },
            'generator': {
                'n_modes': 2,
                'layers': 2,
                'cutoff_dim': 8,
                'encoding_strategy': 'coherent_state'
            },
            'discriminator': {
                'n_modes': 1,
                'layers': 1,
                'cutoff_dim': 8
            },
            'optimizer': {
                'generator_lr': 5e-4,
                'discriminator_lr': 5e-4,
                'beta1': 0.5,
                'beta2': 0.999
            },
            'hardware': {
                'gpu_memory_limit': None,
                'quantum_device': 'cpu',
                'classical_device': 'auto'
            },
            'metrics': {
                'compute_quantum_metrics': True,
                'quantum_metrics_interval': 10,
                'save_interval': 5
            }
        }
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware resources."""
        hardware_info = {
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'gpu_count': len(tf.config.list_physical_devices('GPU')),
            'cpu_count': os.cpu_count(),
            'memory_limit': None
        }
        
        # Try to get GPU memory info
        if hardware_info['gpu_available']:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Get memory info for first GPU
                    gpu_details = tf.config.experimental.get_device_details(gpus[0])
                    hardware_info['gpu_memory'] = gpu_details.get('device_name', 'Unknown GPU')
            except Exception as e:
                logger.debug(f"Could not get GPU details: {e}")
        
        return hardware_info
    
    def get_encoding_strategy(self) -> str:
        """Get current encoding strategy."""
        return self.config.get('generator', {}).get('encoding_strategy', 'coherent_state')
    
    def set_encoding_strategy(self, strategy: str):
        """
        Set encoding strategy.
        
        Args:
            strategy: One of ['coherent_state', 'direct_displacement', 
                     'angle_encoding', 'sparse_parameter', 'classical_neural']
        """
        valid_strategies = [
            'coherent_state', 'direct_displacement', 'angle_encoding', 
            'sparse_parameter', 'classical_neural'
        ]
        
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid encoding strategy. Must be one of: {valid_strategies}")
        
        if 'generator' not in self.config:
            self.config['generator'] = {}
        
        self.config['generator']['encoding_strategy'] = strategy
        logger.info(f"Encoding strategy set to: {strategy}")
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware-optimized configuration."""
        hw_config = self.config.get('hardware', {}).copy()
        
        # Auto-detect optimal device placement
        if hw_config.get('classical_device') == 'auto':
            hw_config['classical_device'] = 'gpu' if self.hardware_info['gpu_available'] else 'cpu'
        
        # Quantum operations always on CPU (Strawberry Fields limitation)
        hw_config['quantum_device'] = 'cpu'
        
        return hw_config
    
    def estimate_memory_usage(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate memory usage for given configuration.
        
        Args:
            batch_size: Training batch size (uses config default if None)
            
        Returns:
            Memory usage estimates in GB
        """
        if batch_size is None:
            batch_size = self.config.get('training', {}).get('batch_size', 8)
        
        # Ensure batch_size is an integer
        batch_size = int(batch_size) if batch_size is not None else 8
        
        gen_config = self.config['generator']
        disc_config = self.config['discriminator']
        
        # Rough estimates based on quantum state dimensions
        gen_cutoff = int(gen_config['cutoff_dim'])
        gen_modes = int(gen_config['n_modes'])
        disc_cutoff = int(disc_config['cutoff_dim'])
        disc_modes = int(disc_config['n_modes'])
        
        # Quantum state memory (complex numbers, 8 bytes each)
        gen_state_size = (gen_cutoff ** gen_modes) * 16 * float(batch_size) / (1024**3)  # GB
        disc_state_size = (disc_cutoff ** disc_modes) * 16 * float(batch_size) / (1024**3)  # GB
        
        # Classical network memory (rough estimate)
        classical_memory = 0.1 * float(batch_size) / 1000  # GB
        
        return {
            'generator_quantum': gen_state_size,
            'discriminator_quantum': disc_state_size,
            'classical_networks': classical_memory,
            'total_estimated': gen_state_size + disc_state_size + classical_memory
        }
    
    def validate_config(self) -> Tuple[bool, list]:
        """
        Validate current configuration.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required sections
        required_sections = ['training', 'generator', 'discriminator', 'optimizer']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
        
        # Helper function to safely convert to int
        def safe_int(value, default=0):
            try:
                return int(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Validate generator config
        gen_config = self.config.get('generator', {})
        cutoff_dim = safe_int(gen_config.get('cutoff_dim', 0))
        n_modes = safe_int(gen_config.get('n_modes', 0))
        layers = safe_int(gen_config.get('layers', 0))
        
        if cutoff_dim < 4:
            issues.append("Generator cutoff_dim should be at least 4")
        if n_modes < 1:
            issues.append("Generator n_modes should be at least 1")
        if layers < 1:
            issues.append("Generator layers should be at least 1")
        
        # Validate discriminator config
        disc_config = self.config.get('discriminator', {})
        disc_cutoff_dim = safe_int(disc_config.get('cutoff_dim', 0))
        disc_n_modes = safe_int(disc_config.get('n_modes', 0))
        disc_layers = safe_int(disc_config.get('layers', 0))
        
        if disc_cutoff_dim < 4:
            issues.append("Discriminator cutoff_dim should be at least 4")
        if disc_n_modes < 1:
            issues.append("Discriminator n_modes should be at least 1")
        if disc_layers < 1:
            issues.append("Discriminator layers should be at least 1")
        
        # Validate training config
        train_config = self.config.get('training', {})
        batch_size = safe_int(train_config.get('batch_size', 0))
        epochs = safe_int(train_config.get('epochs', 0))
        latent_dim = safe_int(train_config.get('latent_dim', 0))
        
        if batch_size < 1:
            issues.append("Batch size should be at least 1")
        if epochs < 1:
            issues.append("Epochs should be at least 1")
        if latent_dim < 1:
            issues.append("Latent dimension should be at least 1")
        
        # Validate optimizer config
        opt_config = self.config.get('optimizer', {})
        gen_lr = safe_float(opt_config.get('generator_lr', 0))
        disc_lr = safe_float(opt_config.get('discriminator_lr', 0))
        
        if gen_lr <= 0:
            issues.append("Generator learning rate should be positive")
        if disc_lr <= 0:
            issues.append("Discriminator learning rate should be positive")
        
        # Validate encoding strategy
        encoding_strategy = str(gen_config.get('encoding_strategy', '')).strip().strip("'\"")
        valid_strategies = [
            'coherent_state', 'direct_displacement', 'angle_encoding', 
            'sparse_parameter', 'classical_neural'
        ]
        if encoding_strategy not in valid_strategies:
            issues.append(f"Invalid encoding strategy: {encoding_strategy}. Must be one of: {valid_strategies}")
        
        return len(issues) == 0, issues
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to YAML file."""
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        logger.info("Configuration updated")
    
    def get_config_summary(self) -> str:
        """Get readable configuration summary."""
        gen_config = self.config['generator']
        disc_config = self.config['discriminator']
        train_config = self.config['training']
        opt_config = self.config['optimizer']
        
        # Estimate memory usage
        memory_est = self.estimate_memory_usage()
        
        summary = f"""
Quantum GAN Configuration Summary:
================================

Generator:
  - Quantum Modes: {gen_config.get('n_modes', 'N/A')}
  - Layers: {gen_config.get('layers', 'N/A')}
  - Cutoff Dimension: {gen_config.get('cutoff_dim', 'N/A')}
  - Encoding Strategy: {gen_config.get('encoding_strategy', 'N/A')}

Discriminator:
  - Quantum Modes: {disc_config.get('n_modes', 'N/A')}
  - Layers: {disc_config.get('layers', 'N/A')}
  - Cutoff Dimension: {disc_config.get('cutoff_dim', 'N/A')}

Training:
  - Epochs: {train_config.get('epochs', 'N/A')}
  - Batch Size: {train_config.get('batch_size', 'N/A')}
  - Latent Dimension: {train_config.get('latent_dim', 'N/A')}

Optimizer:
  - Generator LR: {opt_config.get('generator_lr', 'N/A')}
  - Discriminator LR: {opt_config.get('discriminator_lr', 'N/A')}
  - Beta1: {opt_config.get('beta1', 'N/A')}
  - Beta2: {opt_config.get('beta2', 'N/A')}

Hardware:
  - GPU Available: {self.hardware_info['gpu_available']}
  - GPU Count: {self.hardware_info['gpu_count']}
  - CPU Count: {self.hardware_info['cpu_count']}

Memory Estimation:
  - Total Estimated: {memory_est['total_estimated']:.3f} GB
  - Generator Quantum: {memory_est['generator_quantum']:.3f} GB
  - Discriminator Quantum: {memory_est['discriminator_quantum']:.3f} GB
        """
        
        return summary.strip()
    
    def create_experiment_config(self, experiment_name: str, 
                               generator_modes: int, generator_layers: int,
                               discriminator_modes: int, discriminator_layers: int,
                               cutoff_dim: int, encoding_strategy: str = 'coherent_state') -> Dict[str, Any]:
        """
        Create a new experimental configuration for architecture exploration.
        
        Args:
            experiment_name: Name for this experiment
            generator_modes: Number of generator quantum modes
            generator_layers: Number of generator quantum layers
            discriminator_modes: Number of discriminator quantum modes
            discriminator_layers: Number of discriminator quantum layers
            cutoff_dim: Cutoff dimension for both generator and discriminator
            encoding_strategy: Encoding strategy to use
            
        Returns:
            New configuration dictionary
        """
        experiment_config = self.config.copy()
        
        # Update generator config
        experiment_config['generator'].update({
            'n_modes': generator_modes,
            'layers': generator_layers,
            'cutoff_dim': cutoff_dim,
            'encoding_strategy': encoding_strategy
        })
        
        # Update discriminator config
        experiment_config['discriminator'].update({
            'n_modes': discriminator_modes,
            'layers': discriminator_layers,
            'cutoff_dim': cutoff_dim
        })
        
        # Add experiment metadata
        experiment_config['experiment'] = {
            'name': experiment_name,
            'description': f"G({generator_modes}m,{generator_layers}l) vs D({discriminator_modes}m,{discriminator_layers}l), cutoff={cutoff_dim}, encoding={encoding_strategy}"
        }
        
        return experiment_config
    
    def get_architecture_suggestions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get suggested architecture configurations for experimentation.
        
        Returns:
            Dictionary of suggested configurations
        """
        suggestions = {
            'minimal': {
                'description': 'Minimal configuration for quick testing',
                'generator': {'n_modes': 1, 'layers': 1, 'cutoff_dim': 6},
                'discriminator': {'n_modes': 1, 'layers': 1, 'cutoff_dim': 6},
                'training': {'batch_size': 4, 'epochs': 50}
            },
            'balanced': {
                'description': 'Balanced configuration for general use',
                'generator': {'n_modes': 2, 'layers': 2, 'cutoff_dim': 8},
                'discriminator': {'n_modes': 1, 'layers': 1, 'cutoff_dim': 8},
                'training': {'batch_size': 8, 'epochs': 100}
            },
            'expressive': {
                'description': 'More expressive configuration for complex data',
                'generator': {'n_modes': 3, 'layers': 3, 'cutoff_dim': 10},
                'discriminator': {'n_modes': 2, 'layers': 2, 'cutoff_dim': 8},
                'training': {'batch_size': 8, 'epochs': 200}
            },
            'high_fidelity': {
                'description': 'High fidelity configuration (memory intensive)',
                'generator': {'n_modes': 4, 'layers': 2, 'cutoff_dim': 12},
                'discriminator': {'n_modes': 2, 'layers': 2, 'cutoff_dim': 10},
                'training': {'batch_size': 4, 'epochs': 150}
            }
        }
        
        return suggestions

def test_quantum_gan_config():
    """Test the QuantumGANConfig functionality."""
    print("Testing QuantumGANConfig...")
    
    # Test basic initialization
    config = QuantumGANConfig()
    print("✓ Basic initialization successful")
    
    # Test configuration summary
    print("\nConfiguration Summary:")
    print(config.get_config_summary())
    
    # Test encoding strategy management
    print(f"\nCurrent encoding strategy: {config.get_encoding_strategy()}")
    config.set_encoding_strategy('angle_encoding')
    print(f"Updated encoding strategy: {config.get_encoding_strategy()}")
    
    # Test memory estimation
    memory_usage = config.estimate_memory_usage(batch_size=8)
    print(f"\n✓ Memory usage estimated: {memory_usage['total_estimated']:.3f} GB")
    
    # Test validation
    is_valid, issues = config.validate_config()
    print(f"\n✓ Configuration validation: {'Valid' if is_valid else 'Invalid'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Test experiment configuration creation
    exp_config = config.create_experiment_config(
        experiment_name="test_experiment",
        generator_modes=2,
        generator_layers=2,
        discriminator_modes=2,
        discriminator_layers=1,
        cutoff_dim=8,
        encoding_strategy='coherent_state'
    )
    print(f"\n✓ Experiment config created: {exp_config['experiment']['description']}")
    
    # Test architecture suggestions
    suggestions = config.get_architecture_suggestions()
    print(f"\n✓ Architecture suggestions available: {list(suggestions.keys())}")
    
    print("\nQuantumGANConfig test completed successfully!")
    return config

if __name__ == "__main__":
    test_quantum_gan_config()
