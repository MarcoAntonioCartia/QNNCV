"""
Simple Quantum Neural Network Implementation

This is a clean, simple implementation of a quantum neural network using Strawberry Fields.
It demonstrates the core concepts without the complexity of the full GAN framework.

Key Design Principles:
1. Pure Strawberry Fields quantum circuits
2. Individual sample processing for quantum advantage
3. Clear separation between classical and quantum components
4. Simple gradient flow through quantum parameters
5. Easy to understand and modify

Why this approach:
- Each quantum parameter is a separate tf.Variable for optimal gradient computation
- Individual sample processing preserves quantum correlations
- Native SF Program-Engine execution without classical wrappers
- Clear documentation of the "whys" behind each design choice
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class SimpleQuantumNeuralNetwork:
    """
    Simple Quantum Neural Network using Strawberry Fields.
    
    This implementation demonstrates the core concepts of quantum neural networks
    using continuous variable quantum computing. It processes each input sample
    individually through a quantum circuit to preserve quantum correlations.
    
    Why individual sample processing?
    - Quantum correlations are fragile and can be lost in batch averaging
    - Each sample gets its own quantum circuit instance
    - This preserves the quantum advantage in the computation
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 output_dim: int = 1,
                 n_modes: int = 3,
                 n_layers: int = 2,
                 cutoff_dim: int = 6):
        """
        Initialize the quantum neural network.
        
        Args:
            input_dim: Dimension of classical input data
            output_dim: Dimension of output (e.g., 1 for regression, 2 for classification)
            n_modes: Number of quantum modes (photonic modes)
            n_layers: Number of quantum circuit layers
            cutoff_dim: Fock space cutoff dimension (truncates infinite Hilbert space)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        
        # Why these parameters?
        # - n_modes: Determines the quantum system size (more modes = more quantum capacity)
        # - n_layers: Controls circuit depth (more layers = more expressivity)
        # - cutoff_dim: Practical necessity for infinite-dimensional CV systems
        
        # Initialize Strawberry Fields components
        self._initialize_quantum_circuit()
        self._create_trainable_parameters()
        
        logger.info(f"Quantum Neural Network initialized:")
        logger.info(f"  Input dimension: {input_dim}")
        logger.info(f"  Output dimension: {output_dim}")
        logger.info(f"  Quantum modes: {n_modes}")
        logger.info(f"  Circuit layers: {n_layers}")
        logger.info(f"  Trainable parameters: {len(self.trainable_variables)}")
    
    def _initialize_quantum_circuit(self):
        """
        Initialize the quantum circuit using Strawberry Fields.
        
        Why use SF Program-Engine model?
        - Program: Defines the quantum circuit structure
        - Engine: Executes the circuit and returns quantum states
        - This separation allows for symbolic circuit definition and efficient execution
        """
        # Create SF Program (quantum circuit definition)
        self.prog = sf.Program(self.n_modes)
        
        # Create SF Engine (quantum circuit execution)
        # Why "tf" backend? It integrates with TensorFlow for automatic differentiation
        self.engine = sf.Engine("tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "eval": True  # Return TensorFlow tensors for gradient computation
        })
        
        # Build the symbolic quantum circuit
        self._build_quantum_circuit()
    
    def _build_quantum_circuit(self):
        """
        Build the quantum circuit using symbolic SF operations.
        
        Why this circuit architecture?
        - Displacement gates: Encode classical data into quantum states
        - Squeezing gates: Create quantum correlations and non-linearity
        - Interferometer: Mix quantum modes (like layers in classical NNs)
        - Measurement: Extract classical information from quantum states
        """
        with self.prog.context as q:
            # Layer 1: Input encoding and first quantum processing
            for i in range(self.n_modes):
                # Displacement gates encode classical input into quantum states
                # Why displacement? It's the most natural way to encode classical data in CV systems
                ops.Dgate(self.prog.params(f"disp1_{i}")) | q[i]
                
                # Squeezing gates create quantum correlations
                # Why squeezing? It's a fundamental quantum operation that classical systems can't do
                ops.Sgate(self.prog.params(f"squeeze1_{i}")) | q[i]
            
            # Interferometer: Mix quantum modes (like matrix multiplication in classical NNs)
            # Why interferometer? It creates entanglement between modes
            for i in range(self.n_modes - 1):
                ops.BSgate(self.prog.params(f"bs1_{i}"), self.prog.params(f"bs1_phase_{i}")) | (q[i], q[i+1])
            
            # Layer 2: Additional quantum processing
            if self.n_layers > 1:
                for i in range(self.n_modes):
                    ops.Dgate(self.prog.params(f"disp2_{i}")) | q[i]
                    ops.Sgate(self.prog.params(f"squeeze2_{i}")) | q[i]
                
                for i in range(self.n_modes - 1):
                    ops.BSgate(self.prog.params(f"bs2_{i}"), self.prog.params(f"bs2_phase_{i}")) | (q[i], q[i+1])
            
            # Final measurement: Extract classical information
            # Why measure position quadrature? It's the most natural measurement for CV systems
            for i in range(self.n_modes):
                ops.MeasureX | q[i]
    
    def _create_trainable_parameters(self):
        """
        Create trainable TensorFlow variables for quantum parameters.
        
        Why individual tf.Variables for each parameter?
        - Ensures proper gradient flow through quantum parameters
        - Avoids issues with parameter sharing in quantum circuits
        - Each parameter gets its own gradient during backpropagation
        """
        self.quantum_parameters = {}
        
        # Layer 1 parameters
        for i in range(self.n_modes):
            # Displacement parameters (real-valued)
            self.quantum_parameters[f"disp1_{i}"] = tf.Variable(
                tf.random.normal([], stddev=0.1), name=f"disp1_{i}"
            )
            # Squeezing parameters (real-valued)
            self.quantum_parameters[f"squeeze1_{i}"] = tf.Variable(
                tf.random.normal([], stddev=0.1), name=f"squeeze1_{i}"
            )
        
        # Beam splitter parameters
        for i in range(self.n_modes - 1):
            self.quantum_parameters[f"bs1_{i}"] = tf.Variable(
                tf.random.uniform([], 0, np.pi/2), name=f"bs1_{i}"
            )
            self.quantum_parameters[f"bs1_phase_{i}"] = tf.Variable(
                tf.random.uniform([], 0, 2*np.pi), name=f"bs1_phase_{i}"
            )
        
        # Layer 2 parameters (if using multiple layers)
        if self.n_layers > 1:
            for i in range(self.n_modes):
                self.quantum_parameters[f"disp2_{i}"] = tf.Variable(
                    tf.random.normal([], stddev=0.1), name=f"disp2_{i}"
                )
                self.quantum_parameters[f"squeeze2_{i}"] = tf.Variable(
                    tf.random.normal([], stddev=0.1), name=f"squeeze2_{i}"
                )
            
            for i in range(self.n_modes - 1):
                self.quantum_parameters[f"bs2_{i}"] = tf.Variable(
                    tf.random.uniform([], 0, np.pi/2), name=f"bs2_{i}"
                )
                self.quantum_parameters[f"bs2_phase_{i}"] = tf.Variable(
                    tf.random.uniform([], 0, 2*np.pi), name=f"bs2_phase_{i}"
                )
        
        # Classical output layer (maps quantum measurements to final output)
        # Why classical output layer? Quantum measurements are classical, so we need classical processing
        measurement_dim = self.n_modes  # Each mode gives one measurement
        self.output_weights = tf.Variable(
            tf.random.normal([measurement_dim, self.output_dim], stddev=0.1),
            name="output_weights"
        )
        self.output_bias = tf.Variable(
            tf.zeros([self.output_dim]), name="output_bias"
        )
    
    def _encode_input(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Encode classical input into quantum parameters.
        
        Why encode input into quantum parameters?
        - Classical data needs to be encoded into quantum states
        - Displacement gates are the most natural way to encode classical data in CV systems
        - This creates a quantum representation of the input data
        """
        # Simple linear encoding: map input to displacement parameters
        # Why linear encoding? It's simple and preserves the input structure
        encoding_weights = tf.random.normal([self.input_dim, self.n_modes], stddev=0.1)
        displacement_values = tf.matmul(x, encoding_weights)
        
        # Create parameter dictionary for quantum circuit
        params = {}
        for i in range(self.n_modes):
            params[f"disp1_{i}"] = displacement_values[:, i]
        
        return params
    
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the quantum neural network.
        
        Why process each sample individually?
        - Quantum correlations are fragile and can be lost in batch processing
        - Each sample gets its own quantum circuit instance
        - This preserves the quantum advantage in the computation
        """
        batch_size = tf.shape(x)[0]
        outputs = []
        
        # Process each sample individually through the quantum circuit
        for i in range(batch_size):
            # Get single sample
            sample = x[i:i+1]  # Keep batch dimension for consistency
            
            # Encode input into quantum parameters
            input_params = self._encode_input(sample)
            
            # Combine input parameters with trainable quantum parameters
            all_params = {}
            for name, param in self.quantum_parameters.items():
                if name in input_params:
                    # Use encoded input for displacement parameters
                    all_params[name] = input_params[name]
                else:
                    # Use trainable parameters for other gates
                    all_params[name] = tf.expand_dims(param, 0)  # Add batch dimension
            
            # Execute quantum circuit
            # Why execute individually? Each sample needs its own quantum state
            result = self.engine.run(self.prog, args=all_params)
            
            # Extract measurements (classical outputs from quantum states)
            # Why measure position quadrature? It's the most natural measurement for CV systems
            measurements = result.samples[0]  # Shape: [n_modes]
            
            # Apply classical output layer
            # Why classical output layer? Quantum measurements are classical, so we need classical processing
            output = tf.matmul(tf.expand_dims(measurements, 0), self.output_weights) + self.output_bias
            outputs.append(output[0])  # Remove batch dimension
        
        return tf.stack(outputs)
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables."""
        variables = list(self.quantum_parameters.values())
        variables.extend([self.output_weights, self.output_bias])
        return variables
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return len(self.trainable_variables)


class SimpleQuantumTrainer:
    """
    Simple trainer for the quantum neural network.
    
    This trainer demonstrates how to train quantum neural networks
    with proper gradient handling and monitoring.
    """
    
    def __init__(self, 
                 model: SimpleQuantumNeuralNetwork,
                 learning_rate: float = 1e-3):
        """
        Initialize the trainer.
        
        Args:
            model: Quantum neural network to train
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Training history
        self.loss_history = []
        self.gradient_history = []
    
    def compute_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        Compute the loss function.
        
        Why mean squared error for regression?
        - It's a standard loss function for regression tasks
        - It's differentiable and works well with gradient-based optimization
        - It penalizes large errors more heavily than small errors
        """
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        
        Why use GradientTape?
        - It allows us to compute gradients through quantum circuits
        - It's the standard way to do custom training in TensorFlow
        - It gives us full control over the training process
        """
        with tf.GradientTape() as tape:
            # Forward pass through quantum neural network
            y_pred = self.model.forward(x)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Monitor gradient flow
        gradient_norm = tf.linalg.global_norm(gradients) if gradients[0] is not None else 0.0
        
        return {
            'loss': float(loss),
            'gradient_norm': float(gradient_norm)
        }
    
    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray,
              epochs: int = 100,
              batch_size: int = 8,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the quantum neural network.
        
        Args:
            x_train: Training input data
            y_train: Training target data
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
        
        Returns:
            Dictionary containing training history
        """
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_gradients = []
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            for batch in range(n_batches):
                # Get batch
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = tf.convert_to_tensor(x_shuffled[start_idx:end_idx], dtype=tf.float32)
                y_batch = tf.convert_to_tensor(y_shuffled[start_idx:end_idx], dtype=tf.float32)
                
                # Training step
                step_results = self.train_step(x_batch, y_batch)
                
                epoch_losses.append(step_results['loss'])
                epoch_gradients.append(step_results['gradient_norm'])
            
            # Record epoch statistics
            avg_loss = np.mean(epoch_losses)
            avg_gradient = np.mean(epoch_gradients)
            
            self.loss_history.append(avg_loss)
            self.gradient_history.append(avg_gradient)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Gradient Norm = {avg_gradient:.6f}")
        
        return {
            'loss': self.loss_history,
            'gradient_norm': self.gradient_history
        }


def generate_synthetic_data(n_samples: int = 200) -> tuple:
    """
    Generate synthetic data for testing the quantum neural network.
    
    Why synthetic data?
    - It's simple and allows us to focus on the quantum implementation
    - We can control the complexity of the learning task
    - It's fast to generate and doesn't require external dependencies
    """
    # Generate input data (2D features)
    x = np.random.uniform(-2, 2, (n_samples, 2))
    
    # Generate target data (simple non-linear function)
    # Why non-linear function? To test the quantum network's ability to learn non-linear patterns
    y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + 0.1 * np.random.normal(size=n_samples)
    y = y.reshape(-1, 1)  # Add output dimension
    
    return x, y


def visualize_results(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                     loss_history: List[float]):
    """
    Visualize the training results.
    
    Why visualize?
    - It helps us understand how well the quantum network is learning
    - We can see if the training is stable and converging
    - It's useful for debugging and understanding the model behavior
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: True vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('True vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    axes[0, 1].plot(loss_history)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Input space visualization
    scatter = axes[1, 0].scatter(x[:, 0], x[:, 1], c=y_true.flatten(), cmap='viridis')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    axes[1, 0].set_title('Input Space (True Values)')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Plot 4: Prediction space visualization
    scatter = axes[1, 1].scatter(x[:, 0], x[:, 1], c=y_pred.flatten(), cmap='viridis')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].set_title('Input Space (Predicted Values)')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating the simple quantum neural network.
    
    This function shows how to:
    1. Create a quantum neural network
    2. Generate synthetic data
    3. Train the network
    4. Evaluate and visualize results
    """
    print("Simple Quantum Neural Network Demo")
    print("=" * 40)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    x_train, y_train = generate_synthetic_data(n_samples=200)
    print(f"Generated {len(x_train)} training samples")
    
    # Create quantum neural network
    print("\nCreating quantum neural network...")
    model = SimpleQuantumNeuralNetwork(
        input_dim=2,
        output_dim=1,
        n_modes=3,
        n_layers=2,
        cutoff_dim=6
    )
    print(f"Model has {model.get_parameter_count()} trainable parameters")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = SimpleQuantumTrainer(model, learning_rate=1e-3)
    
    # Train the model
    print("\nTraining quantum neural network...")
    history = trainer.train(
        x_train=x_train,
        y_train=y_train,
        epochs=50,
        batch_size=8,
        verbose=True
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    x_test, y_test = generate_synthetic_data(n_samples=50)
    y_pred = model.forward(tf.convert_to_tensor(x_test, dtype=tf.float32))
    y_pred = y_pred.numpy()
    
    # Calculate final loss
    final_loss = trainer.compute_loss(y_pred, tf.convert_to_tensor(y_test, dtype=tf.float32))
    print(f"Final test loss: {final_loss:.6f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(x_test, y_test, y_pred, history['loss'])
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 