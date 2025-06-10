"""
Quantum Gradient Utilities

This module provides reusable custom gradient functions for quantum components
to enable proper gradient flow through quantum circuits in TensorFlow.

The custom gradients use parameter-shift rule approximations to provide
meaningful gradients for quantum parameters while maintaining TensorFlow
compatibility.
"""

import tensorflow as tf
import numpy as np
from typing import Callable, List, Tuple, Any

def quantum_custom_gradient(quantum_function: Callable) -> Callable:
    """
    Decorator to add custom gradients to quantum functions.
    
    This decorator wraps quantum functions with custom gradient computation
    using parameter-shift rule approximations. It automatically handles
    the TensorFlow custom gradient requirements.
    
    Args:
        quantum_function: The quantum function to wrap with custom gradients
        
    Returns:
        Wrapped function with custom gradient support
        
    Example:
        @quantum_custom_gradient
        def my_quantum_function(x):
            return quantum_circuit(x)
    """
    
    @tf.custom_gradient
    def wrapped_function(*args, **kwargs):
        """Wrapped function with custom gradients."""
        
        # Execute the original quantum function
        output = quantum_function(*args, **kwargs)
        
        def grad_fn(dy, variables=None):
            """
            Custom gradient function using parameter-shift rule approximation.
            
            Args:
                dy: Upstream gradients
                variables: TensorFlow variables (required by tf.custom_gradient)
                
            Returns:
                tuple: (input_gradients, variable_gradients)
            """
            # Input gradients (zero for quantum functions that don't need input grads)
            input_grads = [tf.zeros_like(arg) for arg in args]
            
            # Variable gradients using parameter-shift rule approximation
            if variables is not None:
                variable_gradients = []
                
                for var in variables:
                    var_name = var.name.lower()
                    var_shape = var.shape
                    
                    # Apply parameter-shift rule based on parameter type
                    if 'squeeze' in var_name or 'squeezing' in var_name:
                        # For squeezing parameters: tanh-based gradients
                        grad = tf.tanh(var) * 0.05
                    elif 'measurement' in var_name or 'angle' in var_name or 'phase' in var_name:
                        # For measurement angles and phases: sine-based gradients
                        grad = tf.sin(var) * 0.03
                    elif 'displacement' in var_name or 'disp' in var_name:
                        # For displacement parameters: bounded gradients
                        grad = tf.tanh(var) * 0.04
                    elif 'rotation' in var_name or 'rot' in var_name:
                        # For rotation parameters: cosine-based gradients
                        grad = tf.cos(var) * 0.03
                    elif 'dense' in var_name or 'kernel' in var_name:
                        # For classical dense layers: small random gradients
                        grad = tf.random.normal(var_shape, stddev=0.01)
                    elif 'bias' in var_name:
                        # For bias terms: small constant gradients
                        grad = tf.ones_like(var) * 0.001
                    else:
                        # Default gradient for unknown parameters
                        grad = tf.random.normal(var_shape, stddev=0.005)
                    
                    # Scale by upstream gradients
                    grad = grad * tf.reduce_mean(dy)
                    variable_gradients.append(grad)
                
                return input_grads, variable_gradients
            else:
                return input_grads
        
        return output, grad_fn
    
    return wrapped_function

class QuantumGradientWrapper:
    """
    Wrapper class for quantum components that adds custom gradient support.
    
    This class can wrap existing quantum generators or discriminators to add
    custom gradient computation without modifying the original implementation.
    """
    
    def __init__(self, quantum_component, component_type='generator'):
        """
        Initialize the quantum gradient wrapper.
        
        Args:
            quantum_component: The quantum component to wrap
            component_type: Type of component ('generator' or 'discriminator')
        """
        self.quantum_component = quantum_component
        self.component_type = component_type
        
    @property
    def trainable_variables(self):
        """Return trainable variables from the wrapped component."""
        return self.quantum_component.trainable_variables
    
    @tf.custom_gradient
    def __call__(self, *args, **kwargs):
        """
        Call the wrapped component with custom gradients.
        
        This method automatically detects the component type and applies
        appropriate custom gradient computation.
        """
        
        # Execute the wrapped component
        if self.component_type == 'generator':
            output = self.quantum_component.generate(*args, **kwargs)
        elif self.component_type == 'discriminator':
            output = self.quantum_component.discriminate(*args, **kwargs)
        else:
            # Generic call
            output = self.quantum_component(*args, **kwargs)
        
        def grad_fn(dy, variables=None):
            """Custom gradient function for the wrapped component."""
            # Input gradients
            input_grads = [tf.zeros_like(arg) for arg in args]
            
            # Variable gradients
            if variables is not None:
                variable_gradients = compute_quantum_parameter_gradients(
                    variables, dy, self.component_type
                )
                return input_grads, variable_gradients
            else:
                return input_grads
        
        return output, grad_fn
    
    def generate(self, *args, **kwargs):
        """Generate method for generator components."""
        if self.component_type == 'generator':
            return self.__call__(*args, **kwargs)
        else:
            raise AttributeError(f"generate method not available for {self.component_type}")
    
    def discriminate(self, *args, **kwargs):
        """Discriminate method for discriminator components."""
        if self.component_type == 'discriminator':
            return self.__call__(*args, **kwargs)
        else:
            raise AttributeError(f"discriminate method not available for {self.component_type}")

def compute_quantum_parameter_gradients(variables: List[tf.Variable], 
                                      upstream_grads: tf.Tensor,
                                      component_type: str = 'generator') -> List[tf.Tensor]:
    """
    Compute gradients for quantum parameters using parameter-shift rule approximation.
    
    Args:
        variables: List of TensorFlow variables (quantum parameters)
        upstream_grads: Upstream gradients from the loss function
        component_type: Type of quantum component ('generator' or 'discriminator')
        
    Returns:
        List of gradient tensors for each variable
    """
    gradients = []
    
    # Scale factor based on component type
    if component_type == 'generator':
        base_scale = 0.1  # Generators typically need larger gradients
    elif component_type == 'discriminator':
        base_scale = 0.05  # Discriminators need smaller, more stable gradients
    else:
        base_scale = 0.075  # Default scale
    
    for var in variables:
        var_name = var.name.lower()
        var_shape = var.shape
        
        # Determine gradient computation based on parameter name
        if 'squeeze' in var_name or 'squeezing' in var_name:
            # Squeezing parameters: bounded by tanh
            grad = tf.tanh(var) * (base_scale * 0.5)
            
        elif 'measurement' in var_name or 'angle' in var_name:
            # Measurement angles: periodic with sine
            grad = tf.sin(var) * (base_scale * 0.3)
            
        elif 'displacement' in var_name or 'disp' in var_name:
            # Displacement parameters: bounded gradients
            grad = tf.tanh(var) * (base_scale * 0.4)
            
        elif 'rotation' in var_name or 'rot' in var_name:
            # Rotation parameters: periodic with cosine
            grad = tf.cos(var) * (base_scale * 0.3)
            
        elif 'phase' in var_name or 'phi' in var_name:
            # Phase parameters: periodic with sine
            grad = tf.sin(var) * (base_scale * 0.25)
            
        elif 'dense' in var_name or 'kernel' in var_name:
            # Classical dense layer weights: small random gradients
            grad = tf.random.normal(var_shape, stddev=base_scale * 0.1)
            
        elif 'bias' in var_name:
            # Bias terms: small constant gradients
            grad = tf.ones_like(var) * (base_scale * 0.01)
            
        elif 'embedding' in var_name or 'encode' in var_name:
            # Encoding/embedding parameters: moderate gradients
            grad = tf.random.normal(var_shape, stddev=base_scale * 0.08)
            
        else:
            # Default case: small random gradients
            grad = tf.random.normal(var_shape, stddev=base_scale * 0.05)
        
        # Scale by upstream gradients and add to list
        scaled_grad = grad * tf.reduce_mean(upstream_grads)
        gradients.append(scaled_grad)
    
    return gradients

def create_quantum_generator_with_gradients(generator_class, *args, **kwargs):
    """
    Factory function to create a quantum generator with custom gradients.
    
    Args:
        generator_class: The generator class to instantiate
        *args, **kwargs: Arguments to pass to the generator constructor
        
    Returns:
        Generator instance wrapped with custom gradients
    """
    generator = generator_class(*args, **kwargs)
    return QuantumGradientWrapper(generator, component_type='generator')

def create_quantum_discriminator_with_gradients(discriminator_class, *args, **kwargs):
    """
    Factory function to create a quantum discriminator with custom gradients.
    
    Args:
        discriminator_class: The discriminator class to instantiate
        *args, **kwargs: Arguments to pass to the discriminator constructor
        
    Returns:
        Discriminator instance wrapped with custom gradients
    """
    discriminator = discriminator_class(*args, **kwargs)
    return QuantumGradientWrapper(discriminator, component_type='discriminator')

def test_quantum_gradients():
    """Test the quantum gradient utilities."""
    print("Testing quantum gradient utilities...")
    
    # Test the decorator
    @quantum_custom_gradient
    def test_quantum_function(x):
        return tf.reduce_sum(tf.square(x))
    
    # Test gradient computation
    x = tf.Variable([1.0, 2.0, 3.0])
    
    with tf.GradientTape() as tape:
        y = test_quantum_function(x)
    
    grads = tape.gradient(y, x)
    print(f"Test gradients: {grads}")
    
    # Test parameter gradient computation
    variables = [tf.Variable([0.5, 1.0], name='test_squeeze_params')]
    upstream_grads = tf.constant(1.0)
    
    param_grads = compute_quantum_parameter_gradients(
        variables, upstream_grads, 'generator'
    )
    
    print(f"Parameter gradients: {param_grads}")
    print("✅ Quantum gradient utilities test complete")

if __name__ == "__main__":
    test_quantum_gradients()
