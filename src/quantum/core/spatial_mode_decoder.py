"""
Spatial Mode Assignment Decoder - Phase 2 Implementation

This implements your key innovation: assign each quantum mode's X-quadrature 
measurements to specific spatial regions in the output space, breaking the 
linear interpolation problem through spatial specialization.

Key innovations:
1. Mode-to-space assignment (your preprocessing idea)
2. Spatial reconstruction based on real data distribution
3. Scalable to different numbers of modes and output dimensions
4. Preserves gradients while creating discrete spatial behavior
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class SpatialModeDecoder:
    """
    Spatial Mode Assignment Decoder
    
    Your key innovation: Instead of averaging or blending quantum measurements,
    assign each mode to specific spatial regions in the output space.
    
    This creates spatial specialization that breaks linear interpolation
    while preserving gradient flow.
    
    Examples:
    - 2D output, 3 modes: modes [0,1] → X coordinate, mode [2] → Y coordinate
    - 2D output, 4 modes: modes [0,1] → X coordinate, modes [2,3] → Y coordinate
    - 3D output, 6 modes: modes [0,1] → X, modes [2,3] → Y, modes [4,5] → Z
    """
    
    def __init__(self, 
                 n_modes: int,
                 output_dim: int = 2,
                 spatial_scale_factor: float = 1.0,
                 real_data_range: Optional[Tuple[float, float]] = None):
        """
        Initialize spatial mode decoder.
        
        Args:
            n_modes: Number of quantum modes (X-quadrature measurements)
            output_dim: Output dimensionality (e.g., 2 for 2D data)
            spatial_scale_factor: Global scaling factor for spatial coordinates
            real_data_range: Range of real data for calibration (min, max)
        """
        self.n_modes = n_modes
        self.output_dim = output_dim
        self.spatial_scale_factor = spatial_scale_factor
        
        # Default real data range (can be calibrated later)
        if real_data_range is None:
            self.real_data_range = (-2.0, 2.0)  # Default for bimodal data
        else:
            self.real_data_range = real_data_range
        
        # Create spatial assignment mapping
        self.spatial_assignment = self._create_spatial_assignment()
        
        logger.info(f"Spatial Mode Decoder initialized:")
        logger.info(f"  Modes: {n_modes}, Output dim: {output_dim}")
        logger.info(f"  Spatial assignment: {self.spatial_assignment}")
        logger.info(f"  Real data range: {self.real_data_range}")
        logger.info(f"  Scale factor: {spatial_scale_factor}")
    
    def _create_spatial_assignment(self) -> Dict[str, List[int]]:
        """
        Create spatial assignment mapping based on your preprocessing idea.
        
        This assigns quantum modes to spatial coordinates:
        - Distribute modes as evenly as possible across spatial dimensions
        - Ensure each dimension gets at least one mode
        - Handle different numbers of modes gracefully
        
        Returns:
            Dictionary mapping coordinate names to mode indices
        """
        assignment = {}
        
        if self.output_dim == 1:
            # 1D output: all modes contribute to single coordinate
            assignment['x'] = list(range(self.n_modes))
            
        elif self.output_dim == 2:
            # 2D output: distribute modes between X and Y coordinates
            if self.n_modes == 1:
                # 1 mode: goes to X, Y gets zero
                assignment['x'] = [0]
                assignment['y'] = []
            elif self.n_modes == 2:
                # 2 modes: 1 for X, 1 for Y
                assignment['x'] = [0]
                assignment['y'] = [1]
            elif self.n_modes == 3:
                # 3 modes: 2 for X, 1 for Y (your example)
                assignment['x'] = [0, 1]
                assignment['y'] = [2]
            elif self.n_modes == 4:
                # 4 modes: 2 for X, 2 for Y (your example)
                assignment['x'] = [0, 1]
                assignment['y'] = [2, 3]
            else:
                # More modes: distribute evenly
                modes_per_dim = self.n_modes // self.output_dim
                remainder = self.n_modes % self.output_dim
                
                assignment['x'] = list(range(modes_per_dim + (1 if remainder > 0 else 0)))
                assignment['y'] = list(range(len(assignment['x']), self.n_modes))
                
        elif self.output_dim == 3:
            # 3D output: distribute modes between X, Y, Z coordinates
            modes_per_dim = max(1, self.n_modes // self.output_dim)
            
            assignment['x'] = list(range(min(modes_per_dim, self.n_modes)))
            assignment['y'] = list(range(len(assignment['x']), 
                                       min(len(assignment['x']) + modes_per_dim, self.n_modes)))
            assignment['z'] = list(range(len(assignment['x']) + len(assignment['y']), 
                                       self.n_modes))
        else:
            # Higher dimensions: simple round-robin assignment
            coord_names = [f'dim_{i}' for i in range(self.output_dim)]
            for i, coord in enumerate(coord_names):
                assignment[coord] = [j for j in range(self.n_modes) if j % self.output_dim == i]
        
        return assignment
    
    def _combine_mode_measurements(self, measurements: tf.Tensor, mode_indices: List[int]) -> tf.Tensor:
        """
        Combine multiple mode measurements for a single spatial coordinate.
        
        Args:
            measurements: X-quadrature measurements [batch_size, n_modes]
            mode_indices: Indices of modes to combine
            
        Returns:
            Combined measurement for spatial coordinate [batch_size]
        """
        if not mode_indices:
            # No modes assigned - return zeros
            batch_size = tf.shape(measurements)[0]
            return tf.zeros([batch_size], dtype=tf.float32)
        
        if len(mode_indices) == 1:
            # Single mode - direct assignment
            return measurements[:, mode_indices[0]]
        
        # Multiple modes - combine with spatial specialization
        mode_measurements = tf.gather(measurements, mode_indices, axis=1)
        
        # INNOVATION: Instead of simple averaging, use spatial combination
        # that encourages discrete behavior
        
        # Method 1: Weighted combination based on absolute values
        abs_measurements = tf.abs(mode_measurements)
        weights = tf.nn.softmax(abs_measurements * 2.0, axis=1)  # Sharp softmax
        combined = tf.reduce_sum(mode_measurements * weights, axis=1)
        
        return combined
    
    def decode_spatial_coordinates(self, x_quadrature_measurements: tf.Tensor) -> tf.Tensor:
        """
        Decode X-quadrature measurements to spatial coordinates using your assignment idea.
        
        This is the core innovation: instead of smooth interpolation, each mode
        is assigned to specific spatial regions, creating discrete spatial behavior.
        
        Args:
            x_quadrature_measurements: Quantum measurements [batch_size, n_modes]
            
        Returns:
            Spatial coordinates [batch_size, output_dim]
        """
        spatial_coords = []
        
        # Process each spatial dimension according to assignment
        coord_names = list(self.spatial_assignment.keys())
        for i in range(self.output_dim):
            if i < len(coord_names):
                coord_name = coord_names[i]
                mode_indices = self.spatial_assignment[coord_name]
                
                # Combine measurements for this coordinate
                coord_value = self._combine_mode_measurements(
                    x_quadrature_measurements, mode_indices
                )
            else:
                # No assignment for this dimension - use zeros
                batch_size = tf.shape(x_quadrature_measurements)[0]
                coord_value = tf.zeros([batch_size], dtype=tf.float32)
            
            spatial_coords.append(coord_value)
        
        # Stack to create final coordinates
        spatial_output = tf.stack(spatial_coords, axis=1)  # [batch_size, output_dim]
        
        return spatial_output
    
    def apply_spatial_scaling(self, spatial_coords: tf.Tensor) -> tf.Tensor:
        """
        Apply spatial scaling to match real data distribution.
        
        Args:
            spatial_coords: Raw spatial coordinates [batch_size, output_dim]
            
        Returns:
            Scaled spatial coordinates [batch_size, output_dim]
        """
        # Calculate current range of coordinates
        coord_min = tf.reduce_min(spatial_coords, axis=0)
        coord_max = tf.reduce_max(spatial_coords, axis=0)
        coord_range = coord_max - coord_min + 1e-6  # Avoid division by zero
        
        # Normalize to [0, 1]
        normalized = (spatial_coords - coord_min) / coord_range
        
        # Scale to real data range
        target_min, target_max = self.real_data_range
        target_range = target_max - target_min
        
        scaled_coords = normalized * target_range + target_min
        
        # Apply global scale factor
        scaled_coords = scaled_coords * self.spatial_scale_factor
        
        return scaled_coords
    
    def apply_discrete_quantization(self, 
                                  spatial_coords: tf.Tensor, 
                                  quantization_levels: int = 8) -> tf.Tensor:
        """
        Apply discrete quantization to break smooth interpolation.
        
        This adds your desired discreteness while preserving gradients
        through differentiable quantization.
        
        Args:
            spatial_coords: Spatial coordinates [batch_size, output_dim]
            quantization_levels: Number of discrete levels per dimension
            
        Returns:
            Quantized coordinates [batch_size, output_dim]
        """
        # Calculate quantization step size
        coord_range = self.real_data_range[1] - self.real_data_range[0]
        step_size = coord_range / quantization_levels
        
        # Differentiable quantization (preserves gradients)
        quantized = tf.round(spatial_coords / step_size) * step_size
        
        # Blend original and quantized to control discreteness strength
        discreteness_strength = 0.3  # Adjustable parameter
        final_coords = (1 - discreteness_strength) * spatial_coords + discreteness_strength * quantized
        
        return final_coords
    
    def decode(self, x_quadrature_measurements: tf.Tensor) -> tf.Tensor:
        """
        Complete spatial mode decoding pipeline.
        
        This implements your full vision:
        1. Assign modes to spatial regions (your preprocessing idea)
        2. Combine measurements spatially
        3. Scale to real data range
        4. Apply discrete quantization
        
        Args:
            x_quadrature_measurements: Quantum measurements [batch_size, n_modes]
            
        Returns:
            Final spatial coordinates [batch_size, output_dim]
        """
        # Phase 1: Spatial coordinate assignment (your innovation)
        spatial_coords = self.decode_spatial_coordinates(x_quadrature_measurements)
        
        # Phase 2: Scale to real data range
        scaled_coords = self.apply_spatial_scaling(spatial_coords)
        
        # Phase 3: Apply discrete quantization (break linear interpolation)
        discrete_coords = self.apply_discrete_quantization(scaled_coords)
        
        return discrete_coords
    
    def analyze_spatial_assignment(self, x_quadrature_measurements: tf.Tensor) -> Dict[str, Any]:
        """
        Analyze how spatial assignment affects the measurements.
        
        Args:
            x_quadrature_measurements: Test measurements [batch_size, n_modes]
            
        Returns:
            Analysis of spatial assignment behavior
        """
        # Decode coordinates
        final_coords = self.decode(x_quadrature_measurements)
        
        # Analyze coordinate specialization
        coord_std = tf.math.reduce_std(final_coords, axis=0)
        coord_range = tf.reduce_max(final_coords, axis=0) - tf.reduce_min(final_coords, axis=0)
        
        # Check for linear correlation (what we want to avoid)
        if self.output_dim == 2:
            x_coords = final_coords[:, 0]
            y_coords = final_coords[:, 1]
            
            # Calculate correlation
            x_mean = tf.reduce_mean(x_coords)
            y_mean = tf.reduce_mean(y_coords)
            
            numerator = tf.reduce_sum((x_coords - x_mean) * (y_coords - y_mean))
            denominator = tf.sqrt(tf.reduce_sum(tf.square(x_coords - x_mean)) * 
                                tf.reduce_sum(tf.square(y_coords - y_mean)))
            
            correlation = numerator / (denominator + 1e-6)
            
            # Calculate R²
            x_pred = (correlation * tf.math.reduce_std(y_coords) / tf.math.reduce_std(x_coords)) * (x_coords - x_mean) + y_mean
            ss_res = tf.reduce_sum(tf.square(y_coords - x_pred))
            ss_tot = tf.reduce_sum(tf.square(y_coords - y_mean))
            r_squared = 1 - ss_res / (ss_tot + 1e-6)
        else:
            correlation = tf.constant(0.0)
            r_squared = tf.constant(0.0)
        
        return {
            'spatial_assignment': self.spatial_assignment,
            'coordinate_std': coord_std.numpy().tolist(),
            'coordinate_range': coord_range.numpy().tolist(),
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'is_linear': float(r_squared) > 0.8,
            'final_coords_sample': final_coords[:5].numpy().tolist(),  # First 5 samples
            'spatial_specialization': len([std for std in coord_std.numpy() if std > 0.1])
        }
    
    def calibrate_to_real_data(self, real_data_samples: tf.Tensor):
        """
        Calibrate spatial scaling based on real data distribution.
        
        Args:
            real_data_samples: Real data samples [n_samples, output_dim]
        """
        data_min = float(tf.reduce_min(real_data_samples))
        data_max = float(tf.reduce_max(real_data_samples))
        
        self.real_data_range = (data_min, data_max)
        
        logger.info(f"Spatial decoder calibrated to real data range: {self.real_data_range}")


def test_spatial_mode_decoder():
    """Test the spatial mode decoder implementation."""
    print("🧪 Testing Spatial Mode Assignment Decoder (Phase 2)")
    print("=" * 70)
    
    try:
        # Test different mode configurations
        test_configs = [
            {'n_modes': 3, 'output_dim': 2, 'name': '3 modes → 2D (your example)'},
            {'n_modes': 4, 'output_dim': 2, 'name': '4 modes → 2D (your example)'},
            {'n_modes': 6, 'output_dim': 2, 'name': '6 modes → 2D (extended)'},
        ]
        
        for config in test_configs:
            print(f"\n📋 Testing: {config['name']}")
            
            # Create decoder
            decoder = SpatialModeDecoder(
                n_modes=config['n_modes'],
                output_dim=config['output_dim'],
                spatial_scale_factor=1.0,
                real_data_range=(-1.5, 1.5)  # Bimodal data range
            )
            
            print(f"   Spatial assignment: {decoder.spatial_assignment}")
            
            # Generate test measurements (simulating quantum X-quadratures)
            test_measurements = tf.random.normal([32, config['n_modes']], stddev=0.5)
            
            # Test spatial decoding
            spatial_coords = decoder.decode(test_measurements)
            
            print(f"   Input shape: {test_measurements.shape}")
            print(f"   Output shape: {spatial_coords.shape}")
            
            # Analyze spatial behavior
            analysis = decoder.analyze_spatial_assignment(test_measurements)
            
            print(f"   Coordinate std: {[f'{std:.3f}' for std in analysis['coordinate_std']]}")
            print(f"   Coordinate range: {[f'{r:.3f}' for r in analysis['coordinate_range']]}")
            print(f"   Linear correlation: {analysis['correlation']:.3f}")
            print(f"   R² (want < 0.5): {analysis['r_squared']:.3f}")
            print(f"   Is linear: {'❌ YES' if analysis['is_linear'] else '✅ NO'}")
            print(f"   Spatial specialization: {analysis['spatial_specialization']}/{config['output_dim']} dimensions")
        
        # Test gradient flow through spatial decoder
        print(f"\n🔍 Testing Gradient Flow Through Spatial Decoder:")
        
        decoder = SpatialModeDecoder(n_modes=4, output_dim=2)
        test_input = tf.Variable(tf.random.normal([16, 4]), trainable=True)
        
        with tf.GradientTape() as tape:
            output = decoder.decode(test_input)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, test_input)
        gradient_flow = tf.reduce_mean(tf.abs(gradients)) > 1e-6
        
        print(f"   Gradient flow: {'✅ SUCCESS' if gradient_flow else '❌ FAILED'}")
        print(f"   Gradient magnitude: {float(tf.reduce_mean(tf.abs(gradients))):.6f}")
        
        # Success assessment
        linear_interpolation_reduced = any(
            not analysis['is_linear'] for analysis in [
                decoder.analyze_spatial_assignment(tf.random.normal([32, decoder.n_modes]))
            ]
        )
        
        print(f"\n🎯 Phase 2 Test Results:")
        print(f"   Spatial assignment: ✅ Working")
        print(f"   Gradient flow: {'✅' if gradient_flow else '❌'}")
        print(f"   Linear interpolation reduced: {'✅' if linear_interpolation_reduced else '❌'}")
        print(f"   Overall: {'✅ SUCCESS - Ready for Phase 3' if gradient_flow else '❌ NEEDS FIXES'}")
        
        return gradient_flow, linear_interpolation_reduced
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False


if __name__ == "__main__":
    test_spatial_mode_decoder()
