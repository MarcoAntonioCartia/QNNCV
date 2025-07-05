"""
Quantum Generator Performance Benchmarking

This script provides comprehensive benchmarking of the complete quantum generator
solution to identify performance bottlenecks and optimization opportunities.

Focus areas:
1. Single forward pass timing
2. Component-level breakdown 
3. Batch size scaling analysis
4. Memory usage profiling
5. Bottleneck identification
"""

import numpy as np
import tensorflow as tf
import time
import tracemalloc
import os
import sys
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.input_dependent_quantum_circuit import InputDependentQuantumCircuit
from src.quantum.core.spatial_mode_decoder import SpatialModeDecoder
from src.examples.train_phase3_complete_quantum_generator import CompleteQuantumGenerator


class QuantumGeneratorBenchmark:
    """
    Comprehensive performance benchmarking for quantum generator components.
    
    Measures:
    - Component-level execution times
    - Memory usage patterns
    - Batch size scaling behavior
    - Bottleneck identification
    """
    
    def __init__(self):
        """Initialize benchmarking suite."""
        self.results = {
            'single_pass_times': [],
            'component_breakdown': [],
            'batch_scaling': [],
            'memory_usage': [],
            'bottlenecks': {}
        }
        
        # Test configurations
        self.test_configs = [
            {'batch_size': 1, 'latent_dim': 4, 'n_modes': 4, 'n_layers': 2},
            {'batch_size': 4, 'latent_dim': 4, 'n_modes': 4, 'n_layers': 2},
            {'batch_size': 8, 'latent_dim': 4, 'n_modes': 4, 'n_layers': 2},
            {'batch_size': 16, 'latent_dim': 4, 'n_modes': 4, 'n_layers': 2},
            {'batch_size': 32, 'latent_dim': 4, 'n_modes': 4, 'n_layers': 2},
        ]
        
        print("Quantum Generator Performance Benchmark initialized")
        print(f"Test configurations: {len(self.test_configs)}")
        
    def setup_components(self, config: Dict) -> Tuple[Any, Any, Any]:
        """Setup individual components for testing."""
        # Individual components
        quantum_circuit = InputDependentQuantumCircuit(
            n_modes=config['n_modes'],
            n_layers=config['n_layers'],
            latent_dim=config['latent_dim'],
            cutoff_dim=4,
            input_scale_factor=0.1
        )
        
        spatial_decoder = SpatialModeDecoder(
            n_modes=config['n_modes'],
            output_dim=2,
            spatial_scale_factor=1.0,
            real_data_range=(-1.5, 1.5)
        )
        
        # Complete generator
        complete_generator = CompleteQuantumGenerator(
            latent_dim=config['latent_dim'],
            output_dim=2,
            n_modes=config['n_modes'],
            n_layers=config['n_layers'],
            cutoff_dim=4
        )
        
        return quantum_circuit, spatial_decoder, complete_generator
    
    def time_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    def benchmark_single_forward_pass(self, config: Dict) -> Dict[str, Any]:
        """Benchmark single forward pass with detailed component breakdown."""
        print(f"\n📊 Benchmarking single forward pass - Batch size: {config['batch_size']}")
        
        quantum_circuit, spatial_decoder, complete_generator = self.setup_components(config)
        
        # Generate test input
        latent_batch = tf.random.normal([config['batch_size'], config['latent_dim']])
        
        # Warm-up run (important for accurate timing)
        _ = complete_generator.generate(latent_batch)
        
        # Start memory tracing
        tracemalloc.start()
        
        # 1. Complete forward pass timing
        complete_result, complete_time = self.time_function(
            complete_generator.generate, latent_batch
        )
        
        # 2. Component breakdown
        print("   Measuring component breakdown...")
        
        # Quantum circuit timing
        quantum_result, quantum_time = self.time_function(
            quantum_circuit.process_batch, latent_batch
        )
        
        # Spatial decoder timing  
        spatial_result, spatial_time = self.time_function(
            spatial_decoder.decode, quantum_result
        )
        
        # Individual sample processing timing (the suspected bottleneck)
        print("   Measuring individual vs batch processing...")
        individual_times = []
        for i in range(config['batch_size']):
            single_latent = latent_batch[i:i+1]  # Single sample
            _, single_time = self.time_function(
                complete_generator.generate, single_latent
            )
            individual_times.append(single_time)
        
        total_individual_time = sum(individual_times)
        avg_individual_time = np.mean(individual_times)
        
        # Memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate overhead
        component_sum = quantum_time + spatial_time
        overhead = complete_time - component_sum
        
        results = {
            'config': config,
            'complete_forward_pass_time': complete_time,
            'quantum_circuit_time': quantum_time,
            'spatial_decoder_time': spatial_time, 
            'component_sum_time': component_sum,
            'overhead_time': overhead,
            'avg_individual_time': avg_individual_time,
            'total_individual_time': total_individual_time,
            'batch_efficiency': complete_time / total_individual_time if total_individual_time > 0 else 1.0,
            'memory_current_mb': current / 1024 / 1024,
            'memory_peak_mb': peak / 1024 / 1024,
            'samples_per_second': config['batch_size'] / complete_time,
            'time_per_sample_ms': (complete_time / config['batch_size']) * 1000
        }
        
        # Print results
        print(f"   Complete forward pass: {complete_time:.4f}s")
        print(f"   Quantum circuit: {quantum_time:.4f}s ({quantum_time/complete_time*100:.1f}%)")
        print(f"   Spatial decoder: {spatial_time:.4f}s ({spatial_time/complete_time*100:.1f}%)")
        print(f"   Overhead: {overhead:.4f}s ({overhead/complete_time*100:.1f}%)")
        print(f"   Time per sample: {results['time_per_sample_ms']:.2f}ms")
        print(f"   Samples per second: {results['samples_per_second']:.2f}")
        print(f"   Memory peak: {results['memory_peak_mb']:.2f}MB")
        print(f"   Batch efficiency: {results['batch_efficiency']:.3f}")
        
        return results
    
    def benchmark_batch_scaling(self) -> List[Dict]:
        """Benchmark how performance scales with batch size."""
        print("\n🔍 Analyzing Batch Size Scaling Performance")
        print("=" * 60)
        
        scaling_results = []
        
        for config in self.test_configs:
            print(f"\nTesting batch size: {config['batch_size']}")
            result = self.benchmark_single_forward_pass(config)
            scaling_results.append(result)
            
            # Add to results
            self.results['batch_scaling'].append(result)
        
        # Analyze scaling patterns
        print(f"\n📈 Batch Scaling Analysis:")
        print(f"{'Batch Size':<12} {'Time (s)':<10} {'Time/Sample (ms)':<16} {'Samples/s':<12} {'Efficiency':<12}")
        print("-" * 70)
        
        for result in scaling_results:
            print(f"{result['config']['batch_size']:<12} "
                  f"{result['complete_forward_pass_time']:<10.4f} "
                  f"{result['time_per_sample_ms']:<16.2f} "
                  f"{result['samples_per_second']:<12.2f} "
                  f"{result['batch_efficiency']:<12.3f}")
        
        return scaling_results
    
    def benchmark_component_bottlenecks(self) -> Dict[str, Any]:
        """Identify the main performance bottlenecks."""
        print("\n🔍 Bottleneck Analysis")
        print("=" * 50)
        
        # Use a medium batch size for analysis
        config = {'batch_size': 8, 'latent_dim': 4, 'n_modes': 4, 'n_layers': 2}
        quantum_circuit, spatial_decoder, complete_generator = self.setup_components(config)
        
        latent_batch = tf.random.normal([config['batch_size'], config['latent_dim']])
        
        # Detailed quantum circuit breakdown
        print("\n1. Quantum Circuit Bottleneck Analysis:")
        
        # Individual sample processing (current implementation)
        individual_quantum_times = []
        for i in range(config['batch_size']):
            single_latent = latent_batch[i]
            _, time_taken = self.time_function(
                quantum_circuit.process_single_input, single_latent
            )
            individual_quantum_times.append(time_taken)
        
        avg_individual_quantum = np.mean(individual_quantum_times)
        total_individual_quantum = sum(individual_quantum_times)
        
        # Batch processing
        _, batch_quantum_time = self.time_function(
            quantum_circuit.process_batch, latent_batch
        )
        
        print(f"   Individual processing: {total_individual_quantum:.4f}s")
        print(f"   Batch processing: {batch_quantum_time:.4f}s")
        print(f"   Individual overhead: {(total_individual_quantum / batch_quantum_time):.2f}x slower")
        
        # Spatial decoder analysis
        print("\n2. Spatial Decoder Bottleneck Analysis:")
        test_measurements = tf.random.normal([config['batch_size'], config['n_modes']])
        
        # Component breakdown
        _, coord_decode_time = self.time_function(
            spatial_decoder.decode_spatial_coordinates, test_measurements
        )
        _, scaling_time = self.time_function(
            spatial_decoder.apply_spatial_scaling, 
            spatial_decoder.decode_spatial_coordinates(test_measurements)
        )
        _, quantization_time = self.time_function(
            spatial_decoder.apply_discrete_quantization,
            spatial_decoder.apply_spatial_scaling(
                spatial_decoder.decode_spatial_coordinates(test_measurements)
            )
        )
        
        print(f"   Coordinate decoding: {coord_decode_time:.6f}s")
        print(f"   Spatial scaling: {scaling_time:.6f}s") 
        print(f"   Quantization: {quantization_time:.6f}s")
        
        # Overall bottleneck identification
        bottlenecks = {
            'individual_quantum_processing': {
                'time': total_individual_quantum,
                'percentage': (total_individual_quantum / batch_quantum_time) * 100,
                'severity': 'HIGH' if total_individual_quantum / batch_quantum_time > 2 else 'MEDIUM'
            },
            'quantum_circuit_total': {
                'time': batch_quantum_time,
                'percentage': 100,  # Base reference
                'severity': 'HIGH' if batch_quantum_time > 0.1 else 'MEDIUM'
            },
            'spatial_decoder_total': {
                'time': coord_decode_time + scaling_time + quantization_time,
                'percentage': ((coord_decode_time + scaling_time + quantization_time) / batch_quantum_time) * 100,
                'severity': 'LOW'  # Spatial decoder is typically fast
            }
        }
        
        print(f"\n🚨 Primary Bottlenecks Identified:")
        for name, data in bottlenecks.items():
            if data['severity'] == 'HIGH':
                print(f"   HIGH: {name} - {data['time']:.4f}s")
        
        self.results['bottlenecks'] = bottlenecks
        return bottlenecks
    
    def benchmark_optimization_opportunities(self) -> Dict[str, Any]:
        """Benchmark potential optimization strategies."""
        print("\n🚀 Optimization Opportunity Analysis")
        print("=" * 50)
        
        config = {'batch_size': 16, 'latent_dim': 4, 'n_modes': 4, 'n_layers': 2}
        
        # Current implementation timing
        generator_config = {k: v for k, v in config.items() if k != 'batch_size'}
        complete_generator = CompleteQuantumGenerator(**generator_config, output_dim=2, cutoff_dim=4)
        latent_batch = tf.random.normal([config['batch_size'], config['latent_dim']])
        
        _, current_time = self.time_function(complete_generator.generate, latent_batch)
        
        print(f"Current implementation: {current_time:.4f}s")
        
        # Theoretical optimizations
        optimizations = {
            'vectorized_quantum_processing': {
                'description': 'Vectorized quantum state preparation',
                'estimated_speedup': '5-10x',
                'implementation_difficulty': 'HIGH',
                'estimated_time': current_time / 7  # Conservative estimate
            },
            'cached_quantum_programs': {
                'description': 'Cache quantum programs between calls',
                'estimated_speedup': '2-3x', 
                'implementation_difficulty': 'MEDIUM',
                'estimated_time': current_time / 2.5
            },
            'optimized_spatial_decoding': {
                'description': 'Vectorized spatial operations',
                'estimated_speedup': '1.2-1.5x',
                'implementation_difficulty': 'LOW',
                'estimated_time': current_time / 1.3
            },
            'reduced_precision': {
                'description': 'Lower cutoff dimensions',
                'estimated_speedup': '2-4x',
                'implementation_difficulty': 'LOW',
                'estimated_time': current_time / 3
            }
        }
        
        print(f"\nOptimization Opportunities:")
        for name, opt in optimizations.items():
            print(f"   {name}:")
            print(f"     Description: {opt['description']}")
            print(f"     Estimated speedup: {opt['estimated_speedup']}")
            print(f"     Difficulty: {opt['implementation_difficulty']}")
            print(f"     Estimated time: {opt['estimated_time']:.4f}s")
        
        return optimizations
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("=" * 80)
        report.append("QUANTUM GENERATOR PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        if self.results['batch_scaling']:
            batch_1_result = next(r for r in self.results['batch_scaling'] if r['config']['batch_size'] == 1)
            batch_16_result = next((r for r in self.results['batch_scaling'] if r['config']['batch_size'] == 16), None)
            
            report.append(f"\nPERFORMANCE SUMMARY:")
            report.append(f"  Single sample time: {batch_1_result['time_per_sample_ms']:.2f}ms")
            if batch_16_result:
                report.append(f"  Batch-16 efficiency: {batch_16_result['batch_efficiency']:.3f}")
                report.append(f"  Batch-16 samples/sec: {batch_16_result['samples_per_second']:.2f}")
        
        # Bottleneck analysis
        if self.results['bottlenecks']:
            report.append(f"\nBOTTLENECK ANALYSIS:")
            for name, data in self.results['bottlenecks'].items():
                if data['severity'] == 'HIGH':
                    report.append(f"  HIGH PRIORITY: {name} ({data['time']:.4f}s)")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        report.append(f"  1. Implement vectorized quantum processing (5-10x speedup)")
        report.append(f"  2. Cache quantum programs between calls (2-3x speedup)")
        report.append(f"  3. Consider reducing cutoff dimensions (2-4x speedup)")
        report.append(f"  4. Optimize spatial decoder operations (1.2-1.5x speedup)")
        
        return "\n".join(report)
    
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmarking suite."""
        print("🔬 Starting Comprehensive Quantum Generator Performance Benchmark")
        print("=" * 80)
        
        # Run all benchmarks
        scaling_results = self.benchmark_batch_scaling()
        bottleneck_analysis = self.benchmark_component_bottlenecks()
        optimization_analysis = self.benchmark_optimization_opportunities()
        
        # Generate report
        report = self.generate_performance_report()
        print(f"\n{report}")
        
        # Final results summary
        final_results = {
            'scaling_results': scaling_results,
            'bottleneck_analysis': bottleneck_analysis,
            'optimization_analysis': optimization_analysis,
            'performance_report': report
        }
        
        return final_results


def main():
    """Main benchmarking execution."""
    print("🔬 Quantum Generator Performance Benchmarking Suite")
    print("=" * 80)
    print("Analyzing performance bottlenecks in the complete quantum generator solution")
    print("Focus: Training time concerns and optimization opportunities")
    print("=" * 80)
    
    # Suppress TF warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Initialize and run benchmark
        benchmark = QuantumGeneratorBenchmark()
        results = benchmark.run_complete_benchmark()
        
        # Save results
        results_dir = "results/performance_benchmarks"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        import json
        with open(f"{results_dir}/quantum_generator_benchmark_results.txt", "w") as f:
            f.write(results['performance_report'])
        
        print(f"\n📊 Benchmark Results Summary:")
        print(f"   Results saved to: {results_dir}/")
        print(f"   Key finding: Performance bottlenecks identified")
        print(f"   Next steps: Implement optimization recommendations")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
