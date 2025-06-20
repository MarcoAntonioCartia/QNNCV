"""
Enhanced QGAN Visualization Integration Example
==============================================

This example shows how to integrate the visualization system with your existing
QGAN architecture and demonstrates all the visualization capabilities.

Run this from your src directory to see your quantum circuits and states!
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from datetime import datetime

# Import your QGAN components (adapt paths as needed)
# from quantum.core.quantum_circuit import PureQuantumCircuit
# from models.generators.quantum_generator import PureQuantumGenerator
# from models.discriminators.quantum_discriminator import PureQuantumDiscriminator
# from utils.warning_suppression import suppress_all_quantum_warnings

# Import our visualization system
# from utils.qgan_visualizer import QuantumGANVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QGANVisualizationDemo:
    """Complete demonstration of QGAN visualization capabilities."""
    
    def __init__(self):
        self.visualizer = None  # Will be initialized with the QuantumGANVisualizer
        
    def demo_circuit_visualization(self):
        """Demonstrate circuit visualization capabilities."""
        print("🔬 CIRCUIT VISUALIZATION DEMO")
        print("=" * 50)
        
        # Method 1: Using SF's built-in prog.print()
        print("\n1. 📋 Built-in SF Circuit Printing:")
        prog = sf.Program(3)
        with prog.context as q:
            ops.Sgate(0.54) | q[0]
            ops.Sgate(0.8) | q[1]
            ops.BSgate(0.43, 0.1) | (q[0], q[1])
            ops.Dgate(0.2+0.3j) | q[2]
            ops.MeasureHomodyne(0) | q[0]  # X quadrature
            ops.MeasureHomodyne(np.pi/2) | q[1]  # P quadrature
        
        prog.print()
        
        # Method 2: SF's LaTeX circuit drawing
        print("\n2. 📐 LaTeX Circuit Diagram (generates LaTeX code):")
        try:
            tex_code = prog.draw_circuit()
            print("LaTeX code generated successfully!")
            print("First few lines of LaTeX:")
            if isinstance(tex_code, list):
                print(tex_code[1][:200] + "..." if len(tex_code[1]) > 200 else tex_code[1])
            else:
                print(str(tex_code)[:200] + "...")
        except Exception as e:
            print(f"LaTeX generation failed: {e}")
        
        return prog
    
    def demo_state_visualization(self):
        """Demonstrate quantum state visualization."""
        print("\n🎭 QUANTUM STATE VISUALIZATION DEMO")
        print("=" * 50)
        
        # Create interesting quantum states
        states_to_demo = {
            "Vacuum": (lambda: self._create_vacuum_state(), "The ground state"),
            "Squeezed": (lambda: self._create_squeezed_state(), "Squeezed vacuum state"),
            "Displaced": (lambda: self._create_displaced_state(), "Coherent state"),
            "Cat State": (lambda: self._create_cat_state(), "Superposition of coherent states"),
            "Multi-mode": (lambda: self._create_multimode_state(), "Entangled multi-mode state")
        }
        
        for name, (state_func, description) in states_to_demo.items():
            print(f"\n{name} State: {description}")
            try:
                state = state_func()
                
                # 3D Mountain visualization
                self._create_mountain_viz(state, f" - {name} State")
                
                # Fock probabilities
                self._show_fock_probs(state, name)
                
            except Exception as e:
                print(f"Failed to create {name} state: {e}")
    
    def _create_vacuum_state(self):
        """Create vacuum state for visualization."""
        prog = sf.Program(2)
        with prog.context as q:
            # Vacuum is the default - no operations needed
            pass
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})
        result = eng.run(prog)
        return result.state
    
    def _create_squeezed_state(self):
        """Create squeezed state for visualization."""
        prog = sf.Program(2)
        with prog.context as q:
            ops.Sgate(1.0) | q[0]  # Strong squeezing
            ops.Sgate(0.5) | q[1]  # Moderate squeezing
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 8})
        result = eng.run(prog)
        return result.state
    
    def _create_displaced_state(self):
        """Create displaced (coherent) state."""
        prog = sf.Program(2)
        with prog.context as q:
            ops.Dgate(1.0+0.5j) | q[0]  # Displaced in x+ip
            ops.Dgate(-0.8+1.2j) | q[1]  # Different displacement
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 8})
        result = eng.run(prog)
        return result.state
    
    def _create_cat_state(self):
        """Create cat state for visualization."""
        prog = sf.Program(1)
        with prog.context as q:
            # Create cat state: superposition of ±α
            alpha = 1.0
            ops.Catstate(alpha, p=0) | q[0]
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 12})
        result = eng.run(prog)
        return result.state
    
    def _create_multimode_state(self):
        """Create entangled multi-mode state."""
        prog = sf.Program(3)
        with prog.context as q:
            # Create entangled state
            ops.Sgate(0.8) | q[0]
            ops.Sgate(0.6) | q[1]
            ops.Sgate(0.4) | q[2]
            
            # Entangling beamsplitters
            ops.BSgate(0.7, 0.2) | (q[0], q[1])
            ops.BSgate(0.5, -0.3) | (q[1], q[2])
            
            # Some displacement
            ops.Dgate(0.3+0.2j) | q[0]
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})
        result = eng.run(prog)
        return result.state
    
    def _create_mountain_viz(self, state, title_suffix=""):
        """Create 3D mountain visualization."""
        try:
            print(f"  Creating 3D Wigner function visualization...")
            
            # Single mode visualization
            x = np.linspace(-4, 4, 80)
            p = np.linspace(-4, 4, 80)
            X, P = np.meshgrid(x, p)
            
            # Calculate Wigner function for mode 0
            W = state.wigner(0, x, p)
            
            # Create 3D mountain plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Surface plot
            surface = ax.plot_surface(X, P, W, cmap='RdYlBu_r', 
                                    alpha=0.9, linewidth=0.5,
                                    rstride=2, cstride=2)
            
            # Add contour lines at base
            ax.contour(X, P, W, levels=12, colors='gray', 
                      alpha=0.6, offset=np.min(W)-0.02)
            
            # Styling
            ax.set_xlabel('Position (x)', fontsize=12)
            ax.set_ylabel('Momentum (p)', fontsize=12)
            ax.set_zlabel('Wigner Function', fontsize=12)
            ax.set_title(f'3D Wigner Function Mountain{title_suffix}', fontsize=14)
            
            # Add colorbar
            fig.colorbar(surface, shrink=0.6, aspect=20)
            
            plt.tight_layout()
            plt.show()
            
            # Multi-mode visualization if applicable
            if state.num_modes > 1:
                self._create_all_modes_viz(state, title_suffix)
                
        except Exception as e:
            print(f"    Mountain visualization failed: {e}")
    
    def _create_all_modes_viz(self, state, title_suffix=""):
        """Create all-modes mountain visualization."""
        try:
            print(f"  Creating multi-mode visualization...")
            
            n_modes = min(state.num_modes, 4)  # Limit to 4 modes for display
            
            fig = plt.figure(figsize=(15, 4*((n_modes+1)//2)))
            
            x = np.linspace(-3, 3, 60)
            p = np.linspace(-3, 3, 60)
            X, P = np.meshgrid(x, p)
            
            for mode in range(n_modes):
                ax = fig.add_subplot(2, 2, mode+1, projection='3d')
                
                W = state.wigner(mode, x, p)
                
                surface = ax.plot_surface(X, P, W, cmap='RdYlBu_r',
                                        alpha=0.9, linewidth=0.3,
                                        rstride=2, cstride=2)
                
                ax.contour(X, P, W, levels=8, colors='gray',
                          alpha=0.5, offset=np.min(W)-0.01)
                
                ax.set_title(f'Mode {mode}', fontsize=12)
                ax.set_xlabel('x', fontsize=10)
                ax.set_ylabel('p', fontsize=10)
                ax.set_zlabel('W', fontsize=10)
            
            fig.suptitle(f'Multi-Mode Quantum State{title_suffix}', fontsize=16)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"    Multi-mode visualization failed: {e}")
    
    def _show_fock_probs(self, state, state_name):
        """Show Fock state probabilities."""
        try:
            print(f"  Fock state probabilities for {state_name}:")
            
            # Calculate Fock probabilities for first few states
            n_modes = min(state.num_modes, 2)
            cutoff = min(6, getattr(state, 'cutoff_dim', 6))
            
            fig, axes = plt.subplots(1, n_modes, figsize=(5*n_modes, 4))
            if n_modes == 1:
                axes = [axes]
            
            for mode in range(n_modes):
                probs = []
                for n in range(cutoff):
                    fock_state = [0] * state.num_modes
                    fock_state[mode] = n
                    try:
                        prob = state.fock_prob(fock_state)
                        probs.append(prob)
                    except:
                        probs.append(0)
                
                axes[mode].bar(range(cutoff), probs, alpha=0.7)
                axes[mode].set_title(f'Mode {mode} Fock Probabilities')
                axes[mode].set_xlabel('Fock Number')
                axes[mode].set_ylabel('Probability')
                axes[mode].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"    Fock probability visualization failed: {e}")
    
    def demo_training_visualization(self):
        """Demonstrate training progress visualization."""
        print("\n📈 TRAINING PROGRESS VISUALIZATION DEMO")
        print("=" * 50)
        
        # Generate synthetic training data
        epochs = 100
        
        # Realistic QGAN training curves
        g_loss = self._generate_loss_curve(epochs, start=2.5, end=0.8, noise=0.3)
        d_loss = self._generate_loss_curve(epochs, start=1.8, end=0.6, noise=0.2)
        w_distance = self._generate_wasserstein_curve(epochs)
        gradient_penalty = np.random.exponential(0.1, epochs)
        g_gradients = np.random.exponential(0.5, epochs) * np.exp(-np.linspace(0, 2, epochs))
        d_gradients = np.random.exponential(0.3, epochs) * np.exp(-np.linspace(0, 1.5, epochs))
        
        training_history = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'w_distance': w_distance,
            'gradient_penalty': gradient_penalty,
            'g_gradients': g_gradients,
            'd_gradients': d_gradients
        }
        
        # Create training visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(g_loss, label='Generator', color='blue', linewidth=2)
        axes[0, 0].plot(d_loss, label='Discriminator', color='red', linewidth=2)
        axes[0, 0].set_title('Training Losses', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Wasserstein distance
        axes[0, 1].plot(w_distance, color='green', linewidth=2)
        axes[0, 1].set_title('Wasserstein Distance', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient penalty
        axes[0, 2].plot(gradient_penalty, color='purple', linewidth=2)
        axes[0, 2].set_title('Gradient Penalty', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Penalty')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Gradient norms
        axes[1, 0].plot(g_gradients, label='Generator', color='blue', linewidth=2)
        axes[1, 0].plot(d_gradients, label='Discriminator', color='red', linewidth=2)
        axes[1, 0].set_title('Gradient Norms', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parameter evolution heatmap
        n_params = 24  # Typical for 3-mode, 2-layer circuit
        param_evolution = np.random.randn(n_params, epochs)
        param_evolution = np.cumsum(param_evolution * 0.1, axis=1)  # Smooth evolution
        
        im = axes[1, 1].imshow(param_evolution, cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('Quantum Parameter Evolution', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Parameter Index')
        
        # Learning rate schedule
        lr_schedule = 0.001 * np.exp(-np.linspace(0, 2, epochs))
        axes[1, 2].plot(lr_schedule, color='orange', linewidth=2)
        axes[1, 2].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Quantum GAN Training Dashboard', fontsize=18, y=0.95)
        plt.tight_layout()
        plt.show()
        
        print("Training visualization features:")
        print("  ✓ Loss curves (Generator & Discriminator)")
        print("  ✓ Wasserstein distance tracking")
        print("  ✓ Gradient penalty monitoring")
        print("  ✓ Gradient norm evolution")
        print("  ✓ Parameter evolution heatmap")
        print("  ✓ Learning rate scheduling")
    
    def _generate_loss_curve(self, epochs, start, end, noise):
        """Generate realistic loss curve."""
        trend = np.linspace(start, end, epochs)
        noise_component = np.random.normal(0, noise, epochs)
        oscillation = 0.1 * np.sin(np.linspace(0, 4*np.pi, epochs))
        return trend + noise_component + oscillation
    
    def _generate_wasserstein_curve(self, epochs):
        """Generate realistic Wasserstein distance curve."""
        # Should converge to zero but oscillate
        base = np.exp(-np.linspace(0, 3, epochs))
        noise = np.random.normal(0, 0.05, epochs)
        oscillation = 0.02 * np.sin(np.linspace(0, 8*np.pi, epochs))
        return base + noise + oscillation
    
    def demo_qgan_integration(self):
        """Show how to integrate with your QGAN architecture."""
        print("\n🤖 QGAN INTEGRATION DEMO")
        print("=" * 50)
        
        print("""
Integration with your QGAN architecture:

1. 📋 Circuit Visualization:
   ```python
   from utils.qgan_visualizer import QuantumGANVisualizer
   
   visualizer = QuantumGANVisualizer(your_qgan_model)
   
   # Show generator circuit
   visualizer.print_circuit_diagram(
       qgan.generator.circuit, 
       show_parameters=True, 
       show_values=True
   )
   
   # Show discriminator circuit  
   visualizer.print_circuit_diagram(
       qgan.discriminator.circuit,
       show_parameters=True,
       show_values=True
   )
   ```

2. 🏔️  State Visualization During Training:
   ```python
   # After each epoch or batch
   z_sample = tf.random.normal([1, latent_dim])
   generated_samples = generator.generate(z_sample)
   
   # Get quantum state (you'd need to modify your generator)
   quantum_state = generator.get_quantum_state(z_sample)
   
   # Visualize
   visualizer.visualize_state_3d_mountain(
       quantum_state, 
       mode=0, 
       title_suffix=f" - Epoch {epoch}"
   )
   ```

3. 📊 Training Dashboard:
   ```python
   # Collect metrics during training
   training_history = {
       'g_loss': generator_losses,
       'd_loss': discriminator_losses,  
       'w_distance': wasserstein_distances,
       'gradient_penalty': gradient_penalties
   }
   
   # Visualize progress
   visualizer.visualize_training_progress(training_history)
   ```

4. 🎛️  Parameter Evolution Tracking:
   ```python
   # Save parameters at each epoch
   parameter_snapshots = []
   for epoch in range(n_epochs):
       # ... training step ...
       
       # Save current parameters
       current_params = [var.numpy() for var in generator.trainable_variables]
       parameter_snapshots.append(current_params)
   
   # Visualize evolution
   visualizer.visualize_parameter_evolution(
       parameter_snapshots, 
       parameter_names=[var.name for var in generator.trainable_variables]
   )
   ```
        """)
    
    def run_complete_demo(self):
        """Run the complete visualization demonstration."""
        print("🎨 COMPLETE QGAN VISUALIZATION DEMO")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Circuit visualization
        self.demo_circuit_visualization()
        
        # 2. State visualization
        self.demo_state_visualization()
        
        # 3. Training visualization
        self.demo_training_visualization()
        
        # 4. Integration guide
        self.demo_qgan_integration()
        
        print("\n🎉 COMPLETE DEMO FINISHED!")
        print("=" * 60)
        print("Available visualization capabilities:")
        print("  🔬 Circuit diagrams with parameter details")
        print("  🏔️  3D Wigner function 'mountain maps'") 
        print("  🗻 Multi-mode state visualization")
        print("  📊 Fock state probability distributions")
        print("  📈 Training progress dashboards")
        print("  🎛️  Parameter evolution tracking")
        print("  🤖 Easy integration with your QGAN architecture")
        print("\nReady to enhance your quantum machine learning research! 🚀")


# Example usage
if __name__ == "__main__":
    demo = QGANVisualizationDemo()
    demo.run_complete_demo()
