"""
Simple explanation of why quantum measurements don't create bimodal distributions
"""

import numpy as np
import matplotlib.pyplot as plt

def explain_quantum_modes_vs_bimodal():
    """Visual explanation of the misconception about quantum modes and bimodal distributions."""
    
    print("="*60)
    print("QUANTUM MODES vs BIMODAL DISTRIBUTIONS")
    print("="*60)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Understanding Quantum Modes vs Bimodal Distributions', fontsize=16)
    
    # 1. What quantum modes actually produce
    print("\n1. WHAT QUANTUM MODES ACTUALLY PRODUCE")
    print("-" * 40)
    
    # Simulate photon number measurements from 4 modes
    n_samples = 1000
    
    # Each mode produces continuous values (photon numbers)
    mode_outputs = []
    for i in range(4):
        # Quantum modes typically have low photon numbers (exponential-like distribution)
        photon_numbers = np.random.exponential(0.5, n_samples)
        mode_outputs.append(photon_numbers)
        
        # Plot distribution
        ax = axes[0, i] if i < 3 else axes[1, 0]
        ax.hist(photon_numbers, bins=30, alpha=0.7, color=f'C{i}', density=True)
        ax.set_title(f'Mode {i} Output')
        ax.set_xlabel('Photon Number')
        ax.set_ylabel('Probability')
        ax.set_xlim(0, 3)
        
        print(f"Mode {i}: Continuous values, mean={np.mean(photon_numbers):.3f}")
    
    mode_outputs = np.array(mode_outputs).T
    
    # 2. Combining modes - Wrong interpretation
    print("\n2. WRONG INTERPRETATION: 'Each mode = coordinate of a blob'")
    print("-" * 40)
    
    ax = axes[1, 1]
    ax.text(0.5, 0.8, "MISCONCEPTION:", ha='center', fontsize=14, weight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.6, "Mode 0 → X of blob 1", ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.5, "Mode 1 → Y of blob 1", ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.4, "Mode 2 → X of blob 2", ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.3, "Mode 3 → Y of blob 2", ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.1, "❌ This doesn't work!", ha='center', fontsize=14, color='red', weight='bold', transform=ax.transAxes)
    ax.axis('off')
    
    # 3. What actually happens
    print("\n3. WHAT ACTUALLY HAPPENS")
    print("-" * 40)
    
    ax = axes[1, 2]
    
    # Method 1: Use modes 0,1 for X and 2,3 for Y
    x = mode_outputs[:, 0] + mode_outputs[:, 1]
    y = mode_outputs[:, 2] + mode_outputs[:, 3]
    
    # Scale to reasonable range
    x = x * 2 - 2
    y = y * 2 - 2
    
    ax.scatter(x, y, alpha=0.3, s=10)
    ax.scatter([-2, 2], [-2, 2], c='red', s=200, marker='x', linewidth=3, label='Target Modes')
    ax.set_title('Actual Result: Single Blob!')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend()
    
    print("Result: A single continuous distribution, NOT bimodal!")
    
    plt.tight_layout()
    plt.savefig('quantum_mode_explanation.png', dpi=150)
    plt.show()
    
    # Text explanation
    print("\n4. WHY THIS HAPPENS")
    print("-" * 40)
    print("• Quantum modes output CONTINUOUS values (photon numbers)")
    print("• These are like measuring the 'energy' in each mode")
    print("• Combining continuous values gives another continuous distribution")
    print("• There's NO mechanism to create discrete clusters")
    
    print("\n5. THE KEY INSIGHT")
    print("-" * 40)
    print("Quantum states are fundamentally CONTINUOUS, not discrete!")
    print("To get bimodal output, you need:")
    print("  1. A CLASSICAL decision: 'Generate at location A or B'")
    print("  2. QUANTUM modulation: Add quantum noise/correlations")
    
    return mode_outputs

def show_the_fix():
    """Show how the fix works."""
    
    print("\n\n" + "="*60)
    print("HOW THE FIX WORKS")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('The Solution: Hybrid Classical-Quantum Approach', fontsize=16)
    
    n_samples = 500
    
    # 1. Classical mode selection
    ax = axes[0]
    latent = np.random.randn(n_samples, 6)  # Random latent vectors
    mode_selector = np.random.randn(6, 1) * 0.1  # Mode selector weights
    mode_scores = latent @ mode_selector  # Linear projection
    mode_scores = mode_scores.squeeze()
    
    ax.hist(mode_scores, bins=30, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.text(-2, 40, 'Mode 1', fontsize=12, weight='bold')
    ax.text(2, 40, 'Mode 2', fontsize=12, weight='bold')
    ax.set_title('Step 1: Classical Mode Selection')
    ax.set_xlabel('Mode Score')
    ax.set_ylabel('Count')
    
    # 2. Mode assignment
    ax = axes[1]
    mode1_mask = mode_scores < 0
    mode2_mask = ~mode1_mask
    
    # Pie chart of mode distribution
    sizes = [np.sum(mode1_mask), np.sum(mode2_mask)]
    ax.pie(sizes, labels=['Mode 1', 'Mode 2'], autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
    ax.set_title('Step 2: Mode Assignment')
    
    # 3. Final generation
    ax = axes[2]
    
    # Generate samples based on mode
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    # Mode 1: centered at (-2, -2)
    x[mode1_mask] = -2 + np.random.normal(0, 0.3, np.sum(mode1_mask))
    y[mode1_mask] = -2 + np.random.normal(0, 0.3, np.sum(mode1_mask))
    
    # Mode 2: centered at (2, 2)
    x[mode2_mask] = 2 + np.random.normal(0, 0.3, np.sum(mode2_mask))
    y[mode2_mask] = 2 + np.random.normal(0, 0.3, np.sum(mode2_mask))
    
    ax.scatter(x[mode1_mask], y[mode1_mask], alpha=0.5, s=20, c='lightblue', label='Mode 1')
    ax.scatter(x[mode2_mask], y[mode2_mask], alpha=0.5, s=20, c='lightgreen', label='Mode 2')
    ax.scatter([-2, 2], [-2, 2], c='red', s=200, marker='x', linewidth=3)
    ax.set_title('Step 3: Bimodal Output!')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('quantum_fix_explanation.png', dpi=150)
    plt.show()
    
    print("\nTHE FIX IN DETAIL:")
    print("1. Latent vector z → Mode selector → Mode score")
    print("2. If score < 0: Use Mode 1 quantum parameters & generate around (-2,-2)")
    print("3. If score ≥ 0: Use Mode 2 quantum parameters & generate around (2,2)")
    print("4. Quantum circuit adds realistic noise/correlations")
    
    print("\nKEY: The bimodality comes from CLASSICAL selection,")
    print("     while quantum adds realistic physical properties!")

def show_n_mode_extension():
    """Show how this extends to N modes."""
    
    print("\n\n" + "="*60)
    print("EXTENDING TO N MODES")
    print("="*60)
    
    # Example with 5 modes
    n_modes = 5
    n_samples = 1000
    
    # Define mode centers (pentagon arrangement)
    angles = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
    mode_centers = np.array([[2*np.cos(a), 2*np.sin(a)] for a in angles])
    
    # Simulate mode selection
    latent = np.random.randn(n_samples, 6)
    mode_selector = np.random.randn(6, n_modes) * 0.1
    mode_logits = latent @ mode_selector
    mode_probs = np.exp(mode_logits) / np.sum(np.exp(mode_logits), axis=1, keepdims=True)
    
    # Sample modes
    modes = np.array([np.random.choice(n_modes, p=p) for p in mode_probs])
    
    # Generate samples
    samples = []
    for i in range(n_samples):
        center = mode_centers[modes[i]]
        sample = center + np.random.normal(0, 0.3, 2)
        samples.append(sample)
    
    samples = np.array(samples)
    
    # Plot
    plt.figure(figsize=(8, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_modes))
    
    for mode in range(n_modes):
        mask = modes == mode
        plt.scatter(samples[mask, 0], samples[mask, 1], 
                   alpha=0.5, s=20, c=[colors[mode]], 
                   label=f'Mode {mode+1}')
    
    # Plot centers
    plt.scatter(mode_centers[:, 0], mode_centers[:, 1], 
               c='black', s=200, marker='x', linewidth=3)
    
    plt.title(f'{n_modes}-Mode Quantum Generation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('n_mode_generation.png', dpi=150)
    plt.show()
    
    print(f"\nFor {n_modes} modes:")
    print("1. Mode selector outputs {n_modes} logits")
    print("2. Softmax gives probability for each mode")
    print("3. Sample from categorical distribution")
    print("4. Generate around selected mode's center")
    print("\nThis creates a true {n_modes}-modal distribution!")

if __name__ == "__main__":
    # Run explanations
    print("Running quantum mode explanation...")
    mode_outputs = explain_quantum_modes_vs_bimodal()
    
    print("\n" + "="*60)
    show_the_fix()
    
    print("\n" + "="*60)
    show_n_mode_extension()
    
    print("\n✅ Explanation complete! Check the generated images:")
    print("   - quantum_mode_explanation.png")
    print("   - quantum_fix_explanation.png")
    print("   - n_mode_generation.png")
