# QNNCV - Continuous Variable Quantum Neural Networks

Implementation of CV-QNNs using Strawberry Fields for learning probability distributions with quantum circuits.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train Killoran CV-QNN (recommended)
python qnncv.py killoran --n-peaks 3 --epochs 300

# Or use direct entry points
python train_killoran_qgan.py --n-peaks 4 --epochs 300 --n-layers 8
```

## Training Modes

| Mode | Entry Point | Description |
|------|-------------|-------------|
| Killoran | `qnncv.py killoran` | CV-QNN with Kerr gates for N-modal distributions |
| Distribution | `qnncv.py distribution` | Hermite transform for Gaussian targets |
| Sample | `qnncv.py sample` | Sample-based QGAN (legacy) |

## Key Results: Single-Qumode Expressivity

A single qumode CV-QNN with Kerr gates can learn multi-modal distributions:

| N-peaks | Layers | Cutoff | Best W₁ | Success |
|---------|--------|--------|---------|---------|
| 1       | 6      | 8      | 0.144   | ✅ |
| 2       | 6      | 7      | 0.019   | ✅ |
| 3       | 7      | 9      | 0.064   | ✅ |
| 4       | 8      | 8      | 0.046   | ✅ |
| 5       | 10     | 10     | 0.092   | ✅ |

**Critical finding:** Kerr gate is essential. Without it, the circuit can only produce Gaussian (single-peak) outputs.

## Architecture

Based on [Killoran et al. (2018)](https://arxiv.org/abs/1806.06871):

```
Layer L := Φ ◦ D ◦ U₂ ◦ S ◦ U₁

┌────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐   ┌──────┐
│Rotate  │ → │ Squeeze │ → │Rotate  │ → │ Displace │ → │ Kerr │
│  R₁    │   │  S(r,φ) │   │  R₂    │   │  D(r,φ)  │   │ K(κ) │
└────────┘   └─────────┘   └────────┘   └──────────┘   └──────┘
                                                          ↑
                                                   NON-GAUSSIAN
```

- **R₁, R₂**: Rotation gates (interferometer for single mode)
- **S**: Squeeze gate (state compression)
- **D**: Displacement gate (state translation)
- **K**: Kerr gate (nonlinear activation) — **Essential for multi-modal**

## Project Structure

```
QNNCV/
├── qnncv.py                  # Unified entry point
├── train_killoran_qgan.py    # Direct Killoran training
├── src/
│   ├── models/
│   │   ├── generators/       # Quantum generators
│   │   └── discriminators/   # Classical discriminators
│   ├── training/             # Training loops
│   └── utils/                # Utilities & monitoring
├── docs/                     # Documentation
└── logs/                     # Training outputs
```

## Command Reference

### Killoran Mode (Recommended)
```bash
# Basic 2-modal
python qnncv.py killoran --n-peaks 2 --epochs 300

# 4-modal with deeper circuit
python qnncv.py killoran --n-peaks 4 --n-layers 8 --cutoff-dim 10 --epochs 400

# Ablation: without Kerr gate (will fail on multi-modal)
python qnncv.py killoran --n-peaks 3 --no-kerr --epochs 200

# Custom peak placement
python qnncv.py killoran --n-peaks 3 --x-min -3 --x-max 3 --peak-std 0.4
```

### Distribution Mode
```bash
# Learn Gaussian N(2.0, 0.5²)
python qnncv.py distribution --target-mean 2.0 --target-std 0.5 --epochs 200
```

## Output Files

Training creates `./logs/<experiment_name>/`:
- `final_comparison.png` - Distribution comparison
- `training_dashboard.png` - Full metrics dashboard
- `weight_evolution.png` - Parameter evolution
- `wigner_epoch_*.png` - Quantum state snapshots
- `history.json` - All metrics
- `best_weights.npy` - Best checkpoint

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [Key Learnings](docs/KEY_LEARNINGS.md)
- [Training Analysis](docs/QGAN_TRAINING_ANALYSIS_20260123.md)

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- Strawberry Fields
- NumPy, SciPy, Matplotlib

## Citation

```bibtex
@article{killoran2019continuous,
  title={Continuous-variable quantum neural networks},
  author={Killoran, Nathan and Bromley, Thomas R and Arrazola, Juan Miguel and 
          Schuld, Maria and Quesada, Nicol{\'a}s and Lloyd, Seth},
  journal={Physical Review Research},
  year={2019}
}
```

## License

MIT
