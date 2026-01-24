# Training - src/training/

Training loops and utilities for CV-QNN GANs.

## Active Files

| File | Class | Description |
|------|-------|-------------|
| `killoran_trainer.py` | `KilloranQGANTrainer`, `KilloranTrainerConfig`, `WeightTracker` | Full-featured trainer with monitoring, Wigner snapshots, weight evolution tracking |
| `distribution_trainer.py` | `DistributionQGANTrainer` | Training loop for distribution-based QGAN |
| `trainer.py` | `QGANTrainer`, `QGANConfig` | Sample-based QGAN trainer |
| `data_generators.py` | Various | Data loading and target distribution utilities |

## Key Class: KilloranQGANTrainer

The most complete trainer with all monitoring features.

### Configuration

```python
from src.training.killoran_trainer import KilloranTrainerConfig

config = KilloranTrainerConfig(
    # Target distribution
    n_peaks=3,
    x_min=-2.0,
    x_max=2.0,
    peak_std=0.3,
    
    # Architecture
    n_layers=7,
    cutoff_dim=10,
    use_kerr=True,
    
    # Training
    batch_size=16,
    g_lr=0.005,
    d_lr=0.001,
    
    # Monitoring
    save_wigner=True,
    wigner_epochs=[1, 10, 50, 100, 200, 300],
    smooth_window=10
)
```

### Training

```python
from src.training.killoran_trainer import KilloranQGANTrainer

trainer = KilloranQGANTrainer(generator, discriminator, config)
summary = trainer.train(epochs=300, log_dir='./logs/experiment')
```

### Outputs

Training produces:
- `final_comparison.png` - Target vs generated distribution
- `training_dashboard.png` - 6-panel metrics view
- `weight_evolution.png` - Parameter tracking over epochs
- `wigner_epoch_*.png` - Quantum state snapshots
- `history.json` - All metrics (serializable)
- `best_weights.npy` - Best checkpoint

## Weight Tracker

Monitors quantum gate parameters during training:

```python
tracker = WeightTracker(generator)

for epoch in range(epochs):
    # ... training step ...
    tracker.record(epoch)

tracker.plot_weight_evolution('weights.png')
```

Tracks:
- Overall weight statistics (mean, std, L2 norm)
- Kerr parameters (κ) per layer
- Squeeze magnitudes (r) per layer
- Displacement magnitudes per layer

## Target Distribution Generation

```python
from src.models.generators.killoran_cvqnn import n_modal_gaussian

# Create 4-peak target
prob, peak_positions = n_modal_gaussian(
    xvec=np.linspace(-5, 5, 100),
    n_peaks=4,
    x_min=-2.0,
    x_max=2.0,
    std=0.3
)
# peak_positions: [-2.0, -0.67, 0.67, 2.0]
```

## Legacy Files (in src/models/legacy/)

- `qgan_sf_trainer.py` - Superseded by trainer.py
- `quantum_gan_trainer.py` - Very old trainer
