# QNNCV Setup Scripts

This directory contains setup scripts for different environments and use cases to handle NumPy 2.0+ and SciPy 1.15+ compatibility issues.

## Quick Start

### For Google Colab (Recommended)
```python
# Run in Colab cell
!python setup/setup_colab_modern.py
```

### For Local Development
```bash
# Run from QNNCV root directory
python setup/setup_local.py
```

### For Conda Environment Management
```bash
# Create complete conda environment
python setup/setup_environment.py
```

## Available Scripts

### `setup_colab_modern.py` - **RECOMMENDED FOR COLAB**
- **Purpose**: Modern Colab setup with current package versions
- **Use Case**: Google Colab notebooks with GPU acceleration
- **Features**: 
  - Works with NumPy 2.0.2, SciPy 1.15.3, TensorFlow 2.18.0
  - No package downgrades or runtime restarts required
  - Comprehensive compatibility patches
  - GPU configuration and optimization
  - Robust error handling
- **Usage**: `!python setup/setup_colab_modern.py`

### `setup_local.py` - **RECOMMENDED FOR LOCAL DEV**
- **Purpose**: Local development environment matching Colab
- **Use Case**: Local development with pip-based installation
- **Features**:
  - Installs exact package versions from Colab
  - Tests all QNNCV components locally
  - CPU/GPU support with automatic detection
  - Comprehensive validation and testing
- **Usage**: `python setup/setup_local.py`

### `setup_environment.py` - **CONDA ENVIRONMENT MANAGEMENT**
- **Purpose**: Complete conda environment creation and management
- **Use Case**: Full conda-based development environment
- **Features**:
  - Creates dedicated 'qnncv' conda environment with Python 3.11
  - Installs packages via conda (preferred) and pip (fallback)
  - Interactive environment management (remove/recreate existing)
  - Comprehensive package testing and validation
  - Legacy function support for backward compatibility
- **Usage**: `python setup/setup_environment.py`
- **Creates**: Conda environment named 'qnncv'

### `setup.py` - **MAIN PROJECT SETUP ORCHESTRATOR**
- **Purpose**: Complete project setup and validation
- **Use Case**: Initial project setup with full environment creation
- **Features**:
  - Orchestrates complete project setup workflow
  - Verifies project structure and required files
  - Calls `setup_environment.py` for conda environment creation
  - Runs comprehensive installation verification
  - Executes project tests to validate functionality
  - Provides detailed usage instructions and project guidance
- **Usage**: `python setup/setup.py`
- **Dependencies**: Requires `setup_environment.py` and project structure

### `environment.yml` - **CONDA ENVIRONMENT SPECIFICATION**
- **Purpose**: Declarative conda environment definition
- **Use Case**: Reproducible conda environment creation
- **Usage**: `conda env create -f setup/environment.yml`

## Setup Script Comparison

| Script | Environment | Package Manager | Use Case | Complexity |
|--------|-------------|-----------------|----------|------------|
| `setup_colab_modern.py` | Google Colab | pip | Cloud notebooks | Simple |
| `setup_local.py` | Local | pip | Local development | Simple |
| `setup_environment.py` | Local | conda + pip | Conda environments | Medium |
| `setup.py` | Local | conda (via setup_environment) | Full project setup | Complex |
| `environment.yml` | Local | conda | Declarative setup | Simple |

## Compatibility Issues Solved

### NumPy 2.0+ Compatibility
- **Problem**: NumPy 2.0 removed deprecated aliases like `np.bool`, `np.int`
- **Solution**: Compatibility patches restore these aliases automatically
- **Implementation**: `src/utils/compatibility.py`

### SciPy 1.14+ Compatibility  
- **Problem**: SciPy 1.14+ removed `scipy.integrate.simps` function
- **Solution**: Compatibility patches add `simps` as alias to `simpson`
- **Impact**: Strawberry Fields works with modern SciPy versions

### TensorFlow 2.18+ Integration
- **Features**: GPU configuration, mixed precision handling, memory growth
- **Compatibility**: Works with both local and Colab environments

## Recommended Workflows

### 1. Quick Colab Development
```python
# In Google Colab
!git clone https://github.com/your-username/QNNCV.git
%cd QNNCV
!python setup/setup_colab_modern.py

# Start working immediately
import sys
sys.path.append('/content/QNNCV/src')
import utils  # Auto-applies compatibility patches
```

### 2. Local Development (pip-based)
```bash
# Local machine
git clone https://github.com/your-username/QNNCV.git
cd QNNCV
python setup/setup_local.py

# Start development
python -c "import utils; print('Setup working!')"
```

### 3. Conda Environment Development
```bash
# Create conda environment
cd QNNCV
python setup/setup_environment.py

# Activate and use
conda activate qnncv
python -c "import utils; print('Conda setup working!')"
```

### 4. Complete Project Setup
```bash
# Full project initialization
cd QNNCV
python setup/setup.py

# Follow provided instructions
conda activate qnncv
python test_basic.py
```

## Testing Your Setup

### Quick Test (All Environments)
```python
# Import utils first to apply compatibility patches
import utils

from models.generators.quantum_sf_generator import QuantumSFGenerator
gen = QuantumSFGenerator(n_modes=2, latent_dim=2)
print("Setup working!")
```

### Environment-Specific Tests

**Colab:**
```python
!python setup/setup_colab_modern.py
# Test imports as shown above
```

**Local pip:**
```bash
python setup/setup_local.py
python -c "import utils; from models.generators.quantum_sf_generator import QuantumSFGenerator; print('Local setup working!')"
```

**Conda:**
```bash
python setup/setup_environment.py
conda activate qnncv
python -c "import utils; from models.generators.quantum_sf_generator import QuantumSFGenerator; print('Conda setup working!')"
```

## Package Versions

All setup scripts target these versions to match current Google Colab:

| Package | Version | Notes |
|---------|---------|-------|
| numpy | 2.0.2 | With compatibility patches |
| scipy | 1.15.3 | With simps function restored |
| tensorflow | 2.18.0 | GPU optimized |
| matplotlib | 3.10.0 | Latest stable |
| pandas | 2.2.2 | Latest stable |
| scikit-learn | 1.6.1 | Latest stable |
| strawberryfields | Latest | Quantum computing |
| pyyaml | Latest | Configuration files |
| tqdm | Latest | Progress bars |
| seaborn | Latest | Visualization |
| psutil | Latest | System monitoring |

## Troubleshooting

### Common Issues

**"Module not found" errors:**
- Ensure you're running from the QNNCV root directory
- Import `utils` first to set up paths and apply patches
- Check that `src/` directory exists and contains the modules

**NumPy/SciPy compatibility errors:**
- All setup scripts automatically apply compatibility patches
- If issues persist: `import utils` (this auto-applies patches)
- For manual application: `from utils.compatibility import apply_all_compatibility_patches; apply_all_compatibility_patches()`

**Environment conflicts:**
- **Conda users**: Use `setup_environment.py` or `setup.py` for proper conda environment isolation
- **pip users**: Use `setup_local.py` for system/venv installation
- **Colab users**: Use `setup_colab_modern.py` for cloud environment

**GPU not detected in Colab:**
- Switch to GPU runtime: Runtime → Change runtime type → GPU
- Re-run setup script after switching
- Verify with: `tf.config.list_physical_devices('GPU')`

**Training performance issues:**
- Start with minimal parameters: `n_modes=1`, `layers=1`, `cutoff_dim=4`
- Use small batch sizes: `batch_size=4`
- Test with minimal data: `data[:50]`
- Quantum simulations are computationally intensive

## Performance Tips

### Parameter Scaling Guide
```python
# Start here (fast)
generator = QuantumSFGenerator(n_modes=1, latent_dim=1, layers=1, cutoff_dim=4)

# Scale gradually
generator = QuantumSFGenerator(n_modes=2, latent_dim=2, layers=1, cutoff_dim=6)

# Full complexity (slow)
generator = QuantumSFGenerator(n_modes=2, latent_dim=2, layers=2, cutoff_dim=8)
```

### Environment-Specific Optimization

**Colab:**
- Use GPU runtime for TensorFlow operations
- Quantum operations will run on CPU (normal)
- Monitor memory usage in Colab interface

**Local conda:**
- Conda manages package conflicts better
- Use `conda activate qnncv` for isolated environment
- Better for long-term development

**Local pip:**
- Faster setup for quick testing
- May have package conflicts in complex environments
- Good for CI/CD and automated testing

## Directory Structure

```
setup/
├── setup_colab_modern.py    # Modern Colab setup (ACTIVE)
├── setup_local.py           # Local pip setup (ACTIVE)
├── setup_environment.py     # Conda environment manager (ACTIVE)
├── setup.py                 # Main project orchestrator (ACTIVE)
├── environment.yml          # Conda environment spec (ACTIVE)
└── README.md               # This file
```

## Contributing

When adding new setup features:

1. **Test across environments**: Colab, local pip, conda
2. **Update compatibility module** if needed (`src/utils/compatibility.py`)
3. **Maintain backward compatibility** for existing workflows
4. **Update this README** with new features and use cases
5. **Follow the established patterns** for each setup script type

## Additional Resources

- **Main README**: `../README.md` - Project overview and installation
- **Tutorials**: `../tutorials/` - Working examples and notebooks
- **Compatibility Module**: `../src/utils/compatibility.py` - Technical implementation
- **Source Code**: `../src/` - Main codebase

## License

Same as main project - see `../LICENSE`
