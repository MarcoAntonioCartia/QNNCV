# QNNCV Setup Scripts

This directory contains setup scripts for different environments to handle NumPy 2.0+ and SciPy 1.15+ compatibility issues.

## 🚀 Quick Start

### For Local Development
```bash
# Run from QNNCV root directory
python setup/setup_local.py
```

### For Google Colab
```python
# Run in Colab cell
!python setup/setup_colab_modern.py
```

## 📁 Available Scripts

### `setup_colab_modern.py` ⭐ **RECOMMENDED FOR COLAB**
- **Purpose**: Modern Colab setup with current package versions
- **Features**: 
  - Works with NumPy 2.0.2, SciPy 1.15.3, TensorFlow 2.18.0
  - No package downgrades or runtime restarts required
  - Comprehensive compatibility patches
  - GPU configuration and optimization
  - Robust error handling
- **Usage**: `!python setup/setup_colab_modern.py`

### `setup_local.py` ⭐ **RECOMMENDED FOR LOCAL DEV**
- **Purpose**: Local development environment matching Colab
- **Features**:
  - Installs exact package versions from Colab
  - Tests all QNNCV components locally
  - CPU fallback for development
  - Comprehensive validation
- **Usage**: `python setup/setup_local.py`

### `setup_colab.py` (Legacy)
- **Purpose**: Original Colab setup with strict version control
- **Issues**: Complex NumPy version locking, may require runtime restarts
- **Status**: Superseded by `setup_colab_modern.py`

### `setup_colab_conda_style.py` (Legacy)
- **Purpose**: Conda-style package installation for Colab
- **Issues**: Still has NumPy/SciPy compatibility problems
- **Status**: Superseded by `setup_colab_modern.py`

## 🔧 Compatibility Issues Solved

### NumPy 2.0+ Compatibility
- **Problem**: NumPy 2.0 removed deprecated aliases like `np.bool`, `np.int`
- **Solution**: Compatibility patches restore these aliases automatically
- **Files**: `src/utils/compatibility.py`

### SciPy 1.14+ Compatibility  
- **Problem**: SciPy 1.14+ removed `scipy.integrate.simps` function
- **Solution**: Compatibility patches add `simps` as alias to `simpson`
- **Impact**: Strawberry Fields works with modern SciPy versions

### TensorFlow 2.18+ Integration
- **Features**: GPU configuration, mixed precision, memory growth
- **Compatibility**: Works with both local and Colab environments

## 🎯 Recommended Workflow

### 1. Local Development First
```bash
# Set up local environment
python setup/setup_local.py

# Develop and test locally
jupyter notebook tutorials/minimal_sf_qgan.ipynb

# Commit working changes
git add .
git commit -m "Working local setup"
git push
```

### 2. Deploy to Colab
```python
# In Colab: Clone/update repository
!git clone https://github.com/your-username/QNNCV.git
%cd QNNCV

# Run modern setup
!python setup/setup_colab_modern.py

# Start training with GPU acceleration
from models.generators.quantum_sf_generator import QuantumSFGenerator
# ... your code
```

## 🧪 Testing Your Setup

### Quick Test (Local)
```python
from utils.compatibility import apply_all_compatibility_patches
apply_all_compatibility_patches()

from models.generators.quantum_sf_generator import QuantumSFGenerator
gen = QuantumSFGenerator(n_modes=2, latent_dim=2)
print("✓ Setup working!")
```

### Quick Test (Colab)
```python
!python setup/setup_colab_modern.py

# If successful, try:
from models.generators.quantum_sf_generator import QuantumSFGenerator
gen = QuantumSFGenerator(n_modes=2, latent_dim=2)
print("✓ Colab setup working!")
```

## 📦 Package Versions

The setup scripts install these versions to match current Google Colab:

| Package | Version | Notes |
|---------|---------|-------|
| numpy | 2.0.2 | With compatibility patches |
| scipy | 1.15.3 | With simps function restored |
| tensorflow | 2.18.0 | GPU optimized |
| matplotlib | 3.10.0 | Latest stable |
| pandas | 2.2.2 | Latest stable |
| scikit-learn | 1.6.1 | Latest stable |
| strawberryfields | Latest | Quantum computing |

## 🐛 Troubleshooting

### "Module not found" errors
- Make sure you're running from the QNNCV root directory
- Check that `src/` directory exists and contains the modules

### NumPy compatibility errors
- The setup scripts automatically apply compatibility patches
- If issues persist, manually run: `from utils.compatibility import apply_all_compatibility_patches; apply_all_compatibility_patches()`

### SciPy simps errors
- Compatibility patches automatically add `simps` function
- Strawberry Fields should work after patches are applied

### GPU not detected in Colab
- Switch to GPU runtime: Runtime → Change runtime type → GPU
- Re-run the setup script after switching

### Package installation failures
- Check internet connection
- Try running setup script again
- For local development, ensure you have sufficient disk space

## 🔄 Migration Guide

### From Old Setup Scripts
If you were using the old setup scripts:

1. **Stop using**: `setup_colab.py`, `setup_colab_conda_style.py`
2. **Start using**: `setup_colab_modern.py` for Colab
3. **For local dev**: Use `setup_local.py`
4. **Benefits**: No runtime restarts, better compatibility, GPU optimization

### From Manual Setup
If you were installing packages manually:

1. **Replace manual pip installs** with setup scripts
2. **Remove version constraints** - scripts handle compatibility
3. **Use compatibility module** instead of manual patches

## 📚 Additional Resources

- **Main README**: `../README.md` - Project overview
- **Tutorials**: `../tutorials/` - Working examples
- **Compatibility Module**: `../src/utils/compatibility.py` - Technical details
- **Documentation**: `../docs/` - Detailed documentation

## 🤝 Contributing

When adding new setup features:

1. **Test locally first** with `setup_local.py`
2. **Validate in Colab** with `setup_colab_modern.py`  
3. **Update compatibility module** if needed
4. **Update this README** with new features

## 📄 License

Same as main project - see `../LICENSE`
