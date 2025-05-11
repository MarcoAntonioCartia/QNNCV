# Quantum Generative Adversarial Networks (QGANs) for Chemical Applications

A modular framework for benchmarking **classical**, **discrete-variable quantum (qubit)**, and **continuous-variable quantum (qumode)** GANs. Designed for molecular generation using the QM9 dataset, with industrial validation via RDKit.

---

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training a QGAN](#training-a-qgan)
  - [Validating Molecules](#validating-generated-molecules)
  - [Running Tests](#running-tests)
  - [Jupyter Notebook Tutorial](#jupyter-notebook-tutorial)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Industrial Validation](#industrial-validation)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features
- **Hybrid Architectures**: Mix classical/quantum generators (G) and discriminators (D).
- **Quantum Backends**:
  - **PennyLane** for qubit-based models (discrete-variable).
  - **Strawberry Fields** for photonic/qumode models (continuous-variable).
- **Industrial Relevance**: Generate molecular descriptors validated via RDKit.
- **Benchmarking Tools**: Compare loss curves, sample quality, and hardware efficiency.

---

## Installation

### Dependencies
- Python 3.8+
- TensorFlow 2.10+
- PennyLane
- Strawberry Fields
- RDKit

### Steps

```
git clone https://github.com/yourusername/QGAN_Project.git
cd QGAN_Project
pip install -r requirements.txt
```

---

## Usage

### Training a QGAN

```
python main_qgan.py
```

**Default**: Classical G + Classical D on mock QM9 data.

**Custom Setup**: Edit `config.yaml`:

```
components:
  generator: "quantum_continuous"  # Options: classical, quantum_discrete, quantum_continuous
  discriminator: "quantum_discrete"

training:
  epochs: 200
  batch_size: 64
```

---

### Validating Generated Molecules

```
python pharma_validation.py
```

**Input**: Generated descriptors in `/data/generated_samples.npy`.

**Output**:
- Validity rate (chemically plausible molecules).
- Property distributions (LogP, molecular weight).
- Visualizations of top molecules.

---

### Running Tests

Test all components:

```
python -m pytest tests/
```

Test specific hybrid model (CV-G + Classical-D):

```
python tests/test_hybrid_qgan.py
```

---

### Jupyter Notebook Tutorial

```
jupyter notebook tutorial.ipynb
```

Covers:
- Hybrid QGAN training (CV quantum G + classical D).
- Loss curve analysis.
- Quantum circuit visualization.

---

## Project Structure

```
/QGAN_Project  
├── data/                   # QM9 dataset and generated samples  
├── NN/                     # All generators/discriminators  
│   ├── classical_generator.py  
│   ├── quantum_discrete_generator.py  
│   ├── quantum_continuous_generator.py  
│   ├── classical_discriminator.py  
│   ├── quantum_discrete_discriminator.py  
│   └── quantum_continuous_discriminator.py  
├── tests/                  # Unit tests  
│   ├── test_hybrid_qgan.py  
│   └── test_gan_classical.py  
├── main_qgan.py            # Central training script  
├── utils.py                # Data loading, plotting, metrics  
├── pharma_validation.py    # RDKit-based validation  
├── config.yaml             # Hyperparameters and model settings  
├── requirements.txt        # Dependency list  
└── tutorial.ipynb          # Step-by-step guide  
```

---

## Configuration

Modify `config.yaml` to customize:

```
training:
  epochs: 100               # Number of training iterations
  batch_size: 32            # Samples per batch
  latent_dim: 10            # Noise vector dimension

components:
  generator:
    type: "quantum_continuous"  # Options: classical, quantum_discrete, quantum_continuous
    n_qumodes: 30               # For CV models
    n_layers: 3                 # For qubit models
  discriminator:
    type: "classical"           # Options: classical, quantum_discrete, quantum_continuous
    hidden_units: 64            # For classical D

optimizer:
  learning_rate: 0.001          # Adam optimizer LR
  beta_1: 0.9                   # Adam momentum
```

---

## Industrial Validation

- **Validity Check**: Ensure generated molecules are chemically possible.
- **Property Prediction**: Compute LogP, molecular weight, etc.
- **Novelty Score**: Compare against QM9 training set using Tanimoto similarity.

**Example Output from `pharma_validation.py`:**

```
Valid molecules: 8/10  
Average LogP: 1.2 ± 0.3  
Synthetic accessibility score: 3.4 (easy to synthesize)
```

---

## Contributing

**Add New Components**:
- Add quantum generators/discriminators to `/NN`.
- Include unit tests in `/tests`.

**Improve Validation**:
- Integrate advanced property predictors (e.g., DFT calculators).

**Hardware Support**:
- Add Xanadu Cloud integration for photonic hardware.

---

## License

MIT License. See `LICENSE` file for details.
