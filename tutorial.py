# Cell 1: Install Dependencies  
!pip install pennylane strawberryfields tensorflow  

# Cell 2: Imports  
import tensorflow as tf  
from main_qgan import QGAN  
from NN.quantum_continuous_generator import QuantumContinuousGenerator  
from NN.classical_discriminator import ClassicalDiscriminator  
from utils import load_qm9_data, plot_results  

# Cell 3: Initialize Hybrid QGAN (CV Generator + Classical Discriminator)  
cv_generator = QuantumContinuousGenerator(n_qumodes=30)  
classical_discriminator = ClassicalDiscriminator(input_dim=30)  
qgan = QGAN(cv_generator, classical_discriminator)  

# Cell 4: Load Data and Train  
data = load_qm9_data()  
qgan.train(data, epochs=100)  

# Cell 5: Benchmarking  
# Compare loss curves and generated molecules  
plot_results(qgan.loss_history, qgan.generated_samples)  

# Cell 6: Quantum vs. Classical Comparison  
# (Repeat training with ClassicalGenerator + ClassicalDiscriminator)  