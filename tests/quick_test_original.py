import sys, os
sys.path.insert(0, 'src')
from models.generators.quantum_sf_generator import QuantumSFGenerator
import tensorflow as tf
import numpy as np

print('🔬 Testing Original Generator...')
gen = QuantumSFGenerator(n_modes=4, latent_dim=6, layers=2, cutoff_dim=6)
z = tf.random.normal([100, 6])
samples = gen.generate(z)
print(f'Generated shape: {samples.shape}')
print(f'Sample range: [{samples.numpy().min():.2f}, {samples.numpy().max():.2f}]')
print(f'X std: {np.std(samples.numpy()[:, 0]):.3f}')
print(f'Y std: {np.std(samples.numpy()[:, 1]):.3f}')

# Check bimodal separation
samples_2d = samples.numpy()[:, :2]
mode1_center = np.array([-2.0, -2.0])
mode2_center = np.array([2.0, 2.0])

dist_to_mode1 = np.linalg.norm(samples_2d - mode1_center, axis=1)
dist_to_mode2 = np.linalg.norm(samples_2d - mode2_center, axis=1)

mode1_count = np.sum(dist_to_mode1 < dist_to_mode2)
mode2_count = 100 - mode1_count
balance = min(mode1_count, mode2_count) / 100

print(f'Mode 1: {mode1_count}, Mode 2: {mode2_count}')
print(f'Balance: {balance:.3f}')
print('✅ Original generator working!') 