"""Shared prologue for scripts/verify_*: repo root on sys.path plus the
scipy.integrate.simps compat alias (Strawberry Fields 0.23.0 imports simps,
removed in SciPy 1.14+). Must be imported BEFORE any `src.*` import —
src.quantum.circuit and src.training.qgan_2d import strawberryfields at
module import time. Mirrors the prologue of train_2d_qgan.py and is kept
independent of src/__init__.py side effects.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
