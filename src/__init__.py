# QNNCV - Quantum Neural Networks for Continuous Variables
# Main package initialization

__version__ = "1.0.0"
__author__ = "QNNCV Team"

# Auto-apply compatibility patches when importing. Load-bearing: any direct
# `import src.*` relies on this for the scipy simps alias (via utils ->
# scipy_compat) and the TF config; do not remove.
try:
    from .utils import compatibility
    print("QNNCV: Applying compatibility patches...")
    compatibility.apply_all_compatibility_patches()
except ImportError:
    pass  # Will be handled during setup
