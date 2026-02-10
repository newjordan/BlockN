#!/usr/bin/env python3
"""
Quick test to check if all required dependencies are available
"""

print("Testing imports...")

try:
    import sys
    print(f"✓ Python {sys.version}")
except Exception as e:
    print(f"✗ Python: {e}")

try:
    from PyQt5 import QtCore
    print(f"✓ PyQt5 {QtCore.PYQT_VERSION_STR}")
except Exception as e:
    print(f"✗ PyQt5: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

try:
    import trimesh
    print(f"✓ Trimesh {trimesh.__version__}")
except Exception as e:
    print(f"✗ Trimesh: {e}")

try:
    import pyvista as pv
    print(f"✓ PyVista {pv.__version__}")
except Exception as e:
    print(f"✗ PyVista: {e}")

try:
    from pyvistaqt import QtInteractor
    print("✓ PyVistaQt")
except Exception as e:
    print(f"✗ PyVistaQt: {e}")

# Test core imports
try:
    from core.scene import Scene, Block
    print("✓ Core Scene")
except Exception as e:
    print(f"✗ Core Scene: {e}")

try:
    from core.generators.classical import generate_classical_building
    print("✓ Classical Generator")
except Exception as e:
    print(f"✗ Classical Generator: {e}")

try:
    from gui.main_window import MainWindow
    print("✓ Main Window")
except Exception as e:
    print(f"✗ Main Window: {e}")

print("\nImport test complete!")
