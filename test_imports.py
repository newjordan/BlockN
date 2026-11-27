#!/usr/bin/env python3
"""
Quick test to check if all required dependencies are available
"""

import sys


failures = []


def failed(label, error):
    failures.append(label)
    print(f"✗ {label}: {error}")

print("Testing imports...")

try:
    print(f"✓ Python {sys.version}")
except Exception as e:
    failed("Python", e)

try:
    from PyQt5 import QtCore
    print(f"✓ PyQt5 {QtCore.PYQT_VERSION_STR}")
except Exception as e:
    failed("PyQt5", e)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    failed("NumPy", e)

try:
    import trimesh
    print(f"✓ Trimesh {trimesh.__version__}")
except Exception as e:
    failed("Trimesh", e)

try:
    import pyvista as pv
    print(f"✓ PyVista {pv.__version__}")
except Exception as e:
    failed("PyVista", e)

try:
    from pyvistaqt import QtInteractor
    print("✓ PyVistaQt")
except Exception as e:
    failed("PyVistaQt", e)

# Test core imports
try:
    from core.scene import Scene, Block
    print("✓ Core Scene")
except Exception as e:
    failed("Core Scene", e)

try:
    from core.architecture import BuildRequest, build
    print("✓ BlockN Architecture")
except Exception as e:
    failed("BlockN Architecture", e)

try:
    from core.generators.classical import generate_classical_building
    print("✓ Classical Generator")
except Exception as e:
    failed("Classical Generator", e)

try:
    from gui.main_window import MainWindow
    print("✓ Main Window")
except Exception as e:
    failed("Main Window", e)

if failures:
    print(f"\nImport test failed: {', '.join(failures)}")
    raise SystemExit(1)

print("\nImport test complete: all imports available.")
