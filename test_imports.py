#!/usr/bin/env python3
"""Exit non-zero when a runtime dependency or local module cannot be imported."""

import sys


def _check(label, importer):
    try:
        importer()
    except Exception as exc:
        print(f"FAIL {label}: {exc}")
        return 1
    print(f"ok   {label}")
    return 0


def main():
    failures = 0
    failures += _check("numpy", lambda: __import__("numpy"))
    failures += _check("trimesh", lambda: __import__("trimesh"))
    failures += _check("manifold3d", lambda: __import__("manifold3d"))
    failures += _check("PyQt5", lambda: __import__("PyQt5"))
    failures += _check("pyvista", lambda: __import__("pyvista"))
    failures += _check("pyvistaqt", lambda: __import__("pyvistaqt"))
    failures += _check("core.scene", lambda: __import__("core.scene", fromlist=["Scene"]))
    failures += _check(
        "classical generator",
        lambda: __import__("core.generators.classical", fromlist=["generate_classical_building"]),
    )
    failures += _check("gui.main_window", lambda: __import__("gui.main_window", fromlist=["MainWindow"]))
    if failures:
        print(f"\n{failures} import check(s) failed.")
        return 1
    print("\nImport checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
