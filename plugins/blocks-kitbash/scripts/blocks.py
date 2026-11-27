#!/usr/bin/env python3
"""Portable, shell-free launcher for the Blocks agent interface."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def runtime():
    explicit = os.environ.get("BLOCKS_ROOT")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend(Path(__file__).resolve().parents)
    for root in candidates:
        if (root / "blockn.py").is_file() and (root / "core/architecture").is_dir():
            candidates = ([os.environ["BLOCKS_PYTHON"]] if os.environ.get("BLOCKS_PYTHON") else
                          [str(p) for p in (root / ".venv/bin/python", root / ".venv/Scripts/python.exe") if p.is_file()] + [sys.executable])
            python = candidates[-1]
            for candidate in candidates:
                try:
                    ready = subprocess.run([candidate, "-c", "import importlib.util,sys; sys.exit(any(importlib.util.find_spec(n) is None for n in ('numpy','trimesh')))"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if ready.returncode == 0:
                        python = candidate
                        break
                except OSError:
                    continue
            return [python, "-m", "blockn"], root
    executable = shutil.which("blockn")
    if executable:
        return [executable], None
    if importlib.util.find_spec("blockn"):
        return [sys.executable, "-m", "blockn"], None
    raise RuntimeError("Blocks runtime not found. Install the source bundle with python -m pip install -e . or set BLOCKS_ROOT to its folder.")


def main():
    try:
        command, root = runtime()
        args = sys.argv[1:]
        if args == ["doctor"]:
            check = subprocess.run(command + ["catalog", "--json"], cwd=root, capture_output=True, text=True)
            if check.returncode:
                print(json.dumps({"status": "error", "command": command, "error": check.stderr.strip() or check.stdout.strip(), "install": "python -m pip install -e ."}))
                return check.returncode
            catalog = json.loads(check.stdout)
            print(json.dumps({"status": "pass", "runtime": command, "root": str(root) if root else None,
                              "headless": True, "catalog": catalog, "formats": ["glb", "obj", "stl", "ply"]}))
            return 0
        # Preserve the caller's cwd: relative recipes and outputs belong to the
        # request, while PYTHONPATH locates an unpacked source installation.
        environment = os.environ.copy()
        if root:
            environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
        return subprocess.run(command + args, env=environment).returncode
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
