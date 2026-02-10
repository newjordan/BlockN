import subprocess
import tempfile
import os
import platform
import trimesh
import trimesh.interfaces.blender # Import the blender interface module
import numpy as np # Import numpy
from typing import List, Dict, Optional, Tuple

# Placeholder for actual Blender executable path - needs configuration
# TODO: Make this configurable (e.g., via settings file or UI)
# --- Option A: Assume blender is in PATH (Default) ---
# BLENDER_EXECUTABLE = "blender"
# --- Hardcoded path removed ---

# Define ColorTuple if not imported elsewhere, or import it
ColorTuple = Tuple[int, int, int, int]

# --- Helper function to run Blender scripts ---
def _run_blender_script(blender_executable: str, script_path: str, args: List[str], timeout: int = 120) -> bool:
    """Runs a Blender script via subprocess, returning True on success."""
    if not os.path.exists(script_path):
        print(f"ERROR: Blender script not found at {script_path}")
        return False

    cmd = [
        blender_executable,
        "--background",
        "--python", script_path,
        "--" # Separator for custom arguments
    ]
    cmd.extend(args)

    print(f"Running Blender command: {' '.join(cmd)}")
    try:
        startupinfo = None
        if platform.system() == "Windows":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo, timeout=timeout)
        print("Blender Output:\n", result.stdout)
        if result.stderr:
            print("Blender Errors:\n", result.stderr)
        return True # Success
    except FileNotFoundError:
        print(f"ERROR: Blender executable not found at '{blender_executable}'. Please configure the correct path.")
        # Re-raise a more specific error or return False? Returning False for now.
        # raise FileNotFoundError(f"Blender executable not found at '{blender_executable}'") from None
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Blender script execution failed with return code {e.returncode}.")
        print("Blender Output:\n", e.stdout)
        print("Blender Errors:\n", e.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"ERROR: Blender process timed out after {timeout} seconds.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while running Blender: {e}")
        return False
# --- End Helper Function ---

# --- apply_bevel_via_blender function removed as it's no longer used ---
