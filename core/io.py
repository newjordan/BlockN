import trimesh
from typing import List
import os
import json
from core.scene import Scene # Import Scene for type hinting

def export_meshes_to_glb(meshes: List[trimesh.Trimesh], file_path: str):
    """
    Exports a list of Trimesh meshes to a single GLB file (binary glTF).

    If multiple meshes are provided, they are concatenated into a single mesh
    before export.

    Args:
        meshes: A list of trimesh.Trimesh objects.
        file_path: The full path for the output GLB file.

    Raises:
        ValueError: If the meshes list is empty.
        Exception: Propagates exceptions from trimesh export.
    """
    if not meshes:
        raise ValueError("Cannot export an empty list of meshes.")

    if not file_path.lower().endswith(".glb"):
        file_path += ".glb"

    # Ensure the directory exists (if one is present)
    export_dir = os.path.dirname(file_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    # Concatenate if more than one mesh
    if len(meshes) > 1:
        print(f"Concatenating {len(meshes)} meshes for export...")
        final_mesh = trimesh.util.concatenate(meshes)
    else:
        final_mesh = meshes[0]

    print(f"Exporting mesh to: {file_path}")
    try:
        # Export the mesh to GLB format
        # Trimesh uses pyglet (or other dependencies) for GLB export,
        # which generally handles materials/colors well.
        final_mesh.export(file_type='glb', file_obj=file_path)
        print("Export successful.")
    except Exception as e:
        print(f"Error during GLB export: {e}")
        raise # Re-raise the exception to be caught by the caller


def save_scene_to_json(scene: Scene, file_path: str):
    """
    Saves the scene state (block positions and colors) to a JSON file.

    Args:
        scene: The Scene object to save.
        file_path: The full path for the output JSON file.

    Raises:
        Exception: Propagates exceptions from file I/O or JSON serialization.
    """
    if not file_path.lower().endswith(".json"):
        file_path += ".json"

    # Ensure the directory exists (if one is present)
    export_dir = os.path.dirname(file_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    print(f"Saving scene to: {file_path}")
    try:
        scene_data = scene.to_dict()
        with open(file_path, 'w') as f:
            json.dump(scene_data, f, indent=4) # Use indent for readability
        print("Scene save successful.")
    except Exception as e:
        print(f"Error during JSON scene save: {e}")
        raise


def load_scene_from_json(scene: Scene, file_path: str):
    """
    Loads scene state from a JSON file into the provided Scene object.

    Args:
        scene: The Scene object to load data into (will be cleared first).
        file_path: The full path of the JSON file to load.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        Exception: Propagates exceptions from file I/O or JSON deserialization.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scene file not found: {file_path}")

    print(f"Loading scene from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            scene_data = json.load(f)
        scene.from_dict(scene_data) # Load data into the existing scene object
        print("Scene load successful.")
    except Exception as e:
        print(f"Error during JSON scene load: {e}")
        raise
