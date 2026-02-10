from collections import defaultdict
import trimesh
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .materials import ColorTuple
# Imports for blender operations removed as they are no longer used directly by optimize_blocks
# from .blender_ops import _run_blender_script, apply_bevel_via_blender

# Define ColorTuple if not imported elsewhere - REMOVED, now imported


# --- Helper function to union meshes using Blender ---
# This function is removed as we revert to trimesh union.
# --- End Helper Function ---


def create_block_mesh(position: np.ndarray = np.array([0.0, 0.0, 0.0]),
                      dimensions: np.ndarray = np.array([1.0, 1.0, 1.0])) -> trimesh.Trimesh:
    """
    Creates a rectangular prism mesh centered at a specified position.

    Args:
        position: A numpy array representing the center of the prism (x, y, z).
        dimensions: A numpy array for width(X), height(Y), depth(Z).

    Returns:
        A trimesh.Trimesh object representing the prism.
    """
    # Create a box primitive centered at the origin
    primitive = trimesh.primitives.Box(extents=dimensions)
    # Translate the primitive so the center of its bottom face is at the desired position.
    # The primitive is centered at the origin [0,0,0].
    # Translate the primitive so its center is at the desired 'position'.
    primitive.apply_translation(position)

    # Create a standard Trimesh object from the primitive's data
    mesh = trimesh.Trimesh(vertices=primitive.vertices, faces=primitive.faces)

    # Copy visual properties if needed (though color is applied later in Block)
    # mesh.visual = primitive.visual

    # --- Basic UV Generation (Box Projection Approximation) ---
    # Scale vertex positions relative to bounds to approximate UVs (0 to 1 range)
    # This is a simple approach; results may vary depending on usage.
    bounds = primitive.bounds
    span = dimensions.copy() # Use dimensions directly for span
    # Avoid division by zero for flat dimensions if any
    span[span == 0] = 1.0
    uvs = (mesh.vertices - bounds[0]) / span
    # Use XY for UV coords - other projections might be better depending on need
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs[:, :2])

    return mesh

def create_wedge_mesh(position: np.ndarray = np.array([0.0, 0.0, 0.0]),
                      dimensions: np.ndarray = np.array([1.0, 1.0, 1.0]),
                      orientation: str = '+X') -> trimesh.Trimesh:
    """
    Creates a wedge mesh with its geometric center at the specified position.
    The base lies on the XZ plane relative to the position's Y coordinate minus half the height.
    The slope direction is determined by the 'orientation' parameter.

    Args:
        position: Geometric center of the wedge (x, y, z).
        dimensions: Width(X), Height(Y), Depth(Z) of the wedge's bounding box.
        orientation: Direction of the slope ('+X', '-X', '+Z', '-Z').
                     '+X' means the peak edge is at maximum X.
                     '-X' means the peak edge is at minimum X.
                     '+Z' means the peak edge is at maximum Z.
                     '-Z' means the peak edge is at minimum Z.

    Returns:
        A trimesh.Trimesh object representing the wedge.
    """
    width = dimensions[0]
    height = dimensions[1]
    depth = dimensions[2]
    half_w = width / 2.0
    half_d = depth / 2.0

    # Define vertices relative to the center of the base face (0, 0, 0)
    # Base vertices (Y=0)
    v0 = [-half_w, 0, -half_d] # Back left
    v1 = [ half_w, 0, -half_d] # Back right
    v2 = [ half_w, 0,  half_d] # Front right
    v3 = [-half_w, 0,  half_d] # Front left
    # Top vertices (Y=height) - depends on orientation
    v4, v5 = None, None # Peak edge vertices

    # Define faces based on orientation (5 faces total: Base, 2 Sides, Back/Front, Slope)
    if orientation == '+X': # Peak edge at X = +half_w (v4, v5)
        v4 = [ half_w, height, -half_d] # Top back right
        v5 = [ half_w, height,  half_d] # Top front right
        vertices = np.array([v0, v1, v2, v3, v4, v5])
        faces = np.array([
            # Base (Y=0) - Reversed winding for outward normal
            [2, 1, 0], [3, 2, 0],
            [0, 1, 4],             # Back Side (Z=-hd, Triangle)
            [3, 2, 5],             # Front Side (Z=+hd, Triangle)
            [1, 2, 5], [1, 5, 4],  # Right Vertical Face (X=+hw)
            [0, 3, 5], [0, 5, 4]   # Sloped Face (connecting X=-hw base to X=+hw peak)
        ])
    elif orientation == '-X': # Peak edge at X = -half_w (v4, v5)
        v4 = [-half_w, height, -half_d] # Top back left
        v5 = [-half_w, height,  half_d] # Top front left
        vertices = np.array([v0, v1, v2, v3, v4, v5])
        faces = np.array([
            # Base (Y=0) - Reversed winding for outward normal
            [2, 1, 0], [3, 2, 0],
            [0, 1, 4],             # Back Side (Z=-hd, Triangle)
            [3, 2, 5],             # Front Side (Z=+hd, Triangle)
            [0, 3, 5], [0, 5, 4],  # Left Vertical Face (X=-hw)
            [1, 2, 5], [1, 5, 4]   # Sloped Face (connecting X=+hw base to X=-hw peak)
        ])
    elif orientation == '+Z': # Peak edge at Z = +half_d (v4, v5)
        v4 = [-half_w, height, half_d] # Top front left
        v5 = [ half_w, height, half_d] # Top front right
        vertices = np.array([v0, v1, v2, v3, v4, v5])
        faces = np.array([
            # Base (Y=0) - Reversed winding for outward normal
            [2, 1, 0], [3, 2, 0],
            [0, 3, 4],             # Left Side (X=-hw, Triangle)
            [1, 2, 5],             # Right Side (X=+hw, Triangle)
            [3, 2, 5], [3, 5, 4],  # Front Vertical Face (Z=+hd)
            [0, 1, 5], [0, 5, 4]   # Sloped Face (connecting Z=-hd base to Z=+hd peak)
        ])
    elif orientation == '-Z': # Peak edge at Z = -half_d (v4, v5)
        v4 = [-half_w, height, -half_d] # Top back left
        v5 = [ half_w, height, -half_d] # Top back right
        vertices = np.array([v0, v1, v2, v3, v4, v5])
        faces = np.array([
            # Base (Y=0) - Reversed winding for outward normal
            [2, 1, 0], [3, 2, 0],
            [0, 3, 4],             # Left Side (X=-hw, Triangle)
            [1, 2, 5],             # Right Side (X=+hw, Triangle)
            [0, 1, 5], [0, 5, 4],  # Back Vertical Face (Z=-hd)
            [3, 2, 5], [3, 5, 4]   # Sloped Face (connecting Z=+hd base to Z=-hd peak)
        ])
    else:
        raise ValueError(f"Invalid orientation for wedge: {orientation}")

    # Create the mesh centered at origin (base center)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Keep wedge generation resilient when optional trimesh deps (e.g. scipy) are missing.
    try:
        mesh.fix_normals()
    except Exception:
        # Winding is already explicitly defined above; skipping normal repair is acceptable fallback.
        pass

    # Translate to the desired position (center of the base face)
    # The mesh origin is already at the base center, so just translate by position.
    # However, the position passed from generator assumes bottom-face-center Y.
    # Adjust Y translation like in create_block_mesh. <-- This comment is now misleading
    # The mesh origin is at the base center (0,0,0).
    # We want the geometric center (0, height/2, 0 relative to base center)
    # to end up at the target 'position'.
    # Therefore, translate the mesh (origin at base center) by: position - [0, height/2, 0]
    translation_vector = position - np.array([0, height / 2.0, 0])
    mesh.apply_translation(translation_vector)

    # --- Basic UV Generation (Box Projection Approximation) ---
    # Scale vertex positions relative to bounds to approximate UVs (0 to 1 range)
    bounds = mesh.bounds
    span = mesh.extents.copy() # Create a writable copy
    # Avoid division by zero for flat dimensions if any
    span[span == 0] = 1.0
    uvs = (mesh.vertices - bounds[0]) / span
    # Use XY for UV coords - other projections might be better depending on need
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs[:, :2])

    return mesh


def create_cylinder_mesh(position: np.ndarray = np.array([0.0, 0.0, 0.0]),
                         radius: float = 0.5, height: float = 1.0,
                         parameters: Optional[Dict[str, Any]] = None) -> trimesh.Trimesh:
    """
    Creates a simple cylinder mesh with its geometric center at `position`,
    aligned along the Y-axis (height direction).

    Args:
        position: Geometric center position (x, y, z).
        radius: Radius of the cylinder.
        height: Height of the cylinder (along Y).
        parameters: Optional dictionary of parameters. Expects 'sections' (int).

    Returns:
        A trimesh.Trimesh object representing the cylinder.
    """
    params = parameters or {}
    sections = params.get('sections', 16) # Default to 16 sections if not specified

    # Use trimesh's primitive generator (creates Z-aligned cylinder at origin)
    cylinder_primitive = trimesh.primitives.Cylinder(radius=radius, height=height, sections=sections)

    # Rotate it to align with Y-axis
    # Rotation axis is X-axis, angle is 90 degrees (pi/2 radians)
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    cylinder_primitive.apply_transform(rotation)

    # Translate so the geometric center is at the desired position.
    # The primitive center is at (0, 0, 0) after rotation.
    cylinder_primitive.apply_translation(position)

    # Create a standard Trimesh object from the primitive's data
    mesh = trimesh.Trimesh(vertices=cylinder_primitive.vertices, faces=cylinder_primitive.faces)

    # --- Basic UV Generation (Cylindrical Projection Approximation) ---
    # This is more complex than box projection. For simplicity, we might
    # use the same box projection or skip UVs for cylinders initially.
    # Let's apply the simple box projection for now.
    bounds = mesh.bounds
    span = mesh.extents.copy() # Create a writable copy
    span[span == 0] = 1.0
    uvs = (mesh.vertices - bounds[0]) / span
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs[:, :2])

    return mesh


# Update signature: Remove bevel_params and blender_executable as they are no longer used here.
def optimize_blocks(blocks: list) -> List[trimesh.Trimesh]:
    """
    Optimizes blocks by merging meshes of the same color using trimesh union.

    Args:
        blocks: A list of Block objects from the scene.

    Returns:
        A list of trimesh.Trimesh objects, representing the merged meshes (one per color group).
    """
    print("Optimizing blocks (trimesh union)...")
    if not blocks:
        return []

    def _color_mesh(mesh: trimesh.Trimesh, color: ColorTuple) -> trimesh.Trimesh:
        colored = mesh.copy()
        if not hasattr(colored, "visual"):
            colored.visual = trimesh.visual.ColorVisuals(mesh=colored)
        face_colors = np.tile(color, (len(colored.faces), 1))
        colored.visual.face_colors = face_colors
        return colored

    # 1. Group blocks by color
    blocks_by_color = defaultdict(list)
    for block in blocks:
        # Ensure color is hashable (it's a tuple, so it should be)
        blocks_by_color[block.color].append(block)

    optimized_meshes = []
    # 2. Process each color group: Perform union using trimesh internal engine
    for color, color_blocks in blocks_by_color.items():
        # Get valid, watertight meshes for this color group
        mesh_list = [b.mesh for b in color_blocks if b.mesh is not None and b.mesh.is_watertight]

        if not mesh_list:
            print(f"  Skipping color {color}: No valid/watertight meshes found.")
            continue

        combined = None
        if len(mesh_list) == 1:
            print(f"  Processing color {color}: Single mesh, no union needed.")
            combined = mesh_list[0].copy() # Copy to avoid modifying original block mesh
        elif len(mesh_list) > 1:
            print(f"  Processing color {color}: {len(mesh_list)} meshes need union. Attempting trimesh internal union...")
            try:
                # Use default engine (engine=None), letting trimesh find a backend
                print("    Trying trimesh union with default engine (engine=None)...")
                combined = trimesh.boolean.union(mesh_list, engine=None)
                if combined and not combined.is_empty:
                    print("    Trimesh default union successful.")
                else:
                    # This case might be hit if the union results in nothing, even if a backend runs
                    print("    Trimesh default union failed or resulted in empty mesh.")
                    combined = None # Ensure it's None if failed
            except ImportError as imp_err:
                # This might catch if trimesh itself is missing a dependency
                print(f"    ERROR during trimesh default union (ImportError): {imp_err}")
                print("    Ensure trimesh and its dependencies are installed correctly.")
                combined = None
            except ValueError as val_err:
                # This is often the "No backends available" error
                print(f"    ERROR during trimesh default union (ValueError): {val_err}")
                combined = None
            except Exception as union_err:
                # Catch any other unexpected errors during the boolean operation
                print(f"    ERROR during trimesh default union (Type: {type(union_err).__name__}): {union_err}")
                combined = None
        else: # Should not happen if mesh_list is not empty
            continue

        if combined is None or combined.is_empty:
            print(f"    Falling back to non-unioned meshes for color {color}.")
            for mesh in mesh_list:
                try:
                    optimized_meshes.append(_color_mesh(mesh, color))
                except Exception as color_err:
                    print(f"    Error applying color to fallback mesh for color {color}: {color_err}")
            continue

        # 3. Add the combined mesh (if successful) to the results and apply color
        try:
            optimized_meshes.append(_color_mesh(combined, color))
            print(f"    Added combined mesh for color {color}.")
        except Exception as color_err:
            print(f"    Error applying color to combined mesh for color {color}: {color_err}")

    print(f"Optimization finished. Reduced {len(blocks)} blocks to {len(optimized_meshes)} final meshes.")
    return optimized_meshes

# Future helper functions for adjacency checks, merging, etc. can go here.
