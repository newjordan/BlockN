from collections import defaultdict
import math
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
    """
    width, height, depth = dimensions
    hw, hh, hd = width / 2.0, height / 2.0, depth / 2.0

    # Vertices relative to geometric center
    # Base is at -hh, Top is at +hh
    v0 = [-hw, -hh, -hd] # Back-Left-Bottom
    v1 = [ hw, -hh, -hd] # Back-Right-Bottom
    v2 = [ hw, -hh,  hd] # Front-Right-Bottom
    v3 = [-hw, -hh,  hd] # Front-Left-Bottom

    if orientation == '+X':
        v4 = [hw, hh, -hd] # Back-Right-Top
        v5 = [hw, hh,  hd] # Front-Right-Top
        vertices = [v0, v1, v2, v3, v4, v5]
        faces = [
            [0, 1, 2], [0, 2, 3], # Base (Down)
            [1, 4, 5], [1, 5, 2], # Right (Out)
            [0, 4, 1],             # Back (Triangle)
            [3, 2, 5],             # Front (Triangle)
            [0, 3, 5], [0, 5, 4]  # Slope
        ]
    elif orientation == '-X':
        v4 = [-hw, hh, -hd] # Back-Left-Top
        v5 = [-hw, hh,  hd] # Front-Left-Top
        vertices = [v0, v1, v2, v3, v4, v5]
        faces = [
            [0, 1, 2], [0, 2, 3], # Base
            [0, 3, 5], [0, 5, 4], # Left (Out)
            [0, 4, 1],             # Back (Triangle)
            [3, 2, 5],             # Front (Triangle)
            [1, 5, 2], [1, 4, 5]  # Slope
        ]
    elif orientation == '+Z':
        v4 = [-hw, hh, hd] # Front-Left-Top
        v5 = [ hw, hh, hd] # Front-Right-Top
        vertices = [v0, v1, v2, v3, v4, v5]
        faces = [
            [0, 1, 2], [0, 2, 3], # Base
            [3, 2, 5], [3, 5, 4], # Front (Out)
            [0, 3, 4],             # Left (Triangle)
            [1, 5, 2],             # Right (Triangle)
            [0, 5, 1], [0, 4, 5]  # Slope
        ]
    elif orientation == '-Z':
        v4 = [-hw, hh, -hd] # Back-Left-Top
        v5 = [ hw, hh, -hd] # Back-Right-Top
        vertices = [v0, v1, v2, v3, v4, v5]
        faces = [
            [0, 1, 2], [0, 2, 3], # Base
            [0, 5, 1], [0, 4, 5], # Back (Out)
            [0, 3, 4],             # Left (Triangle)
            [1, 5, 2],             # Right (Triangle)
            [3, 2, 5], [3, 5, 4]  # Slope
        ]
    else:
        raise ValueError(f"Invalid orientation: {orientation}")

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.apply_translation(position) # position is the geometric center

    # UVs
    bounds = mesh.bounds
    span = mesh.extents.copy()
    span[span == 0] = 1.0
    uvs = (mesh.vertices - bounds[0]) / span
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs[:, :2])
    return mesh

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


def create_arch_mesh(position: np.ndarray = np.array([0.0, 0.0, 0.0]),
                     dimensions: np.ndarray = np.array([2.0, 2.0, 1.0]),
                     parameters: Optional[Dict[str, Any]] = None) -> trimesh.Trimesh:
    """
    Creates a segmented Roman-style arch mesh.

    The mesh is assembled from watertight prisms (two legs + curved voussoirs)
    and centered at `position`.

    Args:
        position: Geometric center position (x, y, z).
        dimensions: Overall width(X), height(Y), depth(Z).
        parameters: Optional settings:
            - thickness (float): arch ring/leg thickness before Y-scaling.
            - segments (int): number of voussoir segments in the curved section.

    Returns:
        A trimesh.Trimesh object representing the arch.
    """
    params = parameters or {}
    width = max(1e-3, float(dimensions[0]))
    height = max(1e-3, float(dimensions[1]))
    depth = max(1e-3, float(dimensions[2]))

    segments = int(params.get("segments", 14))
    segments = max(6, segments)

    outer_radius = width / 2.0
    default_thickness = min(width * 0.16, outer_radius * 0.42)
    thickness = float(params.get("thickness", default_thickness))
    thickness = max(0.04, min(thickness, outer_radius * 0.7))
    inner_radius = max(0.03, outer_radius - thickness)
    leg_height_raw = max(thickness * 1.6, outer_radius * 0.46)
    parts: List[trimesh.Trimesh] = []

    # Side legs.
    leg_extents = np.array([thickness, leg_height_raw, depth], dtype=float)
    left_leg = trimesh.primitives.Box(extents=leg_extents).to_mesh()
    left_leg.apply_translation(np.array([-outer_radius + thickness / 2.0, leg_height_raw / 2.0, 0.0]))
    parts.append(left_leg)

    right_leg = trimesh.primitives.Box(extents=leg_extents).to_mesh()
    right_leg.apply_translation(np.array([outer_radius - thickness / 2.0, leg_height_raw / 2.0, 0.0]))
    parts.append(right_leg)

    # Curved crown formed by rotated voussoir prisms.
    theta_step = math.pi / float(segments)
    radial_mid = inner_radius + thickness / 2.0
    tangential_len = max(thickness * 0.9, radial_mid * theta_step * 1.08)
    voussoir_extents = np.array([thickness, tangential_len, depth], dtype=float)

    for idx in range(segments):
        theta = (idx + 0.5) * theta_step
        voussoir = trimesh.primitives.Box(extents=voussoir_extents).to_mesh()

        rotate = trimesh.transformations.rotation_matrix(theta, [0.0, 0.0, 1.0])
        voussoir.apply_transform(rotate)

        center = np.array(
            [
                radial_mid * math.cos(theta),
                leg_height_raw + radial_mid * math.sin(theta),
                0.0,
            ],
            dtype=float,
        )
        voussoir.apply_translation(center)
        parts.append(voussoir)

    mesh = trimesh.util.concatenate(parts)

    # Rotated voussoirs can extend slightly past the nominal radius. Normalize
    # the completed arch into the requested bounding box so GUI dimensions are
    # authoritative for placement, collision checks, and exports.
    raw_bounds = mesh.bounds
    raw_extents = mesh.extents.copy()
    raw_extents[raw_extents == 0] = 1.0
    mesh.apply_translation(-raw_bounds[0])
    mesh.apply_scale(np.array([width, height, depth], dtype=float) / raw_extents)
    mesh.apply_translation(-np.array([width, height, depth], dtype=float) / 2.0)
    mesh.apply_translation(position)

    bounds = mesh.bounds
    span = mesh.extents.copy()
    span[span == 0] = 1.0
    uvs = (mesh.vertices - bounds[0]) / span
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs[:, :2])
    return mesh


def create_dome_mesh(position: np.ndarray = np.array([0.0, 0.0, 0.0]),
                     dimensions: np.ndarray = np.array([2.0, 1.0, 2.0]),
                     parameters: Optional[Dict[str, Any]] = None) -> trimesh.Trimesh:
    """
    Creates a stepped hemispherical dome cap with a flat base.
    Constructed as a single manifold shell to avoid internal faces and Z-fighting.
    """
    params = parameters or {}
    width = max(1e-3, float(dimensions[0]))
    height = max(1e-3, float(dimensions[1]))
    depth = max(1e-3, float(dimensions[2]))

    layers = int(params.get("layers", 12))
    layers = max(4, layers)
    sections = int(params.get("sections", 24))
    sections = max(8, sections)

    # We build a unit dome (radius 1, height 1) at origin, then scale.
    layer_h = 1.0 / float(layers)

    vertices = []
    faces = []

    # Pre-calculate radii at each horizontal level (y-coordinates)
    # Level 0 is the base (y=0), Level layers is the top (y=1)
    levels_y = np.linspace(0, 1, layers + 1)
    # Use the radius at the middle of each layer for the vertical wall
    # but the steps happen at the boundaries.
    # Actually, for a STEPPED dome, each layer i has a constant radius R_i.
    # Radius of layer i (from y_i to y_{i+1}) is sqrt(1 - y_mid^2).
    layer_radii = []
    for i in range(layers):
        y_mid = (levels_y[i] + levels_y[i+1]) / 2.0
        layer_radii.append(math.sqrt(max(0.0, 1.0 - y_mid**2)))

    # 1. Base Cap (y=0)
    base_center_idx = len(vertices)
    vertices.append([0, 0, 0])
    base_start_idx = len(vertices)
    for s in range(sections):
        angle = 2 * math.pi * s / sections
        vertices.append([math.cos(angle) * layer_radii[0], 0, math.sin(angle) * layer_radii[0]])

    for s in range(sections):
        next_s = (s + 1) % sections
        faces.append([base_center_idx, base_start_idx + next_s, base_start_idx + s])

    # 2. Layers and Steps
    current_ring_start = base_start_idx
    for i in range(layers):
        r_current = layer_radii[i]
        y_bottom = levels_y[i]
        y_top = levels_y[i+1]

        # Upper ring of this layer's side wall
        upper_ring_start = len(vertices)
        for s in range(sections):
            angle = 2 * math.pi * s / sections
            vertices.append([math.cos(angle) * r_current, y_top, math.sin(angle) * r_current])

        # Side faces (quads)
        for s in range(sections):
            next_s = (s + 1) % sections
            v1 = current_ring_start + s
            v2 = current_ring_start + next_s
            v3 = upper_ring_start + next_s
            v4 = upper_ring_start + s
            faces.extend([[v1, v2, v3], [v1, v3, v4]])

        if i < layers - 1:
            # Annular step to the next (smaller) radius
            r_next = layer_radii[i+1]
            next_ring_start = len(vertices)
            for s in range(sections):
                angle = 2 * math.pi * s / sections
                vertices.append([math.cos(angle) * r_next, y_top, math.sin(angle) * r_next])

            # Step faces (quads)
            for s in range(sections):
                next_s = (s + 1) % sections
                v1 = upper_ring_start + s
                v2 = upper_ring_start + next_s
                v3 = next_ring_start + next_s
                v4 = next_ring_start + s
                faces.extend([[v1, v2, v3], [v1, v3, v4]])

            current_ring_start = next_ring_start
        else:
            # Top Cap
            top_center_idx = len(vertices)
            vertices.append([0, y_top, 0])
            for s in range(sections):
                next_s = (s + 1) % sections
                faces.append([top_center_idx, upper_ring_start + s, upper_ring_start + next_s])

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    # Scale to requested dimensions and center it at origin.
    mesh.apply_scale([width / 2.0, height, depth / 2.0])
    mesh.apply_translation(np.array([0.0, -height / 2.0, 0.0], dtype=float))
    mesh.apply_translation(position)

    # The procedural rings are consistently connected but historically faced
    # inward, producing negative signed volume.  Keep outward normals so dome
    # artifacts satisfy the same solid-mesh contract as every other primitive.
    if mesh.volume < 0.0:
        mesh.invert()

    bounds = mesh.bounds
    span = mesh.extents.copy()
    span[span == 0] = 1.0
    uvs = (mesh.vertices - bounds[0]) / span
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs[:, :2])
    return mesh


def get_boolean_backend_status() -> Dict[str, Any]:
    """
    Report available trimesh boolean backends for union operations.
    """
    available = sorted(str(engine) for engine in trimesh.boolean.engines_available if engine is not None)
    return {
        "available": bool(available),
        "engines": available,
    }


def optimize_blocks(blocks: list, return_report: bool = False):
    """
    Optimizes blocks by merging meshes of the same color using trimesh union.

    Args:
        blocks: A list of Block objects from the scene.
        return_report: If True, return (meshes, report) where report contains backend/fallback stats.

    Returns:
        If return_report is False: list of trimesh.Trimesh objects.
        If return_report is True: tuple[list[trimesh.Trimesh], dict].
    """
    print("Optimizing blocks (trimesh union)...")
    backend_status = get_boolean_backend_status()
    available_engines = backend_status["engines"]
    report: Dict[str, Any] = {
        "backend_status": backend_status,
        "groups_total": 0,
        "groups_union_attempted": 0,
        "groups_union_succeeded": 0,
        "groups_fallback": 0,
        "fallback_reasons": {},
    }
    if not blocks:
        return ([], report) if return_report else []

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
        report["groups_total"] += 1
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
            report["groups_union_attempted"] += 1
            print(f"  Processing color {color}: {len(mesh_list)} meshes need union.")
            # Prefer deterministic explicit engines; skip attempts when no backend exists.
            engine_sequence: List[Optional[str]] = []
            if "manifold" in available_engines:
                engine_sequence.append("manifold")
            if "blender" in available_engines:
                engine_sequence.append("blender")

            if not engine_sequence:
                reason = "no_boolean_backend"
                report["groups_fallback"] += 1
                report["fallback_reasons"][reason] = report["fallback_reasons"].get(reason, 0) + 1
                print("    No boolean backend available. Falling back to non-unioned meshes.")
                combined = None
            else:
                for engine in engine_sequence:
                    try:
                        print(f"    Trying trimesh union with engine='{engine}'...")
                        combined = trimesh.boolean.union(mesh_list, engine=engine)
                        if combined and not combined.is_empty:
                            report["groups_union_succeeded"] += 1
                            print(f"    Trimesh union successful with engine='{engine}'.")
                            break
                        print(f"    Trimesh union returned empty with engine='{engine}'.")
                        combined = None
                    except Exception as union_err:
                        print(
                            f"    ERROR during trimesh union (engine='{engine}',"
                            f" {type(union_err).__name__}): {union_err}"
                        )
                        combined = None

                if combined is None:
                    reason = "backend_union_failed"
                    report["groups_fallback"] += 1
                    report["fallback_reasons"][reason] = report["fallback_reasons"].get(reason, 0) + 1
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
    return (optimized_meshes, report) if return_report else optimized_meshes

# Future helper functions for adjacency checks, merging, etc. can go here.
