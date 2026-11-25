from dataclasses import dataclass, field
import numpy as np
import trimesh
from typing import List, Optional, Dict, Any, Tuple

from .geometry import create_block_mesh, create_wedge_mesh, create_cylinder_mesh # Import geometry functions
from .materials import ColorTuple # Import color definitions

# A new block counts as supported when its base is this close to the ground
# or to another block's top face.
SUPPORT_TOLERANCE = 1e-5
# Plan overlap must have positive area. Edge contact is not support.
SUPPORT_OVERLAP = 1e-6
# Boolean intersection thinner than this is treated as a shared face, not a collision.
OVERLAP_VOLUME = 1e-6

# Define block type constants
CUBE = "CUBE"
WEDGE = "WEDGE" # Renamed from SLOPE_X_POS
CYLINDER = "CYLINDER"

@dataclass
class Block:
    """Represents a single block in the scene."""
    position: np.ndarray
    block_type: str = CUBE # Default to cube
    color: ColorTuple = (200, 200, 200, 255)  # Default gray color
    dimensions: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0])) # W, H, D
    parameters: Dict[str, Any] = field(default_factory=dict) # Type-specific parameters
    # Allow mesh to be passed during initialization
    mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False) # Don't include mesh in default repr
    block_id: int = -1 # Unique identifier for the block within the scene

    def __post_init__(self):
        """Generate the mesh after initialization if not provided."""
        # Only generate mesh if it wasn't provided during __init__
        if self.mesh is None:
            self._generate_mesh()

        # Apply color to vertex colors if mesh exists (moved from _generate_mesh)
        # This ensures color is applied even if mesh was passed in __init__
        if self.mesh:
            self._apply_color_to_mesh()

    def _generate_mesh(self):
        """Internal helper to generate the mesh based on block type and parameters."""
        # Create mesh based on block type
        if self.block_type == CUBE:
            # Pass dimensions to the updated function
            self.mesh = create_block_mesh(self.position, dimensions=self.dimensions)
        elif self.block_type == WEDGE:
            # Scene positions are geometric centers; the wedge primitive takes the base center.
            orientation = self.parameters.get('orientation', '+X')
            base_position = np.asarray(self.position, dtype=float) - np.array([0.0, self.dimensions[1] / 2.0, 0.0])
            self.mesh = create_wedge_mesh(base_position, dimensions=self.dimensions, orientation=orientation)
        elif self.block_type == CYLINDER:
            # Prioritize parameters for cylinder dimensions
            radius = self.parameters.get('radius', self.dimensions[0] / 2.0) # Default radius from X dim if not in params
            height = self.parameters.get('height', self.dimensions[1])       # Default height from Y dim if not in params
            base_position = np.asarray(self.position, dtype=float) - np.array([0.0, float(height) / 2.0, 0.0])
            self.mesh = create_cylinder_mesh(base_position, radius=radius, height=height, parameters=self.parameters)
        else:
            # Default or fallback to cube if type is unknown
            print(f"Warning: Unknown block type '{self.block_type}'. Creating cube.")
            self.mesh = create_block_mesh(self.position, dimensions=self.dimensions)
        # Color application moved to __post_init__ to handle meshes passed via __init__

    def _apply_color_to_mesh(self):
        """Applies the block's current color to its mesh."""
        if not self.mesh:
            return
        # Ensure mesh visual exists and handle potential vertex color issues
        if not hasattr(self.mesh, 'visual'):
            self.mesh.visual = trimesh.visual.ColorVisuals(mesh=self.mesh)
        try:
            # Use face colors for robustness with boolean ops, but vertex colors are needed for smooth shading
            # Let's stick to vertex colors for now, as optimization applies face colors later.
            vertex_colors = np.tile(self.color, (len(self.mesh.vertices), 1))
            self.mesh.visual.vertex_colors = vertex_colors
        except Exception as e:
            print(f"Warning: Could not apply vertex colors to block {self.block_id}: {e}")
            # Fallback: Apply face colors if vertex colors fail? # TODO: Consider fallback
            # face_colors = np.tile(self.color, (len(self.mesh.faces), 1))
            # self.mesh.visual.face_colors = face_colors


    def update_color(self, color: ColorTuple):
        """Updates the block's color and its mesh's vertex colors."""
        self.color = color
        self._apply_color_to_mesh() # Use the helper method

    def to_dict(self) -> Dict[str, Any]:
        """Serializes block data to a dictionary (excluding the mesh)."""
        return {
            "block_id": self.block_id,
            "position": self.position.tolist(), # Convert numpy array to list
            "color": list(self.color), # Convert tuple to list
            "block_type": self.block_type,
            "dimensions": self.dimensions.tolist(), # Add dimensions
            "parameters": _json_value(self.parameters) # Add parameters
        }


def _json_value(value: Any) -> Any:
    """Convert numpy values so scene dictionaries can be written as JSON."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _bounds_overlap(a_bounds: np.ndarray, b_bounds: np.ndarray, tolerance: float = SUPPORT_TOLERANCE) -> bool:
    """True when two AABBs overlap by more than a shared face."""
    return bool(
        np.all(a_bounds[0] < b_bounds[1] - tolerance)
        and np.all(a_bounds[1] > b_bounds[0] + tolerance)
    )


def _positive_overlap(mesh: trimesh.Trimesh, other: trimesh.Trimesh) -> bool:
    """True when two meshes share volume. Face contact returns False."""
    if not _bounds_overlap(mesh.bounds, other.bounds):
        return False
    try:
        intersection = trimesh.boolean.intersection([mesh, other], engine="manifold")
    except Exception:
        # Fail closed when the volume test cannot run: a strict AABB overlap
        # is reported as a collision rather than silently allowed.
        return True
    if intersection is None or intersection.is_empty:
        return False
    volume = getattr(intersection, "volume", 0.0)
    if volume != volume:
        return True
    return float(volume) > OVERLAP_VOLUME


class _MeshCollisionIndex:
    """Overlap index used when python-fcl cannot be loaded.

    Names are stored in ``_objs`` so callers can count tracked meshes the same
    way they do with trimesh's CollisionManager.
    """

    def __init__(self):
        self._objs: Dict[str, trimesh.Trimesh] = {}

    def add_object(self, name: str, mesh: trimesh.Trimesh):
        self._objs[name] = mesh

    def in_collision_single(self, mesh: trimesh.Trimesh) -> bool:
        return any(_positive_overlap(mesh, other) for other in self._objs.values())

    def in_collision_internal(self, return_names: bool = False, return_data: bool = False):
        pairs = set()
        items = list(self._objs.items())
        for index, (name, mesh) in enumerate(items):
            for other_name, other in items[index + 1:]:
                if _positive_overlap(mesh, other):
                    pairs.add(tuple(sorted((name, other_name))))
        collided = bool(pairs)
        if return_names and return_data:
            return collided, pairs, []
        if return_names:
            return collided, pairs
        if return_data:
            return collided, []
        return collided


def _create_collision_manager(use_collision: bool):
    """Create an overlap index. python-fcl is preferred; volume checks are the fallback."""
    if not use_collision:
        return None
    try:
        return trimesh.collision.CollisionManager()
    except Exception as exc:
        print(f"Info: python-fcl unavailable ({exc}). Using mesh-volume overlap checks.")
        return _MeshCollisionIndex()


class Scene:
    """Manages a collection of blocks."""
    def __init__(self, use_collision: bool = True):
        self.blocks: List[Block] = []
        self._next_block_id = 0
        # Collision manager for accurate overlap detection
        self.collision_manager = _create_collision_manager(use_collision)
        collision_enabled = self.collision_manager is not None
        # Setting to control strict overlap prevention (Task 3.1)
        self.enable_strict_overlap_prevention: bool = collision_enabled
        # Setting to control structural support checking level ("HYBRID", "GRID", "OFF")
        self.support_check_mode: str = "HYBRID" if collision_enabled else "OFF"
        # Legacy cell keys, kept in sync with the support surfaces below.
        self.occupied_cells = set() # Stores (x, y, z) integer tuples
        # (block_id, x_min, x_max, z_min, z_max, top_y)
        self._support_surfaces: List[Tuple[int, float, float, float, float, float]] = []

    def add_block(self, position: np.ndarray, block_type: str = CUBE, color: Optional[ColorTuple] = None,
                  parameters: Optional[Dict[str, Any]] = None, dimensions: Optional[np.ndarray] = None) -> Optional[Block]:
        """
        Adds a new block to the scene after checking for volumetric collisions and support.

        Args:
            position: The geometric center position (x, y, z) for the block.
            block_type: The type of block (e.g., CUBE, WEDGE, CYLINDER).
            color: Optional color tuple (RGBA).
            parameters: Optional dictionary of type-specific parameters.
            dimensions: Optional dimensions array (W, H, D).

        Returns:
            The created Block object if successfully added, otherwise None.
        """
        block_id = self._next_block_id # Get ID for potential use

        # --- Determine Actual Dimensions/Height ---
        # Use dimensions if provided, else default. Crucially, determine actual height for support check.
        actual_dimensions = np.array(dimensions if dimensions is not None else [1.0, 1.0, 1.0], dtype=float)
        actual_parameters = parameters or {}
        block_height = float(actual_dimensions[1]) # Default height from Y dimension
        if block_type == CYLINDER:
            # For cylinders, height might be in parameters, overriding dimensions[1]
            block_height = float(actual_parameters.get('height', actual_dimensions[1]))
            radius = float(actual_parameters.get('radius', actual_dimensions[0] / 2.0))
            # Persist cylinder dimensions in block state/serialization for consistent reload behavior.
            actual_dimensions = np.array([radius * 2.0, block_height, radius * 2.0], dtype=float)
        # Add elif for other block types if their height isn't simply dimensions[1]

        # --- 1. Generate Candidate Mesh ---
        # Use the determined actual dimensions/parameters
        new_mesh: Optional[trimesh.Trimesh] = None
        try:
            if block_type == CUBE:
                # Cube height is actual_dimensions[1], which is already block_height
                new_mesh = create_block_mesh(position, dimensions=actual_dimensions)
            elif block_type == WEDGE:
                # Wedge height is actual_dimensions[1], which is already block_height.
                # Primitive position is the base center; scene position is the geometric center.
                orientation = actual_parameters.get('orientation', '+X')
                base_position = np.asarray(position, dtype=float) - np.array([0.0, block_height / 2.0, 0.0])
                new_mesh = create_wedge_mesh(base_position, dimensions=actual_dimensions, orientation=orientation)
            elif block_type == CYLINDER:
                # Use actual parameters and pre-calculated block_height
                radius = actual_parameters.get('radius', actual_dimensions[0] / 2.0)
                base_position = np.asarray(position, dtype=float) - np.array([0.0, block_height / 2.0, 0.0])
                new_mesh = create_cylinder_mesh(base_position, radius=radius, height=block_height, parameters=actual_parameters)
            else:
                print(f"Warning: Unknown block type '{block_type}' requested. Cannot generate mesh.")
                return None # Cannot proceed without a mesh

            if new_mesh is None or new_mesh.is_empty:
                 print(f"Warning: Mesh generation failed or resulted in empty mesh for {block_type} at {position}. Skipping.")
                 return None

        except Exception as e:
            print(f"Error generating mesh for {block_type} at {position}: {e}")
            return None

        # --- 2. Check for Collisions (Conditional) ---
        if self.enable_strict_overlap_prevention and self.collision_manager is not None:
            try:
                is_collision = self.collision_manager.in_collision_single(new_mesh)
            except Exception as e:
                 # Handle potential errors within the collision check itself
                 print(f"Error during collision check for block at {position}: {e}. Skipping block.")
                 return None

            if is_collision:
                print(f"Warning: Overlap detected at {position} for {block_type}. Skipping block addition.")
                return None # Collision detected, do not add block

        # --- 2b. Check for Support (Based on Mode) ---
        if self.support_check_mode != "OFF":
            base_y = float(new_mesh.bounds[0, 1])
            if not self._surface_supports(new_mesh, base_y):
                if base_y < -SUPPORT_TOLERANCE:
                    print(f"Warning: Block at {position} is attempting to place below ground level. Skipping.")
                else:
                    print(f"Warning: Block at {position} is floating (no supporting surface within tolerance). Skipping.")
                return None
        # --- END SUPPORT CHECK ---

        # --- 3. All Checks Passed - Create Block Object and Add to Scene ---
        block_args = {
            "position": position,
            "block_type": block_type,
            "block_id": block_id,
            "mesh": new_mesh # Pass the pre-generated mesh
        }
        if color:
            block_args["color"] = color
        if parameters: # Use original parameters dict passed to function
            block_args["parameters"] = parameters
        # Always persist resolved dimensions (including inferred cylinder dimensions).
        block_args["dimensions"] = actual_dimensions

        try:
            block = Block(**block_args)
            # Mesh was already generated and passed in block_args["mesh"]
            # Color is applied in Block.__post_init__ if mesh exists

        except Exception as e:
             print(f"Error creating Block object for ID {block_id} at {position}: {e}")
             return None # Failed to create block instance

        # --- 4. Add to Scene Lists and Collision Manager ---
        # Occupied cell coord was calculated before block creation using actual height
        self.blocks.append(block)
        if self.collision_manager is not None:
            try:
                # Add the validated mesh to the collision manager
                self.collision_manager.add_object(name=f'block_{block_id}', mesh=new_mesh)
            except Exception as e:
                 # If adding to collision manager fails, we should probably roll back adding the block
                 print(f"CRITICAL ERROR: Failed to add mesh for block {block_id} to collision manager: {e}. Removing block from scene list.")
                 self.blocks.pop() # Remove the block we just added
                 # Do not increment block ID in this error case
                 return None

        # --- 5. Update support records and finalize ---
        self._remember_support(block)
        self._next_block_id += 1 # Increment ID only on successful addition
        return block # Return the created block

    def _surface_supports(self, mesh: trimesh.Trimesh, base_y: float) -> bool:
        """True when ``mesh`` rests on the ground or on a recorded top face."""
        if abs(base_y) <= SUPPORT_TOLERANCE:
            return True
        if base_y < -SUPPORT_TOLERANCE:
            return False
        bounds = np.asarray(mesh.bounds, dtype=float)
        x0, x1 = float(bounds[0, 0]), float(bounds[1, 0])
        z0, z1 = float(bounds[0, 2]), float(bounds[1, 2])
        for _block_id, sx0, sx1, sz0, sz1, top_y in self._support_surfaces:
            gap = base_y - top_y
            if gap < -SUPPORT_TOLERANCE or gap > SUPPORT_TOLERANCE:
                continue
            overlap_x = min(x1, sx1) - max(x0, sx0)
            overlap_z = min(z1, sz1) - max(z0, sz0)
            if overlap_x > SUPPORT_OVERLAP and overlap_z > SUPPORT_OVERLAP:
                return True
        return False

    def _remember_support(self, block: Block):
        """Record the top face used by later support checks."""
        if block.mesh is None:
            return
        bounds = np.asarray(block.mesh.bounds, dtype=float)
        self._support_surfaces.append(
            (
                int(block.block_id),
                float(bounds[0, 0]),
                float(bounds[1, 0]),
                float(bounds[0, 2]),
                float(bounds[1, 2]),
                float(bounds[1, 1]),
            )
        )
        center = bounds.mean(axis=0)
        self.occupied_cells.add(
            (
                int(np.floor(center[0])),
                int(np.floor(float(bounds[1, 1]) - 1e-9)),
                int(np.floor(center[2])),
            )
        )

    def get_block_by_id(self, block_id: int) -> Optional[Block]:
        """Retrieves a block by its unique ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def get_all_meshes(self) -> List[trimesh.Trimesh]:
        """Returns a list of all meshes in the scene."""
        return [block.mesh for block in self.blocks if block.mesh is not None]

    def check_collision(self, position: np.ndarray, dimensions: np.ndarray) -> bool:
        """Check if a block with given position and dimensions would collide with existing blocks.
        
        Args:
            position: Center position of the block to check
            dimensions: Dimensions (width, height, depth) of the block to check
            
        Returns:
            True if collision detected, False otherwise
        """
        if self.collision_manager is None:
            return False
        # Create a test block mesh
        test_mesh = create_block_mesh(position, dimensions)
        
        # Check against collision manager
        try:
            return self.collision_manager.in_collision_single(test_mesh)
        except Exception as e:
            print(f"Error in collision check: {e}")
            return False

    def clear(self, use_collision: Optional[bool] = None, reset_checks: bool = True):
        """Removes all blocks from the scene and resets the collision manager."""
        previous_overlap = self.enable_strict_overlap_prevention
        previous_support = self.support_check_mode
        self.blocks = []
        self._next_block_id = 0
        # Reset the collision manager; keep previous state if not specified
        if use_collision is None:
            use_collision = self.collision_manager is not None
        self.collision_manager = _create_collision_manager(use_collision)
        collision_enabled = self.collision_manager is not None
        if reset_checks:
            self.enable_strict_overlap_prevention = collision_enabled
            self.support_check_mode = "HYBRID" if collision_enabled else "OFF"
        else:
            self.enable_strict_overlap_prevention = previous_overlap if collision_enabled else False
            self.support_check_mode = previous_support
        self.occupied_cells = set()
        self._support_surfaces = []
        print("Scene cleared.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the scene state to a dictionary."""
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "next_block_id": self._next_block_id
            # Add other scene properties here if needed (e.g., generation params)
        }

    def from_dict(self, data: Dict[str, Any]):
        """Reconstructs the scene state from a dictionary."""
        self.clear(reset_checks=False) # Start with an empty scene (clears blocks and collision manager)
        block_data_list = data.get("blocks", [])
        # self.blocks = [] # Already cleared by self.clear()
        max_id = -1
        # No longer need loaded_coords

        for block_data in block_data_list:
            # Convert position list back to numpy array
            position = np.array(block_data.get("position", [0, 0, 0]))
            # Convert color list back to tuple
            color = tuple(block_data.get("color", (200, 200, 200, 255)))
            block_id = block_data.get("block_id", -1)
            # Load type, default to CUBE, handle potential old SLOPE_X_POS type name
            block_type_loaded = block_data.get("block_type", CUBE)
            if block_type_loaded == "SLOPE_X_POS": # Handle old name from saved files
                block_type = WEDGE
                print("Info: Loaded old block type 'SLOPE_X_POS', converting to 'WEDGE'.")
            elif block_type_loaded not in [CUBE, WEDGE, CYLINDER]: # Add new types here
                 print(f"Warning: Loaded unknown block type '{block_type_loaded}', defaulting to CUBE.")
                 block_type = CUBE
            else:
                 block_type = block_type_loaded

            raw_parameters = block_data.get("parameters", {})
            parameters = dict(raw_parameters) if isinstance(raw_parameters, dict) else {}
            # Load dimensions, default to [1,1,1]
            dimensions = np.array(block_data.get("dimensions", [1.0, 1.0, 1.0]))

            # Create block (mesh will be generated in __post_init__)
            # Create block (mesh will be generated in __post_init__)
            try:
                block = Block(position=position, block_type=block_type, color=color, dimensions=dimensions, parameters=parameters, block_id=block_id)
                self.blocks.append(block)
                max_id = max(max_id, block_id)

                # --- Add Mesh to Collision Manager ---
                if block.mesh and self.collision_manager is not None:
                    try:
                        self.collision_manager.add_object(name=f'block_{block.block_id}', mesh=block.mesh)
                    except Exception as e:
                         print(f"Warning: Failed to add mesh for loaded block {block.block_id} to collision manager: {e}")
                elif block.mesh is None:
                     print(f"Warning: Loaded block {block.block_id} has no mesh, cannot add to collision manager.")
                # --- End Add Mesh ---

            except Exception as e:
                 print(f"Error creating loaded block object for ID {block_id}: {e}. Skipping this block.")
                 continue # Skip to the next block in the file

            self._remember_support(block)

        # Restore the next block ID counter
        self._next_block_id = data.get("next_block_id", max_id + 1)

        # Optional: Post-load collision check (Task 2.3)
        try:
            if self.collision_manager is not None and self.collision_manager.in_collision_internal():
                print("Warning: Loaded scene contains overlapping blocks according to collision manager.")
                # Could potentially iterate through self.collision_manager.manage.items()
                # and check collisions individually to report specific overlapping block IDs.
        except Exception as e:
             print(f"Warning: Error during post-load collision check: {e}")

        print(f"Scene loaded from dict. Blocks: {len(self.blocks)}, Next ID: {self._next_block_id}")
