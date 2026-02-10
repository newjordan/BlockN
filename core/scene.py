import time # For benchmarking collision checks
from dataclasses import dataclass, field
import numpy as np
import trimesh
import trimesh.collision # Import CollisionManager
from typing import List, Optional, Dict, Any

# Import pyembree if available for faster raycasting, but don't require it
try:
    import pyembree
    _pyembree_available = True
except ImportError:
    _pyembree_available = False
    print("Info: pyembree not found. Raycasting for support checks may be slower.")


from .geometry import create_block_mesh, create_wedge_mesh, create_cylinder_mesh # Import geometry functions
from .materials import ColorTuple # Import color definitions

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
            # Use the updated create_wedge_mesh signature
            # Get orientation from parameters, default to '+X' if not specified
            orientation = self.parameters.get('orientation', '+X')
            self.mesh = create_wedge_mesh(self.position, dimensions=self.dimensions, orientation=orientation)
        elif self.block_type == CYLINDER:
            # Prioritize parameters for cylinder dimensions
            radius = self.parameters.get('radius', self.dimensions[0] / 2.0) # Default radius from X dim if not in params
            height = self.parameters.get('height', self.dimensions[1])       # Default height from Y dim if not in params
            # Pass parameters dict itself for other potential settings (like sections)
            self.mesh = create_cylinder_mesh(self.position, radius=radius, height=height, parameters=self.parameters)
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
            "parameters": self.parameters # Add parameters
        }


def _create_collision_manager(use_collision: bool) -> Optional[trimesh.collision.CollisionManager]:
    """Create a collision manager if available; otherwise gracefully disable collision."""
    if not use_collision:
        return None
    try:
        return trimesh.collision.CollisionManager()
    except Exception as exc:
        print(f"Info: Collision backend unavailable ({exc}). Running with use_collision=False.")
        return None


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
        # Grid-based tracking for support checks
        self.occupied_cells = set() # Stores (x, y, z) integer tuples

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
                # Wedge height is actual_dimensions[1], which is already block_height
                orientation = actual_parameters.get('orientation', '+X')
                new_mesh = create_wedge_mesh(position, dimensions=actual_dimensions, orientation=orientation)
            elif block_type == CYLINDER:
                # Use actual parameters and pre-calculated block_height
                radius = actual_parameters.get('radius', actual_dimensions[0] / 2.0)
                new_mesh = create_cylinder_mesh(position, radius=radius, height=block_height, parameters=actual_parameters)
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
            start_time = time.time() # Task 3.3 Benchmarking
            try:
                is_collision = self.collision_manager.in_collision_single(new_mesh)
            except Exception as e:
                 # Handle potential errors within the collision check itself
                 print(f"Error during collision check for block at {position}: {e}. Skipping block.")
                 return None
            check_duration = time.time() - start_time
            # Optional: More verbose logging for performance analysis
            # print(f"Collision check took: {check_duration:.6f}s")

            if is_collision:
                print(f"Warning: Overlap detected at {position} for {block_type}. Skipping block addition.")
                return None # Collision detected, do not add block

        # --- 2b. Check for Support (Based on Mode) ---
        if self.support_check_mode != "OFF":
            # Define constants for support check
            support_tolerance = 1e-5 # How close to ground or another block counts as supported
            min_y = new_mesh.bounds[0, 1] # Bottom Y coordinate of the new mesh

            # Check if block is close enough to the ground (Y=0)
            is_on_ground = abs(min_y) < support_tolerance

            if not is_on_ground:
                # --- Grid-Based Support Check ---
                # Calculate the base Y coordinate using the actual block_height determined earlier.
                base_y = position[1] - block_height / 2.0 # Use actual block_height
                # Use floor for X/Z grid index calculation
                check_x = int(np.floor(position[0]))
                # Calculate the integer grid cell containing the *base* face of the new block.
                base_cell_y = int(base_y - 1e-9) # Y index of the cell containing the bottom face
                # Use floor for Z grid index calculation
                check_z = int(np.floor(position[2]))

                # The support coordinate we need to find in occupied_cells is the one matching the base cell
                # of the new block. occupied_cells stores the top-surface cell index of existing blocks.
                support_coord_needed = (check_x, base_cell_y, check_z)

                # Check if the cell containing the base of the new block matches a cell
                # known to contain the top surface of an existing block.
                grid_check_passed = support_coord_needed in self.occupied_cells
                # print(f"DEBUG: Support Check for Block at {position} (BaseY={base_y:.3f}): Needs support in cell {support_coord_needed}. Found={grid_check_passed}") # DEBUG PRINT

                if not grid_check_passed:
                    # If the grid check fails, it means the cell where the base should be is not marked as occupied (by a top surface below).
                    # Check if the required support cell is below ground (Y=-1). If so, it's an invalid placement.
                    if base_cell_y < 0:
                         print(f"Warning: Block at {position} is attempting to place below ground level (base in cell {support_coord_needed}). Skipping.")
                    else:
                         # Only print floating warning if not below ground
                         print(f"Warning: Block at {position} is floating (Grid Check Failed: required support cell {support_coord_needed} not occupied). Skipping.")
                    return None
                # else:
                    # print(f"DEBUG: Grid check passed for block at {position}.") # DEBUG PRINT
                # --- End Grid-Based Support Check ---

                # --- Volumetric Support Check (Attempt if Mode is Hybrid and Grid Check Passed) ---
                # This check only runs if the grid check passed, providing a finer check using raycasting
                if self.support_check_mode == "HYBRID": # No need to check grid_check_passed again, we wouldn't be here if it failed
                    if self.collision_manager is None:
                        # Collision system disabled: silently skip volumetric support checks.
                        volumetric_check_passed = True
                    else:
                        volumetric_check_passed = False # Assume failure unless proven otherwise
                    try:
                        # Create a ray intersector using the collision manager's current state
                        # Note: This might be inefficient if called very frequently. Consider caching.
                        if self.collision_manager is None:
                             volumetric_check_passed = True
                        elif not self.collision_manager._objs:
                             # No objects in collision manager yet, volumetric check trivially fails (no support)
                             print(f"Debug: Volumetric check skipped for block at {position} - collision manager empty.")
                             # If grid check passed but CM is empty, something is wrong, treat as failure.
                             volumetric_check_passed = False
                        elif not _pyembree_available:
                             print(f"Warning: Volumetric support check skipped (pyembree not available). Relying on grid check result for block at {position}.")
                             volumetric_check_passed = True # Rely on grid check result (which was True)
                        else:
                            # Import RayMeshIntersector specifically from the pyembree backend path
                            from trimesh.ray.ray_pyembree import RayMeshIntersector
                            intersector = RayMeshIntersector(self.collision_manager)

                            # Define ray origin slightly above the center of the bottom face
                            bottom_center_x = new_mesh.bounds.mean(axis=0)[0]
                            bottom_center_z = new_mesh.bounds.mean(axis=0)[2]
                            # Start ray slightly above the bottom face to avoid self-intersection issues
                            ray_origins = np.array([[bottom_center_x, min_y + support_tolerance, bottom_center_z]])
                            ray_directions = np.array([[0, -1, 0]]) # Cast downwards

                            # Perform the ray intersection
                            locations, index_ray, index_tri = intersector.intersects_location(
                                ray_origins=ray_origins,
                                ray_directions=ray_directions
                            )

                            if len(locations) == 0:
                                # Volumetric check confirms no support below
                                print(f"Warning: Block at {position} is floating (Volumetric Check: no support found below). Skipping.")
                                volumetric_check_passed = False
                            else:
                                # Hits found, check if the closest hit is near enough
                                # Calculate distances from ray origin to hit locations
                                distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
                                min_distance = np.min(distances) # Find the closest hit distance

                                # If the closest hit is further than tolerance * 2 (relative to ray origin just above base), it's floating
                                if min_distance > support_tolerance * 2:
                                     print(f"Warning: Block at {position} is floating (Volumetric Check: support too far below: {min_distance:.4f} > {support_tolerance * 2:.4f}). Skipping.")
                                     volumetric_check_passed = False
                                else:
                                     # Volumetric check confirms support is close enough.
                                     # print(f"DEBUG: Volumetric check passed for block at {position} (Hit distance: {min_distance:.4f})") # DEBUG PRINT
                                     volumetric_check_passed = True

                    except ImportError as ie:
                         # Handle case where pyembree/other backend might be missing for the intersector
                         # The specific error might be "No module named 'pyembree'" or "No module named 'embreex'"
                         print(f"Warning: Volumetric support check failed due to missing backend ({ie}). Is 'pyembree' installed? Relying on grid check result for block at {position}.")
                         volumetric_check_passed = True # Rely on grid check result (which was True)
                    except Exception as e:
                         # Catch other potential errors during volumetric check
                         print(f"Warning: Error during volumetric support check for block at {position}: {e}. Relying on grid check result.")
                         volumetric_check_passed = True # Rely on grid check result (which was True)

                    # If volumetric check was performed and failed, return None
                    if not volumetric_check_passed:
                        return None
                    # --- End Volumetric Support Check ---

                # Else (mode is "GRID"): Proceed based on grid check only (which already passed).
                # Else (mode is "HYBRID" and volumetric check passed): Proceed.

            # Else (is_on_ground is True): Block is supported by the ground. Proceed.
        # --- END SUPPORT CHECK ---

        # --- 3. All Checks Passed - Create Block Object and Add to Scene ---
        # Calculate occupied coordinate based on the TOP surface cell using actual block_height
        top_y = position[1] + block_height / 2.0 # Use actual block_height
        # Store the integer cell containing the top surface (using floor for X/Z consistency)
        occupied_coord = (int(np.floor(position[0])), int(top_y - 1e-9), int(np.floor(position[2])))

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

        # --- 5. Update Occupied Grid and Finalize ---
        # Add the pre-calculated occupied coordinate (top-surface-based) to the set *after* all checks passed
        self.occupied_cells.add(occupied_coord)
        # print(f"Debug: Added block {block.block_id} occupying top-surface cell {occupied_coord}") # Optional debug

        self._next_block_id += 1 # Increment ID only on successful addition
        return block # Return the created block

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

    def clear(self, use_collision: Optional[bool] = None):
        """Removes all blocks from the scene and resets the collision manager."""
        self.blocks = []
        self._next_block_id = 0
        # Reset the collision manager; keep previous state if not specified
        if use_collision is None:
            use_collision = self.collision_manager is not None
        self.collision_manager = _create_collision_manager(use_collision)
        collision_enabled = self.collision_manager is not None
        # Reset flags to default
        self.enable_strict_overlap_prevention = collision_enabled
        self.support_check_mode = "HYBRID" if collision_enabled else "OFF" # Reset to default mode
        # Clear occupied grid cells
        self.occupied_cells = set()
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
        self.clear() # Start with an empty scene (clears blocks and collision manager)
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

            parameters = block_data.get("parameters", {}) # Load parameters, default to empty dict
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

            # --- Populate Occupied Cells for Loaded Block (Top-Surface-based, Floor X/Z) ---
            loaded_height = block.dimensions[1]
            if block.block_type == CYLINDER:
                loaded_height = block.parameters.get('height', block.dimensions[1])
            top_y_load = block.position[1] + loaded_height / 2.0
            # Use floor for X/Z grid index calculation
            occupied_coord_load = (int(np.floor(block.position[0])), int(top_y_load - 1e-9), int(np.floor(block.position[2])))
            self.occupied_cells.add(occupied_coord_load)
            # --- End Populate Occupied Cells ---

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
