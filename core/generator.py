import numpy as np
import random
from typing import Optional

from core.scene import CUBE, Scene


def generate_podium(
    scene: Scene,
    width: int,
    depth: int,
    height: int,
    seed: Optional[int] = None,
    clear_scene: bool = True,
    base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
):
    """
    Generates a simple solid podium using CUBE blocks.
    Uses X=Width, Y=Height, Z=Depth convention.

    Args:
        scene: The Scene object to add blocks to.
        width: Dimension in the X axis.
        height: Dimension in the Y axis.
        depth: Dimension in the Z axis.
        seed: An optional integer seed for randomization (currently unused).
        clear_scene: If True, clears the scene before adding blocks.
        base_position: The bottom-corner position (min x, y, z) to start generation from.
    """
    seed_info = f" with seed {seed}" if seed is not None else " (no seed)"
    print(f"Generating Podium: Width(X)={width}, Height(Y)={height}, Depth(Z)={depth}{seed_info}")

    # Use the seed if provided (e.g., for potential color variations later)
    if seed is not None:
        random.seed(seed)

    # Clear previous blocks if requested
    if clear_scene:
        scene.clear()

    # Use a consistent color for the podium
    podium_color = (180, 180, 180, 255) # Light gray

    # Iterate using new convention: X=Width, Y=Height, Z=Depth
    for x in range(width):
        for z in range(depth): # Iterate Z for depth
            for y in range(height): # Iterate Y for height
                # Calculate corner position (min x, y, z of the unit cell)
                corner_pos = base_position + np.array([x, y, z])
                # Calculate geometric center position (assuming unit dimensions 1x1x1)
                center_pos = corner_pos + np.array([0.5, 0.5, 0.5])
                # Add CUBE blocks at calculated geometric center position.
                scene.add_block(center_pos, block_type=CUBE, color=podium_color, dimensions=np.array([1.0, 1.0, 1.0]))

    print("Podium generation complete.")


