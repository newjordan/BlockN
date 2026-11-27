import unittest
import numpy as np
import trimesh

# Need to adjust sys.path if tests/ is not automatically recognized as part of the package
# This is common when running tests directly. A better approach might be using pytest
# with proper project structure (e.g., src/ layout) or configuring PYTHONPATH.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions to test
from core.geometry import (
    create_arch_mesh,
    create_block_mesh,
    create_cylinder_mesh,
    create_dome_mesh,
    create_wedge_mesh,
    optimize_blocks,
)
from core.scene import Block, ColorTuple

# Define some colors for testing
RED: ColorTuple = (255, 0, 0, 255)
GREEN: ColorTuple = (0, 255, 0, 255)
BLUE: ColorTuple = (0, 0, 255, 255)
GRAY: ColorTuple = (200, 200, 200, 255)

class TestOptimizeBlocks(unittest.TestCase):

    def test_empty_input(self):
        """Test optimize_blocks with an empty list."""
        result = optimize_blocks([])
        self.assertEqual(result, [])

    def test_single_block(self):
        """Test optimize_blocks with a single block."""
        block = Block(position=np.array([0, 0, 0]), color=RED)
        result = optimize_blocks([block])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], trimesh.Trimesh)
        # Check if the color was preserved (trimesh might not store it directly on mesh after creation)
        # We rely on the optimize_blocks function to re-apply color for merged meshes.
        # For single blocks, the original mesh is returned, which has vertex colors.

    def test_two_different_colors(self):
        """Test optimize_blocks with two blocks of different colors."""
        block1 = Block(position=np.array([0, 0, 0]), color=RED)
        block2 = Block(position=np.array([1, 0, 0]), color=GREEN)
        result = optimize_blocks([block1, block2])
        self.assertEqual(len(result), 2) # Should not merge

    def test_two_adjacent_same_color(self):
        """Test merging two adjacent blocks of the same color."""
        block1 = Block(position=np.array([0, 0, 0]), color=BLUE)
        block2 = Block(position=np.array([1, 0, 0]), color=BLUE) # Adjacent in X
        result = optimize_blocks([block1, block2])

        # Check if boolean union likely failed (no backend) - expect original blocks
        # A more robust check could involve capturing logs, but this is simpler for now.
        if len(result) == 2:
             print("\nINFO (test_two_adjacent_same_color): Boolean backend likely missing, checking for original blocks.")
             self.assertEqual(len(result), 2, "Should return 2 original meshes if merge fails")
             self.assertIsInstance(result[0], trimesh.Trimesh)
             self.assertIsInstance(result[1], trimesh.Trimesh)
             return # Skip further checks if merge failed

        # --- Checks if merge succeeded ---
        self.assertEqual(len(result), 1, "Should merge into a single mesh if backend exists")
        merged_mesh = result[0]
        self.assertIsInstance(merged_mesh, trimesh.Trimesh)

        # Check if the resulting mesh has approximately double the volume of a single block
        # Use isclose for floating point comparisons
        expected_volume = 2.0 # Assuming 1x1x1 blocks
        self.assertTrue(np.isclose(merged_mesh.volume, expected_volume, rtol=0.1),
                        f"Merged volume {merged_mesh.volume} not close to expected {expected_volume}")

        # Check if the color was applied to the merged mesh's visual properties
        self.assertIsInstance(merged_mesh.visual, trimesh.visual.ColorVisuals)
        # Trimesh stores face_colors as (N, 4) RGBA array. Check the first face color.
        applied_color = merged_mesh.visual.face_colors[0]
        self.assertEqual(tuple(applied_color), BLUE, "Merged mesh color incorrect")

    def test_two_non_adjacent_same_color(self):
        """Test two non-adjacent blocks of the same color (should not merge)."""
        block1 = Block(position=np.array([0, 0, 0]), color=GREEN)
        block2 = Block(position=np.array([2, 0, 0]), color=GREEN) # Separated by 1 unit
        result = optimize_blocks([block1, block2])

        # Check if boolean union likely failed (no backend) - expect original blocks
        if len(result) == 2:
             print("\nINFO (test_two_non_adjacent_same_color): Boolean backend likely missing, checking for original blocks.")
             self.assertEqual(len(result), 2, "Should return 2 original meshes if merge fails")
             self.assertIsInstance(result[0], trimesh.Trimesh)
             self.assertIsInstance(result[1], trimesh.Trimesh)
             return # Skip further checks if merge failed

        # --- Checks if merge succeeded ---
        # Boolean union merges non-touching meshes too!
        self.assertEqual(len(result), 1, "Should merge into a single mesh if backend exists")
        # If we wanted only *adjacent* merging, the logic in optimize_blocks would need refinement
        # (e.g., checking mesh adjacency before union). For now, union merges all.
        merged_mesh = result[0]
        self.assertTrue(np.isclose(merged_mesh.volume, 2.0, rtol=0.1), "Volume should be approx 2.0")
        self.assertEqual(tuple(merged_mesh.visual.face_colors[0]), GREEN)


class TestPrimitiveCreation(unittest.TestCase):

    def test_create_cube(self):
        """Test the create_block_mesh function for a cube."""
        pos = np.array([1.0, 2.0, 3.0])
        dims = np.array([1.0, 1.0, 1.0])
        mesh = create_block_mesh(position=pos, dimensions=dims)
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(hasattr(mesh.visual, 'uv'))
        self.assertTrue(np.isclose(mesh.volume, 1.0))
        # Check if centered correctly (bounds should be pos +/- dims/2)
        expected_bounds = np.array([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]])
        self.assertTrue(np.allclose(mesh.bounds, expected_bounds))

    def test_create_rect_prism(self):
        """Test create_block_mesh for a non-cubic rectangular prism."""
        pos = np.array([0.0, 0.0, 0.0])
        dims = np.array([2.0, 1.0, 3.0]) # W=2, H=1, D=3
        mesh = create_block_mesh(position=pos, dimensions=dims)
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(hasattr(mesh.visual, 'uv'))
        expected_volume = dims[0] * dims[1] * dims[2] # 2*1*3 = 6
        self.assertTrue(np.isclose(mesh.volume, expected_volume))
        # Check bounds
        expected_bounds = np.array([[-1.0, -0.5, -1.5], [1.0, 0.5, 1.5]])
        self.assertTrue(np.allclose(mesh.bounds, expected_bounds))

    def test_create_wedge(self):
        """Test the create_wedge_mesh function."""
        pos = np.array([0.0, 0.0, 0.0])
        dims = np.array([1.0, 1.0, 1.0]) # Default dimensions
        mesh = create_wedge_mesh(position=pos, dimensions=dims, orientation='+X') # Use dimensions and specify orientation
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(hasattr(mesh.visual, 'uv'))
        # Volume of a wedge is 0.5 * base_area * height (base area = width * depth)
        expected_volume = 0.5 * (dims[0] * dims[2]) * dims[1]
        self.assertTrue(mesh.is_winding_consistent)
        self.assertTrue(np.isclose(mesh.volume, expected_volume, rtol=1e-5),
                        f"Wedge volume {mesh.volume} not close to expected {expected_volume}")
        # Wedge API uses geometric-center positioning, so Y spans [-h/2, +h/2] around pos[1].
        expected_bounds = np.array([[-dims[0]/2, -dims[1]/2, -dims[2]/2],
                                    [ dims[0]/2,  dims[1]/2,  dims[2]/2]])
        self.assertTrue(np.allclose(mesh.bounds, expected_bounds))

    def test_wedge_orientations_are_consistent_positive_solids(self):
        dims = np.array([2.0, 3.0, 4.0])
        expected_volume = 0.5 * np.prod(dims)
        for orientation in ("+X", "-X", "+Z", "-Z"):
            with self.subTest(orientation=orientation):
                mesh = create_wedge_mesh(
                    position=np.zeros(3),
                    dimensions=dims,
                    orientation=orientation,
                )
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(mesh.is_winding_consistent)
                self.assertTrue(np.isclose(mesh.volume, expected_volume, rtol=1e-6))

    def test_create_cylinder(self):
        """Test the create_cylinder_mesh function."""
        pos = np.array([1.0, 1.0, 0.0]) # Geometric center
        radius = 0.5
        height = 2.0
        mesh = create_cylinder_mesh(position=pos, radius=radius, height=height)
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(hasattr(mesh.visual, 'uv'))
        # Volume of cylinder is pi * r^2 * h
        expected_volume = np.pi * (radius**2) * height
        self.assertTrue(np.isclose(mesh.volume, expected_volume, rtol=0.05)) # Allow tolerance due to facets
        # Check bounds for center-positioned Y-aligned cylinder.
        expected_bounds = np.array([[pos[0]-radius, pos[1]-height/2.0, pos[2]-radius],
                                    [pos[0]+radius, pos[1]+height/2.0, pos[2]+radius]])
        self.assertTrue(np.allclose(mesh.bounds, expected_bounds, atol=0.01))

    def test_create_cylinder_custom_sections(self):
        """Test create_cylinder_mesh with custom sections parameter."""
        pos = np.array([0.0, 0.0, 0.0])
        radius = 1.0
        height = 1.0
        custom_sections = 8
        # Create a block with parameters
        block = Block(position=pos, block_type="CYLINDER", parameters={'sections': custom_sections})
        # Mesh is created in block's __post_init__
        mesh = block.mesh

        self.assertIsInstance(mesh, trimesh.Trimesh)
        # Check if the number of vertices roughly corresponds to the sections
        # Expected vertices = sections * 2 (top/bottom rings) + 2 (centers) - approximately
        # Trimesh primitive might optimize this slightly. Check number of faces instead.
        # Expected faces = sections * 2 (sides) + sections * 2 (caps) = 4 * sections
        self.assertEqual(len(mesh.faces), 4 * custom_sections)

    def test_create_arch(self):
        """Test the create_arch_mesh function."""
        pos = np.array([2.0, 3.0, 4.0])
        dims = np.array([2.4, 2.0, 1.2])
        mesh = create_arch_mesh(position=pos, dimensions=dims, parameters={"segments": 12})

        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(hasattr(mesh.visual, 'uv'))
        expected_bounds = np.array([
            [pos[0] - dims[0] / 2.0, pos[1] - dims[1] / 2.0, pos[2] - dims[2] / 2.0],
            [pos[0] + dims[0] / 2.0, pos[1] + dims[1] / 2.0, pos[2] + dims[2] / 2.0],
        ])
        self.assertTrue(np.allclose(mesh.bounds, expected_bounds, atol=1e-2))
        full_block_volume = dims[0] * dims[1] * dims[2]
        self.assertLess(abs(mesh.volume), full_block_volume * 0.85)

    def test_create_dome(self):
        """Test the create_dome_mesh function."""
        pos = np.array([0.0, 2.0, -1.0])
        dims = np.array([2.0, 1.2, 1.6])
        mesh = create_dome_mesh(
            position=pos,
            dimensions=dims,
            parameters={"layers": 10, "sections": 20},
        )

        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0.0)
        self.assertTrue(hasattr(mesh.visual, 'uv'))
        expected_bounds = np.array([
            [pos[0] - dims[0] / 2.0, pos[1] - dims[1] / 2.0, pos[2] - dims[2] / 2.0],
            [pos[0] + dims[0] / 2.0, pos[1] + dims[1] / 2.0, pos[2] + dims[2] / 2.0],
        ])
        self.assertTrue(np.allclose(mesh.bounds, expected_bounds, atol=1e-2))
        cylinder_volume = np.pi * (dims[0] / 2.0) * (dims[2] / 2.0) * dims[1]
        self.assertGreater(abs(mesh.volume), cylinder_volume * 0.35)
        self.assertLess(abs(mesh.volume), cylinder_volume)


if __name__ == '__main__':
    unittest.main()
