"""Walls are coursed human-scale stones, not monolithic slabs."""
from collections import Counter
from types import SimpleNamespace
import unittest

from core.architectural_geometry import extrude_profile, mesh_parameters
from core.architecture import BuildDocument, BuildRequest, build, rebuild
from core.architecture.masonry import (
    MAX_COURSE,
    MAX_STRETCHER,
    AshlarStone,
    course_profile_mesh,
    course_profile_tiles,
    iter_ashlar,
    iter_masonry_solid,
    iter_profile_courses,
    placement_is_masonry_cube,
    placement_is_oversized,
    radial_divisions,
    running_bond_holds,
    stone_is_human_scale,
)


def _inside(placements, x, y, z):
    for placement in placements:
        cx, cy, cz = placement.position
        dx, dy, dz = placement.dimensions
        if abs(x - cx) < dx / 2 - 1e-6 and abs(y - cy) < dy / 2 - 1e-6 and abs(z - cz) < dz / 2 - 1e-6:
            return placement
    return None


class AshlarEngineTests(unittest.TestCase):
    def test_a_plain_wall_is_running_bond_human_scale_stones(self):
        stones = iter_ashlar(length=8, height=3.2, thickness=0.3, axis="x")
        self.assertGreater(len(stones), 20)
        self.assertGreater(max(stone.course for stone in stones), 2)
        self.assertTrue(all(stone_is_human_scale(stone) for stone in stones))
        self.assertTrue(running_bond_holds(stones))

    def test_openings_are_voids_with_sill_and_lintel_stones(self):
        openings = [
            {"id": "door.000", "kind": "door", "center": -2.0, "width": 1.2, "height": 2.2, "sill": 0.0},
            {"id": "window.000", "kind": "window", "center": 2.0, "width": 1.4, "height": 1.2, "sill": 0.8},
        ]
        stones = iter_ashlar(length=8, height=3.2, thickness=0.3, axis="x", openings=openings)
        units = Counter(stone.unit for stone in stones)
        self.assertGreaterEqual(units["lintel"], 2)
        self.assertGreaterEqual(units["sill"], 1)

        def hit(x, y):
            for stone in stones:
                cx, cy, cz = stone.position
                dx, dy, dz = stone.dimensions
                if abs(x - cx) < dx / 2 - 1e-6 and abs(y - cy) < dy / 2 - 1e-6 and abs(cz) < dz / 2 - 1e-6:
                    return stone
            return None

        self.assertIsNone(hit(-2.0, 1.1))
        self.assertIsNone(hit(2.0, 1.4))
        self.assertIsNotNone(hit(2.0, 0.4))

    def test_a_thick_pier_is_several_stones_in_plan(self):
        stones = iter_masonry_solid((0, 2.5, 0), (0.6, 5.0, 3.2))
        self.assertGreater(len(stones), 8)
        self.assertTrue(all(stone_is_human_scale(stone) for stone in stones))

    def test_radial_courses_stagger_like_running_bond(self):
        even, odd = {}, {}
        for theta0, theta1, y0, y1, course, index in radial_divisions(
                0.0, 1.0, 0.0, 2.4, radius=4.0, detail="standard", facets=4):
            bucket = even if course % 2 == 0 else odd
            bucket.setdefault(course, []).append((theta0, theta1))
        self.assertGreater(len(even), 0)
        self.assertGreater(len(odd), 0)
        even_spans = next(iter(even.values()))
        odd_spans = next(iter(odd.values()))
        self.assertNotEqual(even_spans, odd_spans)
        self.assertAlmostEqual(even_spans[0][0], 0.0)
        self.assertAlmostEqual(odd_spans[0][0], 0.0)
        self.assertLess(odd_spans[0][1] - odd_spans[0][0], even_spans[0][1] - even_spans[0][0])


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _assert_simple_ring(test, ring):
    test.assertGreaterEqual(len(ring), 3)
    test.assertNotEqual(ring[0], ring[-1])
    test.assertEqual(len(ring), len(set(ring)))
    area = 0.0
    for index, point in enumerate(ring):
        previous = ring[index - 1]
        following = ring[(index + 1) % len(ring)]
        ab = (point[0] - previous[0], point[1] - previous[1])
        bc = (following[0] - point[0], following[1] - point[1])
        test.assertGreater(abs(_cross2(ab, bc)), 1e-6, f"collinear vertex {point} in {ring}")
        test.assertGreater((ab[0] ** 2 + ab[1] ** 2), 0.0)
        area += _cross2(point, following)
    test.assertGreater(area, 0.0)


def _mesh_placement(mesh, course, unit="profile_course", construction="ashlar stack"):
    position, dimensions, payload = mesh_parameters(mesh)
    return SimpleNamespace(
        id=f"course.{course:03d}",
        kind="wall",
        block_type="MESH",
        role="wall_spandrel",
        position=position,
        dimensions=dimensions,
        parameters={**payload, "unit": unit, "course": course, "construction": construction},
    )


def _cross2_area(ring):
    area = 0.0
    for index, point in enumerate(ring):
        other = ring[(index + 1) % len(ring)]
        area += _cross2(point, other)
    return 0.5 * area


class ProfileCourseTests(unittest.TestCase):
    def test_tall_rectangular_plate_splits_into_max_course_bands(self):
        plate = [(0.0, 0.0), (2.0, 0.0), (2.0, 5.0), (0.0, 5.0)]
        bands = iter_profile_courses(plate, 0.4, detail="standard")
        self.assertGreater(len(bands), 4)
        heights = [y1 - y0 for _polygon, y0, y1, _course in bands]
        self.assertTrue(all(height <= MAX_COURSE + 1e-6 for height in heights))
        self.assertAlmostEqual(sum(heights), 5.0, places=6)
        self.assertAlmostEqual(bands[0][1], 0.0, places=6)
        self.assertAlmostEqual(bands[-1][2], 5.0, places=6)
        for previous, current in zip(bands, bands[1:]):
            self.assertAlmostEqual(previous[2], current[1], places=6)
            self.assertEqual(current[3], previous[3] + 1)
        for polygon, y0, y1, _course in bands:
            _assert_simple_ring(self, polygon)
            self.assertTrue(all(y0 - 1e-6 <= point[1] <= y1 + 1e-6 for point in polygon))
            extrude_profile(polygon, 0.4)

    def test_right_triangle_and_pointed_gable_clip_without_empty_or_inverted_bands(self):
        triangle = [(0.0, 0.0), (2.0, 0.0), (0.0, 5.0)]
        gable = [(-1.0, 0.0), (1.0, 0.0), (1.0, 3.0), (0.0, 5.0), (-1.0, 3.0)]
        for polygon in (triangle, gable):
            with self.subTest(polygon=polygon):
                bands = iter_profile_courses(polygon, 0.35, detail="standard")
                self.assertGreater(len(bands), 3)
                y_lo = min(point[1] for point in polygon)
                y_hi = max(point[1] for point in polygon)
                self.assertAlmostEqual(bands[0][1], y_lo, places=6)
                self.assertAlmostEqual(bands[-1][2], y_hi, places=6)
                for ring, y0, y1, _course in bands:
                    self.assertGreater(y1, y0)
                    self.assertLessEqual(y1 - y0, MAX_COURSE + 1e-6)
                    _assert_simple_ring(self, ring)
                    self.assertGreater(_cross2_area(ring), 0.0)
                    ys = [point[1] for point in ring]
                    self.assertGreaterEqual(min(ys), y0 - 1e-6)
                    self.assertLessEqual(max(ys), y1 + 1e-6)
                    mesh = extrude_profile(ring, 0.35)
                    self.assertGreater(mesh.volume, 0.0)
                    self.assertLessEqual(mesh.bounds[1, 1] - mesh.bounds[0, 1], MAX_COURSE + 1e-6)

    def test_duplicate_and_collinear_vertices_are_not_emitted(self):
        dirty = [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 5.0), (2.0, 5.0),
            (0.0, 5.0), (0.0, 2.5), (0.0, 0.0),
        ]
        bands = iter_profile_courses(dirty, 0.3)
        self.assertGreater(len(bands), 1)
        for ring, _y0, _y1, _course in bands:
            _assert_simple_ring(self, ring)
        bottom = bands[0][0]
        self.assertEqual(len(bottom), 4)

    def test_emitted_meshes_are_human_scale_on_aabb_height(self):
        plate = [(0.0, 0.0), (2.0, 0.0), (2.0, 5.0), (0.0, 5.0)]
        depth = 0.25
        bands = iter_profile_courses(plate, depth, detail="standard")
        meshes = course_profile_mesh(plate, depth, "standard")
        self.assertGreater(len(meshes), len(bands))
        self.assertGreater(len(meshes), 4)
        for mesh, course in meshes:
            height = float(mesh.bounds[1, 1] - mesh.bounds[0, 1])
            along = max(
                float(mesh.bounds[1, 0] - mesh.bounds[0, 0]),
                float(mesh.bounds[1, 2] - mesh.bounds[0, 2]),
            )
            self.assertAlmostEqual(float(mesh.bounds[1, 2] - mesh.bounds[0, 2]), depth, places=5)
            self.assertLessEqual(height, MAX_COURSE + 1e-6)
            self.assertLessEqual(along, MAX_STRETCHER + 1e-6)
            self.assertGreater(mesh.volume, 0.0)
            position, dimensions, _payload = mesh_parameters(mesh)
            stone = AshlarStone(position, dimensions, course, 0, "profile_course", "x")
            self.assertTrue(stone_is_human_scale(stone))
            placement = _mesh_placement(mesh, course)
            self.assertTrue(placement_is_masonry_cube(placement))
            self.assertFalse(placement_is_oversized(placement))
            self.assertAlmostEqual(placement.dimensions[1], height, places=6)
        monolithic = SimpleNamespace(
            id="slab", kind="wall", block_type="MESH", role="wall_spandrel",
            dimensions=(2.0, 5.0, 0.4),
            parameters={"unit": "profile_course", "construction": "ashlar stack"},
        )
        self.assertTrue(placement_is_masonry_cube(monolithic))
        self.assertTrue(placement_is_oversized(monolithic))
        bare_mesh = SimpleNamespace(
            id="carve", kind="wall", block_type="MESH", role="wall_spandrel",
            dimensions=(2.0, 5.0, 0.4), parameters={},
        )
        self.assertFalse(placement_is_masonry_cube(bare_mesh))
        ashlar_mesh = SimpleNamespace(
            id="flagged", kind="wall", block_type="MESH", role="rose_surround",
            dimensions=(1.0, 0.4, 0.3), parameters={"construction": "ashlar stack"},
        )
        self.assertTrue(placement_is_masonry_cube(ashlar_mesh))

    def test_profile_tiles_keep_extruded_aabb_under_human_scale(self):
        gable = [(-4.0, 0.0), (0.0, 1.7), (0.0, 2.0), (-4.0, 0.3)]
        tiles = course_profile_tiles(gable, 6.0, detail="standard")
        self.assertGreater(len(tiles), 8)
        for mesh, _course, _tile in tiles:
            height = float(mesh.bounds[1, 1] - mesh.bounds[0, 1])
            along = max(
                float(mesh.bounds[1, 0] - mesh.bounds[0, 0]),
                float(mesh.bounds[1, 2] - mesh.bounds[0, 2]),
            )
            self.assertLessEqual(height, MAX_COURSE + 1e-6)
            self.assertLessEqual(along, MAX_STRETCHER + 1e-6)
            self.assertGreater(mesh.volume, 0.0)
            placement = _mesh_placement(mesh, 0)
            self.assertTrue(placement_is_masonry_cube(placement))
            self.assertFalse(placement_is_oversized(placement))


class BuiltWallTests(unittest.TestCase):
    def test_wall_product_courses_ashlar_and_keeps_opening_voids(self):
        result = build(BuildRequest(product="wall", openings="mixed", detail="presentation"))
        self.assertEqual(result.quality.status, "pass",
                         [c.to_dict() for c in result.quality.checks if c.status == "fail"])
        roles = result.quality.metrics["role_counts"]
        self.assertGreaterEqual(roles.get("envelope.wall.ashlar", 0), 8)
        self.assertGreaterEqual(roles.get("envelope.wall.header", 0), 2)
        self.assertGreaterEqual(roles.get("envelope.wall.sill", 0), 1)
        self.assertTrue(all(
            p.dimensions[1] <= MAX_COURSE + 1e-6
            or p.role.endswith(".header")
            or str(p.parameters.get("unit")) in {"lintel", "profile_course", "radial_ashlar"}
            for p in result.document.placements
        ))
        self.assertIsNone(_inside(result.document.placements, -2.0, 1.1, 0.0))

    def test_a_monolithic_wall_slab_fails_human_scale(self):
        result = build(BuildRequest(product="wall", openings="none"))
        document = BuildDocument.from_dict(result.document.to_dict())
        document.placements = document.placements[:1]
        document.placements[0].dimensions = (8.0, 3.2, 0.3)
        document.placements[0].parameters = {
            **document.placements[0].parameters,
            "unit": "ashlar",
            "construction": "ashlar running bond",
        }
        broken = rebuild(document)
        checks = {check.code: check.status for check in broken.quality.checks}
        self.assertEqual(checks["craft.human_scale_blocks"], "fail")
        self.assertEqual(broken.quality.status, "fail")

    def test_stoa_and_room_shell_walls_are_stones(self):
        stoa = build(BuildRequest(product="classical-stoa", seed=73))
        shell = build(BuildRequest(product="room-shell", openings="mixed", seed=73))
        for result, role_prefix in ((stoa, "rear_wall"), (shell, "space.boundary")):
            with self.subTest(product=result.document.request.product):
                self.assertEqual(result.quality.status, "pass",
                                 [c.to_dict() for c in result.quality.checks if c.status == "fail"])
                walls = [p for p in result.document.placements
                         if p.role == role_prefix or p.role.startswith(role_prefix)]
                self.assertGreater(len(walls), 4)
                self.assertTrue(all(
                    p.dimensions[1] <= MAX_COURSE + 1e-6 or str(p.parameters.get("unit")) == "lintel"
                    or p.role.endswith(".header")
                    for p in walls if p.block_type == "CUBE"
                ))
                self.assertTrue(all(
                    max(p.dimensions[0], p.dimensions[2]) <= MAX_STRETCHER + 0.01
                    or str(p.parameters.get("unit")) == "lintel"
                    or p.role.endswith(".header")
                    for p in walls if p.block_type == "CUBE"
                ))
        floors = [p for p in shell.document.placements if p.role == "space.floor"]
        self.assertGreater(len(floors), 1)
        self.assertTrue(all(p.block_type == "CUBE" for p in floors))
        self.assertEqual([p.id for p in floors if placement_is_oversized(p)], [])

    def test_temple_cella_is_coursed_with_a_real_doorway(self):
        result = build(BuildRequest(product="classical-building", variant="doric", seed=73))
        self.assertEqual(result.quality.status, "pass",
                         [c.to_dict() for c in result.quality.checks if c.status == "fail"])
        walls = [p for p in result.document.placements if p.role == "cella_wall"]
        lintels = [p for p in result.document.placements if p.role == "cella_lintel"]
        self.assertGreater(len(walls), 8)
        self.assertTrue(lintels)
        self.assertTrue(all(p.dimensions[1] <= MAX_COURSE + 1e-6 for p in walls))
        door = (
            sum(p.position[0] for p in lintels) / len(lintels),
            min(p.position[1] - p.dimensions[1] / 2 for p in lintels) - 0.1,
            lintels[0].position[2],
        )
        self.assertIsNone(_inside(walls, *door))

    def test_construction_report_names_ashlar_masonry(self):
        from core.architecture.construction import construction_report
        result = build(BuildRequest(product="wall", openings="door"))
        report = construction_report(result.document, result.quality)
        self.assertIn("ashlar-masonry", report["traditions"])
        self.assertGreater(report["masonry"]["course_count"], 1)
        self.assertEqual(report["masonry"]["bond"], "running")


BUILTIN_PRODUCTS = (
    "wall", "slab", "column", "beam", "stair", "ramp", "roof", "site-pad",
    "platform", "bridge", "room-shell", "arcade", "structural-grid", "facade",
    "spire-cluster",
)


def _oversized_masonry(placements):
    return [p.id for p in placements if placement_is_masonry_cube(p) and placement_is_oversized(p)]


class BuiltInMasonryTests(unittest.TestCase):
    def test_catalog_products_have_no_masonry_megaliths(self):
        variants = {
            "wall": "x",
            "slab": "floor",
            "column": "square",
            "beam": "x",
            "stair": "+z",
            "ramp": "+z",
            "roof": "gable",
            "site-pad": "rectangular",
            "platform": "supported",
            "bridge": "deck",
            "room-shell": "rectangular",
            "arcade": "roman",
            "structural-grid": "slab-frame",
            "facade": "bayed",
            "spire-cluster": "connected",
        }
        for name in BUILTIN_PRODUCTS:
            with self.subTest(product=name):
                kwargs = {"product": name, "seed": 73, "detail": "standard"}
                if name in variants:
                    kwargs["variant"] = variants[name]
                if name == "wall":
                    kwargs["openings"] = "arch"
                elif name == "facade":
                    kwargs["openings"] = "arch"
                elif name == "room-shell":
                    kwargs["openings"] = "mixed"
                result = build(BuildRequest(**kwargs))
                self.assertEqual(result.quality.status, "pass",
                                 [c.to_dict() for c in result.quality.checks if c.status == "fail"])
                leftover = _oversized_masonry(result.document.placements)
                self.assertEqual(leftover, [])
                self.assertFalse(any(p.block_type == "ARCH" for p in result.document.placements))

    def test_roof_gable_and_dome_are_coursed_not_one_mesh(self):
        gable = build(BuildRequest(product="roof", variant="gable", seed=3))
        dome = build(BuildRequest(product="roof", variant="dome", seed=3))
        for result, role in ((gable, "roof.gable_plane"), (dome, "roof.dome")):
            with self.subTest(role=role):
                parts = [p for p in result.document.placements if p.role == role]
                self.assertGreater(len(parts), 4)
                self.assertTrue(all(p.block_type == "MESH" for p in parts))
                self.assertEqual(_oversized_masonry(parts), [])
                self.assertFalse(any(p.block_type in {"WEDGE", "DOME"} for p in result.document.placements))
                extents = [max(p.dimensions) for p in parts]
                self.assertLess(max(extents), 8.0)

    def test_arcade_arches_are_coursed_and_keep_the_bay_void(self):
        result = build(BuildRequest(product="arcade", seed=5, bays=4, detail="standard"))
        self.assertEqual(result.quality.status, "pass")
        self.assertFalse(any(p.block_type == "ARCH" for p in result.document.placements))
        arches = [p for p in result.document.placements if p.role == "facade.arcade_arch"]
        self.assertGreater(len(arches), 8)
        self.assertEqual(_oversized_masonry(arches), [])
        cubes = [p for p in result.document.placements if p.block_type == "CUBE"]
        self.assertIsNone(_inside(cubes, -4.36, 2.0, 0.0))

    def test_facade_arch_surrounds_are_not_leftover_plates(self):
        result = build(BuildRequest(
            product="facade", openings="arch", detail="presentation", seed=9, bays=4,
        ))
        self.assertEqual(result.quality.status, "pass")
        surrounds = [p for p in result.document.placements if p.role.endswith(".arch_surround")]
        self.assertGreater(len(surrounds), 4)
        self.assertFalse(any(p.block_type == "ARCH" for p in surrounds))
        self.assertEqual(_oversized_masonry(result.document.placements), [])

    def test_spire_towers_and_decks_are_human_scale_ashlar(self):
        result = build(BuildRequest(product="spire-cluster", seed=12, detail="presentation"))
        self.assertEqual(result.quality.status, "pass")
        towers = [p for p in result.document.placements if p.role == "spire.tower"]
        decks = [p for p in result.document.placements if p.role in {
            "circulation.spire_platform", "circulation.spire_bridge", "site.spire_pad",
        }]
        self.assertGreater(len(towers), 4)
        self.assertGreater(len(decks), 4)
        self.assertTrue(all(p.block_type == "CUBE" for p in towers))
        self.assertEqual(_oversized_masonry(towers + decks), [])

    def test_slab_beam_and_bridge_decks_course_when_masonry(self):
        slab = build(BuildRequest(product="slab", variant="floor", width=8, depth=6, height=0.3))
        beam = build(BuildRequest(product="beam", variant="x", width=8, height=0.5, depth=0.4))
        bridge = build(BuildRequest(product="bridge", variant="deck", width=8, depth=1.8, height=2.5))
        for result, role in (
            (slab, "horizontal.floor"),
            (beam, "structure.beam"),
            (bridge, "circulation.bridge_deck"),
        ):
            with self.subTest(role=role):
                parts = [p for p in result.document.placements if p.role == role]
                self.assertGreater(len(parts), 1)
                self.assertTrue(all(p.block_type == "CUBE" for p in parts))
                self.assertTrue(all(p.parameters.get("unit") == "ashlar" for p in parts))
                self.assertEqual(_oversized_masonry(parts), [])


if __name__ == "__main__":
    unittest.main()
