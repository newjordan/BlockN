"""Per-stone tone, weathering, real-scale UVs, and physics-ready exports."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from core.architecture import (
    ArchitectInfo,
    BuildDocument,
    BuildRequest,
    Placement,
    build,
    export_bundle,
    export_mesh_artifact,
)
from core.architecture.io import format_requirement
from core.architecture.substances import (
    BIOFILM_GREEN,
    FINISH_HUE_REACH,
    IRON_RED,
    LEAD_WHITE,
    SILVER_GREY,
    SOOT_BLACK,
    SUBSTANCES,
    VERDIGRIS,
    WEATHERING,
    finish_tone,
)
from core.architecture.surfaces import (
    TONE_COUNT,
    UV_SCALE_M,
    _shelter_fractions,
    _weather_vertices,
    absolute_vertex_colors,
    face_colors,
    face_shelter,
    tone_color,
    tone_index,
)
from trimesh.visual.color import linear_to_srgb, srgb_to_linear


def _glb_json(path: Path) -> dict:
    data = path.read_bytes()
    length, = struct.unpack_from("<I", data, 12)
    return json.loads(data[20:20 + length])


def _pair(substance: str, age: float) -> BuildDocument:
    """A sheltered course beside an open one, plus a tall pier for the grime band."""
    request = BuildRequest(product="surface-pair", seed=17, age=age)
    shared = dict(kind="structure", block_type="CUBE", material=(200, 200, 200, 255), role="course")
    return BuildDocument(
        architect=ArchitectInfo("surface-test", "1.0", "Surface test"),
        request=request,
        placements=[
            Placement(id="open", position=(0.0, 0.5, 0.0), dimensions=(2.0, 1.0, 2.0), substance=substance, **shared),
            Placement(id="lower", position=(6.0, 0.5, 0.0), dimensions=(2.0, 1.0, 2.0), substance=substance, **shared),
            Placement(id="eave", position=(6.0, 1.2, 0.0), dimensions=(2.4, 0.4, 2.4), substance=substance, role="eave", kind="roof", block_type="CUBE", material=(180, 180, 180, 255)),
            Placement(id="pier", position=(12.0, 1.5, 0.0), dimensions=(1.0, 3.0, 1.0), substance=substance, **shared),
        ],
    )


def _distance(color: np.ndarray, target) -> float:
    return float(np.linalg.norm(np.asarray(color, dtype=float) - np.asarray(target, dtype=float)))


def _upward_color(mesh) -> np.ndarray:
    colors = face_colors(mesh).astype(float)
    upward = np.asarray(mesh.face_normals)[:, 1] >= 0.5
    return colors[upward].mean(axis=0)


def _cube(name, position, dimensions, material, *, role="course", kind="structure", substance="limestone"):
    return Placement(
        id=name, position=position, dimensions=dimensions, substance=substance,
        kind=kind, block_type="CUBE", material=material, role=role,
    )


class WeatheringProfileTests(unittest.TestCase):
    def test_every_substance_has_one_profile_and_keeps_its_text(self):
        self.assertEqual(set(WEATHERING), set(SUBSTANCES))
        for substance_id, substance in SUBSTANCES.items():
            profile = WEATHERING[substance_id]
            self.assertEqual(profile.substance_id, substance_id)
            self.assertTrue(substance.weathering)
            self.assertGreater(profile.grime_height, 0.0)
            self.assertIn(profile.patina_on, {"open", "shelter"})

    def test_tones_are_deterministic_and_quantised_to_eight(self):
        ids = [f"stone-{index:03d}" for index in range(40)]
        first = [tone_index(11, member_id) for member_id in ids]
        second = [tone_index(11, member_id) for member_id in ids]
        self.assertEqual(first, second)
        self.assertTrue(set(first) <= set(range(TONE_COUNT)))
        self.assertGreater(len(set(first)), 1)
        self.assertEqual(tone_index(11, ids[0]), tone_index(11, ids[0]))
        limestone = SUBSTANCES["limestone"]
        colours = {tone_color(limestone, index) for index in range(TONE_COUNT)}
        self.assertLessEqual(len(colours), TONE_COUNT)
        self.assertEqual(tone_color(limestone, 0), limestone.color_low)
        self.assertEqual(tone_color(limestone, TONE_COUNT - 1), limestone.color_high)

    def test_age_zero_is_the_unweathered_tone(self):
        from core.architecture.compiler import compile_document

        fresh = compile_document(_pair("limestone", 0.0))
        again = compile_document(_pair("limestone", 0.0))
        for block, other in zip(fresh.scene.blocks, again.scene.blocks):
            colors = absolute_vertex_colors(block.mesh)
            other_colors = absolute_vertex_colors(other.mesh)
            np.testing.assert_array_equal(colors, other_colors)
            placement = next(
                item for item in fresh.document.placements
                if item.id == block.parameters["primitive_id"]
            )
            tone = finish_tone(
                SUBSTANCES["limestone"], placement.material[:3],
                tone_index(17, block.parameters["primitive_id"]),
            )
            factor = np.asarray(block.mesh.visual.material.baseColorFactor, dtype=float).reshape(-1)
            if float(factor[:3].max()) > 1.0:
                factor = factor / 255.0
            np.testing.assert_allclose(factor[:3], np.asarray(tone, dtype=float) / 255.0, atol=1e-6)
            raw = block.mesh.visual.vertex_attributes["color"]
            self.assertTrue(np.all(raw[:, :3] == 255))
            self.assertLessEqual(int(np.max(np.abs(colors[:, :3].astype(int) - np.asarray(tone)))), 1)

    def test_authored_alpha_stays_on_the_block_and_the_vertices(self):
        from core.architecture.compiler import compile_document

        document = _pair("window-glass", 0.35)
        document.placements[0].material = (0, 55, 14, 90)
        result = compile_document(document)
        pane = next(block for block in result.scene.blocks if block.parameters["primitive_id"] == "open")
        pier = next(block for block in result.scene.blocks if block.parameters["primitive_id"] == "pier")
        self.assertEqual(pane.color[3], 90)
        self.assertEqual(pier.color[3], 255)
        self.assertTrue(np.all(pane.mesh.visual.vertex_attributes["color"][:, 3] == 90))
        self.assertTrue(np.all(pier.mesh.visual.vertex_attributes["color"][:, 3] == 255))
        self.assertEqual(pane.mesh.visual.material.alphaMode, "BLEND")

    def test_exposure_weathers_limestone_sandstone_timber_copper_and_lead(self):
        limestone = WEATHERING["limestone"]
        tone = (220, 214, 198)
        exposed = _weather_vertices(tone, limestone, 1.0, np.array([1.0]), 0.0, np.array([2.0]))[0]
        sheltered = _weather_vertices(tone, limestone, 1.0, np.array([1.0]), 1.0, np.array([2.0]))[0]
        base = _weather_vertices(tone, limestone, 1.0, np.array([0.0]), 0.0, np.array([0.05]))[0]
        crown = _weather_vertices(tone, limestone, 1.0, np.array([0.0]), 0.0, np.array([2.0]))[0]
        self.assertLess(_distance(exposed, BIOFILM_GREEN), _distance(sheltered, BIOFILM_GREEN))
        self.assertLess(int(base.sum()), int(crown.sum()))

        sandstone = WEATHERING["sandstone"]
        sand_tone = (210, 180, 140)
        sand_fresh = _weather_vertices(sand_tone, sandstone, 0.0, np.array([1.0]), 0.0, np.array([2.0]))[0]
        sand_aged = _weather_vertices(sand_tone, sandstone, 1.0, np.array([1.0]), 0.0, np.array([2.0]))[0]
        self.assertLess(_distance(sand_aged, IRON_RED), _distance(sand_fresh, IRON_RED))

        oak = WEATHERING["oak"]
        oak_tone = (120, 78, 46)
        silvered = _weather_vertices(oak_tone, oak, 1.0, np.array([1.0]), 0.0, np.array([3.0]))[0]
        shaded = _weather_vertices(oak_tone, oak, 1.0, np.array([0.2]), 1.0, np.array([3.0]))[0]
        self.assertLess(_distance(silvered, SILVER_GREY), _distance(shaded, SILVER_GREY))
        self.assertGreater(int(silvered.sum()), int(np.sum(oak_tone)))

        copper = WEATHERING["copper-sheet"]
        copper_tone = (160, 96, 72)
        verdigris = _weather_vertices(copper_tone, copper, 1.0, np.array([0.2]), 1.0, np.array([4.0]))[0]
        open_copper = _weather_vertices(copper_tone, copper, 1.0, np.array([1.0]), 0.0, np.array([4.0]))[0]
        self.assertLess(_distance(verdigris, VERDIGRIS), _distance(open_copper, VERDIGRIS))

        lead = WEATHERING["lead-sheet"]
        lead_tone = (120, 122, 128)
        whitened = _weather_vertices(lead_tone, lead, 1.0, np.array([1.0]), 0.0, np.array([5.0]))[0]
        self.assertGreater(int(whitened.sum()), int(np.sum(lead_tone)))
        self.assertLess(_distance(whitened, LEAD_WHITE), _distance(lead_tone, LEAD_WHITE))

    def test_compiled_neighbours_share_a_seed_and_differ_where_exposed(self):
        from core.architecture.compiler import compile_document

        aged = compile_document(_pair("limestone", 0.8))
        again = compile_document(_pair("limestone", 0.8))
        by_id = {block.parameters["primitive_id"]: absolute_vertex_colors(block.mesh) for block in aged.scene.blocks}
        repeated = {block.parameters["primitive_id"]: absolute_vertex_colors(block.mesh) for block in again.scene.blocks}
        for primitive_id, colors in by_id.items():
            np.testing.assert_array_equal(colors, repeated[primitive_id])
        lo = np.array([[-1.0, 0.0, -1.0], [5.0, 0.0, -1.0], [4.8, 1.0, -1.2]], dtype=float)
        hi = np.array([[1.0, 1.0, 1.0], [7.0, 1.0, 1.0], [7.2, 1.4, 1.2]], dtype=float)
        shelter = _shelter_fractions(lo, hi)
        self.assertEqual(shelter[0], 0.0)
        self.assertEqual(shelter[1], 1.0)
        open_mesh = next(block.mesh for block in aged.scene.blocks if block.parameters["primitive_id"] == "open")
        lower_mesh = next(block.mesh for block in aged.scene.blocks if block.parameters["primitive_id"] == "lower")
        open_face = _upward_color(open_mesh)
        lower_face = _upward_color(lower_mesh)
        self.assertLess(_distance(open_face, BIOFILM_GREEN), _distance(lower_face, BIOFILM_GREEN))
        self.assertFalse(np.array_equal(open_face, lower_face))
        pier = next(block.mesh for block in aged.scene.blocks if block.parameters["primitive_id"] == "pier")
        colours = face_colors(pier)
        heights = pier.triangles_center[:, 1]
        self.assertLess(int(colours[np.argmin(heights)].sum()), int(colours[np.argmax(heights)].sum()))

    def test_box_uvs_use_metres(self):
        result = build(BuildRequest(product="slab", seed=3, width=4.0, height=0.25, depth=3.0, variant="floor", age=0.0))
        mesh = result.scene.blocks[0].mesh
        span = float(np.ptp(mesh.visual.uv, axis=0).max())
        self.assertGreater(span, 1.5)
        self.assertEqual(result.scene.export_metadata["uv_scale_m"], UV_SCALE_M)

    def test_pigmented_finishes_keep_the_authored_colour(self):
        lapis = (43, 92, 158)
        court = build(BuildRequest(
            product="iwan-court", detail="concept", seed=7, age=0.0,
            parameters={"finish": "sandstone-lapis"},
        ))
        cells = [
            block for block in court.scene.blocks
            if block.parameters.get("component_role") == "muqarnas_cell"
        ]
        self.assertTrue(cells)
        for block in cells:
            colour = np.asarray(block.color[:3], dtype=float)
            self.assertLess(_distance(colour, lapis), 0.06 * np.linalg.norm(lapis) + 2.0)
            self.assertGreater(colour[2], colour[0])
            self.assertGreater(colour[2], colour[1])

        ochre = (236, 198, 144)
        house = build(BuildRequest(
            product="timber-house", detail="concept", seed=7, age=0.0,
            parameters={"finish": "honey-clay"},
        ))
        infill = [
            block for block in house.scene.blocks
            if block.parameters.get("component_role") == "lime_infill"
        ]
        self.assertTrue(infill)
        for block in infill:
            colour = np.asarray(block.color[:3], dtype=float)
            self.assertLess(_distance(colour, ochre), 0.06 * np.linalg.norm(ochre) + 2.0)
            self.assertGreater(colour[0], colour[1])
            self.assertGreater(colour[1], colour[2])

    def test_buff_and_rose_brick_stay_apart_and_stylised_stone_stays_in_range(self):
        limestone = SUBSTANCES["limestone"]
        stylised = finish_tone(limestone, (0, 255, 65), 3)
        self.assertGreater(_distance((0, 255, 65), stylised), FINISH_HUE_REACH)
        for channel, low, high in zip(stylised, limestone.color_low, limestone.color_high):
            self.assertGreaterEqual(channel, min(low, high))
            self.assertLessEqual(channel, max(low, high))

        def brick_mean(finish):
            result = build(BuildRequest(
                product="iwan-court", detail="concept", seed=7, age=0.0,
                parameters={"finish": finish},
            ))
            colours = [
                block.color[:3] for block in result.scene.blocks
                if (block.parameters.get("surface") or {}).get("substance") == "fired-clay-brick"
            ]
            self.assertTrue(colours, finish)
            return np.mean(np.asarray(colours, dtype=float), axis=0)

        buff = brick_mean("buff-turquoise")
        rose = brick_mean("rose-azure")
        self.assertGreater(_distance(buff, rose), 8.0)
        self.assertGreater(rose[2], buff[2])

    def test_flush_wall_is_washed_and_cornice_and_soffit_are_sheltered(self):
        from core.architecture.compiler import compile_document

        stone = (200, 196, 184, 255)
        document = BuildDocument(
            architect=ArchitectInfo("surface-test", "1.0", "Surface test"),
            request=BuildRequest(product="surface-pair", seed=3, age=0.8),
            placements=[
                _cube("flush-wall", (0.0, 1.5, 0.0), (0.4, 1.0, 1.0), stone),
                _cube("flush-course", (0.0, 2.25, 0.0), (0.4, 0.5, 1.0), stone),
                _cube("cornice-wall", (4.0, 1.7, 0.0), (0.4, 0.4, 1.0), stone),
                _cube("cornice", (4.0, 2.05, 0.0), (0.9, 0.2, 1.3), stone, role="cornice", kind="roof"),
            ],
        )
        result = compile_document(document)
        by_id = {block.parameters["primitive_id"]: block.mesh for block in result.scene.blocks}

        def side_shelter(mesh, axis):
            normals = np.asarray(mesh.face_normals)
            shelter = face_shelter(mesh)
            selected = np.abs(normals[:, axis]) > 0.5
            self.assertTrue(np.any(selected))
            return float(np.mean(shelter[selected]))

        self.assertLess(side_shelter(by_id["flush-wall"], 0), 0.05)
        self.assertGreater(side_shelter(by_id["cornice-wall"], 0), 0.9)
        cornice = by_id["cornice"]
        soffit = np.asarray(cornice.face_normals)[:, 1] <= -0.5
        self.assertTrue(np.any(soffit))
        self.assertTrue(np.all(face_shelter(cornice)[soffit] == 1.0))

    def test_cathedral_walls_are_mostly_rain_washed(self):
        result = build(BuildRequest(product="gothic-cathedral", detail="concept", seed=7, age=0.8))
        washed = []
        shadowed = []
        wall_faces = 0
        soot_dominated = 0
        for block in result.scene.blocks:
            if block.parameters.get("kind") != "wall":
                continue
            substance = (block.parameters.get("surface") or {}).get("substance")
            if substance == "window-glass":
                continue
            mesh = block.mesh
            normals = np.asarray(mesh.face_normals)
            vertical = np.abs(normals[:, 1]) < 0.5
            if not np.any(vertical):
                continue
            shelter = face_shelter(mesh)[vertical]
            colours = face_colors(mesh)[vertical].astype(float)
            wall_faces += int(vertical.sum())
            soot_dominated += int(np.sum(shelter >= 0.5))
            washed.extend(colours[shelter <= 0.15].tolist())
            shadowed.extend(colours[shelter >= 0.85].tolist())
        self.assertGreater(wall_faces, 0)
        self.assertLess(soot_dominated / float(wall_faces), 0.5)
        self.assertTrue(washed)
        self.assertTrue(shadowed)
        washed_mean = np.mean(np.asarray(washed), axis=0)
        shadowed_mean = np.mean(np.asarray(shadowed), axis=0)
        self.assertGreater(float(washed_mean.sum()), float(shadowed_mean.sum()) + 8.0)
        self.assertLess(_distance(shadowed_mean, SOOT_BLACK), _distance(washed_mean, SOOT_BLACK))


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.result = build(BuildRequest(
            product="wall", seed=5, width=4.0, height=2.4, depth=0.4,
            openings="none", age=0.8,
        ))

    def test_gltf_materials_extras_and_physics_masses(self):
        with tempfile.TemporaryDirectory() as temporary:
            stem = Path(temporary) / "wall"
            paths = export_bundle(self.result, stem, formats=("glb",))
            payload = _glb_json(paths["glb"])
            self.assertTrue(payload["materials"])
            names = [material["name"] for material in payload["materials"]]
            by_substance: dict[str, set] = {}
            for name in names:
                substance_id = name.split(".tone", 1)[0]
                by_substance.setdefault(substance_id, set()).add(name)
            for group in by_substance.values():
                self.assertLessEqual(len(group), TONE_COUNT)
            for material in payload["materials"]:
                pbr = material["pbrMetallicRoughness"]
                self.assertIn("baseColorFactor", pbr)
                self.assertIn("roughnessFactor", pbr)
                self.assertIn("metallicFactor", pbr)
            scene_extras = payload["scenes"][0]["extras"]
            self.assertEqual(scene_extras["uv_scale_m"], UV_SCALE_M)
            self.assertEqual(scene_extras["total_mass_kg"], round(self.result.physical.total_mass_kg, 3))
            self.assertEqual(scene_extras["bill_of_materials"], self.result.physical.to_dict()["bill_of_materials"])
            nodes = {node["name"]: node["extras"] for node in payload["nodes"] if "mesh" in node}
            physical = {member.primitive_id: member for member in self.result.physical.members}
            for primitive_id, extras in nodes.items():
                member = physical[primitive_id]
                self.assertEqual(extras["substance"], member.substance)
                self.assertEqual(extras["mass_kg"], round(member.mass_kg, 3))
                self.assertEqual(extras["centroid"], [round(value, 6) for value in member.centroid_m])
                self.assertIn("density", extras)
                self.assertIn("role", extras)
                self.assertIn("kind", extras)
            self.assertTrue(any(
                "COLOR_0" in primitive["attributes"]
                for mesh in payload["meshes"] for primitive in mesh["primitives"]
            ))
            loaded = trimesh.load(paths["glb"], force="scene", process=False)
            self.assertEqual(loaded.metadata["total_mass_kg"], scene_extras["total_mass_kg"])
            self.assertEqual(set(loaded.geometry), set(nodes))
            weathered = absolute_vertex_colors(next(iter(loaded.geometry.values())))
            self.assertFalse(np.all(weathered[:, :3] == 255))
            self._assert_glb_linear(paths["glb"], self.result)

            physics = json.loads(paths["physics"].read_text(encoding="utf-8"))
            self.assertEqual(physics["total_mass_kg"], round(self.result.physical.total_mass_kg, 3))
            self.assertEqual(len(physics["members"]), len(self.result.physical.members))
            for entry, member in zip(physics["members"], self.result.physical.members):
                self.assertEqual(entry["id"], member.primitive_id)
                self.assertEqual(entry["substance"], member.substance)
                self.assertEqual(entry["mass_kg"], round(member.mass_kg, 3))
                self.assertEqual(entry["centroid"], [round(value, 6) for value in member.centroid_m])
                self.assertIsNotNone(entry["friction"])
                self.assertEqual(len(entry["bounds_min"]), 3)
                self.assertEqual(len(entry["bounds_max"]), 3)
                self.assertTrue(all(low <= high for low, high in zip(entry["bounds_min"], entry["bounds_max"])))

    def test_listed_formats_load_back_and_optional_ones_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mesh_format in ("glb", "gltf", "obj", "ply", "stl", "off"):
                path = export_mesh_artifact(self.result.scene, root / f"wall.{mesh_format}")
                self.assertGreater(path.stat().st_size, 0, mesh_format)
                loaded = trimesh.load(path, process=False, force="mesh")
                geometry = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
                vertices = sum(len(mesh.vertices) for mesh in geometry)
                self.assertGreater(vertices, 0, mesh_format)
            if (root / "wall.mtl").is_file():
                self.assertIn("mtllib", (root / "wall.obj").read_text(encoding="utf-8"))
            ply = trimesh.load(root / "wall.ply", process=False, force="mesh")
            self.assertGreater(len(np.unique(ply.visual.vertex_colors[:, :3], axis=0)), 1)

            for mesh_format in ("dae", "3mf", "usda", "usdc", "fbx"):
                requirement = format_requirement(mesh_format)
                path = root / f"wall.{mesh_format}"
                if requirement:
                    with self.assertRaisesRegex(ValueError, "needs"):
                        export_mesh_artifact(self.result.scene, path)
                    self.assertFalse(path.exists(), mesh_format)
                    continue
                export_mesh_artifact(self.result.scene, path)
                self.assertGreater(path.stat().st_size, 0, mesh_format)
                if mesh_format == "fbx":
                    continue
                if mesh_format in {"usda", "usdc"}:
                    self._assert_usd_linear(path)
                    continue
                loaded = trimesh.load(path, process=False, force="scene")
                geometry = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
                self.assertGreater(sum(len(mesh.vertices) for mesh in geometry), 0, mesh_format)
                if mesh_format == "dae":
                    self.assertTrue(any(
                        getattr(getattr(mesh, "visual", None), "material", None) is not None
                        for mesh in geometry
                    ))
                if mesh_format == "3mf":
                    self._assert_3mf_colors(path)

    def test_cli_age_and_format_list(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as temporary:
            stem = Path(temporary) / "slab"
            completed = subprocess.run(
                [sys.executable, "-m", "blockn", "make", "slab",
                 "--width", "4", "--height", "0.25", "--depth", "3", "--variant", "floor",
                 "--seed", "9", "--age", "0", "--format", "glb,off",
                 "--bundle", str(stem), "--json-summary"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["age"], 0.0)
            self.assertTrue((Path(f"{stem}.glb")).is_file())
            self.assertTrue((Path(f"{stem}.off")).is_file())
            self.assertTrue((Path(f"{stem}.physics.json")).is_file())

    def _assert_glb_linear(self, path, result):
        payload = _glb_json(path)
        data = path.read_bytes()
        bin_length, bin_type = struct.unpack_from("<II", data, 12 + 8 + struct.unpack_from("<I", data, 12)[0])
        json_length = struct.unpack_from("<I", data, 12)[0]
        blob = data[12 + 8 + json_length + 8:12 + 8 + json_length + 8 + bin_length]
        self.assertEqual(bin_type, 0x004E4942)
        by_name = {
            block.parameters["primitive_id"]: block
            for block in result.scene.blocks
            if block.parameters.get("primitive_id")
        }
        mesh_names = {}
        for node in payload.get("nodes", []):
            if "mesh" in node and node.get("name"):
                mesh_names.setdefault(node["mesh"], node["name"])
        checked = 0
        for mesh_index, mesh in enumerate(payload["meshes"]):
            block = by_name.get(mesh_names.get(mesh_index))
            if block is None:
                continue
            intended = face_colors(block.mesh).astype(int)
            if len(intended) == 0:
                continue
            for primitive in mesh["primitives"]:
                color_index = primitive["attributes"]["COLOR_0"]
                accessor = payload["accessors"][color_index]
                self.assertEqual(accessor["componentType"], 5126)
                self.assertNotIn("normalized", accessor)
                view = payload["bufferViews"][accessor["bufferView"]]
                start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
                count = int(accessor["count"])
                raw = np.frombuffer(blob[start:start + count * 16], dtype="<f4").reshape(count, 4)
                factor = np.asarray(
                    payload["materials"][primitive["material"]]["pbrMetallicRoughness"]["baseColorFactor"],
                    dtype=float,
                )
                memory = np.asarray(block.mesh.visual.material.baseColorFactor, dtype=float).reshape(-1)
                if float(memory[:3].max()) > 1.0:
                    memory = memory / 255.0
                expected_linear = srgb_to_linear(memory[:3])
                np.testing.assert_allclose(factor[:3], expected_linear, atol=1e-5)
                if float(np.mean(memory[:3])) > 0.2:
                    self.assertGreater(float(np.max(np.abs(factor[:3] - memory[:3]))), 0.02)
                decoded = np.rint(linear_to_srgb(np.clip(raw[:, :3] * factor[:3], 0.0, None)) * 255.0)
                decoded = np.clip(decoded, 0, 255)
                difference = np.abs(decoded[:, None, :] - intended[None, :, :]).max(axis=2)
                self.assertLessEqual(int(difference.min(axis=1).max()), 2)
                checked += 1
        self.assertGreater(checked, 0)
        self.assertEqual(bin_length, bin_length)

    def _assert_usd_linear(self, path):
        from pxr import Usd, UsdGeom, UsdShade

        stage = Usd.Stage.Open(str(path))
        meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        self.assertTrue(meshes)
        found_color = False
        found_material = False
        for prim in meshes:
            primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
            values = primvar.Get()
            if values:
                found_color = True
                self.assertTrue(all(0.0 <= channel <= 1.0 for color in values for channel in color))
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if bound:
                found_material = True
        self.assertTrue(found_color)
        self.assertTrue(found_material)

    def _assert_3mf_colors(self, path):
        import zipfile
        from lxml import etree

        with zipfile.ZipFile(path) as archive:
            model_name = next(name for name in archive.namelist() if name.lower().endswith("3dmodel.model"))
            root = etree.fromstring(archive.read(model_name))
        material = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
        colors = root.findall(f".//{{{material}}}color")
        self.assertTrue(colors)
        self.assertTrue(any(element.get("color", "").startswith("#") for element in colors))
        core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        triangles = root.findall(f".//{{{core}}}triangle")
        self.assertTrue(any(triangle.get("pid") and triangle.get("p1") for triangle in triangles))


if __name__ == "__main__":
    unittest.main()
