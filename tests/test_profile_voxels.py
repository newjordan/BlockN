"""Player profiles, voxel grids and playability checks."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.architecture import BuildRequest, build
from core.architecture.gameplay import Body, _grid_checks, audit, headroom, walk
from core.architecture.profile import PRESETS, PlayerProfile, preset
from core.architecture.voxels import CHUNK, VOXEL_SCHEMA, VoxelGrid, minecraft_block, voxelize, winding_number


def _game(result):
    return {c.code: c for c in result.quality.checks if c.code.startswith("game.")}


class TestProfile(unittest.TestCase):
    def test_human_clearances(self):
        c = preset("human").clearances()
        self.assertAlmostEqual(c.door_width, 2 * 0.4 + 0.3)
        self.assertAlmostEqual(c.door_height, 1.8 + 0.3)
        self.assertAlmostEqual(c.corridor_width, 4 * 0.4 + 0.45)
        self.assertAlmostEqual(c.takeoff_speed, round(math.sqrt(2 * 20.0 * 2.4), 4))
        self.assertAlmostEqual(c.jump_reach, round(7.0 * 2 * math.sqrt(2 * 20.0 * 2.4) / 20.0, 4))

    def test_bigger_bodies_need_bigger_doors(self):
        human, giant = preset("human").clearances(), preset("giant").clearances()
        self.assertGreater(giant.door_width, human.door_width)
        self.assertGreater(giant.door_height, human.door_height)

    def test_overrides_and_validation(self):
        p = PlayerProfile.from_dict({"preset": "giant", "height": 3.0, "camera": {"mode": "first_person"}})
        self.assertEqual(p.height, 3.0)
        self.assertEqual(p.radius, PRESETS["giant"].radius)
        self.assertEqual(p.camera.mode, "first_person")
        self.assertEqual(PlayerProfile.from_dict(p.to_dict()), p)
        with self.assertRaises(ValueError):
            PlayerProfile.from_dict({"wingspan": 2})
        with self.assertRaises(ValueError):
            PlayerProfile.from_dict({"radius": 2.0, "height": 1.8})
        with self.assertRaises(ValueError):
            preset("dragon")


class TestRequestCompatibility(unittest.TestCase):
    # Fingerprints of these builds before profiles existed.
    KNOWN = {
        ("room-shell", 1, "door"): "027f739ca635a86303fa1dd38a43547075ae9d06a9bee104df099296d9b30d24",
        ("stair", 2, "auto"): "19ea6bc82259cccce4d1c8162984e1037e15595a163d8d6394458ce3f6e78ed2",
        ("spire-cluster", 3, "auto"): "a42164d5c563837e82d69d94545e8a6907e1b2fda30589875ca1a74434424148",
    }

    def test_requests_without_a_profile_keep_their_fingerprints(self):
        for (product, seed, openings), fingerprint in self.KNOWN.items():
            with self.subTest(product=product):
                result = build(BuildRequest(product=product, seed=seed, openings=openings))
                self.assertEqual(result.document.fingerprint(), fingerprint)
                self.assertNotIn("profile", result.document.request.to_dict())
                self.assertNotIn("ports", result.document.metadata)
                self.assertFalse(_game(result))

    def test_profile_roundtrips_through_the_request(self):
        request = BuildRequest.from_dict({"product": "room-shell", "profile": "brute"})
        data = request.to_dict()
        self.assertEqual(data["profile"]["name"], "brute")
        self.assertEqual(BuildRequest.from_dict(json.loads(json.dumps(data))).to_dict(), data)


class TestVoxelize(unittest.TestCase):
    def test_winding_number_of_a_cube(self):
        import trimesh

        cube = trimesh.creation.box(extents=(2, 2, 2))
        w = winding_number(np.array([[0, 0, 0], [0.9, 0.9, 0.9], [1.5, 0, 0], [0, 3, 0]]), cube.triangles)
        np.testing.assert_allclose(w, [1, 1, 0, 0], atol=1e-6)

    def test_thin_tilted_slab_stays_closed_at_coarse_pitch(self):
        # A roof course 0.2 m thick, pitched at 40 degrees, cut into 1 m
        # blocks: every column under it keeps at least one block, so the roof
        # does not break into a scatter.
        import trimesh
        from types import SimpleNamespace
        from core.architecture.voxels import voxelize_parts

        slab = trimesh.creation.box(extents=(6.0, 0.2, 4.0))
        slab.apply_transform(trimesh.transformations.rotation_matrix(math.radians(40), [0, 0, 1]))
        slab.apply_translation((0.0, 4.0, 0.0))
        grid = voxelize_parts([SimpleNamespace(mesh=slab)], [SimpleNamespace(substance="slate")], pitch=1.0)
        filled = grid.labels > 0
        cols = filled.any(axis=1)
        xs = grid.origin[0] + (np.arange(filled.shape[0]) + .5) * grid.pitch
        zs = grid.origin[2] + (np.arange(filled.shape[2]) + .5) * grid.pitch
        span = 6.0 * math.cos(math.radians(40)) / 2 - .5
        under = (np.abs(xs)[:, None] < span) & (np.abs(zs)[None, :] < 1.5)
        self.assertTrue(cols[under].all())

    def test_slab_fills_its_box(self):
        result = build(BuildRequest(product="slab", seed=1, width=4.0, height=0.5, depth=2.0))
        grid = voxelize(result.document, result.scene, pitch=0.25)
        self.assertEqual(int(grid.solid.sum()), 16 * 2 * 8)
        self.assertEqual([s % CHUNK for s in grid.shape], [0, 0, 0])
        self.assertEqual(grid.origin[1], 0.0)

    def test_round_column_volume(self):
        result = build(BuildRequest(product="column", seed=1, variant="round", width=1.0, height=2.0, depth=1.0))
        grid = voxelize(result.document, result.scene, pitch=0.1)
        volume = grid.solid.sum() * 0.1 ** 3
        meshes = sum(abs(b.mesh.volume) for b in result.scene.blocks)
        self.assertAlmostEqual(volume, meshes, delta=meshes * 0.12)

    def test_manifest_roundtrip_and_chunks(self):
        result = build(BuildRequest(product="room-shell", seed=1, openings="door", profile="human"))
        grid = voxelize(result.document, result.scene)
        with tempfile.TemporaryDirectory() as tmp:
            paths = grid.save(tmp, "room")
            manifest = json.loads(paths["manifest"].read_text())
            self.assertEqual(manifest["schema"], VOXEL_SCHEMA)
            self.assertEqual(manifest["units"], "metre")
            self.assertEqual(manifest["source"]["fingerprint"], result.document.fingerprint())
            back = VoxelGrid.load(paths["manifest"])
        np.testing.assert_array_equal(back.labels, grid.labels)
        self.assertEqual(back.palette, grid.palette)
        total = sum(int((block != 0).sum()) for _, block in grid.chunks())
        self.assertEqual(total, int(grid.solid.sum()))

    def test_minecraft_export(self):
        result = build(BuildRequest(product="slab", seed=1, width=4.0, height=1.0, depth=2.0))
        grid = voxelize(result.document, result.scene, pitch=1.0)
        text = grid.to_mcfunction()
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        self.assertTrue(lines)
        self.assertTrue(all(line.startswith(("fill ", "setblock ")) for line in lines))
        placed = 0
        for line in lines:
            parts = line.split()
            if parts[0] == "fill":
                placed += int(parts[4].lstrip("~")) - int(parts[1].lstrip("~")) + 1
            else:
                placed += 1
        self.assertEqual(placed, int(grid.solid.sum()))
        with self.assertRaises(ValueError):
            voxelize(result.document, result.scene, pitch=0.5).to_mcfunction()
        self.assertEqual(minecraft_block("window-glass"), "minecraft:glass")
        self.assertEqual(minecraft_block("unobtainium"), "minecraft:stone")

    def test_blockout_boxes_cover_the_solid(self):
        result = build(BuildRequest(product="room-shell", seed=1, openings="door"))
        grid = voxelize(result.document, result.scene, pitch=0.25)
        boxes = grid.blockout_boxes()
        cells = sum(round(np.prod(np.subtract(b["max"], b["min"])) / 0.25 ** 3) for b in boxes)
        self.assertEqual(cells, int(grid.solid.sum()))


class TestPlayability(unittest.TestCase):
    def test_headroom_counts_up_to_the_first_solid(self):
        solid = np.zeros((1, 5, 1), dtype=bool)
        solid[0, 3, 0] = True
        self.assertEqual(headroom(solid)[0, :, 0].tolist()[:4], [3, 2, 1, 0])

    def test_body_in_cells(self):
        body = Body.of(preset("human"), 0.13)
        self.assertEqual((body.k, body.h, body.step), (3, 14, 3))
        self.assertGreaterEqual((2 * body.k + 1) * 0.13, 2 * 0.4)

    def test_human_walks_into_a_room(self):
        result = build(BuildRequest(product="room-shell", seed=1, openings="door", profile="human"))
        game = _game(result)
        self.assertEqual(game["game.entry"].status, "pass")
        self.assertEqual(game["game.interior_reach"].status, "pass")
        self.assertEqual(result.quality.status, "pass")
        ports = result.quality.metrics["playability"]["ports"]
        self.assertEqual(len(ports), 1)
        self.assertEqual(ports[0]["facing"], [0.0, 0.0, 1.0])     # the door is in the front (+Z) wall
        self.assertGreaterEqual(ports[0]["width"] + 0.13, preset("human").clearances().door_width)
        declared = result.document.metadata["ports"]
        self.assertEqual(len(declared), 1)
        self.assertAlmostEqual(ports[0]["position"][0], declared[0]["position"][0], delta=0.2)
        self.assertAlmostEqual(ports[0]["position"][2], declared[0]["position"][2], delta=0.2)

    def test_default_door_shuts_out_a_giant(self):
        result = build(BuildRequest(product="room-shell", seed=1, openings="door"))
        grid = voxelize(result.document, result.scene, profile="giant")
        w = walk(grid, "giant")
        self.assertGreater(w.summary()["interior_cells"], 0)
        self.assertEqual(w.summary()["interior_reached"], 0)

    def test_giant_profile_builds_a_giant_door(self):
        result = build(BuildRequest(product="room-shell", seed=1, openings="door", profile="giant"))
        self.assertEqual(_game(result)["game.entry"].status, "pass")
        door = result.document.metadata["ports"][0]
        clear = preset("giant").clearances()
        self.assertGreaterEqual(door["width"], clear.door_width)
        self.assertGreaterEqual(door["height"], clear.door_height)

    def test_low_wall_refuses_a_giant_door(self):
        with self.assertRaises(ValueError) as ctx:
            build(BuildRequest(product="room-shell", seed=1, openings="door", height=2.5, profile="giant"))
        self.assertIn("too low", str(ctx.exception))

    def test_no_door_means_no_entry(self):
        result = build(BuildRequest(product="room-shell", seed=1, openings="window", profile="human"))
        self.assertEqual(_game(result)["game.entry"].status, "fail")
        self.assertEqual(result.quality.status, "fail")

    def test_stairs_follow_the_profile_step(self):
        small = build(BuildRequest(product="stair", seed=2, profile="small"))
        self.assertEqual(_game(small)["game.stair_rise"].status, "pass")
        forced = build(BuildRequest(product="stair", seed=2, profile="small", parameters={"steps": 12}))
        self.assertEqual(_game(forced)["game.stair_rise"].status, "fail")


def _grid(shape, pitch=0.2):
    """An empty synthetic grid with one palette entry, ground at y = 0."""
    return VoxelGrid(labels=np.zeros(shape, dtype=np.uint16), pitch=pitch, origin=(0.0, 0.0, 0.0),
                     palette=[{"label": 1, "substance": "limestone", "minecraft": "minecraft:stone"}])


def _reaches(result, cell):
    hit = np.nonzero((result.nodes == np.array(cell)).all(axis=1))[0]
    assert len(hit) == 1, f"{cell} is not a standing cell"
    return bool(result.reached[hit[0]]), bool(result.reached_walking[hit[0]])


class TestJumps(unittest.TestCase):
    # Human at 0.2 m cells: body 2 cells round, 9 tall, steps 2 cells (0.4 m
    # after rounding), drops 15 cells (3 m), jumps 2.4 m at 7 m/s.

    def test_a_ledge_taller_than_a_step_needs_a_jump(self):
        grid = _grid((60, 24, 16))
        grid.labels[30:, :8, :] = 1                          # a 1.6 m ledge
        result = walk(grid, "human", seeds=[(5, 0, 8)])
        self.assertEqual(_reaches(result, (45, 8, 8)), (True, False))
        flat = walk(grid, "human", seeds=[(5, 0, 8)], jumps=False)
        self.assertEqual(int(flat.reached.sum()), int(flat.reached_walking.sum()))
        self.assertGreater(result.jump_edges, 0)

    def test_a_ledge_above_the_jump_stays_out_of_reach(self):
        grid = _grid((60, 30, 16))
        grid.labels[30:, :14, :] = 1                         # 2.8 m, above a 2.4 m jump
        result = walk(grid, "human", seeds=[(5, 0, 8)])
        self.assertEqual(_reaches(result, (45, 14, 8)), (False, False))

    def test_jump_across_a_two_metre_gap(self):
        grid = _grid((90, 34, 16))
        grid.labels[:20, :15, :] = 1                         # 3 m platforms: too tall to climb back
        grid.labels[30:50, :15, :] = 1                       # 2 m gap
        result = walk(grid, "human", seeds=[(5, 15, 8)])
        self.assertEqual(_reaches(result, (40, 15, 8)), (True, False))
        # Walking off the edge is a 3 m drop the body survives, into the gap.
        self.assertEqual(_reaches(result, (25, 0, 8)), (True, True))

    def test_no_jump_across_nine_metres(self):
        grid = _grid((120, 34, 16))
        grid.labels[:20, :15, :] = 1
        grid.labels[65:85, :15, :] = 1                       # 9 m gap, beyond a 6.9 m reach
        result = walk(grid, "human", seeds=[(5, 15, 8)])
        self.assertEqual(_reaches(result, (75, 15, 8)), (False, False))
        runner = walk(grid, "runner", seeds=[(5, 15, 8)])    # 12 m/s, 3.2 m jump: 9.6 m reach
        self.assertTrue(_reaches(runner, (75, 15, 8))[0])

    def test_jumps_need_headroom_on_the_way(self):
        grid = _grid((60, 24, 16))
        grid.labels[30:, :8, :] = 1
        grid.labels[24:40, 10:12, :] = 1                     # a low ceiling over the ledge edge
        result = walk(grid, "human", seeds=[(5, 0, 8)])
        self.assertFalse(_reaches(result, (45, 8, 8))[0])

    def test_entry_by_jump_only_is_a_warning(self):
        # A closed room on a 1.2 m podium with one doorway and an apron in front.
        grid = _grid((80, 36, 80))
        grid.labels[20:60, :6, 20:64] = 1                    # podium, apron on +Z
        wall = np.zeros((80, 80), dtype=bool)
        wall[20:60, 20:60] = True
        wall[21:59, 21:59] = False
        grid.labels[:, 6:20, :][wall[:, None, :].repeat(14, axis=1)] = 1
        grid.labels[37:43, 6:18, 59] = 0                     # 1.2 m by 2.4 m door in the +Z wall
        grid.labels[20:60, 20, 20:60] = 1                    # roof
        result = walk(grid, "human")
        findings = {f["code"]: f["status"] for f in _grid_checks(grid, result, preset("human"), [])}
        self.assertEqual(findings["game.entry"], "warning")
        self.assertEqual(result.summary()["interior_reached_walking"], 0)
        self.assertGreater(result.summary()["interior_reached"], 0)

    def test_round_body_climbs_shallow_treads(self):
        # 0.18 m risers on 0.34 m treads: shallower than the body is wide.
        grid = _grid((60, 24, 16), pitch=0.1)
        for i in range(4):
            grid.labels[20 + 3 * i:, : 2 * (i + 1), :] = 1
        result = walk(grid, "human", seeds=[(5, 0, 8)], jumps=False)
        self.assertEqual(_reaches(result, (50, 8, 8)), (True, True))


class TestSettlementProfile(unittest.TestCase):
    def test_every_house_is_built_for_the_player(self):
        from core.architecture.settlement import build_document

        plain = build_document(BuildRequest(product="english-village", seed=1))
        giant = build_document(BuildRequest(product="english-village", seed=1, profile="giant"))
        self.assertEqual(giant.request.to_dict()["profile"]["name"], "giant")
        self.assertNotIn("profile", plain.request.to_dict())

        def lintels(document):
            out = {}
            for p in document.placements:
                if p.role == "entry_lintel":
                    out[p.id.split(".")[0]] = max(p.dimensions[0], p.dimensions[2])
            return out

        before, after = lintels(plain), lintels(giant)
        self.assertEqual(set(before), set(after))
        self.assertTrue(before)
        door = preset("giant").clearances().door_width
        for plot, width in after.items():
            self.assertGreaterEqual(width + 1e-6, door, plot)
            self.assertGreater(width, before[plot], plot)

    def test_settlement_audit_walks_each_building(self):
        from dataclasses import replace
        from types import SimpleNamespace

        house = build(BuildRequest(product="room-shell", seed=1, openings="door", profile="human"))
        shut = build(BuildRequest(product="room-shell", seed=1, openings="window", profile="human"))
        blocks, placements = [], []
        for plot, result in (("plot-a", house), ("plot-b", shut)):
            for block, placement in zip(result.scene.blocks, result.document.placements):
                blocks.append(block)
                placements.append(replace(placement, id=f"{plot}.room-shell.{placement.id}"))
        document = SimpleNamespace(
            request=house.document.request, placements=placements,
            metadata={"settlement": {"plots": [{"id": "plot-a", "product": "room-shell"},
                                               {"id": "plot-b", "product": "room-shell"},
                                               {"id": "plot-c", "product": None}]}})
        checks, metrics, grid = audit(document, SimpleNamespace(blocks=blocks))
        self.assertIsNone(grid)
        self.assertEqual([b["plot"] for b in metrics["buildings"]], ["plot-a", "plot-b"])
        entry = {c.code: c for c in checks}["game.entry"]
        self.assertEqual(entry.status, "fail")
        self.assertIn("plot-b", entry.message)
        self.assertNotIn("plot-a", entry.message)
        self.assertTrue(all(p["id"].startswith("plot-a.") for p in metrics["ports"]))


class TestVoxelsCommand(unittest.TestCase):
    def test_cli_writes_the_merge_files(self):
        import contextlib
        import io

        from cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["voxels", "room-shell", "--openings", "door", "--profile", "blocky", "--pitch", "1.0",
                             "--mcfunction", "--plan", "1.0", "--blockout", "--out", tmp, "--name", "room"])
            self.assertEqual(code, 0)
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["playability"]["game.entry"], "pass")
            for name in ("room.voxels.json", "room.voxels.npz", "room.mcfunction", "room.plan.txt", "room.blockout.json"):
                self.assertTrue((Path(tmp) / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
