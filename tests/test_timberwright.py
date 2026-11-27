"""Building-scale contracts for the half-timbered house, not just mesh counts."""
import itertools
import json
import random
import struct
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh

from core.architecture import BuildDocument, BuildRequest, build, describe_product, list_products, rebuild


PRODUCT = "timber-house"
VARIANTS = ("longhouse", "gable-front", "twin-gable")


def members(document, role):
    return [p for p in document.placements if p.role == role]


def bounds(p):
    center, half = np.array(p.position), np.array(p.dimensions) / 2
    return center - half, center + half


class TimberwrightTests(unittest.TestCase):
    def test_discoverable_house_with_detached_design_controls(self):
        self.assertIn(PRODUCT, [p['name'] for p in list_products()])
        product = describe_product(' Timber_House ')
        self.assertEqual(product['architect_id'], 'timberwright')
        self.assertTrue(set(VARIANTS) <= set(product['variants']))
        self.assertEqual(set(product['parameters']), {'finish', 'porch', 'chimney'})
        product['variants'].clear()
        self.assertIn('twin-gable', describe_product(PRODUCT)['variants'])

    def test_real_houses_at_every_detail_with_distinct_roof_plans(self):
        for variant, detail in itertools.product(VARIANTS, ('concept', 'standard', 'presentation')):
            with self.subTest(variant=variant, detail=detail):
                r = build(BuildRequest(product=PRODUCT, variant=variant, detail=detail, seed=73))
                self.assertEqual(r.quality.status, 'pass', [c for c in r.quality.checks if c.status == 'fail'])
                self.assertEqual(len(r.scene.blocks), len(r.document.placements))
                ridges = members(r.document, 'ridge_cap')
                self.assertEqual(len(ridges), 2 if variant == 'twin-gable' else 1)
                self.assertEqual(ridges[0].dimensions[0] > ridges[0].dimensions[2], variant == 'longhouse')
                self.assertEqual(bool(members(r.document, 'valley_flashing')), variant == 'twin-gable')
                self.assertEqual(len(members(r.document, 'principal_rafter')),
                                 2 * len(ridges) * len(members(r.document, 'tie_beam')))
                self.assertEqual(bool(members(r.document, 'roof_tile')), detail == 'presentation')
                self.assertGreaterEqual(len(members(r.document, 'window_glass')), 4)

    def test_front_door_is_clear_and_stairs_meet_an_open_landing(self):
        r = build(BuildRequest(product=PRODUCT, variant='twin-gable', levels=2, detail='concept'))
        doc = r.document
        lintel, = members(doc, 'entry_lintel')
        footing, = members(doc, 'plinth')
        ground_floor = bounds(footing)[1][1]
        # Walk the line through the wall up to the lintel, not a painted-on door.
        probe = np.array(lintel.position)
        for probe[1] in np.linspace(ground_floor + .05, bounds(lintel)[0][1] - .05, 12):
            self.assertFalse(any(np.all(lo < probe) and np.all(probe < hi)
                                 for lo, hi in map(bounds, doc.placements)))
        floors = list(map(bounds, members(doc, 'upper_floor')))
        steps = list(map(bounds, members(doc, 'stair_tread')))
        self.assertGreater(len(steps), 5)
        last = steps[-1][1]
        self.assertAlmostEqual(last[1], floors[0][1][1], places=5)
        self.assertTrue(any(abs(lo[2] - last[2]) < 1e-5 and lo[0] < last[0] <= hi[0] for lo, hi in floors))
        for lo, hi in steps:
            center = (lo + hi) / 2
            self.assertFalse(any(a[0] < center[0] < b[0] and a[2] < center[2] < b[2] for a, b in floors))

    def test_seed_changes_geometry_and_replays_without_global_randomness(self):
        state = random.getstate()
        req = BuildRequest(product=' Timber_House ', seed=12, variant='gable-front', detail='concept')
        before = deepcopy(req.to_dict())
        first = build(req).document
        self.assertEqual(first.fingerprint(), build(req).document.fingerprint())
        self.assertEqual(first.fingerprint(), build(first.request).document.fingerprint())
        replay = rebuild(BuildDocument.from_dict(first.to_dict()))
        self.assertEqual(replay.quality.status, 'pass')
        self.assertEqual(first.fingerprint(), replay.document.fingerprint())
        self.assertEqual(req.to_dict(), before)
        req.seed += 1
        self.assertNotEqual([p.position for p in first.placements], [p.position for p in build(req).document.placements])
        self.assertEqual(random.getstate(), state)

    def test_auto_only_selects_roofs_that_fit(self):
        # Only the transverse longhouse ridge has enough headroom at this height.
        for seed in (0, 2, 5):
            r = build(BuildRequest(product=PRODUCT, width=24, depth=8, height=5.5, levels=1,
                                   bays=6, detail='concept', seed=seed))
            self.assertEqual(r.document.request.variant, 'longhouse')
            self.assertEqual(r.quality.status, 'pass')
            self.assertFalse(members(r.document, 'upper_floor'))

    def test_jetty_adapts_to_tight_bays_and_short_stair_runs(self):
        from core.architecture.timberwright import Design

        for dimensions in (dict(width=9.5, depth=10, height=10, bays=5, variant='gable-front'),
                           dict(width=12, depth=6, height=10.6, bays=4, variant='twin-gable')):
            for seed in range(16):
                d = Design.resolve(BuildRequest(product=PRODUCT, levels=2, seed=seed, **dimensions))
                self.assertGreaterEqual((d.width-2*d.jetty)/d.bays-.20, 1.6-1e-8)
                self.assertGreaterEqual(d.depth-2*d.jetty-1.24*d.storey, 1.9-1e-8)

    def test_inspection_fails_for_a_blocked_door_and_stairwell(self):
        doc = build(BuildRequest(product=PRODUCT, detail='concept')).document
        doc.metadata.pop('invariants')
        lintel, = members(doc, 'entry_lintel')
        lo, hi = bounds(lintel)
        sill = bounds(members(doc, 'plinth')[0])[1][1]
        plug = deepcopy(lintel)
        plug.id, plug.role = 'infill.at.entry', 'lime_infill'
        plug.position = (lintel.position[0], (lo[1]+sill)/2, lintel.position[2])
        plug.dimensions = (hi[0]-lo[0], lo[1]-sill, hi[2]-lo[2])
        doc.placements.append(plug)
        floor = members(doc, 'upper_floor')[0]
        floor.dimensions = (floor.dimensions[0]+4, *floor.dimensions[1:])
        failed = {c.code for c in rebuild(doc).quality.checks if c.status == 'fail'}
        self.assertTrue({'timber.entry_clear', 'timber.stair_access'} <= failed, failed)

    def test_low_eave_can_project_over_but_not_through_a_doorway(self):
        result = build(BuildRequest(product=PRODUCT, seed=7, width=12, height=9, depth=13, bays=4,
                                    levels=1, variant='gable-front', detail='concept'))
        self.assertEqual(result.quality.status, 'pass')
        doc = result.document
        lintel, = members(doc, 'entry_lintel')
        obstruction = deepcopy(members(doc, 'roof_plane')[0])
        obstruction.id = 'loose.roof.member'
        x, _, z = lintel.position
        obstruction.position, obstruction.dimensions = (x, 1.5, z), (1.0, .30, .60)
        doc.placements.append(obstruction)
        check = next(c for c in rebuild(doc).quality.checks if c.code == 'timber.entry_clear')
        self.assertEqual(check.status, 'fail')
        self.assertEqual(check.primitive_ids, [obstruction.id])

    def test_series_samples_multiple_storeys_bays_and_all_roofs_reproducibly(self):
        from core.architecture.agent import series_recipe
        from core.architecture.timberwright import Design

        recipe = series_recipe(24, 81, [PRODUCT])
        self.assertEqual(recipe, series_recipe(24, 81, [PRODUCT]))
        designs = [Design.resolve(BuildRequest(**b['request'])) for b in recipe['buildings']]
        self.assertEqual({d.levels for d in designs}, {1, 2})
        self.assertEqual({d.variant for d in designs}, set(VARIANTS))
        self.assertGreater(len({d.bays for d in designs}), 3)

    def test_pinning_finishes_and_optional_members_does_not_move_the_frame(self):
        def house(finish, optional):
            return build(BuildRequest(product=PRODUCT, seed=32, detail='concept', parameters={
                'finish': finish, 'chimney': optional, 'porch': optional})).document

        plain, dressed = house('chalk-slate', False), house('honey-clay', True)
        for role in ('chimney_stack', 'porch_post'):
            self.assertFalse(members(plain, role))
            self.assertTrue(members(dressed, role))
        posts = [members(doc, 'frame_post') for doc in (plain, dressed)]
        self.assertEqual([p.position for p in posts[0]], [p.position for p in posts[1]])
        self.assertNotEqual(posts[0][0].material, posts[1][0].material)

    def test_exported_glb_keeps_plaster_timber_glazing_and_roof_distinct(self):
        from core.architecture.io import export_mesh_artifact
        from core.architecture.surfaces import absolute_vertex_colors

        result = build(BuildRequest(product=PRODUCT, detail='concept'))
        with tempfile.TemporaryDirectory() as out:
            path = export_mesh_artifact(result.scene, Path(out)/'house.glb')
            glb = trimesh.load(path, force='scene', process=False)
            data = path.read_bytes()
            length, = struct.unpack_from('<I', data, 12)
            payload = json.loads(data[20:20 + length])
            extras = {node.get('name'): node.get('extras', {}) for node in payload['nodes']}
            means = {}
            for role in ('lime_infill', 'frame_post', 'window_glass', 'roof_plane'):
                placement = members(result.document, role)[0]
                means[role] = absolute_vertex_colors(glb.geometry[placement.id])[:, :3].astype(float).mean(axis=0)
                self.assertEqual(extras[placement.id]['substance'], placement.substance, role)
                self.assertIn('mass_kg', extras[placement.id])
            for left, right in itertools.combinations(means, 2):
                self.assertGreater(np.linalg.norm(means[left] - means[right]), 12.0, (left, right))

    def test_neutral_glb_omits_materials_without_recoloring_the_next_export(self):
        from core.architecture.io import export_mesh_artifact

        house = build(BuildRequest(product=PRODUCT, detail='concept', levels=1))
        with tempfile.TemporaryDirectory() as out:
            def gltf(name, **opts):
                data = export_mesh_artifact(house.scene, Path(out) / name, **opts).read_bytes()
                length, = struct.unpack_from('<I', data, 12)
                return json.loads(data[20:20 + length])

            plain = gltf('plain.glb', glb_materials=False)
            colored = gltf('colored.glb')
        self.assertEqual([m['name'] for m in plain['meshes']], [m['name'] for m in colored['meshes']])
        self.assertNotIn('materials', plain)
        for mesh in plain['meshes']:
            for prim in mesh['primitives']:
                self.assertNotIn('COLOR_0', prim['attributes'])
                self.assertNotIn('material', prim)
        self.assertTrue(any('material' in p or 'COLOR_0' in p['attributes']
                            for m in colored['meshes'] for p in m['primitives']))

    def test_rejects_unsupported_combinations_before_compiling(self):
        for kwargs in ({'variant': 'doric'}, {'bays': 900}, {'levels': 3}, {'width': 0},
                       {'detail': 'extreme'}, {'openings': 'none'}, {'bays': 7, 'width': 8},
                       {'parameters': {'porch': 'false'}}, {'parameters': {'chimney': 1}},
                       {'parameters': {'finish': []}}, {'parameters': {'porches': True}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(BuildRequest(product=PRODUCT, **kwargs))
