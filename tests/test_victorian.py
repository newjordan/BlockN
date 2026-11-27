"""Victorian architect: period rules measured from the finished geometry."""
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

from core.architecture import (BuildDocument, BuildRequest, build,
                               describe_product, list_products, rebuild)
from core.architecture.io import export_mesh_artifact
from core.architecture.victorian import (ARCHITECT, BRICK_T, DETAILS, FINISHES,
                                         FRONTAGE_BAND, HOUSE_RANGE, PITCH_BAND,
                                         PRODUCT, VARIANTS, VILLA_PITCH_BAND,
                                         random_controls)

SEEDS = (7, 42)
RAFTER_T = 0.09


def members(document, role):
    return [p for p in document.placements if p.role == role]


def bounds(p):
    center, half = np.array(p.position), np.array(p.dimensions) / 2
    return center - half, center + half


def place_key(p, places=3):
    return (p.role, round(p.position[0], places), round(p.position[1], places),
            round(p.position[2], places), round(p.dimensions[0], places),
            round(p.dimensions[1], places), round(p.dimensions[2], places))


def failing(result):
    return [c for c in result.quality.checks if c.status == "fail"]


def damaged(result, edit):
    document = deepcopy(result.document)
    document.metadata.clear()
    edit(document)
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


def build_variant(variant, **kwargs):
    return build(BuildRequest(product=PRODUCT, variant=variant, detail="concept",
                              seed=kwargs.pop("seed", 7), **kwargs))


def party_lines(document):
    return sorted({round(p.position[0], 3) for p in members(document, "party_wall")})


def front_inner_back_inner(document, variant):
    front_band = [p for p in document.placements
                  if p.role == ("front_wall" if variant != "gothic-villa" else "house_wall")
                  and p.block_type == "CUBE"
                  and (variant != "gothic-villa"
                       or p.parameters.get("wall_face") == "front")]
    back_band = [p for p in document.placements
                 if p.role == ("back_wall" if variant != "gothic-villa" else "house_wall")
                 and p.block_type == "CUBE"
                 and (variant != "gothic-villa"
                      or p.parameters.get("wall_face") == "back")]
    front_inner = max(p.position[2] + p.dimensions[2] / 2 for p in front_band)
    back_inner = min(p.position[2] - p.dimensions[2] / 2 for p in back_band)
    return front_inner, back_inner


def ridge_of(document):
    caps = [p for p in document.placements if p.role == "ridge_cap"
            and "ridge_y" in p.parameters]
    return max(float(p.parameters["ridge_y"]) for p in caps)


def member_edge(placements, samples, axis, highest=True):
    """Envelope of meshed members, sampled along one horizontal axis.

    Each member's recorded vertices are restored to world space and its mesh
    edges interpolated, so a raking course reports its true cut at every
    sample instead of the corner of its box. ``highest`` selects the top
    edge; the foot of a bargeboard is the low edge.
    """
    out = np.full(len(samples), np.nan)
    for p in placements:
        params = p.parameters or {}
        verts, faces = params.get("vertices"), params.get("faces")
        if not verts or not faces:
            continue
        verts = (np.asarray(verts, dtype=float) * np.asarray(p.dimensions, dtype=float)
                 + np.asarray(p.position, dtype=float))
        tri = np.asarray(faces, dtype=int)
        edges = np.concatenate((tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        a, b = verts[edges[:, 0]], verts[edges[:, 1]]
        low = np.minimum(a[:, axis], b[:, axis])
        high = np.maximum(a[:, axis], b[:, axis])
        run = b[:, axis] - a[:, axis]
        flat = np.abs(run) <= 1e-9
        for k, sample in enumerate(samples):
            mask = (low - 1e-6 <= sample) & (high + 1e-6 >= sample)
            if not mask.any():
                continue
            frac = np.where(flat[mask], 0.5,
                            (sample - a[mask, axis]) / np.where(flat[mask], 1.0, run[mask]))
            y = a[mask, 1] + (b[mask, 1] - a[mask, 1]) * np.clip(frac, 0.0, 1.0)
            y = np.where(flat[mask], np.maximum(a[mask, 1], b[mask, 1]), y)
            value = float(y.max() if highest else y.min())
            if np.isnan(out[k]):
                out[k] = value
            else:
                out[k] = max(out[k], value) if highest else min(out[k], value)
    return out


def member_top(placements, samples, axis):
    return member_edge(placements, samples, axis, highest=True)


class VictorianTests(unittest.TestCase):
    def test_product_is_registered_with_rules_and_finishes(self):
        names = [p['name'] for p in list_products()]
        self.assertIn(PRODUCT, names)
        product = describe_product(' Victorian_Terrace ')
        self.assertEqual(product['architect_id'], ARCHITECT.id)
        self.assertEqual(product['architect_title'], ARCHITECT.title)
        self.assertEqual(product['architect'], ARCHITECT.to_dict())
        self.assertLessEqual(set(VARIANTS), set(product['variants']))
        self.assertEqual(set(DETAILS), set(product['detail_levels']))
        self.assertEqual(list(product['parameters']), ['finish'])
        self.assertTrue(set(FINISHES) <= {f[0] for f in product['finishes']})
        rules = product['rules']
        for variant in VARIANTS:
            self.assertEqual(rules['frontage'][variant], list(FRONTAGE_BAND[variant]))
            self.assertEqual(rules['houses'][variant], list(HOUSE_RANGE[variant]))
        self.assertEqual(rules['roof_pitch_deg'], list(PITCH_BAND))
        self.assertEqual(rules['levels'], [2, 3])
        self.assertEqual(rules['wall_thickness'], [BRICK_T, 0.33])
        # the published vocabulary is data, not an alias of the module dict
        product['variants'].clear()
        self.assertIn('two-up-two-down', describe_product(PRODUCT)['variants'])

    def test_same_seed_replays_the_same_document_and_rebuild_matches(self):
        req = BuildRequest(product=' Victorian_Terrace ', variant='bay-fronted',
                           detail='concept', seed=12)
        before = deepcopy(req.to_dict())
        first = build(req)
        self.assertEqual(req.to_dict(), before)
        self.assertEqual(first.document.request.style, 'victorian')
        self.assertEqual(first.document.request.architect, ARCHITECT.id)
        self.assertEqual(first.document.fingerprint(),
                         build(req).document.fingerprint())
        replay = rebuild(BuildDocument.from_dict(first.document.to_dict()))
        self.assertEqual(replay.quality.status, 'pass')
        self.assertEqual(first.document.fingerprint(), replay.document.fingerprint())
        second = build(BuildRequest(product=PRODUCT, variant='bay-fronted',
                                    detail='concept', seed=13))
        self.assertNotEqual(first.document.fingerprint(), second.document.fingerprint())

    def test_every_variant_two_seeds_three_details_passes_quality(self):
        for variant, seed, detail in itertools.product(VARIANTS, SEEDS, DETAILS):
            with self.subTest(variant=variant, seed=seed, detail=detail):
                r = build(BuildRequest(product=PRODUCT, variant=variant,
                                       detail=detail, seed=seed))
                self.assertEqual(r.quality.status, 'pass', failing(r))
                self.assertEqual(len(r.scene.blocks), len(r.document.placements))

    def test_every_finish_builds_for_every_variant(self):
        for variant, finish in itertools.product(VARIANTS, FINISHES):
            with self.subTest(variant=variant, finish=finish):
                r = build(BuildRequest(product=PRODUCT, variant=variant,
                                       detail='concept', seed=2,
                                       parameters={'finish': finish}))
                self.assertEqual(r.quality.status, 'pass', failing(r))
                self.assertTrue(all(p.substance for p in r.document.placements))

    def test_bounds_match_the_request(self):
        cases = {
            'two-up-two-down': dict(width=32, depth=7.0),
            'bay-fronted': dict(width=30, depth=7.0),
            'gothic-villa': dict(width=8.2, depth=8.0),
        }
        for variant, controls in cases.items():
            with self.subTest(variant=variant):
                r = build_variant(variant, **controls)
                size = r.quality.metrics['bounds_size']
                # the resolved request records the terrace actually laid out;
                # end-stack copings and flanking villa stacks project past it
                width = r.document.request.width
                allowance = 0.7 if variant == 'gothic-villa' else 1.05
                self.assertGreaterEqual(size[0], width - 0.01)
                self.assertLessEqual(size[0], width + allowance)
                # depth grows by the outshut, the garden, or the villa's
                # projecting porch and wing, never more than one room
                depth = r.document.request.depth
                self.assertGreaterEqual(size[2], depth - 0.01)
                self.assertLessEqual(size[2], depth + 4.0)
                # terraces carry the ridge past stacks and pots; the villa's
                # gables are its own height, its stacks on the eaves
                ridge = ridge_of(r.document)
                if variant == 'gothic-villa':
                    self.assertGreaterEqual(size[1], ridge - 0.1)
                    self.assertLessEqual(size[1], ridge + 0.6)
                else:
                    self.assertGreaterEqual(size[1], ridge + 1.7)
                    self.assertLessEqual(size[1], ridge + 2.6)
                self.assertLessEqual(r.document.request.height, ridge + 0.01)

    def test_party_walls_rise_through_the_roof(self):
        for variant, seed in itertools.product(('two-up-two-down', 'bay-fronted'), SEEDS):
            with self.subTest(variant=variant, seed=seed):
                doc = build_variant(variant, seed=seed).document
                ridge = ridge_of(doc)
                lines = party_lines(doc)
                self.assertGreaterEqual(len(lines), HOUSE_RANGE[variant][0] + 1)
                for line in lines:
                    near = [p for p in members(doc, 'party_wall')
                            if abs(p.position[0] - line) <= 0.16]
                    top = max(p.position[1] + p.dimensions[1] / 2 for p in near)
                    self.assertGreaterEqual(top, ridge + 0.25, line)
                    # the fire break spans the whole main block, front to back
                    self.assertLessEqual(min(p.position[2] - p.dimensions[2] / 2
                                             for p in near),
                                         -doc.request.depth / 2 + 0.01)
                    self.assertGreaterEqual(max(p.position[2] + p.dimensions[2] / 2
                                                for p in near),
                                            doc.request.depth / 2 - 0.01)

    def test_every_opening_bears_and_every_light_is_pierced(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            with self.subTest(variant=variant, seed=seed):
                doc = build_variant(variant, seed=seed).document
                records = [p for p in doc.placements
                           if p.role in ('gauged_arch', 'porch_arch', 'pointed_window')]
                self.assertGreater(len(records), 8)
                walls = [p for p in doc.placements if p.kind in ('wall', 'foundation')]
                for record in records:
                    pr = record.parameters
                    axis = pr['axis']
                    along_i, cross_i = (0, 2) if axis == 'x' else (2, 0)
                    center = float(pr['center'])
                    wo, half = float(pr['span_half']), float(pr['half'])
                    spring, sill = float(pr['spring_y']), float(pr.get('sill', 0.0))
                    cross_c = record.position[cross_i]
                    with self.subTest(opening=pr['opening_id']):
                        for s in (-1, 1):
                            strip = (min(center + s * half - 0.05, center + s * (wo + 0.08)),
                                     max(center + s * half - 0.05, center + s * (wo + 0.08)))
                            jamb = [q for q in walls
                                    if bounds(q)[0][along_i] < strip[1]
                                    and bounds(q)[1][along_i] > strip[0]
                                    and bounds(q)[1][1] >= spring - 0.02
                                    and bounds(q)[0][1] < spring
                                    and abs((bounds(q)[0][cross_i] + bounds(q)[1][cross_i]) / 2
                                            - cross_c) <= 0.6]
                            self.assertGreaterEqual(len(jamb), 1, pr['opening_id'])
                        for light_c, light_h in pr['lights']:
                            light_c, light_h = float(light_c), float(light_h)
                            inside = [q for q in walls
                                      if bounds(q)[0][along_i] > light_c - light_h + 0.08
                                      and bounds(q)[1][along_i] < light_c + light_h - 0.08
                                      and bounds(q)[0][1] < spring - 0.1
                                      and bounds(q)[1][1] > sill + 0.1
                                      and abs((bounds(q)[0][cross_i] + bounds(q)[1][cross_i]) / 2
                                              - cross_c) <= 0.6]
                            self.assertEqual(inside, [], pr['opening_id'])

    def test_joists_span_front_to_back_between_party_walls(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            with self.subTest(variant=variant, seed=seed):
                doc = build_variant(variant, seed=seed).document
                joists = members(doc, 'joist')
                self.assertGreater(len(joists), 8)
                front_inner, back_inner = front_inner_back_inner(doc, variant)
                lines = party_lines(doc)
                walls = [p for p in doc.placements if p.kind in ('wall', 'foundation')]
                for j in joists:
                    lo, hi = bounds(j)
                    with self.subTest(joist=j.id):
                        self.assertLessEqual(lo[2], front_inner + 0.02, j.id)
                        self.assertGreaterEqual(hi[2], back_inner - 0.02, j.id)
                        if lines:
                            ok = any(lines[k] + 0.08 <= lo[0]
                                     and hi[0] <= lines[k + 1] - 0.08
                                     for k in range(len(lines) - 1))
                            self.assertTrue(ok, j.id)
                        # each end pockets into masonry whose top course is
                        # the joist's own base
                        for z_a, z_b in ((lo[2], lo[2] + 0.2), (hi[2] - 0.2, hi[2])):
                            bearing = [q for q in walls
                                       if bounds(q)[0][2] < z_b and bounds(q)[1][2] > z_a
                                       and abs(bounds(q)[1][1] - lo[1]) <= 0.008
                                       and bounds(q)[0][1] < lo[1]]
                            self.assertGreaterEqual(len(bearing), 1, j.id)
                        # boards lie over the joists at the same floor line
                floors = {round(j.parameters['floor_y'], 4) for j in joists}
                self.assertGreaterEqual(len(floors), 1)
                boards = members(doc, 'floorboard')
                self.assertGreaterEqual(len(boards), len(joists))

    def test_roof_pitch_falls_in_the_band_and_is_slated(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            with self.subTest(variant=variant, seed=seed):
                doc = build_variant(variant, seed=seed).document
                band = VILLA_PITCH_BAND if variant == 'gothic-villa' else PITCH_BAND
                rafters = [p for p in doc.placements if p.role == 'rafter']
                common = [p for p in rafters
                          if (p.parameters or {}).get('construction') == 'common rafter']
                self.assertGreater(len(common), 4)
                pitches = set()
                for p in rafters:
                    recorded = (p.parameters or {}).get('pitch_deg')
                    # a lean-to records its own pitch; it is never 0 or absent
                    self.assertIsNotNone(recorded, p.id)
                    self.assertGreater(float(recorded), 1.0, p.id)
                    lo, hi = bounds(p)
                    run = max(hi[0] - lo[0], hi[2] - lo[2])
                    rise = (hi[1] - lo[1]) - RAFTER_T
                    pitch = float(np.degrees(np.arctan2(rise, run)))
                    self.assertAlmostEqual(pitch, float(recorded), delta=0.6, msg=p.id)
                    if (p.parameters or {}).get('construction') != 'common rafter':
                        continue
                    pitches.add(round(pitch, 2))
                    self.assertGreaterEqual(pitch, band[0] - 0.6, p.id)
                    self.assertLessEqual(pitch, band[1] + 0.6, p.id)
                # slate courses cover eaves to ridge over the rafters
                slates = members(doc, 'slate_course')
                self.assertGreater(len(slates), 8)
                slate_ys = [bounds(p)[1][1] for p in slates]
                ridge = ridge_of(doc)
                self.assertGreaterEqual(max(slate_ys), ridge - 0.30)
                self.assertLessEqual(min(slate_ys), ridge - 1.5)
                self.assertGreaterEqual(len(members(doc, 'ridge_cap')), 1)

    def test_party_walls_and_gables_rake_above_the_roof_with_coping(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            with self.subTest(variant=variant, seed=seed):
                doc = build_variant(variant, seed=seed).document
                walls = [p for p in doc.placements
                         if p.role in ('party_wall', 'house_wall')
                         and (p.parameters or {}).get('parapet')]
                copes = members(doc, 'parapet_coping')
                slates = members(doc, 'slate_course')
                self.assertTrue(walls)
                self.assertTrue(copes)
                groups = {}
                for p in walls:
                    pr = p.parameters
                    key = (pr['parapet_axis'], round(float(pr['parapet_at']), 3),
                           pr.get('gable', ''))
                    groups.setdefault(key, []).append(p)
                measured = []
                for (axis, at, kind), wall in groups.items():
                    cope = [p for p in copes
                            if p.parameters.get('parapet_axis') == axis
                            and abs(float(p.parameters['parapet_at']) - at) <= 0.01
                            and p.parameters.get('gable', '') == kind]
                    self.assertTrue(cope, (variant, axis, at))
                    ai = 2 if axis == 'z' else 0
                    lo_a = min(bounds(p)[0][ai] for p in wall)
                    hi_a = max(bounds(p)[1][ai] for p in wall)
                    samples = np.linspace(lo_a + 0.03, hi_a - 0.03, 45)
                    top = member_top(cope, samples, ai)
                    head = member_top(wall, samples, ai)
                    self.assertFalse(np.isnan(top).any(), (variant, axis, at))
                    self.assertFalse(np.isnan(head).any(), (variant, axis, at))
                    # the coping is its own member, laid on the raking head
                    lift = top - head
                    self.assertGreaterEqual(float(lift.min()), 0.02, (variant, axis, at))
                    self.assertLessEqual(float(lift.max()), 0.16, (variant, axis, at))
                    if axis == 'z':
                        sides = []
                        for side in (-1, 1):
                            if side > 0:
                                sides.append([p for p in slates
                                              if bounds(p)[0][0] >= at - 0.02
                                              and bounds(p)[0][0] <= at + 0.24])
                            else:
                                sides.append([p for p in slates
                                              if bounds(p)[1][0] <= at + 0.02
                                              and bounds(p)[1][0] >= at - 0.24])
                        keep = np.abs(samples) > 0.52   # clear of the stack
                    else:
                        roof = '' if kind == 'main' else kind
                        sides = [[p for p in slates
                                  if p.parameters.get('gable', '') == roof]]
                        keep = np.ones(len(samples), dtype=bool)
                    for near in sides:
                        if not near:
                            continue     # an exposed terrace end
                        roof = member_top(near, samples, ai)
                        up = top - roof
                        probe = keep & ~np.isnan(up)
                        self.assertTrue(probe.any(), (variant, axis, at))
                        measured.extend(up[probe].tolist())
                        self.assertGreaterEqual(float(up[probe].min()), 0.25,
                                                (variant, axis, at))
                        self.assertLessEqual(float(up[probe].max()), 0.50,
                                             (variant, axis, at))
                # every parapet in the build sits in the fire-break band
                self.assertGreaterEqual(min(measured), 0.25)
                self.assertLessEqual(max(measured), 0.50)
                if variant == 'gothic-villa':
                    # every gable verge keeps a bargeboard that laps the slate,
                    # so the covering's end is closed in the gable elevation
                    boards = members(doc, 'bargeboard')
                    for kind, per_gable in (('main', 2), ('wing', 1), ('porch', 1)):
                        kind_boards = [p for p in boards if p.parameters.get('gable') == kind]
                        self.assertGreaterEqual(len(kind_boards), 2 * per_gable)
                        roof_name = '' if kind == 'main' else kind
                        kind_slates = [p for p in slates
                                       if p.parameters.get('gable', '') == roof_name]
                        lo_x = min(bounds(p)[0][0] for p in kind_boards)
                        hi_x = max(bounds(p)[1][0] for p in kind_boards)
                        samples = np.linspace(lo_x + 0.05, hi_x - 0.05, 31)
                        foot = member_edge(kind_boards, samples, 0, highest=False)
                        slate_top = member_top(kind_slates, samples, 0)
                        both = ~np.isnan(foot) & ~np.isnan(slate_top)
                        self.assertTrue(both.any(), kind)
                        self.assertLessEqual(float(np.max(foot[both] - slate_top[both])),
                                             0.01, kind)

    def test_frontage_falls_in_the_variant_band(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            with self.subTest(variant=variant, seed=seed):
                r = build_variant(variant, seed=seed)
                doc = r.document
                band = FRONTAGE_BAND[variant]
                if variant == 'gothic-villa':
                    walls = [p for p in doc.placements if p.kind == 'wall']
                    span = (max(bounds(p)[1][0] for p in walls)
                            - min(bounds(p)[0][0] for p in walls))
                    self.assertGreaterEqual(span, band[0] - 0.15)
                    self.assertLessEqual(span, band[1] + 0.15)
                    return
                fronts = members(doc, 'front_wall')
                span = (max(bounds(p)[1][0] for p in fronts)
                        - min(bounds(p)[0][0] for p in fronts) + BRICK_T)
                houses = len(party_lines(doc)) - 1
                facade = span / houses
                self.assertGreaterEqual(facade, band[0] - 1e-4)
                self.assertLessEqual(facade, band[1] + 1e-4)
                self.assertGreaterEqual(houses, HOUSE_RANGE[variant][0])
                self.assertLessEqual(houses, HOUSE_RANGE[variant][1])

    def test_stacks_copings_pots_and_breasts_on_every_party_wall(self):
        for variant, seed in itertools.product(('two-up-two-down', 'bay-fronted'), SEEDS):
            with self.subTest(variant=variant, seed=seed):
                doc = build_variant(variant, seed=seed).document
                ridge = ridge_of(doc)
                lines = party_lines(doc)
                stacks = members(doc, 'stack')
                copings = members(doc, 'stack_coping')
                pots = members(doc, 'chimney_pot')
                breasts = members(doc, 'chimney_breast')
                self.assertGreaterEqual(len(breasts), len(lines))
                for line in lines:
                    with self.subTest(party=line):
                        near = [p for p in stacks if abs(p.position[0] - line) <= 0.35]
                        # concept coursing lays the 1.35 m shaft in two drums
                        self.assertGreaterEqual(len(near), 2)
                        base = min(bounds(p)[0][1] for p in near)
                        self.assertGreaterEqual(base, ridge + 0.2)
                        top = max(bounds(p)[1][1] for p in near)
                        cap = [p for p in copings
                               if abs(bounds(p)[0][1] - (top + 0.02)) <= 0.04
                               and abs(p.position[0] - line) <= 0.5]
                        self.assertGreaterEqual(len(cap), 1)
                        on_cap = [p for p in pots
                                  if abs(bounds(p)[0][1] - bounds(cap[0])[1][1]) <= 0.04
                                  and abs(p.position[0] - line) <= 0.5]
                        self.assertGreaterEqual(len(on_cap), 2)
                        breast = [p for p in breasts if abs(p.position[0] - line) <= 0.75]
                        self.assertGreaterEqual(len(breast), 2)
                        self.assertLessEqual(min(bounds(p)[0][1] for p in breast), 0.01)

    def test_detail_level_never_moves_the_anchors(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            with self.subTest(variant=variant, seed=seed):
                anchors = {}
                for detail in DETAILS:
                    doc = build(BuildRequest(product=PRODUCT, variant=variant,
                                             detail=detail, seed=seed)).document
                    party = tuple(sorted({round(p.position[0], 3)
                                          for p in members(doc, 'party_wall')}))
                    doors = tuple(sorted((round(p.position[0], 3), round(p.position[2], 3))
                                         for p in members(doc, 'door_leaf')))
                    copings = tuple(sorted((round(p.position[0], 3), round(p.position[2], 3))
                                           for p in members(doc, 'stack_coping')))
                    arches = tuple(sorted((round(p.parameters['center'], 3),
                                           round(p.parameters['spring_y'], 3))
                                          for p in doc.placements
                                          if p.role in ('gauged_arch', 'porch_arch',
                                                        'pointed_window')))
                    floors = tuple(sorted({round(j.parameters['floor_y'], 4)
                                           for j in members(doc, 'joist')}))
                    plates = tuple(sorted((round(p.position[0], 3), round(p.position[2], 3))
                                           for p in members(doc, 'plate')))
                    anchors[detail] = (party, doors, copings, arches, floors, plates)
                self.assertEqual(anchors['concept'], anchors['standard'], variant)
                self.assertEqual(anchors['standard'], anchors['presentation'], variant)

    def test_variant_contract_is_wired_throughout(self):
        two_up = build_variant('two-up-two-down').document
        self.assertGreaterEqual(len(two_up.placements), 1000)
        self.assertGreaterEqual(len(members(two_up, 'outshut_wall')), 20)
        pairs = [p for p in members(two_up, 'gauged_arch')
                  if str(p.parameters.get('opening_id', '')).startswith('door.pair')]
        self.assertGreaterEqual(len(pairs), 4)
        true_pairs = [p for p in pairs if p.parameters['span_half'] > 0.9]
        self.assertGreaterEqual(len(true_pairs), 2)
        for pair in true_pairs:
            self.assertEqual(len(pair.parameters['lights']), 2)
        bay = build_variant('bay-fronted').document
        for role, least in (('bay_wall', 20), ('bay_flank', 12), ('bay_plinth', 4),
                            ('garden_wall', 4), ('railing_bar', 20), ('path_tile', 8)):
            self.assertGreaterEqual(len(members(bay, role)), least, role)
        villa = build_variant('gothic-villa').document
        self.assertGreaterEqual(len(members(villa, 'bargeboard')), 6)
        self.assertGreaterEqual(len(members(villa, 'porch_arch')), 4)
        self.assertGreaterEqual(len(members(villa, 'porch_pier')), 4)
        self.assertGreaterEqual(len(members(villa, 'pointed_window')), 4)
        # the wing breaks the symmetry of the plan
        wing = {p.parameters.get('wall_face') for p in villa.placements
                if p.role == 'house_wall'}
        self.assertIn('wing', wing)
        self.assertIn('side', wing)

    def test_garden_path_reaches_every_door(self):
        doc = build_variant('bay-fronted').document
        front_inner, _ = front_inner_back_inner(doc, 'bay-fronted')
        street = min(bounds(p)[0][2] for p in members(doc, 'garden_wall'))
        tiles = members(doc, 'path_tile')
        doors = [p for p in members(doc, 'door_leaf')
                 if str(p.parameters.get('opening_id', '')).startswith('door.')]
        self.assertGreaterEqual(len(doors), 3)
        for door in doors:
            lo, hi = bounds(door)
            x = (lo[0] + hi[0]) / 2
            near = [p for p in tiles if bounds(p)[0][0] < x < bounds(p)[1][0]]
            self.assertTrue(near, door.id)
            self.assertLessEqual(min(bounds(p)[0][2] for p in near), street + 0.4)
            self.assertGreaterEqual(max(bounds(p)[1][2] for p in near),
                                    front_inner - BRICK_T - 0.1)
        # railings stand over the garden wall
        bars = members(doc, 'railing_bar')
        walls = members(doc, 'garden_wall')
        self.assertGreaterEqual(len(bars), max(4, len(walls)))
        for bar in bars:
            self.assertGreaterEqual(bounds(bar)[0][1], 0.5)

    def test_damage_fails_the_period_checks(self):
        terrace = build_variant('two-up-two-down', seed=9)
        bay = build_variant('bay-fronted', seed=9)
        villa = build_variant('gothic-villa', seed=9)

        def remove_stack(doc):
            keep = next(p for p in doc.placements if p.role == 'stack').parameters['party']
            doc.placements = [p for p in doc.placements
                              if not (p.role == 'stack' and p.parameters.get('party') == keep)]

        self.assertIn('victorian.stacks_on_party_walls', damaged(terrace, remove_stack))

        def block_door(doc):
            leaf = next(p for p in doc.placements if p.role == 'door_leaf')
            plug = deepcopy(leaf)
            plug.id, plug.role, plug.kind = 'plug.at.door', 'front_wall', 'wall'
            plug.position = (leaf.position[0], 1.0, leaf.position[2])
            plug.dimensions = (0.3, 1.5, 0.2)
            doc.placements.append(plug)

        self.assertIn('victorian.openings_bear', damaged(terrace, block_door))

        def break_joist(doc):
            lines = party_lines(doc)
            j = next(p for p in doc.placements if p.role == 'joist')
            j.position = (lines[0], j.position[1], j.position[2])

        self.assertIn('victorian.joists_span_between_party_walls',
                      damaged(terrace, break_joist))

        def drop_firebreak(doc):
            line = party_lines(doc)[1]
            doc.placements = [p for p in doc.placements
                              if not (p.role == 'party_wall'
                                      and abs(p.position[0] - line) <= 0.16
                                      and p.position[1] > 5.0)]

        self.assertIn('victorian.party_walls_through_roof', damaged(terrace, drop_firebreak))

        def remove_outshut(doc):
            doc.placements = [p for p in doc.placements if p.role != 'outshut_wall']

        self.assertIn('victorian.outshut', damaged(terrace, remove_outshut))

        def lose_bay(doc):
            doc.placements = [p for p in doc.placements if p.role != 'bay_wall']

        self.assertIn('victorian.openings_bear', damaged(bay, lose_bay))

        def lose_bargeboards(doc):
            doc.placements = [p for p in doc.placements if p.role != 'bargeboard']

        self.assertIn('victorian.gothic_villa', damaged(villa, lose_bargeboards))

        def detach_pot(doc):
            p = next(p for p in doc.placements if p.role == 'chimney_pot')
            p.position = (p.position[0] + 100.0, p.position[1], p.position[2])

        self.assertIn('victorian.envelope_connectivity', damaged(villa, detach_pot))

    def test_rejects_unknown_variants_details_and_parameters(self):
        bad = [dict(variant='edwardian-terrace'),
               dict(detail='extreme'),
               dict(parameters={'finish': 'marble'}),
               dict(parameters={'turrets': True}),
               dict(openings='mixed'),
               dict(variant='two-up-two-down', bays=3),
               dict(variant='bay-fronted', bays=9),
               dict(variant='two-up-two-down', width=60.0),
               dict(variant='gothic-villa', depth=4.0),
               dict(levels=4),
               dict(levels=1)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(BuildRequest(product=PRODUCT, **kwargs))

    # -- the 20-seed random sweep, in batches so workers share the load ---
    def _sweep(self, seeds):
        for seed in seeds:
            for variant in VARIANTS:
                controls = random_controls(random.Random(seed), variant)
                with self.subTest(variant=variant, seed=seed, controls=controls):
                    r = build(BuildRequest(product=PRODUCT, **controls))
                    self.assertEqual(r.quality.status, 'pass',
                                     [c.to_dict() for c in failing(r)])

    def test_random_controls_batch_0_passes_quality(self):
        self._sweep(range(0, 5))

    def test_random_controls_batch_1_passes_quality(self):
        self._sweep(range(5, 10))

    def test_random_controls_batch_2_passes_quality(self):
        self._sweep(range(10, 15))

    def test_random_controls_batch_3_passes_quality(self):
        self._sweep(range(15, 20))

    def test_exported_glb_loads_back_with_named_members(self):
        result = build_variant('bay-fronted')
        with tempfile.TemporaryDirectory() as out:
            path = export_mesh_artifact(result.scene, Path(out) / 'terrace.glb')
            glb = trimesh.load(path, force='scene', process=False)
            self.assertGreater(len(glb.geometry), 1000)
            data = path.read_bytes()
            length, = struct.unpack_from('<I', data, 12)
            payload = json.loads(data[20:20 + length])
            names = {node.get('name') for node in payload['nodes']}
            ids = {p.id for p in result.document.placements}
            self.assertEqual(ids, names & ids)
            extras = {node.get('name'): node.get('extras', {}) for node in payload['nodes']}
            sample = next(p for p in result.document.placements if p.role == 'sash_glass')
            self.assertEqual(extras[sample.id]['substance'], sample.substance)
            self.assertIn('mass_kg', extras[sample.id])
            brick = next(p for p in result.document.placements if p.role == 'front_wall')
            self.assertEqual(extras[brick.id]['role'], 'front_wall')


if __name__ == '__main__':
    unittest.main()
