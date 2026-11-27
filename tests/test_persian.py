"""Persian architect: rule contracts measured from the finished geometry."""
import itertools
import unittest
from copy import deepcopy

import numpy as np

from core.architecture import (BuildDocument, BuildRequest, build,
                               describe_product, list_products, rebuild)
from core.architecture.persian import (ARCHITECT, DETAILS, FINISHES, PRODUCT,
                                       VARIANTS)

SEEDS = (7, 42)


def members(document, role):
    return [p for p in document.placements if p.role == role]


def bounds(p):
    center, half = np.array(p.position), np.array(p.dimensions) / 2
    return center - half, center + half


def place_key(p, places=3):
    return (p.role, round(p.position[0], places), round(p.position[1], places),
            round(p.position[2], places), round(p.dimensions[0], places),
            round(p.dimensions[1], places), round(p.dimensions[2], places))


class PersianTests(unittest.TestCase):
    def test_product_is_registered_with_rules_and_finishes(self):
        names = [p['name'] for p in list_products()]
        self.assertIn(PRODUCT, names)
        product = describe_product(' Iwan_Court ')
        self.assertEqual(product['architect_id'], ARCHITECT.id)
        self.assertEqual(product['architect_title'], ARCHITECT.title)
        self.assertEqual(product['architect'], ARCHITECT.to_dict())
        self.assertLessEqual(set(VARIANTS), set(product['variants']))
        self.assertEqual(set(DETAILS), set(product['detail_levels']))
        self.assertTrue(set(FINISHES) <= {f[0] for f in product['finishes']})
        rules = product['rules']
        self.assertIn('arch_rise_to_span', rules)
        self.assertEqual(rules['arch_rise_to_span'], [0.45, 0.75])
        self.assertEqual(rules['dome_apex_to_diameter'], [0.55, 0.8])
        self.assertEqual(rules['iwan_depth_to_width'], [0.8, 1.6])
        self.assertEqual(rules['muqarnas_tiers'], [3, 6])

    def test_every_variant_two_seeds_three_details_passes_quality(self):
        for variant, seed, detail in itertools.product(VARIANTS, SEEDS, DETAILS):
            with self.subTest(variant=variant, seed=seed, detail=detail):
                r = build(BuildRequest(product=PRODUCT, variant=variant,
                                        detail=detail, seed=seed))
                failed = [c for c in r.quality.checks if c.status == 'fail']
                self.assertEqual(r.quality.status, 'pass', failed)

    def test_four_centred_arches_have_bounded_rise_and_bear_on_piers(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            r = build(BuildRequest(product=PRODUCT, variant=variant,
                                   detail='concept', seed=seed))
            doc = r.document
            arches = [p for p in doc.placements
                      if p.parameters.get('construction') == 'arch course']
            self.assertGreater(len(arches), 8)
            for p in arches:
                with self.subTest(role=p.role):
                    span = p.parameters['span']
                    rise = p.parameters['rise']
                    self.assertGreaterEqual(rise / span, 0.45)
                    self.assertLessEqual(rise / span, 0.75)
            # every arch springing rests on a pier or wall below it
            solids = [p for p in doc.placements
                      if p.parameters.get('construction') != 'arch course'
                      and p.role != 'muqarnas_cell']
            for p in arches:
                lo, hi = bounds(p)
                spring = lo[1] + 0.05
                foot = [q for q in solids
                        if not (q.position[0] - q.dimensions[0] / 2 > hi[0]
                                or q.position[0] + q.dimensions[0] / 2 < lo[0]
                                or q.position[2] - q.dimensions[2] / 2 > hi[2]
                                or q.position[2] + q.dimensions[2] / 2 < lo[2])
                        and q.position[1] + q.dimensions[1] / 2 >= spring - 0.05
                        and q.position[1] + q.dimensions[1] / 2 <= spring + 0.35]
                self.assertGreaterEqual(len(foot), 1, p.id)

    def test_transition_zone_stacks_without_overhang(self):
        for variant, seed in itertools.product(('four-iwan', 'chahar-taq'), SEEDS):
            r = build(BuildRequest(product=PRODUCT, variant=variant,
                                   detail='concept', seed=seed))
            doc = r.document
            squinches = members(doc, 'squinch')
            bands = members(doc, 'transition_band')
            self.assertEqual(len(squinches), 4)
            self.assertGreaterEqual(len(bands), 16)
            walls = [p for p in doc.placements if p.role in ('chamber_wall', 'sanctuary_wall')]
            for p in squinches:
                lo, hi = bounds(p)
                under = [q for q in walls
                         if not (q.position[0] - q.dimensions[0] / 2 > hi[0]
                                 or q.position[0] + q.dimensions[0] / 2 < lo[0]
                                 or q.position[2] - q.dimensions[2] / 2 > hi[2]
                                 or q.position[2] + q.dimensions[2] / 2 < lo[2])
                         and abs(q.position[1] + q.dimensions[1] / 2
                                 - (lo[1] - 0.05)) < 0.35]
                self.assertGreaterEqual(len(under), 1, p.id)
            # band panels sit on squinches or walls, never overhang > 1/3 depth
            supports = walls + squinches + bands
            for p in bands:
                lo, hi = bounds(p)
                depth = min(p.dimensions[0], p.dimensions[2])
                best = None
                for q in supports:
                    if q is p:
                        continue
                    qlo, qhi = bounds(q)
                    if qhi[1] > lo[1] + 1e-6:
                        continue
                    overlap_x = min(hi[0], qhi[0]) - max(lo[0], qlo[0])
                    overlap_z = min(hi[2], qhi[2]) - max(lo[2], qlo[2])
                    if overlap_x > 1e-6 and overlap_z > 1e-6:
                        gap = lo[1] - qhi[1]
                        if best is None or gap < best:
                            best = gap
                self.assertIsNotNone(best, p.id)
                self.assertLessEqual(best, depth / 3 + 0.05, p.id)
            drums = members(doc, 'drum')
            self.assertGreaterEqual(len(drums), 1)
            dlo, dhi = bounds(drums[0])
            inner = min(hi[0] - lo[0] for lo, hi in map(bounds, bands))
            self.assertLessEqual(dhi[0] - dlo[0], inner + 0.2)

    def test_dome_is_pointed_and_coursed_with_bounded_apex(self):
        for variant, seed in itertools.product(('four-iwan', 'chahar-taq'), SEEDS):
            r = build(BuildRequest(product=PRODUCT, variant=variant,
                                   detail='concept', seed=seed))
            doc = r.document
            dome = next(a for a in doc.metadata['anchors']['dome']
                        if 'apex' in a)
            self.assertGreaterEqual(dome['apex_ratio'], 0.55)
            self.assertLessEqual(dome['apex_ratio'], 0.8)
            courses = sorted({p.parameters['course'] for p in members(doc, 'dome_course')})
            self.assertGreaterEqual(len(courses), 8)
            self.assertTrue(all(members(doc, 'dome_course')))

    def test_iwans_are_vaulted_halls_framed_by_pishtaq(self):
        for variant, seed in itertools.product(('four-iwan', 'caravanserai'), SEEDS):
            r = build(BuildRequest(product=PRODUCT, variant=variant,
                                   detail='concept', seed=seed))
            doc = r.document
            iwans = members(doc, 'iwan')
            self.assertGreaterEqual(len(iwans), 4)
            for p in iwans:
                w = p.parameters['width']
                depth = p.parameters['depth']
                self.assertGreaterEqual(depth / w, 0.8)
                self.assertLessEqual(depth / w, 1.6)
            pishtaqs = members(doc, 'pishtaq')
            self.assertGreaterEqual(len(pishtaqs), len(iwans) * 2)
            arcades = [p for p in doc.placements if p.role == 'riwaq_pier']
            tops = [b for a, b in map(bounds, arcades)] if arcades else []
            if tops:
                arcade_roof = max(t[1] for t in tops)
                for p in pishtaqs:
                    lo, hi = bounds(p)
                    if hi[1] > arcade_roof + 0.5:
                        break
                else:
                    self.fail('no pishtaq stands above the arcade roof')

    def test_muqarnas_hood_corbelling_is_stable(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            r = build(BuildRequest(product=PRODUCT, variant=variant,
                                   detail='concept', seed=seed))
            doc = r.document
            cells = members(doc, 'muqarnas_cell')
            self.assertGreater(len(cells), 20)
            tiers = sorted({p.parameters['tier'] for p in cells})
            self.assertGreaterEqual(min(tiers), 0)
            self.assertLessEqual(max(tiers), 5)
            self.assertLessEqual(max(tiers) - min(tiers) + 1, 6)
            self.assertGreaterEqual(max(tiers) - min(tiers) + 1, 3)
            solids = [p for p in doc.placements if p.role != 'muqarnas_cell']
            for tier in tiers[:-1]:
                # each tier steps out by at most half a cell depth
                radii = {}
                for t in (tier, tier + 1):
                    radii[t] = max(p.parameters['radius']
                                   for p in cells if p.parameters['tier'] == t)
                step_out = radii[tier] - radii[tier + 1]
                depth = cells[0].dimensions[1]
                self.assertGreaterEqual(step_out, 0)
                self.assertLessEqual(step_out, depth / 2 + 1e-6)
            # centre of mass rests over masonry: the cell footprint overlaps
            # support either below (corbel) or around (niche cell)
            for p in cells:
                lo, hi = bounds(p)
                base = lo[1]
                ok = False
                for q in solids:
                    qlo, qhi = bounds(q)
                    if qhi[1] > base + 1e-6:
                        continue
                    if (min(hi[0], qhi[0]) - max(lo[0], qlo[0]) > 1e-6
                            and min(hi[2], qhi[2]) - max(lo[2], qlo[2]) > 1e-6
                            and base - qhi[1] < 0.15):
                        ok = True
                        break
                if not ok:
                    for q in cells:
                        if q is p or q.parameters.get('tier') != p.parameters.get('tier'):
                            continue
                        qlo, qhi = bounds(q)
                        if (min(hi[0], qhi[0]) - max(lo[0], qlo[0]) > 1e-6
                                and min(hi[2], qhi[2]) - max(lo[2], qlo[2]) > 1e-6):
                            ok = True
                            break
                self.assertTrue(ok, p.id)

    def test_symmetry_about_the_qibla_axis_and_the_qibla_is_largest(self):
        for seed in SEEDS:
            r = build(BuildRequest(product=PRODUCT, variant='four-iwan',
                                   detail='concept', seed=seed))
            doc = r.document
            iwans = members(doc, 'iwan')
            by_side = {p.parameters['side']: p for p in iwans}
            self.assertEqual(set(by_side), {'north', 'south', 'east', 'west'})
            qibla = by_side['south']
            for side, p in by_side.items():
                if side != 'south':
                    self.assertGreater(qibla.parameters['width'], p.parameters['width'])
                    self.assertGreater(qibla.parameters['height'], p.parameters['height'])
            # mirror pairs across the axis x = qibla center x: every iwan
            # either sits on the axis (north and south halls) or has a twin
            # mirrored across it with the same dimensions (east and west)
            cx = qibla.position[0]
            signature = {(round(p.position[0], 4), round(p.position[1], 4),
                          round(p.position[2], 4),
                          tuple(round(d, 4) for d in p.dimensions))
                         for p in iwans}
            for p in iwans:
                if abs(p.position[0] - cx) < 1e-6:
                    continue
                mirrored = (round(2 * cx - p.position[0], 4), round(p.position[1], 4),
                            round(p.position[2], 4),
                            tuple(round(d, 4) for d in p.dimensions))
                self.assertIn(mirrored, signature)
            minarets = members(doc, 'minaret')
            self.assertEqual(len(minarets), 2)
            self.assertAlmostEqual(minarets[0].position[0] + minarets[1].position[0],
                                    2 * cx, places=6)
            dome = next(a for a in doc.metadata['anchors']['dome'] if 'apex' in a)
            self.assertAlmostEqual(dome['x'], cx, places=6)

    def test_minarets_taper_with_corbelled_balcony_above_the_dome(self):
        for variant, seed in itertools.product(('four-iwan',), SEEDS):
            r = build(BuildRequest(product=PRODUCT, variant=variant,
                                   detail='concept', seed=seed))
            doc = r.document
            dome = next(a for a in doc.metadata['anchors']['dome'] if 'apex' in a)
            for m in members(doc, 'minaret'):
                drums = [p for p in doc.placements
                         if p.role == 'minaret_drum' and p.parameters.get('host') == m.id]
                self.assertGreaterEqual(len(drums), 2)
                bases = sorted(drums, key=lambda p: p.position[1])
                self.assertLess(bases[-1].dimensions[0], bases[0].dimensions[0])
                balconies = members(doc, 'minaret_balcony')
                self.assertGreaterEqual(len(balconies), 2)
                for b in balconies:
                    lo, hi = bounds(b)
                    self.assertLessEqual(hi[0] - lo[0], 1.6)
            tips = members(doc, 'minaret_cap')
            self.assertTrue(all(t.position[1] > dome['apex'] for t in tips))

    def test_masonry_only_no_timber_and_every_span_is_arched(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            r = build(BuildRequest(product=PRODUCT, variant=variant,
                                   detail='concept', seed=seed))
            doc = r.document
            for p in doc.placements:
                self.assertNotIn('timber', str(p.parameters.get('construction', '')))
                self.assertNotIn('timber', str(p.material))
            lintels = [p for p in doc.placements
                       if p.parameters.get('spanning') == 'lintel']
            self.assertEqual(lintels, [])

    def test_same_seed_replays_the_same_document_and_rebuild_matches(self):
        req = BuildRequest(product=' Iwan_Court ', variant='four-iwan',
                           detail='concept', seed=12)
        before = deepcopy(req.to_dict())
        first = build(req)
        self.assertEqual(req.to_dict(), before)
        self.assertEqual(first.document.fingerprint(),
                         build(req).document.fingerprint())
        replay = rebuild(BuildDocument.from_dict(first.document.to_dict()))
        self.assertEqual(replay.quality.status, 'pass')
        self.assertEqual(first.document.fingerprint(), replay.document.fingerprint())
        second = build(BuildRequest(product=PRODUCT, variant='four-iwan',
                                    detail='concept', seed=13))
        self.assertNotEqual(first.document.fingerprint(), second.document.fingerprint())

    def test_detail_level_never_moves_the_anchors(self):
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            anchors = {}
            for detail in DETAILS:
                r = build(BuildRequest(product=PRODUCT, variant=variant,
                                       detail=detail, seed=seed))
                doc = r.document
                dome = tuple((round(a['x'], 4), round(a['z'], 4),
                               round(a['apex'], 4))
                              for a in doc.metadata['anchors']['dome'] if 'apex' in a)
                hoods = tuple(sorted((round(p.position[0], 4), round(p.position[1], 4),
                                      round(p.position[2], 4))
                                     for p in doc.placements
                                     if p.role == 'muqarnas_cell'
                                     and p.parameters['tier'] == 0))
                minarets = tuple(sorted((round(p.position[0], 4), round(p.position[2], 4))
                                        for p in doc.placements if p.role == 'minaret'))
                iwans = tuple(sorted((p.parameters['side'], round(p.position[0], 4),
                                      round(p.position[2], 4),
                                      round(p.parameters['width'], 4),
                                      round(p.parameters['depth'], 4))
                                      for p in doc.placements if p.role == 'iwan'))
                anchors[detail] = (dome, hoods, minarets, iwans)
            self.assertEqual(anchors['concept'], anchors['standard'], variant)
            self.assertEqual(anchors['standard'], anchors['presentation'], variant)

    def test_rejects_unknown_variants_and_details(self):
        for kwargs in ({'variant': 'persian-garden'}, {'detail': 'extreme'},
                       {'parameters': {'finish': 'marble'}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(BuildRequest(product=PRODUCT, **kwargs))

    def test_variant_contract_is_wired_throughout(self):
        four = build(BuildRequest(product=PRODUCT, variant='four-iwan', detail='concept', seed=3))
        self.assertGreaterEqual(len(members(four.document, 'minaret')), 2)
        self.assertGreater(len(members(four.document, 'dome_course')), 100)
        kiosk = build(BuildRequest(product=PRODUCT, variant='chahar-taq', detail='concept', seed=3))
        self.assertEqual(len(members(kiosk.document, 'squinch')), 4)
        self.assertFalse(members(kiosk.document, 'minaret'))
        fort = build(BuildRequest(product=PRODUCT, variant='caravanserai', detail='concept', seed=3))
        self.assertFalse(members(fort.document, 'minaret'))
        self.assertFalse(members(fort.document, 'dome_course'))
        self.assertGreaterEqual(len(members(fort.document, 'iwan')), 4)
        self.assertGreaterEqual(len(members(fort.document, 'corner_tower')), 4)

    # -- the enclosed, roofed riwaq ---------------------------------------

    def _host_groups(self, doc, roles):
        groups = {}
        for p in doc.placements:
            if p.role in roles:
                groups.setdefault(p.parameters.get('host_id'), []).append(p)
        return groups

    def _group_box(self, stones):
        los, his = zip(*map(bounds, stones))
        return np.min(los, axis=0), np.max(his, axis=0)

    def _supports_at(self, doc, rect, y):
        """A pier or wall reaching a member's soffit inside a plan strip.

        The support either finishes its top course at the soffit, like a
        skewback, or is laid into it, like a slab bonded at a wall joint.
        """
        (x0, z0), (x1, z1) = rect
        for q in doc.placements:
            if q.kind not in ('wall', 'pier'):
                continue
            qlo, qhi = bounds(q)
            if qhi[1] < y - 0.06 or qlo[1] > y + 0.06:
                continue
            if min(x1, qhi[0]) - max(x0, qlo[0]) > 1e-6 and \
                    min(z1, qhi[2]) - max(z0, qlo[2]) > 1e-6:
                return True
        return False

    def test_four_iwan_court_is_a_walled_enclosure(self):
        from core.architecture.persian import WALL_T
        for seed in SEEDS:
            with self.subTest(seed=seed):
                doc = build(BuildRequest(product=PRODUCT, variant='four-iwan',
                                         detail='concept', seed=seed)).document
                walls = [p for p in doc.placements if p.kind == 'wall']
                enc = [p for p in doc.placements if p.role == 'enclosure_wall']
                self.assertGreater(len(enc), 40)
                elo, ehi = self._group_box(enc)
                portal = next(p for p in doc.placements
                              if p.parameters.get('bay_id') == 'portal')
                passage = next(p for p in doc.placements
                               if p.parameters.get('bay_id') == 'qibla_passage')
                mouth = portal.parameters['span'] / 2
                threshold = passage.parameters['span'] / 2
                sanct_line_top = max(bounds(p)[1][1] for p in doc.placements
                                     if p.role == 'sanctuary_wall'
                                     and bounds(p)[0][2] < ehi[2] - WALL_T / 2)
                wall_top = ehi[1]
                stations_x = np.arange(elo[0] + 0.4, ehi[0] - 0.3, 0.5)
                stations_z = np.arange(elo[2] + 0.4, ehi[2] - 0.3, 0.5)

                def side_walls(fixed, axis):
                    out = []
                    for p in walls:
                        lo, hi = bounds(p)
                        spans = (lo[2] <= fixed <= hi[2]) if axis == 'z' \
                            else (lo[0] <= fixed <= hi[0])
                        if spans:
                            out.append((lo, hi))
                    return out

                def covered(rects, axis, station, height):
                    for lo, hi in rects:
                        if not lo[1] <= height <= hi[1]:
                            continue
                        if axis == 'z' and lo[0] <= station <= hi[0]:
                            return True
                        if axis == 'x' and lo[2] <= station <= hi[2]:
                            return True
                    return False

                north = side_walls(elo[2] + WALL_T / 2, 'z')
                south = side_walls(ehi[2] - WALL_T / 2, 'z')
                east = side_walls(ehi[0] - WALL_T / 2, 'x')
                west = side_walls(elo[0] + WALL_T / 2, 'x')
                gaps = []
                for station in stations_x:
                    for height in (1.2, wall_top * 0.5, wall_top - 0.55):
                        if abs(station) > mouth and \
                                not covered(north, 'z', station, height):
                            gaps.append(('north', round(station, 2), height))
                    for height in (1.2, min(wall_top * 0.5, sanct_line_top - 0.3)):
                        if abs(station) > threshold and \
                                not covered(south, 'z', station, height):
                            gaps.append(('south', round(station, 2), height))
                for station in stations_z:
                    for height in (1.2, wall_top * 0.5, wall_top - 0.55):
                        if not covered(east, 'x', station, height):
                            gaps.append(('east', round(station, 2), height))
                        if not covered(west, 'x', station, height):
                            gaps.append(('west', round(station, 2), height))
                self.assertEqual(gaps, [], gaps[:6])

    def test_riwaq_bays_are_roofed_on_both_storeys_and_every_slab_bears(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                doc = build(BuildRequest(product=PRODUCT, variant='four-iwan',
                                          detail='concept', seed=seed)).document
                gals = [self._group_box(g) for g in
                        self._host_groups(doc, {'riwaq_gallery'}).values()]
                roofs = [self._group_box(g) for g in
                         self._host_groups(doc, {'riwaq_roof'}).values()]
                gallery_level = min(lo[1] for lo, _ in gals)
                # every bay, on either storey, has the slab above it
                bays = [p for p in doc.placements if p.role == 'riwaq_bay']
                self.assertGreater(len(bays), 8)
                for p in bays:
                    lo, hi = bounds(p)
                    center = ((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2)
                    storey = (gals if p.parameters['spring_y'] < gallery_level - 0.5
                              else roofs)
                    hit = [box for box in storey
                           if box[0][0] <= center[0] <= box[1][0]
                           and box[0][2] <= center[1] <= box[1][2]]
                    self.assertTrue(hit, p.id)
                # every gallery and roof member bears at both span ends
                for role in ('riwaq_gallery', 'riwaq_roof'):
                    for host, stones in self._host_groups(doc, {role}).items():
                        lo, hi = self._group_box(stones)
                        y = lo[1]
                        if hi[0] - lo[0] < hi[2] - lo[2]:
                            ends = [((lo[0], lo[2]), (lo[0] + 0.6, hi[2])),
                                    ((hi[0] - 0.6, lo[2]), (hi[0], hi[2]))]
                        else:
                            ends = [((lo[0], lo[2]), (hi[0], lo[2] + 0.6)),
                                    ((lo[0], hi[2] - 0.6), (hi[0], hi[2]))]
                        for rect in ends:
                            self.assertTrue(self._supports_at(doc, rect, y),
                                            (role, host))

    def test_no_iwan_back_face_passes_the_outer_wall(self):
        plans = {'four-iwan': 'enclosure_wall', 'caravanserai': 'cell_wall'}
        for variant, role in plans.items():
            for seed in SEEDS:
                with self.subTest(variant=variant, seed=seed):
                    doc = build(BuildRequest(product=PRODUCT, variant=variant,
                                             detail='concept', seed=seed)).document
                    outer = self._group_box(members(doc, role))
                    olo, ohi = outer
                    for p in members(doc, 'iwan'):
                        lo, hi = bounds(p)
                        self.assertGreaterEqual(lo[0], olo[0] - 1e-6, p.id)
                        self.assertLessEqual(hi[0], ohi[0] + 1e-6, p.id)
                        self.assertGreaterEqual(lo[2], olo[2] - 1e-6, p.id)
                        self.assertLessEqual(hi[2], ohi[2] + 1e-6, p.id)

    def test_caravanserai_bays_are_porticoes_roofed_to_the_cell_fronts(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                doc = build(BuildRequest(product=PRODUCT, variant='caravanserai',
                                         detail='concept', seed=seed)).document
                roofs = [self._group_box(g) for g in
                         self._host_groups(doc, {'riwaq_roof'}).values()]
                fronts = [bounds(p) for p in members(doc, 'cell_front')]
                self.assertGreater(len(fronts), 8)
                bays = [p for p in doc.placements if p.role == 'riwaq_bay']
                self.assertGreater(len(bays), 8)
                for p in bays:
                    lo, hi = bounds(p)
                    center = ((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2)
                    cover = [box for box in roofs
                             if box[0][0] <= center[0] <= box[1][0]
                             and box[0][2] <= center[1] <= box[1][2]]
                    self.assertTrue(cover, p.id)
                    rlo, rhi = cover[0]
                    # The portico roof runs back to the cell front behind the
                    # bay. Depth is the roof's short axis; that front is the
                    # wall thin on the same axis, and the roof's outer edge
                    # meets the wall face. A return wall crossing the corner
                    # is a different member, and a joint along it is not the
                    # face the slab bears on.
                    if rhi[0] - rlo[0] < rhi[2] - rlo[2]:
                        edge = rlo[0] if center[0] < 0 else rhi[0]
                        reach = min(min(abs(edge - flo[0]), abs(edge - fhi[0]))
                                    for flo, fhi in fronts
                                    if fhi[0] - flo[0] < fhi[2] - flo[2]
                                    and min(fhi[2], hi[2]) - max(flo[2], lo[2]) > 1e-6)
                    else:
                        edge = rlo[2] if center[1] < 0 else rhi[2]
                        reach = min(min(abs(edge - flo[2]), abs(edge - fhi[2]))
                                    for flo, fhi in fronts
                                    if fhi[2] - flo[2] < fhi[0] - flo[0]
                                    and min(fhi[0], hi[0]) - max(flo[0], lo[0]) > 1e-6)
                    self.assertLessEqual(reach, 0.16, (p.id, float(edge), float(reach)))
                # and the portico roofs bear on piers and front walls
                for host, stones in self._host_groups(doc, {'riwaq_roof'}).items():
                    lo, hi = self._group_box(stones)
                    y = lo[1]
                    if hi[0] - lo[0] < hi[2] - lo[2]:
                        ends = [((lo[0], lo[2]), (lo[0] + 0.6, hi[2])),
                                ((hi[0] - 0.6, lo[2]), (hi[0], hi[2]))]
                    else:
                        ends = [((lo[0], lo[2]), (hi[0], lo[2] + 0.6)),
                                ((lo[0], hi[2] - 0.6), (hi[0], hi[2]))]
                    for rect in ends:
                        self.assertTrue(self._supports_at(doc, rect, y),
                                        (host, rect))

    def test_court_paving_covers_the_sahn_with_mortar_joints_only(self):
        joint = 0.031
        for variant in ('four-iwan', 'caravanserai'):
            for seed in SEEDS:
                with self.subTest(variant=variant, seed=seed):
                    doc = build(BuildRequest(product=PRODUCT, variant=variant,
                                             detail='concept', seed=seed)).document
                    groups = self._host_groups(doc, {'court_paving'})
                    def field_area(g):
                        lo, hi = self._group_box(g)
                        return (hi[0] - lo[0]) * (hi[2] - lo[2])
                    big = max(groups.values(), key=field_area)
                    boxes = np.array([[*lo, *hi] for lo, hi in map(bounds, big)])
                    plo, phi = self._group_box(big)
                    misses = []
                    x = plo[0] + 0.11
                    while x < phi[0]:
                        z = plo[2] + 0.13
                        while z < phi[2]:
                            hit = np.any((boxes[:, 0] - joint <= x)
                                         & (boxes[:, 3] + joint >= x)
                                         & (boxes[:, 2] - joint <= z)
                                         & (boxes[:, 5] + joint >= z))
                            if not hit:
                                misses.append((round(x, 2), round(z, 2)))
                            z += 0.37
                        x += 0.41
                    self.assertEqual(misses, [], misses[:6])


if __name__ == '__main__':
    unittest.main()
