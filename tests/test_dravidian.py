"""Dravidian architect: the tiered gopuram and vimana, their haras, their eaves and their pots."""
from copy import deepcopy
import random
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild
from core.architecture.dravidian import random_controls
from core.architecture.construction import construction_report, craft_catalog
from core.architecture.substances import SUBSTANCES

_BUILDS = {}


def built(product, variant="auto", **controls):
    key = (product, variant, tuple(sorted(controls.items())))
    if key not in _BUILDS:
        _BUILDS[key] = build(BuildRequest(product=product, variant=variant, detail="concept", seed=73, **controls))
    return _BUILDS[key]


def failures(report):
    return {c.code for c in report.checks if c.status == "fail"}


def damaged(result, edit):
    document = deepcopy(result.document)
    document.metadata.clear()
    edit(document)
    return failures(rebuild(document).quality)


def members(result, role):
    return [p for p in result.document.placements if p.role == role]


GATE, SHRINE = "dravidian-gopuram", "dravidian-vimana"


class CatalogTests(unittest.TestCase):
    def test_each_tower_is_described_with_what_it_is_built_of(self):
        gate, shrine = describe_product(GATE), describe_product(SHRINE)
        self.assertEqual(gate["architect_id"], "dravidian")
        self.assertEqual(set(gate["substances"]), {"granite", "fired-clay-brick", "lime-plaster", "copper-sheet"})
        self.assertEqual(shrine["substances"], ["granite"])                 # a vimana is granite throughout
        self.assertEqual(gate["variants"], ["auto", "random", "rajagopuram", "early-gopuram"])
        self.assertEqual(shrine["rules"]["levels"], [3, 9])

    def test_the_tradition_is_in_the_craft_catalog(self):
        self.assertGreaterEqual(len(craft_catalog()["traditions"]["dravidian-tiered"]["rules"]), 4)
        result = built(GATE)
        self.assertIn("dravidian-tiered", construction_report(result.document, result.quality)["traditions"])


class TowerTests(unittest.TestCase):
    def test_every_tala_stands_inside_the_one_below(self):
        for product in (GATE, SHRINE):
            rules = built(product).document.metadata["dravidian_rules"]
            self.assertEqual(rules["tala_heights"], sorted(rules["tala_heights"], reverse=True))
            self.assertLess(rules["tala_heights"][-1], rules["tala_heights"][0])
            self.assertGreaterEqual(rules["summit_share"], .25 if product == GATE else .4)

        def swell_tala(document):
            for p in document.placements:
                if p.role == "tala_wall" and p.parameters["tala"] == 2:
                    p.position = (p.position[0] * 1.6, p.position[1], p.position[2] * 1.6)
        self.assertIn("dravidian.diminishing_talas", damaged(built(GATE), swell_tala))

        def widen_summit(document):
            top = max(p.parameters["tala"] for p in document.placements if p.role == "tala_wall")
            for p in document.placements:
                if p.role == "tala_wall" and p.parameters["tala"] == top:
                    p.position = (p.position[0] * 3, p.position[1], p.position[2] * 3)
        self.assertIn("dravidian.summit_share", damaged(built(GATE), widen_summit))

    def test_every_prastara_carries_its_hara_and_its_kudus(self):
        gate = built(GATE)
        for role in ("kuta_base", "kuta_post", "kuta_roof", "shala_wall", "shala_roof", "panjara", "kudu"):
            self.assertTrue(members(gate, role), role)

        def strip_kutas(document):
            document.placements = [p for p in document.placements
                                   if not (p.role == "kuta_base" and p.parameters.get("tala") == 2)]
        self.assertIn("dravidian.hara_each_tala", damaged(gate, strip_kutas))

        def strip_kudus(document):
            document.placements = [p for p in document.placements
                                   if not (p.role == "kudu" and p.parameters.get("tala") == 1)]
        self.assertIn("dravidian.kudu_each_eave", damaged(gate, strip_kudus))

    def test_pilasters_march_at_even_spacing(self):
        def shift_pilaster(document):
            host = next(p.parameters["host_id"] for p in document.placements
                        if p.role == "pilaster" and p.parameters.get("tala") == 1
                        and tuple(p.parameters["face"]) == ("x", -1))
            for p in document.placements:
                if p.parameters.get("host_id") == host:
                    p.position = (p.position[0] + .6, p.position[1], p.position[2])
        self.assertIn("dravidian.pilasters_march", damaged(built(GATE), shift_pilaster))

    def test_pots_crown_the_tower_in_mirrored_pairs(self):
        def lose_pot(document):
            document.placements.remove(next(p for p in document.placements if p.role == "kalasa"))
        self.assertIn("dravidian.kalasa_crowns_tower", damaged(built(GATE), lose_pot))

        def outgrow_pots(document):
            for p in document.placements:
                if p.role == "sikhara":
                    p.position = (p.position[0], p.position[1] + 6, p.position[2])
        self.assertIn("dravidian.kalasa_crowns_tower", damaged(built(SHRINE, "kuta-vimana"), outgrow_pots))

    def test_the_gopuram_is_a_gate_with_its_passage_open(self):
        result = built(GATE)
        rules = result.document.metadata["dravidian_rules"]
        self.assertGreaterEqual(rules["passage_width"], 2.4)
        self.assertGreater(rules["base_height"], .25 * 46)

        def block_passage(document):
            stone = next(p for p in document.placements if p.role == "base_wall")
            stone.position = (0, 5, 0)
        self.assertIn("dravidian.passage_open", damaged(result, block_passage))

    def test_the_vimana_is_granite_throughout_with_its_hall_before_it(self):
        for variant in ("kuta-vimana", "shala-vimana"):
            result = built(SHRINE, variant)
            self.assertEqual({p.substance for p in result.document.placements}, {"granite"})
            self.assertTrue(members(result, "mandapa_pier") and members(result, "sanctum_wall"))
            self.assertIn("sikhara", {p.role for p in result.document.placements})
            hall = [p for p in result.document.placements if p.parameters.get("mandapa")]
            sanctum = members(result, "sanctum_wall")
            self.assertLess(max(p.position[2] for p in hall), min(p.position[2] for p in sanctum) + 1)

    def test_every_tala_count_the_catalog_allows_fits_the_default_tower(self):
        for product, levels in ((GATE, 13), (SHRINE, 3)):
            with self.subTest(product=product, levels=levels):
                result = built(product, levels=levels)
                self.assertEqual(result.quality.status, "pass", failures(result.quality))
                self.assertEqual(result.document.metadata["dravidian_rules"]["talas"], levels)

    def test_random_towers_stay_inside_their_rules(self):
        for product in (GATE, SHRINE):
            controls = random_controls(product, random.Random(5))
            controls["detail"] = "concept"
            with self.subTest(product=product, controls=controls):
                result = build(BuildRequest(product=product, seed=5, **controls))
                self.assertEqual(result.quality.status, "pass", failures(result.quality))

    def test_controls_are_validated(self):
        bad = [dict(product=GATE, detail="concept", levels=4), dict(product=GATE, detail="concept", bays=13),
               dict(product=SHRINE, detail="concept", levels=10), dict(product=SHRINE, detail="concept", bays=2),
               dict(product=GATE, detail="concept", width=26, height=20, depth=13),
               dict(product=GATE, detail="concept", width=26, height=46, depth=40),
               dict(product=GATE, detail="concept", parameters={"gopura": 1})]
        for controls in bad:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(**controls))


if __name__ == "__main__":
    unittest.main()
