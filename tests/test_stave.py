"""Nordic Stave architect: stave churches and cantilevered timber lofts."""
from copy import deepcopy
import random
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild
from core.architecture.construction import construction_report, craft_catalog
from core.architecture.stave import random_controls

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


CHURCH, LOFT = "stave-church", "stave-loft"


class CatalogTests(unittest.TestCase):
    def test_products_are_described_with_substances_and_variants(self):
        church, loft = describe_product(CHURCH), describe_product(LOFT)
        self.assertEqual(church["architect_id"], "stave")
        self.assertEqual(set(church["substances"]), {"granite", "softwood", "oak", "cedar", "wrought-iron"})
        self.assertEqual(set(loft["substances"]), {"granite", "softwood", "oak", "cedar"})
        self.assertEqual(church["variants"], ["auto", "random", "borgund", "single-nave"])
        self.assertEqual(loft["variants"], ["auto", "random", "loft-house", "stabbur"])
        self.assertEqual(church["rules"]["levels"], [2, 3])
        self.assertEqual(loft["rules"]["bays"], [3, 5])

    def test_tradition_is_in_the_craft_catalog(self):
        self.assertGreaterEqual(len(craft_catalog()["traditions"]["nordic-stave"]["rules"]), 4)
        for product in (CHURCH, LOFT):
            result = built(product)
            self.assertIn("nordic-stave", construction_report(result.document, result.quality)["traditions"])


class StaveChurchTests(unittest.TestCase):
    def test_stone_sill_grounds_timber_staves_clear_of_earth(self):
        church = built(CHURCH, "borgund")
        self.assertEqual(church.quality.status, "pass", failures(church.quality))
        self.assertTrue(members(church, "stone_sill"))
        self.assertTrue(members(church, "nave_stave"))

        def sink_stave(document):
            stave = next(p for p in document.placements if p.role == "nave_stave")
            stave.position = (stave.position[0], stave.dimensions[1] / 2, stave.position[2])
        self.assertIn("stave.stone_sill_grounds_timber", damaged(church, sink_stave))

    def test_roofs_are_steep_and_tiered_above_svalgang(self):
        borgund = built(CHURCH, "borgund")
        single = built(CHURCH, "single-nave")
        self.assertTrue(members(borgund, "aisle_roof"))
        self.assertFalse(members(single, "aisle_roof"))
        for result in (borgund, single):
            self.assertEqual(result.quality.status, "pass", failures(result.quality))
            slopes = result.document.metadata["stave_rules"]["roof_slopes"]
            self.assertTrue(all(s >= 0.9 for s in slopes.values()), slopes)

        def flatten_nave_roof(document):
            for p in document.placements:
                if p.role == "nave_roof":
                    p.position = (p.position[0], p.position[1] - 6.0, p.position[2])
        self.assertIn("stave.tiered_silhouette", damaged(borgund, flatten_nave_roof))

    def test_takrytter_turret_and_dragon_finials_crown_the_church(self):
        church = built(CHURCH, "borgund")
        self.assertTrue(members(church, "turret_roof"))
        self.assertTrue(members(church, "spire_finial"))
        self.assertGreaterEqual(len(members(church, "dragon_finial")), 4)

        def drop_finial(document):
            for p in document.placements:
                if p.role == "spire_finial":
                    p.position = (p.position[0], p.position[1] - 8.0, p.position[2])
        self.assertIn("stave.ridge_turret_crowns_nave", damaged(church, drop_finial))

        def break_dragon_symmetry(document):
            dragon = next(p for p in document.placements if p.role == "dragon_finial")
            dragon.position = (dragon.position[0] + 1.5, dragon.position[1], dragon.position[2])
        self.assertIn("stave.paired_dragon_finials", damaged(church, break_dragon_symmetry))

    def test_choir_and_apse_project_east_of_nave(self):
        church = built(CHURCH, "borgund")
        self.assertTrue(members(church, "choir_plank") and members(church, "apse_plank"))

        def pull_apse_west(document):
            for p in document.placements:
                if p.role == "apse_plank":
                    p.position = (p.position[0], p.position[1], -10.0)
        self.assertIn("stave.apse_at_east", damaged(church, pull_apse_west))


class StaveLoftTests(unittest.TestCase):
    def test_loft_and_stabbur_cantilever_upper_gallery_with_external_stair(self):
        loft = built(LOFT, "loft-house")
        stabbur = built(LOFT, "stabbur")
        self.assertEqual(loft.quality.status, "pass", failures(loft.quality))
        self.assertEqual(stabbur.quality.status, "pass", failures(stabbur.quality))
        self.assertTrue(members(stabbur, "footing_pier"))
        self.assertFalse(members(loft, "footing_pier"))

        def remove_overhang(document):
            for p in document.placements:
                if p.role == "gallery_panel":
                    p.position = (p.position[0] * 0.4, p.position[1], 0.0)
                    p.dimensions = (p.dimensions[0] * 0.4, p.dimensions[1], p.dimensions[2])
        self.assertIn("stave.overhanging_upper_storey", damaged(loft, remove_overhang))

    def test_random_builds_and_controls_validation(self):
        for product in (CHURCH, LOFT):
            controls = random_controls(product, random.Random(11))
            controls["detail"] = "concept"
            with self.subTest(product=product, controls=controls):
                result = build(BuildRequest(product=product, seed=11, **controls))
                self.assertEqual(result.quality.status, "pass", failures(result.quality))
                self.assertEqual(BuildDocument.from_dict(result.document.to_dict()).fingerprint(),
                                 result.document.fingerprint())

        bad = [
            dict(product=CHURCH, detail="concept", bays=2),
            dict(product=CHURCH, detail="concept", levels=4),
            dict(product=CHURCH, detail="concept", width=14, height=40, depth=22),
            dict(product=LOFT, detail="concept", bays=6),
            dict(product=LOFT, detail="concept", levels=2),
            dict(product=LOFT, detail="concept", parameters={"unknown": 1}),
        ]
        for controls in bad:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(**controls))


if __name__ == "__main__":
    unittest.main()
