"""Stupa variants, verses and audits measured from the built members."""
from copy import deepcopy
import math
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture import sonnet
from core.architecture.stupa import GATEWAY, VARIANTS

PRODUCT = "buddhist-stupa"


def failed(result, damage):
    """Codes that fail after ``damage`` edits a copy whose emitted claims have been stripped."""
    document = deepcopy(result.document)
    document.metadata.clear()
    damage(document)
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


def move(p, dx=0., dy=0., dz=0.):
    p.position = (p.position[0] + dx, p.position[1] + dy, p.position[2] + dz)


class StupaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {variant: build(roll_request(PRODUCT, 30 + n, variant=variant, detail="concept"))
                     for n, variant in enumerate(VARIANTS)}

    def test_variants_build_fill_the_plot_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "stupa")
        for variant, result in self.built.items():
            with self.subTest(variant=variant):
                self.assertEqual(result.quality.status, "pass")
                request = result.document.request
                size = result.quality.metrics["bounds_size"]
                for got, asked in zip(size, (request.width, request.height, request.depth)):
                    self.assertAlmostEqual(got, asked, places=4)
                self.assertEqual(result.document.fingerprint(), build(request).document.fingerprint())
                roles = {p.role for p in result.document.placements}
                self.assertEqual("torana_architrave" in roles, variant == "great-stupa")
                self.assertEqual("vahalkada" in roles, variant == "dagoba")
                self.assertEqual("terrace_stair" in roles, variant == "great-stupa")
                self.assertTrue(any(p.parameters.get("unit") == "core" for p in result.document.placements))

    def test_verse_is_readable_without_building_and_lines_can_be_pinned(self):
        for result in self.built.values():
            self.assertEqual(sonnet.verse(result.document.request), result.document.metadata["stupa_rules"]["verse"])
        counts = {}
        for answer, discs in (("single", 1), ("triple", 3), ("seven", 7)):
            request = BuildRequest(product=PRODUCT, detail="concept", seed=5, parameters={"parasols": answer})
            result = build(request)
            self.assertEqual(result.quality.status, "pass")
            self.assertEqual(result.document.metadata["stupa_rules"]["verse"]["parasols"], answer)
            counts[answer] = sum(p.role == "chattra" for p in result.document.placements)
            self.assertEqual(counts[answer], discs)
            self.assertAlmostEqual(result.quality.metrics["bounds_size"][1], 17, places=4)
        two = build(BuildRequest(product=PRODUCT, detail="concept", seed=5, parameters={"torana": "two-architrave"}))
        self.assertEqual(sum(p.role == "torana_architrave" for p in two.document.placements), 4 * 2)
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", parameters={"anda": "onion"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", width=36, height=30, depth=36))

    def test_every_anda_and_terrace_builds_in_both_variants(self):
        combos = [(variant, anda, ("single", "double")[(i + j) % 2])
                  for i, variant in enumerate(VARIANTS) for j, anda in enumerate(("hemisphere", "bubble", "bell"))]
        for variant, anda, terrace in combos:
            with self.subTest(variant=variant, anda=anda, terrace=terrace):
                result = build(BuildRequest(product=PRODUCT, variant=variant, detail="concept", seed=9,
                                            parameters={"anda": anda, "terrace": terrace}))
                self.assertEqual(result.quality.status, "pass")

    def test_path_must_stay_clear(self):
        for variant, result in self.built.items():
            with self.subTest(variant=variant):
                def block(document):
                    rail = max(abs(p.position[0]) for p in document.placements if p.role == "vedika_coping")
                    disc = next(p for p in document.placements if p.role == "chattra")
                    disc.position = (rail - 1.4, .5, .9)
                self.assertIn("stupa.clear_path", failed(result, block))

    def test_gateways_stand_on_all_four_axes(self):
        for variant, result in self.built.items():
            with self.subTest(variant=variant):
                def lose_north(document):
                    document.placements = [p for p in document.placements if not (
                        p.role in GATEWAY and p.position[2] > abs(p.position[0]))]
                self.assertIn("stupa.cardinal_gateways", failed(result, lose_north))

                def shift_east(document):
                    for p in document.placements:
                        if p.role in GATEWAY and p.position[0] > abs(p.position[2]):
                            move(p, dz=.6)
                self.assertIn("stupa.cardinal_gateways", failed(result, shift_east))

    def test_dome_sits_on_the_drum(self):
        result = self.built["great-stupa"]

        def lift(document):
            for p in document.placements:
                if p.role == "anda":
                    move(p, dy=1.0)
        self.assertIn("stupa.dome_on_drum", failed(result, lift))

        def slide(document):
            walk = min(math.hypot(p.position[0], p.position[2]) for p in document.placements
                       if p.role == "terrace_railing")
            for p in document.placements:
                if p.role == "anda":
                    move(p, dx=walk * .25)
        self.assertIn("stupa.dome_on_drum", failed(result, slide))

    def test_crown_keeps_the_axis_and_parasols_diminish(self):
        for variant, result in self.built.items():
            with self.subTest(variant=variant):
                def lean(document):
                    for p in document.placements:
                        if p.role == "chattra":
                            move(p, dx=.4)
                self.assertIn("stupa.axis", failed(result, lean))
        result = build(BuildRequest(product=PRODUCT, detail="concept", seed=5, parameters={"parasols": "triple"}))

        def invert(document):
            discs = sorted((p for p in document.placements if p.role == "chattra"), key=lambda p: p.position[1])
            low, high = discs[0], discs[-1]
            low.position, high.position = ((*low.position[:1], high.position[1], low.position[2]),
                                           (*high.position[:1], low.position[1], high.position[2]))
        self.assertIn("stupa.axis", failed(result, invert))

    def test_variations_deal_different_verses(self):
        requests = sonnet.variations(PRODUCT, 8, seed=3, hold={"anda": "bell"})
        self.assertEqual(len({sonnet.stanza(r) for r in requests}), 8)
        self.assertTrue(all(sonnet.verse(r)["anda"] == "bell" for r in requests))
        self.assertEqual(sonnet.verse_count(PRODUCT), 2 * 3 * 3 * 2 * 2)


if __name__ == "__main__":
    unittest.main()
