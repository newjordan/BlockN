"""Mughal garden tomb: variants, sonnet lines, and rules measured from the members."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture.sonnet import verse

PRODUCT = "mughal-tomb"
TOMB = {"tomb_wall", "chamfer_wall", "pishtaq"}


def failed(document):
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


def top(p):
    return p.position[1] + p.dimensions[1] / 2


def shift(p, dx=0., dy=0., dz=0.):
    p.position = (p.position[0] + dx, p.position[1] + dy, p.position[2] + dz)


class MughalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = roll_request(PRODUCT, 7, variant="taj", detail="concept")
        cls.taj = build(cls.request)

    def stripped(self):
        document = deepcopy(self.taj.document)
        document.metadata.clear()
        return document

    def test_variants_build_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "mughal")
        for seed, (variant, minarets) in enumerate((("taj", 4), ("humayun", 0)), 30):
            with self.subTest(variant=variant):
                request = roll_request(PRODUCT, seed, variant=variant, detail="concept")
                result = build(request)
                self.assertEqual(result.quality.status, "pass")
                document = result.document
                self.assertEqual(document.fingerprint(), build(request).document.fingerprint())
                self.assertEqual(document.metadata["mughal_rules"]["minarets"], minarets)
                shafts = [p for p in document.placements if p.role == "minaret_shaft"]
                self.assertEqual(len({(p.position[0] > 0, p.position[2] > 0) for p in shafts}), minarets)
                size = result.quality.metrics["bounds_size"]
                for got, asked in zip(size, (request.width, request.height, request.depth)):
                    self.assertAlmostEqual(got, asked, places=4)
                stone = {p.substance for p in document.placements if p.role == "tomb_wall"}
                self.assertEqual(stone, {"marble" if variant == "taj" else "sandstone"})

    def test_plain_document_passes_after_metadata_is_stripped(self):
        self.assertEqual(self.taj.quality.status, "pass")
        self.assertFalse(failed(self.stripped()))

    def test_lines_are_seeded_pinned_and_validated(self):
        self.assertEqual(verse(self.request), self.taj.document.metadata["mughal_rules"]["verse"])
        request = BuildRequest(product=PRODUCT, detail="concept", seed=3,
                               parameters={"chhatris": "square", "niches": "single", "dome": "pointed"})
        document = build(request).document
        told = document.metadata["mughal_rules"]["verse"]
        self.assertEqual((told["chhatris"], told["niches"], told["dome"]), ("square", "single", "pointed"))
        self.assertEqual(told, verse(request))
        # Four square kiosks of four posts each.
        columns = [p for p in document.placements if p.role == "chhatri_column"]
        self.assertEqual({p.block_type for p in columns}, {"CUBE"})
        self.assertEqual(len({(round(p.position[0], 3), round(p.position[2], 3)) for p in columns}), 16)
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", parameters={"dome": "geodesic"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", width=44, depth=40, height=34))

    def test_symmetry_is_measured(self):
        document = self.stripped()
        shift(next(p for p in document.placements if p.role == "pishtaq"), dx=1.)
        self.assertIn("mughal.fourfold_symmetry", failed(document))

    def test_dome_must_crown_the_axis(self):
        document = self.stripped()
        for p in document.placements:
            if p.role == "dome":
                shift(p, dx=3.)
        self.assertIn("mughal.crowning_dome", failed(document))

    def test_pishtaq_must_rise_above_the_wall(self):
        document = self.stripped()
        head = max(top(p) for p in document.placements if p.role == "tomb_wall")
        document.placements = [p for p in document.placements
                               if not (p.role in {"pishtaq", "iwan_arch", "guldasta"} and top(p) > head + .4)]
        self.assertIn("mughal.pishtaq_rises", failed(document))

    def test_kiosks_and_minarets_keep_their_diagonals(self):
        document = self.stripped()
        for p in document.placements:
            if p.role.startswith("chhatri_") and p.position[0] > 0 and p.position[2] > 0:
                shift(p, dx=-3.)
        self.assertIn("mughal.diagonal_kiosks", failed(document))
        document = self.stripped()
        document.placements = [p for p in document.placements
                               if not (p.role.startswith("minaret_") and p.position[0] > 0 and p.position[2] > 0)]
        self.assertIn("mughal.diagonal_kiosks", failed(document))

    def test_tomb_must_stand_inside_the_plinth(self):
        document = self.stripped()
        for p in document.placements:
            if p.role in TOMB:
                shift(p, dx=9.)
        self.assertIn("mughal.tomb_on_plinth", failed(document))


if __name__ == "__main__":
    unittest.main()
