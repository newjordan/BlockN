"""The ziggurat's rules are measured from its members and fail when broken."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, rebuild, roll_request
from core.architecture.construction import construction_report

PRODUCT = "mesopotamian-ziggurat"


def failed(document, change):
    document = deepcopy(document)
    document.metadata.clear()           # rules must read the geometry, not the claims about it
    change(document)
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


class ZigguratTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ur = build(BuildRequest(product=PRODUCT, detail="concept", seed=73))

    def test_variants_build_and_replay(self):
        for variant, flights in (("triple-stair", {"axial", "left", "right"}), ("axial-stair", {"axial"})):
            with self.subTest(variant=variant):
                request = roll_request(PRODUCT, 31, variant=variant, detail="concept")
                result = build(request)
                self.assertEqual(result.quality.status, "pass")
                document = result.document
                self.assertEqual({p.parameters["flight"] for p in document.placements if p.role == "stair_step"}, flights)
                self.assertEqual({p.parameters["terrace"] for p in document.placements if p.role == "stage_face"},
                                 set(range(request.levels)))
                self.assertEqual(document.fingerprint(), build(request).document.fingerprint())
                self.assertIn("mesopotamian-ziggurat", construction_report(document, result.quality)["traditions"])

    def test_envelope_is_the_plot_whatever_the_verse(self):
        for seed in (1, 2, 3):
            result = build(BuildRequest(product=PRODUCT, detail="concept", seed=seed, levels=4))
            for got, asked in zip(result.quality.metrics["bounds_size"], (48, 22, 46)):
                self.assertAlmostEqual(got, asked, places=3)

    def test_deep_core_is_baked_brick_and_the_rest_sun_dried(self):
        tall = build(BuildRequest(product=PRODUCT, detail="concept", width=60, height=30, depth=60, seed=4))
        self.assertEqual(tall.quality.status, "pass")
        cores = [p for p in tall.document.placements if p.kind == "core"]
        mud = [p for p in cores if p.substance == "mud-brick"]
        baked = [p for p in cores if p.substance == "fired-clay-brick"]
        self.assertTrue(mud and baked)
        self.assertLessEqual(max(p.position[1] for p in baked), min(p.position[1] for p in mud))

    def test_envelope_outside_the_stated_proportions_is_refused(self):
        for controls in (dict(width=40, height=30, depth=40), dict(width=64, height=20, depth=38), dict(levels=6)):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(product=PRODUCT, **controls))

    def test_rules_fail_when_broken(self):
        document = self.ur.document
        self.assertEqual(self.ur.quality.status, "pass")

        def overhang(doc):
            for p in doc.placements:
                if p.role == "stage_face" and p.parameters["terrace"] == 2:
                    p.position = (p.position[0] + 12, *p.position[1:])
        self.assertIn("mesopotamian.receding_stages", failed(document, overhang))

        def plumb(doc):
            stage = [p for p in doc.placements if p.role == "stage_face" and p.parameters["terrace"] == 0]
            top = max(p.position[1] for p in stage)
            foot = min(p.position[0] - p.dimensions[0] / 2 for p in stage)
            for p in stage:
                if p.position[1] > top - .5 and p.position[0] < foot + 2:
                    p.position = (foot + p.dimensions[0] / 2, *p.position[1:])
        self.assertIn("mesopotamian.battered_faces", failed(document, plumb))

        def break_stair(doc):
            doc.placements = [p for p in doc.placements if not (
                p.role == "stair_step" and p.parameters.get("flight") == "axial" and 10 <= p.parameters["step"] < 16
                and p.parameters["terrace"] == 0)]
        self.assertIn("mesopotamian.processional_stair", failed(document, break_stair))

        def sink_shrine(doc):
            for p in doc.placements:
                if p.role.startswith("shrine"):
                    p.position = (p.position[0], p.position[1] - 6.5, p.position[2])
        self.assertIn("mesopotamian.summit_shrine", failed(document, sink_shrine))


if __name__ == "__main__":
    unittest.main()
