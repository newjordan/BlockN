"""Geodesic dome: both layers build, the verse is the request's, and each rule fails when broken."""
from copy import deepcopy
import unittest

import numpy as np

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture import geodesic, sonnet

PRODUCT = "geodesic-dome"


def failed(document, change):
    """Codes that fail once ``change`` has damaged a copy with its metadata stripped."""
    document = deepcopy(document)
    document.metadata.clear()
    change(document)
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


class GeodesicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = BuildRequest(product=PRODUCT, detail="concept", seed=73, parameters={"base": "legs"})
        cls.result = build(cls.request)

    def test_variants_build_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "geodesic")
        for variant in geodesic.VARIANTS:
            with self.subTest(variant=variant):
                request = roll_request(PRODUCT, 31, variant=variant, detail="concept")
                self.assertEqual(request.width, request.depth)
                result = build(request)
                self.assertEqual(result.quality.status, "pass")
                document = result.document
                self.assertEqual(document.fingerprint(), build(request).document.fingerprint())
                rules = document.metadata["geodesic_rules"]
                self.assertEqual(rules["verse"], sonnet.verse(request))
                self.assertEqual(rules["layers"], 1 + geodesic.VARIANTS.index(variant))
                self.assertEqual({p.parameters.get("layer") for p in document.placements if p.role == "hub"},
                                 {"outer", "inner"} if variant == "double-layer" else {"outer"})
                size = result.quality.metrics["bounds_size"]
                for got, asked in zip(size, (request.width, request.height, request.depth)):
                    self.assertAlmostEqual(got, asked, places=6)

    def test_stripped_document_still_passes_its_rules(self):
        self.assertEqual(failed(self.result.document, lambda document: None), set())

    def test_lattice_counts_follow_the_frequency(self):
        for frequency, families in geodesic.FAMILIES.items():
            nodes, edges, faces = geodesic.lattice(frequency)
            self.assertEqual((len(nodes), len(edges), len(faces)),
                             (10 * frequency ** 2 + 2, 30 * frequency ** 2, 20 * frequency ** 2))
            lengths = [float(np.linalg.norm(nodes[a] - nodes[b])) for a, b in edges]
            self.assertEqual(geodesic._families(lengths), families)
            self.assertAlmostEqual(float(nodes[:, 1].max()), 1.0)

    def test_pinned_lines_are_honoured(self):
        for frequency, skin in (("2v", "open-frame"), ("4v", "alternating")):
            request = BuildRequest(product=PRODUCT, detail="concept", seed=5, variant="single-layer",
                                   parameters={"frequency": frequency, "skin": skin, "entry": "lifted-skirt"})
            result = build(request)
            self.assertEqual(result.quality.status, "pass")
            rules = result.document.metadata["geodesic_rules"]
            self.assertEqual(rules["verse"], sonnet.verse(request))
            self.assertEqual((rules["verse"]["frequency"], rules["frequency"]), (frequency, int(frequency[0])))
            roles = {p.role for p in result.document.placements}
            self.assertEqual("skin_panel" in roles, skin == "alternating")
            self.assertEqual("glass_panel" in roles, skin != "open-frame")
            self.assertNotIn("portal_post", roles)
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, parameters={"frequency": "5v"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, width=40, height=28, depth=36))

    def test_a_hub_off_the_sphere_is_found(self):
        def push(document):
            hub = next(p for p in document.placements if p.role == "hub" and p.position[1] > 12 and abs(p.position[0]) > 3)
            hub.position = (hub.position[0] * 1.15, hub.position[1], hub.position[2] * 1.15)
        self.assertIn("geodesic.spherical", failed(self.result.document, push))

    def test_a_missing_strut_breaks_the_triangulation(self):
        def cut(document):
            high = max((p for p in document.placements if p.role == "strut"), key=lambda p: p.position[1])
            document.placements = [p for p in document.placements if p.id != high.id]
        self.assertIn("geodesic.triangulated", failed(self.result.document, cut))

    def test_an_odd_strut_length_is_found(self):
        def stretch(document):
            strut = max((p for p in document.placements if p.role == "strut"), key=lambda p: p.position[1])
            strut.dimensions = tuple(v * 1.2 for v in strut.dimensions)
        self.assertIn("geodesic.strut_families", failed(self.result.document, stretch))

    def test_a_base_hub_without_a_footing_or_leg_is_found(self):
        def unseat(document):
            footing = next(p for p in document.placements if p.role == "footing")
            document.placements = [p for p in document.placements if p.id != footing.id]
        self.assertIn("geodesic.grounded_base", failed(self.result.document, unseat))

        def unleg(document):
            leg = next(p for p in document.placements if p.role == "leg")
            document.placements = [p for p in document.placements if p.id != leg.id]
        self.assertIn("geodesic.grounded_base", failed(self.result.document, unleg))


if __name__ == "__main__":
    unittest.main()
