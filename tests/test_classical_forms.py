"""Regression coverage for the expanded classical design vocabulary."""
import random
import unittest
from copy import deepcopy

from core.architecture import BuildRequest, BuildDocument, build, describe_product, list_products, rebuild
from core.architecture.classical_forms import (
    FORMS,
    ORDERS,
    MASONRY_FORMS,
    ROLE_KINDS,
    random_controls,
    role_kind,
)


class ClassicalFormsTests(unittest.TestCase):
    def test_all_forms_orders_and_details_pass_and_roundtrip(self):
        for product in FORMS:
            variants = ("auto",) if product in MASONRY_FORMS else ORDERS
            for variant in variants:
                for detail in ("concept", "standard", "presentation"):
                    with self.subTest(product=product, variant=variant, detail=detail):
                        result = build(BuildRequest(product=product, variant=variant, detail=detail, seed=73))
                        self.assertEqual(result.quality.status, "pass",
                            [c.to_dict() for c in result.quality.checks if c.status == "fail"])
                        restored = BuildDocument.from_dict(result.document.to_dict())
                        self.assertEqual(restored.fingerprint(), result.document.fingerprint())
                        self.assertEqual(len(result.scene.blocks), len(result.document.placements))

    def test_five_temple_orders_have_distinct_geometry(self):
        geometries = []
        for order in ORDERS:
            result = build(BuildRequest(product="classical-building", variant=order, seed=13))
            self.assertEqual(result.quality.status, "pass")
            geometries.append([(p.block_type, p.position, p.dimensions) for p in result.document.placements])
            roles = {p.role for p in result.document.placements}
            if order in {"corinthian", "composite"}:
                self.assertIn("column_capital_foliage", roles)
            if order == "composite":
                self.assertIn("column_capital_volute", roles)
            if order == "tuscan":
                self.assertNotIn("triglyph", roles)
                self.assertNotIn("frieze_relief", roles)
        for i, geometry in enumerate(geometries):
            for other in geometries[i + 1:]:
                self.assertNotEqual(geometry, other)

    def test_random_order_is_deterministic_and_does_not_mutate_request(self):
        request = BuildRequest(product="classical-portico", variant="random", seed=4)
        before = deepcopy(request.to_dict())
        first, second = build(request), build(request)
        self.assertEqual(first.document.fingerprint(), second.document.fingerprint())
        self.assertEqual(request.to_dict(), before)

    def test_detached_roof_fails_intrinsic_quality(self):
        document = build(BuildRequest(product="classical-portico")).document
        document.metadata.clear()
        roof = next(p for p in document.placements if p.role == "roof")
        roof.position = (0, 100, 0)
        result = rebuild(document)
        failed = {c.code for c in result.quality.checks if c.status == "fail"}
        self.assertIn("classical.envelope_connectivity", failed)

    def test_portico_roof_is_coursed_ashlar(self):
        from core.architecture.masonry import placement_is_oversized

        document = build(BuildRequest(product="classical-portico")).document
        roofs = [p for p in document.placements if p.role == "roof"]
        self.assertGreater(len(roofs), 4)
        self.assertTrue(all(p.parameters.get("construction") == "ashlar running bond" for p in roofs))
        self.assertEqual([p.id for p in roofs if placement_is_oversized(p)], [])

    def test_missing_defining_components_fail_even_without_metadata(self):
        document = build(BuildRequest(product="classical-basilica")).document
        document.metadata.clear()
        document.placements = [p for p in document.placements if p.role != "apse"]
        result = rebuild(document)
        self.assertIn("classical.required_roles", {c.code for c in result.quality.checks if c.status == "fail"})

    def test_intrinsic_checks_survive_product_name_spelling(self):
        for spelling in ("Classical-Portico", "classical_portico", "  CLASSICAL_PORTICO  "):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    build(BuildRequest(product=spelling)).document.fingerprint(),
                    build(BuildRequest(product="classical-portico")).document.fingerprint(),
                )
                document = build(BuildRequest(product="classical-portico")).document
                document.metadata.clear()
                document.request.product = spelling
                roof = next(p for p in document.placements if p.role == "roof")
                roof.position = (0, 100, 0)
                result = rebuild(document)
                self.assertEqual(result.quality.status, "fail")
                self.assertIn("classical.envelope_connectivity",
                              {c.code for c in result.quality.checks if c.status == "fail"})

    def test_catalog_descriptors_are_detached_from_module_state(self):
        descriptor = describe_product("classical-basilica")
        expected = list(FORMS["classical-basilica"][2])
        self.assertEqual(descriptor["rules"]["required_roles"], expected)
        descriptor["rules"]["required_roles"].clear()
        descriptor["defaults"]["width"] = 999
        descriptor["variants"].append("bogus")
        self.assertEqual(list(FORMS["classical-basilica"][2]), expected)
        self.assertEqual(describe_product("classical-basilica")["rules"]["required_roles"], expected)
        listed = next(item for item in list_products() if item["name"] == "classical-basilica")
        self.assertEqual(listed["rules"]["required_roles"], expected)
        self.assertEqual(listed["defaults"]["width"], FORMS["classical-basilica"][1][0])
        document = build(BuildRequest(product="classical-basilica")).document
        document.metadata.clear()
        document.placements = [p for p in document.placements if p.role != "apse"]
        self.assertIn("classical.required_roles",
                      {c.code for c in rebuild(document).quality.checks if c.status == "fail"})

    def test_court_and_arena_centers_remain_open_above_ground(self):
        for product in ("classical-peristyle", "classical-forum", "classical-villa", "classical-amphitheatre"):
            result = build(BuildRequest(product=product))
            for p in result.document.placements:
                if p.role in {"foundation", "arena"}:
                    continue
                x, _, z = p.position
                w, _, d = p.dimensions
                self.assertFalse(abs(x) < w / 2 and abs(z) < d / 2, (product, p.id))
            from core.architecture.masonry import MAX_STRETCHER, placement_is_oversized
            walks = [p for p in result.document.placements if p.role == "court_walk"]
            if walks:
                self.assertEqual(
                    [p.id for p in walks
                     if placement_is_oversized(p)
                     or max(p.dimensions[0], p.dimensions[2]) > MAX_STRETCHER + 1e-6],
                    [],
                )

    def test_aqueduct_tiers_align_and_channel_remains_open(self):
        for levels in (1, 2, 3):
            result = build(BuildRequest(product="classical-aqueduct", levels=levels, bays=4))
            self.assertEqual(result.quality.status, "pass")
            arches = [p for p in result.document.placements if p.role == "arch"]
            self.assertGreater(len(arches), 4 * levels)
            from core.architecture.masonry import placement_is_oversized
            self.assertEqual([p.id for p in arches if placement_is_oversized(p)], [])
            walls = [p for p in result.document.placements if p.role == "channel_wall"]
            self.assertGreaterEqual(len(walls), 2)
            self.assertTrue(all(max(p.dimensions[0], p.dimensions[2]) <= 1.61 for p in walls))

    def test_incompatible_or_unsafe_requests_rejected(self):
        cases = [
            {"product": "classical-rotunda", "width": 20, "depth": 10},
            {"product": "classical-portico", "bays": 13},
            {"product": "classical-portico", "bays": True},
            {"product": "classical-portico", "width": 1, "height": 20},
            {"product": "classical-portico", "levels": 2},
            {"product": "classical-portico", "parameters": {"ignored": True}},
            {"product": "classical-portico", "openings": "door"},
            {"product": "classical-portico", "variant": "unknown"},
            {"product": "classical-aqueduct", "variant": "ionic"},
            {"product": "classical-aqueduct", "levels": 4},
            {"product": "classical-portico", "width": float("nan")},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                build(BuildRequest(**case))


    def test_masonry_forms_never_claim_a_column_order(self):
        for product in FORMS:
            with self.subTest(product=product):
                result = build(BuildRequest(product=product))
                declared = result.document.metadata["classical_rules"]["order"]
                stamped = {p.parameters.get("order") for p in result.document.placements}
                if product in MASONRY_FORMS:
                    self.assertIsNone(declared)
                    self.assertEqual(stamped, {None})
                else:
                    self.assertEqual(stamped, {declared})

    def test_placement_kind_is_coarser_than_role(self):
        catalog_kinds, catalog_roles = set(), set()
        for product in FORMS:
            for detail in ("concept", "standard", "presentation"):
                with self.subTest(product=product, detail=detail):
                    placements = build(BuildRequest(product=product, detail=detail)).document.placements
                    pairs = {(p.kind, p.role) for p in placements}
                    kinds = {kind for kind, _ in pairs}
                    roles = {role for _, role in pairs}
                    catalog_kinds |= kinds
                    catalog_roles |= roles
                    self.assertLessEqual(len(kinds), len(roles))
                    self.assertTrue(kinds <= set(ROLE_KINDS.values()))
                    for kind, role in pairs:
                        if role.startswith("column_") or role.startswith("capital_"):
                            self.assertEqual(kind, "column")
                        else:
                            self.assertEqual(kind, role_kind(role))
        self.assertLess(len(catalog_kinds), len(catalog_roles))

    def test_column_shafts_are_cubit_drums(self):
        from core.architecture.masonry import MAX_COURSE, MAX_STRETCHER, placement_is_oversized
        result = build(BuildRequest(product="classical-portico", detail="concept", seed=73))
        shafts = [p for p in result.document.placements if p.role == "column_shaft"]
        self.assertGreater(len(shafts), 12)
        self.assertTrue(all(p.parameters.get("unit") == "drum" for p in shafts))
        self.assertEqual(
            [p.id for p in shafts
             if placement_is_oversized(p)
             or p.dimensions[1] > MAX_COURSE + 1e-6
             or max(p.dimensions[0], p.dimensions[2]) > MAX_STRETCHER + 1e-6],
            [],
        )

    def test_column_parts_collapse_into_one_kind_across_orders(self):
        for order in ORDERS:
            with self.subTest(order=order):
                placements = build(BuildRequest(product="classical-portico", variant=order)).document.placements
                column_roles = {p.role for p in placements if p.kind == "column"}
                self.assertIn("column_shaft", column_roles)
                self.assertGreater(len(column_roles), 1)

    def test_required_kinds_invariant_is_declared_and_enforced(self):
        result = build(BuildRequest(product="classical-basilica"))
        self.assertEqual(result.document.metadata["invariants"]["required_kinds"],
                         describe_product("classical-basilica")["rules"]["required_kinds"])
        self.assertIn("invariant.required_kinds", {c.code for c in result.quality.checks})
        document = result.document
        document.placements = [p for p in document.placements if p.kind != "roof"]
        self.assertIn("invariant.required_kinds",
                      {c.code for c in rebuild(document).quality.checks if c.status == "fail"})

    def test_unmapped_role_is_rejected_rather_than_silently_kinded(self):
        with self.assertRaises(ValueError):
            role_kind("not-a-classical-role")

    def test_forum_hall_roof_stays_connected_across_deep_plans(self):
        for width, depth in ((28, 24), (16, 44), (14, 48), (20, 60), (16, 60), (44, 16)):
            with self.subTest(width=width, depth=depth):
                result = build(BuildRequest(product="classical-forum", width=width, depth=depth, height=9))
                self.assertEqual(result.quality.status, "pass",
                    [c.to_dict() for c in result.quality.checks if c.status == "fail"])

    def test_random_controls_stay_inside_every_order_envelope(self):
        for product in FORMS:
            variants = ("auto",) if product in MASONRY_FORMS else ORDERS
            drawn = set()
            for seed in range(3):
                controls = random_controls(product, random.Random(seed))
                drawn.add(tuple(sorted(controls.items(), key=str)))
                for variant in variants:
                    with self.subTest(product=product, seed=seed, variant=variant):
                        result = build(BuildRequest(product=product, variant=variant, **controls))
                        self.assertEqual(result.quality.status, "pass",
                            [c.to_dict() for c in result.quality.checks if c.status == "fail"])
            self.assertGreater(len(drawn), 1, product)

    def test_named_random_envelopes_keep_cavea_and_vaults_human_scale(self):
        """Controls from random_controls that welded cavea facets and tall vault cells.

        Seeds 0..2 of the order-envelope test drew these dimensions. A large
        cavea collapsed every seating facet into one slab, and a long bath
        vault stopped halving while the springing was still above one course.
        """
        from core.architecture.masonry import placement_is_oversized

        cases = (
            ("classical-amphitheatre", 0, "seating_tier",
             {"width": 39, "height": 10, "depth": 23, "bays": 6, "detail": "presentation"}),
            ("classical-theatre", 0, "seating_tier",
             {"width": 36, "height": 9, "depth": 21, "bays": 6, "detail": "presentation"}),
            ("classical-baths", 0, "hall_roof",
             {"width": 36, "height": 13, "depth": 23, "bays": 6, "detail": "presentation"}),
            ("classical-baths", 1, "hall_roof",
             {"width": 21, "height": 14, "depth": 29, "bays": 6, "detail": "concept"}),
            ("classical-baths", 2, "hall_roof",
             {"width": 38, "height": 15, "depth": 16, "bays": 3, "detail": "standard"}),
        )
        for product, seed, role, controls in cases:
            with self.subTest(product=product, seed=seed, role=role, **controls):
                result = build(BuildRequest(product=product, variant="auto", **controls))
                oversized = [p.id for p in result.document.placements
                             if p.role == role and placement_is_oversized(p)]
                self.assertEqual(oversized, [])
                self.assertEqual(result.quality.status, "pass",
                    [c.to_dict() for c in result.quality.checks if c.status == "fail"])

    def test_named_apse_roof_controls_keep_every_stone_envelope_distinct(self):
        """Narrow basilicas with many bays repeated apse roof primitive envelopes.

        The reported control is the first case: 13 m wide, 7 m high and 31 m
        deep at concept detail used to emit classical-basilica.apse_roof.0010,
        .0018, .0033 and .0038 twice. Each facet of the semi-conical shell was
        coursed on its own, so two neighbouring facets meeting on a shared
        meridian could stop on cells with the same axis-aligned envelope. The
        other cases are further random-control heights and depths that did the
        same thing at each detail level.
        """
        cases = (
            {"width": 13, "height": 7, "depth": 31, "bays": 7, "detail": "concept"},
            {"width": 13, "height": 7, "depth": 26, "bays": 3, "detail": "concept"},
            {"width": 16, "height": 7, "depth": 27, "bays": 9, "detail": "standard"},
            {"width": 19, "height": 7, "depth": 32, "bays": 6, "detail": "standard"},
            {"width": 13, "height": 7, "depth": 25, "bays": 4, "detail": "presentation"},
            {"width": 18, "height": 7, "depth": 32, "bays": 8, "detail": "presentation"},
        )
        facets = {"concept": 7, "standard": 11, "presentation": 15}
        for controls in cases:
            with self.subTest(**controls):
                result = build(BuildRequest(product="classical-basilica", variant="ionic", **controls))
                self.assertEqual(result.quality.status, "pass",
                    [c.to_dict() for c in result.quality.checks if c.status == "fail"])
                roofs = [p for p in result.document.placements if p.role == "apse_roof"]
                self.assertEqual({p.parameters.get("facet") for p in roofs},
                                 set(range(facets[controls["detail"]])))
                envelopes = [(p.position, p.dimensions) for p in roofs]
                self.assertEqual(len(envelopes), len(set(envelopes)))
        # The GUI draws this helper with the unseeded global RNG; replaying
        # fixed seeds keeps the same path deterministic.
        for seed in range(8):
            controls = random_controls("classical-basilica", random.Random(seed))
            with self.subTest(seed=seed, **controls):
                result = build(BuildRequest(product="classical-basilica", variant="ionic", **controls))
                self.assertEqual(result.quality.status, "pass",
                    [c.to_dict() for c in result.quality.checks if c.status == "fail"])


class ProductLookupTests(unittest.TestCase):
    def test_unknown_product_error_names_the_whole_catalog(self):
        with self.assertRaises(ValueError) as caught:
            build(BuildRequest(product="classical-portic"))
        message = str(caught.exception)
        self.assertIn("unknown product 'classical-portic'", message)
        for name in ("wall", "classical-building", *FORMS):
            self.assertIn(name, message)

    def test_blank_product_is_rejected_before_canonicalization(self):
        for value in ("", "   ", None, 7):
            with self.subTest(value=value), self.assertRaises(ValueError):
                describe_product(value)

    def test_request_serialization_uses_the_routing_spelling(self):
        request = BuildRequest(product="  CLASSICAL_PORTICO ")
        self.assertEqual(request.to_dict()["product"], "classical-portico")
        self.assertEqual(build(request).document.request.product, "classical-portico")


if __name__ == "__main__":
    unittest.main()
