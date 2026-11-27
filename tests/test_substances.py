"""Substances, physical member properties, and the vertical-load check.

These tests cover the material layer of the semantic pipeline: the substance
registry and its sanity, every architect assigning a registered substance to
every member it emits, the physical aggregates (mass, centre of mass, bill of
materials, ground bearing) summing correctly, the substance-free document
contract keeping its fingerprints, and the reported vertical-load check on
every default product.
"""

import unittest

from core.architecture import BuildDocument, BuildRequest, build, rebuild, list_products
from core.architecture.schema import ArchitectInfo, Placement
from core.architecture.substances import REQUIRED_SUBSTANCE_IDS, SUBSTANCES
from core.architecture.physics import GRAVITY


# Fingerprint of a 4 x 0.25 x 3 floor slab (seed 909) as produced by the
# document contract before substances existed.  A document whose placements
# carry no substance must keep hashing to exactly this value.
SUBSTANCE_FREE_SLAB_FINGERPRINT = "361b404606d62f677408faca6fb25fc778553658698a51c3b763b896404494d0"

_MATERIAL = (200, 200, 200, 255)


def _placement(pid, substance, position, dimensions):
    return Placement(
        id=pid,
        kind="structure",
        block_type="CUBE",
        position=position,
        dimensions=dimensions,
        material=_MATERIAL,
        role="structure.test_member",
        parameters={"product": "physics-test"},
        parent_id=None,
        substance=substance,
    )


def _document(*placements):
    return BuildDocument(
        architect=ArchitectInfo("physics-test", "1.0.0", "Physics Test"),
        request=BuildRequest(architect="physics-test", product="physics-test"),
        placements=list(placements),
    )


class SubstanceRegistryTests(unittest.TestCase):
    def test_registry_covers_the_required_building_substances(self):
        self.assertGreaterEqual(len(SUBSTANCES), 25)
        for substance_id in REQUIRED_SUBSTANCE_IDS:
            self.assertIn(substance_id, SUBSTANCES, substance_id)
        self.assertEqual(len(set(SUBSTANCES)), len(SUBSTANCES))
        families = {substance.category for substance in SUBSTANCES.values()}
        self.assertLessEqual(
            {"stone", "earth", "binder", "timber", "roofing", "metal", "glass"},
            families,
        )

    def test_every_substance_has_physically_sane_properties(self):
        for substance in SUBSTANCES.values():
            with self.subTest(substance=substance.id):
                low, high = substance.density_range
                self.assertGreater(substance.density, 0.0)
                self.assertLess(low, high)
                self.assertLessEqual(low, substance.density)
                self.assertLessEqual(substance.density, high)
                self.assertGreater(substance.compressive_strength, 0.0)
                self.assertGreater(substance.tensile_strength, 0.0)
                self.assertGreater(substance.elastic_modulus, 0.0)
                self.assertGreater(substance.friction, 0.0)
                self.assertLess(substance.friction, 1.5)
                for field in ("roughness", "metallic"):
                    self.assertGreaterEqual(getattr(substance, field), 0.0)
                    self.assertLessEqual(getattr(substance, field), 1.0)
                for color in (substance.color_low, substance.color_high):
                    self.assertEqual(len(color), 3)
                    for channel in color:
                        self.assertIsInstance(channel, int)
                        self.assertTrue(0 <= channel <= 255)
                for low_channel, high_channel in zip(substance.color_low, substance.color_high):
                    self.assertLessEqual(low_channel, high_channel)
                self.assertTrue(substance.weathering)
                for line in substance.weathering:
                    self.assertGreater(len(line), 5)
                self.assertGreater(len(substance.basis), 10)
                self.assertGreater(len(substance.name), 0)


class ProductPhysicalTestCase(unittest.TestCase):
    """Shared once-per-session builds of every default catalog product."""

    @classmethod
    def setUpClass(cls):
        cls.records = _product_records()


_PRODUCT_RECORDS = {}


def _product_records():
    """Build every catalog product once and keep only measured records."""
    if not _PRODUCT_RECORDS:
        for descriptor in list_products():
            name = descriptor["name"]
            result = build(BuildRequest(product=name, seed=1729))
            physical = result.quality.metrics["physical"]
            substances = [str(p.substance) for p in result.document.placements]
            _PRODUCT_RECORDS[name] = {
                "descriptor_substances": descriptor.get("substances", []),
                "quality_status": result.quality.status,
                "member_count": len(result.document.placements),
                "substances": substances,
                "missing": [
                    str(p.id) for p in result.document.placements
                    if not isinstance(p.substance, str) or not p.substance
                ],
                "unknown": [
                    str(p.id) for p in result.document.placements
                    if isinstance(p.substance, str) and p.substance not in SUBSTANCES
                ],
                "masses": [member.mass_kg for member in result.physical.members],
                "volumes": [member.volume_m3 for member in result.physical.members],
                "physical": physical,
                "bill_raw": {
                    substance_id: dict(entry)
                    for substance_id, entry in result.physical.bill_of_materials.items()
                },
            }
    return _PRODUCT_RECORDS


class SubstanceAssignmentTests(ProductPhysicalTestCase):
    def test_every_products_every_member_has_a_registered_substance(self):
        for name, record in sorted(self.records.items()):
            with self.subTest(product=name, members=record["member_count"]):
                self.assertEqual(record["missing"], [])
                self.assertEqual(record["unknown"], [])
                self.assertEqual(len(record["substances"]), record["member_count"])
                self.assertTrue(record["substances"])
                # The catalog descriptor advertises what the architect uses.
                self.assertTrue(set(record["descriptor_substances"]))
                used = set(record["substances"])
                self.assertLessEqual(used, set(record["descriptor_substances"]))

    def test_describe_exposes_each_products_substance_vocabulary(self):
        for descriptor in list_products():
            with self.subTest(product=descriptor["name"]):
                substances = descriptor.get("substances")
                self.assertTrue(substances)
                for substance_id in substances:
                    self.assertIn(substance_id, SUBSTANCES)


class PhysicalAggregationTests(ProductPhysicalTestCase):
    def test_total_mass_equals_the_sum_over_members(self):
        for name, record in sorted(self.records.items()):
            with self.subTest(product=name, members=record["member_count"]):
                total = record["physical"]["total_mass_kg"]
                member_sum = sum(record["masses"])
                self.assertGreater(total, 0.0)
                self.assertAlmostEqual(total, member_sum, delta=max(1e-3, total * 1e-9))

    def test_bill_of_materials_sums_to_the_document_totals(self):
        for name, record in sorted(self.records.items()):
            with self.subTest(product=name):
                bill = record["bill_raw"]
                self.assertTrue(bill)
                mass_sum = sum(entry["mass_kg"] for entry in bill.values())
                volume_sum = sum(entry["volume_m3"] for entry in bill.values())
                member_mass = sum(record["masses"])
                member_volume = sum(record["volumes"])
                member_count = sum(entry["members"] for entry in bill.values())
                self.assertAlmostEqual(mass_sum, member_mass, delta=member_mass * 1e-9 + 1e-3)
                self.assertAlmostEqual(volume_sum, member_volume, delta=member_volume * 1e-9 + 1e-6)
                self.assertEqual(member_count, record["member_count"])
                # The exported bill rounds each substance, so its published
                # total stays within the rounding of its entries.
                published = record["physical"]["bill_of_materials"]
                published_mass = sum(entry["mass_tonnes"] * 1000.0 for entry in published.values())
                self.assertAlmostEqual(published_mass, mass_sum, delta=len(published))
                # Every substance in the bill is one the members actually use.
                self.assertEqual(set(bill), set(record["substances"]))
                for substance_id in bill:
                    self.assertIn(substance_id, SUBSTANCES)

    def test_footprint_and_ground_bearing_are_positive_and_finite(self):
        for name, record in sorted(self.records.items()):
            with self.subTest(product=name):
                physical = record["physical"]
                self.assertGreater(physical["footprint_area_m2"], 0.0)
                pressure = physical["ground_bearing_pressure_kpa"]
                self.assertGreater(pressure, 0.0)
                self.assertLess(pressure, 2000.0)
                centre = physical["center_of_mass_m"]
                self.assertEqual(len(centre), 3)
                # X and Z live in the world frame; the centre of mass is a
                # finite point whose height sits above the ground plane.
                for value in centre:
                    self.assertNotEqual(float(value), float("inf"))
                    self.assertNotEqual(float(value), float("-inf"))
                self.assertGreaterEqual(centre[1], 0.0)


class StructuralCheckTests(ProductPhysicalTestCase):
    def test_default_products_build_with_no_member_over_utilisation(self):
        """The reported vertical-load check on every default product.

        The per-product mass and maximum utilisation are recorded in the
        subTest description and the printed table so a regression shows the
        numbers it regressed from.
        """
        rows = []
        for name, record in sorted(self.records.items()):
            physical = record["physical"]
            rows.append(
                f"{name:28s} {record['quality_status']:>7s} "
                f"mass={physical['total_mass_kg'] / 1000.0:9.1f}t "
                f"max_util={physical['max_member_utilisation']:6.3f} "
                f"at={physical['max_utilisation_member_id'] or '-'}"
            )
            with self.subTest(
                product=name,
                mass_tonnes=round(physical["total_mass_kg"] / 1000.0, 1),
                max_utilisation=round(physical["max_member_utilisation"], 3),
                unresolved=physical["members_without_resolved_bearing"],
            ):
                self.assertEqual(record["quality_status"], "pass")
                self.assertEqual(physical["members_over_utilisation"], [])
                self.assertGreater(physical["max_member_utilisation"], 0.0)
                self.assertLessEqual(physical["max_member_utilisation"], 1.0)
        print("\n".join(rows))

    def test_stacked_members_accumulate_vertical_load(self):
        base = _placement("base", "limestone", (0.0, 0.25, 0.0), (2.0, 0.5, 2.0))
        cap = _placement("cap", "granite", (0.0, 1.0, 0.0), (1.0, 1.0, 1.0))
        result = rebuild(_document(base, cap))
        members = {member.primitive_id: member for member in result.physical.members}

        cap_mass = 1.0 * 2700.0
        base_mass = 2.0 * 0.5 * 2.0 * 2200.0
        self.assertAlmostEqual(members["cap"].mass_kg, cap_mass, places=3)
        self.assertAlmostEqual(members["cap"].supported_load_kg, cap_mass, places=3)
        self.assertAlmostEqual(members["cap"].contact_area_m2, 1.0, places=6)
        expected_cap_stress = cap_mass * GRAVITY / 1.0 / 1.0e6
        self.assertAlmostEqual(members["cap"].bearing_stress_mpa, expected_cap_stress, places=9)
        self.assertAlmostEqual(
            members["base"].supported_load_kg, base_mass + cap_mass, places=3
        )
        # The base beds on the ground across its own footprint.
        self.assertAlmostEqual(members["base"].contact_area_m2, 4.0, places=6)
        expected_base_stress = (base_mass + cap_mass) * GRAVITY / 4.0 / 1.0e6
        self.assertAlmostEqual(members["base"].bearing_stress_mpa, expected_base_stress, places=9)
        self.assertEqual(result.quality.status, "pass")

    def test_a_plate_shares_its_load_between_the_piers_it_rests_on(self):
        pier_a = _placement("pier.a", "limestone", (-1.0, 0.5, 0.0), (0.5, 1.0, 0.5))
        pier_b = _placement("pier.b", "limestone", (1.0, 0.5, 0.0), (0.5, 1.0, 0.5))
        plate = _placement("plate", "sandstone", (0.0, 1.1, 0.0), (2.0, 0.2, 0.5))
        result = rebuild(_document(pier_a, pier_b, plate))
        members = {member.primitive_id: member for member in result.physical.members}

        plate_mass = 2.0 * 0.2 * 0.5 * 2200.0
        self.assertAlmostEqual(members["plate"].supported_load_kg, plate_mass, places=3)
        received = members["pier.a"].supported_load_kg - members["pier.a"].mass_kg
        other = members["pier.b"].supported_load_kg - members["pier.b"].mass_kg
        self.assertAlmostEqual(received, plate_mass / 2.0, places=3)
        self.assertAlmostEqual(other, plate_mass / 2.0, places=3)

    def test_over_utilisation_is_reported_without_failing_the_build(self):
        weak = _placement("mud", "mud-brick", (0.0, 0.1, 0.0), (0.5, 0.2, 0.5))
        burden = _placement("load", "granite", (0.0, 10.2, 0.0), (0.4, 20.0, 0.4))
        result = rebuild(_document(weak, burden))
        self.assertEqual(result.quality.status, "warning")
        checks = {check.code: check for check in result.quality.checks}
        structure = checks["structure.vertical_load"]
        self.assertEqual(structure.status, "warning")
        self.assertEqual(structure.severity, "warning")
        self.assertIn(weak.id, structure.primitive_ids)
        self.assertGreater(
            result.quality.metrics["physical"]["max_member_utilisation"], 1.0
        )
        # The bearing gate is reported, never a hard failure.
        self.assertEqual(result.quality.counts["fail"], 0)

    def test_a_document_that_predates_substances_still_passes(self):
        base = _placement("base", None, (0.0, 0.25, 0.0), (2.0, 0.5, 2.0))
        cap = _placement("cap", None, (0.0, 1.0, 0.0), (1.0, 1.0, 1.0))
        result = rebuild(_document(base, cap))
        checks = {check.code: check for check in result.quality.checks}
        self.assertEqual(checks["document.substances"].status, "pass")
        self.assertEqual(checks["structure.vertical_load"].status, "pass")
        self.assertEqual(result.quality.metrics["physical"]["members_without_substance"], 2)

    def test_a_partly_assigned_document_is_a_quality_failure(self):
        base = _placement("base", "limestone", (0.0, 0.25, 0.0), (2.0, 0.5, 2.0))
        cap = _placement("cap", None, (0.0, 1.0, 0.0), (1.0, 1.0, 1.0))
        result = rebuild(_document(base, cap))
        checks = {check.code: check for check in result.quality.checks}
        self.assertEqual(checks["document.substances"].status, "fail")
        self.assertIn(cap.id, checks["document.substances"].primitive_ids)
        self.assertEqual(result.quality.status, "fail")

    def test_an_unknown_substance_is_a_quality_failure(self):
        base = _placement("base", "limestone", (0.0, 0.25, 0.0), (2.0, 0.5, 2.0))
        cap = _placement("cap", "unobtainium", (0.0, 1.0, 0.0), (1.0, 1.0, 1.0))
        result = rebuild(_document(base, cap))
        checks = {check.code: check for check in result.quality.checks}
        self.assertEqual(checks["document.substances"].status, "fail")
        self.assertEqual(result.quality.status, "fail")


class DocumentContractTests(unittest.TestCase):
    def test_documents_without_substances_keep_their_fingerprints(self):
        result = build(
            BuildRequest(
                product="slab",
                seed=909,
                width=4.0,
                height=0.25,
                depth=3.0,
                variant="floor",
            )
        )
        with_substances = result.document.fingerprint()
        self.assertNotEqual(with_substances, SUBSTANCE_FREE_SLAB_FINGERPRINT)

        for placement in result.document.placements:
            placement.substance = None
        stripped = result.document.fingerprint()
        self.assertEqual(stripped, SUBSTANCE_FREE_SLAB_FINGERPRINT)

        payload = result.document.to_dict()
        for placement in payload["placements"]:
            self.assertNotIn("substance", placement)

    def test_placement_payload_omits_an_unset_substance(self):
        placement = _placement("member", None, (0.0, 0.5, 0.0), (1.0, 1.0, 1.0))
        self.assertNotIn("substance", placement.to_dict())
        restored = Placement.from_dict(placement.to_dict())
        self.assertIsNone(restored.substance)

        placement.substance = "limestone"
        payload = placement.to_dict()
        self.assertEqual(payload["substance"], "limestone")
        self.assertEqual(Placement.from_dict(payload).substance, "limestone")

    def test_documents_round_trip_with_substances(self):
        result = build(BuildRequest(product="wall", openings="mixed", seed=41))
        original = result.document.fingerprint()
        substances = {p.id: p.substance for p in result.document.placements}
        self.assertTrue(all(substances.values()))

        first = BuildDocument.from_dict(result.document.to_dict())
        second = BuildDocument.from_dict(first.to_dict())
        self.assertEqual(first.fingerprint(), original)
        self.assertEqual(second.fingerprint(), original)
        self.assertEqual(
            {p.id: p.substance for p in second.placements},
            substances,
        )
        # A loaded document recompiles with the same quality outcome.
        rebuilt = rebuild(second)
        self.assertEqual(rebuilt.quality.status, result.quality.status)
        self.assertEqual(
            rebuilt.quality.metrics["substance_counts"],
            result.quality.metrics["substance_counts"],
        )


if __name__ == "__main__":
    unittest.main()
