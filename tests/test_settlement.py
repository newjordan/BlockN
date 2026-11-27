"""Settlements: streets, plots, and buildings borrowed from the other architects."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.architecture import BuildDocument, BuildRequest, build, rebuild
from core.architecture.schema import Placement
from core.architecture.service import author_document
from core.architecture.settlement import (
    DEFAULT_SITES,
    ENGLISH_VARIANTS,
    ROMAN_VARIANTS,
    build_document,
    turn_members,
    unturn_members,
)
from core.architecture.io import export_bundle
from core.architecture.substances import SUBSTANCES


SEEDS = (
    17, 42, 99, 128, 256, 512, 1024, 1729, 2048, 3001,
    4096, 5555, 7777, 8192, 9001, 12345, 23456, 34567, 45678, 54321,
)


def _concept(product, variant, seed):
    width, depth = DEFAULT_SITES[variant]
    return build(BuildRequest(
        product=product, variant=variant, width=width, depth=depth,
        detail="concept", seed=seed,
    ))


def _by_plot(document):
    groups = {}
    for placement in document.placements:
        plot_id = (placement.parameters or {}).get("plot_id")
        if plot_id:
            groups.setdefault(plot_id, []).append(placement)
    return groups


def _root(placement, by_id):
    seen = set()
    current = placement
    while current.parent_id is not None:
        if current.id in seen:
            raise AssertionError(f"parent cycle at {current.id}")
        seen.add(current.id)
        current = by_id[current.parent_id]
    return current


def _statuses(result):
    return {check.code: check.status for check in result.quality.checks}


def _close(left, right, tol=1e-6):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_close(left[key], right[key], tol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_close(a, b, tol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return abs(float(left) - float(right)) <= tol
    return left == right


class SettlementIdentityTests(unittest.TestCase):
    def test_same_seed_repeats_and_one_plot_override_leaves_the_others(self):
        width, depth = DEFAULT_SITES["street-village"]
        request = BuildRequest(
            product="english-village", variant="street-village", detail="concept", seed=41,
            width=width, depth=depth, height=12, bays=6, levels=2, style="timber",
        )
        first = build_document(request)
        second = build_document(BuildRequest.from_dict(request.to_dict()))
        self.assertEqual(first.fingerprint(), second.fingerprint())
        # Height, bays and levels are single-building controls. Dropping them
        # must not change the town.
        plain = build_document(BuildRequest(
            product="english-village", variant="street-village", detail="concept", seed=41,
            width=width, depth=depth,
        ))
        self.assertEqual(plain.fingerprint(), first.fingerprint())
        other = build_document(BuildRequest(
            product="english-village", variant="street-village", detail="concept", seed=42,
            width=width, depth=depth,
        ))
        self.assertNotEqual(other.fingerprint(), first.fingerprint())

        overridden = build_document(BuildRequest(
            product="english-village", variant="street-village", detail="concept", seed=41,
            width=width, depth=depth,
            parameters={"plot_overrides": {"plot-house-01": {
                "variant": "twin-gable",
                "parameters": {"finish": "rose-shale", "porch": False, "chimney": False},
            }}},
        ))
        base = _by_plot(first)
        changed = _by_plot(overridden)
        self.assertEqual(set(base), set(changed))
        house = next(item for item in changed["plot-house-01"] if item.role == "plot")
        self.assertEqual(house.parameters["building_request"]["variant"], "twin-gable")
        self.assertNotEqual(
            [item.to_dict() for item in base["plot-house-01"]],
            [item.to_dict() for item in changed["plot-house-01"]],
        )
        for plot_id in base:
            if plot_id == "plot-house-01":
                continue
            self.assertEqual(
                [item.to_dict() for item in base[plot_id]],
                [item.to_dict() for item in changed[plot_id]],
                plot_id,
            )
        street_roles = {"street", "kerb", "stepping_stone", "square"}
        base_ground = [item.to_dict() for item in first.placements if item.role in street_roles]
        changed_ground = [item.to_dict() for item in overridden.placements if item.role in street_roles]
        self.assertEqual(base_ground, changed_ground)

    def test_ids_are_unique_and_parent_chains_end_at_a_plot_or_a_street(self):
        for product, variant in (("english-village", "street-village"), ("roman-town", "forum-quarter")):
            document = _concept(product, variant, 5).document
            ids = [placement.id for placement in document.placements]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(ids))
            by_id = {placement.id: placement for placement in document.placements}
            for placement in document.placements:
                root = _root(placement, by_id)
                self.assertIn(root.role, {"plot", "street"}, placement.id)
                self.assertIsNone(root.parent_id)
            for placement in document.placements:
                self.assertIn(placement.substance, SUBSTANCES, placement.id)

    def test_unknown_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            build_document(BuildRequest(
                product="english-village", detail="concept", parameters={"moat": True},
            ))


class SettlementGateTests(unittest.TestCase):
    def test_defaults_pass_overlap_frontage_and_network(self):
        for product, variant in (("english-village", "street-village"), ("roman-town", "forum-quarter")):
            result = _concept(product, variant, 1729)
            self.assertEqual(result.quality.status, "pass", product)
            status = _statuses(result)
            for code in (
                "settlement.overlap", "settlement.frontage",
                "settlement.street_network", "settlement.building_quality",
                "settlement.ground_coverage", "settlement.street_frontage",
                "settlement.insula_coverage",
            ):
                self.assertEqual(status[code], "pass", code)
            physical = result.quality.metrics["physical"]
            self.assertGreater(physical["total_mass_kg"], 0)
            bill = physical["bill_of_materials"]
            self.assertTrue(bill)
            used = {placement.substance for placement in result.document.placements}
            self.assertEqual(set(bill), used)

    def test_broken_overlap_frontage_and_network_each_fail(self):
        result = _concept("english-village", "street-village", 9)
        document = result.document

        overlap = BuildDocument.from_dict(document.to_dict())
        moved = False
        for placement in overlap.placements:
            if (placement.parameters or {}).get("plot_id") == "plot-house-02" and placement.role != "plot":
                x, y, z = placement.position
                placement.position = (x + 8.0, y, z)
                moved = True
        self.assertTrue(moved)
        overlap_status = _statuses(rebuild(overlap))
        self.assertEqual(overlap_status["settlement.overlap"], "fail")

        frontage = BuildDocument.from_dict(document.to_dict())
        plot = next(placement for placement in frontage.placements if placement.id == "plot-house-01")
        plot.parameters["yaw"] = (int(plot.parameters["yaw"]) + 180) % 360
        frontage_status = _statuses(rebuild(frontage))
        self.assertEqual(frontage_status["settlement.frontage"], "fail")
        self.assertEqual(frontage_status["settlement.overlap"], "pass")
        self.assertEqual(frontage_status["settlement.street_network"], "pass")

        network = BuildDocument.from_dict(document.to_dict())
        streets = [placement for placement in network.placements if placement.role == "street"]
        self.assertEqual(len(streets), 1)
        street = streets[0]
        y = street.position[1]
        height = street.dimensions[1]
        street.position = (20.0, y, 20.0)
        street.dimensions = (2.0, height, 2.0)
        network_status = _statuses(rebuild(network))
        self.assertEqual(network_status["settlement.street_network"], "fail")

        bare = BuildDocument.from_dict(document.to_dict())
        bare.placements = [
            placement for placement in bare.placements
            if not (
                placement.role in {"plot", "garden"}
                and (placement.parameters or {}).get("plot_id") == "plot-house-03"
            )
        ]
        self.assertLess(len(bare.placements), len(document.placements))
        self.assertEqual(_statuses(rebuild(bare))["settlement.ground_coverage"], "fail")

        pulled = BuildDocument.from_dict(document.to_dict())
        facing = {
            placement.id: (placement.parameters or {}).get("facing")
            for placement in pulled.placements if placement.role == "plot"
        }
        moved = False
        for placement in pulled.placements:
            plot_id = (placement.parameters or {}).get("plot_id")
            if facing.get(plot_id) != "south" or placement.role in _INFRA_ROLES:
                continue
            x, y, z = placement.position
            placement.position = (x, y, z + 10.0)
            moved = True
        self.assertTrue(moved)
        self.assertEqual(_statuses(rebuild(pulled))["settlement.street_frontage"], "fail")

        roman = _concept("roman-town", "residential-quarter", 9).document
        shrunk = BuildDocument.from_dict(roman.to_dict())
        residential = {
            item["id"] for item in shrunk.metadata["settlement"]["insulae"]
            if item.get("use") == "residential"
        }
        plot_insula = {
            placement.id: (placement.parameters or {}).get("insula_id")
            for placement in shrunk.placements if placement.role == "plot"
        }
        centres = {
            item["id"]: (
                (item["rect"][0] + item["rect"][2]) / 2,
                (item["rect"][1] + item["rect"][3]) / 2,
            )
            for item in shrunk.metadata["settlement"]["insulae"]
            if item["id"] in residential
        }
        for placement in shrunk.placements:
            plot_id = (placement.parameters or {}).get("plot_id")
            insula_id = plot_insula.get(plot_id)
            if insula_id not in residential or placement.role in _INFRA_ROLES:
                continue
            # The coverage check uses the union box, so the pieces have to move
            # inward as well as shrink. Shrinking each one in place leaves the box.
            cx, cz = centres[insula_id]
            x, y, z = placement.position
            width, height, depth = placement.dimensions
            placement.position = (cx + (x - cx) * 0.2, y, cz + (z - cz) * 0.2)
            placement.dimensions = (width * 0.2, height, depth * 0.2)
        self.assertEqual(_statuses(rebuild(shrunk))["settlement.insula_coverage"], "fail")


_INFRA_ROLES = {
    "street", "kerb", "stepping_stone", "square", "plot", "garden",
    "pavement", "podium", "fence", "enclosure",
}


def _assert_seed(test, product, variant, seed):
    result = _concept(product, variant, seed)
    test.assertEqual(result.quality.status, "pass", f"{variant} seed {seed}")
    status = _statuses(result)
    for code in (
        "settlement.overlap", "settlement.frontage", "settlement.street_network",
        "settlement.ground_coverage", "settlement.street_frontage", "settlement.insula_coverage",
    ):
        test.assertEqual(status[code], "pass", f"{variant} {code} seed {seed}")


class SettlementCountTests(unittest.TestCase):
    def test_defaults_meet_the_density_counts(self):
        for variant, minimum in (
            ("street-village", 10), ("green-village", 10), ("market-town", 14),
        ):
            width, depth = DEFAULT_SITES[variant]
            document = build_document(BuildRequest(
                product="english-village", variant=variant, width=width, depth=depth,
                detail="concept", seed=1729,
            ))
            houses = [
                plot for plot in document.metadata["settlement"]["plots"]
                if plot["kind"] == "house" and plot["product"] == "timber-house"
            ]
            self.assertGreaterEqual(len(houses), minimum, variant)
        for variant, residential_min, total_min in (
            ("forum-quarter", 3, 4),
            ("residential-quarter", 4, 4),
            ("colonia-grid", 8, 9),
        ):
            width, depth = DEFAULT_SITES[variant]
            document = build_document(BuildRequest(
                product="roman-town", variant=variant, width=width, depth=depth,
                detail="concept", seed=1729,
            ))
            insulae = document.metadata["settlement"]["insulae"]
            residential = [item for item in insulae if item["use"] == "residential"]
            self.assertGreaterEqual(len(insulae), total_min, variant)
            self.assertGreaterEqual(len(residential), residential_min, variant)
            for item in residential:
                self.assertGreaterEqual(item["buildings"], 3, item["id"])
            if variant != "residential-quarter":
                self.assertEqual(sum(item["use"] == "forum" for item in insulae), 1)


class EnglishSeedTests(unittest.TestCase):
    def test_street_village_seeds(self):
        for seed in SEEDS:
            _assert_seed(self, "english-village", "street-village", seed)

    def test_green_village_seeds(self):
        for seed in SEEDS:
            _assert_seed(self, "english-village", "green-village", seed)

    def test_market_town_seeds(self):
        for seed in SEEDS:
            _assert_seed(self, "english-village", "market-town", seed)


class RomanSeedTests(unittest.TestCase):
    def test_forum_quarter_seeds(self):
        for seed in SEEDS:
            _assert_seed(self, "roman-town", "forum-quarter", seed)

    def test_residential_quarter_seeds(self):
        for seed in SEEDS:
            _assert_seed(self, "roman-town", "residential-quarter", seed)

    def test_colonia_grid_seeds(self):
        for seed in SEEDS:
            _assert_seed(self, "roman-town", "colonia-grid", seed)


class SettlementGeometryTests(unittest.TestCase):
    def test_quarter_turn_restores_a_timber_house_member_for_member(self):
        house = author_document(BuildRequest(
            product="timber-house", width=8, depth=8, height=7, levels=1, bays=2,
            variant="gable-front", detail="concept", seed=5,
            parameters={"finish": "chalk-slate", "porch": True, "chimney": True},
        ))
        self.assertTrue(any(placement.block_type == "MESH" for placement in house.placements))
        anchor = (12.5, 0.0, -4.25)
        for yaw in (90, 180, 270):
            restored = unturn_members(turn_members(house.placements, yaw, anchor), yaw, anchor)
            self.assertEqual(len(restored), len(house.placements))
            for original, back in zip(house.placements, restored):
                self.assertEqual(original.id, back.id)
                self.assertEqual(original.role, back.role)
                self.assertEqual(original.substance, back.substance)
                self.assertEqual(original.block_type, back.block_type)
                self.assertTrue(_close(original.position, back.position), original.id)
                self.assertTrue(_close(original.dimensions, back.dimensions), original.id)
                self.assertTrue(_close(original.parameters, back.parameters), original.id)

        wedge = Placement(
            id="wedge", kind="site", block_type="WEDGE",
            position=(1.0, 2.0, 3.0), dimensions=(4.0, 5.0, 6.0),
            material=(10, 20, 30, 255), role="marker", substance="limestone",
            parameters={"orientation": "+X", "axis": "x"},
        )
        turned = turn_members([wedge], 90, (0.0, 0.0, 0.0))[0]
        self.assertEqual(turned.parameters["orientation"], "+Z")
        self.assertEqual(turned.parameters["axis"], "z")
        self.assertTrue(_close(turned.dimensions, (6.0, 5.0, 4.0)))
        self.assertTrue(_close(turned.position, (-3.0, 2.0, 1.0)))
        back = unturn_members([turned], 90, (0.0, 0.0, 0.0))[0]
        self.assertTrue(_close(back.position, wedge.position))
        self.assertTrue(_close(back.dimensions, wedge.dimensions))
        self.assertEqual(back.parameters["orientation"], "+X")
        self.assertEqual(back.parameters["axis"], "x")

        placed = _concept("english-village", "market-town", 5)
        hall = next(placement for placement in placed.document.placements if placement.id == "plot-hall")
        self.assertEqual(hall.parameters["yaw"], 90)
        members = [
            placement for placement in placed.document.placements
            if (placement.parameters or {}).get("plot_id") == "plot-hall"
            and (placement.parameters or {}).get("source_id")
        ]
        restored = unturn_members(members, 90, hall.parameters["anchor"])
        authored = {
            placement.id: placement
            for placement in author_document(
                BuildRequest.from_dict(hall.parameters["building_request"])
            ).placements
        }
        self.assertEqual({item.parameters["source_id"] for item in restored}, set(authored))
        for member in restored:
            source = authored[member.parameters["source_id"]]
            self.assertEqual(member.role, source.role)
            self.assertEqual(member.substance, source.substance)
            self.assertEqual(member.block_type, source.block_type)
            self.assertTrue(_close(member.position, source.position), member.id)
            self.assertTrue(_close(member.dimensions, source.dimensions), member.id)
            if member.block_type == "MESH":
                self.assertTrue(_close(
                    member.parameters["vertices"], source.parameters["vertices"],
                ), member.id)
            if "axis" in (source.parameters or {}):
                self.assertEqual(member.parameters.get("axis"), source.parameters.get("axis"))

    def test_english_frontages_tofts_and_lanes_fall_in_range(self):
        for variant in ENGLISH_VARIANTS:
            result = _concept("english-village", variant, 11)
            self.assertEqual(result.quality.status, "pass", variant)
            houses = []
            for placement in result.document.placements:
                if placement.role != "plot":
                    continue
                kind = placement.parameters.get("kind_name")
                frontage = float(placement.parameters["frontage_m"])
                depth = float(placement.parameters["depth_m"])
                facing = placement.parameters["facing"]
                # House plots face the lane on their short side. The yard slab
                # spans that frontage; the croft behind is the toft depth.
                if kind == "house":
                    houses.append(placement)
                    self.assertGreaterEqual(frontage, 6.0, placement.id)
                    self.assertLessEqual(frontage, 12.0, placement.id)
                    self.assertGreaterEqual(depth, 30.0, placement.id)
                    self.assertLessEqual(depth, 60.0, placement.id)
                    grounds = [
                        item for item in result.document.placements
                        if item.role in {"plot", "garden"}
                        and (item.parameters or {}).get("plot_id") == placement.id
                    ]
                    xs, zs = [], []
                    for item in grounds:
                        x, _, z = item.position
                        w, _, d = item.dimensions
                        xs.extend((x - w / 2, x + w / 2))
                        zs.extend((z - d / 2, z + d / 2))
                    span_x, span_z = max(xs) - min(xs), max(zs) - min(zs)
                    if facing in ("south", "north"):
                        self.assertAlmostEqual(span_x, frontage, delta=0.05, msg=placement.id)
                        self.assertAlmostEqual(span_z, depth, delta=0.05, msg=placement.id)
                    else:
                        self.assertAlmostEqual(span_z, frontage, delta=0.05, msg=placement.id)
                        self.assertAlmostEqual(span_x, depth, delta=0.05, msg=placement.id)
            self.assertTrue(any(
                item.role == "fence" and item.substance == "oak"
                for item in result.document.placements
            ))
            self.assertTrue(any(
                item.role == "enclosure" and item.substance == "limestone"
                for item in result.document.placements
            ))
            self.assertGreaterEqual(len(houses), 2, variant)
            lanes = [placement for placement in result.document.placements if placement.role == "street"]
            self.assertTrue(lanes)
            for lane in lanes:
                narrow = min(lane.dimensions[0], lane.dimensions[2])
                self.assertGreaterEqual(narrow, 3.0)
                self.assertLessEqual(narrow, 5.0)
                self.assertEqual(lane.substance, "earth-fill")
            if variant == "market-town":
                square = next(item for item in result.document.placements if item.role == "square")
                self.assertEqual(square.substance, "limestone")
            else:
                square = next(item for item in result.document.placements if item.role == "square")
                self.assertEqual(square.substance, "earth-fill")

    def test_roman_streets_and_insulae_fall_in_range(self):
        for variant in ROMAN_VARIANTS:
            result = _concept("roman-town", variant, 11)
            self.assertEqual(result.quality.status, "pass", variant)
            for placement in result.document.placements:
                if placement.role == "street":
                    narrow = min(float(placement.dimensions[0]), float(placement.dimensions[2]))
                    street_class = placement.parameters["street_class"]
                    if street_class in {"cardo", "decumanus"}:
                        self.assertGreaterEqual(narrow, 6.0, placement.id)
                        self.assertLessEqual(narrow, 10.0, placement.id)
                        self.assertEqual(placement.substance, "limestone")
                    else:
                        self.assertGreaterEqual(narrow, 4.0, placement.id)
                        self.assertLessEqual(narrow, 6.0, placement.id)
                if placement.role != "plot":
                    continue
                insula = placement.parameters.get("insula_m")
                self.assertIsNotNone(insula, placement.id)
                self.assertGreaterEqual(insula, 35.0)
                self.assertLessEqual(insula, 75.0)
            residential = [
                item for item in result.document.metadata["settlement"]["insulae"]
                if item.get("use") == "residential"
            ]
            self.assertGreaterEqual(len(residential), 3, variant)
            for item in residential:
                self.assertGreaterEqual(item["buildings"], 3, item["id"])
            self.assertTrue(any(placement.role == "kerb" for placement in result.document.placements))
            self.assertTrue(any(placement.role == "pavement" for placement in result.document.placements))
            self.assertTrue(any(
                placement.role == "stepping_stone" for placement in result.document.placements
            ))

    def test_green_village_can_omit_the_manor(self):
        width, depth = DEFAULT_SITES["green-village"]
        result = build(BuildRequest(
            product="english-village", variant="green-village", width=width, depth=depth,
            detail="concept", seed=4, parameters={"manor": False},
        ))
        self.assertEqual(result.quality.status, "pass")
        products = {
            (placement.parameters or {}).get("building_product")
            for placement in result.document.placements
        }
        self.assertNotIn("norman-keep", products)
        self.assertTrue(any(placement.id == "plot-manor" for placement in result.document.placements))


class SettlementExportTests(unittest.TestCase):
    def test_default_settlements_export_glb_and_physics(self):
        for product in ("english-village", "roman-town"):
            result = build(BuildRequest(product=product, seed=1729))
            self.assertEqual(result.quality.status, "pass", product)
            self.assertEqual(result.document.request.detail, "standard")
            with tempfile.TemporaryDirectory() as temporary:
                paths = export_bundle(result, Path(temporary) / product, formats=("glb",))
                payload = _glb_json(paths["glb"])
                nodes = [node for node in payload["nodes"] if "mesh" in node]
                self.assertEqual(len(nodes), len(result.document.placements))
                extras = {node["name"]: node["extras"] for node in nodes}
                for placement in result.document.placements:
                    extra = extras[placement.id]
                    self.assertIn("plot_id", extra)
                    self.assertIn("building_product", extra)
                    self.assertEqual(extra.get("plot_id"), placement.parameters.get("plot_id"))
                    self.assertEqual(
                        extra.get("building_product"), placement.parameters.get("building_product"),
                    )
                loaded = __import__("trimesh").load(paths["glb"], force="scene", process=False)
                self.assertGreater(len(loaded.geometry), 0)
                physics = json.loads(paths["physics"].read_text(encoding="utf-8"))
                self.assertEqual(len(physics["members"]), len(result.physical.members))
                self.assertGreater(physics["total_mass_kg"], 0)
                self.assertTrue(np.isfinite(physics["total_mass_kg"]))
                by_id = {placement.id: placement for placement in result.document.placements}
                labelled = 0
                streets = 0
                for entry in physics["members"]:
                    placement = by_id[entry["id"]]
                    parameters = placement.parameters or {}
                    self.assertIn("plot_id", entry)
                    self.assertIn("building_product", entry)
                    self.assertEqual(entry["plot_id"], parameters.get("plot_id"))
                    self.assertEqual(entry["building_product"], parameters.get("building_product"))
                    if "street_id" in parameters:
                        self.assertEqual(entry.get("street_id"), parameters.get("street_id"))
                    if entry["plot_id"] and entry["building_product"]:
                        labelled += 1
                    if placement.role == "street" and entry.get("street_id"):
                        streets += 1
                self.assertGreater(labelled, 10, product)
                self.assertGreater(streets, 0, product)


def _glb_json(path):
    data = Path(path).read_bytes()
    length, = struct.unpack_from("<I", data, 12)
    return json.loads(data[20:20 + length])


if __name__ == "__main__":
    unittest.main()
