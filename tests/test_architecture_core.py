import unittest

import numpy as np

from core.architecture import (
    ArchitectureCompileError,
    BuildDocument,
    BuildRequest,
    build,
    rebuild,
    list_architects,
    list_products,
)
from core.primitives import create_primitive_mesh, primitive_catalog, primitive_names
from core.scene import Scene


class TestArchitecturePipeline(unittest.TestCase):
    def test_every_catalog_product_compiles_and_passes_quality(self):
        products = list_products()
        self.assertEqual(len(products), 67)
        self.assertGreaterEqual(len(list_architects()), 8)

        for descriptor in products:
            with self.subTest(product=descriptor["name"]):
                result = build(BuildRequest(product=descriptor["name"], seed=1729))
                self.assertEqual(result.quality.status, "pass")
                self.assertEqual(result.document.request.architect, result.document.architect.id)
                self.assertEqual(len(result.document.placements), len(result.scene.blocks))
                self.assertEqual(
                    len({placement.id for placement in result.document.placements}),
                    len(result.document.placements),
                )

    def test_document_roundtrip_and_fingerprint_are_stable(self):
        result = build(
            BuildRequest(
                product="facade",
                seed=41,
                style="classical",
                detail="presentation",
                bays=5,
                openings="mixed",
            )
        )
        fingerprint = result.document.fingerprint()
        restored = BuildDocument.from_dict(result.document.to_dict())
        self.assertEqual(restored.fingerprint(), fingerprint)

        restored.metadata["timestamp"] = "2099-01-01T00:00:00Z"
        restored.metadata["quality"] = {"status": "stale"}
        self.assertEqual(restored.fingerprint(), fingerprint)

    def test_wall_openings_are_semantic_splits_not_boolean_subtraction(self):
        result = build(BuildRequest(product="wall", openings="mixed", detail="presentation"))
        roles = result.quality.metrics["role_counts"]
        self.assertEqual(result.quality.status, "pass")
        self.assertGreaterEqual(roles.get("envelope.wall.ashlar", 0), 8)
        self.assertGreaterEqual(roles.get("envelope.wall.header", 0), 2)
        self.assertGreaterEqual(roles.get("envelope.wall.sill", 0), 1)
        self.assertTrue(all(placement.block_type == "CUBE" for placement in result.document.placements))

    def test_seeded_spire_architect_is_deterministic_but_varies(self):
        request = BuildRequest(product="spire-cluster", seed=12, detail="presentation")
        first = build(request)
        second = build(BuildRequest.from_dict(request.to_dict()))
        different = build(BuildRequest(product="spire-cluster", seed=13, detail="presentation"))

        self.assertEqual(first.document.fingerprint(), second.document.fingerprint())
        self.assertNotEqual(first.document.fingerprint(), different.document.fingerprint())
        self.assertIn("circulation.spire_bridge", first.quality.metrics["role_counts"])
        checks = {check.code: check.status for check in first.quality.checks}
        self.assertEqual(checks["domain.spire_bridge_connectivity"], "pass")
        self.assertEqual(checks["domain.spire_ground_access"], "pass")
        self.assertEqual(
            first.quality.metrics["spire_connected_platform_count"],
            first.quality.metrics["spire_platform_count"],
        )

        disconnected = BuildDocument.from_dict(first.document.to_dict())
        bridge = next(
            placement
            for placement in disconnected.placements
            if placement.role == "circulation.spire_bridge"
        )
        bridge.position = (bridge.position[0] + 100.0, bridge.position[1], bridge.position[2])
        broken_result = rebuild(disconnected)
        broken_checks = {check.code: check.status for check in broken_result.quality.checks}
        self.assertEqual(broken_checks["domain.spire_bridge_connectivity"], "fail")
        self.assertEqual(broken_result.quality.status, "fail")

    def test_classical_alias_uses_uniform_result_contract(self):
        result = build(
            BuildRequest(
                product="classical-temple",
                variant="doric",
                detail="concept",
                width=8,
                height=6,
                depth=6,
                seed=7,
                parameters={"arch_portal": True},
            )
        )
        self.assertEqual(result.document.architect.id, "classical")
        self.assertEqual(result.document.request.product, "classical-building")
        self.assertEqual(result.quality.status, "pass")
        self.assertIn("arch_portal", result.quality.metrics["role_counts"])
        from core.architecture.masonry import placement_is_oversized
        portals = [p for p in result.document.placements if p.role == "arch_portal"]
        self.assertGreater(len(portals), 1)
        self.assertFalse(any(p.block_type == "ARCH" for p in portals))
        self.assertEqual([p.id for p in portals if placement_is_oversized(p)], [])

    def test_invalid_requests_fail_before_artifact_generation(self):
        with self.assertRaisesRegex(ValueError, "width"):
            build(BuildRequest(product="wall", width=0))
        with self.assertRaisesRegex(ValueError, "variant"):
            build(BuildRequest(product="roof", variant="unsupported"))
        with self.assertRaisesRegex(ValueError, "unknown product"):
            build(BuildRequest(product="not-a-product"))
        with self.assertRaisesRegex(ValueError, "unsupported parameter"):
            build(BuildRequest(product="stair", parameters={"stepps": 3}))
        with self.assertRaisesRegex(ValueError, "does not support --bays"):
            build(BuildRequest(product="wall", bays=3))
        with self.assertRaisesRegex(ValueError, "does not support --openings"):
            build(BuildRequest(product="classical-building", openings="mixed"))
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            build(BuildRequest(product="wall", parameters={"unused": float("nan")}))

    def test_schema_identity_parent_cycles_and_materials_are_enforced(self):
        result = build(BuildRequest(product="wall"))
        payload = result.document.to_dict()
        payload["schema_version"] = "blockn.architecture/v999"
        with self.assertRaisesRegex(ValueError, "Unsupported architecture schema_version"):
            BuildDocument.from_dict(payload)

        payload = result.document.to_dict()
        payload["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            BuildDocument.from_dict(payload)

        cyclic = BuildDocument.from_dict(result.document.to_dict())
        cyclic.placements[0].parent_id = cyclic.placements[0].id
        cyclic_result = rebuild(cyclic)
        checks = {check.code: check.status for check in cyclic_result.quality.checks}
        self.assertEqual(checks["document.parent_acyclic"], "fail")

        invalid_material = BuildDocument.from_dict(result.document.to_dict())
        invalid_material.placements[0].material = (300, -1, 0, 255)
        with self.assertRaisesRegex(ArchitectureCompileError, "RGBA"):
            rebuild(invalid_material)

    def test_dome_products_are_outward_facing_positive_solids(self):
        roof = build(BuildRequest(product="roof", variant="dome"))
        classical = build(
            BuildRequest(product="classical-building", parameters={"dome_cap": True})
        )
        with self.subTest(product="roof"):
            self.assertEqual(roof.quality.status, "pass")
            shells = [
                block.mesh for block in roof.scene.blocks
                if block.parameters.get("component_role") == "roof.dome"
            ]
            self.assertGreater(len(shells), 4)
            self.assertFalse(any(block.block_type == "DOME" for block in roof.scene.blocks))
            self.assertTrue(all(mesh.is_winding_consistent for mesh in shells))
            self.assertTrue(all(mesh.volume > 0.0 for mesh in shells))
        with self.subTest(product="classical-building"):
            self.assertEqual(classical.quality.status, "pass")
            dome_meshes = [
                block.mesh for block in classical.scene.blocks
                if block.parameters.get("component_role") == "dome_cap"
            ]
            self.assertTrue(dome_meshes)
            self.assertTrue(all(mesh.is_winding_consistent for mesh in dome_meshes))
            self.assertTrue(all(mesh.volume > 0.0 for mesh in dome_meshes))

    def test_scene_roundtrip_retains_semantics_and_generation_metadata(self):
        result = build(BuildRequest(product="column", variant="classical", seed=5))
        payload = result.scene.to_dict()
        restored = Scene(use_collision=False)
        restored.from_dict(payload)

        self.assertEqual(restored.generation_metadata, result.scene.generation_metadata)
        self.assertEqual(
            [block.parameters["primitive_id"] for block in restored.blocks],
            [block.parameters["primitive_id"] for block in result.scene.blocks],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported scene schema_version"):
            restored.from_dict({"schema_version": "blockn.scene/v999", "blocks": []})


class TestRenderPrimitiveRegistry(unittest.TestCase):
    def test_registry_is_authoritative_and_factories_are_valid(self):
        self.assertEqual(primitive_names(), ("CUBE", "WEDGE", "CYLINDER", "ARCH", "DOME", "MESH"))
        self.assertEqual([item["name"] for item in primitive_catalog()], list(primitive_names()))
        parameters = {
            "MESH": {"vertices": [[-.5,-.5,-.5],[.5,-.5,-.5],[0,.5,-.5],[0,0,.5]], "faces": [[0,2,1],[0,1,3],[1,2,3],[2,0,3]]},
            "WEDGE": {"orientation": "+Z"},
            "CYLINDER": {"radius": 0.5, "height": 1.0, "sections": 12},
            "ARCH": {"segments": 8},
            "DOME": {"layers": 6, "sections": 12},
        }
        for name in primitive_names():
            with self.subTest(name=name):
                mesh = create_primitive_mesh(
                    name,
                    np.array([0.0, 0.5, 0.0]),
                    np.array([1.0, 1.0, 1.0]),
                    parameters.get(name),
                )
                self.assertFalse(mesh.is_empty)
                self.assertTrue(np.isfinite(mesh.vertices).all())

        with self.assertRaisesRegex(ValueError, "Unknown block type"):
            create_primitive_mesh(
                "UNKNOWN",
                np.zeros(3),
                np.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
