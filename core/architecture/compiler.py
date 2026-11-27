"""Compiler from semantic architecture documents to the existing BlockN scene."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Optional

import numpy as np

from core.scene import Scene

from .physics import PhysicalReport, physical_report
from .quality import QualityReport, audit_build
from .schema import BuildDocument
from .surfaces import apply_surfaces


class ArchitectureCompileError(RuntimeError):
    """Raised when a semantic placement cannot be compiled exactly once."""

    def __init__(self, primitive_id: str, message: str):
        self.primitive_id = str(primitive_id)
        super().__init__(f"Failed to compile primitive '{self.primitive_id}': {message}")


@dataclass
class BuildResult:
    document: BuildDocument
    scene: Scene
    quality: QualityReport
    physical: Optional[PhysicalReport] = None

    def manifest(self) -> Dict[str, Any]:
        """Return the complete, JSON-compatible build and audit manifest."""
        manifest = self.document.to_dict()
        manifest["fingerprint"] = self.document.fingerprint()
        manifest["generation_metadata"] = dict(
            getattr(self.scene, "generation_metadata", {}) or {}
        )
        manifest["quality"] = self.quality.to_dict()
        return manifest


def _generation_metadata(document: BuildDocument, scene: Scene) -> Dict[str, Any]:
    kind_counts = Counter(str(placement.kind) for placement in document.placements)
    role_counts = Counter(str(placement.role) for placement in document.placements)
    block_type_counts = Counter(str(placement.block_type) for placement in document.placements)
    metadata = {
        "schema_version": document.schema_version,
        "document_fingerprint": document.fingerprint(),
        "architect": document.architect.id,
        "architect_version": document.architect.version,
        "architect_title": document.architect.title,
        "product": document.request.product,
        "seed": int(document.request.seed),
        "placement_count": len(document.placements),
        "compiled_block_count": len(scene.blocks),
        "kind_counts": dict(sorted(kind_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "block_type_counts": dict(sorted(block_type_counts.items())),
    }
    marble = (document.metadata or {}).get("marble_run")
    if marble:
        metadata["marble_run"] = marble
    return metadata


def compile_document(
    document: BuildDocument,
    use_collision: bool = False,
) -> BuildResult:
    """Compile a semantic build document into a scene and audit the result.

    Compilation is intentionally exact: every placement must add one scene
    block.  The legacy scene can otherwise return ``None`` for unsupported or
    rejected geometry, which would silently corrupt a semantic build.
    """
    if not isinstance(document, BuildDocument):
        raise TypeError("document must be a BuildDocument")

    scene = Scene(use_collision=bool(use_collision))
    # Architecture documents are already planned compositions.  Legacy grid
    # support and collision rejection are unsuitable for overhangs, arches,
    # nested assemblies, and touching semantic members; quality audits them
    # after exact compilation instead.
    scene.support_check_mode = "OFF"
    scene.enable_strict_overlap_prevention = False

    for placement in document.placements:
        primitive_id = str(placement.id)
        try:
            material = tuple(placement.material)
            valid_material = (
                len(material) == 4
                and all(
                    not isinstance(channel, bool) and isinstance(channel, Integral)
                    for channel in material
                )
                and all(0 <= int(channel) <= 255 for channel in material)
            )
        except TypeError:
            valid_material = False
            material = ()
        if not valid_material:
            raise ArchitectureCompileError(
                primitive_id,
                "material must be four integer RGBA channels in the range 0..255",
            )
        parameters = dict(placement.parameters or {})
        parameters.update(
            {
                "primitive_id": primitive_id,
                "kind": str(placement.kind),
                "architect": str(document.architect.id),
                "architect_version": str(document.architect.version),
                "component_role": str(placement.role),
                "parent_id": None if placement.parent_id is None else str(placement.parent_id),
            }
        )

        before_count = len(scene.blocks)
        try:
            block = scene.add_block(
                position=np.asarray(placement.position, dtype=float),
                block_type=str(placement.block_type),
                color=tuple(int(channel) for channel in material),
                dimensions=np.asarray(placement.dimensions, dtype=float),
                parameters=parameters,
            )
        except Exception as exc:
            raise ArchitectureCompileError(primitive_id, str(exc)) from exc

        added_count = len(scene.blocks) - before_count
        if block is None or added_count != 1:
            if added_count:
                # This branch guards against a future Scene implementation that
                # mutates and then reports rejection.  Do not return a silently
                # divergent result.
                reason = f"scene mutated by {added_count} block(s) but returned no exact result"
            else:
                reason = "scene rejected the placement"
            raise ArchitectureCompileError(primitive_id, reason)

        compiled_id = (getattr(block, "parameters", {}) or {}).get("primitive_id")
        if compiled_id != primitive_id:
            raise ArchitectureCompileError(
                primitive_id,
                f"semantic id was not retained (observed {compiled_id!r})",
            )

    if len(scene.blocks) != len(document.placements):
        raise ArchitectureCompileError(
            "<document>",
            f"compiled {len(scene.blocks)} blocks for {len(document.placements)} placements",
        )

    scene.generation_metadata = _generation_metadata(document, scene)
    physical = physical_report(document, scene)
    quality = audit_build(document, scene, physical=physical)
    # Paint after the audit so quality still sees the compiled solids.
    # Placements, substance ids, and the physical report are not rewritten.
    apply_surfaces(document, scene, physical)
    return BuildResult(document=document, scene=scene, quality=quality, physical=physical)


__all__ = [
    "ArchitectureCompileError",
    "BuildResult",
    "compile_document",
]
