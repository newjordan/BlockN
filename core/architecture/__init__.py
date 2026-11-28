"""Public API for BlockN's semantic architecture pipeline."""

from .compiler import ArchitectureCompileError, BuildResult
from .io import (
    SUPPORTED_MESH_FORMATS,
    export_bundle,
    export_document,
    export_manifest,
    export_mesh_artifact,
    export_physics,
    export_quality,
)
from .schema import (
    SCHEMA_VERSION,
    ArchitectInfo,
    BuildDocument,
    BuildRequest,
    Placement,
)
from .historic import roll_request
from . import sonnet
from .service import (
    build,
    describe_product,
    list_architects,
    list_products,
    load_document,
    rebuild,
)


__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_MESH_FORMATS",
    "ArchitectInfo",
    "ArchitectureCompileError",
    "BuildDocument",
    "BuildRequest",
    "BuildResult",
    "Placement",
    "build",
    "rebuild",
    "roll_request",
    "sonnet",
    "list_architects",
    "load_document",
    "list_products",
    "describe_product",
    "export_document",
    "export_quality",
    "export_manifest",
    "export_mesh_artifact",
    "export_physics",
    "export_bundle",
]
