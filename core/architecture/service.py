"""Public orchestration service for semantic BlockN architecture builds."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from . import builtin, classical_forms, cathedral, historic, marble_run, persian, settlement, timberwright, victorian
from .classical import PRODUCT as CLASSICAL_PRODUCT, build_classical_document
from .compiler import BuildResult, compile_document
from .schema import BuildDocument, BuildRequest, canonical_product_name


def _request_from(value: BuildRequest | Mapping[str, Any]) -> BuildRequest:
    if isinstance(value, BuildRequest):
        return value
    if isinstance(value, Mapping):
        return BuildRequest.from_dict(value)
    raise TypeError("request must be a BuildRequest or mapping")


def _document_from(value: BuildDocument | Mapping[str, Any]) -> BuildDocument:
    if isinstance(value, BuildDocument):
        return value
    if isinstance(value, Mapping):
        return BuildDocument.from_dict(value)
    raise TypeError("document must be a BuildDocument or mapping")


def author_document(request: BuildRequest | Mapping[str, Any]) -> BuildDocument:
    """Resolve a request through its architect without compiling a scene.

    Settlements call this for each plot, so a town is authored once and compiled
    once. Routing stays here; architects do not import the service.
    """
    # Detach from the caller because resolving aliases and architect routing is
    # part of normalization, not a mutation the caller should observe.
    normalized_request = BuildRequest.from_dict(_request_from(request).to_dict())
    product = canonical_product_name(normalized_request.product)
    if product == "classical-temple":
        product = "classical-building"
    normalized_request.product = product
    descriptor = describe_product(product)
    architect = descriptor.get("architect") or {}
    normalized_request.architect = str(
        architect.get("id") or descriptor.get("architect_id") or "primitive"
    )
    if product in settlement.PRODUCT_NAMES:
        document = settlement.build_document(normalized_request)
    elif product == CLASSICAL_PRODUCT["name"]:
        document = build_classical_document(normalized_request)
    elif product == cathedral.PRODUCT:
        normalized_request.style = "gothic"
        document = cathedral.build_document(normalized_request)
    elif product == marble_run.PRODUCT:
        document = marble_run.build_document(normalized_request)
    elif product == timberwright.PRODUCT:
        document = timberwright.build_document(normalized_request)
    elif product == persian.PRODUCT:
        normalized_request.style = "persian"
        document = persian.build_document(normalized_request)
    elif product == victorian.PRODUCT:
        normalized_request.style = "victorian"
        document = victorian.build_document(normalized_request)
    elif product in classical_forms.FORMS:
        document = classical_forms.build_document(normalized_request)
    elif historic.style_for(product):
        normalized_request.style = historic.style_for(product)[0].STYLE
        document = historic.build_document(normalized_request)
    else:
        document = builtin.build_document(normalized_request)
    return document


def build(request: BuildRequest | Mapping[str, Any]) -> BuildResult:
    """Resolve a request through a built-in architect and compile its document."""
    # compile_document owns scene construction, quality auditing, and the
    # stable generation_metadata record.  Replacing that record with
    # result.manifest() would make every subsequent manifest recursively embed
    # the previous one, so the compiler metadata remains authoritative here.
    return compile_document(author_document(request), use_collision=False)


def rebuild(document: BuildDocument | Mapping[str, Any]) -> BuildResult:
    """Compile and audit an existing semantic architecture document."""
    return compile_document(_document_from(document), use_collision=False)


def list_architects() -> List[Dict[str, Any]]:
    """Return the built-in product catalog grouped by architect.

    Architect groups and their products retain catalog order, which is stable
    in :func:`builtin.list_products`.  Each product is the full detached catalog
    record returned by the built-in registry.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for product in list_products():
        architect = product.get("architect") or {}
        architect_id = str(
            architect.get("id")
            or product.get("architect_id")
            or "unknown"
        )
        if architect_id not in groups:
            groups[architect_id] = {
                "id": architect_id,
                "title": str(
                    architect.get("title")
                    or product.get("architect_title")
                    or architect_id
                ),
                "version": str(
                    architect.get("version")
                    or product.get("architect_version")
                    or ""
                ),
                "products": [],
            }
        groups[architect_id]["products"].append(product)
    return list(groups.values())


def list_products() -> List[Dict[str, Any]]:
    """Return every built-in product, including the classical adapter."""

    return [*builtin.list_products(), deepcopy(CLASSICAL_PRODUCT), *classical_forms.list_products(),
            *cathedral.list_products(), *historic.list_products(), *marble_run.list_products(),
            *timberwright.list_products(), *persian.list_products(), *victorian.list_products(),
            *settlement.list_products()]


def describe_product(name: str) -> Dict[str, Any]:
    """Describe one product, accepting ``classical-temple`` as an alias.

    Unknown names raise against the whole merged catalog so the suggestion list
    names every product the service can actually route.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("product must be a non-empty string")
    normalized = canonical_product_name(name)
    if normalized == "classical-temple":
        normalized = "classical-building"
    catalog = list_products()
    for descriptor in catalog:
        if descriptor["name"] == normalized:
            return descriptor
    choices = ", ".join(str(descriptor["name"]) for descriptor in catalog)
    raise ValueError(f"unknown product '{name}'; expected one of: {choices}")


def load_document(path: str | os.PathLike[str]) -> BuildDocument:
    """Load a JSON architecture document from disk.

    Export-only fields are ignored after canonical fields are reconstructed;
    when a top-level fingerprint is present it must match the document.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Architecture document JSON must contain an object at the top level.")
    return BuildDocument.from_dict(payload)


__all__ = [
    "author_document",
    "build",
    "rebuild",
    "list_architects",
    "list_products",
    "describe_product",
    "load_document",
]
