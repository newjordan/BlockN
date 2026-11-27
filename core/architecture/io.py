"""Artifact exporters for BlockN architecture builds.

The functions in this module intentionally operate on the small public
protocol shared by the architecture pipeline: documents and quality reports
provide ``to_dict()``, build results provide ``manifest()``, and scenes expose
their generated blocks.  Keeping the exporters protocol-based avoids coupling
artifact output to a particular architect implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from trimesh.visual.color import srgb_to_linear

from .surfaces import UV_SCALE_M, _split_hard_edges, absolute_vertex_colors


# Always-on writers, then formats that need an optional library or Blender.
SUPPORTED_MESH_FORMATS: Tuple[str, ...] = (
    "glb",
    "gltf",
    "obj",
    "ply",
    "stl",
    "off",
    "dae",
    "3mf",
    "usda",
    "usdc",
    "fbx",
)
_ALWAYS_MESH_FORMATS = {"glb", "gltf", "obj", "ply", "stl", "off"}
_NODE_EXTRA_KEYS = (
    "primitive_id",
    "substance",
    "mass_kg",
    "density",
    "role",
    "kind",
    "centroid",
    "roughness",
    "color",
    "tone",
    "friction",
    "uv_scale_m",
    "plot_id",
    "building_product",
    "street_id",
)


def _mapping_from(value: Any, *, label: str) -> Dict[str, Any]:
    """Return a detached mapping from a protocol object or mapping value."""
    if isinstance(value, Mapping):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"{label} must provide a to_dict() method or be a mapping.")

    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label}.to_dict() must return a mapping.")
    return dict(payload)


def _fingerprint_for(document: Any, payload: Mapping[str, Any]) -> str:
    """Resolve the document fingerprint, with a deterministic fallback."""
    fingerprint = getattr(document, "fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()

    if fingerprint is None:
        fingerprint = payload.get("fingerprint")

    if fingerprint is None:
        # Hash a compact canonical representation.  Exclude any absent/empty
        # fingerprint field so the digest describes the document itself.
        fingerprint_payload = dict(payload)
        fingerprint_payload.pop("fingerprint", None)
        canonical = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()

    if isinstance(fingerprint, bytes):
        return fingerprint.hex()
    return str(fingerprint)


def _atomic_write(path: Path, data: bytes) -> Path:
    """Write bytes beside the destination and atomically replace when possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _write_json(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write canonical, human-readable JSON with a terminating newline."""
    target = Path(path)
    serialized = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return _atomic_write(target, f"{serialized}\n".encode("utf-8"))


def _validate_json_payload(payload: Mapping[str, Any], label: str) -> None:
    try:
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid finite JSON: {exc}") from exc


def export_document(document: Any, path: str | os.PathLike[str]) -> Path:
    """Export an architecture document as canonical pretty JSON.

    The top-level ``fingerprint`` is always present.  A document's own
    ``fingerprint()`` value is authoritative; a SHA-256 digest of its canonical
    dictionary is used only for protocol-compatible documents without one.
    """
    payload = _mapping_from(document, label="Architecture document")
    payload["fingerprint"] = _fingerprint_for(document, payload)
    return _write_json(payload, path)


def export_quality(report: Any, path: str | os.PathLike[str]) -> Path:
    """Export a quality report as canonical pretty JSON."""
    return _write_json(_mapping_from(report, label="Quality report"), path)


def export_manifest(result: Any, path: str | os.PathLike[str]) -> Path:
    """Export the build result's manifest as canonical pretty JSON."""
    manifest = getattr(result, "manifest", None)
    if not callable(manifest):
        raise TypeError("Build result must provide a manifest() method.")
    payload = manifest()
    if not isinstance(payload, Mapping):
        raise TypeError("Build result manifest() must return a mapping.")
    return _write_json(dict(payload), path)


def _unique_semantic_name(raw_name: Any, index: int, used: Dict[str, int]) -> str:
    base = str(raw_name).strip() if raw_name is not None else ""
    if not base:
        base = f"primitive_{index:04d}"

    occurrence = used.get(base, 0) + 1
    used[base] = occurrence
    return base if occurrence == 1 else f"{base}__{occurrence}"


def _plain(value: Any) -> Any:
    """JSON-safe copy. glTF drops a whole extras object if one value is not."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _node_extras(mesh: trimesh.Trimesh, name: str) -> Dict[str, Any]:
    raw = dict(getattr(mesh, "metadata", {}) or {})
    extras: Dict[str, Any] = {}
    for key in _NODE_EXTRA_KEYS:
        if key in raw:
            extras[key] = _plain(raw[key])
    extras["primitive_id"] = name
    extras.setdefault("uv_scale_m", UV_SCALE_M)
    return extras


def _mesh_export_scene(scene: Any) -> tuple[trimesh.Scene, list[trimesh.Trimesh]]:
    """Create a named trimesh scene and return its non-empty mesh members."""
    if isinstance(scene, trimesh.Scene):
        meshes = [
            geometry
            for geometry in scene.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and not geometry.is_empty
        ]
        if not meshes:
            raise ValueError("Cannot export mesh artifact: scene contains no mesh geometry.")
        copied = scene.copy()
        export_meta = getattr(scene, "metadata", None)
        if isinstance(export_meta, Mapping) and export_meta:
            copied.metadata = _plain(dict(export_meta))
        return copied, meshes

    blocks = getattr(scene, "blocks", None)
    named_scene = trimesh.Scene()
    export_meta = getattr(scene, "export_metadata", None)
    if isinstance(export_meta, Mapping) and export_meta:
        named_scene.metadata = _plain(dict(export_meta))
    meshes: list[trimesh.Trimesh] = []
    used_names: Dict[str, int] = {}

    if blocks is not None:
        for index, block in enumerate(blocks):
            mesh = getattr(block, "mesh", None)
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                continue

            parameters = getattr(block, "parameters", None)
            primitive_id = parameters.get("primitive_id") if isinstance(parameters, Mapping) else None
            name = _unique_semantic_name(primitive_id, index, used_names)

            exported_mesh = _split_hard_edges(mesh)
            extras = _node_extras(exported_mesh, name)
            exported_mesh.metadata = extras
            named_scene.add_geometry(
                exported_mesh,
                node_name=name,
                geom_name=name,
                metadata=extras,
            )
            meshes.append(exported_mesh)
    else:
        get_all_meshes = getattr(scene, "get_all_meshes", None)
        if callable(get_all_meshes):
            for index, mesh in enumerate(get_all_meshes()):
                if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                    continue
                name = _unique_semantic_name(None, index, used_names)
                exported_mesh = _split_hard_edges(mesh)
                extras = _node_extras(exported_mesh, name)
                exported_mesh.metadata = extras
                named_scene.add_geometry(
                    exported_mesh,
                    node_name=name,
                    geom_name=name,
                    metadata=extras,
                )
                meshes.append(exported_mesh)

    if not meshes:
        raise ValueError("Cannot export mesh artifact: scene contains no mesh geometry.")
    return named_scene, meshes


def format_requirement(mesh_format: str) -> Optional[str]:
    """Return the install message for a format whose library is missing.

    ``None`` means the writer can run. Unknown names are not reported here.
    """
    if mesh_format in _ALWAYS_MESH_FORMATS:
        return None
    if mesh_format == "dae":
        try:
            import collada  # noqa: F401
        except ImportError:
            return (
                "Collada export needs pycollada. Install pycollada and retry; "
                "other formats do not need it."
            )
        return None
    if mesh_format == "3mf":
        try:
            import lxml  # noqa: F401
            import networkx  # noqa: F401
        except ImportError:
            return (
                "3MF export needs lxml. Install lxml and retry; "
                "other formats do not need it."
            )
        return None
    if mesh_format in {"usda", "usdc"}:
        try:
            import pxr  # noqa: F401
        except ImportError:
            return (
                "USD export needs usd-core. Install usd-core and retry; "
                "other formats do not need it."
            )
        return None
    if mesh_format == "fbx":
        if shutil.which("blender") is None:
            return (
                "FBX export needs Blender (`blender` on PATH). "
                "Install Blender and retry; other formats do not need it."
            )
        return None
    return None


def _format_from_path(path: Path) -> str:
    mesh_format = path.suffix.lower().lstrip(".")
    if mesh_format not in SUPPORTED_MESH_FORMATS:
        supported = ", ".join(f".{item}" for item in SUPPORTED_MESH_FORMATS)
        suffix = path.suffix or "<none>"
        raise ValueError(
            f"Unsupported mesh artifact format '{suffix}'. Supported formats: {supported}."
        )
    missing = format_requirement(mesh_format)
    if missing:
        raise ValueError(missing)
    return mesh_format


def _as_bytes(exported: Any, mesh_format: str) -> bytes:
    if isinstance(exported, str):
        return exported.encode("utf-8")
    if isinstance(exported, (bytes, bytearray, memoryview)):
        return bytes(exported)
    raise TypeError(
        f"Trimesh returned unsupported {mesh_format.upper()} export payload "
        f"of type {type(exported).__name__}."
    )


def _colored_copy(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """PLY wants vertex colours. Texture visuals keep them only as an attribute."""
    colors = absolute_vertex_colors(mesh)
    copied = mesh.copy()
    if len(colors) != len(copied.vertices):
        colors = np.tile(np.array([[200, 200, 200, 255]], dtype=np.uint8), (len(copied.vertices), 1))
    copied.visual = trimesh.visual.ColorVisuals(mesh=copied, vertex_colors=colors)
    return copied


def _geometry_copy(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces).copy(),
        process=False,
    )


def _flatten(meshes: list[trimesh.Trimesh], *, colored: bool) -> trimesh.Trimesh:
    prepared = [_colored_copy(mesh) if colored else _geometry_copy(mesh) for mesh in meshes]
    if len(prepared) == 1:
        return prepared[0]
    if not colored:
        return trimesh.util.concatenate(prepared)
    # Concatenate drops per-vertex colour when the visuals disagree. Stack it
    # explicitly so a flattened PLY keeps every stone's weathering.
    vertices = np.vstack([np.asarray(mesh.vertices) for mesh in prepared])
    faces = []
    colors = []
    offset = 0
    for mesh in prepared:
        faces.append(np.asarray(mesh.faces) + offset)
        colors.append(np.asarray(mesh.visual.vertex_colors, dtype=np.uint8))
        offset += len(mesh.vertices)
    merged = trimesh.Trimesh(vertices=vertices, faces=np.vstack(faces), process=False)
    merged.visual.vertex_colors = np.vstack(colors)
    return merged


def _write_sidecars(target: Path, files: Mapping[str, Any], main_name: str) -> bytes:
    main: Optional[bytes] = None
    for name, blob in files.items():
        data = blob if isinstance(blob, (bytes, bytearray)) else str(blob).encode("utf-8")
        payload = bytes(data)
        if name == main_name:
            main = payload
            continue
        _atomic_write(target.parent / name, payload)
    if main is None:
        raise ValueError(f"{target.suffix} export did not include {main_name}.")
    return main


def _export_obj(named_scene: trimesh.Scene, target: Path) -> bytes:
    from trimesh.exchange.obj import export_obj

    text, textures = export_obj(
        named_scene,
        return_texture=True,
        mtl_name=f"{target.stem}.mtl",
        header=None,
    )
    for name, blob in textures.items():
        payload = blob if isinstance(blob, (bytes, bytearray)) else str(blob).encode("utf-8")
        _atomic_write(target.parent / name, bytes(payload))
    return text.encode("utf-8")


def _export_usd(named_scene: trimesh.Scene, meshes: list[trimesh.Trimesh], target: Path) -> bytes:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    with tempfile.TemporaryDirectory() as temporary:
        stage_path = Path(temporary) / f"stage{target.suffix.lower()}"
        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        metadata = dict(named_scene.metadata or {})
        metadata.setdefault("uv_scale_m", UV_SCALE_M)
        stage.GetRootLayer().customLayerData = _plain(metadata)

        for index, mesh in enumerate(meshes):
            token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(mesh.metadata.get("primitive_id", index)))
            if not token or token[0].isdigit():
                token = f"m_{token}"
            prim_path = f"/World/{token}_{index}"
            usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
            points = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in np.asarray(mesh.vertices)]
            usd_mesh.CreatePointsAttr(points)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            colors = absolute_vertex_colors(mesh)
            # displayColor is linear. The sRGB bytes are the weathered colour.
            linear = srgb_to_linear(colors[:, :3].astype(np.float64) / 255.0)
            display = [Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in linear]
            color_primvar = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex,
            )
            color_primvar.Set(display)
            visual = getattr(mesh, "visual", None)
            uv = getattr(visual, "uv", None)
            if uv is not None and len(uv) == len(mesh.vertices):
                st = [Gf.Vec2f(float(u), float(v)) for u, v in np.asarray(uv)]
                texcoord = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex,
                )
                texcoord.Set(st)
            material = getattr(visual, "material", None)
            factor = getattr(material, "baseColorFactor", None)
            if factor is not None:
                base = np.asarray(factor, dtype=float).reshape(-1)
                if float(base[:3].max()) > 1.0:
                    base = base / 255.0
                base_linear = srgb_to_linear(base[:3])
                shade = UsdShade.Material.Define(stage, f"{prim_path}/Material")
                shader = UsdShade.Shader.Define(stage, f"{prim_path}/Material/Preview")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(float(base_linear[0]), float(base_linear[1]), float(base_linear[2]))
                )
                roughness = getattr(material, "roughnessFactor", 0.85)
                metallic = getattr(material, "metallicFactor", 0.0)
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
                shade.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(usd_mesh).Bind(shade)
            usd_mesh.GetPrim().SetCustomData(_plain(dict(mesh.metadata or {})))
        stage.GetRootLayer().Save()
        return stage_path.read_bytes()


def _export_fbx(named_scene: trimesh.Scene, target: Path) -> bytes:
    blender = shutil.which("blender")
    if blender is None:
        requirement = format_requirement("fbx")
        raise ValueError(requirement or "FBX export needs Blender (`blender` on PATH).")
    script = Path(__file__).with_name("blender_fbx_script.py")
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.glb"
        output = Path(temporary) / "output.fbx"
        source.write_bytes(_linearize_glb(_as_bytes(named_scene.export(file_type="glb"), "glb")))
        completed = subprocess.run(
            [blender, "-b", "--python", str(script), "--", str(source), str(output)],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            tail = detail[-1] if detail else "Blender produced no FBX."
            raise ValueError(f"Blender failed to write FBX: {tail}")
        return output.read_bytes()


_GLB_MAGIC = 0x46546C67
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_GL_UNSIGNED_BYTE = 5121
_GL_FLOAT = 5126


def _parse_glb(data: bytes) -> tuple[dict, bytearray]:
    if len(data) < 20 or struct.unpack_from("<I", data, 0)[0] != _GLB_MAGIC:
        raise ValueError("glTF binary header is not a GLB.")
    offset = 12
    header = None
    blob = bytearray()
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == _JSON_CHUNK:
            header = json.loads(chunk.decode("utf-8").rstrip(" \x00"))
        elif chunk_type == _BIN_CHUNK:
            blob = bytearray(chunk)
    if header is None:
        raise ValueError("GLB is missing its JSON chunk.")
    return header, blob


def _pack_glb(header: dict, blob: bytes) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(encoded)) % 4
    encoded += b" " * pad
    binary = bytes(blob)
    binary += b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return b"".join((
        struct.pack("<III", _GLB_MAGIC, 2, total),
        struct.pack("<II", len(encoded), _JSON_CHUNK),
        encoded,
        struct.pack("<II", len(binary), _BIN_CHUNK),
        binary,
    ))


def _read_vec4(header: dict, buffers: List[bytearray], accessor_index: int) -> np.ndarray:
    accessor = header["accessors"][accessor_index]
    view = header["bufferViews"][accessor["bufferView"]]
    buffer = buffers[int(view.get("buffer", 0))]
    count = int(accessor["count"])
    component = int(accessor["componentType"])
    dtype = np.uint8 if component == _GL_UNSIGNED_BYTE else np.float32
    width = 4
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", width * np.dtype(dtype).itemsize))
    row = width * np.dtype(dtype).itemsize
    if stride == row:
        return np.frombuffer(buffer[start:start + count * row], dtype=dtype).reshape(count, width).copy()
    rows = []
    for index in range(count):
        row_start = start + index * stride
        rows.append(np.frombuffer(buffer[row_start:row_start + row], dtype=dtype).copy())
    return np.vstack(rows)


def _linear_multipliers(raw: np.ndarray, factor_srgb: Sequence[float]) -> np.ndarray:
    """Linear COLOR_0 that reconstructs ``factor_srgb * raw`` once both are linear."""
    factor = np.asarray(factor_srgb[:3], dtype=np.float64)
    if raw.dtype == np.uint8:
        ratio = raw[:, :3].astype(np.float64) / 255.0
        alpha = raw[:, 3].astype(np.float64) / 255.0 if raw.shape[1] >= 4 else np.ones(len(raw))
    else:
        ratio = raw[:, :3].astype(np.float64)
        alpha = raw[:, 3].astype(np.float64) if raw.shape[1] >= 4 else np.ones(len(raw))
    absolute = np.clip(ratio * factor, 0.0, 1.0)
    absolute_linear = srgb_to_linear(absolute)
    reference_linear = srgb_to_linear(np.clip(factor, 0.0, 1.0))
    scale = np.maximum(reference_linear, 1e-8)
    rgb = np.clip(absolute_linear / scale, 0.0, 1.0)
    rgb[:, factor <= 1e-8] = 0.0
    return np.column_stack((rgb, alpha)).astype(np.float32)


def _append_float_colors(header: dict, buffers: List[bytearray], colors: np.ndarray) -> int:
    payload = np.ascontiguousarray(colors, dtype=np.float32).tobytes()
    target = buffers[0]
    if len(target) % 4:
        target.extend(b"\x00" * ((-len(target)) % 4))
    offset = len(target)
    target.extend(payload)
    header.setdefault("bufferViews", []).append({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(payload),
    })
    header.setdefault("accessors", []).append({
        "bufferView": len(header["bufferViews"]) - 1,
        "byteOffset": 0,
        "componentType": _GL_FLOAT,
        "count": int(colors.shape[0]),
        "type": "VEC4",
        "min": [float(value) for value in colors.min(axis=0)],
        "max": [float(value) for value in colors.max(axis=0)],
    })
    return len(header["accessors"]) - 1


def _rewrite_linear_colors(header: dict, buffers: List[bytearray]) -> None:
    """Move baseColorFactor and COLOR_0 from sRGB into linear float colour."""
    if not buffers or "accessors" not in header:
        return
    materials = header.get("materials", [])
    stored_factors: List[List[float]] = []
    for material in materials:
        factor = (material.get("pbrMetallicRoughness") or {}).get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        stored_factors.append([float(channel) for channel in factor])
    for mesh in header.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes") or {}
            color_index = attributes.get("COLOR_0")
            if color_index is None:
                continue
            accessor = header["accessors"][color_index]
            if int(accessor.get("componentType", 0)) != _GL_UNSIGNED_BYTE:
                continue
            raw = _read_vec4(header, buffers, int(color_index))
            material_index = int(primitive.get("material", 0) or 0)
            factor = stored_factors[material_index] if material_index < len(stored_factors) else [1.0, 1.0, 1.0, 1.0]
            floats = _linear_multipliers(raw, factor)
            attributes["COLOR_0"] = _append_float_colors(header, buffers, floats)
    for material, factor in zip(materials, stored_factors):
        pbr = material.get("pbrMetallicRoughness")
        if not pbr or "baseColorFactor" not in pbr:
            continue
        linear = srgb_to_linear(np.clip(np.asarray(factor[:3], dtype=np.float64), 0.0, 1.0))
        alpha = float(factor[3]) if len(factor) > 3 else 1.0
        pbr["baseColorFactor"] = [float(channel) for channel in linear] + [alpha]
    if header.get("buffers"):
        header["buffers"][0]["byteLength"] = len(buffers[0])


def _linearize_glb(data: bytes) -> bytes:
    header, blob = _parse_glb(data)
    _rewrite_linear_colors(header, [blob])
    return _pack_glb(header, blob)


def _linearize_gltf_files(files: Dict[str, Any]) -> None:
    raw = files.get("model.gltf")
    if raw is None:
        return
    header = json.loads(raw if isinstance(raw, str) else bytes(raw).decode("utf-8"))
    buffers: List[bytearray] = []
    owned: List[Optional[str]] = []
    for entry in header.get("buffers", []):
        uri = str(entry.get("uri", ""))
        if uri.startswith("data:"):
            payload = uri.split(",", 1)[1]
            buffers.append(bytearray(base64.b64decode(payload)))
            owned.append(None)
        elif uri:
            stored = files.get(uri, b"")
            buffers.append(bytearray(stored if isinstance(stored, (bytes, bytearray)) else bytes(stored)))
            owned.append(uri)
        else:
            buffers.append(bytearray())
            owned.append(None)
    if not buffers:
        return
    _rewrite_linear_colors(header, buffers)
    for index, entry in enumerate(header.get("buffers", [])):
        data = bytes(buffers[index])
        entry["byteLength"] = len(data)
        if index < len(owned) and owned[index]:
            files[owned[index]] = data
        else:
            entry["uri"] = "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")
    files["model.gltf"] = json.dumps(header, separators=(",", ":")).encode("utf-8")


def _colorize_3mf(payload: bytes, meshes: List[trimesh.Trimesh]) -> bytes:
    """Add per-vertex sRGB colours. The trimesh 3MF writer emits geometry only."""
    import io
    import zipfile

    from lxml import etree

    core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    material = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
    by_name = {}
    for mesh in meshes:
        name = str((getattr(mesh, "metadata", {}) or {}).get("primitive_id") or "")
        if name:
            by_name[name] = mesh
    archive_in = zipfile.ZipFile(io.BytesIO(payload))
    blobs = {name: archive_in.read(name) for name in archive_in.namelist()}
    archive_in.close()
    model_name = next(name for name in blobs if name.lower().endswith("3d/3dmodel.model") or name.lower().endswith("3dmodel.model"))
    root = etree.fromstring(blobs[model_name])
    resources = root.find(f"{{{core}}}resources")
    if resources is None:
        return payload
    next_id = 1
    for element in root.iter(f"{{{core}}}object"):
        next_id = max(next_id, int(element.get("id", "0")) + 1)
    for obj in list(resources.findall(f"{{{core}}}object")):
        mesh_element = obj.find(f"{{{core}}}mesh")
        if mesh_element is None:
            continue
        source = by_name.get(obj.get("name") or "")
        if source is None:
            continue
        colors = absolute_vertex_colors(source)
        if len(colors) == 0:
            continue
        group_id = str(next_id)
        next_id += 1
        group = etree.Element(f"{{{material}}}colorgroup")
        group.set("id", group_id)
        for red, green, blue, alpha in np.asarray(colors, dtype=np.uint8):
            entry = etree.SubElement(group, f"{{{material}}}color")
            entry.set("color", f"#{int(red):02X}{int(green):02X}{int(blue):02X}{int(alpha):02X}")
        resources.insert(0, group)
        triangles = mesh_element.find(f"{{{core}}}triangles")
        if triangles is None:
            continue
        for triangle in triangles:
            triangle.set("pid", group_id)
            triangle.set("p1", triangle.get("v1", "0"))
            triangle.set("p2", triangle.get("v2", "0"))
            triangle.set("p3", triangle.get("v3", "0"))
    blobs[model_name] = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            archive.writestr(name, blob)
    return out.getvalue()


def _export_payload(
    named_scene: trimesh.Scene,
    meshes: list[trimesh.Trimesh],
    mesh_format: str,
    target: Path,
    *,
    glb_materials: bool,
) -> bytes:
    if mesh_format == "glb" and not glb_materials:
        # Only alter export copies. No material or COLOR_0 means the viewer
        # supplies its default surface; this is not a baked neon wireframe.
        for member in named_scene.geometry.values():
            if isinstance(member, trimesh.Trimesh):
                member.visual = trimesh.visual.ColorVisuals(mesh=member)

    if mesh_format == "glb":
        payload = _as_bytes(named_scene.export(file_type="glb"), mesh_format)
        if glb_materials:
            payload = _linearize_glb(payload)
        return payload
    if mesh_format == "gltf":
        from trimesh.exchange.gltf import export_gltf

        files = export_gltf(named_scene, embed_buffers=True)
        _linearize_gltf_files(files)
        return _write_sidecars(target, files, "model.gltf")
    if mesh_format == "obj":
        return _export_obj(named_scene, target)
    if mesh_format == "ply":
        return _as_bytes(_flatten(meshes, colored=True).export(file_type="ply"), mesh_format)
    if mesh_format == "stl":
        return _as_bytes(_flatten(meshes, colored=False).export(file_type="stl"), mesh_format)
    if mesh_format == "off":
        from trimesh.exchange.off import export_off

        return export_off(_flatten(meshes, colored=False)).encode("utf-8")
    if mesh_format == "dae":
        from trimesh.exchange.dae import export_collada

        return _as_bytes(export_collada(meshes), mesh_format)
    if mesh_format == "3mf":
        payload = _as_bytes(named_scene.export(file_type="3mf"), mesh_format)
        return _colorize_3mf(payload, meshes)
    if mesh_format in {"usda", "usdc"}:
        return _export_usd(named_scene, meshes, target)
    if mesh_format == "fbx":
        return _export_fbx(named_scene, target)
    supported = ", ".join(f".{item}" for item in SUPPORTED_MESH_FORMATS)
    raise ValueError(
        f"Unsupported mesh artifact format '.{mesh_format}'. Supported formats: {supported}."
    )


def export_mesh_artifact(scene: Any, path: str | os.PathLike[str], *, glb_materials: bool = True) -> Path:
    """Export a non-empty scene to a mesh file.

    GLB, glTF, OBJ, Collada, 3MF, USD and FBX keep separate members where the
    format allows. STL, PLY and OFF are one mesh. A block's
    ``parameters['primitive_id']`` supplies the semantic name when present.
    PLY carries per-vertex weathered colour. Materials are written for every
    format that has a material slot.
    """
    target = Path(path)
    mesh_format = _format_from_path(target)
    named_scene, meshes = _mesh_export_scene(scene)
    data = _export_payload(
        named_scene, meshes, mesh_format, target, glb_materials=glb_materials,
    )
    return _atomic_write(target, data)


def _normalize_formats(formats: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(formats, str):
        formats = (formats,)

    normalized = []
    for requested in formats:
        for part in str(requested).split(","):
            mesh_format = part.strip().lower().lstrip(".")
            if not mesh_format:
                continue
            if mesh_format not in SUPPORTED_MESH_FORMATS:
                supported = ", ".join(SUPPORTED_MESH_FORMATS)
                raise ValueError(
                    f"Unsupported mesh artifact format '{part.strip()}'. Supported formats: {supported}."
                )
            missing = format_requirement(mesh_format)
            if missing:
                raise ValueError(missing)
            if mesh_format not in normalized:
                normalized.append(mesh_format)
    return tuple(normalized)


def _round_list(values: Any, places: int) -> list[float]:
    return [round(float(value), places) for value in values]


def _member_bounds(mesh: Any, placement: Any) -> tuple[list[float], list[float]]:
    bounds = getattr(mesh, "bounds", None) if mesh is not None and not getattr(mesh, "is_empty", True) else None
    if bounds is not None:
        array = np.asarray(bounds, dtype=float)
        if array.shape == (2, 3) and np.isfinite(array).all():
            return _round_list(array[0], 6), _round_list(array[1], 6)
    center = np.asarray(placement.position, dtype=float)
    half = np.asarray(placement.dimensions, dtype=float) / 2.0
    return _round_list(center - half, 6), _round_list(center + half, 6)


def physics_manifest(result: Any) -> Dict[str, Any]:
    """One rigid-body record per member. Mass and centroid come from the physical report."""
    physical = getattr(result, "physical", None)
    if physical is None or getattr(physical, "members", None) is None:
        from .physics import physical_report

        physical = physical_report(result.document, result.scene)

    placements = {str(placement.id): placement for placement in result.document.placements}
    blocks: Dict[str, Any] = {}
    for block in getattr(result.scene, "blocks", ()) or ():
        parameters = getattr(block, "parameters", {}) or {}
        primitive_id = parameters.get("primitive_id")
        if primitive_id is not None:
            blocks[str(primitive_id)] = block

    from .substances import SUBSTANCES

    members = []
    for member in physical.members:
        placement = placements.get(str(member.primitive_id))
        block = blocks.get(str(member.primitive_id))
        mesh = getattr(block, "mesh", None) if block is not None else None
        if placement is None:
            bounds_min = [0.0, 0.0, 0.0]
            bounds_max = [0.0, 0.0, 0.0]
            substance_id = str(member.substance) if member.substance else None
        else:
            bounds_min, bounds_max = _member_bounds(mesh, placement)
            substance_id = str(placement.substance) if getattr(placement, "substance", None) else None
        substance = SUBSTANCES.get(substance_id) if substance_id else None
        friction = None if substance is None else round(float(substance.friction), 4)
        record = {
            "id": str(member.primitive_id),
            "substance": substance_id,
            "mass_kg": round(float(member.mass_kg), 3),
            "centroid": _round_list(member.centroid_m, 6),
            "friction": friction,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
        }
        parameters = {} if placement is None else (getattr(placement, "parameters", {}) or {})
        for key in ("plot_id", "building_product", "street_id"):
            if key in parameters:
                record[key] = parameters[key]
        members.append(record)
    age = getattr(getattr(result.document, "request", None), "age", 0.0)
    return {
        "units": "m",
        "uv_scale_m": UV_SCALE_M,
        "age": round(float(age), 4),
        "total_mass_kg": round(float(physical.total_mass_kg), 3),
        "members": members,
    }


def export_physics(result: Any, path: str | os.PathLike[str]) -> Path:
    """Export the physics manifest as canonical pretty JSON."""
    return _write_json(physics_manifest(result), path)


def export_bundle(
    result: Any,
    stem: str | os.PathLike[str],
    formats: Sequence[str] | str = ("glb",),
    *, glb_materials: bool = True,
) -> Dict[str, Path]:
    """Export all standard artifacts for a build result.

    Returned keys are ``architecture``, ``quality``, ``construction``,
    ``manifest``, ``physics``, plus one key for each requested mesh format.
    """
    normalized_formats = _normalize_formats(formats)
    stem_text = os.fspath(stem)

    # Validate every non-mesh payload, the scene, and optional writers before
    # creating the first file. A missing library must not leave a partial bundle.
    document_payload = _mapping_from(result.document, label="Architecture document")
    document_payload["fingerprint"] = _fingerprint_for(result.document, document_payload)
    quality_payload = _mapping_from(result.quality, label="Quality report")
    manifest_method = getattr(result, "manifest", None)
    if not callable(manifest_method):
        raise TypeError("Build result must provide a manifest() method.")
    manifest_payload = manifest_method()
    if not isinstance(manifest_payload, Mapping):
        raise TypeError("Build result manifest() must return a mapping.")
    _validate_json_payload(document_payload, "Architecture document")
    _validate_json_payload(quality_payload, "Quality report")
    _validate_json_payload(manifest_payload, "Build manifest")
    physics_payload = physics_manifest(result)
    from .construction import construction_report

    construction_payload = construction_report(result.document, result.quality)
    _validate_json_payload(physics_payload, "Physics manifest")
    _validate_json_payload(construction_payload, "Construction report")
    _mesh_export_scene(result.scene)

    paths: Dict[str, Path] = {}
    for mesh_format in normalized_formats:
        mesh_path = Path(f"{stem_text}.{mesh_format}")
        paths[mesh_format] = export_mesh_artifact(result.scene, mesh_path, glb_materials=glb_materials)

    # Reuse the exact snapshots already validated above. Sculpted assemblies
    # can contain hundreds of thousands of vertices; rebuilding and hashing
    # these same document/manifest payloads doubles a substantial export cost.
    paths["architecture"] = _write_json(
        document_payload,
        Path(f"{stem_text}.architecture.json"),
    )
    paths["quality"] = _write_json(
        quality_payload,
        Path(f"{stem_text}.quality.json"),
    )
    paths["construction"] = _write_json(
        construction_payload,
        Path(f"{stem_text}.construction.json"),
    )
    paths["manifest"] = _write_json(
        manifest_payload,
        Path(f"{stem_text}.manifest.json"),
    )
    paths["physics"] = _write_json(
        physics_payload,
        Path(f"{stem_text}.physics.json"),
    )
    return paths
