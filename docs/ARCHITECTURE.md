# BlockN Architecture

This document describes BlockN's semantic architecture pipeline, its stable data contracts, and the supported ways to extend it.

## Design Goal

BlockN separates architectural intent from render geometry.

An architect decides that a build contains a foundation, structural bays, openings, stairs, roof members, or a classical entablature. It emits those decisions as semantic placements. The compiler then lowers each placement to the small render vocabulary understood by the existing scene and mesh system.

```text
request
  -> product resolution
  -> architect
  -> semantic document
  -> exact compiler
  -> scene
  -> quality audit
  -> document / quality / manifest / mesh exports
```

This division gives BlockN two useful properties:

1. Architecture is inspectable before and after mesh generation.
2. Mesh implementation details can evolve without discarding product-level meaning.

## Runtime Modules

| Module | Responsibility |
|---|---|
| `core/architecture/schema.py` | Canonical request, architect, placement, and document records; SHA-256 fingerprints. |
| `core/architecture/substances.py` | Registry of real building substances: density, strength, stiffness, friction, appearance, and a numeric weathering profile for each one. |
| `core/architecture/builtin.py` | Product catalog, palettes, deterministic built-in architects, and semantic assembly builders. |
| `core/architecture/classical.py` | Adapter from the five-order temple generator to semantic placements. |
| `core/architecture/masonry.py` | Human-scale ashlar coursing: running bond, opening voids, stacked drums. |
| `core/architecture/classical_forms.py` | Twelve classical typologies, shared column geometry, bounded composition rules. |
| `core/architecture/historic.py` | Shared composer and registry for the historic architects: coursed skins, arches, ring walls, batter, turned columns. Its `styles()` is the registry through which every historic product is listed, routed and audited. |
| `core/architecture/castle.py`, `egyptian.py`, `mesoamerican.py`, `pagoda.py` | Norman castle, Egyptian temple, Mesoamerican pyramid and timber pagoda architects with measured rule checks. |
| `core/architecture/chinese.py`, `dravidian.py` | Chinese timber halls on stone terraces and Dravidian tiered temple towers, on the historic composer with measured rule checks. |
| `core/architecture/marble_run.py` | Closed aqueduct/spline canal and a looping marble path. |
| `core/architecture/compiler.py` | Exact conversion from placements to `Scene` blocks and creation of `BuildResult`. |
| `core/architecture/physics.py` | Per-member mass, centroid, and bill of materials from compiled meshes; the reported vertical-load check. |
| `core/architecture/surfaces.py` | Deterministic stone tones, age and exposure weathering, and metre-scale box UVs. |
| `core/architecture/quality.py` | Ordered hard gates, invariant checks, and build metrics. |
| `core/architecture/io.py` | Canonical JSON, the physics manifest, and mesh exports for the common interchange formats. |
| `core/architecture/service.py` | Public orchestration API for catalog discovery, build, rebuild, and document loading. |
| `core/architecture/settlement.py` | Street and plot planner. A settlement is assembled from other architects' documents. See `docs/SETTLEMENTS.md`. |
| `core/primitives.py` | Authoritative registry for the internal render-level mesh vocabulary. |
| `cli.py` | `blockn` command parsing and request/output routing. |

## Pipeline Lifecycle

### 1. Request normalization

`BuildRequest` carries user intent. CLI product names are normalized to lowercase hyphenated names, defaults come from the product descriptor, and product-specific inputs can be supplied through `parameters`.

The product descriptor chooses the architect. The `architect` field in request JSON is useful provenance, but catalog resolution is authoritative for built-in products.

### 2. Architecture authoring

The selected builder produces a `BuildDocument` without mutating the final scene. Built-in builders use a call-local `random.Random(seed)` and deterministic iteration order. They assign stable primitive IDs and semantic roles as they construct the document.

The classical adapter is intentionally different internally: it runs the mature legacy generator in a collision-free temporary scene, suppresses its console output, and translates every generated block and legacy element tag into a deterministic `Placement`. The resulting document follows the same contract as native semantic products.

### 3. Canonical document and identity

`BuildDocument.to_dict()` is the canonical serialized representation. It contains the architect record, normalized request, ordered placements, metadata, units, and coordinate-system declaration.

`BuildDocument.fingerprint()` hashes a compact, key-sorted JSON representation with SHA-256. Volatile quality and timestamp fields are removed before hashing. The fingerprint therefore identifies semantic build content rather than the time at which it was audited or exported.

Readers reject unknown architecture schema versions and verify an exported top-level fingerprint before accepting the document. Non-finite values are rejected recursively during normalization.

### 4. Exact compilation

`compile_document(document, use_collision=False)` creates a fresh `Scene` and lowers placements in document order.

For each placement, the compiler injects these authoritative block parameters:

- `primitive_id`
- `kind`
- `architect`
- `architect_version`
- `component_role`
- `parent_id`

The compiler requires every placement to produce exactly one block and retain its semantic ID. Rejected geometry, unsupported block types, or a scene mutation that does not return an exact result raises `ArchitectureCompileError`; silent partial builds are not returned.

Legacy grid-support and overlap rejection are disabled during semantic compilation. Those systems cannot correctly represent every planned overhang, touching assembly, arch, or nested member. The semantic document is compiled exactly and then audited as a whole.

### 5. Quality audit

`audit_build(document, scene)` runs deterministic checks in a fixed order and returns a `QualityReport`. The report includes overall status, status counts, detailed observed/expected values, affected primitive IDs, and build metrics.

### 6. Exports

The exporter writes the semantic source, audit, combined manifest, and optional mesh formats. GLB and OBJ retain one named scene member per semantic primitive. PLY and STL flatten the geometry because they do not preserve scene graphs.

## Data Contracts

All schema types use only JSON-compatible values at their public serialization boundary.

### `ArchitectInfo`

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable machine identifier such as `envelope` or `classical`. |
| `version` | string | Architect implementation version included in provenance. |
| `title` | string | Human-readable catalog label. |

### `BuildRequest`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `architect` | string | `primitive` | Requested/catalog architect identifier. |
| `product` | string | `wall` | Product name. |
| `seed` | integer | `0` | Local deterministic seed. |
| `width` | number or null | `null` | Requested X extent; product default when null. |
| `height` | number or null | `null` | Requested Y extent; product default when null. |
| `depth` | number or null | `null` | Requested Z extent; product default when null. |
| `style` | string | `contemporary` | Coordinated material palette. |
| `detail` | string | `standard` | Composition detail level. |
| `variant` | string | `auto` | Product-specific form variant. |
| `bays` | integer or null | `null` | Product-specific repeated bay count. |
| `levels` | integer or null | `null` | Product-specific level count. |
| `openings` | string | `auto` | Product-specific opening strategy. |
| `parameters` | object | `{}` | Typed product-specific options. |
| `age` | number | `0.35` | Surface age from 0 (new) to 1 (centuries). Changes weathering only. |

`BuildRequest.normalized_dict()` lowercases and trims categorical fields, resolves numeric Python types, sorts mapping keys recursively, and returns the fingerprint-ready request record.

### `Placement`

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique deterministic primitive ID. |
| `kind` | string | Broad semantic class, such as `structure`, `opening`, `roof`, or `circulation`. |
| `block_type` | string | Internal render primitive: `CUBE`, `WEDGE`, `CYLINDER`, `ARCH`, or `DOME`. |
| `position` | three numbers | Geometric center `(x, y, z)`. |
| `dimensions` | three positive numbers | Full `(width, height, depth)` extents. |
| `material` | four integers | RGBA color. |
| `role` | string | Specific component role used for inspection and metrics. |
| `parameters` | object | Mesh and product-specific metadata. |
| `parent_id` | string or null | Optional reference to another placement in the same document. |
| `substance` | string or null | Substance registry id of the member's physical material. Serialized only when set, so pre-substance documents keep their exact fingerprints. |

Positions are center-based. X is width, Y is up/height, and Z is depth. Ground is `Y=0`. Geometry is authored at human scale in metres; the physical layer treats one block unit as one metre. Every architect assigns a registered substance to every member it emits; see [MATERIALS.md](MATERIALS.md).

### `BuildDocument`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | string | `blockn.architecture/v1` | Serialized contract version. |
| `units` | string | `block` | Logical unit declaration. |
| `coordinate_system` | object | X/Y/Z, right-handed, precision 6 | Coordinate convention and canonical precision. |
| `architect` | `ArchitectInfo` | required | Author and implementation provenance. |
| `request` | `BuildRequest` | required | Normalized source intent. |
| `placements` | array | `[]` | Ordered semantic primitive records. |
| `metadata` | object | `{}` | Resolved parameters, palette, bounds, invariants, and design metrics. |

Canonical document JSON has this shape:

```json
{
  "schema_version": "blockn.architecture/v1",
  "units": "block",
  "coordinate_system": {
    "axes": {"x": "width", "y": "height", "z": "depth"},
    "handedness": "right",
    "precision": 6
  },
  "architect": {
    "id": "primitive",
    "version": "1.0.0",
    "title": "Primitive Architect"
  },
  "request": {},
  "placements": [],
  "metadata": {}
}
```

The abbreviated empty objects above are illustrative; normal exports contain the complete normalized request and placement records.

### `QualityCheck` and `QualityReport`

A check contains:

- `code`: stable check/invariant identifier.
- `status`: `pass`, `warning`, or `fail`.
- `severity`: `info`, `warning`, or `error`.
- `message`: human-readable result.
- `primitive_ids`: sorted IDs involved in a failure.
- `observed`: measured value.
- `expected`: required value.

A report contains overall `status`, per-status `counts`, ordered `checks`, and `metrics` such as placement count, block count, mesh health, bounds, ground height, parent validity, semantic-role counts, substance counts, and the physical aggregates described in [MATERIALS.md](MATERIALS.md) (total mass, centre of mass, bill of materials, footprint, ground bearing pressure, maximum member utilisation).

### `BuildResult` and manifest

`BuildResult` holds:

- `document`: canonical semantic source.
- `scene`: compiled legacy-compatible `Scene`.
- `quality`: deterministic audit report.
- `physical`: per-member physical records (`members`, in document order) plus document aggregates. Stored on the result, never on the placements, so a build's fingerprint is unaffected by physical measurement.

`BuildResult.manifest()` starts with the canonical document fields and adds:

- `fingerprint`
- `generation_metadata`
- `quality`

This makes the manifest self-contained while keeping `.architecture.json` and `.quality.json` useful as focused artifacts.

## Quality Contract

Core hard gates run in this order:

1. `document.unique_ids`
2. `document.transforms`
3. `document.materials`
4. `document.substances`
5. `document.unique_geometry`
6. `compile.parity`
7. `compile.semantic_tags`
8. `mesh.nonempty`
9. `mesh.finite`
10. `mesh.watertight`
11. `mesh.winding`
12. `mesh.positive_volume`
13. `structure.vertical_load` (reported: a member over its factored bearing strength warns without failing)
14. `build.ground_contact`
15. `document.parent_links`
16. `document.parent_acyclic`
17. Domain checks such as `domain.spire_bridge_connectivity` and `domain.spire_ground_access`
18. Product-declared invariants, ordered by invariant name

Product metadata may declare these invariant forms:

- `min_placements` / `max_placements`
- exact `placement_count`, `compiled_block_count`, or `mesh_count`
- `ground_contact`
- `parent_links_valid`
- `required_roles`
- `required_kinds`
- `required_block_types`
- a metric comparison descriptor with `metric`, `operator`, and `expected`/`value`

Example:

```json
{
  "invariants": {
    "min_placements": 1,
    "ground_contact": true,
    "parent_links_valid": true,
    "required_roles": ["structure.column", "structure.beam"],
    "at_least_four_blocks": {
      "metric": "compiled_block_count",
      "operator": ">=",
      "expected": 4
    }
  }
}
```

Quality checks validate the document/mesh contract; they are not a substitute for structural engineering, code compliance, accessibility review, or fabrication analysis. The physical layer ([MATERIALS.md](MATERIALS.md)) adds measured mass and a reported vertical-bearing check, which is a plausibility diagnostic rather than a structural analysis.

## Deterministic Contract

Within a fixed BlockN and architect version, the following are deterministic for the same normalized request:

- product defaults and parameter resolution;
- placement order, IDs, transforms, roles, materials, and substances;
- architecture document JSON content;
- document SHA-256 fingerprint;
- compiler semantic tags and block parity;
- quality-check order and metrics.

Architect implementations must:

1. Use only call-local seeded randomness.
2. Avoid unordered iteration when it affects placements or IDs.
3. Round authored coordinates consistently before creating placements.
4. Generate IDs from stable semantic structure, not object identity or wall-clock time.
5. Put volatile timestamps and audit data outside geometry-driving metadata.

The fingerprint recursively excludes keys named `age`, `quality`, `quality_report`, `timestamp`, `timestamps`, `created_at`, `updated_at`, and `generated_at`. Do not place architectural decisions under those names. `age` is excluded because it changes the weathered surface, not the building.

The semantic fingerprint does not promise byte-identical GLB/OBJ output across different versions of Python, Trimesh, or exporter backends. It identifies the architecture document that those exporters consumed.

## Artifact Contract

For bundle stem `builds/example`, the default export is:

```text
builds/example.glb
builds/example.architecture.json
builds/example.quality.json
builds/example.construction.json
builds/example.manifest.json
builds/example.physics.json
```

`STEM.physics.json` is one rigid-body record per member: mass, centroid, substance, friction, and bounds. Mass and centroid are copied from the physical report. Additional requested formats use the same stem (`--format glb,obj,off`, or repeated flags). Always available: `glb`, `gltf`, `obj`, `ply`, `stl`, `off`. Collada, 3MF, USD, and FBX are written when their library or Blender is installed, and refused with a direct message otherwise. See [MATERIALS.md](MATERIALS.md) for the weathering and material payload. JSON is pretty-printed, key-sorted, UTF-8, newline-terminated, and written through an atomic replace. Architecture exports include the top-level fingerprint.

Bundle export preflights the document, quality report, manifest, and nonempty scene before writing its first artifact, preventing malformed JSON from leaving a partial mesh-only bundle.

Mesh node/geometry names come from `parameters['primitive_id']`. Duplicate names receive deterministic occurrence suffixes during export, although duplicate document IDs should already fail quality validation.

## Extension Points

### Add a semantic product

For a built-in product:

1. Add a serializable product descriptor to `_PRODUCTS` in `core/architecture/builtin.py`.
2. Implement a builder that receives `_BuildContext` and emits semantic placements through `context.add(...)` or the existing assembly helpers.
3. Register the builder in `_BUILDERS` using the same product name.
4. Resolve abstract parent groups before returning the document.
5. Declare deterministic metadata, design metrics, and invariants.
6. Add catalog, determinism, compile-parity, and quality tests.

Product descriptors define the catalog-visible name, category, architect, description, defaults, variants, and opening modes. The catalog is the source used by CLI discovery and request defaulting.

For a standalone architect, construct `ArchitectInfo`, `BuildRequest`, `Placement`, and `BuildDocument` directly, then call `compile_document`. The compiler and exporters do not require the document to originate in the built-in registry.

### Add an architect adapter

Use `core/architecture/classical.py` as the adapter pattern when a mature generator already emits scene blocks:

1. Run the generator in an isolated, collision-free scene.
2. Convert every output block to one placement without dropping geometry.
3. Map legacy tags to stable semantic kinds and roles.
4. Preserve source parameters and design metrics.
5. Assign deterministic IDs and document the architect version.
6. Compile the new document through the standard compiler and require quality parity.

### Add a render primitive

Semantic products should reuse the existing render vocabulary whenever possible. When a genuinely new mesh form is required:

1. Implement a mesh factory with center-position and full-extent conventions.
2. Add one `PrimitiveSpec` to `PRIMITIVE_REGISTRY` in `core/primitives.py`.
3. Define parameter defaults and discovery tags.
4. Ensure scene creation and loading resolve through the registry.
5. Add tests for dimensions, bounds, finite vertices, watertightness, serialization, compilation, and export.

The registry validates three-component finite positions and positive finite dimensions before invoking a factory.

### Add a quality invariant

Prefer product-declared generic metric comparisons first. Add a new core check only when it applies to every architecture document or requires scene/mesh evidence not present in metrics.

Core checks must have stable codes, deterministic order, explicit observed/expected values, and sorted affected primitive IDs.

### Add a substance

New materials extend the registry rather than the architects:

1. Add one `Substance` record to `SUBSTANCES` in `core/architecture/substances.py` with honest typical values, a stated weathering profile, and a `basis` note naming the source family.
2. Reference the new id from an architect's substance mapping (data, never a scattered literal).
3. Extend `tests/test_substances.py` registry coverage if a new category appears.
4. Update `docs/MATERIALS.md`.

Substance ids are a stable vocabulary: architects, documents, and quality gates all resolve through the registry.

### Add an exporter

Exporters operate on the small public protocols:

- documents and reports provide `to_dict()`;
- results provide `manifest()`;
- scenes expose compiled blocks and meshes.

New exporters should preserve semantic primitive names where the target format allows, reject empty output, and use atomic file replacement.

## Public API Example

```python
from core.architecture.io import export_bundle
from core.architecture.schema import BuildRequest
from core.architecture.service import build

request = BuildRequest(
    architect="structure",
    product="structural-grid",
    seed=11,
    width=18,
    height=4,
    depth=12,
    style="industrial",
    detail="presentation",
    variant="slab-frame",
    bays=4,
    levels=3,
)

result = build(request)
if result.quality.status == "fail":
    raise RuntimeError(result.quality.to_dict())

paths = export_bundle(result, "builds/frame", formats=("glb", "obj"))
print(result.document.fingerprint())
print(paths)
```

## Compatibility Boundary

The existing `Scene`, mesh geometry, classical generator, JSON save format, optimizer, and desktop GUI remain supported. They are the compilation/render backend, not the semantic source of truth for new BlockN workflows.

Direct construction with an unknown render `block_type` is now rejected by the authoritative primitive registry. Legacy scene JSON remains load-compatible: a missing type loads as `CUBE`, the historical `SLOPE_X_POS` name migrates to `WEDGE`, and an unknown legacy type is warned about and loaded as `CUBE`.

Use the semantic pipeline for new automation because it provides explicit intent, deterministic identity, exact compile parity, and auditable artifacts. Use the legacy `generate` command or GUI when an existing workflow depends on the original classical scene format or interactive desktop controls.

## Classical composition rules

See [CLASSICAL_RULES.md](CLASSICAL_RULES.md) for the order/typology distinction,
supported forms, historical references, input limits and intrinsic quality gates.
