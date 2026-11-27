# Materials: substances, physical properties, and the load check

A placement used to carry only an RGBA colour. It now also carries a
**substance**: a real building material with a density, strengths, a stiffness,
a friction coefficient, an appearance, and a stated way of weathering. With
substances assigned, every compiled build can be weighed member by member,
totalled into a bill of materials, and checked against gravity.

The registry lives in `core/architecture/substances.py`. The measurement and
the vertical-load check live in `core/architecture/physics.py`.

## The substance registry

Every substance is a frozen `Substance` record:

| Field | Unit | Meaning |
|---|---|---|
| `id` | – | Stable identifier used by architects and documents. |
| `name`, `category` | – | Display name and family (`stone`, `earth`, `soil`, `binder`, `timber`, `roofing`, `metal`, `glass`, `textile`). |
| `density`, `density_range` | kg/m³ | A typical value plus a low/high range. Mass uses the typical value. |
| `compressive_strength` | MPa | Typical strength; the value the load check compares against. |
| `tensile_strength` | MPa | Typical tensile/flexural strength, recorded for later use. |
| `elastic_modulus` | GPa | Typical Young's modulus, recorded for later use. |
| `friction` | – | Static friction coefficient, recorded for later use. |
| `color_low`, `color_high` | sRGB 0..255 | Base appearance range, weathered end to fresh end. |
| `roughness`, `metallic` | 0..1 | PBR surface parameters. |
| `weathering` | – | Data statements: how the surface ages (limestone gains biofilm, copper turns verdigris, lead whitens, oak silvers). |
| `basis` | – | Where the values come from, e.g. "typical engineering-handbook range for sedimentary limestone". |

The numbers are honest typical values from standard engineering sources, not
invented precision. Treat strengths and moduli as order-of-magnitude figures:
real stone varies as much as the density ranges shown. The pipeline authors
human-scale geometry in metres, so one block unit is treated as one metre and
densities are kg/m³.

The registry (29 substances):

| id | category | density (kg/m³) | compressive (MPa) | tensile (MPa) | E (GPa) |
|---|---|---|---|---|---|
| limestone | stone | 2200 | 55 | 3.5 | 30 |
| sandstone | stone | 2200 | 60 | 3.0 | 20 |
| granite | stone | 2700 | 160 | 10 | 60 |
| marble | stone | 2700 | 100 | 7 | 55 |
| basalt | stone | 2900 | 150 | 10 | 70 |
| tuff | stone | 1500 | 10 | 0.8 | 8 |
| flint | stone | 2600 | 200 | 15 | 70 |
| fired-clay-brick | earth | 1800 | 30 | 2 | 15 |
| engineering-brick | earth | 2000 | 80 | 4 | 25 |
| mud-brick | earth | 1550 | 3 | 0.3 | 1.5 |
| earth-fill | soil | 1800 | 1.0 | 0.02 | 0.05 |
| terracotta | earth | 1900 | 30 | 3 | 15 |
| glazed-tile | earth | 2000 | 40 | 4 | 25 |
| lime-mortar | binder | 1650 | 2 | 0.2 | 2 |
| lime-plaster | binder | 1600 | 2.5 | 0.5 | 4 |
| stucco | binder | 1850 | 5 | 1 | 8 |
| roman-concrete | binder | 1700 | 8 | 0.7 | 10 |
| oak | timber | 750 | 50 | 90 | 12 |
| softwood | timber | 500 | 35 | 70 | 11 |
| cedar | timber | 420 | 30 | 60 | 8.5 |
| painted-timber | timber | 520 | 35 | 70 | 11 |
| slate | roofing | 2700 | 120 | 15 | 20 |
| clay-roof-tile | roofing | 1900 | 25 | 3 | 15 |
| thatch | roofing | 150 | 0.8 | 2 | 0.15 |
| lead-sheet | metal | 11340 | 12 | 15 | 16 |
| copper-sheet | metal | 8960 | 70 | 220 | 120 |
| wrought-iron | metal | 7800 | 200 | 350 | 190 |
| window-glass | glass | 2500 | 800 | 45 | 70 |
| painted-linen | textile | 420 | 1 | 40 | 1.5 |

`painted-linen` exists because two traditions genuinely emit cloth members —
castle banners and temple pennants on their cedar masts. It is recorded as
non-structural.

`painted-timber` exists for the members a visitor sees painted or lacquered:
the vermilion columns, doors and rafters and the blue-green architraves and
bracket sets of a Chinese hall. Its strength is the fir beneath the coating.

Timber strengths are parallel to grain, which is how the load check uses them.

## How architects assign substances

Every architect assigns a substance to **every member it emits**. The
assignment is data — a module-level mapping of member role (or palette key) to
substance id — never a scattered literal:

| Module | Mapping | Reading of the tradition |
|---|---|---|
| `builtin.py` | `STYLE_SUBSTANCES` (style → palette role → substance) | The four material palettes: contemporary limestone/sandstone on slate, classical marble on terracotta tile, industrial brick and iron under lead, timber oak under clay tile. |
| `classical.py` | `CLASSICAL_SUBSTANCES` (role → substance) | Five-order temple: marble order and roof members on a limestone podium, stair, and cella. |
| `classical_forms.py` | `FORM_SUBSTANCES` (role → substance) | Greek and Roman typologies: marble order members, limestone walls and piers, terracotta tiled roofs; Roman concrete domes, groin vaults, channels, and pools with brick bath walls in the masonry forms. |
| `cathedral.py` | `CATHEDRAL_SUBSTANCES` (role → substance) | High Gothic: limestone ashlar everywhere, slated nave, transept, aisle, and apse roofs (the geometry lays them as slate courses), lead hip and eave rolls, glass in the lights. |
| `castle.py` | `SUBSTANCES` (role → substance) | Norman: limestone ashlar facings over a **flint** rubble core, oak floors, joists, stairs and drawbridges, lead on the boarded keep roofs, slated tower cones, a shingled cedar turret cap, a motte of compacted **earth fill**. |
| `egyptian.py` | `SUBSTANCES` (role → substance) | New Kingdom: coursed sandstone with **granite** for the members that do the work — architraves, lintels, abaci, obelisks — limestone floors and ramps, cedar masts and naos. |
| `mesoamerican.py` | `SUBSTANCES` (role → substance) | Cut **limestone** facing and core; the plastered and painted surfaces (tablero panels, roof comb) are lime plaster; the temple slab over the corbelled vault is lime mortar. |
| `pagoda.py` | `SUBSTANCES` (role → substance) | **Cedar/hinoki** frame (posts, beams, brackets, doors), lime-plaster infill panels, clay tile eaves, granite step and foundation stone on a limestone podium, copper-alloy finial. |
| `chinese.py` | `SUBSTANCES` (role → substance) | Ming and Qing halls: **softwood** fir for the frame nobody sees (cross-beams, king posts, corner beams) and **painted timber** for what is seen (columns, doors, rafters, architraves, plates, bracket sets); a white **marble** terrace, stairs and balustrade over a rammed **earth** core; **fired brick** walls; **lead-glazed tile** on every roof and ridge. |
| `dravidian.py` | `SUBSTANCES` for the gopuram and `VIMANA` for the vimana (role → substance) | A gopuram is **granite** below (plinth, base, passage and its ceiling) and **fired brick** above (talas, pilasters, the frames of its niches, prastaras, haras), with **lime plaster** figures and kudu arches and **gilded copper** pots. A vimana is **granite** throughout, pot included. |
| `timberwright.py` | `MATERIAL_SUBSTANCES` + `ROOF_SUBSTANCES` (palette key → substance, finish-aware roofs) | Oak frame and joinery, lime-plaster infill and gables, brick and terracotta chimney work, lead flashings; the finish palette names the roof: slate or clay tile. |
| `persian.py` | `FINISH_MASONRY` (finish → masonry, stone) and `MATERIAL_SUBSTANCES` (palette key → substance) | Load-bearing walls, piers, arches, vaults and domes in **fired brick** laid in lime mortar, or **sandstone** for the sandstone-lapis finish; glazed tile for turquoise, lapis and azure work and the minaret caps; lime-plaster render; limestone paving and plinths. Nothing is timber. |
| `victorian.py` | `FINISH_BRICK` (finish → brick), `FINISH_DRESSING` (finish → dressings) and `MATERIAL_SUBSTANCES` (palette key → substance) | British domestic 1840–1900: bonded walls, arches, stacks, breasts and garden walls in **fired clay** brick or **engineering brick** by finish; dressings, sills, steps and copings in limestone or sandstone; **slate** roofs over **softwood** joists, rafters, plates, boards, sashes and bargeboards; **window glass** glazing; **wrought-iron** railings; **lime mortar** beds under plates and copings; **glazed-tile** paths; **terracotta** pots. |
| `marble_run.py` | `SUBSTANCES` (role → substance) | The aqueduct tradition: limestone piers, arcades and channel walls over a concrete specus, iron riding rail, and a marble sphere. |

Each architect's catalog descriptor publishes the substances it may assign
(`substances` key in every product description), so `blockn describe` shows a
product's physical vocabulary.

### The document contract

`Placement` gained an optional `substance` field. It is serialized **only when
set**, so documents authored before substances existed keep their exact
canonical JSON and fingerprints; loading and saving round-trips either way.
Two new quality checks enforce the contract:

- `document.substances` (hard gate): every placement carries a substance id
  that exists in the registry. An unknown id fails, and so does a document
  where some members have a substance and others do not. A document with no
  substances at all predates them. It still passes, so older and
  hand-authored documents keep working; every built-in architect assigns
  every member, and `tests/test_substances.py` holds them to it.
- `structure.vertical_load` (reported, see below): the vertical-load check.
  Members without a substance still count toward mass and footprint at zero
  density. They are left out of the load check and counted as
  `members_without_substance`.

Substance data itself never changes a fingerprint once assigned: mass, load,
and stress are measured after compilation and stored on the build result, not
in the placements.

## What the physical numbers mean

After compilation, `core/architecture/physics.py` measures every member:

- **Volume** from the member's real compiled mesh (for a `CUBE` this is exact
  from its dimensions; other primitives use their true geometry). Meshes whose
  volume cannot be read fall back to the placement's extent box.
- **Mass** = volume x the substance's typical density.
- **Centroid** from the mesh when it is watertight, otherwise the placement's
  center.
- **Supported load** = own mass plus the share of everything resting on the
  member (below).
- **Bearing stress** = supported load x gravity / contact area at the base.
- **Utilisation** = bearing stress x factor of safety / compressive strength.

Per document, the physical report aggregates:

- **Total mass** and **centre of mass** (mass-weighted centroid).
- **Bill of materials**: volume in m³, mass in tonnes, and member count per
  substance.
- **Footprint area**: the union of the ground-plane rectangles of every member
  touching Y=0, so overlapping plinths are not double-counted.
- **Average ground bearing pressure**: total weight / footprint area, in kPa.
- **Max member utilisation**, the member it occurs in, and any members over 1.0.

These appear under `quality.metrics["physical"]` (and therefore in the
`.quality.json` and manifest exports), as `metrics["substance_counts"]`, and in
`blockn make` output. Per-member records (`result.physical.members`) stay off
the placements so build identity is untouched. `blockn describe` lists the
substances a product's architect may assign.

Typical magnitudes: the default products weigh between 3.2 t (a single beam)
and 28,222 t (the 36 m pyramid); ground bearing runs 20–450 kPa; a 50 m
cathedral tower base reaches utilisation 0.77 under its own limestone.

## The structural check, v1

`structure.vertical_load` accumulates gravity down through the members:

1. A member **rests on** another when its base sits within 1 mm of the other's
   top and their ground-plane footprints overlap.
2. A member carries its own mass **plus the share of every member resting on
   it**, shared in proportion to the footprint overlap each support provides.
3. Bearing stress at a member's base is that load divided by its contact area
   (its own footprint when it beds on the ground).
4. Utilisation compares that stress with the substance's compressive strength
   through a factor of safety: **10 for masonry-like materials** (stone,
   earth, binders, mineral roofing, glass), **4 for timber and metal**, and
   **3 for soil** (a motte's earth fill), the usual geotechnical margin on an
   ultimate bearing capacity.

The check reports:

- each member's utilisation (per-member records on the build result),
- the maximum utilisation and the member it occurs in,
- any member above 1.0.

It runs on every product build and is **reported, not yet a hard gate**: a
member over 1.0 raises a warning — never a failure — with the offending ids in
the check's `primitive_ids`. As of this writing every default catalog product
builds with no member over 1.0; the highest is the cathedral's west tower at
0.77.

### Bearing resolution

A contact that covers less than a quarter of a member's own bed (or is thinner
than a 10 cm square) is not treated as a resolved bed. Such members keep their
mass, are counted as `members_without_resolved_bearing`, report an
as-if-bedded stress, and drop out of the utilisation ranking and of the
downward accumulation. This is a honesty guard, not a structural model: an
edge kiss at a running-bond joint, a spandrel wedge against a curved arch
ring, or a floor joist seated in a wall pocket has no meaningful axis-aligned
bearing, and pretending it does would report fiction (a 16 t funnel through a
50 mm sliver, for instance).

### Known limits

V1 is deliberately narrow. It does **not** evaluate:

- anything but vertical gravity: no wind, no lateral load, no arch thrust;
- bending, tension, shear, or buckling — a spanning beam is not checked
  (and members whose span the model cannot resolve are excluded above);
- connections, mortar joints, or reinforcement;
- foundations or soil: the ground bearing pressure is a report of total
  weight over footprint, not a soil capacity check;
- lateral stability, progressive collapse, or building-code compliance.

It is a plausibility check that keeps architects honest about mass and
material strength, at the scale of a game starter composition — and a base to
build a real solver on later.

## Weathering

`Substance.weathering` stays a short prose account of how the material ages.
Beside it, `WEATHERING` in `core/architecture/substances.py` holds one frozen
`WeatheringProfile` for every substance id. Each profile records:

| Field | Meaning |
|---|---|
| `biofilm`, `soot`, `patina` | A `TintDrive`: target sRGB colour, plus how strongly exposure and age push toward it. |
| `patina_on` | `open` for sun and rain (silvered timber, whitened lead), or `shelter` (verdigris on copper). |
| `erosion_color`, `erosion_strength` | The tone rain cuts toward, strongest on sandstone. |
| `roughness_age` | Added to the substance roughness at age 1. |
| `grime_strength`, `grime_height` | A dark band at the base. Height is metres above the ground. |

Shared targets are biofilm green `(78, 112, 68)`, soot black `(36, 34, 32)`,
verdigris `(72, 150, 122)`, silver-grey `(174, 172, 164)`, and lead white
`(228, 226, 216)`. A drive's exposure and age terms sum to at most 1, so a
fully exposed, fully aged face reaches the tint and stops.

`BuildRequest.age` runs from 0 (newly worked) to 1 (centuries). The default is
`0.35`, a maintained building. It is serialized with the request, accepted by
`blockn make --age`, and shown on the workbench as an Age spin box. The
fingerprint ignores `age`: the same building at two ages is the same building.

`core/architecture/surfaces.py` paints after the quality audit. It does not
move geometry or change a placement's substance. For each member:

1. A tone is chosen from the authored finish, seeded by
   `sha256(f"{seed}:{member id}")`, and quantised to at most 8 variants.
   Pigmented and glazed substances (`glazed-tile`, `lime-plaster`, `stucco`,
   `painted-linen`, `window-glass`) keep the authored hue. The eight tones
   only scale luminance, from −6% to +6% (`TONE_LUMINANCE_SPAN`). Natural
   substances project the authored RGB onto the segment from `color_low` to
   `color_high`:

       t = clamp(dot(authored − low, high − low) / |high − low|², 0, 1)

   A window of width 0.5 (`NATURAL_TONE_WINDOW`) is centred on `t` and clamped
   inside 0..1, and the eight tones are spread evenly across that window.
   Hue the segment does not express is kept while the authored colour is
   within `FINISH_HUE_REACH` (160 RGB levels) of its projection, and dropped
   beyond it. Buff and rose brick both land past the pale end of fired clay;
   the reach keeps those two finishes apart. A stylised colour such as the
   cathedral's `(0, 255, 65)` is beyond the reach, so it sits on the segment
   inside the real limestone range. Age 0 reproduces that tone exactly.
2. Exposure is measured per face, not per stacked course. Upward faces
   (`normal Y ≥ 0.5`) take biofilm and erosion from sky visibility: the
   fraction of a 3 by 3 sample of the member's top with no other member
   anywhere above it. Side faces are rain-washed unless they lie in a rain
   shadow. A point at height `y` is in shadow when some member above projects
   past the face plane by `p ≥ 0.05 m`, covers the point along the face, and
   has its base no more than `RAIN_SHADOW_RATIO * p` above `y`.
   `RAIN_SHADOW_RATIO` is 2.0: one metre of projection shades two metres of
   wall, which is wind-driven rain at one unit out for two units down. A flush
   course has `p = 0` and casts no shadow, so a stone in a plain wall stays
   clean apart from a small age drift. Cornices, string courses, eaves,
   porticoes and iwan hoods do cast a shadow, and those faces take soot and
   gypsum crust. Downward faces are soffits and are always sheltered. The
   base band still climbs `grime_height` metres. Biofilm and soot age drives
   are at most 0.08, so a face with no exposure term stays near its tone at
   age 1. Lead whitening and copper verdigris keep a stronger age drive.
3. The profile mixes those channels by age. Open tops green, rain-washed
   walls stay close to the finish, shadowed mouldings darken, open timber
   silvers, sheltered copper goes to verdigris, and lead whitens.
4. The same seed and age always produce the same colours. The compiled mesh
   stays welded, so the solid remains watertight. Exports split vertices where
   face normals differ by more than about 40°, which stops a top face's green
   bleeding down a wall and puts a UV seam on that edge. Curved members weld
   back together and keep smooth shading.

In memory, and in OBJ, PLY and OFF, colours are sRGB. glTF multiplies
`baseColorFactor` by `COLOR_0` in linear space, so the writer converts the
factor with the sRGB transfer and stores the vertex multipliers as float
`COLOR_0` (component type 5126). `absolute_vertex_colors` decodes that pair
back to sRGB. USD `displayColor` and the preview-surface `diffuseColor` are
linear as well. The Qt viewport keeps `block.color` in sRGB. Roughness on the
material is the substance roughness plus `roughness_age * age`. One material
name is shared by every member of the same substance and tone; a different
authored finish is a different factor. The placement's alpha is kept: a pane
authored below 255 stays translucent, the vertex colour carries that alpha,
and the material uses `alphaMode` BLEND. Opaque members of the same tone keep
alpha 255.

## Exports

Compiled meshes carry box-projected UVs at one UV unit per metre
(`uv_scale_m` is 1). A texture applied later in Blender, Unreal, or Godot
tiles at real-world scale. The scale is stored on the scene extras, on each
member, and in the physics manifest.

`blockn make … --format glb,obj,off` (repeatable or comma-separated) adds mesh
files to the bundle. The default is still a single GLB. Every bundle also
writes `STEM.physics.json`: units, age, `uv_scale_m`, total mass, and one
record per member with `id`, `substance`, `mass_kg`, `centroid`, `friction`,
and `bounds_min` / `bounds_max`. Mass and centroid are the figures already
measured by `physical_report`. They are not measured again.

| Format | What it carries |
|---|---|
| `.glb`, `.gltf` | One PBR material per substance and tone (`baseColorFactor`, `roughnessFactor`, `metallicFactor`) and float vertex colours for weathering. Both the factor and `COLOR_0` are linear, as the glTF spec requires. Each node's `extras` carry `substance`, `mass_kg`, `density`, `role`, `kind`, and `centroid`. The scene `extras` carry the bill of materials, total mass, age, and `uv_scale_m`. |
| `.obj` | Named members, metre-scale UVs, and an MTL sidecar. `Kd` stays sRGB. |
| `.ply` | One mesh with per-vertex weathered sRGB colour. |
| `.stl`, `.off` | Geometry. These formats have no material slot. OFF vertex colours are not written. |
| `.dae` | Collada materials and UVs when `pycollada` is installed. Diffuse is the sRGB factor. |
| `.3mf` | Geometry plus a per-vertex sRGB colour group when `lxml` is installed. The library writer emits triangles only; the colour group is added after it. |
| `.usda`, `.usdc` | UsdPreviewSurface and linear `displayColor`, with metre-scale UVs, when `usd-core` is installed. |
| `.fbx` | Through `blender -b --python` when `blender` is on `PATH`. |

Asking for a format whose library is missing raises a direct error and writes
nothing: Collada needs `pycollada`, 3MF needs `lxml`, USD needs `usd-core`,
and FBX needs Blender. The other formats do not. An unknown suffix is still
rejected as an unsupported mesh format.
