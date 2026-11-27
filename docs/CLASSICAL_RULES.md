# Classical architecture rules and concepts

BlockN separates **order** (column, capital and entablature proportions),
**typology** (the arrangement and purpose of spaces), and **detail** (geometry
resolution or composition density). A column order alone does not define a
building type. The catalog contains 30 products, including the legacy temple
and 12 additional classical forms.

## Scope and historical basis

The five-order vocabulary is Tuscan, Doric, Ionic, Corinthian and Composite.
The [Metropolitan Museum's architecture overview](https://www.metmuseum.org/essays/architecture-in-ancient-greece)
provides the Greek order context; the [Getty edition of Perrault's five orders](https://www.getty.edu/publications/resources/virtuallibrary/0892362332.pdf)
provides the wider five-order reference. These sources inform the vocabulary,
not a claim that BlockN's ratios reproduce any particular monument.

This release covers common Greco-Roman building concepts. It does **not** claim
to enumerate every historical regional tradition, later classical revival,
temple plan, or archaeological building. Gothic composition has a dedicated
[Cathedral Architect](CATHEDRAL_ARCHITECT.md). Byzantine, Egyptian, Persian,
Indian and East Asian traditions would require additional rule families.

The classical products now use sculpted solid members: fluted and tapered
Doric/Ionic shafts, shaped mouldings, real spiral volutes and foliate capitals.
The repaired rotunda and baths use hollow roof/vault shells; seating follows
radial sectors. These remain design interpretations, not archaeological
reconstructions or structural certificates. See [the craft audit](CRAFT_AUDIT.md).

## Order rules

`core/generators/classical.py::ORDER_SPECS` is the common proportion source for
the temple and new colonnaded forms. Doric/Ionic profiles and ratios were revised in version 3.0.0; legacy
Random seed selection is preserved. Random selection in new colonnaded forms
uses all five orders and a local seeded RNG.

| Order | Composition rule |
|---|---|
| Tuscan | Sturdy smooth shaft, simple base/capital, plain frieze in the temple. |
| Doric | Shorter, broader shaft; echinus and abacus; temple triglyph rhythm. |
| Ionic | 24 flutes, torus/scotia base, spiral volutes and a thin abacus. |
| Corinthian | Slender shaft, taller capital and two tiers of attached foliage blocks. |
| Composite | Tall foliated capital plus paired geometric volute accents. |

Ratios are authored design presets, not universal historical measurements.
New colonnades derive diameter from available column height and reject requests
whose capitals exceed bay clearance. Columns share one reusable geometry
builder with the legacy temple. Walls, piers, cellae and podiums are coursed
as human-scale ashlars in running bond rather than as monolithic slabs. The
classical architect version is now 3.1.0.

## Supported forms

| Product | Defining composition | Order selection |
|---|---|---|
| `classical-building` (`classical-temple`) | Established stepped temple and roof composer | All five |
| `classical-portico` | Two rows of supports, entablature and roof | All five |
| `classical-stoa` | Front colonnade, rear wall and long roof | All five |
| `classical-peristyle` | Four covered walks around an open court | All five |
| `classical-basilica` | Central raised nave, lower aisles, polygonal apse | All five |
| `classical-rotunda` | Circular colonnade, solid drum and dome | All five |
| `classical-forum` | Open colonnaded civic court and rear hall massing | All five |
| `classical-villa` | Open court with subdivided room wings | All five |
| `classical-triumphal-arch` | One or three arched openings and attic | Masonry; `auto` |
| `classical-aqueduct` | Aligned arcades and open channel | Masonry; `auto` |
| `classical-theatre` | Semicircular stepped seating and stage wall | Masonry; `auto` |
| `classical-amphitheatre` | Elliptical stepped seating around an arena | Masonry; `auto` |
| `classical-baths` | Three domed halls, side chambers, pool basin | Masonry; `auto` |

Masonry forms reject column-order variants instead of silently ignoring them,
and their placements carry no `order` parameter, matching the `null` order in
`metadata.classical_rules`. Theatres use block terraces, not curved wedge
seating. Forum, villa and baths are conceptual plans, without detailed internal
doors or service systems.

## Inputs and composition limits

New forms accept dimensions from 1 to 200, bays from 2 to 12, and concept,
standard or presentation detail. Descriptor defaults are the recommended
starting proportions. Dimensions are composition controls, not a guarantee
that ornaments and terrace blocks fit an exact external bounding box.

- `bays` sets colonnade/arcade rhythm or radial sector density. On the triumphal
  arch, 2 selects one opening and 3–12 select three. Baths retain three halls.
- Only aqueducts accept `levels` (1–3; default 2).
- Rotundas require width and depth to agree within 5%.
- Invalid aspect ratios and overlapping colonnade capital clearances are rejected.
- New forms author openings themselves and accept only `openings=auto`.
- Unknown free-form parameters are rejected. The legacy temple keeps its
  existing documented parameters and dimension semantics.
- Theatre detail selects 3/6/9 terrace tiers. Other forms currently retain the
  same geometry across detail levels; detail does not promise ornament variation.
- The order supplies a fixed stone/roof palette, as with the legacy temple;
  the general `style` field does not recolor these products.

## Roles and kinds

Every placement carries a fine `role` and a coarse `kind`. Roles name the
typology-specific member (`nave_roof`, `channel_wall`, `seating_tier`); kinds
come from a small shared vocabulary — `foundation`, `column`, `entablature`,
`roof`, `wall`, `pier`, `arch`, `slab`, `circulation`, `platform`, `seating` —
so `kind_counts` metrics and `required_kinds` invariants stay comparable across
the catalog. Every part of a column collapses to kind `column` regardless of
order. Descriptors publish both `required_roles` and `required_kinds`, and
documents declare both as invariants.

## Quality contract

Every form goes through the existing exact compiler and mesh audit. Intrinsic
classical checks additionally require the typology's defining roles and check
that every member's bounding envelope connects to the ground through other
members. These checks run even if document invariant metadata is removed.
The 0.0001 connectivity tolerance accommodates the legacy column layer gaps.
Members rest on what carries them rather than relying on reaching a neighbour
across the plan, so connectivity holds across each form's accepted aspect range.

Envelope connectivity is conservative. An arch's empty center is inside its
bounding box, so passing this test does not prove mesh contact, load capacity,
accessibility, human circulation, interpenetration freedom or printability.
Each member must still be finite, watertight and positively wound. Exported
assemblies are collections of solid members, not Boolean-unioned solids.

Regression tests cover all offered order/detail combinations, five distinct
temple geometries, deterministic requests, document round trips, open courts,
aqueduct tier counts, rejected inputs, missing components, detached roofs,
role/kind separation, masonry order absence and randomized control envelopes.

## Build and inspect

```bash
python3 -m blockn make classical-portico --variant corinthian --bundle builds/portico
python3 -m blockn make classical-basilica --variant composite --bundle builds/basilica
python3 -m blockn make classical-aqueduct --bays 6 --levels 3 --bundle builds/aqueduct
python3 -m blockn make classical-amphitheatre --detail presentation --bundle builds/arena
python3 -m blockn inspect builds/portico.architecture.json
```

The desktop workbench's form selector exposes all 13 classical products.
Selecting a new form loads its default dimensions and reveals bays/detail
controls; aqueducts also enable tiers. Temple-only tuning is hidden for new
forms, and masonry forms hide the order selector. Randomize redraws dimensions,
bays and detail from the selected typology's own buildable envelope — validated
against every order it offers — while leaving the chosen form and order alone.
After a build the panel reports the document's real order, typology, bays,
detail and quality tally. New desktop forms use the same quality-gated semantic
service as the CLI. The temple also uses the semantic pipeline and its quality checks in the desktop.

## Adding another tradition or form

Add a real composition builder, discoverable descriptor, declared input limits,
intrinsic required roles, and representative tests. Use an independent order
or tradition profile when its grammar differs. Do not advertise an unimplemented
style by attaching a new name to an existing mesh. Keep historical vocabulary,
authored geometric rules, and verified quality claims distinct.

The [Cathedral Architect](CATHEDRAL_ARCHITECT.md) extends the catalog with Gothic
plan and elevation rules. It has its own styles; classical column orders do not
apply to this product.
