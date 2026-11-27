# Cathedral Architect

BlockN's Gothic architect composes a complete cruciform church from coordinated
plans and elevations. It produces real meshes and named members: clustered nave
piers, pointed arcades, side aisles, clerestory windows, transverse and diagonal
vault ribs, flying buttresses, a projecting transept with walled and lit arms,
a chevet (a pinnacled ambulatory below a clerestoried hemicycle whose half-cone
roof continues the nave roof), recessed west portals with blind tympana and
wimperg gables, towers, pinnacles, and rose tracery. Nave and choir share one
bay pitch. Roofs are laid as staggered slate courses, spires as continuous
octagonal tapers.

The desktop preserves the black-and-green phosphor wireframe. Select **Gothic
Cathedral**, choose a style, set proportions and bays, and generate. **Open roof /
inspect vaults** temporarily hides roof coverings and glazing. Export still
includes the complete building. Individual members remain selectable even when
large assemblies share viewport draw calls.

## Styles and design controls

| Variant | Composition |
| --- | --- |
| `high-gothic` | Paired tall west spires, a secondary crossing lantern, paired lights and a ten-petal rose |
| `early-english` | Capped west towers that clear the ridge, a dominant spire on a broad crossing tower, single lancets with upright trefoils and an eight-petal rose |
| `rayonnant` | Paired west spires, three-light clerestory/belfry windows and denser twelve-petal rose tracery |

These are original stylized compositions informed by Gothic architectural
principles, not replicas of named monuments. The pointed arch, ribbed vault,
and flying buttress organize the design rather than serve as applied ornaments.
See the [Met's Gothic architecture overview](https://www.metmuseum.org/essays/gothic-art)
and [English Heritage's account of medieval architecture](https://www.english-heritage.org.uk/learn/story-of-england/medieval/architecture/).

Width controls the whole composition including the transept. Height controls
the tallest spire, with a small finial allowance; depth sets the longitudinal
plan before the apse projection. The emitted bounds are authoritative for asset
placement. The standard request is width 32, height 50, depth 64, seven nave bays.

| Control | Accepted values |
| --- | --- |
| `variant` | `auto`, `random`, or one of the styles above |
| `detail` | `concept`, `standard`, `presentation` |
| `bays` | 4–10 nave bays |
| `width`, `height`, `depth` | 12–200; depth/width 1.6–2.6 and height/width 1.2–1.9 |
| `--param choir_bays=N` | 2–4 |
| `--param apse_sides=N` | 5, 7, or 9 |
| `--param spire_ratio=N` | 0.22–0.36, west-spire fraction in High Gothic/Rayonnant |

Unsupported controls are rejected. The architect authors its own vertical
levels and openings; omit `levels` and keep `openings=auto`. A fixed request and
architect version replay deterministically. `random` selects a style from the
local seed; agent series also vary overall scale and style.

## Practical exports

```bash
python3 -m blockn describe gothic-cathedral
python3 -m blockn make gothic-cathedral --variant high-gothic \
  --detail presentation --seed 73 --bundle builds/cathedral-73
python3 -m blockn make gothic-cathedral --variant early-english \
  --bays 8 --param choir_bays=3 --param apse_sides=7 \
  --bundle builds/english-73
python3 -m blockn agent series --products gothic-cathedral --kitbash \
  --count 3 --seed 73 --output builds/cathedral-kit-73
python3 -m blockn agent build examples/cathedral-kitbash.json \
  --output builds/cathedral-studies
```

Bundles include GLB geometry, canonical architecture JSON, a manifest, and the
measured quality report. Agent series add individual assets, a combined scene,
and a replayable recipe. Every member retains its role and stable ID; examples
include `vault_rib`, `flying_buttress`, `rose_tracery`, and `portal_gable`.
Window surrounds and tracery are pierced geometry. Glazing is a separate,
translucent mesh member that can be removed; west portals remain open. The
workbench draws panes black to suppress distracting rear edges through the
windows. Exports preserve green member colors and separate glazing alpha.

`MESH` is a validated render primitive for sculpted members. Vertices are stored
in the placement's normalized envelope, with explicit triangle faces. This
keeps translations, batch layout, JSON reload, and GLB exports consistent.
Profile extrusion and transported rib frames are implemented locally without
Blender or an optional triangulation package.

## Quality and visual evidence

The compiler checks finite positive transforms, unique IDs, exact compilation,
watertight meshes, consistent winding, positive volume, and parent links.
Cathedral-specific checks require its defining parts, coherent coarse kinds,
paired primary supports, and every component envelope connected to ground.
These checks run again on rebuild even if the document's mutable metadata is
removed. They are geometric diagnostics for a visual asset, not a structural
engineering calculation or construction certification.

The refinement process uses the requested
[dream-loop skill](https://github.com/newjordan/dream-loop), an imagegen target
based on the existing app, actual Qt/VTK captures, and independent visual review.
Working concepts and review records live in the ignored `.dream-loop` directory.
Generated images are references; the viewport and exported models use actual
procedural geometry.

## Repeatable capture and viewport verification

With the GUI dependencies, Pillow, FFmpeg and a working display installed:

```bash
python3 scripts/verify_cathedral_viewport.py
python3 scripts/render_architect_showcase.py
```

The viewport probe exercises actual cell picking, roof removal and restoration,
export-scene preservation, and appended-member picking. It records the local
renderer, source hash, viewport dimensions, median orbit FPS and frame times.
Run it without competing rendering or test processes for a useful measurement.

The showcase script first runs the documented agent export, then records real
Qt controls and VTK model views. It produces a 58-second 1920x1080 H.264 video,
a silent copy, an original synthesized soundtrack, stills and a source manifest
in `builds/cathedral-upgrade/showcase/`. Frames are rendered offline at 30 fps;
that playback rate is not an interactive-performance measurement. Existing kit
outputs are preserved. Use a new output directory after changing the architect.
