# Classical construction audit

Blocks is a procedural architecture workbench for creating inspectable 3D
assets, composing building kits and exporting repeatable geometry. The craft
layer describes how architectural members belong together; it is separate from
the black and green wireframe presentation.

## Findings and corrections

| Area | Previous construction | Corrected geometry |
|---|---|---|
| Greek Doric | Straight polygonal shaft on a separate block base | Baseless shaft, 20 concave flutes, taper/entasis, annuli and spreading echinus |
| Ionic | Straight cylinder and broad slab capital without scrolls | 24 flutes with fillets, torus/scotia base, cushion and opposed spiral volutes |
| Other orders | Cubes used as foliage and cylindrical scroll substitutes | Shaped leaf solids and spirals; smooth Tuscan shaft and simple moulded base |
| Temple bearings | Side beams projected outside column centerlines | Beam bands centered over abaci, checked for positive projected bearing |
| Temple envelope | Voxel podium and solid triangular roof slices | Three continuous step courses, recessed cella with a doorway, thin roof panels enclosing an open attic |
| Theatre and amphitheatre | Axis-aligned blocks approximating curved seating | Actual radial/elliptical sector courses and stepped aisle openings |
| Basilica | Separate apse blocks and flat roof slab | Joined polygonal apse walls, connected apse roof and pitched nave roof |
| Rotunda | Filled drum/disc and solid domed mass | Annular bearing and hollow dome with an oculus |
| Baths | Domes on incomplete bearings and weak side-wing support | Groin-vault hall, domed caldarium, supported wings and partitions |
| Walls | One full-height slab per run, openings as leftover header/sill boxes | Human-scale ashlar courses in running bond; openings are true voids spanned by lintel stones |

The Doric and Ionic profiles follow the distinctions described by the
[Metropolitan Museum of Art](https://www.metmuseum.org/essays/architecture-in-ancient-greece).
Their modular interpretation and entablature arrangement also refer to
[Vitruvius III](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Vitruvius/3%2A.html)
and [IV](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Vitruvius/4%2A.html).
These are reusable design interpretations, not measured reconstructions of a
particular monument. The Greek Doric choice is intentionally distinct from
later Roman versions with bases. Corinthian and Composite ornament remains
stylized. No generated reference image is used as historical evidence.

## Inspect and reuse

```bash
python3 -m blockn craft
python3 -m blockn describe classical-building
python3 -m blockn make classical-building --variant doric --seed 73 --bundle builds/doric
python3 -m blockn agent build examples/through-the-ages.json --output builds/through-the-ages --format glb --format obj
```

Every bundle includes a `.construction.json` report containing the tradition,
all members grouped into assembly stages, and relevant quality checks. The
catalog covers Greek orders, Roman masonry, later classical orders and Gothic
construction vocabulary. An agent can inspect the rules before composing a kit
and return both geometry and construction reports afterward.

Classical Architect 3 uses `column_*` roles consistently for column members,
including `column_capital_foliage` and `column_capital_volute`. Their original
`element` tags remain available in placement parameters. Bundle receipts now
include the additional `construction` artifact; consumers should read the
returned paths instead of assuming a fixed four-file bundle.

## Validation boundaries

Regression tests measure flute counts, taper and entasis; check spiral openings,
member volume/winding/watertightness, hollow interiors, apse joints, exact export
replay and per-capital bearing. Negative tests remove an Ionic scroll, substitute
a plain cylinder while retaining a flute claim, and displace a temple lintel.
Those damaged builds fail even without editable invariant metadata.

Projected bearing and envelope connectivity are geometric diagnostics. They do
not calculate allowable spans, stress, thrust lines, wind/seismic stability,
connections, soil behavior or code compliance. This release produces architectural
assets; actual construction still requires project-specific engineering.
