# Brutalism Studio: architecture as a set of relations

A Brutalist building is not a gray building. This studio asks what the building
makes legible: where public life happens, which pieces carry which, what repeats
and what interrupts the repetition, and which material is actually doing the work.
It is an original design study, not a reconstruction or a definitive classifier.

## Historical anchors, not a parts catalog

| Precedent | Read it for | What BlockN borrows—and what it does not |
| --- | --- | --- |
| Reyner Banham, “The New Brutalism,” *The Architectural Review*, December 1955 | Image as a whole, clear exhibition of structure, and valuation of materials “as found” | A visually intelligible assembly, with material roles that can be audited. Early British New Brutalism was not limited to bare concrete: the Smithsons’ Hunstanton School exposes steel, brick and services. The studio models one concrete branch of a broader history. |
| [Boston City Hall](https://www.archdaily.com/117442/ad-classics-boston-city-hall-kallmann-mckinnell-knowles), Kallmann, McKinnell & Knowles | Differentiated civic functions given distinct masses and a relation to the plaza | A dominant council volume over a ceremonial threshold, flanked by smaller rhythms. No claim to reproduce the original plan, elevations or its public reception. |
| [Unité d’Habitation, Marseille](https://www.fondationlecorbusier.fr/en/work-architecture/achievements-unite-dhabitation-marseille-france-1945-1952/), Le Corbusier | Collective housing as more than a stack of flats: pilotis, loggias, collective space, roof and section | Repeated dwelling bays over a public undercroft and deep balcony edges. Duplex organization, internal streets, social programming and the original construction system are not modeled. |
| [SESC Pompéia, São Paulo](https://www.sescsp.org.br/unidades/pompeia/), Lina Bo Bardi | Existing fabric and new construction serving cultural life; bridges making relationships between separate volumes visible | Distinct gallery blocks joined by exposed walkways over a public passage. The factory reuse, rough irregular apertures and site history are not replicated. |

These are not interchangeable examples of one look. Boston’s municipal visibility,
Marseille’s collective dwelling and Pompéia’s public leisure imply different
briefs. The abstract portal field removes those programs altogether to test what
remains: threshold, cadence, shadow, and the experience of moving through solids.

## A rule is an invariant, not an object

| Rule | Geometry it audits | Transfer example |
| --- | --- | --- |
| `protected_void` | Reserve a named volume with a minimum width and headroom; boundaries may touch it but no member may occupy it. | Civic gateway → housing promenade → arts-center passage and skywalks → the axis of a sculpture field. In a gallery layout, the same predicate can preserve a sight or movement corridor. |
| `visible_bearing` | A named upper member must contact each named support on a horizontal face; default contact is at least 65% of the smaller span **in each** horizontal direction. | Girder on civic piers → dwelling slabs on party walls → bridge on seats → lintel on portal piers. The Rumah Gadang’s post/padstone and girder/post checks now call the **same predicate**, with full seating and its own centering rules. |
| `bay_rhythm` | The expected members must meet signed-off datums; the arts center additionally checks deck height alignment, adjacency and shared landing width. | Even dwelling bays → unequal gallery ranges → alternating intervals between portals. Rhythm need not mean a uniform grid. |
| `material_expression` | The exposed cast members use `cast-concrete`; glazing uses `window-glass` and must meet its flanking ribs behind a deep reveal. | Both housing and civic work distinguish the fabric that gives a building depth from the infill. The portal field keeps the concrete and legitimately drops the glazing. |
| `connected_circulation` | In the arts center, each plaza-to-bridge route follows actual treads, landings, and gallery doors. The audit checks edge width, riser height, tread depth, and headroom against the edited solids. | A continuous route is a stronger requirement than an empty corridor or two slabs at the same height. |

The build sequence is **brief → reserve voids → place supports and spans → add
repetition and infill → audit the edited placements**. Rule expectations are
reconstructed from the normalized request rather than read from cached metadata.
Audit failures identify the affected member IDs. The reusable rectangular
predicates are in [`core/architecture/grammar.py`](../core/architecture/grammar.py);
brief-specific rules belong to the architect, not to the predicate library.

## Comparative atlas

| Product | Matched pair | What changes |
| --- | --- | --- |
| `brutalist-civic-center` | `forum` / `assembly-gate` | Wider forum with lower threshold vs. narrower, taller ceremonial opening and a deeper council setback. |
| `brutalist-housing` | `slab` / `stepped` | Straight loggia edge vs. successive setbacks; party-wall bay datums remain aligned. |
| `brutalist-arts-center` | `sky-bridges` / `stepped-galleries` | Bridge levels and positions, gallery separation, and top gallery setbacks change without severing the landings. |
| `brutalist-portal-field` | `procession` / `syncopated` | Even intervals and a terminal gate vs. alternating short and long intervals and different lintel proportions. |

The [sixteen-model recipe](../examples/brutalism-atlas.json) groups four studies
per program. The first two in each group use the same default envelope, bays,
and seed with `variation=0`, isolating the variant. The last two replay explicit
controls sampled by `roll`. Seeds vary envelope, bay count, feasible housing
levels, and named geometry proportions; they do not waive the rules. Variation
is independent of surface color. `concept`, `standard`, and `presentation`
change the amount of shallow casting relief on the gateway and portals without
changing the main structural scheme. Other briefs currently share one level of
geometric detail.

```bash
# Replay and export all 16 studies and the laid-out district.
python -m blockn agent build examples/brutalism-atlas.json --output builds/brutalism-atlas

# Render the actual exported GLBs (optional: requires PyVista and Pillow).
python scripts/render_brutalism_atlas.py builds/brutalism-atlas

# Repeat one family; keep the variant while varying the controls.
python -m blockn roll brutalist-housing --variant stepped --seed 722 > housing.json
python -m blockn build housing.json --bundle builds/housing-722

# Fix the envelope, levels and variation for a controlled comparison.
python -m blockn make brutalist-housing --variant slab --width 45 --height 25 \
  --depth 23 --bays 6 --levels 6 --param variation=0 --bundle builds/housing-study
```

## Limits and the next meaningful tests

These are geometric studies, not permit-ready buildings or engineering checks.
Bearing does not model reinforcement, lateral stability, moment transfer or
capacity. The void predicate certifies an empty rectangular corridor in the
model, not accessibility, floodlighting, a fire exit, or a whole-building
circulation network. The arts-center switchbacks connect the plaza to both
bridges; lifts, apartment fit-out, daylight and
acoustics are not simulated. No style label alone can establish social success.

Next research priorities: (1) extend circulation checks to accessible and egress
routes, including sections through split levels; (2) explicit cantilever and
transfer-frame topology, distinct from a face-contact check; (3) adaptive reuse
and a broader material grammar for steel, brick and board-marked concrete;
(4) site-bound briefs that can test daylit courts, acoustic separation and
multiple access patterns instead of assuming a flat plinth.
