# Architecture Sonnets

A sonnet has a fixed form and no two are alike. A building tradition works the
same way: a stupa is circled by a railing, a ziggurat recedes, an onion dome
sits on a drum, and inside those rules the builder still chooses. A *sonnet
form* is a historic architect that states both halves as data: the rules it
will not break, and the *lines* it leaves open.

Nine architects are written this way. Each one is audited against rules
measured from the finished members, like every [historic
architect](HISTORIC_ARCHITECTS.md).

| Architect | Product | Variants | Verses |
| --- | --- | --- | --- |
| Mesopotamian | `mesopotamian-ziggurat` | `triple-stair`, `axial-stair` | 48 |
| Buddhist Stupa | `buddhist-stupa` | `great-stupa`, `dagoba` | 72 |
| Muscovite | `muscovite-church` | `five-domed`, `tent-roof` | 32 |
| Great Zimbabwe | `zimbabwe-enclosure` | `great-enclosure`, `valley-enclosure` | 48 |
| Lalibela | `lalibela-church` | `cruciform`, `colonnaded` | 48 |
| Art Deco | `artdeco-tower` | `spire-crown`, `stepped-crown` | 72 |
| Geodesic | `geodesic-dome` | `single-layer`, `double-layer` | 72 |
| Tibetan | `tibetan-palace` | `potala`, `dzong` | 32 |
| Mughal | `mughal-tomb` | `taj`, `humayun` | 48 |

```bash
python3 -m blockn sonnet mesopotamian-ziggurat --describe     # the form: its lines and answers
python3 -m blockn sonnet muscovite-church --count 12 > churches.json
python3 -m blockn sonnet muscovite-church --count 12 --detail concept --output builds/churches
python3 -m blockn sonnet mughal-tomb --count 6 --variant taj --hold dome=onion
python3 -m blockn make buddhist-stupa --randomize --seed 7 --param anda=bell --bundle builds/stupa
```

`examples/architecture-sonnets.json` and `examples/architecture-sonnets-stone.json` build one of
each form between them, with every verse written out. They are two recipes because one agent batch
holds at most 30,000 placements.

`scripts/render_sonnet_sheet.py` draws the same sets as phosphor contact sheets: `PRODUCT --count 12`
for variations on one theme, or `--anthology` for a row of verses from every form. The sheets in
`public/sonnets/` were made this way.

## Lines, verses and variations

A **line** is one named decision with a short list of answers the tradition
allows. The first answer is the canonical one. The ziggurat has four lines:

| Line | Answers |
| --- | --- |
| `stages` | `even`, `diminishing` |
| `buttresses` | `broad`, `close`, `paired` |
| `shrine` | `cella`, `twin-cella` |
| `crown` | `stepped-merlons`, `plain` |

The seed answers every line. The answers together are the building's
**verse**. With two variants, the ziggurat can be written in 48 verses. Inside
each verse the seed also moves the plot, the controls (bays, levels) and the
continuous proportions (a batter, a stair width), so two buildings with the
same verse are still not the same building.

Three things follow from stating the lines as data.

**A line can be pinned.** Name it in the request's parameters and the seed
writes the rest:

```bash
python3 -m blockn make mesopotamian-ziggurat --seed 12 --param shrine=twin-cella
```

An answer the tradition does not allow is refused, with the allowed answers in
the message. `variation=0` builds the canonical verse and proportions.

**A verse can be read without building.** It depends only on the request:

```python
from core.architecture import roll_request, sonnet

request = roll_request("mesopotamian-ziggurat", seed=11)
sonnet.verse(request)       # {'stages': 'diminishing', 'buttresses': 'close', ...}
```

The built document records the same verse in its `<architect>_rules` metadata.

**Variations can be dealt without repeats.** `sonnet.variations(product, count,
seed)` returns requests whose verses are all different for as long as the form
has verses left; after that, verses return with new proportions. `hold` pins
lines for the whole set. Each request has its full verse written into its
parameters, so a saved recipe says what it builds and replays the same way if
the form later gains an answer.

```python
sonnet.variations("muscovite-church", 12, seed=3, hold={"domes": "onion"})
sonnet.recipe("buddhist-stupa", 8)          # a kitbash recipe for `blockn agent build`
sonnet.describe("geodesic-dome")            # lines, answers and the verse count
```

`blockn sonnet` is the same thing on the command line. It prints a
`blockn.kitbash/v1` recipe, or builds it with `--output`. It works for every
historic product; a form with no stated lines varies by variant and proportion
only.

## What stays fixed

The envelope is the request's. A sonnet form fills the requested width and
depth and reaches the requested height exactly, whatever the verse, so a
variation never needs a larger plot than the one asked for. Each form states
its allowed width, height and depth, the ratios between them and the range of
its controls in `describe`, and `roll` samples only inside them.

The rules are checked on the geometry, not on what the architect says it
built. Each form's checks are listed with it below. They are geometric
diagnostics for visual assets and not a structural calculation; the one
physical check that applies to every build, bearing stress under accumulated
self-weight, is noted where it shaped a form.

## The forms

### Mesopotamian ziggurat

`mesopotamian-ziggurat` draws on the Ziggurat of Ur ([Ur, within the Ahwar of
Southern Iraq](https://whc.unesco.org/en/list/1481/)) as built by Ur-Namma
about 2100 BCE and on later Babylonian stage towers. A ziggurat is a solid
mass with no interior: sun-dried brick inside, a jacket of baked brick outside.
`triple-stair` has one stair on the axis and two laid against the front face,
all meeting at an arched gate on the first stage. `axial-stair` keeps the
single flight. `--levels` sets three to five stages.

The model's stages are set back further on the front than on the other sides,
so each upper flight has room to land. Shallow buttresses rib every face that
is not behind a stair. The summit shrine is glazed brick under a flat earth
roof; nothing of any real summit shrine survives, so it is a plain cella, or
two side by side, under a plain or stepped parapet.

One decision comes from the physics check and not from the record. Sun-dried
brick is held to a factor of safety of ten, which a core more than about 15 m
deep does not meet under its own weight. Core lifts deeper than that below the
summit are therefore laid in baked brick. Weep holes, reed matting and bitumen
courses are not modelled.

| Check | Meaning |
| --- | --- |
| `mesopotamian.receding_stages` | Each stage stands on the one below and inside its edge |
| `mesopotamian.battered_faces` | Every stage face leans inward as it rises |
| `mesopotamian.processional_stair` | The axial stair climbs without a break from the ground to the summit |
| `mesopotamian.summit_shrine` | The shrine stands on the highest stage, inside its edge |

### Buddhist stupa

`buddhist-stupa` draws on the Great Stupa at [Sanchi](https://whc.unesco.org/en/list/524/) and the
brick dagobas of [Anuradhapura](https://whc.unesco.org/en/list/200/). A stupa is a solid mound over
a relic. It is not entered; it is walked round. The plan is circular and is set out from the
shorter side of the plot, on a paved court that fills the plot.

`great-stupa` has a stone railing of posts, three rails and a coping, broken by four toranas. A
terrace drum is reached by two stairs laid against it on the south side, so the ground path stays
clear. The dome is laid in rings of radial stones over a brick core, with a square harmika on its
flat crown and a mast of parasol discs. `dagoba` has three basal rings instead of a walked terrace,
a low boundary wall, a stepped frontispiece (vahalkada) between two stelae on each axis, and the
parasols fused into a ringed cone. Its visible surfaces are lime plaster.

| Line | Answers | What it changes |
| --- | --- | --- |
| `anda` | `hemisphere`, `bubble`, `bell` | Profile of the dome |
| `parasols` | `triple`, `single`, `seven` | Parasol discs on the mast; in a dagoba, 7, 3 or 13 rings in the spire |
| `torana` | `three-architrave`, `two-architrave` | Architraves on each gateway; in a dagoba, registers of each frontispiece |
| `terrace` | `single`, `double` | One or two walked terraces, each with its own stairs; in a dagoba, `double` broadens the lowest basal ring |

The dome takes between 0.8 and 1.3 of its radius in height and the terrace takes up the rest. Not
modelled: relief carving, the curve and volute ends of the torana architraves, the relic chamber,
and the balustrade of the Sanchi stairs. The Sanchi stair is a double ramp; here it is two stepped
flights. The dagoba's basal rings are a simplification, not a survey of any one monument.

| Check | Meaning |
| --- | --- |
| `stupa.clear_path` | An annulus at least 1.5 m wide is clear between the drum, stairs and frontispieces and the railing |
| `stupa.cardinal_gateways` | Gateway members stand on all four axes and repeat exactly under a quarter turn |
| `stupa.dome_on_drum` | The dome starts at the terrace walk, is round, and sits at least 0.8 m inside the drum edge |
| `stupa.axis` | Harmika, mast and parasols are centred on the dome; the harmika sits on the crown; parasols diminish upward |

### Muscovite church

`muscovite-church` draws on the masonry churches of Muscovy from the late 15th to the 17th century.
`five-domed` follows the cathedrals of the [Moscow Kremlin](https://whc.unesco.org/en/list/545/).
`tent-roof` follows the [Church of the Ascension at Kolomenskoye](https://whc.unesco.org/en/list/634/).

The church is a tall white cube on a faced podium. A porch and steps stand on the west (-Z), and
one or three half-round apses project on the east (+Z). Each wall bay ends in a gable instead of a
level eave. Slit windows and the west portal have round arches with real voussoirs. In
`five-domed`, a great drum stands over four interior piers and four lesser drums stand on the
diagonals. In `tent-roof`, an octagon stands on the cube behind a ring of kokoshniki, and an
eight-sided tent narrows to one small drum. Every drum carries a dome and every dome a cross.
`--bays` (2–4) sets the gabled bays on each face of the cube.

| Line | Answers | Meaning |
| --- | --- | --- |
| `domes` | `onion`, `helmet` | The onion swells past its springing; the older helmet does not |
| `gables` | `zakomary`, `kokoshniki` | Round bay gables, or keeled gables with a second tier on the vault |
| `apses` | `three`, `one` | Altar apses on the east wall |
| `porch` | `gabled`, `tent` | Roof of the west porch |

Not modelled: the walls are white limestone, as in the Dormition Cathedral, and brick under
whitewash is not a separate finish. The vaults are a flat deck on four piers, not barrel vaults;
the drums and the low hip roof stand on that deck. The domes are sheet-covered gores without their
timber cradle. There are no openings from the nave into the apses, no iconostasis and no galleries.

| Check | Meaning |
| --- | --- |
| `muscovite.domes_on_drums` | Every dome springs from its own drum, no narrower than it, and every drum stands on the vault deck or the tent |
| `muscovite.crown_hierarchy` | The central cross is highest and four equal lesser domes stand on the diagonals, or the tent narrows to one drum |
| `muscovite.eastern_apses` | One or three apses stand against the east wall, inside its width and below the eaves |
| `muscovite.crosses` | Every dome carries a cross that starts at its tip and stands on its axis |

### Great Zimbabwe

`zimbabwe-enclosure` draws on the Great Enclosure of [Great Zimbabwe](https://whc.unesco.org/en/list/364/),
built by Shona masons between about 1100 and 1450. Granite is laid dry in level courses, with no
mortar. The outer wall is one closed curve fitted to the plot, laid as two battered faces over a
packed core. It rises in whole courses from a lower front to its full height at the rear.
Entrances are gaps in the wall with a stepped threshold. A solid conical tower stands free inside.
A decorated band and upright monoliths crown the full-height stretch of the wall. Round daga huts
under conical thatch stand in the yard.

`great-enclosure` adds the inner wall that runs beside the outer one and makes the narrow passage
toward the tower. `valley-enclosure` is a single wall, lower at the front.

| Line | Answers |
| --- | --- |
| `band` | `chevron`, `herringbone`, `dentelle` |
| `tower` | `single`, `twin` |
| `entrances` | `three`, `two` |
| `huts` | `ring`, `cluster` |

The model is a study of the type and not a survey. The plan is an ellipse, not the irregular curve
of the real enclosure, and the plot is about half the real size. Entrances are plain gaps without
the rounded wall ends. The interior platforms and smaller enclosures are not modelled. The band is
a schematic two-row pattern.

| Check | Meaning |
| --- | --- |
| `zimbabwe.curved_walls` | Outer-face stones lie on one closed curve; the foot course never runs straight or turns a corner |
| `zimbabwe.battered_walls` | At the rear, the wall is narrower at its head than at its foot, on both faces |
| `zimbabwe.solid_tower` | Each conical tower is filled along its axis from ground to head, tapers, and stands clear of every wall |
| `zimbabwe.crowning_band` | Every band stone sits at or above the wall head, over the full-height stretch at the rear |

### Lalibela rock-hewn church

`lalibela-church` draws on the monolithic churches of [Lalibela](https://whc.unesco.org/en/list/18/)
in the Ethiopian highlands. These churches were not built. A trench was cut down into volcanic
rock until a block stood free in a pit, and the block was hollowed into a church. In this model
y=0 is the pit floor and y=height is the natural ground, so `height` is the depth of the pit and
the church roof lies level with the plot surface.

`cruciform` follows Bete Giyorgis: a Greek cross of four equal arms on a stepped plinth.
`colonnaded` follows Bete Medhane Alem: a rectangular church ringed by square piers under its
roof, with two rows of pillars inside. Both have walls left in horizontal bands: a plain base,
blind lower windows, a projecting string course, open upper windows, a frieze and a cornice. A
rock-cut stair descends from the front edge of the plot to the pit floor.

| Line | Answers | Meaning |
| --- | --- | --- |
| `windows` | `ogee`, `cross`, `aksumite` | Pointed head, a cross left in the opening, or monkey-head bosses at the corners |
| `relief` | `nested-crosses`, `arcade` | Relief on the roof |
| `approach` | `open-stair`, `tunnel` | An open cut, or rock left standing over the foot of the stair |
| `plinth` | `three-step`, `two-step` | Steps under the church |

The rock is tuff. Rock members are large blocks tagged `living rock monolith`; the joints between
them are modelling cells, not masonry joints, and the roof, each plinth step and each wall band are
single blocks. Not modelled: the wider complex and its drainage trenches, interior arches and
carved ceilings, the pitched roof of Bete Medhane Alem, and the true double curve of an ogee head.
The stair is one straight flight at about 40 degrees; the real approaches are longer and wind.

| Check | Meaning |
| --- | --- |
| `lalibela.freestanding_in_pit` | The church stands wholly inside its pit with at least 2 m of clear trench on every side |
| `lalibela.roof_at_ground` | The roof is level with the surrounding rock surface and nothing rises above it |
| `lalibela.stepped_plinth` | Each plinth step stands on the one below and inside its edge |
| `lalibela.plan` | Cruciform: a Greek cross of four equal arms with open corners. Colonnaded: an unbroken ring of piers enclosing the church |
| `lalibela.descending_stair` | The rock-cut stair descends without a break from the ground to the pit floor |

### Art Deco setback tower

`artdeco-tower` is a law before it is a style. New York's 1916 Zoning Resolution let a building
fill its lot up to a street-wall height. Above that, every part had to stay under a sky-exposure
plane leaning in from the lot line. A tower on no more than a quarter of the lot could rise without
limit. The architect steps the mass back in tiers under that plane, raises a slender shaft, and
ends it in a crown. `spire-crown` ends in four diminishing copper tiers and a needle.
`stepped-crown` ends in three clad tiers, a glazed lantern and a mast.

The model takes a full-block site, so the plane applies on all four lot lines, at one metre of
setback for every four of rise. One bay grid serves every tier, so upper piers stand on the bay
lines of the tier below. Stone piers are cladding one tier high; each stands on the granite podium
or on a concealed iron frame column, so no pier carries the tiers above it.

| Line | Answers | Meaning |
| --- | --- | --- |
| `ornament` | `chevron`, `sunburst`, `ziggurat` | The one motif set on every setback parapet and crown tier |
| `piers` | `single`, `paired`, `finned` | Plain piers; a slender mullion in each bay; or piers that rise past the parapet as fins |
| `setbacks` | `symmetric`, `street-stepped` | Tower centred, or stood one bay back from the street with deeper front setbacks |
| `portal` | `triple-bay`, `stepped-arch` | Three glazed bays three storeys high, or one bay four storeys high closed by corbelled steps |

Not modelled: floor plates, the frame beyond the columns under the piers, lifts and entrance
steps. The district height multipliers of the real law are reduced to one slope. Crown tiers are
hollow boxes of plate. The frame columns are solid iron the size of a pier, so the reported mass
is too high. `sunburst` and `chevron` are straight-edged gablets, not arcs.

| Check | Meaning |
| --- | --- |
| `artdeco.sky_exposure` | Everything above the street wall stays under the sky-exposure plane or inside the tower footprint |
| `artdeco.quarter_lot_tower` | The tower covers no more than a quarter of the lot |
| `artdeco.receding_tiers` | Each tier starts at the roof of the tier below and stands inside its edge |
| `artdeco.unbroken_piers` | Each pier is one member from tier base to crest, stands on a support, and keeps the bay lines of the tier below |
| `artdeco.tapering_crown` | The crown narrows tier by tier, stays inside the tower, and ends in a finial at the very top |

### Geodesic dome

`geodesic-dome` builds Buckminster Fuller's lattice dome ([US patent 2,682,235](https://patents.google.com/patent/US2682235A/en)).
The faces of an icosahedron are divided at a frequency, with one corner at the zenith, and every
division point is pushed onto one sphere. Each point is a hub, each edge a straight strut between
two hubs, each face a triangle. A level plane cuts the sphere above the floor, and every hub on the
cut edge stands on a leg and a concrete footing. The plan is circular, so depth must equal width.
The shell is a hemisphere or more: height may be 0.45 to 0.76 of the width.

`single-layer` is the plain lattice. `double-layer` adds a second concentric lattice inside the
first, with a radial web strut at every hub and a diagonal web strut for every lattice edge. This
follows the principle of the Montreal Biosphere's space frame, not its geometry: the Biosphere's
inner layer is hexagonal, and the model repeats the triangulated lattice.

| Line | Answers | Meaning |
| --- | --- | --- |
| `frequency` | `3v`, `2v`, `4v` | Divisions of each icosahedron edge |
| `skin` | `glazed`, `open-frame`, `alternating` | Glass in every triangle, no panels, or sheet and glass in alternate triangles |
| `entry` | `portal`, `lifted-skirt` | A framed porch in the gap, or the bare gap |
| `base` | `legs`, `v-legs` | One plumb leg under each edge hub, or two raking legs |

A triangulated shell has no door-sized opening, so both entries lift the skirt: the hubs nearest
the entrance are left out and the hubs round the gap stand on longer legs. The seed also varies the
strut gauge, the height of the cut and the depth between the layers. Struts and hubs are
`wrought-iron`, the only ferrous substance in the catalog, and the opaque panels are `lead-sheet`
standing in for aluminium. Bolts, gaskets and the hub connection are not modelled, and the checks
are geometric, not an analysis of the lattice.

| Check | Meaning |
| --- | --- |
| `geodesic.spherical` | Every hub of a layer fits one sphere, and the layers are concentric |
| `geodesic.triangulated` | Every strut ends at two hubs; hubs not on a leg gather five or six struts; five-strut hubs are no closer than icosahedron corners |
| `geodesic.strut_families` | Lattice struts of a layer come in no more lengths than the frequency allows (2v: 2, 3v: 3, 4v: 6) |
| `geodesic.grounded_base` | Every hub with an open side stands on a leg whose foot is in a footing on the floor |

### Tibetan fortress-palace

`tibetan-palace` draws on the [Potala Palace](https://whc.unesco.org/en/list/707/) at Lhasa and on
the dzongs of Tibet and Bhutan. A faced, battered base fills the plot. Above it every wall leans
inward on one shared taper, so the masses narrow together as they rise. The lowest storey of each
block is blind. Windows start on the next storey and grow toward the roof. Each has a timber lintel
through the wall and a black surround that is wider at the sill than at the head. The central
block is red, rises above everything else, and carries a dark frieze (penbey) directly under its
parapet. Its roof holds gilded pavilion roofs or banner cylinders.

`potala` sets white wings on either side of the red palace, in one or two steps. `dzong` sets four
ranges under pitched timber roofs around a court, with a central tower (utse).

| Line | Answers |
| --- | --- |
| `wings` | `stepped`, `level` |
| `roofs` | `gilded-pavilions`, `banners` |
| `stair` | `zigzag`, `axial` |
| `frieze` | `single`, `double` |

The model is an exterior massing study. Interior floors, rooms and the court entrance of the dzong
are not modelled, and flat roofs span from wall to wall without the timber beneath them. The hill
is a level faced base, not terrain. Wall faces hidden behind a lower neighbour are not built.

| Check | Meaning |
| --- | --- |
| `tibetan.battered_walls` | Every block is narrower at its head than at its foot |
| `tibetan.windows_grow` | No window opens in the lowest storey, and window area does not shrink going up |
| `tibetan.central_block` | The central block rises at least 1.5 m above every wing or range |
| `tibetan.frieze_band` | The dark frieze lies above the central wall and directly under its parapet |
| `tibetan.approach_stair` | Stair treads climb without a break from the ground to the entrance sill |

### Mughal garden tomb

`mughal-tomb` draws on the [Taj Mahal](https://whc.unesco.org/en/list/252/) at Agra and
[Humayun's Tomb](https://whc.unesco.org/en/list/232/) at Delhi. The plot is square. A plinth fills
it, with a stair cut into its front edge. On the plinth stands a square tomb with cut corners. Each
main face has a pishtaq, a framed portal that rises above the wall head, holding a pointed iwan
with a lintelled door in the screen behind it. Pointed niches stand beside the portal and on the
cut corners. Eight piers carry a drum and the dome; four chhatris stand on the roof at the
diagonals.

`taj` is white marble with a high drum, a tall dome and four tapering minarets at the plinth
corners. `humayun` is red sandstone with marble arches, niche backs and dome, a taller plinth, a
broad low dome and no minarets.

| Line | Answers |
| --- | --- |
| `dome` | `onion`, `bulbous`, `pointed` |
| `niches` | `two-storey`, `single` |
| `chhatris` | `octagonal`, `square` |
| `finial` | `lotus-kalash`, `spire` |

One face, one cut corner, one chhatri and one minaret are composed and then turned onto the other
three sides, so the symmetry is constructed and then checked. The arches are two-centred pointed
arches of cut voussoirs with coursed spandrels; they are not the four-centred Mughal arch. The
dome is a single shell laid in rings of radial stones, solid where it closes; the inner dome of
the real double shell is not modelled. The niches are blind recesses backed by thin slabs. The
interior is the eight piers and a roof deck, with no chamber, cenotaph or vaults. The garden, the
water channels, the inlay and the calligraphy are not modelled.

| Check | Meaning |
| --- | --- |
| `mughal.fourfold_symmetry` | Every face, cut corner, kiosk and minaret has its twin a quarter turn away |
| `mughal.crowning_dome` | The dome sits on its drum over the centre, and the finial is the highest point |
| `mughal.pishtaq_rises` | On each of the four faces the portal rises above the wall head |
| `mughal.diagonal_kiosks` | Chhatris, and the four minarets of `taj`, stand at the diagonals below the dome |
| `mughal.tomb_on_plinth` | The tomb rests on the plinth, at least a metre inside its edge |

## Writing a new form

A sonnet form is one module beside `core/architecture/mesopotamian.py`, and one
name in `historic.ARCHITECTS`.

1. State the form with `sonnet.form(...)`: defaults, variants, required roles,
   `lines`, the `size` ranges, the `ratios` between them and any `controls`.
   The catalog, the desktop controls, `roll` and `blockn sonnet` read this.
2. Subclass `sonnet.Poem`. It checks the envelope against the stated ranges
   and answers the lines; `self.line("crown")` returns this building's answer.
   Use `self.vary(key, spread)` for continuous proportions.
3. Build from the shared members: `shell`, `run`, `ring`, `revolve` (a dome or
   drum in radial stone rings), `podium`, `fill`, `flight`, `conic`. Masonry is
   laid in stones a gang can lift; only fill and declared monoliths are large.
4. Fill the requested envelope exactly in every verse.
5. Write `rule_checks(document, form)` from the placements alone, and a test
   that breaks each rule and sees its check fail.
6. Add the tradition to `construction.TRADITIONS` with its rules and load path.
