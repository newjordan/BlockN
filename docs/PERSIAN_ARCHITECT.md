# Persian Architect

BlockN's Persian architect composes court buildings from the rules of the
Iranian building tradition. It produces real meshes and named members: four
centred pointed arches, a two-storey riwaq of roofed aisles around a paved
sahn, iwans recessed into that band and framed by pishtaqs, a squinch
transition carrying a drum and a pointed dome, corbelled muqarnas hoods over
the principal portals, and tapering minarets with corbelled balconies. Nothing
is timber; every span
is an arch, a vault or a dome. Walls, piers, drums, dome rings and muqarnas
courses are laid as human-scale stones so each member carries its own role
and stable ID.

The desktop preserves the black-and-green phosphor wireframe. Select **Iwan
Court (Persian)**, choose a court plan and a finish, and generate. The
emitted bounds are authoritative for asset placement.

## Court plans

| Variant | Composition |
| --- | --- |
| `four-iwan` | A rectangular sahn inside one load-bearing outer wall. The two-storey riwaq fills the band between the court and that wall: piers and four-centred bays face the court on both storeys, and the gallery floor and the roof span the aisle back to the wall. An iwan is recessed into each side and opens through its pishtaq; the widest, the qibla iwan, continues through a passage arch into a square domed sanctuary. The outer wall is continuous up to the riwaq roof except for the entrance pishtaq on the north face, flanked by paired minarets, and it meets the sanctuary on the qibla side |
| `chahar-taq` | A square domed kiosk: four massive walls each pierced by one tall four-centred arch, corner squinches raising the octagonal transition to a sixteen-sided band, then a drum and a pointed dome |
| `caravanserai` | A fortified court with a single pishtaq entrance piercing the north rampart, a ring of hujra cells against that rampart, and a court arcade whose every bay is a portico roofed back to the cell front. Four iwans sit in the cell band and corner towers mark the rampart; no dome, no minarets |

These are original compositions built from the tradition's rules, not
replicas of named monuments. The four-centred arch, the squinch transition
and the muqarnas corbel organize the design rather than serve as applied
ornament.

## Vocabulary

- **Sahn** — the paved court.
- **Riwaq** — the roofed arcade between the open court and the outer wall.
  Coursed piers and four-centred bays face the court on both storeys. Behind
  each bay the aisle runs back to the enclosure, one aisle deep, or two when
  the qibla band is deeper than about 7 m, and back piers or the wall carry
  it. The gallery slab and the roof each span that full depth and bear on
  piers or walls at both ends. In the caravanserai the same bay is a
  single-storey portico: the roof runs from the arcade piers to the hujra
  front, so the walk in front of each cell is covered.
- **Iwan** — a vaulted hall walled on three sides, open to the court on the
  fourth. In the four-iwan court and the caravanserai it sits inside the
  band and opens through its pishtaq; its back is the outer wall, except the
  qibla iwan, whose back opens into the sanctuary. Its mouth carries a
  four-centred arch on jamb plinths, its roof is a pointed barrel vault of
  four-centred rings, and its depth stays between 0.8 and 1.6 times its width.
- **Pishtaq** — the tall rectangular portal frame that fronts an iwan or the
  entrance. Its top stands above the adjoining arcade roof.
- **Qibla iwan** — the widest and tallest iwan, on the axis of the
  sanctuary. Its back wall opens through the sanctuary's passage arch.
- **Chahar-taq** — a square chamber whose four arches carry corner squinches,
  the classic kiosk transition to a dome.
- **Squinch** — a small four-centred arch bridging each corner of the square
  chamber, springing from the wall top so the chamber becomes an octagon.
- **Muqarnas** — the corbelled hood of tiered niche cells over a portal arch
  head. Each tier steps back by at most half a cell depth over the tier
  below, and every row of cells stands on a masonry head course, so each
  cell's centre of mass rests on stones below it.
- **Hujra** — a cell of the caravanserai, walled between partitions against
  the rampart and opening onto the court through the roofed portico bay in
  front of it. The cell front carries that portico roof.

## Finishes

| Finish | Materials |
| --- | --- |
| `buff-turquoise` | Buff brick with turquoise tile |
| `sandstone-lapis` | Sandstone with lapis |
| `rose-azure` | Rose brick with azure tile |

## Design controls

| Control | Accepted values |
| --- | --- |
| `variant` | `auto`, `random`, or one of the plans above |
| `detail` | `concept`, `standard`, `presentation` |
| `finish` | one of the finishes above |
| `width`, `height`, `depth` | 18–60 by 8–22 by 20–70 |

The standard request is width 36, height 14, depth 46, `auto` plan,
`standard` detail. Unsupported controls are rejected. The architect authors
its own levels and openings; omit `levels` and keep `openings=auto`.
Randomness is confined to the design phase: changing `detail` never moves an
iwan, the dome or a minaret, only the coursing of the stones between them.
A fixed request and architect version replay deterministically.

Height sets the qibla axis: the qibla iwan scales with it, the arcade stacks
two storeys beside it, and the sanctuary dome apex lands near the requested
height with the minaret tips a fixed allowance above.

## Practical exports

```bash
python3 -m blockn describe iwan-court
python3 -m blockn make iwan-court --variant four-iwan \
  --detail presentation --seed 7 --bundle builds/iwan-7
python3 -m blockn make iwan-court --variant chahar-taq \
  --param finish=sandstone-lapis --bundle builds/kiosk
python3 -m blockn make iwan-court --variant caravanserai --seed 42 \
  --bundle builds/serai-42
```

Bundles include GLB geometry, canonical architecture JSON, a manifest, and
the measured quality report. Every member retains its role and stable ID;
examples include `riwaq_bay`, `iwan`, `entrance_arch`, `squinch`,
`transition_band`, `drum`, `dome_course`, `muqarnas_cell` and
`minaret_balcony`.

## What is checked

The compiler checks finite positive transforms, unique IDs, exact
compilation, watertight meshes, consistent winding, positive volume, parent
links, and that every component envelope is connected to ground. The
Persian-specific rule set measures the finished geometry:

- Every opening and arcade bay is a four-centred pointed arch with
  rise-to-span between 0.45 and 0.75; the arch is laid as ring courses of
  stones that follow the curve, and every springing bears on a pier or wall
  course below its own footprint.
- A square chamber becomes an octagon through four corner squinches, then a
  sixteen-sided band, then a drum whose diameter is at most the chamber
  side; the dome springs from the drum. Each stage sits on the stage below,
  with no band panel overhanging its support by more than a third of its
  depth.
- The dome's meridian is the four-centred curve rotated about the drum, so
  it springs vertically and closes at a true point; its apex rises between
  0.55 and 0.8 of its diameter, in at least eight named courses.
- Each iwan's depth is between 0.8 and 1.6 times its width, its three walls
  are present, and its pishtaq stands above the arcade roof. The east, west
  and north iwans are recessed into the band; their backs are the outer wall.
- The four-iwan court reads as one walled rectangle: an outer wall rises to
  the riwaq roof on every side, open only at the north portal and at the
  qibla passage into the sanctuary. Every riwaq bay has a gallery slab and a
  roof across its depth, each bearing on a pier or wall at both ends.
- Every caravanserai arcade bay is a portico roofed back to the hujra front
  behind it, bearing on the arcade piers and on that front wall.
- The qibla and portal hoods, and the arch heads of the kiosk, are muqarnas
  of three to six corbelled tiers, each tier stepping back no more than half
  a cell depth, every cell resting on masonry.
- Court, iwans, sanctuary and portal are symmetric about the qibla axis,
  and the qibla iwan is strictly the largest.
- Minarets are cylindrical and taper upward, each balcony corbels out no
  more than half its step depth, and the tips stand above the dome apex.
- Nothing is timber and no span is a lintel: every span is an arch, a vault
  or a dome.

These checks run again on rebuild even if the document's mutable metadata is
removed. They are geometric diagnostics for a visual asset, not a structural
engineering calculation or construction certification.
