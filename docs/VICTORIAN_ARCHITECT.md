# Victorian Architect

BlockN's Victorian architect composes British domestic and civic building
from about 1840 to 1900. It produces real meshes and named members: one-brick
load-bearing walls laid in running bond, timber joist floors spanning front
to back between the party walls, slated roofs pitched between 30 and 45
degrees over common rafters (50 to 60 on the Gothic villa), sash windows
seated in reveals under gauged segmental brick arches, bay windows on their
own plinths, and a chimney breast and stack on every party wall, carried up
through the roof as a raked, coped fire break. Floors are joisted, roofs
are rafters under slate courses, and every member carries its own role and
stable ID.

The desktop preserves the black-and-green phosphor wireframe. Select
**Victorian Terrace**, choose a plan and a finish, and generate. The emitted
bounds are authoritative for asset placement.

## Plans

| Variant | Composition |
| --- | --- |
| `two-up-two-down` | A workers' terrace of 4 to 10 houses, each with about a 4 to 5.15 m frontage. Every house has one front door, and the doors of adjoining houses pair under one shared gauged arch at the party wall. The fronts are flat, with a coursed plinth, one living-room sash at ground level and one sash per upper storey. A single-storey outshut runs across the back of each house behind the rear wall, with its own rear door, its own kitchen light and a slated lean-to roof |
| `bay-fronted` | A middle-class terrace of 3 to 8 houses with about a 5.1 to 7.15 m frontage. Each house carries a canted bay window on its own grounded plinth — a ground-floor bay, or a full two-storey bay that opens through the main wall under a banded lintel — with sashes above, set clear of the bay opening. A front garden wall with wrought-iron railings encloses a tiled path from the street gate to each door |
| `gothic-villa` | A single detached Gothic Revival house, 6.8 to 9.4 m wide, with an asymmetric plan: a gabled wing projects on one side, its flank walls returned to the main front, while the other half carries the sashes. The roofs are steep — 50 to 60 degrees — with raked, coped gables dressed by decorative bargeboards on every verge, polychrome brick banding through the walls, a pointed-arch porch on grounded piers before the wing door, and a pointed light in the wing gable |

These are original compositions built from the tradition's rules, not
replicas of named streets. The gauged arch, the party-wall fire break and
the canted bay organize the design rather than serve as applied ornament.

## Vocabulary

- **Party wall** — the one-brick (0.22 m) load-bearing wall between every
  pair of houses, running the full depth from front to back. It is coursed
  through the door-arch head and carried up through the roof as a raked
  parapet: its head follows the pitch of the slates on both sides, stands
  about 0.38 m (15 in, the London Acts' fire break) clear of the roof
  surface, and is finished with its own projecting stone raking coping. The
  raking top course is cut to the rake while the courses below keep level
  beds, so the brickwork steps up the slope in human-scale units. It carries
  a chimney breast on its inner face, and the stack rises out of the coping
  at the ridge.
- **Front wall / back wall** — one-brick walls coursed band by band, each
  band breaking its joints at the floor lines the joists pocket into. A
  three-storey front stands on a one-and-a-half brick (0.33 m) base that
  projects its extra half brick forward of the face above. Openings are true
  voids left by the coursing, not painted-on holes.
- **Gauged arch** — the segmental brick arch over every door and window:
  voussoir rings springing past the light's edges so each end embeds into
  the jamb masonry, with the spandrels above filled to the band top. The
  arch over a door pair spans both leaves and the party pier between them.
- **Sash window** — a timber case seated 9 cm back in the reveal: two
  stiles, a meeting rail, two panes, and a stone sill bedded on the coursing
  below the light, projecting past the face.
- **Joist floor** — softwood joists, 0.22 m deep, spanning front to back
  inside one house bay and pocketed 15 cm into both walls on the floor
  line; floorboards lie across them. Joist lines that would cross a
  chimney breast are framed around the hearth zone.
- **Outshut** — the single-storey rear extension of the workers' house:
  party-wall returns, its own back wall with a door and a small light, and
  a slated lean-to roof rising to the main wall.
- **Bay** — the canted projecting window: a stone plinth, a front face
  coursed with its sash lights, flank walls of bricks turned 45 degrees in
  plan, and a slated lean-to roof from the main wall down to the bay face,
  overhanging the canted corners. The bay is its own load path to its
  plinth. A full two-storey bay opens through the main wall under a
  banded lintel course bearing on the jambs.
- **Garden** — the bay-fronted front garden: brick piers at the party
  lines, a low coursed wall with wrought-iron railing bars and a handrail,
  a gate opening before each door, and encaustic path tiles from the
  street to the step.
- **Stack** — the two-flue chimney on its fire break: coursed drums, a
  lime-mortar bed, a stone coping, and two terracotta pots. Flues are not
  modelled; the stack mass is real.
- **Bargeboard** — the decorative softwood board hung under the coping along
  every gable verge of the Gothic villa, from eaves to apex. Its foot laps
  the slate, closing the verge so no rafter end and no slate edge shows in
  the gable elevation.

## Finishes

| Finish | Materials |
| --- | --- |
| `london-stock` | Yellow stock brick with red banding, limestone dressings |
| `red-engineering` | Red engineering brick with blue banding, sandstone dressings |
| `polychrome` | Red brick with yellow banding, sandstone dressings |

The banding is honest work: on the villa it alternates through the wall
courses; on the other plans it waits for the finish that asks for it.

## Design controls

| Control | Accepted values |
| --- | --- |
| `variant` | `auto`, `random`, or one of the plans above |
| `detail` | `concept`, `standard`, `presentation` |
| `finish` | one of the finishes above |
| `width` | 16.4 to 51.5 for the workers' terrace, 15.3 to 57.2 for the bay terrace, 6.8 to 9.4 for the villa |
| `depth` | 5.8 to 9.5 by plan |
| `bays` | the number of houses, 4 to 10 or 3 to 8; the villa ignores it |
| `levels` | 2 or 3 storeys |
| `height` | optional; read as the ridge target and clamped into the pitch band |
| `age` | 0 (newly built) to 1 (centuries); weathering only |

The standard request is width 32, depth 7.5, 6 houses over 2 storeys, `auto`
plan, `standard` detail, and the default surface age. Unsupported controls
are rejected. Randomness is confined to the design phase: changing `detail`
never moves a party wall, a door, a joist, a stack or a window arch, only
the coursing of the bricks between them. A fixed request and architect
version replay deterministically.

`bays` fixes the house count; `width` then fixes each frontage inside the
plan's band. Without `bays`, the width is shared out at the seeded
frontage. The villa clamps a wide request into its own band so a catalogue
default still resolves under `auto`.

Height sets the ridge: the eaves follow from the storey heights, and the
roof pitch is solved toward the request and clamped into 30 to 45 degrees on
the terraces and 50 to 60 degrees on the villa's steep gables.

## What is checked

The compiler checks finite positive transforms, unique IDs, exact
compilation, watertight meshes, consistent winding, positive volume, parent
links, and that every component envelope is connected to ground. The
Victorian-specific rule set measures the finished geometry:

- Load-bearing walls are one bonded brick thick — 0.205 to 0.35 m measured
  across the wall on the axis the stone was laid — with the one-and-a-half
  brick base allowed at street level.
- Every party wall and every gable is carried up through the roof as a
  raked parapet: at every point outside the stack it stands 0.25 to 0.5 m
  above the slate surface on either side, and each parapet carries its own
  projecting raking coping.
- Every opening carries a gauged arch whose ends embed into jamb masonry
  on both sides, or — for the through bay — a banded lintel course that
  spans the void and bears on both jambs; and every light is a true void,
  pierced through the wall.
- Joists span front to back inside one house bay, pocket into wall masonry
  whose top course is the joist's own base at both ends, and floorboards
  lie over them.
- The main roof pitch, measured from the common-rafter envelopes, falls
  inside 30 to 45 degrees on the terraces and 50 to 60 degrees on the villa,
  and the slate courses run from the eaves to the ridge. Every rafter
  records that envelope pitch. A lean-to outshut or bay rafter records its
  own shallower pitch and is not held to the main-roof band.
- Every party wall carries a grounded breast and a stack above the ridge,
  capped with a stone coping on a mortar bed and two pots; the stack rises
  out of the raked parapet's coping.
- Each villa gable is raked and coped to the same fire-break band, and every
  verge is closed by a decorative bargeboard hung under the coping and
  lapped over the slate.
- Each house keeps its frontage inside the plan's period band, and the
  house count inside the plan's range.
- The workers' terrace keeps its grounded rear outshut; the bay terrace
  keeps its bays, garden walls, railings and tiled path to every door; the
  villa keeps its bargeboards, pointed porch, pointed gable light and
  polychrome banding.

These checks run again on rebuild even if the document's mutable metadata
is removed. They are geometric diagnostics for a visual asset, not a
structural engineering calculation or construction certification.

## Practical exports

```bash
python3 -m blockn describe victorian-terrace
python3 -m blockn make victorian-terrace --variant two-up-two-down \
  --detail presentation --seed 7 --bundle builds/terrace-7
python3 -m blockn make victorian-terrace --variant bay-fronted \
  --param finish=red-engineering --bundle builds/bay-terrace
python3 -m blockn make victorian-terrace --variant gothic-villa --seed 42 \
  --levels 3 --bundle builds/villa-42
```

Bundles include GLB geometry, canonical architecture JSON, a manifest, and
the measured quality report. Every member retains its role and stable ID;
examples include `front_wall`, `gauged_arch`, `sash_stile`, `sill`,
`door_leaf`, `joist`, `floorboard`, `plate`, `rafter`, `slate_course`,
`bay_flank`, `railing_bar`, `path_tile`, `chimney_breast`, `parapet_coping`,
`stack_coping` and `chimney_pot`.

## Substances

Every member carries a physical substance. Brick walls are **fired clay**
(London stock, polychrome) or **engineering brick** (the red finish); the
roof covering is **slate**; joists, rafters, plates, boards, sashes and
bargeboards are **softwood**; glazing is **window glass**; railings are
**wrought iron**; the plates and copings bed in **lime mortar**; dressings,
sills, steps and copings are **limestone** or **sandstone**; the path is
**glazed tile**; the pots are **terracotta**. See
[MATERIALS.md](MATERIALS.md) for the measured densities and weathering.

## Limits

Flues are not modelled, though the stacks that gather them are real mass.
The outshut and bay roofs are honest lean-tos; the villa's wing roof dies
into the main front gable rather than forming a valley. Joists pocketed
into walls and sloping rafters are excluded from the reported vertical-load
ranking, as the coarse axis-aligned contact model cannot resolve their
beds — the walls, stacks and lintels that carry the real load are the ones
measured. Interior partitions and stairs are not modelled.
