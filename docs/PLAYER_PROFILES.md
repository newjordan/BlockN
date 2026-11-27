# Player profiles, voxels and playability

Status: step one plus settlements, jumps and the historic architects, 2026-11-23. Merged to main 2026-11-26.

A build can name the player it's for. The architects then size doors and
stairs for that body, and the quality report adds checks that the player can
actually get in and around. Without a profile nothing changes: requests,
documents and fingerprints are exactly what they were.

## Asking for a profile

```json
{"product": "room-shell", "openings": "door", "profile": "giant"}
```

`profile` is a preset name or an object. An object may start from a preset and
override any field:

```json
{"profile": {"preset": "brute", "height": 2.8, "camera": {"mode": "first_person"}}}
```

| Preset | Radius | Height | Step | Jump | Notes |
|---|---|---|---|---|---|
| `human` | 0.40 | 1.80 | 0.35 | 2.4 | the default |
| `small` | 0.25 | 1.10 | 0.22 | 1.4 | child or small creature |
| `brute` | 0.60 | 2.40 | 0.45 | 1.2 | |
| `giant` | 0.70 | 2.60 | 0.50 | 1.0 | the "8-foot character" |
| `runner` | 0.35 | 1.80 | 0.40 | 3.2 | runs at 12 m/s |
| `blocky` | 0.30 | 1.80 | 0.60 | 1.25 | Minecraft-like, first person |

Derived sizes (`PlayerProfile.clearances()`):

- door width = 2 × radius + 0.30, door height = height + 0.30
- corridor width = 4 × radius + 0.45 (two characters passing)
- ceiling = door height, or more for a third-person camera boom
- stair rise ≤ step height, going ≥ 1.6 × radius
- take-off speed v0 = √(2 g J), air time 2 v0 / g, jump reach = run speed × air time
- sight distance = run speed × reaction + run speed² / (2 × braking)

## What a profile changes in a build

- Named opening modes (`door`, `arch`, `mixed`) grow their doors to the
  profile's door size. A wall too low to fit that door plus a header is
  refused with an error that says so.
- Default stair step counts grow until every riser is one the player can walk.
- Explicit opening specs and explicit `steps` are left as written; the checks
  report them instead.
- `metadata.ports` lists each door and arch the wall runs cut, and
  `metadata.player` the profile name and derived sizes.
- Timber houses (`timber-house`) grow the front door, the entry steps and
  porch, and the stair and the hole above it to the profile's door, and raise
  the storey (and, if needed, the frame height) until that door fits under
  the lintel. A frame that can't fit it under 18 m is refused with an error.
- Settlements (`english-village`, `roman-town` and the rest) pass the profile
  to every building they place, so a giant's village is a village of giant
  doors.
- Norman keeps and castles (`norman-keep`, `norman-shell-keep`,
  `norman-castle`) grow every door and gate arch to the profile's door (wider,
  and sprung high enough to clear the door height across the body), widen the
  forebuilding stair, the round keep's stair and bridge to the door width, and
  draw the portcullis up clear of the door height. Behind the raised doors they
  add what a walking player needs: a vestibule floor in the forebuilding, a
  sill across the recess at the keep door, a floored passage through the wall
  core, and a boarded entrance floor inside the round keep. Inside, the
  square keep gets a newel stair in each half, in a well against the outer
  wall, with doors through the spine wall front and back, so every floor is
  walked to; the round keep gets a stair round its back wall with crescent
  board floors at each storey, down to the basement too. A last flight in
  each keep climbs on through a hatch in the roof to the wall-walk, and the
  square keep's corner turrets get doors on to the walks either side. For a
  body taller than 2 m the round keep's drum grows with the player, so the
  stair has headroom on every turn. Every riser stays within the player's step as
  the voxel walk measures it (a step rounds to whole 0.13 m cells). A
  castle's keep is smaller than a lone keep, so for a body taller than 2 m it
  grows with the player (by their height against 1.8 m, and to at least 21 m)
  until those stairs fit, staying inside the bailey.
- Roman baths (`classical-baths`) put a door-sized doorway into each chamber
  wall that sealed the back rooms, and size the range end doors to the
  player. Theatres and amphitheatres
  (`classical-theatre`, `classical-amphitheatre`) cut their aisles into steps
  no taller than the player's step and at least a door wide, and cut each
  tier of seating into rows, two aisle steps to a row, so the stairs read as
  scalae between fine rows of seats as in a real cavea; the theatre pulls
  its stage back to leave a door-wide passage beside it, and the amphitheatre
  opens two door-wide gates through the seating on its long axis.
- Gothic cathedral portals and the other classical forms are already far
  larger than any preset's door, so they build as they always have.
- Without a profile every one of these builds is exactly as before: the same
  placements and the same fingerprint. The one exception is the baths, which
  were reworked for every build on 2026-11-24: hall walls rise to the vault's
  springing with a thermal (half-moon) window in each bay, the chamber ranges
  get tiled gables, windows and end walls with doors.

## Playability checks

Run on a voxel grid of the compiled build, only for profiled requests:

| Code | Fails or warns when |
|---|---|
| `game.entry` | fail: there is enclosed floor but no way in from the ground outside; warning: the only way in is a jump |
| `game.interior_reach` | warning: under 90% of the enclosed floor is reachable (walking or jumping) |
| `game.stair_rise` | fail: a stair riser is taller than the player's step |
| `game.port_clearance` | warning: a doorway measures tighter than the profile's door |
| `game.headroom` | warning: interior headroom is under the profile's ceiling |

A settlement is checked one building at a time, each on its own grid, and
each check names the plots that fail it. `metrics.playability.buildings` has
the per-plot results.

### What the checks still report

- The castle's curtain wall-walks and mural towers have no stairs to them
  yet, so the castle passes entry and warns on interior reach. Wall stairs
  from the bailey are the next step.
- Enclosed voids nobody should enter, the roof space over a cathedral's vaults
  and the top of a rotunda's drum under its dome, count as enclosed floor. A
  cathedral's portals are wider than the closing that finds doorways, so its
  nave counts as outside. Both read as a failed entry though the floors people
  use are reached.

### How the walk works

- **Body.** A capsule: a disc of cells `radius` wide, `height` tall. Its rim
  may overlap anything a step tall, and its rounded ends pull in toward the
  rim (at distance d from the centre, r − √(r² − d²)), so it climbs treads
  shallower than it is wide and ducks a beam at its shoulder, as a character
  controller does.
- **Walking.** 4-way, up a step or down a safe drop, and off a ledge to land
  up to a body width out.
- **Jumps.** Straight ballistic arcs from the edge of a ledge or a gap:
  take-off speed v0 = √(2 g J), any run speed up to `run_speed`, landing dh
  above take-off at t = (v0 + √(v0² − 2 g dh)) / g. The body must fit along
  the arc. Only jumps the walk can't make are kept: up a ledge taller than a
  step, or across a gap.
- **Cells.** The default pitch is a third of the radius, never coarser than
  0.13 m, so a doorway built to the profile (0.15 m clear each side) measures
  passable. Heights and steps round to whole cells. A finer `--pitch` is more
  exact and slower.

## Pictures

```
python scripts/render_profiles.py norman-keep:square norman-keep:round --profile giant --detail standard --section --out docs/renders
python scripts/render_profiles.py classical-theatre classical-amphitheatre --profile giant --detail standard --max-cell 0 --out docs/renders
python scripts/render_gallery.py gothic-cathedral norman-castle --detail presentation --pitch 0.35 --azimuth 35 215 --out docs/renders/gallery
```

`core/architecture/render.py` draws a voxel grid as a shaded view with Pillow
and numpy (no GPU). `render_profiles.py` builds each product twice, historic
and sized for the profile, walks both with that player, and writes a sheet of
four views: each version from outside, and each cut just above the player's
head with the walk on the floor. The blue figure is the player at full size,
stood at the widest doorway the walk found. Green floor is walked to, amber
is reached only by a jump, red is enclosed floor the player never reaches.
`docs/renders/` holds the sheets and an `index-PROFILE.json` with the walk
numbers behind each one. The committed sheets are at standard detail, walked
at 0.13 m and drawn pooled to 0.3 m so a sheet renders in under two minutes.
The keep sheets add a third view, `--section`, cut through the middle front to
back so every floor and stair shows; the theatre sheets are drawn from every
cell (`--max-cell 0`), because pooling makes the aisle steps look ragged.
`render_gallery.py` draws exteriors only, unpooled, with no player, for
judging how a build looks; those are in `docs/renders/gallery/`. The
cathedral's az35 view is at 0.15 m: at 0.35 m its thin roof courses alias
into a dark lattice.

Still open in these pictures: the castle's curtain wall-walks and mural
towers are not reached; the square keep's forebuilding upper room is not
reached; the round keep's roof hatch is wide, because its stair turns close
to the middle of the drum, so much of the top storey stands open to the
sky.

## Voxels

```
blockn voxels room-shell --openings door --profile giant --out out/vox
blockn voxels room-shell --openings door --profile blocky --pitch 1.0 --mcfunction --out out/vox
```

Writes `NAME.voxels.json` (manifest) and `NAME.voxels.npz` (labels), and
optionally a Minecraft `.mcfunction`, an ASCII plan and merged blockout boxes.
The manifest is the merge contract, `blockn.voxels/v1`; its fields are
documented at the top of `core/architecture/voxels.py`: metres, Y up, ground
at y = 0, origin snapped to whole cells, 16-cell chunks, a substance palette
with suggested Minecraft blocks, and measured ports with facing for a world
system to snap to.

## Block mode

For a Minecraft export, `voxels --blocks` builds the castle, village and houses straight onto the 1 m grid instead of voxelizing meshes, sized to the profile in whole blocks, and walks them with the same checks. See `docs/BLOCK_MODE.md`.
