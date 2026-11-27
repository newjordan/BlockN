# Architecture rules for block mode

Status: seventh pass, 25 Nov 2026. Three regional layouts in: a Pueblo plaza
town (`blockregion.pueblo`), a Norse farm village (`blockregion.norse`) and
a Mediterranean town of courtyard-house blocks (`blockregion.mediterranean`),
each 5 of 5 town rules with every floor walked, Pueblo upper floors by ladder. Knowledge base, checker and seven style
sub-architects are in: Tudor and Hanseatic merchant houses (these line each
town's square and main street), a Norse longhouse, an Alpine chalet, a
Mediterranean courtyard house, a Pueblo terraced block and an English seld
(a lean-to lock-up shop, standalone, in a row between houses, or with its
door on the street line). Each passes every rule on seeds 0 to 3 but one
Hanseatic seed. Four of five test towns now pass all five town rules, with
street fronts 71 to 75 percent lined against the 70 percent bar; valley-wall
seed 2 is at 67 percent and its square reads a little too tall for its width.
Towns of the four new traditions build, but their houses fit our English plot
layout badly.

Block mode's architects knew how to make a building a player can walk. They did
not know how buildings are put together, so they made the same boxy medieval
house in four material sets. This is the layer that fixes that, in three parts:

1. **World knowledge** (`core/architecture/lore.py`): how buildings stand up,
   how roofs are covered, how facades are composed, what each kind of building
   is for, how towns are laid out, and the traditions that combine those by
   region and era. Every primary architect reads it.
2. **A checker** (`core/architecture/blockrules.py`): scores any block build
   against the rules, measured from the blocks alone. Because it needs nothing
   from our generator, it can score a rival's world too.
3. **Style sub-architects** (`core/architecture/blockstyles.py`): a sub-architect is a tradition and a
   purpose, say a Tudor merchant house or a Norse longhouse. It does not invent
   structure. It picks a system, a covering and ranges from the knowledge base,
   and the primary architect builds inside them.

## What the knowledge base holds

| Table | What it says |
| --- | --- |
| `SYSTEMS` | 8 structural systems: rubble masonry, ashlar, brick, timber box frame, cruck, log, adobe, East Asian post and beam. Each has its cantilever limit, storey range, widest opening, and rules. |
| `COVERINGS` | 11 roof coverings with their usual pitch range and eave depth, and which Minecraft blocks read as each. |
| `COMPOSITION` | Facade rules: openings stacked in bays, storey height, base, middle and top, window ratio, at least two depth planes and no long flat run, two to five material families, roof oversailing the walls, a chimney that clears the ridge, a load path. |
| `PURPOSES` | 14 building types (cottage, longhouse, hall house, row house, merchant house, tower house, courtyard house, barn, smithy, inn, chapel, market hall, granary, mill) with footprint, storeys, rooms and door rules. |
| `TOWN` | Street hierarchy, street enclosure, square size and enclosure, frontage, plot widths, landmarks. |
| `TRADITIONS` | 18 traditions by region and era, each naming its systems, coverings, purposes, palette, window ratio, features and climate, and which mode already builds it. |

Every rule carries a `basis`. `read` means a page read while writing it,
listed below. `practice` means common building practice that has not yet been
checked against a source. Those numbers are first guesses and should be
replaced as sources are read.

Pages read for this pass:

- North Norfolk design guide, roofs: pantile 35 to 45 degrees and never below
  30, medieval thatch 50 or over, slate as low as 25.
- Marley, minimum pitch per product: clay plain tiles 35 to 40, cedar
  shingles 14 (modern), Double Roman 22.5.
- Livable Cities, neighbourhood squares part 6: a square's long side under
  300 ft unless divided, recognising faces at 150 ft.
- SIF Hg, enclosure in urban design: 3:1 street width to building height in
  modern guides (and that it has no scientific basis), 6:1 for squares, Sitte's
  1:1 to 2:1.
- Looking at Buildings and Wikipedia on jetties: how they were built and when
  they were banned. Neither gives a projection figure; one block per storey is
  practice.

One rule was corrected after reading: the modern 1:1 to 3:1 street ratio
fails every medieval lane, so medieval towns are held to 0.3:1 to 1.5:1.

## What the checker measures

Per building:

- **load-path**: every structural block is carried. A block is carried when
  the one below is; when it reaches no further than its material's cantilever
  (two blocks for timber and roof coverings, one for stone) from a carried
  block; or when it lies on a straight run between two carried blocks no longer
  than a beam (9) or lintel (5) can span. Leaves, plants and cloth are not
  structural; a mill's sails are declared a hung mechanism.
- **roof-pitch**: the slope of the top blocks, held to the covering's range.
- **roof-covers**, **chimney**.
- Per facade, from a march in from each side to the first solid block:
  **facade-depth** (planes and relief), **flat-run** (the largest blank
  rectangle at one depth, in one material family, with no opening, counting
  only panels at least 3 by 3 so piers and chimney stacks are not walls; at
  most 12 blocks),
  **window-ratio**, **stacked-openings**, **materials**, **base-course**.
- **footprint** and **storeys** against the building's purpose.

Per town: **street-enclosure**, **frontage** (share of street edge with a
building within 2 blocks), **square-enclosure**, **square-size**,
**square-frontage**.

## Where our architects stand

Before this pass (the "after" is further down). All 22 buildings in a town's pool (four house styles at three seeds, three
towers, workshops and apartments, a windmill):

| Rule | Passed |
| --- | --- |
| roof-pitch, roof-covers, chimney, load-path, storeys, stacked-openings | all |
| facade-depth | 84 of 88 facades |
| materials | 81 of 88 |
| base-course | 77 of 88 |
| window-ratio | 62 of 88 (houses sit at 4.8 to 4.9 percent, just under the 5 percent floor) |
| covering-fits | 15 of 22 (deepslate "slate" roofs on brick houses and towers; slate is not an English medieval covering) |
| footprint | 15 of 22 (a third of houses are inn-sized; the house architect has no building type) |
| **flat-run** | **73 of 88** (apartments have blank panels of 42 to 80 blocks on every side; the windmill's tower 24 to 32) |

Towns (valley-wall and hill, seed 1): the square passes for size and
enclosure, but only 34 to 50 percent of its rim has a building on it, and only
46 to 60 percent of street edge is fronted. Hill passes street enclosure;
valley-wall's streets read too narrow for their buildings.

So the checker agrees with the pictures. The worst faults are flat walls,
gaps in the street fronts, and a square half ringed by grass. And block mode
builds one tradition out of eighteen.

### After the third pass

- **Walls**: an apartment's back stands against rock, so it is declared a
  party wall (`b.party_walls`), which the checker skips as it is no facade.
  The other walls get windows in a steady rhythm up every storey
  (`blockstyles.window_rhythm`), and any blank panel left is broken by a
  window, or a post where something solid stands behind
  (`blockstyles.relieve_flat_walls`, which reads the checker's own finding).
  The windmill gets the same repair. Flat-run now passes on **85 of 85**
  facades of the pool (was 73 of 88). Window ratio 65 of 85; the misses are
  the plain houses at 4.8 to 4.9 percent.
- **Town**: the merchant houses (a tradition per town) take the square's rim
  and the main street first. The outer ring street now stays a merchant's plot
  clear of the square, and the side streets leave from the square's far
  corners, not the middle of its long sides. Street and square heights are
  measured to the eaves, not the ridge.

| Town, seed 1 | Square rim built | Street edge fronted | Street enclosure | Square enclosure | Buildings |
| --- | --- | --- | --- | --- | --- |
| Valley wall, before | 50% | 46% | fails | passes | 61 |
| Valley wall, after | **82%** | 44% | fails (50% in range) | passes | 69 |
| Hill, before | 34% | 60% | passes | passes | 68 |
| Hill, after | **82%** | 57% | passes | passes | 75 |

Frontage did not move: the gaps are mostly street edge with open ground
beside it, on the outer ring where it runs close to the map's edge and on
the stair streets up steep ground. Keeping the ring a plot's depth from the
edge was tried and made it worse (40 to 55 percent, fewer houses), so it was
taken out. Merchant houses are three storeys: at four, a 15-block square
reads too enclosed. The windmill gets its own placement pass so a town keeps
one.

The merchant houses' backs get windows in the same rhythm, so both now pass
31 of 31 rules on seven of eight seeds (Hanseatic seed 0 has a rear window
ratio of 12.5 percent against 15).

## Styles: what exists and what is missing

| Tradition | Block mode | Mesh mode |
| --- | --- | --- |
| English medieval | yes (four material sets) | timberwright |
| Norman | towers and castle | castle |
| Greek and Roman, Gothic, Egyptian, Mesoamerican, Persian, Japanese, Victorian | no | yes |
| Tudor, Hanseatic | merchant house (sub-architect) | no |
| Norse | longhouse (sub-architect) | no |
| Alpine | chalet (sub-architect) | no |
| Mediterranean | courtyard house (sub-architect) | no |
| Pueblo | terraced room block (sub-architect) | no |
| Dutch, Aegean, Georgian | no | no |

## The first sub-architects

Each asks the primary house architect for the body (rooms, stair, floors, a
walkable plan), then dresses the gable end as the street front and turns the
house so the street door opens north, the way the town lays plots.

- **Tudor merchant house** (`tudor/merchant-house`): three storeys,
  seven or eight blocks of frontage, close studding in dark oak with lime
  infill, each upper storey jettied a block further over the street on a
  bressumer and brackets, a shop front with counters, mullioned windows, a
  plain-tile roof. Rules passed on seeds 0 to 3: 31 of 31 on each.
- **Hanseatic gable house** (`hanseatic/merchant-house`): brick, gable to the
  street, pilaster strips proud of the wall framing two- and three-block bays,
  stone string courses at the floors, tall windows set back, whitewashed blind
  niches, a stepped gable capped in stone, a hoist beam and loading doors, a
  pantile roof, side lights near the street end. Rules passed on seeds 0 to 3:
  30, 31, 31, 31 of 31 (seed 0 misses the rear window ratio).

The next four build their own body, because none of them is the primary
house with new clothes:

- **Norse longhouse** (`norse/longhouse`): 18 to 24 blocks long and 8 wide,
  bow-sided (the walls bulge at mid-length), log walls on a stone footing,
  outside raking posts, a turf roof of sods over a plank deck following the
  bow, an earth floor, a long hearth under a smoke hole, benches along the
  walls and hall posts carrying the ridge. No chimney, by tradition.
  26 of 26 on seeds 0 to 3.
- **Alpine chalet** (`alpine/cottage`): a rubble ground storey with quoins on
  an andesite plinth, a log storey above, a balcony on brackets round three
  sides, a wide shingle roof two blocks out over the walls, stacked windows,
  an outside stone chimney. 31 of 31 on seeds 0 to 3.
- **Mediterranean courtyard house** (`mediterranean/courtyard-house`): four
  two-storey wings round an open court, lime-washed (or sandstone) walls on a
  stone plinth, quoins and a string course, small windows with painted
  shutters in bays centred on each wall, a passage entrance, a stone stair up
  the court to the upper floor, low terracotta roofs at 27 degrees, a cistern.
  30 of 30 on seeds 0 to 3.
- **Pueblo terraced block** (`pueblo/row-house`): one house of a contiguous
  row (party walls east and west), two or three storeys of packed earth
  stepping back from the plaza, flat earth roofs on vigas whose ends run out
  through the wall head, mud-brick parapets with rain gaps, a mud-brick
  footing, few small openings, doors onto each terrace, ladders roof to roof,
  a beehive oven on the first terrace. 16 of 16 on seeds 0 to 3.

Building them showed four places where the checker was wrong, now fixed:
the eave is read from the outside only (a courtyard's stair is not the
eave the street sees); an open trapdoor is a shutter lying on the wall, not
the wall; a jump of more than two blocks a block is a setback or a parapet,
not a roof slope; a flat roof counts storeys to its highest roof. A ladder
is now a thin layer on a wall face, not a solid block.

`blockstyles.build(tradition, purpose, player, seed)` picks one;
`tests/test_blockstyles.py` holds every sub-architect to standing, facing the
street and at most two misses.

## Town wall

`blocktown.place_town_wall` runs a curtain wall with towers along the free
ground beyond each outer street, so a street that runs along the edge of the
town is lined on that side. Frontage went from 44 to 57 percent (valley) and
59 to 65 percent (hill); the hill's street enclosure now passes.

## Regional towns: first cut

`town(climate=...)` or `town(street_style=...)` now builds a town of another
region in that region's way: the tradition's own houses throughout, keeping
only the other buildings that belong there (no windmills in a Pueblo town).
It shows the next problem plainly. The layout was drawn for narrow English
plots, so the regional houses fit few of them: seed 1 holds 35 to 51
buildings (the temperate town holds 70), only 1 to 5 of the 10 street-front
houses find a plot near the square, and frontage drops to 43 to 53 percent.
The castle, the square's plinth and the workshops are still English. A
regional town needs its own layout: a Pueblo plaza ringed by joined room
blocks, Norse farmsteads round a hall, Mediterranean blocks of courtyard
houses wall to wall.

## Regional layout: Pueblo plaza

`blockregion.pueblo(player, seed)` lays out a Pueblo town its own way rather
than on English plots. A plaza about 36 by 29 blocks is ringed by joined
three-storey terraced room blocks that step back as they rise, all facing in;
a second ring behind a three-block lane faces the lane. One passage three
blocks wide runs from the south edge into the plaza. A kiva sits sunk five
blocks in the plaza, entered by a ladder down through its roof, with hornos
(beehive ovens), drying racks and walled fields of wheat, beetroot and
squash watered by ditches outside the rings.

Upper floors are reached the Pueblo way, by ladders, so the walker now
climbs ladders: `BlockBuild.climbs()` returns a box over each ladder run and
the wall behind it, and `gameplay.walk(climbs=...)` links every standing spot
inside a box both ways. Each ladder head opens a gap in the parapet above it.
Parapets are a low unbroken earth wall with wooden spouts (canales) through
their foot; the roof check now reads a thin upstand round a flat roof as a
wall, not a slope, so a parapet no longer fails a flat roof on pitch. A
building counts one floor for each level its doors open onto, so a roof
that is only a roof is not counted as a floor.

Seed 1: 61 blocks, 5 of 5 town rules, frontage 0.887, square enclosure
4.83:1, 183 of 183 floors reached from the entrance, kiva floor reached. The
lane between the rings is four blocks wide; at three, the taller unbroken
parapets pushed street enclosure under the bar. Without ladder climbing,
only ground floors are reached.

## Regional layout: Norse farm village

`blockregion.norse(player, seed)` follows Vorbasse in Jutland: fenced
farmsteads in two rows either side of one village street, each longhouse
long side to the street a block back, its yard behind it with one or two pit
houses (sunken-floored huts for weaving), a haystack and a woodpile. Halfway
along, a green opens off the street with the hall at its head (the longest
longhouse) and two farms facing in from its sides; a runestone, a well and
an old tree stand on it. Strip fields lie beyond the yards. The street runs
only as far as the last farms.

Seeds 0 to 3: 21 to 22 longhouses, 5 of 5 town rules; seed 1 frontage 0.72,
square enclosure 5.75:1 (the green is 23 by 24 and the longhouses' walls are
low), every longhouse walked from the street's west end.

## Regional layout: Mediterranean courtyard blocks

`blockregion.mediterranean(player, seed)` builds the insula grid of Roman
towns and the old towns of Andalusia and the south of Italy: blocks of two
rows of courtyard houses back to back, wall to wall along the row (each pair
shares its plinth), every house turned in on its court with only its door
and small shuttered windows on the street. Streets five blocks wide run
between the blocks; the outer rows look out over gardens. The middle block
is left open as the piazza, paved, with a fountain and potted cypresses, its
rim lined by the fronts of the blocks round it.

Seeds 0 to 3: 64 houses in 8 blocks, 5 of 5 town rules; seed 1 frontage
0.773, every street edge's width to eave inside the bar (median 0.56:1),
piazza 43 by 75 at 4.78:1, 128 of 128 floors walked from the west.

## Selds

`english-medieval/shop` is a seld: a booth one to four blocks deep, a timber frame on a rubble
sill, door and shop window to the street, a loft under a single-pitch
shingle roof rising to a rubble back wall that stands against the bank. 21 of
21 rules on seeds 0 to 15. Towns place them last, on street fronts no house
fitted. Seed 1: valley frontage 57 to 63 percent with 22 selds, hill 65 to
66 percent with 6 (69 in one run; the hill moves a few points run to run);
buildings 69 to 94 and 75 to 80. Most strips left beside an unlined street
are one to five cells deep, so the booth-depth selds matter.

### Sixth pass: row selds and street-door selds

Two more seld forms close most of the gap. A row seld has party walls both
sides, three to six blocks wide, for the gap between two houses. A street-door
seld stands its front wall on the street line itself, its door opening
straight onto the street, the counter in the window and a pentice board
over it; it fits a strip only as deep as the shop, most often the strip in
front of the back of a house on the next street. Rules: 21 of 21 standalone,
11 of 11 in a row, on every seed tried.

Two town fixes came with it. A house's eave over a street had been turning
the street cells under it into yard, so a lamp post could land in a shop
doorway; street cells now stay street. Street props now go in house yards
away from doors as well as on open ground, since built-up fronts left little
open ground (valley-wall seed 1: 52 props, 16 benches; every floor of all 140
buildings walked).

| Town | Frontage | Town rules |
|---|---|---|
| valley-wall seed 1 | 0.747 | 5 of 5 |
| hill seed 1 | 0.714 | 5 of 5 |
| hill seed 2 | 0.724 | 5 of 5 |
| valley seed 3 | 0.745 | 5 of 5 |
| valley-wall seed 2 | 0.673 | 3 of 5 (frontage, square enclosure 0.94:1) |

## Next

1. Valley-wall seed 2: lower the eaves round a square that reads too tall
   for its width, and line its last stair streets.
2. Regional layouts on real ground: hill and valley sites for the three
   regional towns, and a town wall for the Mediterranean grid.
3. Score a rival's build with the same checker once one of their saves can be
   read here.

## Commands

```python
from core.architecture import blockrules, lore
blockrules.score_building(build, tradition="english-medieval", purpose="cottage")
blockrules.score_town(town_build)
lore.coverage()
```
