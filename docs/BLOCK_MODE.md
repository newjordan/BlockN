Status: eleventh pass, 2026-11-24. Block mode builds a castle, a village street, single houses with panelled rooms furnished by use, towers, workshops, windmills, mines and apartment blocks, and a castle town fitted to a given height map that calls them by plot role, sets them into the rock and stands flats against tall faces with doors out the back onto the ground above, and dresses the streets (a fountain and striped stalls on the square, benches, barrels, hay, flower tubs and wells off the streets, lamp posts every five blocks), straight onto Minecraft's 1 m grid, sized to a player profile. Chests, barrels, banners, item frames, paintings and mine carts are exported as block entities and entities (Java 1.21 form). `blockworld` writes a whole build as a Java 1.21 world save, which the Chunky path tracer loads and renders. Builds can be pictured from a player's eye with Minecraft-style light. Mesh mode is unchanged.

# Block mode

Mesh mode builds exact geometry and voxelizes it. At Minecraft's 1 m block size that aliases: a 40 degree roof becomes a ragged staircase, a thin sill vanishes or doubles, tracery breaks into loose blocks. Block mode designs to the grid instead.

- Walls, roof pitches, stairs and doors are whole blocks. Roofs are 45 degree courses of stair blocks with slab ridges and overhangs; turret roofs are steep cones.
- Every block carries its Minecraft block state (`minecraft:spruce_stairs[facing=east,half=bottom]`), so the export is what was designed.
- Materials are mixed by seed (stone bricks with mossy and cracked ones, cobble bases, timber frames, plaster, brick), with windows, shutters, lanterns, banners, furniture, gardens, fields and trees.

## Using it

```
python cli.py voxels castle --blocks --profile blocky --seed 1 --out out/castle --name castle --render
python cli.py voxels village --blocks --out out/village --name village --render
python cli.py voxels house --blocks --profile giant --out out/house --name house
python cli.py voxels mine --blocks --profile giant --seed 2 --out out/mine --name mine
```

`tower`, `workshop`, `windmill` and `mine` are block-mode products too.

`norman-castle` and `english-village` are accepted as names too. Each run writes:

| File | What it is |
|---|---|
| `NAME.mcfunction` | `setblock` and `fill` commands with full block states, solid blocks first and attached ones (doors, lanterns, banners, plants) last, then `summon` commands for entities. Block entity data rides on its `setblock` as SNBT. Run it standing where block (0, 0, 0) should be; paths and fields go in the layer below your feet. A mine's `cave_air` carves the hill it is run in. |
| `NAME.voxels.json` / `.npz` | A `blockn.voxels/v1` grid at 1 m, one cell per block, its palette holding each block state. |
| `NAME.blocks.json` | The build's manifest: the sizes used, blocks by type, declared doorways, the walk summary and the doorways the walk measured. |
| `NAME.png` | With `--render`, a picture. |

From Python: `core.architecture.blockcraft.build("castle", "blocky", seed)` returns a `BlockBuild`; see `core/architecture/blockmode.py`.

## Sized to the player

`BlockSizes.of(profile)` turns a profile into whole blocks:

| Profile | Door (w x h) | Room height | Floor to floor | Stair, walk lane |
|---|---|---|---|---|
| blocky (Minecraft-like) | 1 x 2 | 4 | 5 | 1 |
| human | 1 x 2 | 4 | 5 | 1 |
| brute | 2 x 3 | 5 | 6 | 2 |
| giant | 2 x 3 | 6 | 7 | 2 |

Openings follow Minecraft's own rule: the smallest whole-block opening the body fits through, so a 1.8 m player takes a 2-block door. Rooms are two blocks taller than the door, and a block taller than the body's own clearance: a room a door's height plus one reads as cramped from a player's eye. Stair flights are a lane wide with the floor above cut back far enough for the body's head; the castle's wall-walk is a lane plus one wide.

Stair blocks rise 0.5 m. A body whose step is below 0.5 m (the `human` preset steps 0.35 m) can only climb them by jumping; the walk reports that as reached by a jump, not walking. Block mode is meant for Minecraft-like bodies.

## Checked the same way as mesh mode

The walk (`gameplay.walk`) runs unchanged on a collision grid in which every block has its true shape at 0.25 m: a bottom slab is its lower half, a stair its lower half plus the back of the upper half, fences and walls 1.5 m tall, doors, carpets and plants nothing. Where a body rounded to 0.25 m cells would not fit its own doorway (`human`), the grid is 0.125 m.

Every door an architect places is recorded as a port with its world position, facing, clear width and height in metres and blocks, and the profile it was sized for. The walk's measured ports are written beside them.

First results, seed 1:

- House (blocky, giant): every interior cell reached; the stair to the upper floor is walked.
- Village of eight houses (blocky): every interior cell reached.
- Castle (blocky): the bailey, both stairs to the wall-walk, the whole wall-walk through all four corner towers and the gatehouse, every floor of the keep and its roof walk are reached.

Corner towers end in a stone deck with a crenellated parapet and a slender spire (each ring a layer taller than the one outside it, topped with a finial), over a ring of corbels; the keep's turrets get the same spires. Roofs are spruce, dark oak or deepslate tiles by seed. Lanterns hang from stone brackets every seven blocks along both faces of the curtain, and the keep has a balcony over its door at the hall, which the hall opens onto. Houses of two or more storeys get a balcony with a fence rail over the door and a lantern under it, and flower boxes under their upper windows.

Castles carry timber work as ones that grew a town about them do: the keep's top level is timber-framed (posts every third block, plaster or planks between, windows kept), and stretches of the curtain, either side of the gatehouse and along the other three walls, carry roofed timber hoardings over the wall-walk, their framed face where the parapet was, on joists out of the outer face, with a lantern under each roof. Vines hang in strands down the outer faces of the curtain, the towers and the keep, each clinging to the stone behind it; vines are drawn as a sheet on that face and are not collided with.

The keep has three levels: a guard room, a great hall two blocks taller than a storey with tall windows, and the lord's chamber. The guard room and the hall each get a long table of top slabs down the middle with chairs either side and lit candles, a carpet round it, and lanterns hung on chains over it; the hall also has a throne on the wall away from the stairs under banners and torches. Each piece is taken out again if it would cut floor off from the player, so every floor cell of the keep is still walked.

## Furnished rooms

Each storey of a house gets a use: a kitchen on the ground floor (furnace, smoker, crafting table, a table with chairs, cauldron, bookcase, chests and barrels with plants and lanterns on them), a study and a bedroom above (beds with a nightstand and candle, lectern, cartography table, bookcases), or all of it in one room for a cottage. Each house has a fireplace in its chimney wall with a brick hearth, a bordered rug on its open floor, potted plants in empty corners, a banner on a bare wall and stripped-log beams laid into each ceiling.

Two-storey houses deep enough for it (seven in ten of them) have a gallery: the upper floor covers only the back of the house, over the stair, and the front of the main room is open to the rafters, two storeys high, with a fence rail along the gallery's edge and a lantern hung from the roof on a chain. Its panelling runs up to the roof and turns to glass behind the upper windows, so no top or sill is a ledge, and its furniture stops a block short of the gallery's floor.

Each house is finished in its own palette, chosen by seed: a two-tone wall in each panelled room (a darker wainscot at the foot and a frieze under the ceiling, a lighter field between: plaster, birch, packed mud, or brick and mud brick in a kitchen, dark wood or bookcases in a study), stripped-log trim or, in some houses, posts grown over with leaves, and a floor of its own in each storey with a darker border inside the walls (polished andesite or mud brick in some kitchens). Each house has a trade, the baker's, smith's, weaver's, brewer's, scribe's or farmer's, whose workstation and goods stand in the ground-floor room (a lit smoker and hay bales, a blast furnace, smithing table and anvil, a loom and wool, a brewing stand and cauldron, a cartography table and chiseled bookshelf). Past the essentials a room's pieces come in a seeded order, with crates stacked in corners, decorated pots and barrels or chests turned by seed, and up to three built-in shelves let into the panelling above the wainscot (chiseled bookshelves in a study, barrel cubbies in a kitchen), each replacing a panelling block so nothing a body meets changes. Item frames and paintings are entities in Minecraft, not blocks, so block mode does not place them.

Rooms with a ceiling in houses of two or more storeys are panelled: a ring of planks (glass panes where the outer wall is glass) inside the outer wall, with stripped-log pilasters at the corners and every few blocks. The panelling leaves the door, the cell inside it, the hearth and the stair clear, and a storey keeps its panelling only if the walk still reaches all of its floor. Against the panelling stand hutches (a chest, smoker or barrel with a barrel and a candle or pot above), shelves (a top slab over block-high furniture, with a pot, candle or lantern on it), and under the ceiling hang lanterns, one to about fourteen floor blocks, with azalea leaves trailing from the corners where the layer is above every head.

`furnish` places each piece against a wall, turned to face the room, and tries the four turns of larger pieces (a bed's head goes to the wall). It keeps the lane from the door to the stairs clear and refuses any piece that would cut off floor from a body as wide as the player's lane. A room with a ceiling takes bookcases up to it; a room open to the roof takes nothing over a block high, so no furniture top becomes floor that nobody can reach. Every floor cell the furniture leaves is walked; the tops of furniture are left to jumps.

Pictures show furniture detail (books on shelves, chest lids, furnace mouths, pillows, flowers, flames, lantern tops and chains, thin candles) from the visual grid only; the export and the walk use the block states and true shapes.

## Pictures from a player's eye

`render_lit.render_view(build, eye, look_at, fov=70, sun=0.85, glow=0.0)` draws what a player standing in the build sees, for comparing with in-game screenshots; `render_lit.room_camera(build, floor_y)` picks a standing spot and a look across a room, as a screenshot taken on walking in would be.

- Rays from the eye walk the 0.25 m picture grid, so visibility is exact.
- Light is Minecraft's: block light spreads from lanterns, torches, candles, campfires and lit furnaces, falling one level a block; sky light pours down open columns and in through windows and doors; faces take smooth light between block centres and darken in inside corners (ambient occlusion). `sun` sets the time of day (0.2 is dusk).
- `glow` adds what shader packs add: direct warm light from each source near the eye, with distance falloff and hard shadows from opaque blocks. `sun_dir` likewise adds sunlight with cast shadows; the sky has a gradient and blocky clouds, and `ground_y` sets the plain drawn beyond the build.
- Faces carry a 16 by 16 texel pattern from their block's colour: plank seams, bark and log rings, brick and stone courses, cobble and leaf noise.

It uses no game textures, so it is a fair stand-in for lighting, space and composition, not for texture packs.

## Towns on given ground

`town` (also `castle-town`) fits a castle and a dense town to a height map that it is given; it does not make terrain.

```
python cli.py voxels town --blocks --heightmap heights.npy --out out/town --name town --render
python cli.py voxels town --blocks --terrain slope --seed 1 --out out/town --name town --render
```

The height map is a 2-D array of whole-block ground heights indexed `[x][z]` (`.npy` or `.json`). Without one, `--terrain` picks a sample (`hill`, `slope`, `valley`, `valley-wall`) from `blocktown.sample_heights`, for tests and pictures; `valley-wall` is a valley floor under a wall rising some 60 blocks, curving round at both ends.

The planner:

1. puts the castle where the ground is high and level near the middle, turned so its gate has room in front, on a platform cut and filled to one height, never more than 14 blocks deep (so not perched part way up a steep wall);
2. levels a market square in front of the gate, with a well, stalls and lamps;
3. routes a main street from the square to the edge of the map, a lane round the castle, an outer ring two plot depths beyond it, and side streets off the main street and the castle lane that run out across the outer ring, around steep ground (A* with a cost on every block of climb); each street's surface is eased to follow the ground with no step over one block, and a stair block sits at every rise;
4. where a side of the castle rises steeply (14 blocks or more over two plot depths), climbs it in terraces: a street along the contour every two plot depths up the slope, so a row of houses stands either side of it, joined by switchback ramps at alternate ends that rise a block a step at most, and a stair from the lowest terrace down to the streets below; other streets stop where they would leave steep ground by more than four blocks, and terraced ground is walled in stone and planted with bushes;
5. fronts buildings onto every street and the square, each on a pad levelled to the street's height. The type follows the plot: workshops near the square, on the main street and on some ordinary fronts (up to 8, 8 blocks apart), towers on the highest street fronts (up to 4, 22 apart), windmills on the outskirts away from the gate on gentle ground (up to 2, 30 apart), houses everywhere else. Only the ground under the walls and before the door is cut to the floor: rising ground beside a building is left as rock against its walls (set into the slope), falling ground up to five blocks down is crossed on timber stilts, and deeper drops become a crag of rock raised to the floor; half the houses are narrow town houses at the smallest size their stair allows, to fill tight plots; eaves may hang over a street, or meet the eaves of a house on nearly the same terrace so roofs run on in a row, never over another house's walls;
6. digs a mine into the hill off a street where the rock over every passage stays at least a block thick (and three under a street), its portal floor flush with the street; then fields on gentle ground near the streets (never over the mine), lamp posts, trees, flowers and grass;
7. lays the ground as blocks: grass or the street's surface on top, dirt, then stone; where a column stands two or more blocks above a neighbour it shows the hill's rock in bands of stone, andesite, tuff, diorite and granite, or a retaining wall of stone bricks and cobble where it holds up a street, the square, a field or the castle.

`town_walk` walks the whole town at 0.25 m from where the main street leaves the map, and the mine from its portal. On the four samples (seed 1, `blocky`) every floor of every building (56 to 79; 61 on the valley wall, up three terraces), the castle bailey and all of its wall-walk are reached, and on the valley wall every floor cell of the mine (192). The other samples have no hill face deep enough for a mine, and the giant's valley-wall town has no site for its wider, taller one. For the giant the castle wall-walk is reached only at its stairs (share 0.001); that was so before this pass.

Pictures use the block renderer's whole-block path: a block that fills its cube in one material is drawn as one face per side, so a whole town renders in a few seconds.

## Building types and mines

`blockkinds` adds three building types on the house's frame and one dug architect:

- **tower**: square, three or four storeys of stone brick on a deepslate frame, a hipped roof with a lip, corbels under the eaves and a banner flag.
- **workshop**: a one- or two-storey house with an open lean-to on its east end over the trade's stations (smith: anvil, grindstone, blast furnace; carpenter, mason, potter likewise), stores filled with the trade's goods.
- **windmill**: three storeys, cobble below and spruce above, a hipped roof, an axle and four sails of wool on spars, kept above head height.
- **mine**: a portal in a hill face with an adit and rails, an incline down to a chamber under timber sets, two side galleries ending at ore faces, and on most seeds a second flight to a deeper working with amethyst. Passages are two blocks wider than the profile's lane and one block taller than its door; bodies that can't climb stairs get half-block steps instead. Carved blocks are `cave_air` and a skin of rock and ore surrounds the carving. `mine_walk` walks every floor cell from the portal; the tests hold it at all of them for blocky, giant and human.

## Apartments against faces

A street front that has a tall face of ground behind it (rising two storeys or more within five to ten blocks) asks for an **apartment**: one bay deep, one or three wide, three to five storeys, in stone, brick, timber or oak. The town places apartments in a first pass before the other kinds, so the terrace above doesn't take the cells behind them, and lets them cut up to three storeys into the face. After placing one, `back_door` looks along the back wall from the top storey down for the first storey whose floor meets the ground above (within two blocks, or a street or plaza at that level), cuts a door there, clears any shutter or flower box from the landing, lays a path block on it and walls up any back window left buried in rock. On the valley-wall test ground, seeds 1 and 2 place 14 apartments each, 12 and 9 of them with back doors, and every floor of every building is walked; the tests hold a valley-wall town at five or more apartments and three or more back doors.

## Street life

The square starts at the castle's outer face, 21 by 15 and fully paved. It has a fountain in the middle (a basin of water round a carved column with water on top, lanterns on its corners) with a bench on each side, eight market stalls on fence posts under striped wool canopies, each with a counter, wares, a crate of goods and a lantern hanging inside, crates and sacks between them, and lamp posts on its corners and far edge. The square's dressing draws on its own random stream so changing it never moves the town plan. `dress_streets` then walks each street and every fifth cell puts one prop on the free ground beside it, never on the street or within a block of a door's landing and only where the ground is level with the pavement: a bench facing the street, stacked barrels, hay, or a flower tub. It also sinks up to three wells on level patches beside the streets. Street lamps go every five blocks. On the valley wall, seed 1: 107 props (36 benches), 168 lamps, a well; every house floor and the mine are still walked.

## Roofs

Houses take one of three roofs: the gable along the house, a hipped roof, or a **cross gable**: a second, narrower gable over the door bay running back from the front wall to the main ridge, with a window in its gable. The cross gable is laid over the main roof, not cut into it, and only its parts on or above the main roof's slope are kept, so no pocket of attic opens under it and the walk check is unchanged. Town houses cycle through gable, cross and hip by seed; apartments pick one of the three.

## World saves and path-traced pictures

`blockworld.to_world(build, path, dy=0, ground=True, fill=None)` writes a world folder: region files in the 1.18+ section format (palette and packed longs), block entities in the chunk, entities in their own region files and a gzipped `level.dat` for a flat creative world, data version 3955 (1.21.1). `fill="stone"` packs the ground under a build's footprint so a mine sits in rock rather than over the void. The tests read a house's world back block by block and check chests keep their items.

These saves load in Chunky (open source, built from source here) and render with a freely licensed texture pack (Pixel Perfection Community Edition). That pack stops at 1.16, so the render copy adds stand-in textures for the newer blocks block mode uses (deepslate tiles, calcite, tuff, mud bricks, azalea leaves and a few ores), recoloured from the pack's own stone, brick and leaf textures. Chunky showed nothing below y 0 from these saves (cause not found yet), so builds that dig below the ground are written with `dy=64`.

## Decoration and the walk check

Decoration the player can't reach is allowed: a storage loft under a gable roof, the tops of tall shelves, a windmill's sails. The walk check is on floors and routes: every floor cell the furniture leaves free, on every storey of every building, and every mine floor cell, must be reached by the profile's body. A house's interior share as a whole may fall to 0.6 because lofts and furniture tops count in it.

Set dressing that Minecraft keeps as block entities and entities (`blockdress`): every chest and barrel gets two to five stacks of its building's goods in random slots; every banner two or three patterns; walls get item frames (fixed, holding a theme item) and one-block paintings, never two side by side; a mine gets a minecart on its rails and a loaded chest cart at a gallery end. None of it touches a block or floor, so the walk check is the same with or without it. The format is Java 1.21's (`count`, `patterns:[{pattern,color}]`, painting `variant`); it has not been loaded in game yet.

## Not done yet

- Other architects: the castle, village, house, tower, workshop, windmill and mine have block-mode versions; the rest don't yet.
- The export's entities and block entity data have not been loaded in a running game; Chunky loads the blocks.
- Rival pictures use shader packs and weather; ours use Chunky's plain sun and sky.
- Stairs for bodies that step less than 0.5 m.
