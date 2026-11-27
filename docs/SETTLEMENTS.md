# Settlements

A settlement is a street and plot plan that calls the architects Blocks already has. It does not generate building meshes of its own. Each plot becomes one `BuildDocument` from the normal authoring path, and that document is placed into the town.

Two products are registered: `english-village` and `roman-town`. Both take the site `width` and `depth` in metres, plus `seed`, `detail`, `variant` and `age`. `detail` is passed through to every building. `height`, `bays`, `levels`, `style` and `openings` are ignored, because those controls belong to a single building. `plot_overrides`, keyed by plot id, can replace one plot's product, size, variant, seed or parameters. `manor` is a boolean and applies only to `green-village`.

The same request always produces the same town. A plot seed is the first four bytes of SHA-256 of `{settlement seed}:{plot id}`, masked to 31 bits. The street plan does not depend on the finished size of a building, so changing one plot leaves the other plots where they were. An identical child request is reused from a cache, so a repeated house or stoa is not authored twice. The cache does not change ids: each placed member is still prefixed with its plot id.

## Medieval English village

`english-village` is a nucleated village of the sort that stood in England between the 12th and the 15th century. The plan is a continuous frontage, not a few houses on an empty green.

Houses stand at the street end of a burgage toft, facing the house opposite. Neighbours are 0.5 to 2 metres apart. There is no back lane; the gap between eaves is the only break in the row. The plot seed picks one of two frames. A one-storey house is 8 by 8 by 7 metres on a 10.4 metre frontage. About one plot in three is two-storey, 9 by 9 by 11 metres on an 11.7 metre frontage. Variant (`longhouse`, `gable-front`, `twin-gable`), finish (`chalk-slate`, `honey-clay`, `rose-shale`), porch and chimney also change per plot, so a row does not repeat. A 6 metre frontage will not hold the timber house this planner can call: the frame is at least 8 metres wide and the eaves grow a little past that. Tofts run back 30 to 60 metres. The yard is `earth-fill` and fills the plot around the building, out to the boundary. It is not a pad under the house.

The boundary is a post-and-rail fence in `oak`: posts about 1.1 metres high and a rail between them, set just inside the plot. A shared edge is drawn once. The street front is open. The substance registry has no wattle id, and `oak` is already one of the timber substances, so the fence does not add a material.

Lanes are 4 metres of `earth-fill`. There are no kerbs. Stone paving is reserved for the market place. The registry has no gravel or beaten-earth id, so earth lanes, yards and the green use `earth-fill`, and stone uses `limestone`.

The church is a `gothic-cathedral` at 12 by 15 by 20 metres, Early English, four nave bays, two choir bays, five apse sides, and no spire. That is the smallest integer size the cathedral's own ratio rule accepts. A Norman keep reads as a tower, and there is no parish-church architect to call. The cathedral's entrance is its west front, which is the local −Z side, so the plot is turned until that front faces the street or the green. The churchyard is `earth-fill`, enclosed by a low `limestone` wall with a gateway on the side that faces the public space.

The manor, where it is built, is a `norman-keep` of 12 by 16 by 14 metres with `variation` 0, so the seed does not move its stones. A shell keep's mound is too large for the edge of a village.

Variants, and the smallest site each one will build:

- `street-village` (the default), 100 by 72 metres. One lane through the middle. Tofts face each other across it, at least ten houses. A green closes the south-east end of the lane. The church closes the north-east, in its churchyard, facing south onto the lane.
- `green-village`, 120 by 96 metres. Lanes on the west, south and north. At least ten houses ring a green and face onto it. The church occupies the east end of the north side, facing south. A small manor keep stands at the north-east, facing the north lane. `manor` defaults to true; `manor: false` keeps that plot as an empty yard. Extra depth lengthens the tofts up to 60 metres and leaves a field beyond the north lane. It does not stretch the green's short side.
- `market-town`, 120 by 68 metres. At least fourteen houses line both sides of the lane. A limestone market place sits on the north side, with a timber hall immediately west of it, turned to face the market and pulled up to the lane, and the churchyard to the east of the market. There is no market-cross architect, so the landmark is the hall.

## Roman town quarter

`roman-town` is one quarter of a provincial town of the 1st or 2nd century AD, not a whole city. An insula is built out to its streets.

The plan is an orthogonal grid. The cardo and the decumanus maximus are 8 metres wide, inside the 6 to 10 metre range of a main street, paved in `limestone`, with kerbs, raised pavements (crepidines) and stepping stones. The pavement is omitted where the street is too narrow to keep a carriageway, so side streets stay 4.5 metres, inside the 4 to 6 metre range, and have no kerb. Kerbs and pavements break at junctions. Insulae are 35 to 75 metres on a side.

Where an insula fronts the cardo or the decumanus, that edge is a `classical-stoa`, turned so the colonnade faces the street. The catalog has no shop-row or taberna form. The stoa is the narrow street-front block, used along those two streets and along the forum. Behind the stoae, `classical-villa` courtyard houses sit back to back, each turned to face a street. A corner insula that touches neither main street is four domus, one to a corner. One insula carries the baths. Another carries a short aqueduct along a side street, facing the street rather than standing in it. Orders are Doric or Tuscan, except the baths, arch and aqueduct, which the masonry forms require to be `auto`. The classical forms accept no extra parameters, so a plot cannot ask a villa for a shop range.

The forum is the insula at the crossing. The square is open `limestone` paving. A stoa faces it from the south, another from the east, and a basilica faces it from the west. At the north end of the court a portico stands in for the temple, raised on a limestone podium, with a triumphal arch beside it. Both face the square. The arch and the aqueduct are inside an insula. Neither spans a carriageway: a building whose footprint crosses a street fails the overlap rule, and neither form is a bridge in this catalog.

Whatever the buildings do not cover becomes courtyard and garden, so the insula has no hole. Streets, pavements and the forum paving cover the rest of the site.

Variants:

- `forum-quarter` (the default), 90 by 90 metres. Two by two insulae: the forum plus three built-out insulae.
- `residential-quarter`, 90 by 90 metres. The same grid with no forum. Four insulae, each with at least three buildings.
- `colonia-grid`, 128 by 128 metres. Three by three insulae, with a forum at the crossing. Four by four when both sides can hold an insula of at least 35 metres at that count (about 166 metres).

## How a plot calls an architect

The planner writes a `BuildRequest` per plot: product, variant, finish, size and seed. Timber houses roll those choices from the plot seed. Civic buildings (the church, the keep, and the classical forms) are pinned, and their geometry does not use the seed.

`author_document` runs the same routing as `build`, and returns the child document before any mesh is compiled. The settlement is compiled once, as one document.

Placement is a quarter turn about Y: 0, 90, 180 or 270 degrees. South is −Z. Positions rotate, width and depth swap on 90 and 270, a sculpted mesh's normalized vertices rotate with the member, an ashlar `axis` of `x` or `z` swaps with them, and a wedge's `+X +Z -X -Z` orientation cycles. Every placed member stays axis-aligned. Ids are prefixed with the plot id, including ashlar host ids, so repeated buildings do not merge their bonds. The original parent links are kept, rewritten to those prefixed ids. A member that had no parent is parented to the plot yard. Street slabs, kerbs, pavements and stepping stones parent to the street. A square parents to the street it opens from. Fences and the churchyard wall parent to their plot.

Each plot and each street is real geometry with a substance, not an empty group. The ground of a plot is the yard and garden left after the building's plan is cut out. The cut follows the members, not their bounding box, so the empty corners of a cruciform church or a porch stay ground. The largest piece is the plot root. Both the building and the yard sit on the ground. The temple is the exception in height: its anchor is raised by the podium, so the child document still touches Y=0 when it is checked in its own frame, while the world building stands on the podium.

`age` weathers the finished town. Children are authored at age 0, so the age does not change the fingerprint of the geometry.

The physics manifest and the glTF node extras carry `plot_id`, `building_product` and `street_id` for every member that has them. Streets carry a null plot and product. The compiler records the settlement as the document's architect; the plot and the product on each member are what still name the building that was called.

## Quality gates

Reported through the normal quality report:

- `settlement.overlap`. No two buildings overlap in plan, and no building crosses a street or a square. Edges that merely touch are allowed. Yards, gardens, kerbs, pavements, fences and the churchyard wall are not buildings.
- `settlement.frontage`. Every called architect puts its entrance on the local −Z side: the timber door, the cathedral west front, the keep gate, the basilica's public end opposite the apse, the stoa and portico colonnade, and the bath, arch and aqueduct entrances. After the quarter turn, a point 0.9 metres out from the middle of that face must land in a street or a square.
- `settlement.street_network`. The street slabs form one connected network, and that network reaches the site edge.
- `settlement.building_quality`. Each building is turned back into the frame it was authored in and run through its own architect's checks, including ashlar coursing. The world document is also checked, so a rotated wall still has to keep a running bond.
- `settlement.ground_coverage`. Ground members plus building footprints cover at least 99.5 percent of the site. Ground is street, kerb, pavement, yard, garden, green, churchyard, market, forum paving or the temple podium. A missing yard fails.
- `settlement.street_frontage`. Along each side of the main street, the green or the market, built frontage is at least 65 percent of that side. The green, the churchyard and the market place are left out of the length, and so are the crossings where the cardo and the decumanus meet. A side with nothing to measure does not fail.
- `settlement.insula_coverage`. On each residential insula, building footprints cover at least 45 percent of the insula. The forum court is not a residential insula. An English village has none, and the check passes.

Every member has a substance. The physical report covers the whole town: total mass, and the bill of materials by substance.

## Limits

- The planner will not build a variant below its default site. The shared catalog bounds are wider than that, 90 to 200 metres by 68 to 200, because the two products share one rule. A street village is 100 by 72 metres, a green village 120 by 96, a market town 120 by 68, a forum quarter and a residential quarter 90 by 90, and a colonia 128 by 128. Larger sites stay legal up to 200 metres. They lengthen the rows or, for a colonia, add insulae once a side can hold them.
- The church is a small cathedral, not a parish church. Its west front and its cruciform plan are doing the job of a village church because that is the closest architect in the catalog.
- The manor is a small keep, and only on the green village. A shell keep does not fit the edge of the site.
- There is no market cross. The market landmark is a timber hall.
- There is no taberna or shop-row form. A `classical-stoa` is the street-front colonnade on the cardo, the decumanus and the forum.
- An arch or an aqueduct cannot span a street under the overlap rule. Both stand inside an insula.
- Burgage frontages are 10.4 or 11.7 metres, not 6, because of the timber house's minimum width and its eaves.
- Lanes, yards, gardens and the green are `earth-fill`. Paving, kerbs, pavements, stepping stones and the churchyard wall are `limestone`. Fences are `oak`.
- Classical houses are the villa, stoa, basilica, portico, baths, arch and aqueduct forms that already exist. A plot cannot pass them extra parameters, and it does not call `classical-forum` or the Greek temple generator.
- Height, bay count and level count on the settlement request do not change the town. Set them on a `plot_overrides` entry if one building needs them.
