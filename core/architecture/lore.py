"""Construction knowledge shared by every block-mode architect.

This is the world knowledge the primary architects (house, tower, hall,
workshop, town) consult, and the rulebook the checker in ``blockrules``
scores against. It is data, not geometry: structural systems, roof coverings,
composition rules, building purposes, town rules, and the building
traditions that combine them by region and era.

A style sub-architect is a (tradition, purpose) pair. It never invents its own
structure: it picks a system, a covering and ranges from here, and the primary
architect builds inside them.

Units are metres, and one block is one metre. Every rule carries ``basis``:
where its numbers come from. ``read`` means a page that was read while writing
this module, named in docs/ARCHITECTURE_RULES.md. ``practice`` means common
building practice that has not been checked against a source yet; treat those
numbers as first guesses to be replaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Range = Tuple[float, float]


# ------------------------------------------------------- structural systems
# How a building stands up. ``cantilever`` is the furthest a member may reach
# past the support below it without a bracket, corbel or arch, in blocks.
SYSTEMS: Dict[str, Dict[str, Any]] = {
    "rubble-masonry": {
        "name": "Load-bearing rubble stone", "cantilever": 1, "storeys": (1, 3), "opening_max": 1.5,
        "pier_min": 1.0, "spans": "lintel or arch over every opening",
        "rules": ["Walls stand on walls: nothing heavy over a void without an arch or lintel",
                  "Corners are solid and dressed (quoins); openings keep away from corners",
                  "Small openings; piers between them at least as wide as half the opening"],
        "basis": "practice"},
    "ashlar": {
        "name": "Dressed stone (ashlar)", "cantilever": 1, "storeys": (1, 5), "opening_max": 2.0,
        "pier_min": 1.0, "spans": "lintel up to 2 m, arch beyond",
        "rules": ["Coursed in running bond", "String courses mark floor lines",
                  "Wider openings are arched"],
        "basis": "practice"},
    "brick": {
        "name": "Load-bearing brick", "cantilever": 1, "storeys": (1, 5), "opening_max": 1.8,
        "pier_min": 1.0, "spans": "arch or lintel; corbelled eaves and gables",
        "rules": ["Openings stack in bays", "Arches or lintels over openings",
                  "Gables may be stepped or shaped, rising above the roof as parapets"],
        "basis": "practice"},
    "box-frame": {
        "name": "Timber box frame (half-timbered)", "cantilever": 1, "storeys": (1, 4), "opening_max": 1.5,
        "pier_min": 0.0, "bay": (3.0, 5.0), "jetty": 1,
        "spans": "posts at every bay line, plates at every floor",
        "rules": ["Posts at every bay line and every corner, plates along each floor",
                  "Openings sit between posts, below a plate",
                  "Upper storeys may project (jetty) on the joists below, one block per storey at most",
                  "Infill panels are lighter than the frame"],
        "basis": "read: jettying pages (no projection figure); one block per storey is practice"},
    "cruck": {
        "name": "Cruck frame", "cantilever": 1, "storeys": (1, 2), "opening_max": 1.2, "pier_min": 0.0,
        "span": (4.0, 6.0), "spans": "paired curved blades from ground to ridge",
        "rules": ["One storey and a loft", "Low eaves, steep roof", "Span no more than about 6 m"],
        "basis": "practice"},
    "log": {
        "name": "Log construction (blockbau)", "cantilever": 2, "storeys": (1, 2), "opening_max": 1.2,
        "pier_min": 1.0, "spans": "horizontal logs notched at the corners, ends left proud",
        "rules": ["Logs cross at the corners and stick out past them", "Small openings, cut after raising",
                  "Often on a stone base above the snow or damp"],
        "basis": "practice"},
    "adobe": {
        "name": "Mud brick and earth", "cantilever": 1, "storeys": (1, 4), "opening_max": 1.0,
        "pier_min": 1.0, "spans": "timber beams (vigas) through the walls, flat earth roof",
        "rules": ["Thick walls, few and small openings", "Flat roofs behind a parapet, beams showing at the wall head",
                  "Upper storeys step back to make terraces"],
        "basis": "practice"},
    "post-and-beam": {
        "name": "East Asian post and beam", "cantilever": 2, "storeys": (1, 3), "opening_max": 4.0,
        "pier_min": 0.0, "spans": "columns carry the roof; walls are screens",
        "rules": ["Raised floor on a stone base", "Deep eaves (at least 1 m) on bracket sets",
                  "Walls between columns are light screens and may be mostly open"],
        "basis": "practice"},
}


# ----------------------------------------------------------- roof coverings
# ``pitch`` is the usual range in degrees for that covering on historic
# buildings. Block roofs come in fixed steps: stairs on stairs is 45, stairs
# every row with a slab between is about 27, two stairs per step is about 63,
# and a flat roof is 0.
COVERINGS: Dict[str, Dict[str, Any]] = {
    "thatch": {"pitch": (45, 60), "eave": (0.5, 1.0),
               "basis": "read: North Norfolk design guide (medieval thatch 50 or over); 45 lower bound is practice"},
    "plain-tile": {"pitch": (35, 60), "eave": (0.3, 0.8),
                   "basis": "read: Marley product minimums (clay plain types 35 to 40); upper bound is practice"},
    "slate": {"pitch": (25, 50), "eave": (0.3, 0.6),
              "basis": "read: North Norfolk design guide (slate as low as 25); upper bound is practice"},
    "stone-slate": {"pitch": (30, 55), "eave": (0.2, 0.5), "basis": "practice"},
    "pantile": {"pitch": (30, 45), "eave": (0.2, 0.5),
                "basis": "read: North Norfolk design guide (35 to 45 normal, not below 30)"},
    "roman-tile": {"pitch": (15, 30), "eave": (0.3, 0.8),
                   "basis": "read: Marley Double Roman minimum 22.5; Mediterranean range is practice"},
    "shingle": {"pitch": (25, 55), "eave": (0.5, 1.5),
                "basis": "read: Marley cedar shingle minimum 14 (modern); historic range is practice"},
    "turf": {"pitch": (20, 35), "eave": (0.3, 0.8), "basis": "practice"},
    "flat-earth": {"pitch": (0, 5), "eave": (0.0, 0.3), "parapet": True, "basis": "practice"},
    "east-asian-tile": {"pitch": (25, 45), "eave": (1.0, 2.5), "curved": True, "basis": "practice"},
    "metal": {"pitch": (0, 65), "eave": (0.0, 0.5), "basis": "practice"},
}

# Which covering a Minecraft roof block reads as. Checked by prefix, longest
# first, so ``dark_oak`` wins over ``oak``.
BLOCK_COVERING: Dict[str, str] = {
    "hay_block": "thatch", "dried_kelp_block": "thatch", "deepslate_tile": "slate", "deepslate_brick": "slate",
    "polished_deepslate": "slate", "cobbled_deepslate": "stone-slate", "blackstone": "slate",
    "polished_blackstone": "slate", "stone_brick": "stone-slate", "cobblestone": "stone-slate",
    "mossy_cobblestone": "turf", "moss_block": "turf", "grass_block": "turf",
    "brick": "plain-tile", "bricks": "plain-tile", "mud_brick": "plain-tile", "granite": "pantile",
    "polished_granite": "pantile", "terracotta": "roman-tile", "orange_terracotta": "roman-tile",
    "red_terracotta": "roman-tile", "smooth_sandstone": "flat-earth", "sandstone": "flat-earth",
    "cut_sandstone": "flat-earth", "packed_mud": "flat-earth", "smooth_quartz": "flat-earth",
    "spruce": "shingle", "dark_oak": "shingle", "oak": "shingle", "birch": "shingle", "jungle": "shingle",
    "mangrove": "shingle", "acacia": "shingle", "cherry": "shingle", "crimson": "shingle", "warped": "shingle",
    "copper": "metal", "cut_copper": "metal", "exposed_cut_copper": "metal", "weathered_cut_copper": "metal",
    "oxidized_cut_copper": "metal", "waxed_cut_copper": "metal",
}


def covering_of(block_name: str) -> Optional[str]:
    """The roof covering a block name reads as, or None."""
    name = block_name.split(":")[-1].split("[")[0]
    for suffix in ("_stairs", "_slab", "_planks", "_wall"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for key in sorted(BLOCK_COVERING, key=len, reverse=True):
        if name == key or name.startswith(key + "_") or name.endswith("_" + key) or name == key + "s":
            return BLOCK_COVERING[key]
    return None


# ------------------------------------------------------------- composition
# Facade rules every building is scored on. Ranges a tradition overrides are
# named in its ``composition``.
COMPOSITION: Dict[str, Dict[str, Any]] = {
    "stacked-openings": {"text": "Windows and doors line up from storey to storey in bays",
                         "min": 0.6, "basis": "practice"},
    "storey-height": {"text": "A storey is 3 to 5 blocks floor to floor; the ground storey is not lower than "
                              "the ones above", "range": (3, 5), "basis": "practice"},
    "base-middle-top": {"text": "A building reads as a base (plinth or ground storey in another material), "
                                "a middle, and a top (eave line, cornice or roof overhang)", "basis": "practice"},
    "window-ratio": {"text": "Glass and openings as a share of wall area", "range": (0.08, 0.30),
                     "basis": "practice"},
    "facade-depth": {"text": "No flat walls: a facade has at least two depth planes (frame, jetty, sill, "
                             "string course, buttress, porch, bay)", "planes": 2,
                     "flat_run": 12, "basis": "practice"},
    "materials": {"text": "Two to five material families on a facade: one reads as flat, more as noise",
                  "range": (2, 5), "basis": "practice"},
    "roof-covers": {"text": "The roof oversails the walls (eave and verge) unless a parapet closes it",
                    "basis": "practice"},
    "roof-pitch": {"text": "Roof pitch suits its covering", "basis": "see COVERINGS"},
    "chimney": {"text": "A domestic building in a cold or temperate climate has a chimney that clears the "
                        "ridge", "basis": "practice"},
    "load-path": {"text": "Every block rests on the one below it, or reaches no further than the system's "
                          "cantilever from support", "basis": "see SYSTEMS"},
}


# ---------------------------------------------------------------- purposes
# What a building is for, and how that shapes its plan. Sizes are footprint
# (frontage, depth) and storeys, in blocks.
PURPOSES: Dict[str, Dict[str, Any]] = {
    "cottage": {"footprint": ((5, 8), (7, 10)), "storeys": (1, 2), "rooms": "one or two rooms and a loft",
                "door": "long side or gable", "chimney": True},
    "longhouse": {"footprint": ((5, 7), (14, 30)), "storeys": (1, 1), "rooms": "people and animals under one roof",
                  "door": "long side, cross passage", "chimney": False, "hearth": "central"},
    "hall-house": {"footprint": ((6, 9), (12, 16)), "storeys": (1, 2), "rooms": "open hall between two-storey end bays",
                   "door": "long side, into a cross passage", "chimney": True},
    "row-house": {"footprint": ((4, 7), (9, 15)), "storeys": (2, 4), "rooms": "one room per floor front, one back",
                  "door": "street front", "party_walls": True, "chimney": True},
    "merchant-house": {"footprint": ((5, 8), (10, 16)), "storeys": (3, 4), "rooms": "shop on the street, hall and "
                       "chambers above, store behind", "door": "street front", "shopfront": True, "chimney": True},
    "tower-house": {"footprint": ((7, 12), (7, 12)), "storeys": (3, 5), "rooms": "one room per floor",
                    "door": "raised or on the ground behind a yett", "small_low_openings": True, "chimney": True},
    "courtyard-house": {"footprint": ((12, 20), (12, 20)), "storeys": (1, 2), "rooms": "rooms around a court",
                        "door": "one entrance to the court", "blank_outer_walls": True},
    "barn": {"footprint": ((7, 12), (12, 30)), "storeys": (1, 1), "rooms": "threshing floor between bays",
             "door": "wagon doors in the long sides", "window_ratio": (0.0, 0.05), "chimney": False},
    "smithy": {"footprint": ((5, 8), (6, 10)), "storeys": (1, 1), "rooms": "forge open to the street",
               "door": "wide front", "chimney": True},
    "inn": {"footprint": ((8, 14), (10, 16)), "storeys": (2, 3), "rooms": "taproom, kitchen, chambers above, "
            "yard behind", "door": "street front and carriage arch", "chimney": True},
    "chapel": {"footprint": ((6, 9), (12, 20)), "storeys": (1, 1), "rooms": "nave and chancel",
               "door": "west end or south side", "tall_windows": True},
    "market-hall": {"footprint": ((6, 10), (10, 18)), "storeys": (2, 2), "rooms": "open arcade, room above",
                    "door": "open ground floor", "open_ground": True},
    "shop": {"footprint": ((4, 8), (4, 7)), "storeys": (1, 2), "rooms": "one lock-up shop, a loft under the roof",
             "door": "street front, a shop window beside it", "chimney": False,
             "basis": "selds: the small lock-up shops that lined medieval market streets, often lean-tos "
                      "against a wall or bank; practice"},
    "granary": {"footprint": ((3, 5), (4, 6)), "storeys": (1, 1), "rooms": "one store on staddle stones",
                "door": "one small door up steps", "raised": True, "chimney": False},
    "mill": {"footprint": ((6, 9), (6, 9)), "storeys": (2, 4), "rooms": "stones, bins, sack floor",
             "door": "ground", "window_ratio": (0.02, 0.12), "chimney": False},
}


# ------------------------------------------------------------------- towns
TOWN: Dict[str, Dict[str, Any]] = {
    "street-hierarchy": {"text": "Main streets 6 to 10 m, lanes 3 to 5 m, alleys 1.5 to 2.5 m",
                         "main": (6, 10), "lane": (3, 5), "alley": (1.5, 2.5), "basis": "practice"},
    "street-enclosure": {"text": "Street width to fronting building height: 1:1 to 3:1 in modern guides; "
                                 "medieval lanes ran narrower, about 0.3:1 to 1.5:1",
                         "range": (1.0, 3.0), "medieval": (0.3, 1.5),
                         "basis": "read: SIF Hg enclosure essay (3:1 street, citing Southampton guide; "
                                  "it notes no scientific basis)"},
    "square-enclosure": {"text": "Square width to surrounding eave height between 1:1 and 6:1, best near 2:1",
                         "range": (1.0, 6.0), "best": (1.0, 2.0),
                         "basis": "read: SIF Hg essay (6:1 squares; Sitte's 1:1 to 2:1 via Kim and Kim)"},
    "square-size": {"text": "A square's long side under about 90 m unless divided; its short side under about "
                            "45 m, so faces can be recognised across it", "long_max": 90, "short_max": 45,
                    "basis": "read: Livable Cities squares part 6 (300 ft, 150 ft)"},
    "frontage": {"text": "Main streets are lined: at least 70 percent of each street edge has a building front "
                         "within 2 m", "min": 0.7, "basis": "practice"},
    "plots": {"text": "Town plots are narrow and deep (frontage 4 to 8 m) and a street keeps one rule for "
                      "gable or eave to the street", "frontage": (4, 8), "basis": "practice"},
    "landmarks": {"text": "Streets end on something: a gate, a church, a hall, a tower", "basis": "practice"},
}


# --------------------------------------------------------------- traditions
@dataclass
class Tradition:
    """One building tradition: a region and an era, its systems, coverings,
    palette and composition ranges, and the purposes it has types for."""
    key: str
    name: str
    region: str
    era: str
    systems: List[str]
    coverings: List[str]
    purposes: List[str]
    palette: Dict[str, Any]
    composition: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    climate: str = "temperate"
    block_mode: bool = False          # a block-mode sub-architect exists
    mesh_mode: Optional[str] = None   # the mesh architect that already builds it

    def window_ratio(self) -> Range:
        return tuple(self.composition.get("window-ratio", COMPOSITION["window-ratio"]["range"]))  # type: ignore


TRADITIONS: Dict[str, Tradition] = {t.key: t for t in [
    Tradition("english-medieval", "English medieval vernacular", "England", "1200-1500",
              ["box-frame", "cruck", "rubble-masonry"], ["thatch", "plain-tile", "stone-slate", "shingle"],
              ["cottage", "hall-house", "longhouse", "barn", "inn", "chapel", "smithy", "market-hall", "row-house",
               "shop"],
              {"frame": "dark_oak_log", "infill": "calcite", "base": "cobblestone", "roof": "dark_oak"},
              {"window-ratio": (0.05, 0.18)}, ["jetties on street fronts", "close studding", "cross wings"],
              block_mode=True, mesh_mode="timberwright"),
    Tradition("tudor", "Tudor town house", "England", "1485-1603", ["box-frame", "brick"], ["plain-tile"],
              ["merchant-house", "row-house", "inn", "market-hall"],
              {"frame": "dark_oak_log", "infill": "calcite", "base": "bricks", "roof": "deepslate_tile"},
              {"window-ratio": (0.12, 0.30)},
              ["one jetty per storey on the street", "close studding and herringbone brick", "oriel windows",
               "gables to the street", "tall clustered brick chimneys"]),
    Tradition("norman-romanesque", "Norman stone", "England and Normandy", "1066-1200",
              ["rubble-masonry", "ashlar"], ["plain-tile", "stone-slate"], ["tower-house", "chapel", "hall-house"],
              {"wall": "stone_bricks", "dressing": "chiseled_stone_bricks", "roof": "deepslate_tile"},
              {"window-ratio": (0.03, 0.10)}, ["round arches", "pilaster buttresses", "small high windows"],
              block_mode=True, mesh_mode="castle"),
    Tradition("hanseatic", "Hanseatic brick", "North Germany and the Baltic", "1300-1600", ["brick"],
              ["plain-tile", "pantile"], ["merchant-house", "row-house", "chapel", "market-hall"],
              {"wall": "bricks", "trim": "mud_bricks", "roof": "deepslate_tile"},
              {"window-ratio": (0.15, 0.35)},
              ["stepped gables to the street", "blind niches in the gable", "hoist beam and loading doors",
               "pointed arches"]),
    Tradition("dutch-golden-age", "Dutch canal house", "Netherlands", "1600-1750", ["brick"], ["plain-tile"],
              ["row-house", "merchant-house"], {"wall": "bricks", "trim": "smooth_quartz", "roof": "deepslate_tile"},
              {"window-ratio": (0.25, 0.45)},
              ["narrow and tall, 3 to 5 storeys", "spout, step, neck or bell gable", "hoist hook at the apex",
               "front leans out slightly"]),
    Tradition("norse", "Norse longhouse and stave", "Scandinavia", "800-1100", ["log", "post-and-beam"],
              ["turf", "shingle"], ["longhouse", "hall-house", "barn", "chapel"],
              {"wall": "spruce_log", "roof": "moss_block", "base": "cobblestone"},
              {"window-ratio": (0.0, 0.05)},
              ["long curved walls", "turf or shingle roof to near the ground", "carved gable finials",
               "central hearth with a smoke hole, no chimney"], climate="cold"),
    Tradition("alpine", "Alpine chalet", "Alps", "1500-1900", ["log", "rubble-masonry"], ["shingle", "stone-slate"],
              ["cottage", "barn", "inn"], {"wall": "spruce_log", "base": "stone_bricks", "roof": "spruce"},
              {"window-ratio": (0.10, 0.25)},
              ["stone ground storey, log above", "low wide gable to the valley", "eaves 1.5 m or more",
               "balconies along the upper storeys", "stones holding the shingles"], climate="cold"),
    Tradition("mediterranean", "Mediterranean farmhouse and courtyard", "Italy, Spain, Provence", "1300-1800",
              ["rubble-masonry"], ["roman-tile"], ["courtyard-house", "cottage", "barn", "chapel"],
              {"wall": "smooth_sandstone", "base": "stone", "roof": "terracotta"},
              {"window-ratio": (0.05, 0.15)},
              ["low roofs of curved tile", "rendered walls in warm colours", "shutters", "loggias and pergolas",
               "outside stairs"], climate="warm"),
    Tradition("aegean", "Aegean island", "Greek islands", "1700-1900", ["rubble-masonry"], ["flat-earth"],
              ["cottage", "chapel"], {"wall": "white_concrete", "trim": "light_blue_terracotta", "roof": "smooth_quartz"},
              {"window-ratio": (0.05, 0.15)}, ["whitewashed cubes", "flat roofs", "domed chapels", "outside stairs"],
              climate="warm"),
    Tradition("pueblo", "Pueblo adobe", "American Southwest", "900-1900", ["adobe"], ["flat-earth"],
              ["courtyard-house", "row-house", "chapel"],
              {"wall": "packed_mud", "beam": "stripped_spruce_log", "roof": "packed_mud"},
              {"window-ratio": (0.02, 0.10)},
              ["terraced storeys stepping back", "ladders between roofs", "beams through the wall head",
               "rounded parapets"], climate="dry"),
    Tradition("persian", "Persian courtyard", "Iran", "1500-1800", ["brick", "adobe"], ["flat-earth"],
              ["courtyard-house"], {"wall": "sandstone", "trim": "cyan_glazed_terracotta"},
              {"window-ratio": (0.10, 0.30)}, ["iwans on the court", "wind catchers", "domes"],
              climate="dry", mesh_mode="persian"),
    Tradition("japanese-minka", "Japanese minka and machiya", "Japan", "1600-1900", ["post-and-beam"],
              ["east-asian-tile", "thatch"], ["cottage", "row-house", "merchant-house"],
              {"frame": "dark_oak_log", "screen": "white_concrete", "roof": "deepslate_tile"},
              {"window-ratio": (0.30, 0.70)},
              ["deep eaves", "lattice screens", "raised floor", "hip-and-gable roof"], mesh_mode="pagoda"),
    Tradition("georgian", "Georgian", "Britain and colonies", "1714-1830", ["brick", "ashlar"],
              ["slate", "plain-tile"], ["row-house", "merchant-house", "inn"],
              {"wall": "bricks", "trim": "smooth_quartz", "roof": "deepslate_tile"},
              {"window-ratio": (0.18, 0.32)},
              ["strict symmetry", "sash windows in vertical stacks, tallest on the first floor",
               "parapet hiding the roof", "pedimented door"]),
    Tradition("victorian", "Victorian", "Britain", "1837-1901", ["brick"], ["slate"],
              ["row-house", "merchant-house"], {"wall": "bricks", "roof": "deepslate_tile"},
              {"window-ratio": (0.20, 0.35)}, ["bay windows", "polychrome brick"], mesh_mode="victorian"),
    Tradition("greco-roman", "Greek and Roman", "Mediterranean", "-600-400", ["ashlar"], ["roman-tile"],
              ["chapel", "market-hall", "courtyard-house"], {"wall": "smooth_quartz", "roof": "terracotta"},
              mesh_mode="classical", climate="warm"),
    Tradition("gothic", "Gothic", "Western Europe", "1150-1500", ["ashlar"], ["slate", "metal"],
              ["chapel"], {"wall": "stone_bricks"}, mesh_mode="cathedral"),
    Tradition("egyptian", "Egyptian", "Egypt", "-2600--30", ["ashlar"], ["flat-earth"], ["chapel"],
              {"wall": "sandstone"}, mesh_mode="egyptian", climate="dry"),
    Tradition("mesoamerican", "Mesoamerican", "Mexico and Central America", "-200-1500", ["ashlar"],
              ["flat-earth"], ["chapel"], {"wall": "mossy_stone_bricks"}, mesh_mode="mesoamerican", climate="warm"),
]}


def tradition(key: str) -> Tradition:
    return TRADITIONS[key]


def coverage() -> Dict[str, Any]:
    """Which traditions each mode builds, and the purposes each still lacks."""
    out = {"block_mode": [], "mesh_only": [], "missing": []}
    for t in TRADITIONS.values():
        if t.block_mode:
            out["block_mode"].append(t.key)
        elif t.mesh_mode:
            out["mesh_only"].append(t.key)
        else:
            out["missing"].append(t.key)
    return out
