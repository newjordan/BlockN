"""Inspectable craft traditions and measured checks for classical column work."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import numpy as np


TRADITIONS = {
    "timber-framing": {
        "name": "Half-timbered house framing", "orders": [],
        "system": "Braced post-and-beam bays with limewashed infill",
        "rules": ["Openings sit between posts below continuous plates", "Gables close the roof ends",
                  "Upper floors project on joists and leave a stairwell"],
        "load_path": ["pitched roof", "top plate", "braced posts", "plinth"],
    },
    "greek-doric": {
        "name": "Greek Doric stonework", "orders": ["doric"],
        "system": "Stone post-and-lintel order with timber and tile roof",
        "rules": ["Shaft meets the stylobate without an individual base", "20 concave shaft flutes with taper and entasis",
                  "Annuli, spreading echinus and square abacus", "Triglyph rhythm over supports and between columns"],
        "load_path": ["roof framing", "entablature", "abacus and echinus", "column shaft", "stylobate and foundation"],
    },
    "greek-ionic": {
        "name": "Greek Ionic stonework", "orders": ["ionic"],
        "system": "Slender stone post-and-lintel order with moulded bases and scroll capitals",
        "rules": ["24 concave shaft flutes separated by fillets", "Torus and scotia base profiles",
                  "Opposed spiral volutes and a thin abacus", "Three advancing architrave fasciae and continuous frieze"],
        "load_path": ["roof framing", "entablature", "abacus and capital cushion", "shaft", "moulded base", "foundation"],
    },
    "roman-masonry": {
        "name": "Roman masonry and vaults", "orders": [],
        "system": "Arches, annular seating, groin vaults and domed chambers",
        "rules": ["Arcade tiers align with their piers", "Seat courses follow radial or elliptical sectors",
                  "Domes retain a hollow interior and continuous bearing ring", "Groin-vault hall has corner supports",
                  "Walls are coursed ashlars, not monolithic slabs"],
        "load_path": ["vault or arch", "springing and bearing", "ashlar courses", "foundation"],
    },
    "ashlar-masonry": {
        "name": "Human-scale ashlar masonry", "orders": [],
        "system": "Coursed stone blocks in running bond; openings are voids closed by lintels",
        "rules": ["Walls are stacked courses of lift-able stones, not one slab",
                  "Vertical joints stagger between courses",
                  "A doorway is the void left in the bond, spanned by a lintel stone",
                  "A stone is about one cubit high and two cubits long"],
        "load_path": ["lintel or arch", "ashlar courses", "foundation"],
    },
    "classical-revival": {
        "name": "Roman and later classical orders", "orders": ["tuscan", "corinthian", "composite"],
        "system": "Reusable classical orders also used by Renaissance and later architects",
        "rules": ["Tuscan shafts remain unfluted", "Corinthian bell carries shaped foliage",
                  "Composite capitals combine foliage and volutes", "Members retain a continuous column stack"],
        "load_path": ["entablature", "capital", "shaft", "base", "foundation"],
    },
    "gothic-masonry": {
        "name": "Gothic ribbed construction", "orders": [],
        "system": "Pointed vault ribs, clustered piers and flying buttresses",
        "rules": ["Ribs follow the nave and choir bays", "Flying arches land on external buttress piers",
                  "Pointed portals remain open", "Roof inspection preserves the complete export scene"],
        "load_path": ["vault ribs and webs", "clustered piers and flying buttresses", "buttress piers", "foundation"],
    },
    "norman-masonry": {
        "name": "Norman castle masonry", "orders": [],
        "system": "Thick walls of ashlar facing over a rubble core, round arches, battered plinths",
        "rules": ["A keep is entered at the first floor, by an external stair or forebuilding",
                  "The plinth spreads toward its foot",
                  "Shallow pilasters and clasping corner turrets stiffen the wall",
                  "Roofs sit below the wall-walk either side of a spine wall",
                  "Merlons cover a standing defender; curtain towers are rectangular until about 1200"],
        "load_path": ["roof and floors", "spine wall and outer walls", "battered plinth", "foundation"],
    },
    "egyptian-trabeated": {
        "name": "Egyptian trabeated stonework", "orders": [],
        "system": "Stone beams on plant-form columns under flat slab roofs; battered pylons and walls",
        "rules": ["No arch: every span is a single stone beam or slab",
                  "Slab span sets the column spacing",
                  "Pylon towers and outer walls lean inward as they rise",
                  "A taller nave is lit by a clerestory of stone grilles",
                  "Roofs fall and floors rise from pylon to sanctuary"],
        "load_path": ["roof slabs", "architraves", "abacus and capital", "column drums", "foundation"],
    },
    "mesoamerican-terraced": {
        "name": "Mesoamerican terraced platforms", "orders": [],
        "system": "Cut-stone facing over a rubble and earth core, raised in receding terraces",
        "rules": ["Every terrace stands inside the one below",
                  "Openings are lintelled; there is no true arch",
                  "Talud-tablero: a sloping talud under a framed vertical panel",
                  "A radial pyramid counts the year: four stairs of 91 steps and the platform",
                  "The stair arrives at the temple floor between sloping balustrades"],
        "load_path": ["temple", "summit platform", "terrace facings and core", "ground"],
    },
    "japanese-timber": {
        "name": "Japanese timber pagoda framing", "orders": [],
        "system": "Post-and-beam frame with bracket sets and deep eaves around a free central pillar",
        "rules": ["An odd number of storeys, each narrower than the one below",
                  "Each storey is a three-bay ring of twelve posts tied at the head",
                  "Bracket arms step outward to carry eaves nearly as deep as the body is wide",
                  "Upper storeys stand on beams laid across the frame below",
                  "The central pillar is independent of the floors and carries only the finial"],
        "load_path": ["tile and rafters", "eave purlin and bracket arms", "posts", "cross beams and struts", "stone podium"],
    },
    "chinese-timber": {
        "name": "Chinese timber hall framing",
        "orders": [],
        "system": "Post-and-beam hall on a stone terrace, with bracket sets (dougong) carrying deep tiled eaves",
        "rules": ["Bays are odd in number and shorten outward from a widest middle bay",
                  "An eave column stands no higher than its bay is wide",
                  "A bracket set steps out three doukou at a jump, and every set under an eave steps out alike",
                  "The roof steepens at every purlin step toward the ridge (jujia)",
                  "A corner shoots three doukou out and lifts four; an owl-tail bites each end of the ridge"],
        "load_path": ["tiles and rafters", "eave and frame purlins", "bracket sets and cross-beams",
                      "columns", "stone terrace"],
    },
    "dravidian-tiered": {
        "name": "Dravidian tiered temple towers",
        "orders": [],
        "system": "A stack of talas (storeys), each roofed by a prastara that carries a hara of miniature shrines",
        "rules": ["Every tala stands inside the one below it and is lower than it, giving the tower its concave profile",
                  "The summit of a gopuram is about a third of its base",
                  "Each prastara carries a hara: kutas at the corners, shalas along the sides, panjaras between",
                  "Kudu arches run along every eave, and pilasters march at even spacing along every face",
                  "A gopuram is a gate: its passage runs open through the stone base; kalasas crown every tower"],
        "load_path": ["crown and kalasa", "hara and prastara of each tala", "pilastered tala wall",
                      "stone base or sanctum", "moulded plinth"],
    },
    "nordic-stave": {
        "name": "Nordic stave timber framing",
        "orders": [],
        "system": "Load-bearing vertical pine staves on a stone foundation sill under steep shingled roofs and an encircling svalgang gallery",
        "rules": ["Every timber stave stands on the stone foundation sill, clear of bare earth",
                  "Shingled roof slopes hold a steep snow-shedding Nordic pitch",
                  "Raised-nave churches tier cleanly from svalgang gallery to aisle to nave to takrytter spire",
                  "Dragon-head finials and pierced ridge crests crown the gable peaks symmetrically",
                  "Choir and semicircular apse project east of the nave; lofts cantilever their upper gallery over the ground floor"],
        "load_path": ["wood shingles and ridge turret", "wall plates and scissor braces", "vertical pine staves",
                      "oak ground sills", "granite foundation sill"],
    },
}
HISTORIC_TRADITIONS = {"castle": "norman-masonry", "egyptian": "egyptian-trabeated",
                       "mesoamerican": "mesoamerican-terraced", "pagoda": "japanese-timber",
                       "chinese": "chinese-timber", "dravidian": "dravidian-tiered",
                       "stave": "nordic-stave"}


def craft_catalog():
    return {"schema_version": "blockn.craft/v1", "traditions": deepcopy(TRADITIONS),
        "scope": "Architectural geometry and construction conventions; not a structural capacity calculation",
        "sources": ["https://www.metmuseum.org/essays/architecture-in-ancient-greece",
                    "https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Vitruvius/3%2A.html",
                    "https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Vitruvius/4%2A.html",
                    "https://www.english-heritage.org.uk/learn/story-of-england/medieval/architecture/",
                    "https://whc.unesco.org/en/list/87/", "https://whc.unesco.org/en/list/414/",
                    "https://whc.unesco.org/en/list/483/", "https://whc.unesco.org/en/list/660/",
                    "https://whc.unesco.org/en/list/439/", "https://whc.unesco.org/en/list/250/"]}


def _stage(placement):
    role, kind = placement.role, placement.kind
    if kind in {"foundation", "slab"} or role == "podium":
        return "01_foundation"
    if kind in {"column", "pier", "wall"}:
        return "02_supports"
    # Timber houses tag their beams with the product. Historic beams carry the
    # architect id in the placement id. Other beams stay finishes.
    historic_member = bool(set(HISTORIC_TRADITIONS) & set(placement.id.split(".")))
    if kind in {"entablature", "arch", "bracket"} or "vault" in role or "buttress" in role or (
            kind == "beam" and (placement.parameters.get("product") == "timber-house" or historic_member)):
        return "03_spanning_members"
    if kind == "roof":
        return "04_roof"
    if kind in {"circulation", "seating", "platform"}:
        return "05_access"
    return "06_finish"


def construction_report(document, quality=None):
    groups = {}
    for p in document.placements:
        groups.setdefault(_stage(p), []).append(p.id)
    orders = sorted({str(p.parameters.get("order", "")).lower() for p in document.placements
                     if p.parameters.get("order")})
    product = document.request.product
    traditions = [key for key, value in TRADITIONS.items() if set(orders) & set(value["orders"])]
    roles = {p.role for p in document.placements}
    # Generic ribs also occur in Roman baths. Preserve product provenance in
    # composed IDs instead of assigning a historical tradition from one role.
    products = {product} | {token for p in document.placements for token in p.id.split(".")}
    if "timber-house" in products:
        traditions.append("timber-framing")
    if "gothic-cathedral" in products or {"nave_arcade", "flying_buttress"} <= roles:
        traditions.append("gothic-masonry")
    if products & {"classical-rotunda", "classical-baths", "classical-triumphal-arch",
                   "classical-aqueduct", "classical-theatre", "classical-amphitheatre",
                   "marble-run"} or roles & {"seating_tier", "channel_bed", "marble"}:
        traditions.append("roman-masonry")
    for token, tradition in HISTORIC_TRADITIONS.items():
        if token in products and tradition not in traditions:
            traditions.append(tradition)
    masonry_units = Counter(str(p.parameters.get("unit", "")) for p in document.placements if p.parameters.get("unit"))
    if masonry_units.keys() & {"ashlar", "sill", "lintel", "jamb", "drum", "paving", "radial_ashlar"} or any(
            "ashlar" in str(p.parameters.get("construction", "")) for p in document.placements):
        if "ashlar-masonry" not in traditions:
            traditions.append("ashlar-masonry")
    checks = [] if quality is None else [c.to_dict() for c in quality.checks
                                        if c.code.startswith(("craft.", "classical.", "cathedral.", "timber.", "mesh.",
                                                              *(f"{t}." for t in HISTORIC_TRADITIONS)))]
    courses = {int(p.parameters["course"]) for p in document.placements
               if isinstance(p.parameters.get("course"), int) or str(p.parameters.get("course", "")).isdigit()}
    physical = ((quality.metrics.get("physical") or {}) if quality is not None else {})
    return {"schema_version": "blockn.construction/v1", "product": product,
        "fingerprint": document.fingerprint(), "traditions": traditions, "orders": orders,
        "quality": None if quality is None else quality.status,
        "assembly_stages": [{"stage": stage, "count": len(ids), "primitive_ids": ids}
                            for stage, ids in sorted(groups.items())],
        "role_counts": dict(sorted(Counter(p.role for p in document.placements).items())),
        "materials": {
            "bill_of_materials": physical.get("bill_of_materials", {}),
            "total_mass_kg": physical.get("total_mass_kg"),
            "center_of_mass_m": physical.get("center_of_mass_m"),
            "footprint_area_m2": physical.get("footprint_area_m2"),
            "ground_bearing_pressure_kpa": physical.get("ground_bearing_pressure_kpa"),
            "max_member_utilisation": physical.get("max_member_utilisation"),
            "basis": "per-substance mass and volume from compiled meshes at one block unit per metre",
        },
        "masonry": {"stone_units": dict(sorted(masonry_units.items())),
                    "course_count": len(courses),
                    "bond": "running" if masonry_units else None},
        "checks": checks, "scope": "Geometry, member identity and bearing diagnostics",
        "not_evaluated": ["lateral stability and wind", "thrust lines in arches and vaults",
                          "tension and bending capacity", "foundations and soil",
                          "connections", "building-code compliance"]}


def order_checks(document, scene):
    """Measure actual shafts and require each ordered capital's defining parts.

    Bearings below are projected envelope/contact diagnostics. They do not
    infer compression stress, thrust lines, material strength or capacity.
    """
    from .quality import _check
    placements = document.placements
    shafts = [p for p in placements if p.role == "column_shaft"]
    if not shafts:
        return []
    meshes = {b.parameters.get("primitive_id"): b.mesh for b in scene.blocks}
    failures, profiles, bearings = [], [], []
    structural = [p for p in placements if p.role in {"architrave", "entablature", "drum", "bearing_ring"}]
    stacks = defaultdict(list)
    for shaft in shafts:
        host = shaft.parameters.get("host_id")
        key = host or (round(float(shaft.position[0]), 4), round(float(shaft.position[2]), 4))
        stacks[key].append(shaft)
    for stack in stacks.values():
        stack.sort(key=lambda item: item.position[1])
        top = stack[-1]
        order = str(top.parameters.get("order", document.request.variant)).lower()
        diameters = []
        for shaft in stack:
            item = meshes.get(shaft.id)
            if item is None:
                continue
            lo, hi = item.bounds
            diameters.append(max(hi[0]-lo[0], hi[2]-lo[2]))
        if not diameters:
            continue
        diameter = max(diameters)
        y0 = min(p.position[1] - p.dimensions[1] / 2 for p in stack)
        y1 = max(p.position[1] + p.dimensions[1] / 2 for p in stack)
        parts = [p for p in placements if p.kind == "column"
                 and abs(p.position[0]-top.position[0]) < diameter*1.35
                 and abs(p.position[2]-top.position[2]) < diameter*1.35
                 and y0 - diameter <= p.position[1] <= y1 + diameter]
        elements = {p.parameters.get("element") for p in parts}
        if order == "doric" and (elements & {"base", "base_lower", "base_upper", "base_scotia"}
                                 or not {"capital_echinus", "capital_abacus", "capital_annuli"} <= elements):
            failures.append(top.id)
        if order == "ionic" and (not {"base_lower", "base_upper", "base_scotia", "capital_volute", "capital_cushion", "capital_abacus"} <= elements
                                  or sum(p.parameters.get("element") == "capital_volute" for p in parts) < 4):
            failures.append(top.id)
        if order in {"doric", "ionic"}:
            stack_bottom = meshes.get(stack[0].id)
            stack_top = meshes.get(stack[-1].id)
            taper_ok = False
            if stack_bottom is not None and stack_top is not None:
                blo, bhi = stack_bottom.bounds
                tlo, thi = stack_top.bounds
                bcenter = (blo + bhi) / 2
                tcenter = (tlo + thi) / 2
                bottom = stack_bottom.vertices[np.abs(stack_bottom.vertices[:,1]-blo[1])<1e-7]
                top_ring = stack_top.vertices[np.abs(stack_top.vertices[:,1]-thi[1])<1e-7]
                if len(bottom) and len(top_ring):
                    br = np.max(np.linalg.norm(bottom[:,[0,2]]-bcenter[[0,2]],axis=1))
                    tr = np.max(np.linalg.norm(top_ring[:,[0,2]]-tcenter[[0,2]],axis=1))
                    taper_ok = tr < br * .99
            drum = stack_bottom
            flutes_ok = False
            if drum is not None:
                lo, hi = drum.bounds
                center = (lo + hi) / 2
                vertices = drum.vertices
                ys = np.unique(np.round(vertices[:,1],9))
                y = ys[np.argmin(abs(ys-center[1]))]
                ring = vertices[np.abs(vertices[:,1]-y)<1e-7]
                radii = np.linalg.norm(ring[:,[0,2]]-center[[0,2]],axis=1)
                radii = radii[radii > diameter*.25]
                flutes_ok = len(radii)>=20 and np.ptp(radii)>diameter*.012
            if not (flutes_ok and taper_ok):
                profiles.append(top.id)
        abaci = [p for p in parts if p.parameters.get("element") == "capital_abacus"]
        host = top.parameters.get("host_id")
        if host:
            capitals = [p for p in abaci if p.parameters.get("host_id") == host]
        else:
            capitals = [p for p in abaci
                        if abs(p.position[0]-top.position[0]) < diameter
                        and abs(p.position[2]-top.position[2]) < diameter]
        if capitals:
            clo = np.min([np.asarray(p.position)-np.asarray(p.dimensions)/2 for p in capitals], axis=0)
            chi = np.max([np.asarray(p.position)+np.asarray(p.dimensions)/2 for p in capitals], axis=0)
            supporting=[]
            for beam in structural:
                blo=np.asarray(beam.position)-np.asarray(beam.dimensions)/2
                bhi=np.asarray(beam.position)+np.asarray(beam.dimensions)/2
                overlap=np.minimum(chi[[0,2]],bhi[[0,2]])-np.maximum(clo[[0,2]],blo[[0,2]])
                if abs(blo[1]-chi[1])<max(1e-4,diameter*.003) and np.all(overlap>diameter*.03):
                    supporting.append(beam.id)
            if not supporting:
                bearings.append(capitals[0].id)
    return [
        _check("craft.order_members", not failures, "Doric and Ionic columns retain their defining members.",
               "A column has incompatible or missing order-specific members.", primitive_ids=sorted(set(failures))),
        _check("craft.shaft_profiles", not profiles, "Classical shafts have measured concave flutes and taper.",
               "A Doric or Ionic shaft lacks physical flute depth or taper.", primitive_ids=profiles),
        _check("craft.capital_bearings", not bearings, "Every capital meets a spanning member with positive projected bearing.",
               "A capital lacks a spanning member at its top bearing surface.", primitive_ids=bearings),
    ]
