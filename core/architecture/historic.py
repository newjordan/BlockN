"""Shared composer for the historic architects.

Each tradition (Norman castle, Egyptian temple, Mesoamerican pyramid, timber
pagoda) is its own module with its own rules. They share this vocabulary of
built members: coursed ashlar skins, battered masses, ring walls, round arches
with real voussoirs, crenellation, and plain timber members.

Masonry is always emitted as lift-able stones. A thick wall is two ashlar
skins; the rubble core between them is not modelled.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import math
import random

import numpy as np
import trimesh

from core.architectural_geometry import extrude_profile, mesh_parameters
from .masonry import (MAX_COURSE, MAX_STRETCHER, MIN_STONE, course_target, iter_ashlar_box,
                      iter_masonry_solid, stretcher_target, _divisions, _merge_marks)
from .schema import BuildDocument, BuildRequest, Placement, canonical_product_name

TAU = math.tau
DETAILS = ("concept", "standard", "presentation")
STONE = (214, 204, 182, 255)


@dataclass(frozen=True)
class Form:
    """One catalog product of an architect."""

    name: str
    description: str
    defaults: tuple                 # width, height, depth
    variants: tuple
    required_roles: frozenset
    controls: dict = field(default_factory=dict)    # default bays / levels
    parameters: tuple = ()
    rules: dict = field(default_factory=dict)


# Architect modules in catalog order. A new tradition is one module and one name here.
ARCHITECTS = ("castle", "egyptian", "mesoamerican", "pagoda", "chinese", "dravidian", "stave", "sahelian", "inka",
              "khmer", "swahili", "minangkabau", "chacoan", "brutalist", "byzantine",
              "mesopotamian", "stupa", "muscovite", "zimbabwe", "lalibela", "artdeco", "geodesic", "tibetan",
              "mughal")


def styles():
    """Product name -> (architect module, form), in catalog order."""
    from importlib import import_module
    modules = [import_module(f".{name}", __package__) for name in ARCHITECTS]
    return {name: (module, form) for module in modules
            for name, form in module.FORMS.items()}


def style_for(product):
    return styles().get(canonical_product_name(product))


def roll_request(product, seed, *, variant=None, detail=None):
    """Sample explicit controls with the same seed in the CLI, desktop, and agent API."""
    found = style_for(product)
    if found is None:
        raise ValueError(f"{product} does not have a historic reroll sampler")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    module, form = found
    if variant not in (None, "auto", "random", *form.variants):
        raise ValueError(f"{form.name} variant must be one of {', '.join(form.variants)}")
    if detail is not None and detail not in DETAILS:
        raise ValueError(f"detail must be one of {', '.join(DETAILS)}")
    rng = random.Random(f"{form.name}:{seed}:controls")
    controls = module.random_controls(form.name, rng)
    selected = variant if variant in form.variants else controls.get("variant") or rng.choice(form.variants)
    for name in ("bays", "levels"):
        if name not in form.controls:
            controls.pop(name, None)
    for name, choices in form.rules.get("reroll_controls_by_variant", {}).get(selected, {}).items():
        controls[name] = rng.choice(choices)
    controls["variant"] = selected
    controls["detail"] = detail or controls.get("detail") or rng.choice(DETAILS[:2])
    for name in ("width", "height", "depth"):
        controls[name] = round(controls[name])  # same whole-metre grid as the desktop
    return BuildRequest(architect=module.ARCHITECT.id, style=module.STYLE, product=form.name, seed=seed, **controls)


def list_products():
    return [descriptor(module, form) for module, form in styles().values()]


def descriptor(module, form):
    width, height, depth = form.defaults
    defaults = {"width": width, "height": height, "depth": depth,
                "variant": form.variants[0], "detail": "standard", **form.controls}
    substances = getattr(module, "SUBSTANCES", {})
    # An architect whose products are built of different materials says so for each form.
    vocabulary = (module.substances_for(form) if hasattr(module, "substances_for") else
                  set(substances.values()) | {getattr(module, "DEFAULT_SUBSTANCE", "limestone")})
    return {"name": form.name, "category": "assembly", "architect": module.ARCHITECT.to_dict(),
            "architect_id": module.ARCHITECT.id, "architect_title": module.ARCHITECT.title,
            "architect_version": module.ARCHITECT.version, "description": form.description,
            "defaults": defaults, "variants": ["auto", "random", *form.variants],
            "detail_levels": list(DETAILS), "opening_modes": ["auto"],
            "reroll": {"command": f"blockn roll {form.name}", "seeded": True},
            "parameters": sorted({"variation", *form.parameters}),
            "substances": sorted(vocabulary),
            "rules": {"required_roles": sorted(form.required_roles), **deepcopy(form.rules), "variation": [0.0, 1.0],
                      "scope": "architectural visualization and modular asset generation"}}


def build_document(request):
    module, _ = style_for(request.product)
    return module.build_document(request)


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not low <= value <= high:
        raise ValueError(f"{name} must be a number between {low} and {high}")
    return float(value)


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def choice(value, name, options):
    if value not in options:
        raise ValueError(f"{name} must be one of {', '.join(options)}")
    return value


def box_mesh(center, dimensions):
    mesh = trimesh.creation.box(extents=[float(v) for v in dimensions])
    mesh.apply_translation([float(v) for v in center])
    return mesh


def hull(points):
    return trimesh.Trimesh(vertices=np.unique(np.round(np.asarray(points, dtype=float), 9), axis=0)).convex_hull


def sector_mesh(cx, cz, inner, outer, a0, a1, y0, y1, inner_top=None, outer_top=None, segments=2):
    """One annular stone. Radii may differ top and bottom for a battered drum."""
    inner_top = inner if inner_top is None else inner_top
    outer_top = outer if outer_top is None else outer_top
    angles = np.linspace(a0, a1, segments + 1)
    vertices = []
    for y, r_out, r_in in ((y0, outer, inner), (y1, outer_top, inner_top)):
        for a in angles:
            vertices.append((cx + r_out * math.cos(a), y, cz + r_out * math.sin(a)))
            vertices.append((cx + r_in * math.cos(a), y, cz + r_in * math.sin(a)))
    n = 2 * (segments + 1)
    faces = []

    def quad(a, b, c, d):
        faces.extend(((a, b, c), (a, c, d)))
    for i in range(segments):
        o0, i0, o1, i1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        quad(o0, o1, i1, i0)                              # bed
        quad(n + o0, n + i0, n + i1, n + o1)              # top
        quad(o0, n + o0, n + o1, o1)                      # outer face
        quad(i0, i1, n + i1, n + i0)                      # inner face
    quad(0, 1, n + 1, n)                                  # start joint
    last = 2 * segments
    quad(last, n + last, n + last + 1, last + 1)          # end joint
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def oriented_profile(polygon, depth, axis, along, cross):
    """Extrude an (along, y) profile through a wall that runs on X or Z."""
    mesh = extrude_profile(polygon, depth)
    v = mesh.vertices
    if axis == "x":
        vertices = np.column_stack((along + v[:, 0], v[:, 1], cross + v[:, 2]))
    else:
        vertices = np.column_stack((cross + v[:, 2], v[:, 1], along + v[:, 0]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def arch_openings(center, half, sill, spring, ring, opening_id):
    """Void for a round-headed opening: the light, then room for the arch ring."""
    outer = half + ring
    return [{"id": opening_id, "center": center, "width": 2 * half, "sill": sill, "height": spring - sill},
            {"id": opening_id + ".head", "center": center, "width": 2 * outer, "sill": spring, "height": outer}]


class Voids:
    """Openings of one wall run, given in world heights."""

    def __init__(self, base):
        self.base, self.specs, self.arches = base, [], []

    def rect(self, center, width, sill, height, opening_id):
        self.specs.append({"id": opening_id, "center": center, "width": width,
                           "sill": sill - self.base, "height": height})

    def arch(self, center, half, sill, spring, ring, opening_id, role=None):
        self.specs += arch_openings(center, half, sill - self.base, spring - self.base, ring, opening_id)
        self.arches.append((center, half, spring, ring, opening_id, role))

    def slots(self, along, axis, margin=.12):
        """The openings as (a0, a1, y0, y1, axis) in world terms, for cutting the core behind them."""
        return [(along + spec["center"] - spec["width"] / 2 - margin, along + spec["center"] + spec["width"] / 2 + margin,
                 self.base + spec["sill"], self.base + spec["sill"] + spec["height"] + margin, axis) for spec in self.specs]


def clamp(value, low, high):
    return max(low, min(high, value))


def rotation(angle, cx=0.0, cz=0.0):
    """Warp that turns a member about a vertical axis; angle runs from +X toward +Z."""
    c, s = math.cos(angle), math.sin(angle)

    def warp(vertices):
        out = vertices.copy()
        out[:, 0] = cx + vertices[:, 0] * c - vertices[:, 2] * s
        out[:, 2] = cz + vertices[:, 0] * s + vertices[:, 2] * c
        return out
    return warp


class Composer:
    """Base for one architect. Subclasses set ``module`` and write compose()."""

    module = None           # the architect module (ARCHITECT, KINDS, FORMS, ...)
    PALETTE = {}            # role or kind -> RGBA
    TIMBER_KINDS = frozenset()   # coarse kinds this tradition raises as timbers, counted apart from stone
    SIZE = (6.0, 200.0)

    def __init__(self, request):
        m = self.module
        self.request = request
        self.form = form = m.FORMS[canonical_product_name(request.product)]
        values = []
        for name, default in zip(("width", "height", "depth"), form.defaults):
            value = getattr(request, name)
            values.append(number(default if value is None else value, f"{form.name} {name}", *self.SIZE))
        self.w, self.h, self.d = values
        if request.detail not in DETAILS:
            raise ValueError(f"{form.name} detail must be concept, standard, or presentation")
        if request.openings != "auto":
            raise ValueError(f"{form.name} authors its own openings; use openings=auto")
        allowed = {"variation", *form.parameters}
        if set(request.parameters) - allowed:
            raise ValueError(f"{form.name} parameters: {', '.join(sorted(allowed))}")
        # 0 builds the canonical proportions; 1 lets the seed range across the tradition.
        self.variation = number(request.parameters.get("variation", 1.0), f"{form.name} variation", 0.0, 1.0)
        for name in ("bays", "levels"):
            if getattr(request, name) is not None and name not in form.controls:
                raise ValueError(f"{form.name} does not take {name}")
        self.rng = random.Random(request.seed)
        variant = request.variant
        if variant == "auto":
            variant = form.variants[0]
        elif variant == "random":
            variant = self.rng.choice(form.variants)
        if variant not in form.variants:
            raise ValueError(f"{form.name} variant must be auto, random, or one of {', '.join(form.variants)}")
        self.variant = variant
        self.detail = self.grade = request.detail
        self.prefix = m.ARCHITECT.id
        self.placements, self.counts = [], Counter()
        # The player the building is for; None keeps the historic sizes.
        player_of = getattr(request, "player_profile", None)
        self.player = player_of() if callable(player_of) else None

    def door(self, half, sill, spring):
        """A doorway arch's half-width and springing height, grown to fit the
        player's door: as wide as the profile door, and high enough that the
        arch clears the door height over the body's full width. Without a
        player the historic sizes come back unchanged."""
        if self.player is None:
            return half, spring
        clear = self.player.clearances()
        half = max(half, clear.door_width / 2)
        crown = math.sqrt(max(0.0, half * half - self.player.radius ** 2))
        return half, max(spring, sill + clear.door_height - crown)

    def passage(self, width):
        """A stair, bridge or passage width grown to the player's door width."""
        return width if self.player is None else max(width, self.player.clearances().door_width)

    def vary(self, key, spread):
        """A seeded multiplier near 1 for one named proportion; stable whatever else is built."""
        if not self.variation:
            return 1.0
        draw = random.Random(f"{self.request.seed}:{self.form.name}:{key}").uniform(-1, 1)
        return 1 + spread * self.variation * draw

    def pick(self, key, options):
        """A seeded choice; the first option is the canonical one."""
        if not self.variation:
            return options[0]
        return random.Random(f"{self.request.seed}:{self.form.name}:{key}").choice(options)

    def control(self, name, low, high):
        value = getattr(self.request, name)
        return integer(self.form.controls[name] if value is None else value, name, low, high)

    def ratio(self, value, name, low, high):
        if not low - 1e-9 <= value <= high + 1e-9:
            raise ValueError(f"{self.form.name} requires {name} between {low} and {high}")

    # ------------------------------------------------------------ members
    def host(self, role):
        return f"{self.prefix}.{role}.{self.counts[role]:04d}"

    def add(self, role, position, dimensions, block_type="CUBE", **params):
        kind = self.module.KINDS[role]
        pid = self.host(role)
        self.counts[role] += 1
        material = self.PALETTE.get(role) or self.PALETTE.get(kind) or STONE
        self.placements.append(Placement(
            id=pid, kind=kind, role=role, block_type=block_type,
            position=tuple(float(v) for v in position), dimensions=tuple(float(v) for v in dimensions),
            material=material, parent_id=None, parameters={"style": self.variant, **params},
            substance=self.substance(role, kind)))
        return pid

    def substance(self, role, kind):
        """The physical substance of a member: by role, then coarse kind, then the module default.

        The tradition's own data decides role by role. An architect whose
        products are built of different materials overrides this.
        """
        substances = getattr(self.module, "SUBSTANCES", {})
        return (substances.get(role) or substances.get(kind)
                or getattr(self.module, "DEFAULT_SUBSTANCE", "limestone"))

    def mesh(self, role, mesh, **params):
        position, dimensions, data = mesh_parameters(mesh)
        return self.add(role, position, dimensions, "MESH", **data, **params)

    def stone(self, role, mesh, host, course, **params):
        """A cut stone that is not a plain box."""
        return self.mesh(role, mesh, construction="ashlar running bond", unit="profile_course",
                         course=course, host_id=host, **params)

    def ashlar(self, role, center, dimensions, openings=(), axis=None, grade=None, warp=None, **params):
        """Course a box as stones. ``axis`` makes it a single skin and allows openings.

        ``warp`` maps an (n, 3) vertex array to its built position, so a
        battered or sheared member is still laid in the same bond.
        """
        host = self.host(role)
        grade = grade or self.grade
        if axis:
            stones = iter_ashlar_box(center, dimensions, axis, openings, grade)
        else:
            stones = iter_masonry_solid(center, dimensions, grade)
        for stone in stones:
            if warp is None:
                extra = {"opening_id": stone.opening_id} if stone.opening_id else {}
                self.add(role, stone.position, stone.dimensions, construction="ashlar running bond",
                         unit=stone.unit, course=stone.course, stone_index=stone.index,
                         host_id=host, axis=stone.axis, **extra, **params)
            else:
                mesh = box_mesh(stone.position, stone.dimensions)
                mesh.vertices = warp(np.asarray(mesh.vertices))
                self.stone(role, mesh, host, stone.course, **params)
        return host

    def shell(self, role, x0, x1, z0, z1, y0, y1, thickness, openings=None, faces="fblr",
              grade=None, warp=None, **params):
        """Four ashlar skins around a rectangle. Front is -Z, left is -X."""
        o = openings or {}
        cx, cz, cy, t = (x0 + x1) / 2, (z0 + z1) / 2, (y0 + y1) / 2, thickness
        w, d, h = x1 - x0, z1 - z0, y1 - y0
        runs = {"f": ((cx, cy, z0 + t / 2), (w, h, t), "x", "front"),
                "b": ((cx, cy, z1 - t / 2), (w, h, t), "x", "back"),
                "l": ((x0 + t / 2, cy, cz), (t, h, d - 2 * t), "z", "left"),
                "r": ((x1 - t / 2, cy, cz), (t, h, d - 2 * t), "z", "right")}
        for key in faces:
            center, dims, axis, name = runs[key]
            if min(dims) > MIN_STONE:
                self.ashlar(role, center, dims, o.get(name, ()), axis, grade, warp, face=name, **params)

    def run(self, role, axis, along, cross, length, y0, y1, thickness, voids=None, grade=None,
            warp=None, **params):
        """One straight skin with its round-headed openings built as real arches."""
        centre = (along, (y0 + y1) / 2, cross) if axis == "x" else (cross, (y0 + y1) / 2, along)
        dims = (length, y1 - y0, thickness) if axis == "x" else (thickness, y1 - y0, length)
        self.ashlar(role, centre, dims, voids.specs if voids else (), axis, grade, warp, **params)
        for centre, half, spring, ring, opening_id, arch_role in (voids.arches if voids else ()):
            self.arch_head(role, axis, along + centre, cross, thickness, half, spring, ring,
                           opening_id, arch_role, **params)

    def shift(self, start, dx, dz):
        """Move everything placed since ``start``; lets a part be composed at the origin."""
        for p in self.placements[start:]:
            p.position = (p.position[0] + dx, p.position[1], p.position[2] + dz)

    def core(self, x0, x1, y0, y1, z0, z1, voids=(), warp=None, role="core_fill", **params):
        """Fill a wall or platform core with the largest blocks a gang can set.

        ``voids`` are (a0, a1, y0, y1, axis) slots kept open through the core, so
        a window in the facing is an embrasure and not a blind recess.
        """
        host = self.host(role)
        nx, ny, nz = (max(1, int(math.ceil((b - a) / limit - 1e-9)))
                      for a, b, limit in ((x0, x1, MAX_STRETCHER), (y0, y1, MAX_COURSE), (z0, z1, MAX_STRETCHER)))
        long_x = (x1 - x0) >= (z1 - z0)
        for j in range(ny):
            ya, yb = y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny
            xs = _bond(x0, x1, nx, j % 2 == 1 and long_x)
            zs = _bond(z0, z1, nz, j % 2 == 1 and not long_x)
            for xa, xb in xs:
                for za, zb in zs:
                    top = yb
                    cut = [v0 for a0, a1, v0, v1, axis in voids
                           if ya < v1 - 1e-9 and yb > v0 + 1e-9 and
                           ((xa if axis == "x" else za) < a1 and (xb if axis == "x" else zb) > a0)]
                    if cut:
                        # For a player, a block that straddles a sill is cut down to
                        # it, so a doorway through a thick wall has a floor.
                        if self.player is None or min(cut) - ya < .05:
                            continue
                        top = min(cut)
                    centre, dims = ((xa + xb) / 2, (ya + top) / 2, (za + zb) / 2), (xb - xa, top - ya, zb - za)
                    if warp is None:
                        self.add(role, centre, dims, construction="rubble and mortar core", unit="core",
                                 course=j, host_id=host, **params)
                    else:
                        mesh = box_mesh(centre, dims)
                        mesh.vertices = warp(np.asarray(mesh.vertices))
                        self.mesh(role, mesh, construction="rubble and mortar core", unit="core",
                                  course=j, host_id=host, **params)
        return host

    def mirror(self, start):
        """Reflect everything placed since ``start`` across the plan's long axis (x -> -x)."""
        for p in self.placements[start:]:
            p.position = (-p.position[0], p.position[1], p.position[2])
            if p.block_type == "MESH":
                p.parameters["vertices"] = [[-v[0], v[1], v[2]] for v in p.parameters["vertices"]]
                p.parameters["faces"] = [[f[0], f[2], f[1]] for f in p.parameters["faces"]]

    def turn(self, start, quarters):
        """Rotate everything placed since ``start`` by quarter turns about the vertical axis."""
        for p in self.placements[start:]:
            for _ in range(quarters % 4):
                x, y, z = p.position
                p.position = (-z, y, x)
                p.dimensions = (p.dimensions[2], p.dimensions[1], p.dimensions[0])
                if "axis" in p.parameters:
                    p.parameters["axis"] = "z" if p.parameters["axis"] == "x" else "x"
                if p.block_type == "MESH":
                    p.parameters["vertices"] = [[-v[2], v[1], v[0]] for v in p.parameters["vertices"]]

    def clone(self, start, count=None):
        """Copy ``count`` members placed since ``start``; return where the copy begins.

        The copies keep their places, so a caller can then ``turn``, ``mirror``
        or ``shift`` them onto another side or corner.
        """
        count = len(self.placements) - start if count is None else count
        for p in self.placements[start:start + count]:
            self.placements.append(Placement(
                id=self.host(p.role), kind=p.kind, block_type=p.block_type, position=p.position,
                dimensions=p.dimensions, material=p.material, role=p.role, parameters=dict(p.parameters),
                substance=p.substance))
            self.counts[p.role] += 1
        return len(self.placements) - count

    def spin(self, start):
        """Repeat everything placed since ``start`` on the other three sides of a square."""
        count = len(self.placements) - start
        for quarter in (1, 2, 3):
            self.turn(self.clone(start, count), quarter)

    def drum(self, role, cx, cz, y0, y1, r0, r1=None, sides=16, **params):
        """One round drum; radii may differ for a tapered or swelling shaft."""
        r1 = r0 if r1 is None else r1
        points = [(cx + r * math.cos(TAU * i / sides), y, cz + r * math.sin(TAU * i / sides))
                  for y, r in ((y0, r0), (y1, r1)) for i in range(sides)]
        return self.mesh(role, hull(points), **params)

    def lathe(self, role, cx, cz, profile, sides=16, course=MAX_COURSE, **params):
        """Stack drums along a (height, radius) profile, none taller than a course."""
        host = self.host(role)
        index = 0
        for (ya, ra), (yb, rb) in zip(profile, profile[1:]):
            count = max(1, int(math.ceil((yb - ya) / course - 1e-9)))
            for k in range(count):
                u0, u1 = k / count, (k + 1) / count
                self.drum(role, cx, cz, ya + (yb - ya) * u0, ya + (yb - ya) * u1,
                          ra + (rb - ra) * u0, ra + (rb - ra) * u1, sides,
                          construction="ashlar stack", unit="drum", course=index, host_id=host, **params)
                index += 1
        return host

    def stack(self, role, x, bottom, z, width, depth, height, block_type="CUBE", **params):
        """A pier or post laid as drums no taller than a course."""
        host = self.host(role)
        count = _divisions(height, MAX_COURSE, MAX_COURSE)
        size = height / count
        for index in range(count):
            extra = {"construction": "ashlar stack", "unit": "drum", "course": index, "host_id": host, **params}
            if block_type == "CYLINDER":
                extra.update(radius=min(width, depth) / 2, height=size)
            self.add(role, (x, bottom + (index + .5) * size, z), (width, size, depth), block_type, **extra)
        return host

    def ring(self, role, cx, cz, radius, thickness, y0, y1, gaps=(), arc=(0.0, TAU),
             top_radius=None, grade=None, core=False, **params):
        """A curved wall of radial ashlars. ``gaps`` are (angle, half_angle, y0, y1) voids."""
        host = self.host(role)
        grade = grade or self.grade
        start, end = arc
        closed = end - start >= TAU - 1e-9
        top_radius = radius if top_radius is None else top_radius
        ny = _divisions(y1 - y0, course_target(grade), MAX_COURSE)
        count = _divisions((end - start) * max(radius, top_radius), stretcher_target(grade), MAX_STRETCHER)
        module = (end - start) / count
        sliver = MIN_STONE * 1.5 / max(radius, top_radius)
        # A void starts and stops on a bed joint, as in a coursed straight wall.
        protected = [min(max(g - y0, 0.0), y1 - y0) for gap in gaps for g in gap[2:4]]
        marks = [(y1 - y0) * k / ny for k in range(ny + 1)] + protected
        edges = [y0 + mark for mark in _merge_marks(marks, protected, y1 - y0)]
        for course, (ya, yb) in enumerate(zip(edges, edges[1:])):
            mid = (ya + yb) / 2
            ra = radius + (top_radius - radius) * (ya - y0) / (y1 - y0)
            rb = radius + (top_radius - radius) * (yb - y0) / (y1 - y0)
            spans = [(start, end)]
            for angle, half, g0, g1 in gaps:
                if not g0 < mid < g1:
                    continue
                for turn in (-TAU, 0.0, TAU) if closed else (0.0,):
                    lo, hi = angle - half + turn, angle + half + turn
                    spans = [piece for a, b in spans for piece in ((a, min(b, lo)), (max(a, hi), b))
                             if piece[1] - piece[0] > 1e-9]
            shift = module / 2 if course % 2 else 0.0
            for a, b in spans:
                cuts = [a] + [start + shift + k * module for k in range(count + 1)
                              if a + sliver < start + shift + k * module < b - sliver] + [b]
                for index, (c0, c1) in enumerate(zip(cuts, cuts[1:])):
                    self.mesh(role, sector_mesh(cx, cz, ra - thickness, ra, c0, c1, ya, yb,
                                                rb - thickness, rb),
                              construction="rubble and mortar core" if core else "ashlar running bond",
                              unit="core" if core else "radial_ashlar",
                              course=course, stone_index=index, host_id=host, **params)
        return host

    def frustum(self, role, cx, cz, y0, y1, r0, r1, sides=12, rot=0.0, rise=MAX_COURSE, **params):
        """A solid cone or tapering drum, cut in courses and radial segments."""
        host = self.host(role)
        rings = max(1, int(math.ceil((y1 - y0) / rise - 1e-9)))
        for ring in range(rings):
            ya, yb = y0 + (y1 - y0) * ring / rings, y0 + (y1 - y0) * (ring + 1) / rings
            ra, rb = r0 + (r1 - r0) * ring / rings, r0 + (r1 - r0) * (ring + 1) / rings
            for i in range(sides):
                a0, a1 = rot + TAU * i / sides, rot + TAU * (i + 1) / sides
                points = [(cx, ya, cz), (cx, yb, cz)]
                for y, r in ((ya, ra), (yb, rb)):
                    if r > 1e-6:
                        points += [(cx + r * math.cos(a), y, cz + r * math.sin(a)) for a in (a0, a1)]
                self.mesh(role, hull(points), course=ring, host_id=host, **params)
        return host

    def arch_head(self, role, axis, along, cross, thickness, half, spring, ring, opening_id=None,
                  voussoir_role=None, **params):
        """Round arch of voussoirs with coursed spandrels, filling arch_openings()' head."""
        outer = half + ring
        count = max(5, int(math.ceil(math.pi * half / 0.42)) | 1)
        arch_role = voussoir_role or role
        host = self.host(arch_role)
        extra = {"opening_id": opening_id} if opening_id else {}
        for index in range(count):
            a0, a1 = math.pi * index / count, math.pi * (index + 1) / count
            am = (a0 + a1) / 2
            polygon = [(outer * math.cos(a), spring + outer * math.sin(a)) for a in (a0, am, a1)]
            polygon += [(half * math.cos(a), spring + half * math.sin(a)) for a in (a1, am, a0)]
            self.mesh(arch_role, oriented_profile(polygon, thickness, axis, along, cross),
                      construction="ashlar voussoir arch", unit="voussoir", course=index, host_id=host,
                      keystone=index == count // 2, **extra, **params)
        bands = _divisions(outer, course_target(self.grade), MAX_COURSE)
        host = self.host(role)
        for band in range(bands):
            ya, yb = outer * band / bands, outer * (band + 1) / bands
            xa, xm, xb = (math.sqrt(max(0.0, outer * outer - y * y)) for y in (ya, (ya + yb) / 2, yb))
            if outer - xb < MIN_STONE / 2:
                continue
            for side in (-1, 1):
                polygon = [(side * outer, spring + ya), (side * xa, spring + ya),
                           (side * xm, spring + (ya + yb) / 2), (side * xb, spring + yb),
                           (side * outer, spring + yb)]
                polygon = [p for i, p in enumerate(polygon)
                           if i == 0 or math.dist(p, polygon[i - 1]) > 1e-6]
                self.stone(role, oriented_profile(polygon, thickness, axis, along, cross), host, band,
                           **{"register": "spandrel", **extra, **params})

    def crenellate(self, role, axis, a0, a1, cross, y, thickness, merlon=1.3, crenel=0.7, height=0.9,
                   **params):
        """Merlons along a straight parapet; the run starts and ends on a merlon."""
        length = a1 - a0
        gaps = max(1, int(round((length - merlon) / (merlon + crenel))))
        scale = length / ((gaps + 1) * merlon + gaps * crenel)
        m, c = merlon * scale, crenel * scale
        for index in range(gaps + 1):
            centre = a0 + index * (m + c) + m / 2
            position = (centre, y + height / 2, cross) if axis == "x" else (cross, y + height / 2, centre)
            dims = (m, height, thickness) if axis == "x" else (thickness, height, m)
            self.ashlar(role, position, dims, axis=axis, **params)
        return gaps + 1

    def slab(self, role, x0, x1, z0, z1, y0, y1, module=1.2, warp=None, **params):
        """A thin layer cut into plan modules: paving, decking, roof sheets."""
        host = self.host(role)
        nx, nz = (max(1, int(math.ceil((b - a) / module - 1e-9))) for a, b in ((x0, x1), (z0, z1)))
        for i in range(nx):
            for j in range(nz):
                xa, xb = x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx
                za, zb = z0 + (z1 - z0) * j / nz, z0 + (z1 - z0) * (j + 1) / nz
                center, dims = ((xa + xb) / 2, (y0 + y1) / 2, (za + zb) / 2), (xb - xa, y1 - y0, zb - za)
                if warp is None:
                    self.add(role, center, dims, host_id=host, **params)
                else:
                    mesh = box_mesh(center, dims)
                    mesh.vertices = warp(np.asarray(mesh.vertices))
                    self.mesh(role, mesh, host_id=host, **params)
        return host

    def member(self, role, x0, x1, y0, y1, z0, z1, block_type="CUBE", **params):
        """A single axis-aligned member given by its corners (timber, slabs)."""
        return self.add(role, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                        (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), block_type, **params)

    def block_count(self):
        """How many pieces it takes to build this: cut stone, core blocks, timber and the rest."""
        counts = Counter()
        for p in self.placements:
            construction = str(p.parameters.get("construction", ""))
            if p.parameters.get("unit") == "core":
                counts["core_blocks"] += 1
            elif "ashlar" in construction or p.parameters.get("unit") in {"drum", "voussoir", "radial_ashlar"}:
                counts["cut_stones"] += 1
            elif p.kind in self.TIMBER_KINDS:
                counts["timbers"] += 1
            elif "monolith" in construction:
                counts["monoliths"] += 1
            else:
                counts["other_members"] += 1
        return {**dict(sorted(counts.items())), "total": len(self.placements)}

    def document(self, rules):
        m = self.module
        return BuildDocument(m.ARCHITECT, self.request, self.placements, metadata={
            f"{self.prefix}_rules": {"variant": self.variant, "variation": self.variation,
                                     "block_count": self.block_count(), **rules,
                                     "scope": "visual architecture; geometric checks are not structural engineering"},
            "invariants": {"required_roles": sorted(self.form.required_roles)}})


def _bond(a, b, count, shifted):
    """Joints along one run; alternate courses break joint by half a block."""
    step = (b - a) / count
    if not shifted or count < 2:
        return [(a + step * k, a + step * (k + 1)) for k in range(count)]
    cuts = [a, a + step / 2] + [a + step / 2 + step * k for k in range(1, count)] + [b]
    return list(zip(cuts, cuts[1:]))


def taper(cx, cz, y0, y1, scale_x, scale_z):
    """Warp for a battered mass: plan shrinks linearly from y0 to y1 about (cx, cz)."""
    def warp(vertices):
        u = np.clip((vertices[:, 1] - y0) / (y1 - y0), 0.0, 1.0)
        out = vertices.copy()
        out[:, 0] = cx + (vertices[:, 0] - cx) * (1 + (scale_x - 1) * u)
        out[:, 2] = cz + (vertices[:, 2] - cz) * (1 + (scale_z - 1) * u)
        return out
    return warp


def flare(cx, cz, y0, y1, scale_x, scale_z):
    """Warp for a course that is wider at its head than at its foot.

    ``taper`` sets a box to the foot of the course and grows it toward the head, which can
    stretch a closing stone past a stretcher. Lay the box to the head instead and draw it in
    toward the foot: every stone only shrinks, and stays the size it was cut to.
    """
    def warp(vertices):
        u = np.clip((y1 - vertices[:, 1]) / (y1 - y0), 0.0, 1.0)      # 1 at the foot, 0 at the head
        out = vertices.copy()
        out[:, 0] = cx + (vertices[:, 0] - cx) * (1 - (1 - scale_x) * u)
        out[:, 2] = cz + (vertices[:, 2] - cz) * (1 - (1 - scale_z) * u)
        return out
    return warp


def batter(axis_index, sign, projection, top):
    """Warp for a wall whose foot leans out: full projection at ground, none at ``top``."""
    def warp(vertices):
        out = vertices.copy()
        out[:, axis_index] = vertices[:, axis_index] + sign * projection * np.clip(1 - vertices[:, 1] / top, 0, 1)
        return out
    return warp


def shear_y(axis_index, origin, slope):
    """Warp for a pitched slab: height rises with distance along one plan axis."""
    def warp(vertices):
        out = vertices.copy()
        out[:, 1] = vertices[:, 1] + (vertices[:, axis_index] - origin) * slope
        return out
    return warp


# ---------------------------------------------------------------- quality
def historic_checks(document):
    """Checks that hold from the geometry alone, after metadata is stripped."""
    from .quality import _check, _envelope_connectivity_checks
    found = style_for(document.request.product)
    if found is None:
        return []
    module, form = found
    domain = module.ARCHITECT.id
    placements = document.placements
    observed = {p.role for p in placements}
    checks = [_check(f"{domain}.required_roles", form.required_roles <= observed,
                     "Defining members are present.", "A defining member is missing.",
                     observed=sorted(observed), expected=sorted(form.required_roles))]
    checks.extend(_envelope_connectivity_checks(document, domain))
    invalid = [p.id for p in placements if module.KINDS.get(p.role) != p.kind]
    checks.append(_check(f"{domain}.member_semantics", not invalid,
                         "Every member has a recognized role and matching coarse kind.",
                         "Member semantics are inconsistent.", primitive_ids=invalid))
    for code, ok, passed, failed, ids in module.rule_checks(document, form):
        checks.append(_check(f"{domain}.{code}", ok, passed, failed, primitive_ids=ids))
    return checks


def extent(placements):
    lo = np.min([np.asarray(p.position) - np.asarray(p.dimensions) / 2 for p in placements], axis=0)
    hi = np.max([np.asarray(p.position) + np.asarray(p.dimensions) / 2 for p in placements], axis=0)
    return lo, hi


def unmirrored(placements, roles, axis=0, centre=0.0):
    """IDs of members in ``roles`` that have no mirror twin across a plan axis."""
    def key(p, flip):
        position = [float(v) for v in p.position]
        position[axis] -= centre
        if flip:
            position[axis] = -position[axis]
        position = [round(v, 3) for v in position]
        return (p.role, tuple(v + 0.0 for v in position), tuple(round(float(v), 3) for v in p.dimensions))
    chosen = [p for p in placements if p.role in roles]
    seen = {key(p, False) for p in chosen}
    return [p.id for p in chosen if key(p, True) not in seen]
