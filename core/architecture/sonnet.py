"""Architecture sonnets: a fixed form, and every building the seed can write in it.

A sonnet has fourteen lines and a rhyme scheme, and no two are alike. A building
tradition works the same way. Its rules are fixed: a stupa is circled by a
railing, a ziggurat recedes, an onion dome sits on a drum. Inside the rules a
builder still chooses. This module names those choices.

A form states its *lines* in ``Form.rules["sonnet"]``: each line is a named
decision with a short list of answers the tradition allows, the canonical one
first. The seed answers every line; the answers together are the building's
*verse*. A request may pin any line by name in its parameters and let the seed
write the rest, so "the same church, but with a tent roof" is one parameter.

The verse depends only on the request, never on what was built, so it can be
read without building. ``variations`` uses that to deal a set of buildings
whose verses are all different, for as long as the form has verses left.
"""
from __future__ import annotations

import math
import random

from .historic import TAU, Composer, Form, hull, number, roll_request, style_for
from .schema import canonical_product_name

SCHEMA = "blockn.sonnet/v1"
SIZES = ("width", "height", "depth")


def form(name, description, defaults, variants, required, *, lines, size, ratios=None, controls=None, **rules):
    """A catalog form whose envelope, proportions and lines are all stated as data.

    ``size`` maps width, height and depth to [low, high] metres. ``ratios`` maps
    "depth/width"-style names to [low, high]. ``controls`` maps bays or levels
    to (default, low, high). ``lines`` maps each line to its allowed answers.
    """
    controls = controls or {}
    stated = {**{key: list(span) for key, span in size.items()},
              **{key: list(span) for key, span in (ratios or {}).items()},
              **{key: [low, high] for key, (_, low, high) in controls.items()},
              "sonnet": {key: list(options) for key, options in lines.items()}, **rules}
    return Form(name, description, tuple(defaults), tuple(variants), frozenset(required),
                {key: default for key, (default, _, _) in controls.items()}, tuple(lines), stated)


def lines_of(product):
    found = style_for(product)
    if found is None:
        raise ValueError(f"{product} is not a historic form")
    return dict(found[1].rules.get("sonnet", {}))


def _ratios(rules):
    for key, span in rules.items():
        if "/" in key and all(part in SIZES for part in key.split("/")):
            yield key.split("/"), span


def roll(form, rng):
    """Sample a whole-metre envelope and controls the form's own ratios accept."""
    rules = form.rules
    for _ in range(400):
        size = {name: rng.randint(math.ceil(rules[name][0]), math.floor(rules[name][1])) for name in SIZES}
        if all(span[0] <= size[a] / size[b] <= span[1] for (a, b), span in _ratios(rules)):
            break
    else:
        size = dict(zip(SIZES, form.defaults))
    for name in form.controls:
        size[name] = rng.randint(*rules[name])
    return size


def verse(request):
    """Every line of the request's form and its answer, without building anything."""
    found = style_for(request.product)
    if found is None:
        raise ValueError(f"{request.product} is not a historic form")
    form = found[1]
    variation = request.parameters.get("variation", 1.0)
    answers = {}
    for name, options in form.rules.get("sonnet", {}).items():
        pinned = request.parameters.get(name)
        if pinned is not None:
            if pinned not in options:
                raise ValueError(f"{form.name} {name} must be one of {', '.join(options)}")
            answers[name] = pinned
        elif not variation:
            answers[name] = options[0]
        else:       # the same draw as Composer.pick, so a verse read here is the verse that is built
            answers[name] = random.Random(f"{request.seed}:{form.name}:{name}").choice(options)
    return answers


def stanza(request):
    """A request's variant and verse as one hashable signature."""
    found = style_for(request.product)
    variant = request.variant
    if variant == "auto":
        variant = found[1].variants[0]
    elif variant == "random":
        variant = random.Random(request.seed).choice(found[1].variants)
    return (variant, *verse(request).items())


def verse_count(product, *, variant=None, hold=None):
    """How many different verses a form can be written in."""
    form = style_for(product)[1]
    count = 1 if variant else len(form.variants)
    for name, options in form.rules.get("sonnet", {}).items():
        count *= 1 if name in (hold or {}) else len(options)
    return count


def variations(product, count, seed=0, *, variant=None, detail=None, hold=None):
    """``count`` requests on one theme, no verse repeated while the form has verses left.

    ``hold`` pins lines for the whole set: hold the dome and vary the rest.
    Seeds are taken in order from ``seed``, so the set can be dealt again.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 512:
        raise ValueError("count must be an integer from 1 to 512")
    hold = dict(hold or {})
    unknown = set(hold) - set(lines_of(product))
    if unknown:
        raise ValueError(f"{product} has no line named {', '.join(sorted(unknown))}")
    total = verse_count(product, variant=variant, hold=hold)
    seen, requests, tried = set(), [], seed
    while len(requests) < count:
        request = roll_request(product, tried, variant=variant, detail=detail)
        request.parameters.update(hold)
        tried += 1
        key = stanza(request)
        if key in seen and len(seen) < total and tried - seed < 64 * count:
            continue
        seen.add(key)
        request.parameters.update(verse(request))      # the verse is written out, so the recipe reads as built
        requests.append(request)
    return requests


def recipe(product, count=8, seed=0, *, variant=None, detail=None, hold=None, spacing=10, columns=None):
    """A kitbash recipe that sets the variations side by side, each named by its seed."""
    name = canonical_product_name(product)
    requests = variations(name, count, seed, variant=variant, detail=detail, hold=hold)
    return {"schema_version": "blockn.kitbash/v1", "spacing": spacing,
            "columns": columns or math.ceil(math.sqrt(count)),
            "buildings": [{"id": f"{name}-{request.seed:04d}", "request": request.to_dict()} for request in requests]}


def describe(product, *, variant=None, hold=None):
    """The form as data: its lines, their answers, and how many verses that makes."""
    name = canonical_product_name(product)
    module, form = style_for(name) or (None, None)
    if form is None:
        raise ValueError(f"{product} is not a historic form")
    return {"schema_version": SCHEMA, "product": name, "architect": module.ARCHITECT.id,
            "variants": list(form.variants), "lines": lines_of(name),
            "verses": verse_count(name, variant=variant, hold=hold),
            "also_seeded": "envelope, controls and continuous proportions vary inside every verse"}


def dome_profile(shape, radius, base, height, steps=10):
    """(height, radius) stations of a dome for ``Composer.lathe``; the crown closes to a point.

    ``hemisphere`` and ``saucer`` are arcs of one circle. ``pointed`` is struck
    from two centres. ``onion`` swells past its springing before it closes, and
    ``helmet`` is the older, lower Russian form that does not swell.
    """
    points = []
    for k in range(steps + 1):
        u = k / steps
        if shape in ("hemisphere", "saucer"):
            r = math.sqrt(max(0.0, 1 - u * u))
        elif shape == "pointed":
            r = (1 - u ** 1.7) ** .62
        elif shape == "helmet":
            r = (1 - u ** 1.25) ** .8 * (1 + .1 * math.sin(math.pi * min(1.0, u * 2.2)))
        elif shape == "onion":
            r = (1 + .34 * math.sin(math.pi * min(1.0, u / .62)) ** 1.2) * (1 - u ** 2.2) ** .9
        else:
            raise ValueError(f"unknown dome shape {shape}")
        points.append((base + u * height, max(radius * r, .03)))
    return points


class Poem(Composer):
    """Base composer of a sonnet form: checks the stated envelope, then answers the lines."""

    def __init__(self, request):
        super().__init__(request)
        rules = self.form.rules
        size = dict(zip(SIZES, (self.w, self.h, self.d)))
        for name in SIZES:
            number(size[name], f"{self.form.name} {name}", *rules[name])
        for (a, b), span in _ratios(rules):
            self.ratio(size[a] / size[b], f"{a}/{b}", *span)
        for name in self.form.controls:
            setattr(self, name, self.control(name, *rules[name]))
        self.verse = verse(request)

    def line(self, name):
        return self.verse[name]

    def fill(self, x0, x1, y0, y1, z0, z1, module=3.2, rise=1.6, warp=None, role="core_fill", **params):
        """The unseen mass of a platform or mound: rammed fill laid in lifts, in large blocks."""
        if min(x1 - x0, y1 - y0, z1 - z0) <= 1e-6:
            return
        lifts = max(1, math.ceil((y1 - y0) / rise - 1e-9))
        for j in range(lifts):
            self.slab(role, x0, x1, z0, z1, y0 + (y1 - y0) * j / lifts, y0 + (y1 - y0) * (j + 1) / lifts,
                      module=module, warp=warp, construction="rammed fill", unit="core", course=j, **params)

    def flight(self, role, width, foot, head, y0, y1, riser=.2, base=None, piece=1.5, **params):
        """A straight stair facing -Z: it climbs from z=``foot`` at y0 to z=``head`` at y1, solid beneath."""
        steps = max(1, math.ceil((y1 - y0) / riser - 1e-9))
        pieces = max(1, math.ceil(width / piece - 1e-9))
        base = y0 if base is None else base
        for j in range(steps):
            za, zb = foot + (head - foot) * j / steps, foot + (head - foot) * (j + 1) / steps
            for k in range(pieces):
                xa = -width / 2 + width * k / pieces
                self.member(role, xa, xa + width / pieces, base, y0 + (y1 - y0) * (j + 1) / steps,
                            za, zb, step=j, construction="solid stair", **params)
        return steps

    def podium(self, role, x0, x1, z0, z1, y0, y1, paving=None, thickness=.5, deck=.2, module=3.2, **params):
        """A faced platform: four ashlar skins, fill behind them, and an optional paved top."""
        top = y1 - (deck if paving else 0.)
        self.shell(role, x0, x1, z0, z1, y0, top, thickness, **params)
        self.fill(x0 + thickness, x1 - thickness, y0, top, z0 + thickness, z1 - thickness, module=module, **params)
        if paving:
            self.slab(paving, x0, x1, z0, z1, top, y1, module=min(module, 2.4), **params)

    def revolve(self, role, profile, thickness=.6, core=None, cx=0., cz=0., sides=20, **params):
        """A shell of revolution along (height, radius) stations: a dome, a drum, a conical tower.

        Where the shell is wide it is laid in rings of radial stones, with ``core``
        drums filling it if a core role is given. Where it closes it is solid.
        """
        for (ya, ra), (yb, rb) in zip(profile, profile[1:]):
            if min(ra, rb) > max(1.3, thickness * 1.8):
                self.ring(role, cx, cz, ra, thickness, ya, yb, top_radius=rb, **params)
                if core:
                    self.drum(core, cx, cz, ya, yb, ra - thickness, rb - thickness, sides,
                              construction="rammed fill", unit="core", **params)
            else:
                self.lathe(role, cx, cz, [(ya, ra), (yb, rb)], sides, **params)

    def conic(self, role, cx, cz, y0, y1, r0, r1, sides=12, arc=(0., TAU), rot=0., **params):
        """One tier of a cone, tent or pyramid roof in radial pieces; ``arc`` makes a half-cone."""
        host = self.host(role)
        for i in range(sides):
            a0, a1 = (rot + arc[0] + (arc[1] - arc[0]) * k / sides for k in (i, i + 1))
            points = [(cx, y0, cz), (cx, y1, cz)]
            for y, r in ((y0, r0), (y1, r1)):
                if r > 1e-6:
                    points += [(cx + r * math.cos(a), y, cz + r * math.sin(a)) for a in (a0, a1)]
            self.mesh(role, hull(points), host_id=host, **params)
        return host

    def document(self, rules):
        return super().document({"verse": dict(self.verse), **rules})


def controls_for(module):
    """``random_controls`` for a module whose forms state their envelope as data."""
    def random_controls(product, rng):
        return roll(module.FORMS[product], rng)
    return random_controls


__all__ = ["SCHEMA", "Poem", "controls_for", "describe", "dome_profile", "form", "lines_of", "recipe",
           "roll", "stanza", "variations", "verse", "verse_count"]
