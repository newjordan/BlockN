"""Historic architects sized to a player profile.

Without a profile the norman, classical and roman forms build their historic
sizes. With one they grow doors, stairs and passages so the player can walk
in; these tests pin both halves.
"""

import math
import unittest

from core.architecture import BuildRequest, build
from core.architecture.profile import preset
from core.architecture.service import author_document

GIANT = preset("giant")


def doc(product, profile=None, **extra):
    request = {"product": product, "seed": 1, **extra}
    if profile:
        request["profile"] = profile
    return author_document(request)


def registers(document, name):
    return [p for p in document.placements if p.parameters.get("register") == name]


def top(p):
    return p.position[1] + p.dimensions[1] / 2


def bottom(p):
    return p.position[1] - p.dimensions[1] / 2


class TestNormanDoors(unittest.TestCase):
    def test_no_profile_keeps_the_historic_keep(self):
        plain = doc("norman-keep", variant="square")
        self.assertFalse(registers(plain, "vestibule floor"))
        self.assertFalse(registers(plain, "door sill"))

    def test_square_keep_gets_a_floor_behind_its_door(self):
        giant = doc("norman-keep", "giant", variant="square")
        floor = registers(giant, "vestibule floor")
        sill = registers(giant, "door sill")
        self.assertTrue(floor)
        self.assertTrue(sill)
        # The sill is as wide as the giant's door.
        width = max(p.position[0] + p.dimensions[0] / 2 for p in sill) - min(
            p.position[0] - p.dimensions[0] / 2 for p in sill)
        self.assertGreaterEqual(width + 1e-6, GIANT.clearances().door_width)

    def test_forebuilding_stair_is_door_wide(self):
        plain = doc("norman-keep", variant="square")
        giant = doc("norman-keep", "giant", variant="square")
        def width(d):
            stair = [p for p in d.placements if p.role == "stair"
                     and "newel" not in str(p.parameters.get("construction"))]
            return (max(p.position[2] + p.dimensions[2] / 2 for p in stair)
                    - min(p.position[2] - p.dimensions[2] / 2 for p in stair))
        self.assertGreaterEqual(width(giant) + 1e-6, GIANT.clearances().door_width)
        self.assertGreaterEqual(width(giant), width(plain))

    def test_round_keep_bridge_and_stair_are_door_wide(self):
        giant = doc("norman-keep", "giant", variant="round")
        bridge = [p for p in giant.placements if p.role == "bridge"]
        self.assertTrue(bridge)
        self.assertGreaterEqual(max(p.dimensions[0] for p in bridge) + 1e-6, GIANT.clearances().door_width)

    def test_portcullis_is_drawn_up_for_a_player(self):
        plain = doc("norman-shell-keep")
        giant = doc("norman-shell-keep", "giant")
        foot = lambda d: min(bottom(p) for p in d.placements if p.role == "portcullis")
        floor = lambda d: min(bottom(p) for p in d.placements if p.role == "gate_passage")
        self.assertLess(foot(plain) - floor(plain), GIANT.clearances().door_height)
        self.assertGreaterEqual(foot(giant) - floor(giant) + 1e-6, GIANT.clearances().door_height)

    def test_giant_walks_into_a_keep(self):
        result = build(BuildRequest(product="norman-keep", seed=1, profile="giant", detail="concept"))
        codes = {c.code: c.status for c in result.quality.checks}
        self.assertEqual(codes["game.entry"], "pass")
        walk = result.quality.metrics["playability"]["walk"]
        self.assertGreater(walk["interior_reached_walking"], 0)

    def test_plain_keep_has_no_inner_stair(self):
        plain = doc("norman-keep", variant="square")
        self.assertFalse([p for p in plain.placements if "newel" in str(p.parameters.get("construction"))])

    def test_castle_keep_grows_to_fit_its_stairs(self):
        # The castle's keep is smaller than a lone keep; for a big body it
        # grows until a newel stair fits every storey, and stays put for a
        # person. Three storeys, then a last flight up through the roof to
        # the wall-walk.
        def newels(document):
            return {p.parameters.get("level") for p in document.placements
                    if p.parameters.get("construction") == "stone newel step"}

        for name in ("giant", "brute", "human"):
            with self.subTest(profile=name):
                self.assertEqual(len(newels(doc("norman-castle", name, detail="concept"))), 4)

    def test_square_keep_risers_fit_the_step(self):
        for name in ("giant", "brute", "human"):
            player = preset(name)
            steps = [p for p in doc("norman-keep", name, variant="square").placements
                     if p.parameters.get("construction") == "stone newel step"]
            self.assertTrue(steps, name)
            by_level = {}
            for p in steps:
                by_level.setdefault(p.parameters["level"], []).append(top(p))
            for level, tops in by_level.items():
                tops.sort()
                risers = [b - a for a, b in zip(tops, tops[1:])]
                self.assertLessEqual(max(risers), player.step_height + 1e-6, (name, level))

    def test_giant_walks_to_every_floor_of_the_square_keep(self):
        import numpy as np
        from core.architecture.gameplay import walk
        from core.architecture.voxels import profile_pitch, voxelize

        result = build(BuildRequest(product="norman-keep", variant="square", seed=1, profile="giant",
                                    detail="standard"))
        grid = voxelize(result.document, result.scene, pitch=profile_pitch(GIANT), profile=GIANT)
        w = walk(grid, GIANT)
        floors = sorted({round(top(p), 2) for p in result.document.placements
                         if p.parameters.get("construction") == "oak boards"})
        heights = grid.origin[1] + w.nodes[:, 1] * grid.pitch
        for y in floors:
            here = (np.abs(heights - y) < .2) & w.interior
            share = (here & w.reached_walking).sum() / max(1, here.sum())
            self.assertGreater(share, .9, f"floor at {y} m")

    def test_giant_walks_the_round_keep_to_its_wall_walk(self):
        # Basement, every board floor, and on up through the roof hatch to
        # the wall-walk round the top of the drum, all on foot.
        import numpy as np
        from core.architecture.gameplay import walk
        from core.architecture.voxels import profile_pitch, voxelize

        result = build(BuildRequest(product="norman-keep", variant="round", seed=1, profile="giant",
                                    detail="standard"))
        grid = voxelize(result.document, result.scene, pitch=profile_pitch(GIANT), profile=GIANT)
        w = walk(grid, GIANT)
        heights = grid.origin[1] + w.nodes[:, 1] * grid.pitch
        floors = sorted({round(top(p), 2) for p in result.document.placements
                         if p.parameters.get("construction") == "oak boards"})
        self.assertGreaterEqual(len(floors), 3)
        for y in floors:
            here = (np.abs(heights - y) < .2) & w.interior
            if here.sum() > 200:
                share = (here & w.reached_walking).sum() / here.sum()
                self.assertGreater(share, .9, f"floor at {y} m")
        walk_top = max(top(p) for p in result.document.placements if p.role == "wall_walk")
        deck = np.abs(heights - walk_top) < .2
        self.assertGreater(deck.sum(), 500)
        self.assertGreater((deck & w.reached_walking).sum() / deck.sum(), .9)
        self.assertTrue(any(p.role == "keep_roof" for p in result.document.placements))

    def test_sized_cavea_is_cut_into_seat_rows(self):
        # Each tier becomes rows of seats, and every row meets the aisle at
        # the height of one of its steps, two steps to a row.
        def tops(document, role):
            return {round(top(p), 3) for p in document.placements if p.role == role}

        plain = doc("classical-theatre")
        giant = doc("classical-theatre", "giant")
        self.assertGreater(len(tops(giant, "seating_tier")), len(tops(plain, "seating_tier")))
        self.assertLessEqual(tops(giant, "seating_tier"), tops(giant, "aisle_step"))


    def test_baths_chambers_get_doorways(self):
        # Every build closes the ends of the chamber ranges with a door; only
        # a player's build also opens the walls between the chambers, which
        # stand near the middle of each range (|z| < 4 m, |x| > 6 m).
        def jambs(document):
            by = {p.id: p for p in document.placements}
            hosts = {p.parameters.get("host_id") for p in document.placements
                     if p.role == "hall_wall" and p.parameters.get("unit") == "jamb"}
            return [by[h].position for h in hosts if h in by]

        def between(document):
            return [q for q in jambs(document) if abs(q[2]) < 4 and abs(q[0]) > 6]

        plain = doc("classical-baths")
        giant = doc("classical-baths", "giant")
        self.assertTrue([q for q in jambs(plain) if abs(q[2]) > 9])
        self.assertFalse(between(plain))
        self.assertTrue(between(giant))

    def test_giant_reaches_the_whole_baths(self):
        result = build(BuildRequest(product="classical-baths", seed=1, profile="giant"))
        codes = {c.code: c.status for c in result.quality.checks}
        self.assertEqual(codes["game.entry"], "pass")
        self.assertEqual(codes["game.interior_reach"], "pass")

    def _risers(self, document):
        steps = {}
        for p in document.placements:
            if p.role == "aisle_step":
                key = (p.parameters["tier"], p.parameters["aisle"], p.parameters["half"])
                steps[key] = max(steps.get(key, 0.0), top(p))
        risers = []
        for (tier, aisle, half), y in steps.items():
            below = steps.get((tier, aisle, half - 1))
            if below is not None:
                risers.append(y - below)
        return risers

    def test_theatre_aisle_steps_fit_the_step(self):
        plain = self._risers(doc("classical-theatre"))
        giant = self._risers(doc("classical-theatre", "giant"))
        self.assertGreater(max(plain), GIANT.step_height)
        self.assertLessEqual(max(giant), GIANT.step_height + 1e-6)

    def test_amphitheatre_opens_two_gates(self):
        half = GIANT.clearances().door_width / 2

        def in_gate(p):
            # Mesh vertices are stored in the placement's unit box.
            for v in p.parameters.get("vertices") or []:
                x = p.position[0] + v[0] * p.dimensions[0]
                z = p.position[2] + v[2] * p.dimensions[2]
                if abs(x) > 1.0 and abs(z) < half - 1e-6:
                    return True
            return False

        giant = [p for p in doc("classical-amphitheatre", "giant").placements if p.role == "seating_tier"]
        plain = [p for p in doc("classical-amphitheatre").placements if p.role == "seating_tier"]
        self.assertTrue(giant)
        self.assertFalse([p.id for p in giant if in_gate(p)])
        self.assertTrue([p.id for p in plain if in_gate(p)])

if __name__ == "__main__":
    unittest.main()
