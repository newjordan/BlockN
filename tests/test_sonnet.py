"""The sonnet system: lines, verses and dealt variations on one form."""
import json
import subprocess
import sys
import unittest

from core.architecture import BuildRequest, build, describe_product, roll_request, sonnet
from core.architecture.historic import styles

PRODUCT = "mesopotamian-ziggurat"


class SonnetTests(unittest.TestCase):
    def test_every_sonnet_form_states_its_lines_and_envelope_as_data(self):
        forms = {name: form for name, (_, form) in styles().items() if form.rules.get("sonnet")}
        self.assertGreaterEqual(len(forms), 9)
        for name, form in forms.items():
            with self.subTest(product=name):
                described = describe_product(name)
                self.assertEqual(described["rules"]["sonnet"], sonnet.lines_of(name))
                self.assertLessEqual(set(sonnet.lines_of(name)), set(described["parameters"]))
                for options in sonnet.lines_of(name).values():
                    self.assertGreaterEqual(len(options), 2)
                    self.assertTrue(all(isinstance(option, str) for option in options))
                self.assertGreaterEqual(sonnet.verse_count(name), 16)
                for axis, default in zip(("width", "height", "depth"), form.defaults):
                    low, high = form.rules[axis]
                    self.assertTrue(low <= default <= high)

    def test_the_verse_is_read_without_building_and_is_what_gets_built(self):
        request = roll_request(PRODUCT, 11, detail="concept")
        verse = sonnet.verse(request)
        self.assertEqual(set(verse), set(sonnet.lines_of(PRODUCT)))
        document = build(request).document
        self.assertEqual(document.metadata["mesopotamian_rules"]["verse"], verse)
        canonical = BuildRequest(product=PRODUCT, seed=11, parameters={"variation": 0})
        self.assertEqual(sonnet.verse(canonical), {line: options[0] for line, options in sonnet.lines_of(PRODUCT).items()})

    def test_a_pinned_line_changes_only_that_line(self):
        request = roll_request(PRODUCT, 5, detail="concept")
        free = sonnet.verse(request)
        other = next(o for o in sonnet.lines_of(PRODUCT)["shrine"] if o != free["shrine"])
        request.parameters["shrine"] = other
        self.assertEqual(sonnet.verse(request), {**free, "shrine": other})
        result = build(request)
        self.assertEqual(result.quality.status, "pass")
        cells = {p.parameters.get("opening_id") for p in result.document.placements if p.role == "shrine_wall"}
        self.assertEqual("shrine.1" in cells, other == "twin-cella")
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, parameters={"shrine": "pagoda"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, parameters={"minarets": "four"}))

    def test_variations_do_not_repeat_a_verse_and_can_be_dealt_again(self):
        dealt = sonnet.variations(PRODUCT, 24, seed=3, detail="concept")
        self.assertEqual(len({sonnet.stanza(r) for r in dealt}), 24)
        self.assertEqual([r.to_dict() for r in dealt],
                         [r.to_dict() for r in sonnet.variations(PRODUCT, 24, seed=3, detail="concept")])
        # The whole verse is written into the request, so the recipe reads as built.
        self.assertTrue(all(set(sonnet.lines_of(PRODUCT)) <= set(r.parameters) for r in dealt))
        held = sonnet.variations(PRODUCT, 12, variant="axial-stair", hold={"crown": "plain"})
        self.assertEqual({(r.variant, r.parameters["crown"]) for r in held}, {("axial-stair", "plain")})
        self.assertEqual(len({sonnet.stanza(r) for r in held}), 12)
        # A form runs out of verses before it runs out of seeds; then verses repeat with new proportions.
        every = sonnet.variations(PRODUCT, 60)
        self.assertEqual(len({sonnet.stanza(r) for r in every}), sonnet.verse_count(PRODUCT))
        self.assertEqual(len({json.dumps(r.to_dict(), sort_keys=True) for r in every}), 60)
        with self.assertRaises(ValueError):
            sonnet.variations(PRODUCT, 4, hold={"spire": "needle"})
        with self.assertRaises(ValueError):
            sonnet.variations(PRODUCT, 0)

    def test_forms_without_lines_still_vary_by_variant(self):
        self.assertEqual(sonnet.describe("norman-keep")["verses"], 2)
        self.assertEqual({r.variant for r in sonnet.variations("norman-keep", 2)}, {"square", "round"})

    def test_every_rolled_envelope_composes(self):
        for name, (_, form) in styles().items():
            if not form.rules.get("sonnet"):
                continue
            for seed in range(40):
                with self.subTest(product=name, seed=seed):
                    request = roll_request(name, seed)
                    for axis in ("width", "height", "depth"):
                        low, high = form.rules[axis]
                        self.assertTrue(low <= getattr(request, axis) <= high)
                    sonnet.verse(request)

    def test_cli_prints_a_recipe_the_agent_can_build(self):
        def run(*args):
            done = subprocess.run([sys.executable, "-m", "blockn", "sonnet", *args], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads(done.stdout)
        form = run(PRODUCT, "--describe", "--hold", "crown=plain")
        self.assertEqual(form["verses"], sonnet.verse_count(PRODUCT, hold={"crown": "plain"}))
        recipe = run(PRODUCT, "--count", "3", "--seed", "9", "--detail", "concept", "--hold", "crown=plain")
        self.assertEqual(recipe, sonnet.recipe(PRODUCT, 3, 9, detail="concept", hold={"crown": "plain"}))
        self.assertEqual(recipe["schema_version"], "blockn.kitbash/v1")
        self.assertEqual(len({b["id"] for b in recipe["buildings"]}), 3)


if __name__ == "__main__":
    unittest.main()
