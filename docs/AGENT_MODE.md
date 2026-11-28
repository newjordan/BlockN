# Agent mode: request a kitbashed building series

Agent mode is a deterministic JSON-in/files-out interface. An agent translates
a user's request into catalog requests or a kitbash recipe; BlockN does not
need an LLM provider, desktop session, or prompt parser to execute it.

```bash
python3 -m blockn craft
python3 -m blockn agent schema
python3 -m blockn catalog --json
python3 -m blockn describe classical-villa
python3 -m blockn agent series --kitbash --count 6 --seed 73 --output builds/set-73
python3 -m blockn agent build examples/classical-kitbash.json --output builds/kit-73
cat recipe.json | python3 -m blockn agent build - --output builds/from-stdin
```

`--kitbash` adds a fitted podium and approach stair to every generated building,
using measured bounds for placement. Omit it for standalone catalog buildings.

`series` cycles through six default classical products and varies proportions
and supported variants with a local seeded RNG. Use `--products` with a
comma-separated catalog list to choose a different repertoire. The exact same
request and architect version produce the same recipe and geometry. Save the
emitted `recipe.json` to replay or edit individual buildings.

Historic products use their own rule-specific sampler (not just uniform
scaling). Sample or pin individual variants with `blockn roll PRODUCT --seed N
--variant NAME`, or call `core.architecture.roll_request(product, seed)` from
Python. Preview a batch without exporting geometry:

```bash
python3 -m blockn agent series --products swahili-courtyard-house,khmer-temple-mountain \
    --count 2 --seed 42 --detail concept --recipe-only > variation.json
python3 -m blockn agent build variation.json --output builds/variation
```

For many variations on one product, `blockn sonnet PRODUCT --count N` deals
requests whose verses do not repeat and prints them as a recipe; `--hold
LINE=ANSWER` pins a line across the set. See [architecture
sonnets](ARCHITECTURE_SONNETS.md).

`--recipe-only` is an editable recipe, not a quality receipt. The geometry and
placement budget are checked when building; `--kitbash` additionally measures
the buildings when sampling to size their podiums.

`build` accepts `blockn.kitbash/v1`. Each building has a filename-safe `id` and
**exactly one** of `request` (one catalog building) or `parts` (a kitbash).
Each part has its own `id`, a normal build `request`, and an optional `[x,y,z]`
`offset`. Set dimensions in the part request. This version supports translation;
it does not silently approximate unsupported rotation or nonuniform scaling.
The [example recipe](../examples/classical-kitbash.json) combines a podium,
elevated rotunda, and approach stair, alongside two other buildings.

Each source part must pass its own quality audit. The assembled building must
also pass mesh and document checks, and all its part envelopes must connect.
Part-envelope connectivity is conservative, not a physical load-path test.
Detached kitbash parts are rejected; separate buildings belong in separate
building entries. Recompiling a kitbashed document rechecks connectivity.

The combined series uses measured mesh bounds for a grid layout, preserving a
positive gap between buildings. `columns` and `spacing` control that layout.
Individual exports retain their local coordinate systems. The series document
records each building's translation and original fingerprint.

## Output contract

A successful command prints one JSON receipt with `status`, `count`, `output`,
`manifest`, per-building artifact paths and fingerprints, and combined series
artifact paths. Paths within the manifest are relative to its directory.

```text
set-73/
  recipe.json
  batch.manifest.json
  series.glb
  series.architecture.json
  series.quality.json
  series.manifest.json
  buildings/
    building-001.glb
    building-001.architecture.json
    building-001.quality.json
    building-001.manifest.json
    ...
```

Repeated `--format glb --format obj` flags select mesh formats. GLB keeps named
members, and the semantic JSON keeps part IDs and parent links. Exports contain
usable solid geometry; the black/green wireframe is the workbench presentation.

All requested models are built and audited before files are published. Files
are staged beside the destination, then the complete directory is renamed
into place. Existing output directories are refused. Use a new destination for
each run; do not run two writers against the same destination concurrently.
A runtime failure prints a JSON error and returns nonzero: 1 for failed quality,
2 for invalid input or IO errors. Parser/usage errors use normal argparse stderr.

Limits: 1–24 buildings, 1–8 parts per building, 30,000 placements, dimensions at
most 200, and bounded translation offsets. Unknown recipe and request fields,
duplicate IDs, path-like IDs, and unsupported schema versions are rejected.
These are workload guardrails, not a sandbox for hostile requests.

## Appearance

The desktop uses black backgrounds, green phosphor wireframes, green monospace
controls, and a green intensity palette. Model framing excludes the decorative
grid so generated geometry is immediately visible. Agent output remains plain
JSON for automation; it does not mix terminal decoration into machine output.

The [historic architects](HISTORIC_ARCHITECTS.md) can be mixed in a batch.
The [world architecture recipe](../examples/world-architecture.json) builds eight
examples at concept detail to stay within the 30,000-placement budget. The
[Byzantine plans recipe](../examples/byzantine-plans.json) compares four
cross-in-square churches on identical plots in a separate batch. Filled wall
cores make the Egyptian temple and the walled castle best built on their own.
Cathedral series use `--products gothic-cathedral`. See the
[cathedral recipe](../examples/cathedral-kitbash.json) for explicit styles and
[the cathedral guide](CATHEDRAL_ARCHITECT.md) for proportions and member roles.

## Construction reports

Every bundle now includes `<name>.construction.json`: craft traditions, complete
member IDs grouped by assembly stage, and measured order/bearing checks. Run
`python3 -m blockn craft` for the reusable vocabulary spanning Greek orders,
Roman masonry, later classical orders and Gothic work. Reports explicitly list
loads, material capacity and stability as unevaluated; they support architectural
asset work rather than approval of real construction.
