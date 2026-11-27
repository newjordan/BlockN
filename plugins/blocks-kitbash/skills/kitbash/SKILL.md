---
name: kitbash
description: Build and export deterministic architectural kits with Blocks. Use when asked to generate buildings, classical architecture, Gothic cathedrals, modular environments, or a kitbashed series as GLB, OBJ, STL, or PLY assets.
---

# Blocks kitbash

Turn the user's building request into a bounded recipe, run the real geometry
engine, and return usable files. No provider credentials or desktop are needed.

## Locate and inspect

Use `python <plugin>/scripts/blocks.py` as the command prefix. Resolve `<plugin>`
from this skill's location (two directories above this file's folder). Run
`doctor` first. If the runtime is elsewhere, set `BLOCKS_ROOT` to that source
folder, or use an installed `blockn` command. Follow README.md to install missing
dependencies; do not silently modify the user's global Python environment.

Run `craft`, `catalog --json`, `describe <product>`, and `agent schema` to discover the
current products and accepted fields. The included version supports classical orders,
Gothic cathedral styles and Timberwright house plans. Only
request fields and variants actually supported by `describe`.

## Build

For an unspecified building set, choose a count of 6 and seed 73, then run:

```bash
python <plugin>/scripts/blocks.py agent series --kitbash --count 6 --seed 73 --output ./blocks-kit-73 --format glb
```

For a specific composition, write `blockn.kitbash/v1` JSON using `agent schema`.
Each building has an ID and exactly one of `request` or `parts`. Parts have an
ID, request and optional translation offset. The source bundle's
`examples/cathedral-kitbash.json`, `examples/classical-kitbash.json`, and `examples/timber-street.json` are ready
recipes. Run the recipe with repeated format flags:

```bash
python <plugin>/scripts/blocks.py agent build recipe.json --output ./blocks-custom-73 --format glb --format obj
```

Use a fresh output directory; existing directories are refused. Resolve input
and output paths relative to the user's working directory. Keep IDs filename
safe. Limit jobs to 24 buildings, 8 parts per building and 30,000 placements.
Use translations for assembly; this schema does not implement part rotation.
Separate buildings belong in separate entries; parts within a building must
connect. Prefer GLB for named meshes and scene interchange. Always keep the
semantic JSON and quality reports for replay.

Cathedrals support `high-gothic`, `early-english` and `rayonnant`. Query the
product before choosing proportions or style-specific parameters. Do not
promise every historical style or structural engineering certification.

## Verify and deliver

Check the process exit code and parse the JSON receipt. Success requires
`status: pass`; resolve all artifact paths against the receipt's output folder
and verify they exist. Return the combined scene, individual building folder,
recipe, construction report and batch manifest. Construction reports group every
member into assembly stages and expose measured order/bearing checks. These
are architectural geometry diagnostics, not structural capacity calculations.
Include the seed and quality result. Use direct file
links when the client supports them. Never claim a file was exported after an
error. Exit 1 means failed quality; 2 means invalid input or IO failure. Usage
errors go to stderr. Avoid decorating or mixing text into JSON output.

For a visual preview, launch `blockn-gui` after installing the GUI extra. The
app uses black and green wireframes; exported meshes contain actual solid
geometry. Roof inspection changes the view while preserving complete exports.
