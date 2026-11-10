# LegoGen (Public Build)

This repository is the stripped, shareable production branch of LegoGen.
It keeps the stable classical generator, GUI viewer, and export pipeline, and excludes in-progress planning/session artifacts.

## New: Historic Architects (development build)

![Castles, temples, pyramids and pagodas assembling stone by stone](public/historic_architects_preview.gif)

Four new architects join the development build of Blocks. Each one composes a
building from the rules of its own tradition, lays the masonry as lift-able
stones, and is then checked against rules measured from the finished geometry.

- **Castle**: Norman great towers (square or round), a shell keep on its motte, and a walled castle with rectangular or drum towers. Keeps are entered at the first floor and stand on a battered plinth.
- **Egyptian**: an axial cult temple with a battered pylon, colonnaded court, clerestoried hypostyle hall and sanctuary. Every span is a single stone beam or slab; nothing is arched.
- **Mesoamerican**: stepped pyramids in three traditions. The radial form has four stairs of exactly 91 steps, which with the platform count 365.
- **Pagoda**: a timber frame of diminishing storeys with bracketed deep eaves around a free central pillar.

![Norman castle with drum towers](public/historic_norman_castle_drum_towers.png)
![Norman great tower](public/historic_norman_keep.png)
![Egyptian temple](public/historic_egyptian_temple.png)
![Egyptian temple with the roof slabs lifted](public/historic_egyptian_temple_interior.png)
![Calendar pyramid](public/historic_calendar_pyramid.png)
![Five-storey timber pagoda](public/historic_timber_pagoda.png)
![All twelve variants](public/historic_architects_all.png)

These are original compositions built from each tradition's rules, not
replicas of named monuments. Like the cathedral below, these architects are
not yet part of this public branch.

## Latest Renders: Blocks Architecture System (development build)

These images come from the current development build of Blocks, the semantic
architecture system that grew out of LegoGen. Every member is a named,
human-scale stone, and each render shows real generated geometry in the
green phosphor style of the workbench. The Gothic Cathedral Architect, the
expanded classical forms, and the `blockn` CLI shown here are not yet part of
this public branch.

![Gothic cathedral, west front: paired spires, rose window, flying buttresses](public/gothic_cathedral_west.png)
![Gothic cathedral, east end: chevet with ambulatory, clerestoried hemicycle and radiating buttresses](public/gothic_cathedral_chevet.png)
![Gothic cathedral, side elevation](public/gothic_cathedral_side.png)
![One architect, three styles: High Gothic, Early English, Rayonnant](public/gothic_cathedral_styles.png)
![Desktop workbench with a generated cathedral](public/gui_cathedral_workbench.png)
![Workbench: open roof to inspect the vaults](public/gui_cathedral_vaults.png)
![Classical forms: temple, rotunda, basilica, aqueduct, amphitheatre, triumphal arch](public/classical_forms.png)

## Screenshots (this branch)

![Doric Canonical — large build](public/gui_canonical_large.png)
![Doric Monumental](public/gui_monumental.png)
![Doric Canonical — front view](public/gui_canonical_front.png)
![Doric Canonical — angled view](public/gui_canonical_angled.png)
![Classical — custom columns](public/gui_custom_columns.png)

## What Is Included

- PyQt5 + PyVista desktop GUI (`main.py`)
- Headless CLI generation/export (`cli.py`)
- Classical generator with preset/override controls
- GLB, JSON, and PNG export paths
- Unit tests for geometry, scene, and classical generation

## Quick Start

```bash
python3 -m venv .venv-public
source .venv-public/bin/activate  # Windows: .venv-public\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python3 main.py
```

## Environment Separation

Keep this repo independent from your development repo:

- Public repo venv: `.venv-public`
- Dev repo venv: use a different folder name (for example `.venv-dev`)
- Do not share `.env` files between the two repos

This avoids dependency and config drift between production and experimental work.

## CLI Examples

```bash
python3 cli.py generate --classical-order Doric --seed 123 --out builds/classical.glb --summary
python3 cli.py generate --classical-order Ionic --classical-preset monumental --classical-front-columns 8 --classical-side-columns 8 --classical-pediment-slope 1.1 --classical-entablature-depth 1.2 --classical-side-roof-overhang 1.25 --classical-foundation-apron --width 24 --depth 24 --summary
python3 cli.py generate --classical-order Doric --classical-preset canonical --classical-wrap-min-front-span 22 --classical-wrap-min-side-span 14 --classical-wrap-front-coverage 0.60 --width 20 --height 12 --depth 14 --summary
```

## Tests

```bash
python3 -m unittest tests.test_geometry -v
python3 -m unittest tests.test_scene -v
python3 -m unittest tests.test_classical_generator -v
python3 test_imports.py
```

## License

MIT. See [LICENSE](LICENSE).
