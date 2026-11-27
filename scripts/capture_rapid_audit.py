"""Offscreen phosphor captures of current classical / masonry / marble-run builds.

Uses the same VTK feature-edge language as the desktop viewport. Qt/xcb is not
required; PYVISTA_OFF_SCREEN is.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("DISPLAY", None)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageDraw, ImageFont

from core.architecture import BuildRequest, build
from core.architecture.marble_run import point_on_loop
from gui.camera import frame_cathedral

OUT = ROOT / "public" / "rapid-audit"
SHOTS = [
    ("wall-ashlar", {"product": "wall", "openings": "mixed", "detail": "presentation", "width": 12, "height": 3.6, "depth": 0.4, "seed": 73},
     (-1.6, 0.35, 1.1), 1.08, None),
    ("wall-close", {"product": "wall", "openings": "mixed", "detail": "presentation", "width": 12, "height": 3.6, "depth": 0.4, "seed": 73},
     (0.15, 0.12, 1.0), 0.42, "door"),
    ("room-shell", {"product": "room-shell", "openings": "mixed", "detail": "standard", "seed": 73},
     (-1.9, 0.7, 1.6), 1.12, None),
    ("doric-temple", {"product": "classical-building", "variant": "doric", "detail": "standard", "seed": 73},
     (-1.8, 0.55, 2.1), 1.10, None),
    ("stoa-rear", {"product": "classical-stoa", "variant": "ionic", "detail": "standard", "seed": 73},
     (-0.4, 0.45, 2.2), 1.08, None),
    ("aqueduct", {"product": "classical-aqueduct", "bays": 6, "levels": 2, "detail": "standard", "seed": 73},
     (-2.2, 0.35, 0.9), 1.12, None),
    ("basilica", {"product": "classical-basilica", "variant": "doric", "detail": "concept", "seed": 73},
     (-1.7, 0.6, 2.0), 1.10, None),
    ("marble-run", {"product": "marble-run", "bays": 6, "levels": 1, "detail": "concept", "seed": 73},
     (-2.0, 0.85, 1.6), 1.12, None),
    ("marble-chase", {"product": "marble-run", "bays": 6, "levels": 1, "detail": "concept", "seed": 73},
     (-0.6, 0.25, 1.0), 0.62, "chase"),
    ("portico", {"product": "classical-portico", "variant": "corinthian", "detail": "standard", "seed": 73},
     (-1.8, 0.55, 1.8), 1.10, None),
    ("theatre", {"product": "classical-theatre", "detail": "concept", "seed": 73},
     (-1.6, 0.9, 1.5), 1.12, None),
    ("baths", {"product": "classical-baths", "detail": "concept", "seed": 73},
     (-2.0, 0.7, 1.7), 1.12, None),
    ("amphitheatre", {"product": "classical-amphitheatre", "detail": "concept", "seed": 73},
     (-1.8, 1.0, 1.6), 1.12, None),
    ("villa", {"product": "classical-villa", "variant": "doric", "detail": "concept", "seed": 73},
     (-1.7, 0.55, 1.8), 1.10, None),
    ("triumphal-arch", {"product": "classical-triumphal-arch", "detail": "standard", "seed": 73},
     (-1.4, 0.4, 1.6), 1.08, None),
    ("cathedral", {"product": "gothic-cathedral", "detail": "concept", "width": 32, "height": 40, "depth": 52, "seed": 73},
     (-1.8, 0.7, 2.0), 1.12, None),
]


def phosphor_plotter(blocks, extra_meshes=(), marble_meshes=()):
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.set_background("#000000")
    meshes = [block.mesh for block in blocks if block.mesh is not None and not block.mesh.is_empty]
    meshes.extend(extra_meshes)
    vertices = np.zeros((1, 3))
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        poly = pv.wrap(combined)
        if not isinstance(poly, pv.PolyData):
            poly = poly.extract_surface()
        faces = plotter.add_mesh(poly, color="black", lighting=False, show_edges=False, backface_culling=True)
        faces.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        edges = poly.extract_feature_edges(feature_angle=26, boundary_edges=True,
                                           non_manifold_edges=True, manifold_edges=False)
        if edges.n_points:
            plotter.add_mesh(edges, color="#00ff41", line_width=1, lighting=False)
        vertices = combined.vertices
    for marble in marble_meshes:
        poly = pv.wrap(marble)
        if not isinstance(poly, pv.PolyData):
            poly = poly.extract_surface()
        face = plotter.add_mesh(poly, color="black", lighting=False, show_edges=True,
                                edge_color="#7dff7d", line_width=2, backface_culling=True)
        face.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        try:
            rider = poly.extract_all_edges()
        except Exception:
            rider = poly.extract_feature_edges(feature_angle=1.0, boundary_edges=True,
                                               non_manifold_edges=True, manifold_edges=True)
        if rider.n_points:
            plotter.add_mesh(rider, color="#7dff7d", line_width=5, lighting=False)
        vertices = np.vstack((vertices, np.asarray(marble.vertices)))
    return plotter, vertices


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
    receipt = {"source": "offscreen VTK phosphor (same feature-edge language as the desktop viewport)", "shots": {}}
    thumbs = []
    for name, request, direction, margin, close in SHOTS:
        started = time.perf_counter()
        result = build(BuildRequest(**request))
        elapsed = time.perf_counter() - started
        fails = [c.to_dict() for c in result.quality.checks if c.status == "fail"]
        record = {
            "quality": result.quality.status,
            "members": len(result.scene.blocks),
            "seconds": round(elapsed, 3),
            "fails": fails,
        }
        extra = []
        chase_points = None
        if name in {"marble-run", "marble-chase"}:
            run = result.document.metadata["marble_run"]
            path = np.asarray(run["path"])
            distance = run["length"] * (0.18 if name == "marble-run" else 0.31)
            position, tangent = point_on_loop(path, run["length"], distance)
            marble = trimesh.creation.icosphere(
                subdivisions=0, radius=float(run["marble_radius"]) * 1.35)
            marble.apply_translation(position)
            extra.append(marble)
            if close == "chase":
                chase_points = np.vstack((
                    marble.vertices,
                    path[max(0, int(0.31 * len(path)) - 8): int(0.31 * len(path)) + 12],
                ))
        blocks = result.scene.blocks
        if close == "door":
            host = [b for b in blocks if any(token in b.parameters.get("component_role", "")
                                             for token in ("header", "sill", "ashlar"))]
            points = np.vstack([b.mesh.vertices for b in host[:40]])
            plotter, _ = phosphor_plotter(host[:80])
        else:
            architecture = extra[:-1] if extra else extra
            rider = extra[-1:] if extra else []
            plotter, points = phosphor_plotter(
                [b for b in blocks if b.parameters.get("component_role") != "marble"],
                architecture, rider,
            )
            if chase_points is not None:
                points = chase_points
        frame_cathedral(plotter, points, direction, margin=margin)
        image = Image.fromarray(plotter.screenshot(return_img=True))
        plotter.close()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1280, 36), fill=(0, 0, 0))
        draw.text((16, 8), f"{name}  members={record['members']}  {record['quality']}  {record['seconds']}s",
                  fill="#00ff41", font=font)
        image.save(OUT / f"{name}.png")
        thumbs.append(image.resize((640, 360)))
        receipt["shots"][name] = record
        print(json.dumps({"shot": name, **record}), flush=True)
    rows = (len(thumbs) + 1) // 2
    board = Image.new("RGB", (1280, rows * 360), "black")
    for i, thumb in enumerate(thumbs):
        board.paste(thumb, ((i % 2) * 640, (i // 2) * 360))
    board.save(OUT / "board.png")
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"board": str(OUT / "board.png"), "shots": list(receipt["shots"])}), flush=True)


if __name__ == "__main__":
    main()

