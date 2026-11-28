"""Lay out the audited, exported GLBs from examples/brutalism-atlas.json.

Requires the optional PyVista/VTK viewer stack and Pillow. No model is rebuilt
or inferred from a prompt; the same GLBs remain usable in other viewers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import trimesh

W, H = 820, 468
MARGIN, GAP = 40, 16
BACKGROUND = "#202422"
CARD = "#171a18"
INK = "#f0e8da"
SUBTLE = "#bfc3ae"
AMBER = "#e9ae77"
LABELS = {
    "brutalist-civic-center": ("01 / CIVIC CENTER", "A council chamber crowns the public threshold"),
    "brutalist-housing": ("02 / COLLECTIVE HOUSING", "Dwelling bays repeat over an open undercroft"),
    "brutalist-arts-center": ("03 / ARTS CENTER", "Exposed walkways link galleries across a public passage"),
    "brutalist-portal-field": ("04 / PORTAL FIELD", "Interval and compression without a building program"),
}


def font(size, bold=False):
    for family in ("DejaVuSans", "LiberationSans", "JetBrainsMono"):
        try:
            return ImageFont.truetype(f"{family}{'-Bold' if bold else ''}.ttf", size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_asset(path, expected_size):
    scene = trimesh.load_scene(path)
    if len(scene.graph.nodes_geometry) != expected_size:
        raise ValueError(f"{path}: GLB node count differs from the construction report")
    positions, cells, colors = [], [], []
    offset = 0
    for node in scene.graph.nodes_geometry:
        matrix, key = scene.graph[node]
        mesh = scene.geometry[key]
        xyz = trimesh.transform_points(mesh.vertices, matrix)
        visual = mesh.visual
        if visual.kind == "texture":
            factor = np.asarray(visual.material.baseColorFactor, dtype=float)[:3] / 255
            linear = np.broadcast_to(factor, (len(xyz), 3)).copy()
            tint = visual.vertex_attributes.data.get("color")
            if tint is not None:
                tint = np.asarray(tint)
                linear *= tint[:, :3] / (255 if tint.dtype.kind in "ui" else 1)
            srgb = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
            rgb = np.rint(srgb * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = np.asarray(visual.vertex_colors, dtype=np.uint8)[:, :3]
        positions.append(xyz)
        cells.append(np.c_[np.full(len(mesh.faces), 3), mesh.faces+offset])
        colors.append(rgb)
        offset += len(xyz)
    data = pv.PolyData(np.vstack(positions), np.vstack(cells).ravel())
    data["rgb"] = np.vstack(colors)
    return data


def frame(data, eye):
    xmin, xmax, ymin, ymax, zmin, zmax = data.bounds
    target = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    right = np.cross([0, 1, 0], eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    projected = (data.points-target) @ np.column_stack([right, up])
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    mid = (lo+hi)/2
    target += right*mid[0] + up*mid[1]
    scale = .565 * max((hi[0]-lo[0]) / (W/H), hi[1]-lo[1])
    return target, scale


def render(data, eye, target, scale):
    p = pv.Plotter(off_screen=True, window_size=(W, H), lighting="none")
    p.set_background(BACKGROUND)
    p.enable_anti_aliasing("ssaa")
    p.enable_ssao(radius=1.3, bias=.04, kernel_size=48, blur=True)
    for pos, color, power in [((-55, 64, -36), "#fff2d8", 1.00),
                              ((56, 32, -24), "#d0e2ed", .43), ((23, 48, 57), "#fee4bf", .57)]:
        p.add_light(pv.Light(position=pos, focal_point=(0, 6, 0), color=color, intensity=power))
    p.add_mesh(data, scalars="rgb", rgb=True, ambient=.27, diffuse=.72, specular=.08,
               specular_power=22, show_scalar_bar=False)
    p.camera_position = [target + eye*170, target, (0., 1., 0.)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = scale
    return Image.fromarray(p.show(return_img=True)).convert("RGB")


def elevation_measurement(product, document):
    parts = document["placements"]
    if product == "brutalist-housing":
        edges = [p["position"][2] - p["dimensions"][2]/2 for p in parts if p["role"] == "living_deck"]
        return f"FRONT SLAB SETBACK   /   {max(edges)-min(edges):.2f} m total"
    stations = sorted(p["position"][2] for p in parts if p["role"] == "portal_lintel")
    intervals = " / ".join(f"{b-a:.2f}" for a, b in zip(stations, stations[1:]))
    return f"CENTER SPACING   /   {intervals} m"


def elevations(cards, matched, bundle):
    """A side-on comparison exposes the setbacks and intervals foreshortened in perspective."""
    board = Image.new("RGB", (2*MARGIN+2*W+GAP, 1280), CARD)
    draw = ImageDraw.Draw(board)
    draw.text((MARGIN, 18), "MATCHED STUDIES  /  SIDE ELEVATIONS", fill=INK, font=font(43, True))
    for row, (product, caption) in enumerate((
            ("brutalist-housing", "DWELLING TERRACES   /   front at left"),
            ("brutalist-portal-field", "PORTAL INTERVALS   /   spacing and frame proportions both vary"))):
        samples = cards[product][:2]
        eye = np.array((-1., 0., 0.))
        frames = [frame(data, eye) for _, data, _, _ in samples]
        scale = max(scale for _, scale in frames)
        y = 92+row*585
        draw.text((MARGIN, y), caption, fill=AMBER, font=font(24, True))
        for col, ((request, data, _, _), (target, _)) in enumerate(zip(samples, frames)):
            x = MARGIN+col*(W+GAP)
            board.paste(render(data, eye, target, scale), (x, y+39))
            draw.text((x+5, y+515), request["variant"].replace("-", " ").upper(), fill=INK, font=font(28, True))
            draw.text((x+5, y+551), matched[product][col], fill=SUBTLE, font=font(21))
    board.save(bundle / "brutalism-elevations.png", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Directory produced by blockn agent build")
    args = parser.parse_args()
    bundle = args.bundle
    receipt = read_json(bundle / "batch.manifest.json")
    recipe = read_json(bundle / "recipe.json")
    if receipt["status"] != "pass" or receipt["count"] != 16:
        raise ValueError("Expected a passing sixteen-study Brutalism atlas batch")
    assets = {asset["id"]: asset for asset in receipt["buildings"]}
    cards = {product: [] for product in LABELS}
    matched = {product: [] for product in ("brutalist-housing", "brutalist-portal-field")}
    for entry in recipe["buildings"]:
        request = entry["request"]
        product = request["product"]
        files = assets[entry["id"]]["artifacts"]
        report = read_json(bundle / files["construction"])
        if report["quality"] != "pass":
            raise ValueError(f"{entry['id']}: construction audit did not pass")
        count = sum(report["role_counts"].values())
        data = load_asset(bundle / files["glb"], count)
        quality = read_json(bundle / files["quality"])
        if product in matched and len(matched[product]) < 2:
            document = read_json(bundle / files["architecture"])
            matched[product].append(elevation_measurement(product, document))
        cards[product].append((request, data, count, quality["metrics"]["bounds_size"]))
    banner = 194
    row_height = 62 + H + 82 + 24
    width = 2*MARGIN+4*W+3*GAP
    poster = Image.new("RGB", (width, banner+len(LABELS)*row_height+32), CARD)
    draw = ImageDraw.Draw(poster)
    draw.text((MARGIN, 24), "BLOCKN STUDIO   /   RESEARCH ATLAS 001", fill=AMBER, font=font(27, True))
    draw.text((MARGIN-4, 54), "BRUTALISM AS ARCHITECTURE", fill=INK, font=font(79, True))
    draw.text((MARGIN, 150), "VOID  /  BEARING  /  RHYTHM  /  MATERIAL", fill=SUBTLE, font=font(28))
    draw.text((width-MARGIN, 150), "16 BUILT STUDIES   ·   4 BRIEFS   ·   AUDITED GLB", fill=SUBTLE,
              font=font(26), anchor="ra")
    for row, (product, (label, caption)) in enumerate(LABELS.items()):
        samples = cards[product]
        if len(samples) != 4:
            raise ValueError(f"{product}: expected four studies")
        eye = np.array((.68, .58, -1.0) if row != 1 else (.53, .40, -1.0))
        eye /= np.linalg.norm(eye)
        frames = [frame(data, eye) for _, data, _, _ in samples]
        scale = max(scale for _, scale in frames)  # identical metres/pixel within a program
        y = banner+row*row_height
        draw.line((MARGIN, y, width-MARGIN, y), fill="#626757", width=2)
        draw.text((MARGIN, y+13), label, fill=AMBER, font=font(34, True))
        draw.text((width-MARGIN, y+16), caption, fill=SUBTLE, font=font(29), anchor="ra")
        for col, ((r, data, count, size), (target, _)) in enumerate(zip(samples, frames)):
            x = MARGIN+col*(W+GAP)
            poster.paste(render(data, eye, target, scale), (x, y+59))
            draw.text((x+6, y+H+65), r["variant"].replace("-", " ").upper(), fill=INK, font=font(29, True))
            dims = " x ".join(f"{v:g}" for v in size)
            mode = "MATCHED" if col < 2 else f"SEED {r['seed']}"
            draw.text((x+7, y+H+103), f"{mode}   ·   {dims} m   ·   {count} members", fill=SUBTLE, font=font(23))
    poster.save(bundle / "brutalism-atlas.png", optimize=True)
    poster.thumbnail((2300, 2300), Image.Resampling.LANCZOS)
    poster.save(bundle / "brutalism-preview.png", optimize=True)
    elevations(cards, matched, bundle)
    print(f"Rendered {receipt['count']} audited GLBs to {bundle/'brutalism-atlas.png'}")


if __name__ == "__main__":
    main()
