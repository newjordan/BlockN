"""Voxel grids: the canonical, mergeable form of a compiled build.

A ``VoxelGrid`` is a dense uint16 label array over world space. Label 0 is air;
every other label is an entry in the grid's palette, one per substance. The
grid is what the playability checks walk, and the form other world systems
merge with: 2D plans and sections, Minecraft ``.mcfunction`` files and engine
blockouts are all derived from it.

The merge contract (``blockn.voxels/v1``):

* Units are metres. Y is up and the ground plane is y = 0. Axes are BlockN's
  (right-handed, +X east, +Z south in Minecraft terms), so the Minecraft export
  maps axes one to one.
* ``origin`` is the world position of the min corner of cell (0, 0, 0). X and Z
  are snapped to whole cells and the bottom layer of cells sits on the ground,
  so a build placed at a world cell lines up with any grid of the same pitch.
* Sizes are padded to whole chunks of ``chunk_size`` (16) cells on every axis.
* ``palette`` maps each label to a substance, with a suggested Minecraft block.
* ``ports`` are the doorways and entrances found in the grid, each with a
  world position on the ground, a facing, a clear width and height and the
  profile it was measured for. A world system snaps a road, corridor or
  terrain cut to a port.

Arrays are indexed ``labels[x, y, z]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VOXEL_SCHEMA = "blockn.voxels/v1"
AIR = 0
CHUNK = 16
DEFAULT_PITCH = 0.25
DEFAULT_MARGIN = 2.0

# Suggested Minecraft blocks by substance keyword, checked in order. A world
# system is free to remap; these only make the export usable as it stands.
_MINECRAFT = (
    ("glass", "minecraft:glass"),
    ("lead", "minecraft:iron_block"),
    ("iron", "minecraft:iron_bars"),
    ("steel", "minecraft:iron_block"),
    ("bronze", "minecraft:copper_block"),
    ("copper", "minecraft:copper_block"),
    ("gold", "minecraft:gold_block"),
    ("marble", "minecraft:quartz_block"),
    ("limestone", "minecraft:smooth_stone"),
    ("sandstone", "minecraft:sandstone"),
    ("granite", "minecraft:polished_granite"),
    ("basalt", "minecraft:polished_basalt"),
    ("tuff", "minecraft:tuff"),
    ("slate", "minecraft:deepslate_tiles"),
    ("mud", "minecraft:mud_bricks"),
    ("brick", "minecraft:bricks"),
    ("roof-tile", "minecraft:terracotta"),
    ("terracotta", "minecraft:terracotta"),
    ("glazed", "minecraft:white_glazed_terracotta"),
    ("tile", "minecraft:terracotta"),
    ("plaster", "minecraft:white_concrete"),
    ("stucco", "minecraft:white_concrete"),
    ("concrete", "minecraft:light_gray_concrete"),
    ("thatch", "minecraft:hay_block"),
    ("cedar", "minecraft:spruce_planks"),
    ("oak", "minecraft:oak_planks"),
    ("softwood", "minecraft:spruce_planks"),
    ("timber", "minecraft:oak_planks"),
    ("wood", "minecraft:oak_planks"),
    ("stone", "minecraft:stone_bricks"),
)


def minecraft_block(substance: Optional[str]) -> str:
    key = (substance or "").lower()
    for word, block in _MINECRAFT:
        if word in key:
            return block
    return "minecraft:stone"


@dataclass
class VoxelGrid:
    labels: np.ndarray                       # uint16, [x, y, z]
    pitch: float
    origin: Tuple[float, float, float]       # world min corner of cell (0, 0, 0)
    palette: List[Dict[str, Any]]            # index = label - 1
    ports: List[Dict[str, Any]] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.labels.shape)  # type: ignore[return-value]

    @property
    def solid(self) -> np.ndarray:
        return self.labels != AIR

    def cell_of(self, point: Sequence[float]) -> Tuple[int, int, int]:
        return tuple(int(math.floor((float(point[i]) - self.origin[i]) / self.pitch)) for i in range(3))  # type: ignore[return-value]

    def center_of(self, cell: Sequence[int]) -> Tuple[float, float, float]:
        return tuple(round(self.origin[i] + (int(cell[i]) + 0.5) * self.pitch, 6) for i in range(3))  # type: ignore[return-value]

    def manifest(self) -> Dict[str, Any]:
        nx, ny, nz = self.shape
        return {
            "schema": VOXEL_SCHEMA,
            "units": "metre",
            "up": "+y",
            "ground_y": 0.0,
            "pitch": self.pitch,
            "origin": [round(v, 6) for v in self.origin],
            "size": [nx, ny, nz],
            "chunk_size": CHUNK,
            "chunks": [nx // CHUNK, ny // CHUNK, nz // CHUNK],
            "index_order": "labels[x, y, z]",
            "air": AIR,
            "palette": self.palette,
            "solid_cells": int(self.solid.sum()),
            "ports": self.ports,
            "profile": self.profile,
            "source": self.source,
        }

    # ---------------------------------------------------------------- files
    def save(self, directory: str | Path, name: str = "build") -> Dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / f"{name}.voxels.npz"
        manifest_path = directory / f"{name}.voxels.json"
        np.savez_compressed(data_path, labels=self.labels)
        manifest = self.manifest()
        manifest["data"] = data_path.name
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"manifest": manifest_path, "data": data_path}

    @classmethod
    def load(cls, manifest_path: str | Path) -> "VoxelGrid":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != VOXEL_SCHEMA:
            raise ValueError(f"not a {VOXEL_SCHEMA} manifest")
        with np.load(manifest_path.parent / manifest["data"]) as data:
            labels = data["labels"].astype(np.uint16)
        if list(labels.shape) != list(manifest["size"]):
            raise ValueError("voxel data does not match the manifest size")
        return cls(labels, float(manifest["pitch"]), tuple(manifest["origin"]), manifest["palette"],
                   manifest.get("ports") or [], manifest.get("profile"), manifest.get("source") or {})

    def chunks(self) -> Iterator[Tuple[Tuple[int, int, int], np.ndarray]]:
        """Non-empty 16-cube chunks as ((cx, cy, cz), labels)."""
        nx, ny, nz = self.shape
        for cx in range(nx // CHUNK):
            for cy in range(ny // CHUNK):
                for cz in range(nz // CHUNK):
                    block = self.labels[cx * CHUNK:(cx + 1) * CHUNK, cy * CHUNK:(cy + 1) * CHUNK,
                                        cz * CHUNK:(cz + 1) * CHUNK]
                    if block.any():
                        yield (cx, cy, cz), block

    # -------------------------------------------------------------- exports
    def to_mcfunction(self) -> str:
        """Minecraft commands that place the build relative to where they run.

        One cell is one Minecraft block, so the grid must be at pitch 1.0. The
        command origin (``~ ~ ~``) is world (0, 0, 0): the build's own origin on
        the ground. Runs of one block along X become one ``fill``.
        """
        if abs(self.pitch - 1.0) > 1e-9:
            raise ValueError("Minecraft export needs a pitch of 1.0; voxelize again with pitch=1.0")
        blocks = {i + 1: entry["minecraft"] for i, entry in enumerate(self.palette)}
        ox, oy, oz = (int(round(v)) for v in self.origin)
        lines = [f"# {VOXEL_SCHEMA} from {self.source.get('product', 'build')}; "
                 f"run at the spot that should be world (0, 0, 0) on the ground"]
        nx, ny, nz = self.shape
        for y in range(ny):
            layer = self.labels[:, y, :]
            if not layer.any():
                continue
            for z in range(nz):
                row = layer[:, z]
                x = 0
                while x < nx:
                    label = int(row[x])
                    if label == AIR:
                        x += 1
                        continue
                    end = x
                    while end + 1 < nx and int(row[end + 1]) == label:
                        end += 1
                    a = (x + ox, y + oy, z + oz)
                    if end == x:
                        lines.append(f"setblock ~{a[0]} ~{a[1]} ~{a[2]} {blocks[label]}")
                    else:
                        lines.append(f"fill ~{a[0]} ~{a[1]} ~{a[2]} ~{end + ox} ~{a[1]} ~{a[2]} {blocks[label]}")
                    x = end + 1
        return "\n".join(lines) + "\n"

    def plan_ascii(self, height: float = 1.0) -> str:
        """A top-down slice at ``height`` metres: ``#`` solid, ``.`` air, ``D`` a port.

        Rows run along +Z, columns along +X.
        """
        y = max(0, min(self.shape[1] - 1, int(math.floor((height - self.origin[1]) / self.pitch))))
        rows = [["#" if v else "." for v in self.labels[:, y, z]] for z in range(self.shape[2])]
        for port in self.ports:
            cx, _, cz = self.cell_of(port["position"])
            if 0 <= cz < len(rows) and 0 <= cx < len(rows[0]):
                rows[cz][cx] = "D"
        return "\n".join("".join(row) for row in rows) + "\n"

    def section_ascii(self, z: float = 0.0) -> str:
        """A vertical slice at world ``z``, top row highest: ``#`` solid, ``.`` air."""
        k = max(0, min(self.shape[2] - 1, int(math.floor((z - self.origin[2]) / self.pitch))))
        rows = ["".join("#" if v else "." for v in self.labels[:, y, k]) for y in range(self.shape[1] - 1, -1, -1)]
        return "\n".join(rows) + "\n"

    def blockout_boxes(self) -> List[Dict[str, Any]]:
        """Greedy-merged boxes (min, max, label) for an engine blockout.

        Runs along X are merged, then identical runs stacked along Z. Good
        enough to import as collision or greybox geometry; not minimal.
        """
        nx, ny, nz = self.shape
        out: List[Dict[str, Any]] = []
        for y in range(ny):
            layer = self.labels[:, y, :]
            if not layer.any():
                continue
            open_runs: Dict[Tuple[int, int, int], int] = {}
            for z in range(nz + 1):
                runs = set()
                if z < nz:
                    row = layer[:, z]
                    x = 0
                    while x < nx:
                        label = int(row[x])
                        if label == AIR:
                            x += 1
                            continue
                        end = x
                        while end + 1 < nx and int(row[end + 1]) == label:
                            end += 1
                        runs.add((x, end, label))
                        x = end + 1
                for run in list(open_runs):
                    if run not in runs:
                        z0 = open_runs.pop(run)
                        x0, x1, label = run
                        out.append(self._box(x0, x1, y, z0, z - 1, label))
                for run in runs:
                    open_runs.setdefault(run, z)
        return out

    def _box(self, x0: int, x1: int, y: int, z0: int, z1: int, label: int) -> Dict[str, Any]:
        p, o = self.pitch, self.origin
        return {
            "min": [round(o[0] + x0 * p, 6), round(o[1] + y * p, 6), round(o[2] + z0 * p, 6)],
            "max": [round(o[0] + (x1 + 1) * p, 6), round(o[1] + (y + 1) * p, 6), round(o[2] + (z1 + 1) * p, 6)],
            "substance": self.palette[label - 1]["substance"],
        }


# --------------------------------------------------------------- voxelizing
def _pad(n: int) -> int:
    return max(CHUNK, int(math.ceil(n / CHUNK)) * CHUNK)


def _is_axis_box(mesh: Any) -> bool:
    vertices = np.asarray(mesh.vertices)
    if len(vertices) != 8:
        return False
    lo, hi = vertices.min(0), vertices.max(0)
    on_corner = np.isclose(vertices, lo, atol=1e-6) | np.isclose(vertices, hi, atol=1e-6)
    return bool(on_corner.all())


def winding_number(points: np.ndarray, triangles: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Generalized winding number of each point for a triangle soup.

    Van Oosterom and Strackee's solid angle per triangle, summed over 4 pi.
    It is about 1 inside a closed mesh and 0 outside, and degrades gently for
    meshes with small holes, which is why it is used over ray parity.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    out = np.zeros(len(points))
    if not len(points) or not len(triangles):
        return out
    step = max(1, chunk // max(1, len(triangles)))
    for start in range(0, len(points), step):
        p = points[start:start + step, None, None, :]
        d = triangles[None] - p                        # n x m x 3 x 3
        a, b, c = d[:, :, 0], d[:, :, 1], d[:, :, 2]
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        det = np.einsum("nmk,nmk->nm", a, np.cross(b, c))
        dot = lambda u, v: np.einsum("nmk,nmk->nm", u, v)
        denom = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb
        out[start:start + step] = (2.0 * np.arctan2(det, denom)).sum(-1) / (4.0 * np.pi)
    return out


def _thinner_than(mesh: Any, pitch: float) -> bool:
    """Whether a closed piece is thinner than one cell: twice its volume over
    its area is a slab's thickness."""
    area = float(mesh.area)
    return area > 0 and 2.0 * abs(float(mesh.volume)) / area < pitch


def _surface_cells(triangles: np.ndarray, origin: Sequence[float], pitch: float,
                   size: Sequence[int]) -> np.ndarray:
    """The cells a mesh's surface passes through, found by sampling every
    triangle on a barycentric grid finer than a third of a cell."""
    points = []
    for a, b, c in triangles:
        edge = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
        k = max(1, int(math.ceil(edge / (pitch / 3.0))))
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        u, v = i[keep] / k, j[keep] / k
        points.append(a + np.outer(u, b - a) + np.outer(v, c - a))
    if not points:
        return np.zeros((0, 3), dtype=int)
    cells = np.floor((np.concatenate(points) - np.asarray(origin)) / pitch).astype(int)
    cells = np.unique(cells, axis=0)
    ok = np.all((cells >= 0) & (cells < np.asarray(size)), axis=1)
    return cells[ok]


def _axis_range(lo: float, hi: float, origin: float, pitch: float, size: int) -> np.ndarray:
    """Cells whose centres the interval covers; a slab thinner than a cell still
    takes the one cell its middle falls in, so thin walls stay watertight."""
    first = int(math.ceil((lo - origin) / pitch - 0.5 - 1e-9))
    last = int(math.floor((hi - origin) / pitch - 0.5 + 1e-9))
    if last < first:
        first = last = int(math.floor(((lo + hi) / 2.0 - origin) / pitch))
    return np.arange(max(0, first), min(size - 1, last) + 1)


def voxelize(
    document: Any,
    scene: Any,
    *,
    pitch: Optional[float] = None,
    profile: Any = None,
    margin: float = DEFAULT_MARGIN,
) -> VoxelGrid:
    """Voxelize a compiled build.

    ``pitch`` defaults to a third of the player's radius when a profile is
    given (rounded down to 1 cm, between 5 cm and 0.5 m), so the body is seven
    cells across and a doorway built to the profile's clearance always
    measures as wide enough; without a profile it is 0.25 m.
    """
    from .profile import resolve

    player = resolve(profile if profile is not None else getattr(document.request, "profile", None))
    source = {
        "product": document.metadata.get("product") if hasattr(document, "metadata") else None,
        "fingerprint": document.fingerprint() if hasattr(document, "fingerprint") else None,
        "seed": getattr(document.request, "seed", None),
    }
    if source["product"] is None:
        source["product"] = getattr(document.request, "product", None)
    return voxelize_parts(list(scene.blocks), list(document.placements), pitch=pitch, player=player,
                          margin=margin, source=source)


def profile_pitch(player: Any) -> float:
    """The default pitch for a player: a third of its radius, and never coarser
    than the 0.15 m clearance each side of a profile door (so a doorway built
    to the profile measures as passable), to the centimetre, 0.05 to 0.5 m."""
    from .profile import SIDE_CLEARANCE, resolve

    radius = resolve(player).radius
    return min(0.5, max(0.05, math.floor(min(radius / 3.0, SIDE_CLEARANCE - 0.02) * 100 + 1e-9) / 100))


def voxelize_parts(
    blocks: Sequence[Any],
    placements: Sequence[Any],
    *,
    pitch: Optional[float] = None,
    player: Any = None,
    margin: float = DEFAULT_MARGIN,
    source: Optional[Dict[str, Any]] = None,
) -> VoxelGrid:
    """Voxelize matching lists of scene blocks and placements (a whole build or
    one building of a settlement)."""
    from .profile import resolve

    player = resolve(player)
    if pitch is None:
        pitch = DEFAULT_PITCH if player is None else profile_pitch(player)
    pitch = float(pitch)
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("pitch must be a positive number")
    blocks = list(blocks)
    placements = list(placements)
    if len(blocks) != len(placements):
        raise ValueError("scene and document disagree on the number of blocks")
    bounds = np.array([np.asarray(b.mesh.bounds) for b in blocks]) if blocks else np.zeros((1, 2, 3))
    lo = bounds[:, 0].min(0)
    hi = bounds[:, 1].max(0)
    lo_x = math.floor((lo[0] - margin) / pitch) * pitch
    lo_z = math.floor((lo[2] - margin) / pitch) * pitch
    origin = (round(lo_x, 9), 0.0, round(lo_z, 9))
    size = (
        _pad(int(math.ceil((hi[0] + margin - lo_x) / pitch))),
        _pad(int(math.ceil((max(hi[1], 0.0) + margin) / pitch))),
        _pad(int(math.ceil((hi[2] + margin - lo_z) / pitch))),
    )
    labels = np.zeros(size, dtype=np.uint16)

    palette: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}

    def label_for(substance: Optional[str]) -> int:
        key = substance or "unspecified"
        if key not in index:
            palette.append({"label": len(palette) + 1, "substance": key, "minecraft": minecraft_block(substance)})
            index[key] = len(palette)
        return index[key]

    for block, placement in zip(blocks, placements):
        mesh = block.mesh
        b_lo, b_hi = np.asarray(mesh.bounds)
        xs = _axis_range(b_lo[0], b_hi[0], origin[0], pitch, size[0])
        ys = _axis_range(b_lo[1], b_hi[1], origin[1], pitch, size[1])
        zs = _axis_range(b_lo[2], b_hi[2], origin[2], pitch, size[2])
        if not (len(xs) and len(ys) and len(zs)):
            continue
        label = label_for(getattr(placement, "substance", None))
        if _is_axis_box(mesh):
            labels[np.ix_(xs, ys, zs)] = label
            continue
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        cells = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        centres = (cells + 0.5) * pitch + np.asarray(origin)
        triangles = np.asarray(mesh.triangles)
        inside = winding_number(centres, triangles) > 0.5
        if _thinner_than(mesh, pitch):
            # A sloped or turned piece thinner than a cell (a roof course, a
            # tilted board) misses most cell centres and voxelizes as a
            # scatter of blocks. Every cell its surface passes through is
            # taken instead, so it stays one cell thick and closed.
            hit = _surface_cells(triangles, origin, pitch, size)
            if len(hit):
                labels[hit[:, 0], hit[:, 1], hit[:, 2]] = label
        if not inside.any() and len(cells):
            # Thinner than a cell everywhere (a fin, a rail): keep the cell
            # nearest the block's centre so it does not vanish.
            mid = np.asarray(mesh.bounds).mean(0)
            inside[np.argmin(np.linalg.norm(centres - mid, axis=1))] = True
        hit = cells[inside]
        labels[hit[:, 0], hit[:, 1], hit[:, 2]] = label

    return VoxelGrid(labels, pitch, origin, palette, [], None if player is None else player.to_dict(), dict(source or {}))


__all__ = ["profile_pitch", "VOXEL_SCHEMA", "AIR", "CHUNK", "VoxelGrid", "voxelize", "voxelize_parts", "winding_number",
           "minecraft_block"]
