from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from core.scene import Block

# Phosphor vector-monitor palette (P1 green CRT on black).
VIEWPORT_BG = "#000000"
PHOSPHOR = "#00ff41"          # bright trace: feature-edge outlines
PHOSPHOR_DIM = "#0d4f21"      # dim trace: interior tessellation edges
PHOSPHOR_FACE = "#000000"     # near-black faces for hidden-line occlusion
GRID_COLOR = "#002409"        # quiet floor grid beneath architectural traces
HIGHLIGHT = "#00ff41"         # thicker phosphor trace for selection
MARBLE_TRACE = "#7dff7d"      # rider: hotter P1 than canal / masonry traces
MARBLE_LINE_WIDTH = 5
MARBLE_RADIUS_SCALE = 1.35

GRID_SIZE = 220.0
GRID_DIVISIONS = 44
FEATURE_EDGE_ANGLE = 35.0
ORBIT_INTERVAL_MS = 33
ORBIT_STEP_DEGREES = 0.45
MARBLE_INTERVAL_MS = 33

pv.set_plot_theme("dark")
pv.global_theme.background = VIEWPORT_BG
pv.global_theme.anti_aliasing = "fxaa"


class ViewportWidget(QWidget):
    """Qt widget wrapping a PyVista interactor with a phosphor wireframe theme."""

    block_picked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background(VIEWPORT_BG)
        self.plotter.render_window.SetMultiSamples(0)
        self.plotter.renderer.SetUseFXAA(True)
        layout.addWidget(self.plotter.interactor)

        self.plotter.add_axes(color=PHOSPHOR, x_color=PHOSPHOR, y_color=PHOSPHOR, z_color=PHOSPHOR)
        self.plotter.camera_position = "iso"

        # face actor -> block id (pickable), block id -> outline actor (bright trace)
        self._actors: Dict[pv.Actor, int] = {}
        self._outline_actors: Dict[int, pv.Actor] = {}
        self._tints: Dict[int, Tuple[float, float, float]] = {}
        self._highlighted_block_id: Optional[int] = None
        self._grid_actor: Optional[pv.Actor] = None
        self._model_bounds = None
        self._batch_blocks = {}
        self._displayed_blocks = {}
        self._batch_actor = None
        self._batch_cell_ids = np.array([], dtype=int)
        self._batch_highlight = None
        self._cathedral_view = False
        self._camera_points = None

        self._orbit_timer = QTimer(self)
        self._orbit_timer.setInterval(ORBIT_INTERVAL_MS)
        self._orbit_timer.timeout.connect(self._orbit_tick)
        self._marble_timer = QTimer(self)
        self._marble_timer.setInterval(MARBLE_INTERVAL_MS)
        self._marble_timer.timeout.connect(self._marble_tick)
        self._marble_path = None
        self._marble_length = 0.0
        self._marble_speed = 6.0
        self._marble_s = 0.0
        self._marble_actor = None
        self._marble_outline = None
        self._marble_last_tick = None

        self._rebuild_grid()
        self.plotter.enable_mesh_picking(self._handle_pick, use_actor=True, show=False)

    # ------------------------------------------------------------------
    # Gridscape
    # ------------------------------------------------------------------
    def _rebuild_grid(self):
        """Add the phosphor floor grid (XZ plane; scene is Y-up)."""
        plane = pv.Plane(
            center=(0.0, -0.02, 0.0),
            direction=(0.0, 1.0, 0.0),
            i_size=GRID_SIZE,
            j_size=GRID_SIZE,
            i_resolution=GRID_DIVISIONS,
            j_resolution=GRID_DIVISIONS,
        )
        self._grid_actor = self.plotter.add_mesh(
            plane,
            style="wireframe",
            color=GRID_COLOR,
            line_width=1,
            lighting=False,
            pickable=False,
        )

    # ------------------------------------------------------------------
    # Block rendering
    # ------------------------------------------------------------------
    def set_block_tint(self, block_id: int, color) -> None:
        """Override a block's trace color (RGBA 0-255 tuple or hex string)."""
        if isinstance(color, str):
            self._tints[block_id] = color
        else:
            self._tints[block_id] = tuple(channel / 255.0 for channel in color[:3])

    def _trace_color(self, block_id: int):
        return self._tints.get(block_id, PHOSPHOR)

    def _add_phosphor_mesh(
        self,
        pv_mesh: pv.PolyData,
        trace_color,
        pickable: bool = True,
        line_width: float = 2,
        feature_angle: float = FEATURE_EDGE_ANGLE,
        tessellation_color=None,
        tessellation_width: float = 1,
        all_edges: bool = False,
    ) -> Tuple[Optional[pv.Actor], Optional[pv.Actor]]:
        """Render a mesh as hidden-line phosphor wireframe.

        Two actors: near-black faces with dim tessellation edges (occluders),
        plus bright feature-edge outlines on top.
        """
        face_actor = self.plotter.add_mesh(
            pv_mesh,
            color=PHOSPHOR_FACE,
            show_edges=True,
            edge_color=PHOSPHOR_DIM if tessellation_color is None else tessellation_color,
            line_width=tessellation_width,
            lighting=False,
            pickable=pickable,
        )

        outline_actor = None
        try:
            if all_edges:
                try:
                    edges = pv_mesh.extract_all_edges()
                except Exception:
                    edges = pv_mesh.extract_feature_edges(
                        feature_angle=1.0,
                        boundary_edges=True,
                        non_manifold_edges=True,
                        manifold_edges=True,
                    )
            else:
                edges = pv_mesh.extract_feature_edges(
                    feature_angle=feature_angle,
                    boundary_edges=True,
                    non_manifold_edges=True,
                    manifold_edges=False,
                )
            if edges.n_points > 0:
                outline_actor = self.plotter.add_mesh(
                    edges,
                    color=trace_color,
                    line_width=line_width,
                    lighting=False,
                    pickable=False,
                )
        except Exception:
            outline_actor = None

        return face_actor, outline_actor

    def _create_block_actor(self, block: Block) -> Optional[pv.Actor]:
        if block.mesh is None:
            return None

        try:
            pv_mesh = self._as_polydata(pv.wrap(block.mesh))
            face_actor, outline_actor = self._add_phosphor_mesh(
                pv_mesh, self._trace_color(block.block_id)
            )
        except Exception:
            return None

        if outline_actor is not None:
            self._outline_actors[block.block_id] = outline_actor
        return face_actor

    @staticmethod
    def _as_polydata(pv_mesh: pv.DataSet) -> pv.PolyData:
        if isinstance(pv_mesh, pv.PolyData):
            return pv_mesh
        return pv_mesh.extract_surface()

    def apply_stable_camera(self):
        if self._model_bounds is None:
            return
        if self._camera_points is not None:
            from .camera import frame_cathedral
            frame_cathedral(self.plotter, self._camera_points)
            return
        lo, hi = self._model_bounds
        center = (lo + hi) / 2
        radius = max(float(np.max(hi - lo)), 1.0)
        self.plotter.camera_position = [center + radius * np.array([1.0, 0.8, -1.0]), center, (0, 1, 0)]
        # The decorative 220-unit grid must not determine model framing.
        self.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

    def _set_model_bounds(self, meshes):
        bounds = [mesh.bounds for mesh in meshes if mesh is not None and not mesh.is_empty]
        self._model_bounds = (np.min([b[0] for b in bounds], axis=0),
                              np.max([b[1] for b in bounds], axis=0)) if bounds else None
        self._camera_points = np.vstack([m.vertices for m in meshes if m is not None and not m.is_empty]) if bounds else None
        if self._model_bounds is not None:
            if self._grid_actor is not None:
                self.plotter.remove_actor(self._grid_actor, render=False)
            lo,hi=self._model_bounds
            plane=pv.Plane(center=((lo[0]+hi[0])/2,-.02,(lo[2]+hi[2])/2),direction=(0,1,0),
                i_size=(hi[0]-lo[0])*1.22,j_size=(hi[2]-lo[2])*1.13,i_resolution=20,j_resolution=32)
            self._grid_actor=self.plotter.add_mesh(plane,style="wireframe",color=GRID_COLOR,
                line_width=.7,lighting=False,pickable=False)

    def display_meshes(self, blocks: List[Block], clear_existing: bool = True):
        self._clear_highlight()
        if not clear_existing:
            # Rebuild one batch so both old and appended cells keep correct IDs.
            combined = dict(self._displayed_blocks)
            combined.update((block.block_id, block) for block in blocks)
            blocks = list(combined.values())
        self.clear_viewport()
        self._displayed_blocks = {block.block_id: block for block in blocks}

        self._cathedral_view = any(
            b.parameters.get("architect") == "cathedral"
            or "cathedral" in str(b.parameters.get("primitive_id", "")).split(".")
            for b in blocks
        )
        # Native single-pixel traces keep close Gothic moldings separated and
        # avoid a full-frame blur pass on software OpenGL implementations.
        static = [block for block in blocks if block.parameters.get("component_role") != "marble"]
        if static:
            blocks = static
        sculpted = any(b.block_type == "MESH" for b in blocks)
        self.plotter.renderer.SetUseFXAA(not (self._cathedral_view or sculpted))
        if len(blocks) > 500 or sculpted:
            # Large assemblies share a draw call; cell IDs retain exact picking.
            self._display_batch(blocks)
        else:
            for block in blocks:
                actor = self._create_block_actor(block)
                if actor is not None:
                    self._actors[actor] = block.block_id

        if blocks:
            self._set_model_bounds([block.mesh for block in blocks])
            self.apply_stable_camera()
        self.plotter.render()

    def _display_batch(self, blocks):
        vertices, faces, cell_ids = [], [], []
        offset = 0
        for block in blocks:
            if block.mesh is None or block.mesh.is_empty:
                continue
            vertices.append(block.mesh.vertices)
            faces.append(block.mesh.faces + offset)
            offset += len(block.mesh.vertices)
            cell_ids.append(np.full(len(block.mesh.faces), block.block_id, dtype=int))
            self._batch_blocks[block.block_id] = block
        if not faces:
            return
        triangles = np.vstack(faces)
        poly = pv.PolyData(np.vstack(vertices), np.column_stack((np.full(len(triangles), 3), triangles)).ravel())
        self._batch_cell_ids = np.concatenate(cell_ids)

        self._batch_actor = self.plotter.add_mesh(poly, color="black", lighting=False, show_edges=False,
                                                   backface_culling=True)
        self._batch_actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self._batch_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1, 1)
        self._actors[self._batch_actor] = -1
        if self._cathedral_view:
            groups = {}
            for block in blocks:
                if block.mesh is None or block.mesh.is_empty:
                    continue
                role = block.parameters.get("component_role")
                section = block.parameters.get("section")
                if role == "apse_roof":
                    # The shallow hip facets are real geometry. A generic
                    # 26-degree crease filter would erase their roof reading.
                    style = ("#007220", 5)
                elif role == "roof_seam":
                    style = ("#005d18", 26)
                elif role in {"window_tracery", "facade_gallery", "pier_shaft", "finial"} or section == "fillet":
                    style = ("#007e20", 26)
                else:
                    style = ("#00c934", 26)
                groups.setdefault(style, []).append(block.mesh)
            for (color, feature_angle), selected in groups.items():
                edge_poly = self._as_polydata(pv.wrap(trimesh.util.concatenate(selected)))
                edges = edge_poly.extract_feature_edges(feature_angle=feature_angle, boundary_edges=True,
                    non_manifold_edges=True, manifold_edges=False)
                actor = self.plotter.add_mesh(edges, color=color, line_width=1,
                                             lighting=False, pickable=False)
                actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)
        else:
            groups = {}
            for block in blocks:
                role = block.parameters.get("component_role", block.parameters.get("element", ""))
                if role in {"column_shaft", "shaft"}:
                    style = ("#00b92f", 6 if block.parameters.get("order") == "Tuscan" else 20)
                elif "capital" in role or "base" in role:
                    style = ("#00d936", 8)
                elif role in {"dome", "dome_rib", "groin_vault", "apse_roof"}:
                    style = ("#008525", 7)
                elif role == "roof_seam":
                    style = ("#005d18", 26)
                else:
                    style = ("#00cf35", 26)
                if block.mesh is not None and not block.mesh.is_empty:
                    groups.setdefault(style, []).append(block.mesh)
            for (color, angle), selected in groups.items():
                edge_poly = self._as_polydata(pv.wrap(trimesh.util.concatenate(selected)))
                edges = edge_poly.extract_feature_edges(feature_angle=angle, boundary_edges=True,
                                                        non_manifold_edges=True, manifold_edges=False)
                if edges.n_points:
                    actor = self.plotter.add_mesh(edges, color=color, line_width=1, lighting=False, pickable=False)
                    actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                    actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)

        # Smooth vaults have no hard crease along their apparent horizon.
        # Follow that actual mesh silhouette as the camera moves so a dome
        # cannot disappear into the black face pass between its ribs.
        curved = [block.mesh for block in blocks
                  if (block.parameters.get("component_role") in {"dome", "groin_vault"}
                      or block.parameters.get("construction") == "continuous groin-vault shell")
                  and block.mesh is not None and not block.mesh.is_empty]
        if curved:
            contour = self.plotter.add_silhouette(
                self._as_polydata(pv.wrap(trimesh.util.concatenate(curved))),
                color="#00b52e", line_width=1, feature_angle=180, decimate=0)
            contour.SetPickable(False)
            contour.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            contour.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)

    def update_block_visuals(self, block: Block):
        if block.mesh is None:
            return
        if block.block_id in self._batch_blocks:

            # Faces remain black; a per-member trace overlay carries its tint.
            old = self._outline_actors.pop(block.block_id, None)
            if old is not None:
                self.plotter.remove_actor(old, render=False)
            edges = self._as_polydata(pv.wrap(block.mesh)).extract_feature_edges(feature_angle=26)
            color = tuple(float(c)/255 for c in block.color[:3])
            self._tints[block.block_id] = color
            self._outline_actors[block.block_id] = self.plotter.add_mesh(edges, color=color,
                line_width=2, lighting=False, pickable=False)
            self.plotter.render()
            return

        actor_to_remove = None
        for actor, block_id in self._actors.items():
            if block_id == block.block_id:
                actor_to_remove = actor
                break

        if actor_to_remove is None:
            return

        was_highlighted = self._highlighted_block_id == block.block_id

        self.plotter.remove_actor(actor_to_remove)
        del self._actors[actor_to_remove]
        old_outline = self._outline_actors.pop(block.block_id, None)
        if old_outline is not None:
            self.plotter.remove_actor(old_outline)
        if was_highlighted:
            self._highlighted_block_id = None

        new_actor = self._create_block_actor(block)
        if new_actor is None:
            return

        self._actors[new_actor] = block.block_id
        if was_highlighted:
            self._apply_highlight(block.block_id)

        self.plotter.render()

    def display_optimized_result(self, meshes: List[trimesh.Trimesh]):
        self.clear_viewport()

        for mesh in meshes:
            if mesh is None:
                continue
            try:
                pv_mesh = self._as_polydata(pv.wrap(mesh))
                self._add_phosphor_mesh(pv_mesh, PHOSPHOR, pickable=False)
            except Exception:
                continue

        if meshes:
            self._set_model_bounds(meshes)
            self.apply_stable_camera()
        self.plotter.render()

    # ------------------------------------------------------------------
    # Screensaver orbit
    # ------------------------------------------------------------------
    def set_orbit_enabled(self, enabled: bool):
        if enabled:
            self._orbit_timer.start()
        else:
            self._orbit_timer.stop()

    def _orbit_tick(self):
        try:
            self.plotter.camera.Azimuth(ORBIT_STEP_DEGREES)
            self.plotter.render()
        except Exception:
            self._orbit_timer.stop()

    def _stop_marble_run(self):
        self._marble_timer.stop()
        self._marble_path = None
        self._marble_last_tick = None
        for actor in (self._marble_actor, self._marble_outline):
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
        self._marble_actor = None
        self._marble_outline = None

    def start_marble_run(self, record):
        """Ride the closed canal forever. Phosphor traces stay; camera follows."""
        path = np.asarray((record or {}).get("path") or [], dtype=float)
        length = float((record or {}).get("length") or 0.0)
        if len(path) < 8 or length <= 1e-6:
            return
        self._stop_marble_run()
        self.set_orbit_enabled(False)
        self._marble_path = path
        self._marble_length = length
        self._marble_speed = float(record.get("speed") or 6.0)
        self._marble_s = 0.0
        radius = max(0.08, float(record.get("marble_radius") or 0.18)) * MARBLE_RADIUS_SCALE
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        # Icospheres lack 35° creases; the geodesic itself is the rider trace.
        face, outline = self._add_phosphor_mesh(
            self._as_polydata(pv.wrap(sphere)),
            MARBLE_TRACE,
            pickable=False,
            line_width=MARBLE_LINE_WIDTH,
            tessellation_color=MARBLE_TRACE,
            tessellation_width=2,
            all_edges=True,
        )
        self._marble_actor = face
        self._marble_outline = outline
        if outline is not None and getattr(outline, "prop", None) is not None:
            outline.prop.line_width = MARBLE_LINE_WIDTH
            outline.prop.color = MARBLE_TRACE
        try:
            self.plotter.disable_parallel_projection()
        except Exception:
            pass
        self._marble_last_tick = None
        self._marble_timer.start()
        self._marble_tick()

    def _marble_tick(self):
        if self._marble_path is None or self._marble_length <= 1e-6:
            self._marble_timer.stop()
            return
        try:
            from core.architecture.marble_run import point_on_loop
            now = time.monotonic()
            dt = 0.033 if self._marble_last_tick is None else min(0.08, now - self._marble_last_tick)
            self._marble_last_tick = now
            self._marble_s = (self._marble_s + dt * self._marble_speed) % self._marble_length
            position, tangent = point_on_loop(self._marble_path, self._marble_length, self._marble_s)
            if self._marble_actor is not None:
                self._marble_actor.SetPosition(*position)
            if self._marble_outline is not None:
                self._marble_outline.SetPosition(*position)
            camera = position - tangent * 10.0 + np.array([0.0, 3.4, 0.0])
            look = position + tangent * 5.0
            self.plotter.camera_position = [camera.tolist(), look.tolist(), (0, 1, 0)]
            self.plotter.render()
        except Exception:
            self._marble_timer.stop()

    # ------------------------------------------------------------------
    # Snapshot / lifecycle
    # ------------------------------------------------------------------
    def save_snapshot(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        previous_off_screen = getattr(self.plotter, "off_screen", False)
        self.apply_stable_camera()

        try:
            self.plotter.off_screen = True
            self.plotter.show(screenshot=path, auto_close=False)
        finally:
            self.plotter.off_screen = previous_off_screen

    def clear_viewport(self):
        self._stop_marble_run()
        self._clear_highlight()
        self.plotter.clear_actors()
        self._actors.clear()
        self._outline_actors.clear()
        self._tints.clear()
        self._model_bounds = None
        self._batch_blocks.clear()
        self._displayed_blocks.clear()
        self._batch_actor = None
        self._batch_cell_ids = np.array([], dtype=int)
        self._cathedral_view = False
        self._camera_points = None
        self._rebuild_grid()

    # ------------------------------------------------------------------
    # Picking / highlight
    # ------------------------------------------------------------------
    def _apply_highlight(self, block_id: int):
        if block_id in self._batch_blocks:
            edges = self._as_polydata(pv.wrap(self._batch_blocks[block_id].mesh)).extract_feature_edges(feature_angle=26)
            self._batch_highlight = self.plotter.add_mesh(edges, color=HIGHLIGHT,
                line_width=3, lighting=False, pickable=False)
        outline = self._outline_actors.get(block_id)
        if outline is not None and outline.prop:
            outline.prop.color = HIGHLIGHT
            outline.prop.line_width = 3
        self._highlighted_block_id = block_id

    def _clear_highlight(self):
        if self._batch_highlight is not None:
            self.plotter.remove_actor(self._batch_highlight, render=False)
            self._batch_highlight = None
        if self._highlighted_block_id is None:
            return
        outline = self._outline_actors.get(self._highlighted_block_id)
        if outline is not None and outline.prop:
            try:
                outline.prop.color = self._trace_color(self._highlighted_block_id)
                outline.prop.line_width = 2
            except Exception:
                pass
        self._highlighted_block_id = None

    def _handle_pick(self, actor: Optional[pv.Actor]):
        self._clear_highlight()

        if actor is None:
            self.block_picked.emit(None)
            return

        block_id = self._actors.get(actor)
        if block_id == -1 and self._batch_actor is not None:
            cell_id = self.plotter.iren.picker.GetCellId()
            block_id = int(self._batch_cell_ids[cell_id]) if 0 <= cell_id < len(self._batch_cell_ids) else None
        if block_id is None:
            self.block_picked.emit(None)
            return

        self._apply_highlight(block_id)
        self.plotter.render()
        self.block_picked.emit(block_id)

    def closeEvent(self, event):
        self._orbit_timer.stop()
        self._marble_timer.stop()
        self.plotter.close()
        super().closeEvent(event)
