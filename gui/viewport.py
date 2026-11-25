from __future__ import annotations

import os
from typing import Dict, List, Optional

import pyvista as pv
import trimesh
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from core.scene import Block

VIEWPORT_BASE = "#18233a"
VIEWPORT_SKY = "#f5a35f"
VIEWPORT_EDGE = "#2b1b16"
VIEWPORT_HIGHLIGHT = "#ffd36a"
VIEWPORT_FALLBACK = "#d8be95"
VIEWPORT_NORMAL_MODE = "face"  # "face" (Maya Set to Face style) or "smooth"
VIEWPORT_NORMAL_FEATURE_ANGLE = 88.0
VIEWPORT_AMBIENT = 0.34
VIEWPORT_DIFFUSE = 0.62
VIEWPORT_SPECULAR = 0.08
VIEWPORT_SPECULAR_POWER = 14.0

pv.set_plot_theme("document")
pv.global_theme.background = VIEWPORT_BASE
pv.global_theme.anti_aliasing = "fxaa"
pv.global_theme.smooth_shading = True


class ViewportWidget(QWidget):
    """Qt widget wrapping a PyVista interactor."""

    block_picked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background(VIEWPORT_BASE, top=VIEWPORT_SKY)
        layout.addWidget(self.plotter.interactor)

        self.plotter.add_axes()
        self.plotter.camera_position = "iso"

        self._actors: Dict[pv.Actor, int] = {}
        self._highlighted_actor: Optional[pv.Actor] = None

        self.plotter.enable_mesh_picking(self._handle_pick, use_actor=True, show=False)

    def _create_block_actor(self, block: Block) -> Optional[pv.Actor]:
        if block.mesh is None:
            return None

        pv_mesh = self._prepare_render_mesh(pv.wrap(block.mesh))

        try:
            # Block colors are stored as RGBA. Passing that array to PyVista with
            # rgb=True fails once normals split the vertices, and the actor was dropped.
            color_rgb = [channel / 255.0 for channel in block.color[:3]]
            opacity = block.color[3] / 255.0 if len(block.color) > 3 else 1.0
            actor = self.plotter.add_mesh(
                pv_mesh,
                color=color_rgb,
                opacity=opacity,
                show_edges=True,
                edge_color=VIEWPORT_EDGE,
                line_width=1,
                **self._base_mesh_render_kwargs(),
            )
        except Exception:
            return None

        return actor

    def _base_mesh_render_kwargs(self) -> Dict[str, float | bool]:
        return {
            "smooth_shading": VIEWPORT_NORMAL_MODE != "face",
            "ambient": VIEWPORT_AMBIENT,
            "diffuse": VIEWPORT_DIFFUSE,
            "specular": VIEWPORT_SPECULAR,
            "specular_power": VIEWPORT_SPECULAR_POWER,
        }

    def _prepare_render_mesh(self, pv_mesh: pv.DataSet) -> pv.DataSet:
        prepared = pv_mesh.copy(deep=True)
        if not isinstance(prepared, pv.PolyData):
            try:
                prepared = prepared.extract_surface()
            except Exception:
                return prepared

        try:
            if VIEWPORT_NORMAL_MODE == "face":
                prepared = prepared.compute_normals(
                    cell_normals=True,
                    point_normals=False,
                    split_vertices=True,
                    consistent_normals=True,
                    auto_orient_normals=True,
                    non_manifold_traversal=True,
                )
            else:
                prepared = prepared.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    split_vertices=False,
                    feature_angle=VIEWPORT_NORMAL_FEATURE_ANGLE,
                    consistent_normals=True,
                    auto_orient_normals=True,
                    non_manifold_traversal=True,
                )
        except Exception:
            return prepared

        return prepared

    def apply_stable_camera(self):
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()

    def display_meshes(self, blocks: List[Block], clear_existing: bool = True):
        self._clear_highlight()
        if clear_existing:
            self.clear_viewport()

        for block in blocks:
            actor = self._create_block_actor(block)
            if actor is not None:
                self._actors[actor] = block.block_id

        if blocks:
            self.apply_stable_camera()
        self.plotter.render()

    def update_block_visuals(self, block: Block):
        if block.mesh is None:
            return

        actor_to_remove = None
        for actor, block_id in self._actors.items():
            if block_id == block.block_id:
                actor_to_remove = actor
                break

        if actor_to_remove is None:
            return

        was_highlighted = self._highlighted_actor == actor_to_remove

        self.plotter.remove_actor(actor_to_remove)
        del self._actors[actor_to_remove]

        if was_highlighted:
            self._highlighted_actor = None

        new_actor = self._create_block_actor(block)
        if new_actor is None:
            return

        self._actors[new_actor] = block.block_id
        if was_highlighted and new_actor.prop:
            new_actor.prop.edge_color = VIEWPORT_HIGHLIGHT
            new_actor.prop.line_width = 3
            self._highlighted_actor = new_actor

        self.plotter.render()

    def display_optimized_result(self, meshes: List[trimesh.Trimesh]):
        self.clear_viewport()

        for mesh in meshes:
            if mesh is None:
                continue

            pv_mesh = self._prepare_render_mesh(pv.wrap(mesh))
            kwargs = {
                "show_edges": True,
                "edge_color": VIEWPORT_EDGE,
                "line_width": 1,
                **self._base_mesh_render_kwargs(),
            }
            # Optimized meshes are one color per group. Use that color directly so
            # a change in vertex count after normal splitting cannot drop the mesh.
            color = VIEWPORT_FALLBACK
            visual = getattr(mesh, "visual", None)
            face_colors = getattr(visual, "face_colors", None)
            if face_colors is not None and len(face_colors):
                color = [channel / 255.0 for channel in face_colors[0][:3]]
            self.plotter.add_mesh(
                pv_mesh,
                color=color,
                **kwargs,
            )

        if meshes:
            self.apply_stable_camera()
        self.plotter.render()

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
        self._clear_highlight()
        self.plotter.clear_actors()
        self._actors.clear()

    def _clear_highlight(self):
        if self._highlighted_actor and self._highlighted_actor.prop:
            try:
                self._highlighted_actor.prop.edge_color = VIEWPORT_EDGE
                self._highlighted_actor.prop.line_width = 1
            except Exception:
                pass
        self._highlighted_actor = None

    def _handle_pick(self, actor: Optional[pv.Actor]):
        self._clear_highlight()

        if actor is None:
            self.block_picked.emit(None)
            return

        block_id = self._actors.get(actor)
        if block_id is None:
            self.block_picked.emit(None)
            return

        if actor.prop:
            actor.prop.edge_color = VIEWPORT_HIGHLIGHT
            actor.prop.line_width = 3
            self._highlighted_actor = actor

        self.block_picked.emit(block_id)

    def highlight_block(self, block_id: int):
        self._clear_highlight()
        for actor, ident in self._actors.items():
            if ident != block_id or not actor.prop:
                continue
            actor.prop.edge_color = VIEWPORT_HIGHLIGHT
            actor.prop.line_width = 3
            self._highlighted_actor = actor
            break
        self.plotter.render()

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)
