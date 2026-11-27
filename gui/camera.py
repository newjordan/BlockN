"""Consistent CAD framing from visible geometry, independent of decorative grids."""
import numpy as np


def frame_cathedral(plotter, vertices, direction=(-2.1, .50, -1.25), margin=1.045):
    points = np.asarray(vertices)
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    horizontal = (points-center) @ right
    vertical = (points-center) @ up
    center += right * (horizontal.min()+horizontal.max())/2 + up * (vertical.min()+vertical.max())/2
    width, height = plotter.window_size
    scale = max(np.ptp(vertical)/2, np.ptp(horizontal)/2 / (width/height))*margin
    distance = max(np.ptp(points, axis=0))*3
    plotter.enable_parallel_projection()
    plotter.camera_position = [center+forward*distance, center, up]
    plotter.camera.parallel_scale = float(scale)
    plotter.reset_camera_clipping_range()
