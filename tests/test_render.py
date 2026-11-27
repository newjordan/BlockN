"""The voxel renderer draws a build, a cut, a walk overlay and a player figure."""

import unittest

import numpy as np

from core.architecture import BuildRequest, build
from core.architecture.gameplay import walk
from core.architecture.profile import preset
from core.architecture.render import render
from core.architecture.voxels import profile_pitch, voxelize


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        giant = preset("giant")
        result = build(BuildRequest(product="room-shell", seed=1, openings="door", profile="giant"))
        cls.grid = voxelize(result.document, result.scene, pitch=profile_pitch(giant), profile=giant)
        cls.walk = walk(cls.grid, giant)
        cls.giant = giant

    def test_draws_the_build(self):
        img = np.asarray(render(self.grid, width=400))
        background = np.array([246, 244, 239])
        self.assertGreater((np.abs(img - background).sum(axis=2) > 0).mean(), 0.2)

    def test_walk_overlay_and_figure_show(self):
        o = self.grid.origin
        img = np.asarray(render(self.grid, width=500, cut_above=3.2, walk=self.walk,
                                figure={"player": self.giant, "at": (o[0] + 1.0, o[2] + 1.0)}))
        r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
        self.assertGreater(((g > 150) & (r < 120) & (b < 120)).sum(), 50)   # reached floor
        self.assertGreater(((b > 150) & (r < 80)).sum(), 50)                # the figure


if __name__ == "__main__":
    unittest.main()
