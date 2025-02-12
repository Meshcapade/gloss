#!/usr/bin/env python3
"""
Display a point cloud
"""

import numpy as np

from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.types import PointColorType
from gloss.components import Verts, Colors, VisPoints

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()
    mesh = viewer.get_or_create_entity(name = "mesh")

    verts = Verts(np.array([
        [0,0,0],
        [0,1,0],
        [1,0,0],
        [1.5,1.5,-1]
    ], dtype = "float32"))

    colors = Colors(np.array([
        [1,0,0],
        [0,0,1],
        [1,1,0],
        [1,0,1]
    ], dtype = "float32"))

    point_visualisation = VisPoints(show_points=True, \
                        show_points_indices=True, \
                        point_size=10.0, \
                        color_type = PointColorType.PerVert)

    mesh.insert(verts)
    mesh.insert(colors)
    mesh.insert(point_visualisation)

    viewer.run()
