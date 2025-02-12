#!/usr/bin/env python3
"""
Visualise lines
"""

import numpy as np

from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import Verts, Edges, VisLines, VisPoints

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    lines = viewer.get_or_create_entity(name = "lines")
    verts = Verts(np.array([
        [1,0,0],
        [0,1,0],
        [0,0,0],
        [1,1,0]
    ], dtype = "float32"))

    # each row of this matrix represent a line that is drawn by indexing into Verts.
    # So the first line is drawn between verts[0] and verts[1]
    edges = Edges(np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,0],
        [2,0],
        [3,1],
    ], dtype = "uint32"))

    lines.insert(verts)
    lines.insert(edges)

    # These are settings for visualisation of lines and points
    line_visualisation = VisLines(show_lines=True, line_width=10.0, zbuffer=False,
                                  line_color=[1.0, 0.2, 0.2, 1.0])
    point_visualisation = VisPoints(show_points=True, point_size=10.0, zbuffer=False,
                                    point_color=[0.0, 1.0, 1.0, 1.0])

    lines.insert(point_visualisation)
    lines.insert(line_visualisation)

    viewer.run()
