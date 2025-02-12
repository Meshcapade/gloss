#!/usr/bin/env python3
"""
Show a mesh vertices as a point cloud
"""

import os

from gloss import Viewer, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import Verts, Colors, VisMesh, VisPoints

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    data_path = os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data")
    mesh_path = os.path.join(data_path,"bust.obj")

    # Create a Visualiser instance and an entity
    viewer = Viewer()

    smpl_body = viewer.get_or_create_entity(name = "smpl_body")
    smpl_body.insert_builder(geom.build_from_file(mesh_path))

    smpl_body.insert(
        VisMesh(show_mesh = False)
    )
    smpl_body.insert(VisPoints(show_points = True, point_size = 2.0))

    vertices = smpl_body.get(Verts).numpy()
    smpl_body.insert(Colors(vertices))

    viewer.run()
