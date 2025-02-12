#!/usr/bin/env python3
"""
This shows how to set intrinsics (focal point and principal point) 
and also extrinsics (position, rotation) of the camera and how to render from it
"""

import os
import numpy as np

from gloss import ViewerHeadless, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    viewer = ViewerHeadless(2048, 2048)
    cam = viewer.get_camera()

    # Get paths to data
    path_data=os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data")
    path_obj=os.path.join(path_data,"bust.obj")

    mesh = viewer.get_or_create_entity(name = "mesh")
    mesh.insert_builder(geom.build_from_file(path_obj))

    cam.set_width_height(2048, 2048)
    extrinsics = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 1.5],
        [0.0, 0.0, -1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    cam.set_extrinsics(extrinsics)

    #instrinsics
    fx,fy,cx,cy = (1600.0, 1600.0, 1024.0, 1024.0)
    cam.set_intrinsics(fx,fy,cx,cy)

    viewer.render_next_frame()
    viewer.save_last_render("./image_intrinsics_extrinsics.png")
