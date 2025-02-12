#!/usr/bin/env python3
"""
This example shows how to use a headless renderer and save the image to .png file
"""

import os

from gloss import ViewerHeadless, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    viewer = ViewerHeadless(2048, 2048)

    #get paths to data
    data_path = os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data")
    mesh_path = os.path.join(data_path,"bust.obj")

    smpl_body = viewer.get_or_create_entity(name = "smpl_body")
    smpl_body.insert_builder(geom.build_from_file(mesh_path))

    cam = viewer.get_camera()
    cam.set_position([0.0, 1.5, 2.0]) #xyz right-hand coordinate system
    cam.set_lookat([0.0, 1.0, 0.0])
    # or
    # cam.set_extrinsics(np.array([
    #     [1.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0],
    #     [0.0, -1.0, 0.0, 8.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ], dtype=np.float32))

    # This overwrites the resolution you initially set to ViewerHeadless
    # cam.set_width_height(2048, 2048)

    cam.set_intrinsics(1600.0, 1600.0, 1024.0, 1024.0)

    viewer.render_next_frame()
    viewer.save_last_render("./render.png")
