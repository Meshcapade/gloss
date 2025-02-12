#!/usr/bin/env python3
"""
This example shows how to use a headless renderer and get a depth map
One of the primary differences between this example and headless.py is that
MSAA cannot be used when trying to get depths.
"""
import os
import os.path as osp
import cv2
import numpy as np

from gloss import ViewerHeadless, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    root_path = osp.dirname(osp.realpath(__file__))
    config_path = osp.join(root_path, '../config', 'depth_example.toml')
    assert osp.exists(config_path),  f"Config for the test ({config_path}) does not exist!\n \
                                    Configs found - {os.listdir(osp.join(root_path, '../config'))}"

    # We have to use the config here to set msaa samples to 1
    # Retrieving depth maps with msaa with samples > 1 is not supported
    viewer = ViewerHeadless(2048, 2048, config_path) # Resolution of the rendering window
    # Get paths to data
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../../../data")
    mesh_path = os.path.join(data_path,"bust.obj")

    # Create a mesh entity, add a mesh builder to it, and add it to the scene
    smpl_body = viewer.get_or_create_entity(name = "smpl_body")
    smpl_body.insert_builder(geom.build_from_file(mesh_path))

    # Set Camera parameters
    cam = viewer.get_camera()
    # cam.set_position(0.0, 3.0, 4.0) #xyz right-hand coordinate system
    # cam.set_lookat(0.0, 1.0, 0.0)
    # or
    cam.set_extrinsics(np.array([
        [-0.19913687965295618,-0.47458859076048726,0.8573856615794662,0.7913859597626586],
        [0.0,-0.8749086068608676,-0.48428806473087466,1.3802989124878067],
        [0.9799716848777235,-0.09643961406367524,0.17422656995178815,1.689456300513285],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32))

    # cam.set_width_height(2048, 2048) # overwrites the resolution initially set in ViewerHeadless
    cam.set_intrinsics(1600.0, 1600.0, 1024.0, 1024.0)

    # render once
    viewer.render_next_frame()

    # retrieve linearised (in metric) depth map of the last rendered frame
    depth_map = viewer.get_linearised_depth()

    # normalise depth map for easier visualisation
    gloss_depth_image = ((depth_map / depth_map.max()) * 255.0).astype(np.uint8)
    cv2.imwrite("./depth.png", gloss_depth_image)
