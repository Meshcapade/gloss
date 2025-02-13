#!/usr/bin/env python3
"""
This example shows how to create and control the gloss camera
"""
from gloss import Viewer, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    # Create a Viewer instance and get the corresponding camera
    viewer = Viewer()
    cam = viewer.get_camera()

    # Create an empty entity, attach a cube builder to it, and add to Visualiser scene
    # ``name`` is a unique identifier for an entity
    cube = viewer.get_or_create_entity(name = "Cube")
    cube.insert_builder(geom.build_cube(center = [0, 1.0, 0]))

    # Set Camera parameters
    cam.set_position([1.0, 2.5, 5.0]) # xyz right-hand coordinate system
    cam.set_lookat([0.0, 1.0, 0.0])
    # or
    # cam.set_extrinsics(np.array([
    #     [1.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0],
    #     [0.0, -1.0, 0.0, 8.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ], dtype=np.float32))

    # Make the camera orbit at a steady rate
    while True:
        dt = viewer.start_frame()
        cam.orbit_y(degrees = 10.0 * dt)
        viewer.update()
