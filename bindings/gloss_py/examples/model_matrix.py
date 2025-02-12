#!/usr/bin/env python3
"""
Shows functionality of the model matrix which is the transform from object coordinates
to the world coordinates
"""

from gloss import Entity, Viewer, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import ModelMatrix

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    visualiser = Viewer()
    cam = visualiser.get_camera()

    cube = visualiser.get_or_create_entity(name = "cube")
    cube.insert_builder(geom.build_cube(center = [0, 0, 0]))

    model_matrix = ModelMatrix.default()\
            .with_translation([0, 2.0, 0])\
            .with_rotation_euler([1.0, 0.0, 0])

    cube.insert(model_matrix)
    visualiser.run()
