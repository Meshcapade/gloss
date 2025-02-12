#!/usr/bin/env python3
"""
Visualise an obj
"""

import os

from gloss import Viewer, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    # get paths to data
    path_data = os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data")
    path_obj = os.path.join(path_data,"bust.obj")

    mesh = viewer.get_or_create_entity(name = "mesh")
    mesh.insert_builder(geom.build_from_file(path_obj))

    viewer.run()
