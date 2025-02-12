#!/usr/bin/env python3
"""
This example shows the minimal usage of gloss_py. 
It can read vertics and faces from numpy arrays and display the mesh in the viewer
"""

import numpy as np

from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import Verts, Faces

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    visualiser = Viewer()
    mesh = visualiser.get_or_create_entity(name = "test")

    verts = Verts(np.array([
        [0,0,0],
        [0,1,0],
        [1,0,0],
        [1.5,1.5,-1]
    ], dtype = "float32"))

    faces = Faces(np.array([
        [0,2,1],
        [1,2,3]
    ], dtype = "uint32"))

    mesh.insert(verts)
    mesh.insert(faces)
    verts_retrieved = mesh.get(Verts)

    print("Retrieved Verts from the component is ", verts.numpy())
    visualiser.run()
