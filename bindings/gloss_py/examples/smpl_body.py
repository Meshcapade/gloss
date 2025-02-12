#!/usr/bin/env python3
"""
This example shows how to use the SMPLX package to create a body model and visualize it
The body model consists of vertices and faces as numpy arrays which can be easily passed to gloss
SMPLX package: https://github.com/vchoutas/smplx
"""

from smplx import SMPLX

from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import Verts, Faces

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    #smpl model from: https://smpl-x.is.tue.mpg.de/
    MODEL_PATH = "../../../data/smplx"
    smpl = SMPLX(MODEL_PATH)

    #forward pass
    body = smpl.forward()
    verts = Verts(body.v_shaped.squeeze().detach().numpy()) #verts as Nx3 numpy array
    faces = Faces(smpl.faces) #faces as Mx3 numpy array


    body = viewer.get_or_create_entity(name = "body")
    body.insert(verts)
    body.insert(faces)

    viewer.run()
