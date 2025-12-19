#!/usr/bin/env python3
"""
Test passing of torch cuda memory to vulkan directly
"""

import os

from gloss import Viewer, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import VisOutline, DiffuseImg, DiffuseTex
from gloss.types import ImgConfig
from gloss.builders import builders
import torch
import torchvision
import numpy as np

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()
    gpu = viewer.get_gpu()

    # get paths to data
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data"
    )
    path_obj = os.path.join(path_data, "bust.obj")
    path_img = os.path.join(path_data, "bust_alb.jpg")

    mesh = viewer.get_or_create_entity(name="mesh")
    mesh.insert_builder(builders.build_from_file(path_obj))
    mesh.insert(
        DiffuseImg(
            path_img,
            ImgConfig(generate_mipmaps=False),
        )
    )

    # create a cuda tensor corresponding to an image
    t_img = torchvision.io.read_image(path_img)
    t_img = t_img.unsqueeze(0).cuda()

    idx = 0
    while True:
        print(" ")

        # render in order to initialize the diffuseTex component
        viewer.start_frame()
        viewer.update()

        tex = mesh.get(DiffuseTex)

        # modify the cuda tensor
        val = np.abs(np.cos(idx * 0.01))
        t_img_modif = (t_img * val).byte()

        # update the vulkan texture
        tex.from_tensor(t_img_modif, gpu)
        mesh.insert(tex)

        idx += 1
        print("idx, ", idx)
