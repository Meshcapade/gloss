#!/usr/bin/env python3
"""
Test passing of torch cuda memory to vulkan directly
"""

import os

from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import (
    VertsPyTensor,
    ColorsPyTensor,
    VisPoints,
    ModelMatrix,
)
from gloss.types import PointColorType
import torch
import numpy as np

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()
    gpu = viewer.get_gpu()
    cam = viewer.get_camera()

    # get paths to data
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data"
    )

    mesh = viewer.get_or_create_entity(name="mesh")
    point_visualisation = VisPoints(
        show_points=True,
        point_size=10.0,
        color_type=PointColorType.PerVert,
    )
    mesh.insert(ModelMatrix.default())
    mesh.insert(point_visualisation)

    cam.set_position([0.5, 1.0, 3.0])  # xyz right-hand coordinate system
    cam.set_lookat([0.5, 0.5, 0.0])

    idx = 0
    while True:
        print(" ")

        val = np.abs(np.cos(idx * 0.01))
        val = int(2 + val * 25)
        # print("val", val)

        # make a cuda tensor of 3d points in a grid
        grid_size = val
        print("grid_size", grid_size)
        x = torch.arange(start=1, end=grid_size, device="cuda", dtype=torch.float32)
        y = torch.arange(start=1, end=grid_size, device="cuda", dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        zz = torch.zeros_like(xx)
        points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        print("CUDA grid points shape:", points.shape)
        points = points.unsqueeze(0)
        points /= grid_size

        # update the points
        # mesh.insert(VertsGPU.new_from_tensor(points, gpu))
        # mesh.insert(ColorsGPU.new_from_tensor(points, gpu))

        mesh.insert(VertsPyTensor(points))
        mesh.insert(ColorsPyTensor(points))

        # render
        viewer.start_frame()
        viewer.update()

        idx += 1
        print("idx, ", idx)
