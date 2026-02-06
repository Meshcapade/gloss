#!/usr/bin/env python3
"""
Read an image from a file into a numpy array and visualize it in the Gloss gui
"""

import os
import cv2
import numpy as np

from gloss import Viewer
from gloss.components import DiffuseImg
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    # get paths to data
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data"
    )
    path_texture = os.path.join(path_data, "bust_alb.jpg")

    # read the file directly
    alebo_from_file_entity = viewer.get_or_create_gui_entity(name="albedo_from_file")
    alebo_from_file_entity.insert(DiffuseImg(path_texture))

    # create an image from numpy array
    img_bgr_numpy = cv2.imread(path_texture)
    img_rgb_numpy = cv2.cvtColor(img_bgr_numpy, cv2.COLOR_BGR2RGB)
    print("img_numpy shape:", img_rgb_numpy.shape, " dtype:", img_rgb_numpy.dtype)
    alebo_from_numpy_entity = viewer.get_or_create_gui_entity(name="albedo_from_numpy")
    alebo_from_numpy_entity.insert(DiffuseImg.from_numpy_u8_hw3(img_rgb_numpy))

    # viewer.run()
    while True:
        viewer.start_frame()
        # img_rgb_numpy=img_rgb_numpy+1
        # img_rgb_numpy=img_rgb_numpy.astype(np.uint8)
        # alebo_from_numpy_entity.insert(DiffuseImg.from_numpy_u8_hw3(img_rgb_numpy))
        viewer.update()
