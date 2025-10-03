#!/usr/bin/env python3
"""
Functionality to record the screen rendered by gloss into a .png texture
"""

from gloss import Viewer
from gloss.builders import builders
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    visualiser = Viewer()
    cam = visualiser.get_camera()

    cube = visualiser.get_or_create_entity(name="cube")
    cube.insert_builder(builders.build_cube(center=[0, 0, 0], scale=1.0))

    PATH = "./img.png"
    while True:
        visualiser.start_frame()
        visualiser.update()  # updates the main screen

        visualiser.start_frame()
        visualiser.override_dt(
            0.0
        )  # this new render doesn't need to advance the global timer so we just set it to zero

        # updates the offscreen texture that will be later transfered to cpu and saved to png
        visualiser.update_offscreen_texture()
        visualiser.save_last_render(PATH)
