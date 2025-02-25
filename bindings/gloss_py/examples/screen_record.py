#!/usr/bin/env python3
"""
Functionality to record the screen rendered by gloss into a .png texture
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
    
    
    path="./img.png"
    while True:
        visualiser.start_frame()
        visualiser.update()  #updates the main screen

        visualiser.start_frame()
        visualiser.override_dt(0.0) #this new render doesn't need to advance the global timer so we just set it to zero
        visualiser.update_offscreen_texture()  #updates the offscreen texture that will be later transfered to cpu and saved to png

        visualiser.save_last_render(path)