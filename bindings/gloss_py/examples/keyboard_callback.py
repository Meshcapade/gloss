#!/usr/bin/env python3
"""
This example shows how to listen to keyboard events and modify the viewer based on them
"""

import numpy as np
from pynput import keyboard

from gloss import Viewer, geom
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import VisMesh

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    visualiser = Viewer()
    cam = visualiser.get_camera()

    cube = visualiser.get_or_create_entity(name = "cube")
    cube.insert_builder(geom.build_cube(center = [0, 1, 0]))

    with keyboard.Events() as events:
        while True:
            # Don't block waiting for events
            event = events.get(0.0)
            if event is not None:
                print(f'Received event {event}')
                # We have to do this because editing the entity here would edit cube (defined above)
                # We want to edit the entity thats been added to the scene
                cube.insert(VisMesh(solid_color= np.random.rand(4)))

            visualiser.render_next_frame()
