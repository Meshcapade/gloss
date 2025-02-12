#!/usr/bin/env python3
"""
This example shows an example that reads config file from a file. 
by default the configuration is read from gloss_renderer/config/default.toml 
However you can point to viewer to your own local config.toml file that 
shadows the default configuration.
Check the default.toml file to see all the setting you can change
"""

import os

from gloss import Viewer, geom
from gloss.log import LogLevel, LogLevelCaps, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info, \
             log_level_caps=LogLevelCaps(
                                  {"wgpu": LogLevel.Warn,
                                   "winit": LogLevel.Warn,
                                   "naga": LogLevel.Warn}
))


if __name__ == "__main__":
    #get paths to data
    path_root=os.path.dirname( os.path.realpath(__file__) )
    path_config=os.path.join(path_root,"../config/viewer_config_example.toml")
    path_data=os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data")
    path_obj=os.path.join(path_data,"bust.obj")

    viewer = Viewer(path_config)
    mesh = viewer.get_or_create_entity(name = "mesh")
    mesh.insert_builder(geom.build_from_file(path_obj))

    viewer.run()
