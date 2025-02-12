#!/usr/bin/env python3
"""
This example shows an empty window in which you can drag an drop meshes or .smpl bodies
"""
from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level = LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()
    viewer.run()
