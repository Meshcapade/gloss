#!/usr/bin/env python3
"""
Example that should be paired with scene_sender.py This is the receiver part of scene_sender.py
"""

import numpy as np
import time

from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.types import PointColorType
from gloss.components import Verts, Colors, VisPoints
from gloss.network import SceneReceiver, TransportConfig, SceneReceiverPlugin

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    # setup scene for network sending
    transport_config = TransportConfig(
        # local
        address="127.0.0.1",
        port=46377,
        # server
        # address="100.115.140.113",
        # port=8888,
    )
    scene_receiver = SceneReceiver(transport_config)

    # Actually connect to the sender - this will enable the receiver
    # print("Attempting to connect to sender...")
    # scene_receiver.connect_to_sender()
    # print("Successfully connected to sender!")

    # connected=scene_receiver.try_connect_to_sender()
    # while not connected:
    #     print("attempting to connect to sender!")
    #     time.sleep(0.1)
    #     connected=scene_receiver.try_connect_to_sender()

    viewer.add_resource(scene_receiver)
    # plugin also
    scene_receiver_plugin = SceneReceiverPlugin(autorun=True)
    viewer.insert_plugin(scene_receiver_plugin)

    viewer.run()
