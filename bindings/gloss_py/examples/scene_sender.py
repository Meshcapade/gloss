#!/usr/bin/env python3
"""
Add some data to the ECS world and send it over the network
"""

import numpy as np
import time

from gloss import Viewer, ViewerDummy
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.types import PointColorType
from gloss.components import Verts, Colors, VisPoints
from gloss.network import SceneSender, TransportConfig

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = ViewerDummy()

    # setup scene for network sending
    transport_config = TransportConfig(
        # local
        address="127.0.0.1",
        port=46377,
        # server
        # address="0.0.0.0",
        # port=8888,
    )
    scene_sender = SceneSender(transport_config)
    scene_sender.start_listening()
    scene_sender.try_connect_to_receiver()
    time.sleep(2)  # give some time to set up the connection
    viewer.add_resource(scene_sender)
    print("start the scene_receiver_script if you haven't already")

    verts = Verts(
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1.5, 1.5, -1]], dtype="float32")
    )
    colors = Colors(
        np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]], dtype="float32")
    )

    # anything between start_batch_net_sending and end_batch_net_sending is sent together in one message an applied atomically.
    # usually desirable to avoid having entities that are half-updated
    viewer.start_batch_net_sending()

    mesh = viewer.get_or_create_entity(name="mesh")
    point_visualisation = VisPoints(
        show_points=True,
        show_points_indices=True,
        point_size=10.0,
        color_type=PointColorType.PerVert,
    )
    mesh.insert(verts)
    mesh.insert(colors)
    mesh.insert(point_visualisation)

    viewer.end_batch_net_sending()

    while True:
        viewer.start_batch_net_sending()
        mesh = viewer.get_or_create_entity(name="mesh")
        mesh.insert(verts)
        mesh.insert(colors)
        mesh.insert(point_visualisation)
        viewer.end_batch_net_sending()
        time.sleep(1)
        pass
