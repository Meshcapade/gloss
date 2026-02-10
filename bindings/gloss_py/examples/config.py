#!/usr/bin/env python3
"""
Show how to define and use a custom config file
"""

import os

from gloss import Viewer, Config
from gloss.builders import builders
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.components import Verts, Colors, VisMesh, VisPoints

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data"
    )
    mesh_path = os.path.join(data_path, "bust.obj")

    # any fields you set in this config will overwrite the ones set in the default
    # For the fields that you can set in the config check gloss_renderer/config/default.toml
    config = Config.new_from_str(
        "\
                               [core] \n \
                               auto_add_floor = false"
    )
    viewer = Viewer.new_with_config(config)

    smpl_body = viewer.get_or_create_entity(name="smpl_body")
    smpl_body.insert_builder(builders.build_from_file(mesh_path))

    viewer.run()
