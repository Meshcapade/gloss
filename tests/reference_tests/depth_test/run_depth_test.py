#!/usr/bin/env python3

# This test tests whether the depth map generated by gloss matches the one generated by Open3D (created using https://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html)
# The corresponding config for the test can be found in the `configs` folder in `tests/reference_tests`

import gloss
import os, os.path as osp
import numpy as np
import json
import cv2
from rich.progress import Progress

# To be called only once per process. Can select between Off,Error,Warn,Info,Debug,Trace
gloss.log.gloss_setup_logger(log_level=gloss.log.LogLevel.Info) 

def test_depth():
    root_path = osp.dirname(osp.realpath(__file__))
    json_path = osp.join(root_path, '../reference_cameras.json')
    config_file = 'depth_test.toml'
    config_path = osp.join(root_path, config_file)
    assert osp.exists(config_path),  f"Config for the test ({config_file}) does not seem to exist!\n \
                                    Configs found - {os.listdir(osp.join(root_path, 'configs'))}"

    # Get path to data 
    data_path = osp.join(root_path, "../../../data")
    obj_path = os.path.join(data_path, "bust.obj")
    assert osp.exists(obj_path), "Mesh was not found!"

    # Create the HeadlessViewer
    viewer = gloss.ViewerHeadless(2048, 2048, config_path) # Dummy height width, will be overwritten
    device = viewer.get_device()
    queue  = viewer.get_queue()
    scene  = viewer.get_scene()

    depth_error_means = []
    depth_error_max = []

    # Read a mesh from file and add it to the scene
    scene.get_or_create_entity("body") \
        .insert_builder(gloss.geom.build_from_file(obj_path))

    with open(json_path, "r") as f:
        cameras = json.load(f)

    with Progress() as progress:
        task = progress.add_task("Testing gloss against reference depths", total=len(cameras))
        for camera in cameras:

            # Set camera parameters
            cam = viewer.get_camera()
            cam.set_extrinsics(np.array(camera["extrinsics"], dtype = np.float32))
            cam.set_intrinsics(*camera["intrinsics"])
            cam.set_width_height(camera["width"], camera["height"])
            znear, zfar = cam.get_near_far()
            camera_id = camera["id"]
            
            # Render once
            viewer.start_frame()
            viewer.update()

            # Get the last image we rendered as a numpy array
            gloss_depth = viewer.get_final_depth()
            gloss_depth_np_remapped = np.squeeze(gloss_depth.depth_linearize(device, queue, znear, zfar))
            gloss_render = viewer.get_final_tex()
            gloss_render_np = gloss_render.numpy(device, queue)
            gloss_silhouette = (gloss_render_np[..., 3] / 255).astype(np.uint)

            path_to_reference = osp.join(root_path, "references", str(camera_id) + ".tiff")
            path_to_reference_silhouette = osp.join(root_path, "../silhouette_test", "references", str(camera_id) + ".png")
            reference_depth = cv2.imread(path_to_reference, cv2.IMREAD_UNCHANGED)
            reference_silhouette = (cv2.imread(path_to_reference_silhouette, cv2.IMREAD_UNCHANGED) / 255).astype(np.uint)
            silhouette_mask = np.bitwise_and(reference_silhouette, gloss_silhouette) == 1

            image_difference = gloss_depth_np_remapped[silhouette_mask] - reference_depth[silhouette_mask]

            depth_error = (abs(image_difference).mean())
            depth_error_means.append(abs(image_difference).mean())
            depth_error_max.append(abs(image_difference).max())

            assert depth_error < 1e-2, "[TEST FAILED] Image difference is larger than 10^-2"
            progress.update(task, advance=1)

    print("[ALL TESTS SUCCEEDED] with a mean depth difference of less than 10^-4")
    print("Maximum depth error for each of the references - ", depth_error_max)
    print("Mean depth error for each of the references - ", depth_error_means)