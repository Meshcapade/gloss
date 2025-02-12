import json
from rich.progress import Progress
import os, os.path as osp
import open3d as o3d 
import cv2 
import numpy as np

def get_intrinsics_matrix(fx, fy, cx, cy):
    intrinsics_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def save_silhouette_and_depth(extrinsics, intrinsics, width, height, camera_id, scene):
    # Create rays and cast into scene
    rays = scene.create_rays_pinhole(intrinsics, extrinsics, width, height)
    ray_cast = scene.cast_rays(rays)

    # Get a hit mask and depth map and write to file 
    hit_t = ray_cast["t_hit"].numpy()
    bg_fg_arr = ~np.isinf(hit_t) * 255
    depth_map = np.zeros_like(hit_t)
    depth_map[~np.isinf(hit_t)] = hit_t[~np.isinf(hit_t)] 

    reference_dir_silhouette = osp.join(osp.dirname(osp.realpath(__file__)), '../silhouette_test/references')
    save_path_silhouette = osp.join(reference_dir_silhouette, str(camera_id) + '.png')
    cv2.imwrite(save_path_silhouette, bg_fg_arr)

    reference_dir_depth = osp.join(osp.dirname(osp.realpath(__file__)), '../depth_test/references')
    save_path_depth_tif = osp.join(reference_dir_depth, str(camera_id) + '.tiff')
    save_path_depth_png = osp.join(reference_dir_depth, str(camera_id) + '.png')
    cv2.imwrite(save_path_depth_png, (depth_map / depth_map.max()) * 255)
    cv2.imwrite(save_path_depth_tif, depth_map)

if __name__ == "__main__":
    root_path = osp.dirname(osp.realpath(__file__))
    json_path = osp.join(root_path, '../reference_cameras.json')

    data_path = osp.join(root_path, "../../../data")
    obj_tri_path = os.path.join(data_path, "bust.obj")
    assert osp.exists(obj_tri_path), "Mesh was not found!"

    mesh = o3d.io.read_triangle_mesh(obj_tri_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    with open(json_path, "r") as f:
        cameras = json.load(f)

    with Progress() as progress:
        task = progress.add_task("Rendering and saving references", total=len(cameras))
        for camera in cameras:
            extrinsics = np.array(camera["extrinsics"])
            intrinsics = get_intrinsics_matrix(*camera["intrinsics"])
            width = camera["width"]
            height = camera["height"]
            camera_id = camera["id"]

            save_silhouette_and_depth(extrinsics, intrinsics, width, height, camera_id, scene)

            progress.update(task, advance=1)