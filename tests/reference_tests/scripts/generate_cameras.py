import json
import numpy as np
import os.path as osp
from rich.progress import Progress

def generate_extrinsics(lookat, radius, height, theta):
    # Generate extrinsics matrix for a camera orbiting around the lookat point and looking at it
    lookfrom = np.array([radius * np.cos(theta), height, radius * np.sin(theta)])
    direction = lookat - lookfrom
    norm_direction = direction / np.linalg.norm(direction)


    # Compute the right direction by taking the cross product of direction and up direction
    up_direction = np.array([0, 1, 0])
    right = np.cross(norm_direction, up_direction)
    right = right / np.linalg.norm(right)  # Normalize the right direction
    up = np.cross(right, norm_direction)

    # Create the rotation matrix
    rotation_matrix = np.vstack((right, -up, norm_direction)).T
    
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = -rotation_matrix @ lookfrom
    return extrinsics

def generate_intrinsics():
    # Generate intrinsics with random offsets
    # These values are derived from default gloss camera parameters
    fx = 1600 + np.random.randint(-200, 200)
    fy = 1600 + np.random.randint(-200, 200)
    cx = 1024 + np.random.randint(-100, 100)
    cy = 1024 + np.random.randint(-100, 100)
    return [fx, fy, cx, cy]

def generate_viewport():
    # Generate random viewport dimensions with offsets
    width  = 2048 + np.random.randint(-200, 200)
    height = 2048 + np.random.randint(-200, 200)
    return [width, height]

def generate_camera(lookat, radius_range, height_range, num_cameras):
    cameras = []
    with Progress() as progress:
        task = progress.add_task("Generating Cameras", total=num_cameras)
        for i in range(num_cameras):
            progress.update(task, advance=1)
            radius = np.random.uniform(*radius_range)
            height = np.random.uniform(*height_range)
            theta = np.random.uniform(0, 2 * np.pi)
            # theta = np.linspace(0, 2 * np.pi, num_cameras)[i]

            extrinsics = generate_extrinsics(lookat, radius, height, theta)
            intrinsics = generate_intrinsics()
            viewport = generate_viewport()

            camera = {
                "extrinsics": extrinsics.tolist(),
                "intrinsics": intrinsics,
                "height": viewport[1],
                "width": viewport[0],
                "id": i
            }
            cameras.append(camera)
    return cameras

if __name__ == "__main__":
    lookat = np.array([0, 0.9, 0]) # This looks at roughly the center of the mesh (to avoid empty renders)
    radius_range = (1.0, 2.0)  # Range of radii for cameras, the farthest point of the mesh has a radius of 0.48
    height_range = (0.5, 2.0)  # Range of heights for cameras, the range for the mesh is 0 -> 1.66
    num_cameras = 8  # Number of cameras to generate

    cameras = generate_camera(lookat, radius_range, height_range, num_cameras)

    root_path = osp.dirname(osp.realpath(__file__))
    json_path = osp.join(root_path, '../reference_cameras.json')

    with open(json_path, "w") as f:
        json.dump(cameras, f, indent=4)