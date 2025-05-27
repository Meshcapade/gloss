<div align="center">

# ✨ Gloss

**A light-weight Physically-based Renderer, made with Rust and wgpu**

[![Crates.io](https://img.shields.io/crates/v/gloss-rs.svg)](https://crates.io/crates/gloss-rs)
[![PyPI](https://img.shields.io/pypi/v/gloss-rs.svg)](https://pypi.org/project/gloss-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Meshcapade/gloss/LICENSE)

Gloss is a light-weight Physically-based Renderer written in Rust and wgpu. The main functionality includes loading meshes with high-resolution textures, rendering them with advanced graphics features, and allowing a general framework to explore new rendering techniques.
Gloss also compiles for Python and Web, allowing for rendering in multiple different environments.

<img alt="Gloss Banner" src="https://raw.githubusercontent.com/Meshcapade/gloss/main/imgs/banner.png">

</div>

## Documentation 
* [Gloss Rust API Documentation](https://docs.rs/gloss-rs/latest/gloss_rs/): Automatically generated docs for Gloss's Rust API
* [Gloss Rust Examples](https://github.com/Meshcapade/gloss/tree/main/examples): Gloss's runnable examples in Rust, covering basic usage. 
* [Gloss Python Examples](https://github.com/Meshcapade/gloss/tree/main/bindings/gloss_py/examples): Gloss's runnable examples for the Python bindings. Covers a wide range of features of the Python bindings. 


<!-- ## Usage
Below is an example of a python3 script which shows a single mesh using the default viewing parameters. More examples for the python bindings can be found in the `bindings/gloss_py/examples` folder.
```python
import gloss

viewer = gloss.Viewer()
mesh = viewer.get_or_create_entity("mesh")
mesh.insert_builder(gloss.geom.build_from_file("my/mesh.obj")) 
viewer.run()
``` -->
## Getting Started
The easiest way to get started with gloss is to install the Python bindings. 

```sh
$ pip install gloss-rs
```

Below is a basic example of a python script which shows a single mesh using the default viewing parameters. More examples for the python bindings can be found in the `bindings/gloss_py/examples` folder.

```python
import gloss

viewer = gloss.Viewer()

mesh = viewer.get_or_create_entity("mesh")
mesh.insert_builder(gloss.geom.build_from_file("my/mesh.obj")) 

viewer.run()
```

## Installation and Dependencies
The main dependency is installing Rust, as the rest of dependencies are handled by cargo. 
To install Rust, simply run the following in your terminal: 

```sh
$ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

Some additional dependencies for Linux:

```sh
$ sudo apt-get install libvulkan-dev vulkan-tools xorg-dev libxkbcommon-x11-dev
```

For `MacOs`, it should run out of the box.

<!-- Additional dependencies for WSL:
```sh
$ sudo apt-get install libvulkan-dev xorg-dev libxkbcommon-x11-dev
$ sudo add-apt-repository ppa:kisak/kisak-mesa
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install vulkan-tools
```
Afterwards, follow the instructions in: https://github.com/bevyengine/bevy/pull/5040#issuecomment-1494706996
You also cannot run this in docker because vulkan doesn't work in docker and WSL.
Also, you need to run in X11 and not in wayland, so you need to run:
unset WAYLAND_DISPLAY

Additional dependencies for MacOS:
```sh
$ 
``` -->

### For running the Rust examples
```sh
$ cd gloss
$ cargo run --bin gloss_view_mesh
```
![View Mesh Example](https://raw.githubusercontent.com/Meshcapade/gloss/main/imgs/mesh_view.png)

### For running the Python examples
```sh
$ cd gloss/bindings/gloss_py
$ pip install gloss-rs
$ ./examples/empty.py
```

### Build and run the Web Example
For Web, we use wasm-pack:
```sh
$ curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```
To build the web example, run:

```sh
$ cd gloss/examples/web
$ wasm-pack build --target web
```

To run the web example, we can create a dummy web server by opening another terminal and running:
```sh
$ cd gloss/examples/web
$ python -m http.server 
```
Finally, navigate to `http://0.0.0.0:8000/gloss_webpage/` in your browser of choice.

<!-- ## Examples

Various examples can be found in the `./examples` folder. A short description of each one is given here: 

| Name  | Description |
| ------------- | ------------- |
| Mesh View | ![Mesh View](https://raw.githubusercontent.com/Meshcapade/gloss/main/imgs/mesh_view.png) Visualizes a mesh with textures. <br /> Run with [cargo r --bin `gloss_view_mesh`](./examples/view_mesh) | -->

## Planned features
- `PyTorch` integration
- Differentiable rendering
- Area lights 
- Subsurface scattering
- Order-independent transparency 
- Support for Gaussian Splatting 

## Troubleshoot
If you have a laptop with both Intel graphics and NVIDIA, go to nvidia-settings and set the GPU to performance mode. Letting it run "on-demand" can cause issues with an external monitor: `<https://askubuntu.com/a/1447935>`

If there are any exceptions that mention "Maybe there's a driver issue on your machine?", please check that you have GPU vulkan drivers installed (`libvulkan-dev vulkan-tools`) and running `vulkaninfo | grep GPU` shows a real GPU and not a software solution like llvmpipe.

If there are still issues with "Maybe there's a driver issue on your machine", you can switch to GL backend with:
```sh
$ sudo apt-get install mesa-utils libegl-dev
$ sudo usermod -a -G render $USER
$ sudo usermod -a -G video $USER
RELOG 
$ LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 WGPU_BACKEND=gl MY_SCRIPT
```
Partially, this solution comes from "Solution 1" from `<https://stackoverflow.com/a/72427700>`, where it seems that conda can cause issues. 

Another solution might be: `conda install -c conda-forge libstdcxx-ng`

## Acknowledgements and Credits
* [Rerun](https://github.com/rerun-io/rerun)
* [Bevy](https://github.com/bevyengine/bevy)
* [HECS](https://github.com/Ralith/hecs)
* [wasm-log](https://github.com/s1gtrap/wasm-log)
* [Stall 2 mesh](https://www.sharetextures.com/models/building/stall_2)
* [Bust of Róża Loewenfeld](https://sketchfab.com/3d-models/sculpture-bust-of-roza-loewenfeld-fc6e731a0131471ba8e45511c7ea9996)