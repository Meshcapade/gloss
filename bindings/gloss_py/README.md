# Gloss

Gloss is a light-weight Physically-based Renderer, written in Rust + wgpu

The main functionality includes loading meshes with high-resolution textures, rendering them with advanced graphics features, and allowing a general framework to explore new rendering techniques.

## Usage
Below is an example of a python3 script which shows a single mesh using the default viewing parameters. More examples for the python bindings can be found in the `bindings/gloss_py/examples` folder.
```python
import gloss

viewer = gloss.Viewer()
mesh = viewer.get_or_create_entity("mesh")
mesh.insert_builder(gloss.geom.build_from_file("my/mesh.obj")) 
viewer.run()
```
You can find more examples [here](https://github.com/Meshcapade/gloss/tree/main/bindings/gloss_py/examples).