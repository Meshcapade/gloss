#![deny(missing_docs)]

//! ## Crate Items Overview
//!
//! This section provides quick links to the main items in Gloss.
//!
//! ### Modules
//! - [`gloss_renderer`](crate::gloss_renderer) - The core renderer and viewer
//! - [`gloss_img`](crate::gloss_img) - Image-related functionality.
//! - [`easy_wgpu`](crate::easy_wgpu) - Abstractions for wgpu.
//! - [`gloss_hecs`](crate::gloss_hecs) - A wrapper around hecs forECS-related functionality.
//! - [`gloss_utils`](crate::gloss_utils) - Utility functions and helpers.
//! - [`gloss_geometry`](crate::gloss_geometry) - Geometry-related functionality.
//!
//! ## Examples
//! Below are the examples you can explore in the `examples/` folder of the
//! repository:
//!
//! - **Camera Control**: [cam_control.py](https://github.com/Meshcapade/gloss/bindings/gloss_py/examples/cam_control.py)
//! - **Depth Map**: [depth_map.py](https://github.com/Meshcapade/gloss/bindings/gloss_py/examples/depth_map.py)
//! - **Empty Scene**: [empty.py](https://github.com/Meshcapade/gloss/bindings/gloss_py/examples/empty.py)
//! - **Show Mesh as Point Cloud**: [show_mesh_as_point_cloud.py](https://github.com/Meshcapade/gloss/bindings/gloss_py/examples/show_mesh_as_point_cloud.py)
//!
//! These examples demonstrate various features of Gloss and can be run
//! directly.

// Re-exports
pub use easy_wgpu;
pub use gloss_geometry;
pub use gloss_img;
pub use gloss_renderer;
pub use gloss_utils;
