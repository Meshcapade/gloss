#![deny(missing_docs)]

//! ## Crate Items Overview
//!
//! This section provides quick links to the main items in Gloss.
//!
//! ### Modules
//! - [`gloss_renderer`](crate::gloss_renderer) - The core renderer and viewer
//! - [`gloss_img`](crate::gloss_img) - Image-related functionality.
//! - [`easy_wgpu`](crate::easy_wgpu) - Abstractions for wgpu.
//! - [`utils_rs`](crate::utils_rs) - Utility functions and helpers.
//!
//! ## Examples
//! Below are the examples you can explore in the `examples/` folder of the
//! repository:
//!
//! - **Camera Control**: [cam_control.py](https://github.com/<your-repo>/examples/cam_control.py)
//! - **Depth Map**: [depth_map.py](https://github.com/<your-repo>/examples/depth_map.py)
//! - **Empty Scene**: [empty.py](https://github.com/<your-repo>/examples/empty.py)
//! - **Show Mesh as Point Cloud**: [show_mesh_as_point_cloud.py](https://github.com/<your-repo>/examples/show_mesh_as_point_cloud.py)
//!
//! These examples demonstrate various features of Gloss and can be run
//! directly.

#![doc = include_str!("../../../README.md")]

// Re-exports
pub use easy_wgpu;
pub use gloss_img;
pub use gloss_renderer;
pub use utils_rs;
