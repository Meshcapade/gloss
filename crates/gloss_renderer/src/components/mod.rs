// #[macro_use]
pub mod bundles;
pub mod cam_comps;
pub mod light_comps;
pub mod mesh_cpu_comps;
pub mod mesh_gpu_comps;
pub mod misc_comps;
pub mod render_comps;

//when we do "use crate::components::*" make it so that we can just use
// directly the components without mentioning cam_comps for example
pub use bundles::*;
pub use cam_comps::*;
pub use light_comps::*;
pub use mesh_cpu_comps::*;
pub use mesh_gpu_comps::*;
pub use misc_comps::*;
pub use render_comps::*;
