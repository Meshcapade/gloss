pub mod render_passes;
pub mod render_platform;
pub mod renderer;

//when we do "use crate::renderer::*" make it so that we can just use directly
// the components without mentioning cam_comps for example
pub use render_passes::*;
pub use render_platform::*;
pub use renderer::*;
