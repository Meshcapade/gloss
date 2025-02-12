// For WASM builds, `extern` fn uses type `std::string::String`, which is not FFI-safe
// This is added to ignore those warnings
#![cfg_attr(target_arch = "wasm32", allow(improper_ctypes_definitions))]

pub mod gui;
pub mod plugins;
pub mod runner;
pub mod systems;

//when we do "use crate::plugins::*" make it so that we can just use directly
// the components without mentioning cam_comps for example
pub use plugins::*;
pub use runner::*;
pub use systems::*;
