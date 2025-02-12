// #![warn(
//     clippy::all,
//     // clippy::restriction,
//     clippy::pedantic,
//     // clippy::nursery,
//     // clippy::cargo
// )]
// //some lints are really just too pedantic
// // clippy::module_name_repetitions, // This makes code confusing if you do
// widlcard ::* importing // clippy::must_use_candidate,
// // clippy::missing_const_for_fn
// // clippy::inefficient_to_string, // IMO if to_string is hot you have other
// problems // clippy::multiple_crate_versions, // I'm amazed you can deny this
// one in a codebase of any size // clippy::redundant_pub_crate, // This one
// disagrees with the rustc lint unreachable_pub in many cases which is a hard
// no from me // clippy::use_self, // As far as I can tell, this lint bans fn
// new() for types with a lifetime param // clippy::similar_names // Requires a
// lot of contorting if functions are large
// #![allow(clippy::module_name_repetitions)] // This makes code confusing if
// you do widlcard ::* importing #![allow(clippy::must_use_candidate)]

pub mod bind_group;
// pub mod bind_group_collection;
pub mod bind_group_layout;
// pub mod bufferpool;
//TODO this might not be necessary as it lives also in egui itsel
// #[cfg(feature = "with-gui")]
// pub mod egui_renderer;
pub mod framebuffer;
pub mod gpu;
pub mod pipeline;
// pub mod render_pass; //DOES NOT work because we return something that has to
// live within the current context, also it does not save us much coding space
// to use this abstraction
pub mod buffer;
pub mod mipmap;
pub mod texture;
pub mod utils;
