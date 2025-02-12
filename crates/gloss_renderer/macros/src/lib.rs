// // extern crate proc_macro;

// use proc_macro::TokenStream;
// // use quote::quote;
// // use syn::{parse_macro_input, Type};

// #[proc_macro]
// pub fn sys_fn(input: TokenStream) -> TokenStream {
//     #[cfg(target_arch = "wasm32")]
//     let result = input;

//     #[cfg(not(target_arch = "wasm32"))]
//     let result = format!("extern \"C\" {}", input).parse().unwrap(); // Not a type macro

//     result
// }
