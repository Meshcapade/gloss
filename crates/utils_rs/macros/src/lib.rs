// extern crate proc_macro;
// use proc_macro::TokenStream;
// use quote::ToTokens;
// use syn::{DeriveInput, parse_macro_input};

// #[proc_macro_derive(DummyStableAbi)]
// pub fn derive(input: TokenStream) -> TokenStream {
//     parse_macro_input!(input as DeriveInput).to_token_stream().into()
// }

// extern crate proc_macro;
// use proc_macro::TokenStream;

// #[proc_macro_derive(DummyStableAbi, attributes(sabi))]
// pub fn dummy_stable_abi_derive(input: TokenStream) -> TokenStream {
//     input
// }

// #[proc_macro_derive(StableAbi, attributes(sabi))]
// pub fn derive_stable_abi(input: TokenStream1) -> TokenStream1 {
//     parse_or_compile_err(input, stable_abi::derive).into()
// }
