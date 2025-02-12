// https://rustwasm.github.io/docs/book/game-of-life/debugging.html
// A macro to provide `println!(..)`-style syntax for `console.log` logging.
// #[cfg(target_arch = "wasm32")]
#[allow(unused_macros)]
#[macro_export]
macro_rules! log {
    ( $( $t:tt )* ) => {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

// https://stackoverflow.com/questions/59984712/rust-macro-to-convert-between-identical-enums
#[allow(unused_macros)]
#[macro_export]
macro_rules! convert_enum_from{($src: ident, $dst: ident, $($variant: ident,)*)=> {
    impl From<$src> for $dst {
        fn from(src: $src) -> Self {
            match src {
                $($src::$variant => Self::$variant,)*
            }
        }
    }
}}
#[allow(unused_macros)]
#[macro_export]
macro_rules! convert_enum_into{($src: ident, $dst: ident, $($variant: ident,)*)=> {
    impl Into<$dst> for $src {
        fn into(self) -> $dst {
            match self {
                $(Self::$variant => $dst::$variant,)*
            }
        }
    }
}}
