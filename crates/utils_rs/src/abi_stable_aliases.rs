// For non-wasm targets, re-export the real abi_stable
#[cfg(not(target_arch = "wasm32"))]
pub use abi_stable::*;

// For wasm target, define some standard type aliases
#[cfg(target_arch = "wasm32")]
pub mod std_types {
    pub type RString = String;
    pub type RDuration = Duration;

    use std::collections::hash_map::RandomState;
    pub type RVec<T> = Vec<T>;

    //RVec has a function called from_slice that I also want defined on Vec
    pub trait FromSliceExt<T> {
        fn from_slice(slice: &[T]) -> Self
        where
            T: Clone;
    }
    impl<T> FromSliceExt<T> for RVec<T> {
        fn from_slice(slice: &[T]) -> Self
        where
            T: Clone,
        {
            Vec::from(slice)
        }
    }

    pub type RStr<'a> = &'a str;

    //RStr has a function called from_str that I also want defined on str
    pub trait FromStrExt<'a> {
        fn from_str(string: &'a str) -> Self;
    }

    impl<'a> FromStrExt<'a> for RStr<'a> {
        fn from_str(string: &'a str) -> Self {
            string // Just returns the string as it is
        }
    }

    // pub type RHashMap<K, V> = std::collections::HashMap<K, V>;
    pub type RHashMap<K, V, S = RandomState> = std::collections::HashMap<K, V, S>;

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
    pub struct Tuple2<A, B>(pub A, pub B);

    pub mod map {
        pub use std::collections::hash_map::Entry as REntry;
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
    #[repr(u8)]
    pub enum ROption<T> {
        ///
        RSome(T),
        ///
        RNone,
    }

    // use std::ops::Deref;
    use std::time::Duration;

    pub use self::ROption::*;

    #[allow(clippy::missing_const_for_fn)]
    impl<T> ROption<T> {
        #[inline]
        pub const fn as_ref(&self) -> ROption<&T> {
            match self {
                RSome(v) => RSome(v),
                RNone => RNone,
            }
        }
        #[inline]
        pub fn as_mut(&mut self) -> ROption<&mut T> {
            match self {
                RSome(v) => RSome(v),
                RNone => RNone,
            }
        }
        #[inline]
        pub const fn is_some(&self) -> bool {
            matches!(self, RSome { .. })
        }
        #[inline]
        pub const fn is_none(&self) -> bool {
            matches!(self, RNone { .. })
        }
        #[inline]
        pub fn into_option(self) -> Option<T> {
            // self.into()
            match self {
                RSome(v) => Option::Some(v),
                RNone => Option::None,
            }
        }
        #[inline]
        pub fn unwrap(self) -> T {
            self.into_option().unwrap()
        }
        #[inline]
        pub fn map<U, F>(self, f: F) -> ROption<U>
        where
            F: FnOnce(T) -> U,
        {
            match self {
                RSome(x) => RSome(f(x)),
                RNone => RNone,
            }
        }
    }

    /// The default value is `RNone`.
    impl<T> Default for ROption<T> {
        fn default() -> Self {
            RNone
        }
    }
}

// Re-export standard types for wasm target
#[cfg(target_arch = "wasm32")]
pub use std_types::*;

// For WASM, define an empty StableAbi trait so that references compile.
#[cfg(target_arch = "wasm32")]
pub trait StableAbi {}
