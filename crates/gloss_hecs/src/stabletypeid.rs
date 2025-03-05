// StableTypeId is not stable across binaries so dynamic loading will not work.
// For example this will cause python bindings of one create to not be able to
// communicate properly with python bindings of another crate. The two libraries
// are dynamically linked when imporing in python but the component that are
// added by one binding do not have the same type as the oen from the other
// crate, even if it's exactly the same rust type that was imported using
// use::other_crate::component. for this we neeed a stable id
// more info: https://github.com/bevyengine/bevy/issues/32
// https://github.com/PyO3/pyo3/issues/1444
// inspired by the original implementation of TypeId in rust
#![allow(missing_docs)]
#![allow(unused_imports)]
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use gloss_utils::abi_stable_aliases::StableAbi;

use identity_hash::IdentityHasher;
use std::any::TypeId;

// struct StableStableTypeId {}
// impl StableStableTypeId {
//     pub fn of<T: Component>(t: T) -> StableTypeId {
//         let StableTypeId = StableTypeId::
//     }
// }

// #[cfg(feature = "unsafe_typeid")]
// pub type StableTypeId as TypeId;

#[derive(Clone, Copy, Debug, Eq, PartialOrd, Ord)]
#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct StableTypeId {
    t: u64,
}

impl PartialEq for StableTypeId {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t
    }
}

impl Hash for StableTypeId {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // We only hash the lower 64 bits of our (128 bit) internal numeric ID,
        // because:
        // - The hashing algorithm which backs `TypeId` is expected to be unbiased and
        //   high quality, meaning further mixing would be somewhat redundant compared
        //   to choosing (the lower) 64 bits arbitrarily.
        // - `Hasher::finish` returns a u64 anyway, so the extra entropy we'd get from
        //   hashing the full value would probably not be useful (especially given the
        //   previous point about the lower 64 bits being high quality on their own).
        // - It is correct to do so -- only hashing a subset of `self` is still with an
        //   `Eq` implementation that considers the entire value, as ours does.
        (self.t).hash(state);
    }
}

impl StableTypeId {
    #[allow(clippy::similar_names)]
    pub fn of<T: ?Sized + 'static>() -> Self {
        // let t: u128 = intrinsics::type_id::<T>();
        #[cfg(not(feature = "unsafe_typeid"))]
        {
            let name = core::any::type_name::<T>();
            let mut s = DefaultHasher::new();
            name.hash(&mut s);
            let hashed = s.finish();
            Self { t: hashed }
        }
        #[cfg(feature = "unsafe_typeid")]
        {
            let ty = TypeId::of::<T>();
            //I want to get a u64 from the ty.t but since I can't access it I can just use
            // an identity hasher because I know that the hasher just gives as intput the
            // u64 that I want
            let mut hasher = IdentityHasher::<u64>::default();
            ty.hash(&mut hasher);
            let hashed = hasher.finish();
            Self { t: hashed }
        }
    }
}
