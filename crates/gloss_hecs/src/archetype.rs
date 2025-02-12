// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// use crate::alloc::alloc::{alloc, dealloc, Layout};
use crate::{
    alloc::{
        alloc::{alloc, dealloc},
        vec,
    },
    layout::Layout,
    std::string::ToString,
};
// use core::any::{type_name, TypeId};
use crate::stabletypeid::StableTypeId;
use core::{
    any::type_name,
    fmt,
    hash::{BuildHasher, BuildHasherDefault, Hasher},
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};
// use std::println;

use hashbrown::hash_map::DefaultHashBuilder;

use crate::{borrow::AtomicBorrow, query::Fetch, Access, Component, ComponentRef, Query};

use utils_rs::abi_stable_aliases::std_types::{RHashMap, RStr, RVec, Tuple2};
#[cfg(target_arch = "wasm32")]
use utils_rs::abi_stable_aliases::FromSliceExt;
#[cfg(target_arch = "wasm32")]
use utils_rs::abi_stable_aliases::FromStrExt;
#[cfg(not(target_arch = "wasm32"))]
use utils_rs::abi_stable_aliases::StableAbi;

/// A collection of entities having the same component types
///
/// Accessing `Archetype`s is only required in niche cases. Typical use should
/// go through the [`World`](crate::World).
#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct Archetype {
    types: RVec<TypeInfo>,
    // type_ids: RSlice<'a, StableTypeId>,
    type_ids: RVec<StableTypeId>,
    index: OrderedStableTypeIdMap<usize>,
    len: u32,
    entities: RVec<u32>,
    /// One allocation per type, in the same order as `types`
    // data: RSlice<'a, Data>,
    data: RVec<Data>,
}

impl Archetype {
    fn assert_type_info(types: &[TypeInfo]) {
        types.windows(2).for_each(|x| match x[0].cmp(&x[1]) {
            core::cmp::Ordering::Less => (),
            #[cfg(debug_assertions)]
            core::cmp::Ordering::Equal => panic!(
                "attempted to allocate entity with duplicate {} components; \
                 each type must occur at most once!",
                x[0].type_name
            ),
            #[cfg(not(debug_assertions))]
            core::cmp::Ordering::Equal => panic!(
                "attempted to allocate entity with duplicate components; \
                 each type must occur at most once!"
            ),
            core::cmp::Ordering::Greater => panic!("type info is unsorted"),
        });
    }

    pub(crate) fn new(types: RVec<TypeInfo>) -> Self {
        let max_align = types.first().map_or(1, |ty| ty.layout.align());
        Self::assert_type_info(&types);
        let component_count = types.len();

        // let type_ids: RVec<StableTypeId> = types.iter().map(|ty| ty.id()).collect();
        let type_ids: RVec<StableTypeId> = types.iter().map(TypeInfo::id).collect();

        let data: RVec<Data> = (0..component_count)
            .map(|_| Data {
                state: AtomicBorrow::new(),
                storage: NonNull::new(max_align as *mut u8).unwrap(),
                mutated_entities: RVec::new(),
                added_entities: RVec::new(),
            })
            .collect();

        Self {
            // index: OrderedStableTypeIdMap::new(types.iter().enumerate().map(|(i, ty)| (ty.id, i))),
            index: OrderedStableTypeIdMap::new(types.iter().enumerate().map(|(i, ty)| Tuple2(ty.id, i))),
            type_ids,
            types,
            entities: RVec::new(),
            len: 0,
            data,
        }
    }

    pub(crate) fn clear(&mut self) {
        for (ty, data) in self.types.iter().zip(&*self.data) {
            for index in 0..self.len {
                unsafe {
                    let removed = data.storage.as_ptr().add(index as usize * ty.layout.size());
                    (ty.drop)(removed);
                }
            }
        }
        self.len = 0;
    }

    /// Whether this archetype contains `T` components
    pub fn has<T: Component>(&self) -> bool {
        self.has_dynamic(StableTypeId::of::<T>())
    }

    /// Whether this archetype contains components with the type identified by
    /// `id`
    pub fn has_dynamic(&self, id: StableTypeId) -> bool {
        self.index.contains_key(&id)
    }

    /// Find the state index associated with `T`, if present
    pub(crate) fn get_state<T: Component>(&self) -> Option<usize> {
        self.index.get(&StableTypeId::of::<T>()).copied()
    }

    /// Find the state index associated with `StableTypeId`, if present
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub(crate) fn get_state_by_id(&self, id: &StableTypeId) -> Option<usize> {
        self.index.get(id).copied()
    }

    #[allow(missing_docs)]
    #[inline]
    pub(crate) fn get_mutated(&self, state: usize) -> NonNull<bool> {
        // unsafe { NonNull::new_unchecked(self.data.get_unchecked(state).
        // mutated_entities.as_ptr() as *mut bool) }
        unsafe { NonNull::new_unchecked(self.data.get_unchecked(state).mutated_entities.as_ptr().cast_mut()) }
    }

    #[allow(missing_docs)]
    #[inline]
    pub(crate) fn get_added(&self, state: usize) -> NonNull<bool> {
        // unsafe { NonNull::new_unchecked(self.data.get_unchecked(state).
        // added_entities.as_ptr() as *mut bool) }
        unsafe { NonNull::new_unchecked(self.data.get_unchecked(state).added_entities.as_ptr().cast_mut()) }
    }

    #[allow(missing_docs)]
    pub(crate) fn clear_trackers(&mut self) {
        for data in self.data.iter_mut() {
            for mutated in data.mutated_entities.iter_mut() {
                *mutated = false;
            }
            for added in data.added_entities.iter_mut() {
                *added = false;
            }
        }
    }

    #[allow(missing_docs)]
    pub(crate) fn set_trackers_added(&mut self) {
        for data in self.data.iter_mut() {
            for added in data.added_entities.iter_mut() {
                *added = true;
            }
        }
    }

    #[allow(missing_docs)]
    pub(crate) fn set_trackers_changed(&mut self) {
        for data in self.data.iter_mut() {
            for mutated in data.mutated_entities.iter_mut() {
                *mutated = true;
            }
        }
    }

    /// Get the address of the first `T` component using an index from
    /// `get_state::<T>`
    pub(crate) fn get_base<T: Component>(&self, state: usize) -> NonNull<T> {
        assert_eq!(self.types[state].id, StableTypeId::of::<T>());

        // unsafe { NonNull::new_unchecked(self.data.get_unchecked(state).storage.
        // as_ptr().cast::<T>() as *mut T) }
        unsafe { NonNull::new_unchecked(self.data.get_unchecked(state).storage.as_ptr().cast::<T>()) }
    }

    /// Borrow all components of a single type from these entities, if present
    ///
    /// `T` must be a shared or unique reference to a component type.
    ///
    /// Useful for efficient serialization.
    pub fn get<'a, T: ComponentRef<'a>>(&'a self) -> Option<T::Column> {
        T::get_column(self)
    }

    pub(crate) fn borrow<T: Component>(&self, state: usize) {
        assert_eq!(self.types[state].id, StableTypeId::of::<T>());

        assert!(self.data[state].state.borrow(), "{} already borrowed uniquely", type_name::<T>());
        // if !self.data[state].state.borrow() {
        //     panic!("{} already borrowed uniquely", type_name::<T>());
        // }
    }

    pub(crate) fn borrow_mut<T: Component>(&self, state: usize) {
        assert_eq!(self.types[state].id, StableTypeId::of::<T>());
        assert!(self.data[state].state.borrow_mut(), "{} already borrowed", type_name::<T>());
        // if !self.data[state].state.borrow_mut() {
        //     panic!("{} already borrowed", type_name::<T>());
        // }
    }

    pub(crate) fn release<T: Component>(&self, state: usize) {
        assert_eq!(self.types[state].id, StableTypeId::of::<T>());
        self.data[state].state.release();
    }

    pub(crate) fn release_mut<T: Component>(&self, state: usize) {
        assert_eq!(self.types[state].id, StableTypeId::of::<T>());
        self.data[state].state.release_mut();
    }

    /// Number of entities in this archetype
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether this archetype contains no entities
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(crate) fn entities(&self) -> NonNull<u32> {
        // unsafe { NonNull::new_unchecked(self.entities.as_ptr() as *mut _) }
        unsafe { NonNull::new_unchecked(self.entities.as_ptr().cast_mut()) }
    }

    pub(crate) fn entity_id(&self, index: u32) -> u32 {
        self.entities[index as usize]
    }

    #[inline]
    pub(crate) fn set_entity_id(&mut self, index: usize, id: u32) {
        self.entities[index] = id;
    }

    pub(crate) fn types(&self) -> &[TypeInfo] {
        &self.types
    }

    pub(crate) fn type_ids(&self) -> &[StableTypeId] {
        &self.type_ids
    }

    /// Enumerate the types of the components of entities stored in this
    /// archetype.
    ///
    /// Convenient for dispatching logic which needs to be performed on sets of
    /// type ids.  For example, suppose you're building a scripting system,
    /// and you want to integrate the scripting language with your ECS. This
    /// functionality allows you to iterate through all of the archetypes of
    /// the world with [`World::archetypes()`](crate::World::archetypes()) and
    /// extract all possible combinations of component types which are
    /// currently stored in the `World`. From there, you can then create a
    /// mapping of archetypes to wrapper objects for your scripting language
    /// that provide functionality based off of the components of any given
    /// [`Entity`], and bind them onto an [`Entity`] when passed into your
    /// scripting language by looking up the [`Entity`]'s archetype using
    /// [`EntityRef::component_types`](crate::EntityRef::component_types).
    ///
    /// [`Entity`]: crate::Entity
    pub fn component_types(&self) -> impl ExactSizeIterator<Item = StableTypeId> + '_ {
        self.types.iter().map(|typeinfo| typeinfo.id)
    }

    /// `index` must be in-bounds or just past the end
    pub(crate) unsafe fn get_dynamic(&self, ty: StableTypeId, size: usize, index: u32) -> Option<NonNull<u8>> {
        debug_assert!(index <= self.len);
        Some(NonNull::new_unchecked(
            self.data
                .get_unchecked(*self.index.get(&ty)?)
                .storage
                .as_ptr()
                .add(size * index as usize)
                .cast::<u8>(),
        ))
    }

    /// Every type must be written immediately after this call
    pub(crate) unsafe fn allocate(&mut self, id: u32) -> u32 {
        if self.len as usize == self.entities.len() {
            self.grow(64);
        }

        self.entities[self.len as usize] = id;
        self.len += 1;
        self.len - 1
    }

    pub(crate) unsafe fn set_len(&mut self, len: u32) {
        debug_assert!(len <= self.capacity());
        self.len = len;
    }

    pub(crate) fn reserve(&mut self, additional: u32) {
        if additional > (self.capacity() - self.len()) {
            let increment = additional - (self.capacity() - self.len());
            self.grow(increment.max(64));
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn capacity(&self) -> u32 {
        self.entities.len() as u32
    }

    /// Increase capacity by at least `min_increment`
    fn grow(&mut self, min_increment: u32) {
        // Double capacity or increase it by `min_increment`, whichever is larger.
        self.grow_exact(self.capacity().max(min_increment));
    }

    /// Increase capacity by exactly `increment`
    fn grow_exact(&mut self, increment: u32) {
        let old_count = self.len as usize;
        let old_cap = self.entities.len();
        let new_cap = self.entities.len() + increment as usize;
        let mut new_entities = vec![!0; new_cap].into_boxed_slice();
        new_entities[0..old_count].copy_from_slice(&self.entities[0..old_count]);
        // let r_ent: RSlice<u32> = RSlice::from_slice(&new_entities);
        let r_ent: RVec<u32> = RVec::from_slice(&new_entities);
        self.entities = r_ent;

        let new_data: RVec<Data> = self
            .types
            .iter()
            .zip(&mut *self.data)
            .map(|(info, old)| {
                let storage = if info.layout.size() == 0 {
                    NonNull::new(info.layout.align() as *mut u8).unwrap()
                } else {
                    unsafe {
                        let mem = alloc(core::alloc::Layout::from_size_align(info.layout.size() * new_cap, info.layout.align()).unwrap());
                        ptr::copy_nonoverlapping(old.storage.as_ptr(), mem, info.layout.size() * old_count);
                        if old_cap > 0 {
                            dealloc(
                                old.storage.as_ptr(),
                                core::alloc::Layout::from_size_align(info.layout.size() * old_cap, info.layout.align()).unwrap(),
                            );
                        }
                        NonNull::new(mem).unwrap()
                    }
                };
                // let mut mutated_entities = old.mutated_entities.split_off(0);
                let mut mutated_entities: RVec<bool> = old.mutated_entities.drain(..).collect();
                mutated_entities.resize(new_cap, false);
                // let mut added_entities = old.added_entities.split_off(0);
                let mut added_entities: RVec<bool> = old.added_entities.drain(..).collect();
                added_entities.resize(new_cap, true);
                Data {
                    state: AtomicBorrow::new(), // &mut self guarantees no outstanding borrows
                    storage,
                    mutated_entities,
                    added_entities,
                }
            })
            .collect();

        // self.data = RSlice::from_slice(&new_data);
        self.data = new_data;
    }

    /// Returns the ID of the entity moved into `index`, if any
    pub(crate) unsafe fn remove(&mut self, index: u32, drop: bool) -> Option<u32> {
        let last = self.len - 1;
        for (ty, data) in self.types.iter().zip(&mut *self.data) {
            let removed = data.storage.as_ptr().add(index as usize * ty.layout.size());
            if drop {
                (ty.drop)(removed);
            }
            if index != last {
                let moved = data.storage.as_ptr().add(last as usize * ty.layout.size());
                ptr::copy_nonoverlapping(moved, removed, ty.layout.size());
                data.mutated_entities[index as usize] = data.mutated_entities[last as usize];
                data.added_entities[index as usize] = data.added_entities[last as usize];
            }
        }
        self.len = last;
        if index == last {
            None
        } else {
            self.entities[index as usize] = self.entities[last as usize];
            Some(self.entities[last as usize])
        }
    }

    /// Returns the ID of the entity moved into `index`, if any
    pub(crate) unsafe fn move_to(&mut self, index: u32, mut f: impl FnMut(*mut u8, StableTypeId, usize, bool, bool)) -> Option<u32> {
        let last = self.len - 1;
        for (ty, data) in self.types.iter().zip(&mut *self.data) {
            let moved_out = data.storage.as_ptr().add(index as usize * ty.layout.size());
            let is_added = data.added_entities[index as usize];
            let is_mutated = data.mutated_entities[index as usize];
            f(moved_out, ty.id(), ty.layout().size(), is_added, is_mutated);
            if index != last {
                let moved = data.storage.as_ptr().add(last as usize * ty.layout.size());
                ptr::copy_nonoverlapping(moved, moved_out, ty.layout.size());
                data.added_entities[index as usize] = data.added_entities[last as usize];
                data.mutated_entities[index as usize] = data.mutated_entities[last as usize];
            }
        }
        self.len -= 1;
        if index == last {
            None
        } else {
            self.entities[index as usize] = self.entities[last as usize];
            Some(self.entities[last as usize])
        }
    }

    pub(crate) unsafe fn put_dynamic(&mut self, component: *mut u8, ty: StableTypeId, size: usize, index: u32, added: bool, mutated: bool) {
        let data = self.data.get_unchecked_mut(*self.index.get(&ty).unwrap());
        // if added {
        //     data.added_entities[index as usize] = true;
        // }
        // if mutated {
        //     data.mutated_entities[index as usize] = true;
        // }
        data.added_entities[index as usize] = added;
        data.mutated_entities[index as usize] = mutated;
        let ptr = self.get_dynamic(ty, size, index).unwrap().as_ptr().cast::<u8>();
        ptr::copy_nonoverlapping(component, ptr, size);
    }

    /// How, if at all, `Q` will access entities in this archetype
    pub fn access<Q: Query>(&self) -> Option<Access> {
        Q::Fetch::access(self)
    }

    /// Add components from another archetype with identical components
    ///
    /// # Safety
    ///
    /// Component types must match exactly.
    pub(crate) unsafe fn merge(&mut self, mut other: Archetype) {
        self.reserve(other.len);
        for ((info, dst), src) in self.types.iter().zip(&*self.data).zip(&*other.data) {
            dst.storage
                .as_ptr()
                .add(self.len as usize * info.layout.size())
                .copy_from_nonoverlapping(src.storage.as_ptr(), other.len as usize * info.layout.size());
        }
        self.len += other.len;
        other.len = 0;
    }

    /// Raw IDs of the entities in this archetype
    ///
    /// Convertible into [`Entity`](crate::Entity)s with
    /// [`World::find_entity_from_id()`](crate::World::find_entity_from_id).
    /// Useful for efficient serialization.
    #[inline]
    pub fn ids(&self) -> &[u32] {
        &self.entities[0..self.len as usize]
    }
}

impl Drop for Archetype {
    fn drop(&mut self) {
        self.clear();
        if self.entities.is_empty() {
            return;
        }
        for (info, data) in self.types.iter().zip(&*self.data) {
            if info.layout.size() != 0 {
                unsafe {
                    dealloc(
                        data.storage.as_ptr(),
                        core::alloc::Layout::from_size_align_unchecked(info.layout.size() * self.entities.len(), info.layout.align()),
                    );
                }
            }
        }
    }
}

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
struct Data {
    state: AtomicBorrow,
    storage: NonNull<u8>,
    mutated_entities: RVec<bool>,
    added_entities: RVec<bool>,
}

/// A hasher optimized for hashing a single `StableTypeId`.
///
/// `StableTypeId` is already thoroughly hashed, so there's no reason to hash it
/// again. Just leave the bits unchanged.
#[derive(Default)]
pub(crate) struct StableTypeIdHasher {
    hash: u64,
}

impl Hasher for StableTypeIdHasher {
    fn write_u64(&mut self, n: u64) {
        // Only a single value can be hashed, so the old hash should be zero.
        debug_assert_eq!(self.hash, 0);
        self.hash = n;
    }

    // Tolerate StableTypeId being either u64 or u128.
    #[allow(clippy::cast_possible_truncation)]
    fn write_u128(&mut self, n: u128) {
        debug_assert_eq!(self.hash, 0);
        self.hash = n as u64;
    }

    fn write(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.hash, 0);

        // This will only be called if StableTypeId is neither u64 nor u128, which is
        // not anticipated. In that case we'll just fall back to using a
        // different hash implementation.
        let mut hasher = <DefaultHashBuilder as BuildHasher>::Hasher::default();
        hasher.write(bytes);
        self.hash = hasher.finish();
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A `HashMap` with `StableTypeId` keys
///
/// Because `StableTypeId` is already a fully-hashed u64 (including data in the
/// high seven bits, which hashbrown needs), there is no need to hash it again.
/// Instead, this uses the much faster no-op hash.
pub(crate) type StableTypeIdMap<V> = RHashMap<StableTypeId, V, BuildHasherDefault<StableTypeIdHasher>>;

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
// struct OrderedStableTypeIdMap<'a, V>(RSlice<'a, Tuple2<StableTypeId, V>>);
struct OrderedStableTypeIdMap<V>(RVec<Tuple2<StableTypeId, V>>);

impl<V> OrderedStableTypeIdMap<V> {
    fn new(iter: impl Iterator<Item = Tuple2<StableTypeId, V>>) -> Self {
        // fn new(iter: impl Iterator<Item = (StableTypeId, V)>) -> Self {
        // let mut vals = iter.collect::<Box<[_]>>();
        let mut vals: RVec<Tuple2<StableTypeId, V>> = iter.collect();
        // let vals_slice = vals.as_mut_slice()
        // vals.sort_unstable_by_key(|(id, _)| *id);
        vals.sort_unstable_by_key(|id| id.0);
        // vals_slice.sort_unstable_by_key(|(id, _)| id.0);
        // let rvals: RVec<Tuple2<StableTypeId, V>> = vals
        //     .iter()
        //     .map(|x| Tuple2::from_tuple((x.0, x.1)))
        //     .collect();
        let rvals = vals;
        // vals.iter().map(|x| Tuple2::).collect();
        // Self(RSlice::from_slice(&rvals))
        Self(rvals)
    }
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn search(&self, id: &StableTypeId) -> Option<usize> {
        self.0.binary_search_by_key(id, |Tuple2(id, _)| *id).ok()
    }
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn contains_key(&self, id: &StableTypeId) -> bool {
        self.search(id).is_some()
    }
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn get(&self, id: &StableTypeId) -> Option<&V> {
        self.search(id).map(move |idx| &self.0[idx].1)
    }
}

/// Metadata required to store a component.
///
/// All told, this means a [`StableTypeId`], to be able to dynamically
/// name/check the component type; a [`Layout`], so that we know how to allocate
/// memory for this component type; and a drop function which internally calls
/// [`core::ptr::drop_in_place`] with the correct type parameter.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
// #[sabi(unsafe_opaque_fields)]
pub struct TypeInfo {
    id: StableTypeId,
    layout: Layout,
    // drop: extern "C" unsafe fn(*mut u8),
    drop: unsafe extern "C" fn(*mut u8),
    // drop: unsafe fn(*mut u8),
    // #[cfg(debug_assertions)]
    // type_name: &'static RStr<'static>,
    type_name: RStr<'static>,
}

impl TypeInfo {
    /// Construct a `TypeInfo` directly from the static type.
    pub fn of<T: 'static>() -> Self {
        unsafe extern "C" fn drop_ptr<T>(x: *mut u8) {
            x.cast::<T>().drop_in_place();
        }

        Self {
            id: StableTypeId::of::<T>(),
            layout: Layout::new::<T>(),
            drop: drop_ptr::<T>,
            // drop: unsafe { std::mem::transmute(drop_ptr::<T>) },
            // drop: unsafe { drop_ptr::<T> as extern "C" fn(*mut u8) },
            // #[cfg(debug_assertions)]
            // type_name: &'static RStr::from_str(&'static core::any::type_name::<T>()),
            type_name: RStr::from_str(core::any::type_name::<T>()),
        }
    }

    /// Construct a `TypeInfo` from its components. This is useful in the rare
    /// case that you have some kind of pointer to raw bytes/erased memory
    /// holding a component type, coming from a source unrelated to hecs,
    /// and you want to treat it as an insertable component by implementing
    /// the `DynamicBundle` API.
    // pub fn from_parts(id: StableTypeId, layout: Layout, drop: unsafe fn(*mut u8)) -> Self {
    //     Self {
    //         id,
    //         layout,
    //         drop,
    //         // #[cfg(debug_assertions)]
    //         type_name: "<unknown> (TypeInfo constructed from parts)",
    //     }
    // }

    /// Access the `StableTypeId` for this component type.
    pub fn id(&self) -> StableTypeId {
        self.id
    }

    /// Access the `Layout` of this component type.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Access the `Layout` of this component type.
    pub fn name(&self) -> std::string::String {
        self.type_name.to_string()
    }

    /// Directly call the destructor on a pointer to data of this component
    /// type.
    ///
    /// # Safety
    ///
    /// All of the caveats of [`core::ptr::drop_in_place`] apply, with the
    /// additional requirement that this method is being called on a pointer
    /// to an object of the correct component type.
    pub unsafe fn drop(&self, data: *mut u8) {
        (self.drop)(data);
    }

    /// Get the function pointer encoding the destructor for the component type
    /// this `TypeInfo` represents.
    pub fn drop_shim(&self) -> unsafe extern "C" fn(*mut u8) {
        self.drop
    }
}

impl PartialOrd for TypeInfo {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TypeInfo {
    /// Order by alignment, descending. Ties broken with `StableTypeId`.
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.layout
            .align()
            .cmp(&other.layout.align())
            .reverse()
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialEq for TypeInfo {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TypeInfo {}

/// Shared reference to a single column of component data in an [`Archetype`]
pub struct ArchetypeColumn<'a, T: Component> {
    archetype: &'a Archetype,
    column: &'a [T],
}

impl<'a, T: Component> ArchetypeColumn<'a, T> {
    pub(crate) fn new(archetype: &'a Archetype) -> Option<Self> {
        let state = archetype.get_state::<T>()?;
        let ptr = archetype.get_base::<T>(state);
        let column = unsafe { core::slice::from_raw_parts(ptr.as_ptr(), archetype.len() as usize) };
        archetype.borrow::<T>(state);
        Some(Self { archetype, column })
    }
}

impl<T: Component> Deref for ArchetypeColumn<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.column
    }
}

impl<T: Component> Drop for ArchetypeColumn<'_, T> {
    fn drop(&mut self) {
        let state = self.archetype.get_state::<T>().unwrap();
        self.archetype.release::<T>(state);
    }
}

impl<T: Component> Clone for ArchetypeColumn<'_, T> {
    fn clone(&self) -> Self {
        let state = self.archetype.get_state::<T>().unwrap();
        self.archetype.borrow::<T>(state);
        Self {
            archetype: self.archetype,
            column: self.column,
        }
    }
}

impl<T: Component + fmt::Debug> fmt::Debug for ArchetypeColumn<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.column.fmt(f)
    }
}

/// Unique reference to a single column of component data in an [`Archetype`]
pub struct ArchetypeColumnMut<'a, T: Component> {
    archetype: &'a Archetype,
    column: &'a mut [T],
}

impl<'a, T: Component> ArchetypeColumnMut<'a, T> {
    pub(crate) fn new(archetype: &'a Archetype) -> Option<Self> {
        let state = archetype.get_state::<T>()?;
        let ptr = archetype.get_base::<T>(state);
        let column = unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), archetype.len() as usize) };
        archetype.borrow_mut::<T>(state);
        Some(Self { archetype, column })
    }
}

impl<T: Component> Deref for ArchetypeColumnMut<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.column
    }
}

impl<T: Component> DerefMut for ArchetypeColumnMut<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.column
    }
}

impl<T: Component> Drop for ArchetypeColumnMut<'_, T> {
    fn drop(&mut self) {
        let state = self.archetype.get_state::<T>().unwrap();
        self.archetype.release_mut::<T>(state);
    }
}

impl<T: Component + fmt::Debug> fmt::Debug for ArchetypeColumnMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.column.fmt(f)
    }
}
