use crate::stabletypeid::StableTypeId;
use core::{marker::PhantomData, ptr::NonNull};

use crate::{Access, Archetype, Component, Fetch, Query};

/// Query that retrieves mutation state of type `T` component.
/// Added components do not count as mutated.
///
/// It is your responsibility to clear trackers with
/// [`World::clear_trackers()`](crate::World::clear_trackers()) at the start of
/// the frame (or any other suitable moment).
///
/// # Example
/// ```
/// # use gloss_hecs::*;
/// let mut world = World::new();
/// let e = world.spawn((123,));
/// for (_id, (value, value_mut)) in world.query::<(&i32, Mutated<i32>)>().iter() {
///     assert_eq!(*value, 123, "!1");
///     assert_eq!(value_mut, false, "!2");
/// }
/// for (_id, mut value) in world.query::<&mut i32>().iter() {
///     *value = 42;
/// }
/// for (_id, (value, value_mut)) in world.query::<(&i32, Mutated<i32>)>().iter() {
///     assert_eq!(*value, 42, "!3");
///     assert_eq!(value_mut, true, "!3a");
/// }
/// world.clear_trackers();
/// for (_id, value_mut) in world.query::<Mutated<i32>>().iter() {
///     assert_eq!(value_mut, false, "!4");
/// }
/// ```
pub struct Mutated<T>(PhantomData<fn(T)>);

impl<T: Component> Query for Mutated<T> {
    type Fetch = FetchMutated<T>;
}

#[doc(hidden)]
pub struct FetchMutated<T>(NonNull<bool>, PhantomData<fn(T)>);

unsafe impl<'a, T: Component> Fetch<'a> for FetchMutated<T> {
    type Item = bool;

    type State = usize;

    fn dangling() -> Self {
        Self(NonNull::dangling(), PhantomData)
    }

    fn access(archetype: &Archetype) -> Option<Access> {
        if archetype.has::<T>() {
            Some(Access::Read)
        } else {
            None
        }
    }

    fn borrow(_archetype: &Archetype, _state: Self::State) {}
    fn prepare(archetype: &Archetype) -> Option<Self::State> {
        archetype.get_state::<T>()
    }
    fn execute(archetype: &'a Archetype, state: Self::State) -> Self {
        Self(archetype.get_mutated(state), PhantomData)
    }
    fn release(_archetype: &Archetype, _state: Self::State) {}

    fn for_each_borrow(mut f: impl FnMut(StableTypeId, bool)) {
        f(StableTypeId::of::<T>(), false);
    }

    unsafe fn get(&self, n: usize) -> Self::Item {
        *self.0.as_ptr().add(n)
    }
}

/// Query that retrieves added state of type `T` component.
///
/// It is your responsibility to clear trackers with
/// [`World::clear_trackers()`](crate::World::clear_trackers()) at the start of
/// the frame (or any other suitable moment).
///
/// # Example
/// ```
/// # use gloss_hecs::*;
/// let mut world = World::new();
/// let e = world.spawn((123,));
/// for (_id, (value, value_add)) in world.query::<(&i32, Added<i32>)>().iter() {
///     assert_eq!(*value, 123);
///     assert_eq!(value_add, true);
/// }
/// world.clear_trackers();
/// for (_id, value_add) in world.query::<Added<i32>>().iter() {
///     assert_eq!(value_add, false);
/// }
/// ```
pub struct Added<T>(PhantomData<fn(T)>);

impl<T: Component> Query for Added<T> {
    type Fetch = FetchAdded<T>;
}

#[doc(hidden)]
pub struct FetchAdded<T>(NonNull<bool>, PhantomData<fn(T)>);

unsafe impl<'a, T: Component> Fetch<'a> for FetchAdded<T> {
    type Item = bool;

    type State = usize;

    fn dangling() -> Self {
        Self(NonNull::dangling(), PhantomData)
    }

    fn access(archetype: &Archetype) -> Option<Access> {
        if archetype.has::<T>() {
            Some(Access::Read)
        } else {
            None
        }
    }

    fn borrow(_archetype: &Archetype, _state: Self::State) {}
    fn prepare(archetype: &Archetype) -> Option<Self::State> {
        archetype.get_state::<T>()
    }
    fn execute(archetype: &'a Archetype, state: Self::State) -> Self {
        Self(archetype.get_added(state), PhantomData)
    }
    fn release(_archetype: &Archetype, _state: Self::State) {}

    fn for_each_borrow(mut f: impl FnMut(StableTypeId, bool)) {
        f(StableTypeId::of::<T>(), false);
    }

    unsafe fn get(&self, n: usize) -> Self::Item {
        *self.0.as_ptr().add(n)
    }
}

/// Query that retrieves changed state of type `T` component.
/// Changed component is one that have either been mutated or added.
///
/// It is your responsibility to clear trackers with
/// [`World::clear_trackers()`](crate::World::clear_trackers()) at the start of
/// the frame (or any other suitable moment).
///
/// # Example
/// ```
/// # use gloss_hecs::*;
/// let mut world = World::new();
/// let e = world.spawn((123,));
/// for (_id, (value, value_ch)) in world.query::<(&i32, Changed<i32>)>().iter() {
///     assert_eq!(*value, 123);
///     assert_eq!(value_ch, true);
/// }
/// world.clear_trackers();
/// for (_id, value_ch) in world.query::<Changed<i32>>().iter() {
///     assert_eq!(value_ch, false);
/// }
/// for (_id, mut value) in world.query::<&mut i32>().iter() {
///     *value = 42;
/// }
/// for (_id, (value, value_ch)) in world.query::<(&i32, Changed<i32>)>().iter() {
///     assert_eq!(*value, 42);
///     assert_eq!(value_ch, true);
/// }
/// world.clear_trackers();
/// for (_id, value_ch) in world.query::<Changed<i32>>().iter() {
///     assert_eq!(value_ch, false);
/// }
/// ```
pub struct Changed<T>(PhantomData<fn(T)>);

impl<T: Component> Query for Changed<T> {
    type Fetch = FetchChanged<T>;
}

#[doc(hidden)]
pub struct FetchChanged<T>(NonNull<bool>, NonNull<bool>, PhantomData<fn(T)>);

unsafe impl<'a, T: Component> Fetch<'a> for FetchChanged<T> {
    type Item = bool;

    type State = usize;

    fn dangling() -> Self {
        Self(NonNull::dangling(), NonNull::dangling(), PhantomData)
    }

    fn access(archetype: &Archetype) -> Option<Access> {
        if archetype.has::<T>() {
            Some(Access::Read)
        } else {
            None
        }
    }

    fn borrow(_archetype: &Archetype, _state: Self::State) {}
    fn prepare(archetype: &Archetype) -> Option<Self::State> {
        archetype.get_state::<T>()
    }
    fn execute(archetype: &'a Archetype, state: Self::State) -> Self {
        Self(archetype.get_mutated(state), archetype.get_added(state), PhantomData)
    }
    fn release(_archetype: &Archetype, _state: Self::State) {}

    fn for_each_borrow(mut f: impl FnMut(StableTypeId, bool)) {
        f(StableTypeId::of::<T>(), false);
    }

    unsafe fn get(&self, n: usize) -> Self::Item {
        *self.0.as_ptr().add(n) || *self.1.as_ptr().add(n)
    }
}
