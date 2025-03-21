// use core::any::StableTypeId;
use crate::stabletypeid::StableTypeId;
use core::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{archetype::Archetype, ArchetypeColumn, ArchetypeColumnMut, Component, Entity, Fetch, MissingComponent, Query, QueryOne, TypeInfo};

/// Handle to an entity with any component types
#[derive(Copy, Clone)]
pub struct EntityRef<'a> {
    archetype: &'a Archetype,
    entity: Entity,
    index: u32,
}

impl<'a> EntityRef<'a> {
    pub(crate) unsafe fn new(archetype: &'a Archetype, entity: Entity, index: u32) -> Self {
        Self { archetype, entity, index }
    }

    /// Get the [`Entity`] handle associated with this entity
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Determine whether this entity would satisfy the query `Q`
    pub fn satisfies<Q: Query>(&self) -> bool {
        Q::Fetch::access(self.archetype).is_some()
    }

    /// Determine whether this entity has a `T` component without borrowing it
    ///
    /// Equivalent to [`satisfies::<&T>`](Self::satisfies)
    pub fn has<T: Component>(&self) -> bool {
        self.archetype.has::<T>()
    }

    /// Borrow a single component, if it exists
    ///
    /// `T` must be a shared or unique reference to a component type.
    ///
    /// # Example
    /// ```
    /// # use gloss_hecs::*;
    /// let mut world = World::new();
    /// let a = world.spawn((42, "abc"));
    /// let e = world.entity(a).unwrap();
    /// *e.get::<&mut i32>().unwrap() = 17;
    /// assert_eq!(*e.get::<&i32>().unwrap(), 17);
    /// ```
    ///
    /// Panics if `T` is a unique reference and the component is already
    /// borrowed, or if the component is already uniquely borrowed.
    pub fn get<T: ComponentRef<'a>>(&self) -> Option<T::Ref> {
        T::get_component(*self)
    }

    /// Run a query against this entity
    ///
    /// Equivalent to invoking [`World::query_one`](crate::World::query_one) on
    /// the entity. May outlive `self`.
    ///
    /// # Example
    /// ```
    /// # use gloss_hecs::*;
    /// let mut world = World::new();
    /// let a = world.spawn((123, true, "abc"));
    /// // The returned query must outlive the borrow made by `get`
    /// let mut query = world.entity(a).unwrap().query::<(&mut i32, &bool)>();
    /// let (mut number, flag) = query.get().unwrap();
    /// if *flag {
    ///     *number *= 2;
    /// }
    /// assert_eq!(*number, 246);
    /// ```
    pub fn query<Q: Query>(&self) -> QueryOne<'a, Q> {
        unsafe { QueryOne::new(self.archetype, self.index) }
    }

    /// Enumerate the types of the entity's components
    ///
    /// Convenient for dispatching component-specific logic for a single entity.
    /// For example, this can be combined with a `HashMap<StableTypeId,
    /// Box<dyn Handler>>` where `Handler` is some user-defined trait with
    /// methods for serialization, or to be called after spawning or before
    /// despawning to maintain secondary indices.
    pub fn component_types(&self) -> impl Iterator<Item = StableTypeId> + 'a {
        self.archetype.types().iter().map(TypeInfo::id)
    }

    /// Enumate the type info of the entity components
    /// Useful for inspecting the name of all the components
    pub fn component_infos(&self) -> impl Iterator<Item = TypeInfo> + 'a {
        self.archetype.types().iter().copied()
    }

    /// Number of components in this entity
    pub fn len(&self) -> usize {
        self.archetype.types().len()
    }

    /// Shorthand for `self.len() == 0`
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

unsafe impl<'a> Send for EntityRef<'a> {}
unsafe impl<'a> Sync for EntityRef<'a> {}

/// Shared borrow of an entity's component
#[derive(Clone)]
pub struct Ref<'a, T: Component> {
    archetype: &'a Archetype,
    /// State index for `T` in `archetype`
    state: usize,
    target: NonNull<T>,
    index: u32,
}

#[allow(missing_docs)]
impl<'a, T: Component> Ref<'a, T> {
    pub(crate) unsafe fn new(archetype: &'a Archetype, index: u32) -> Result<Self, MissingComponent> {
        let state = archetype.get_state::<T>().ok_or_else(MissingComponent::new::<T>)?;
        let target = NonNull::new_unchecked(archetype.get_base::<T>(state).as_ptr().add(index as usize));
        archetype.borrow::<T>(state);
        Ok(Self {
            archetype,
            state,
            target,
            index,
        })
    }
    pub fn is_added(&self) -> bool {
        unsafe { *self.archetype.get_added(self.state).as_ptr().add(self.index as usize) }
    }
    //keep this one private to the crate because it makes things confusing. I only
    // want added and changed. Changed will be true if it's either added or mutated
    pub(crate) fn is_mutated(&self) -> bool {
        unsafe { *self.archetype.get_mutated(self.state).as_ptr().add(self.index as usize) }
    }
    pub fn is_changed(&self) -> bool {
        self.is_added() || self.is_mutated()
    }
}

unsafe impl<T: Component> Send for Ref<'_, T> {}
unsafe impl<T: Component> Sync for Ref<'_, T> {}

impl<'a, T: Component> Drop for Ref<'a, T> {
    fn drop(&mut self) {
        self.archetype.release::<T>(self.state);
    }
}

impl<'a, T: Component> Deref for Ref<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { self.target.as_ref() }
    }
}

/// Unique borrow of an entity's component
pub struct RefMut<'a, T: Component> {
    archetype: &'a Archetype,
    /// State index for `T` in `archetype`
    state: usize,
    target: NonNull<T>,
    index: u32,
    mutated: &'a mut bool,
}

#[allow(missing_docs)]
impl<'a, T: Component> RefMut<'a, T> {
    pub(crate) unsafe fn new(archetype: &'a Archetype, index: u32) -> Result<Self, MissingComponent> {
        let state = archetype.get_state::<T>().ok_or_else(MissingComponent::new::<T>)?;
        let target = NonNull::new_unchecked(archetype.get_base::<T>(state).as_ptr().add(index as usize));
        archetype.borrow_mut::<T>(state);
        let mutated = &mut *archetype.get_mutated(state).as_ptr().add(index as usize);
        Ok(Self {
            archetype,
            state,
            target,
            index,
            mutated,
        })
    }
    pub fn is_added(&self) -> bool {
        unsafe { *self.archetype.get_added(self.state).as_ptr().add(self.index as usize) }
    }
    //keep this one private to the crate because it makes things confusing. I only
    // want added and changed. Changed will be true if it's either added or mutated
    pub(crate) fn is_mutated(&self) -> bool {
        unsafe { *self.archetype.get_mutated(self.state).as_ptr().add(self.index as usize) }
    }
    pub fn is_changed(&self) -> bool {
        self.is_added() || self.is_mutated()
    }
    pub fn remove_added(&mut self) {
        unsafe { *self.archetype.get_added(self.state).as_ptr().add(self.index as usize) = false }
    }
    //keep this one private to the crate because it makes things confusing. I only
    // want added and changed. Changed will be true if it's either added or mutated
    pub(crate) fn remove_mutated(&mut self) {
        unsafe { *self.archetype.get_mutated(self.state).as_ptr().add(self.index as usize) = false }
    }
    pub fn remove_changed(&mut self) {
        self.remove_added();
        self.remove_mutated();
    }
    // //keep this one private to the crate because it makes things confusing. I
    // only want added and changed. Changed will be true if it's either added or
    // mutated pub(crate) fn is_mutated(&self) -> bool {
    //     unsafe {
    //         *self
    //             .archetype
    //             .get_mutated(self.state)
    //             .as_ptr()
    //             .add(self.index as usize)
    //     }
    // }
}

unsafe impl<T: Component> Send for RefMut<'_, T> {}
unsafe impl<T: Component> Sync for RefMut<'_, T> {}

impl<'a, T: Component> Drop for RefMut<'a, T> {
    fn drop(&mut self) {
        self.archetype.release_mut::<T>(self.state);
    }
}

impl<'a, T: Component> Deref for RefMut<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { self.target.as_ref() }
    }
}

impl<'a, T: Component> DerefMut for RefMut<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        *self.mutated = true;
        unsafe { self.target.as_mut() }
    }
}

/// `&T` or `&mut T` where `T` is some component type
///
/// The interface of this trait is a private implementation detail.
pub trait ComponentRef<'a> {
    /// Smart pointer to a component of the referenced type
    #[doc(hidden)]
    type Ref;

    /// Smart pointer to a column of the referenced type in an
    /// [`Archetype`](crate::Archetype)
    #[doc(hidden)]
    type Column;

    /// Component type referenced by `Ref`
    #[doc(hidden)]
    type Component: Component;

    /// Fetch the component from `entity`
    #[doc(hidden)]
    fn get_component(entity: EntityRef<'a>) -> Option<Self::Ref>;

    /// Construct from a raw pointer
    ///
    /// # Safety
    ///
    /// Dereferencing `raw` for lifetime `'a` must be sound
    #[doc(hidden)]
    unsafe fn from_raw(raw: *mut Self::Component) -> Self;

    /// Borrow a column from an archetype
    #[doc(hidden)]
    fn get_column(archetype: &'a Archetype) -> Option<Self::Column>;
}

impl<'a, T: Component> ComponentRef<'a> for &'a T {
    type Ref = Ref<'a, T>;

    type Column = ArchetypeColumn<'a, T>;

    type Component = T;

    fn get_component(entity: EntityRef<'a>) -> Option<Self::Ref> {
        Some(unsafe { Ref::new(entity.archetype, entity.index).ok()? })
    }

    unsafe fn from_raw(raw: *mut Self::Component) -> Self {
        &*raw
    }

    fn get_column(archetype: &'a Archetype) -> Option<Self::Column> {
        ArchetypeColumn::new(archetype)
    }
}

impl<'a, T: Component> ComponentRef<'a> for &'a mut T {
    type Ref = RefMut<'a, T>;

    type Column = ArchetypeColumnMut<'a, T>;

    type Component = T;

    fn get_component(entity: EntityRef<'a>) -> Option<Self::Ref> {
        Some(unsafe { RefMut::new(entity.archetype, entity.index).ok()? })
    }

    unsafe fn from_raw(raw: *mut Self::Component) -> Self {
        &mut *raw
    }

    fn get_column(archetype: &'a Archetype) -> Option<Self::Column> {
        ArchetypeColumnMut::new(archetype)
    }
}

/// `&T` where `T` is some component type
///
/// Used when consistency demands that references to component types, rather
/// than component types themselves, be supplied as a type parameter to a
/// function that cannot operate on unique references.
pub trait ComponentRefShared<'a>: ComponentRef<'a> {}

impl<'a, T: Component> ComponentRefShared<'a> for &'a T {}
