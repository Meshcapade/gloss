use crate::scene::Scene;

use gloss_hecs::Entity;
use gloss_utils::abi_stable_aliases::std_types::{RString, RVec};
#[cfg(not(target_arch = "wasm32"))]
use gloss_utils::abi_stable_aliases::StableAbi;

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct Selectable {
    pub name: RString,
    pub is_selected: bool,
    pub f_clicked: extern "C" fn(RString, Entity, &mut Scene),
}
impl Selectable {
    pub fn new(name: &str, is_selected: bool, f_clicked: extern "C" fn(RString, Entity, &mut Scene)) -> Self {
        Self {
            name: RString::from(name),
            is_selected,
            f_clicked,
        }
    }
}

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct SelectableList {
    pub items: RVec<Selectable>,
    pub is_horizontal: bool,
}

impl SelectableList {
    pub fn new(items: RVec<Selectable>, is_horizontal: bool) -> Self {
        Self { items, is_horizontal }
    }
}
