use crate::scene::Scene;
use gloss_hecs::Entity;
use gloss_utils::abi_stable_aliases::std_types::RString;
#[cfg(not(target_arch = "wasm32"))]
use gloss_utils::abi_stable_aliases::StableAbi;

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct Button {
    pub name: RString,
    pub f_clicked: extern "C" fn(RString, Entity, &mut Scene),
}

impl Button {
    pub fn new(name: &str, f_clicked: extern "C" fn(RString, Entity, &mut Scene)) -> Self {
        Self {
            name: RString::from(name),
            f_clicked,
        }
    }
}
