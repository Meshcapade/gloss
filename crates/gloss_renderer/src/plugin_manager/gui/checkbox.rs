use crate::scene::Scene;
use gloss_hecs::Entity;

use utils_rs::abi_stable_aliases::std_types::RString;
#[cfg(not(target_arch = "wasm32"))]
use utils_rs::abi_stable_aliases::StableAbi;

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct Checkbox {
    pub name: RString,
    pub init_val: bool,
    pub f_clicked: extern "C" fn(bool, RString, Entity, &mut Scene),
}

impl Checkbox {
    pub fn new(name: &str, init_val: bool, f_clicked: extern "C" fn(bool, RString, Entity, &mut Scene)) -> Self {
        Self {
            name: RString::from(name),
            init_val,
            f_clicked,
        }
    }
}
