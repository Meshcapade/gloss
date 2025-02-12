use crate::scene::Scene;

use gloss_hecs::Entity;
use utils_rs::abi_stable_aliases::std_types::{ROption, RString};
#[cfg(not(target_arch = "wasm32"))]
use utils_rs::abi_stable_aliases::StableAbi;

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct Slider {
    pub name: RString,
    pub init_val: f32,
    pub min: f32,
    pub max: f32,
    pub width: ROption<f32>,
    //TODO change to f_dragged f_released
    pub f_change: extern "C" fn(f32, RString, Entity, &mut Scene),
    pub f_no_change: ROption<extern "C" fn(RString, Entity, &mut Scene)>,
}

impl Slider {
    pub fn new(
        name: &str,
        init_val: f32,
        min: f32,
        max: f32,
        width: ROption<f32>,
        f_change: extern "C" fn(f32, RString, Entity, &mut Scene),
        f_no_change: ROption<extern "C" fn(RString, Entity, &mut Scene)>,
    ) -> Self {
        Self {
            name: RString::from(name),
            init_val,
            min,
            max,
            width,
            f_change,
            f_no_change,
        }
    }
}
