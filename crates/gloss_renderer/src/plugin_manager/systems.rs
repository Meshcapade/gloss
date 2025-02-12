use crate::scene::Scene;

use gloss_hecs::Entity;
use utils_rs::abi_stable_aliases::std_types::{RDuration, RNone, ROption, ROption::RSome, RString};
#[cfg(not(target_arch = "wasm32"))]
use utils_rs::abi_stable_aliases::StableAbi;

use super::{gui::window::GuiWindow, plugins::Event, runner::RunnerState};

#[repr(C)]
// #[derive(StableAbi, Clone)]
#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct GuiSystem {
    pub f: extern "C" fn(selected_entity: ROption<Entity>, scene: &mut Scene) -> GuiWindow,
}
impl GuiSystem {
    pub fn new(f: extern "C" fn(selected_entity: ROption<Entity>, scene: &mut Scene) -> GuiWindow) -> Self {
        Self { f }
    }
}

#[repr(C)]
// #[derive(StableAbi, Clone)]
#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct SystemMetadata {
    pub autorun: bool,
    // pub was_run_last: bool,
    pub execution_time: RDuration,
}
impl Default for SystemMetadata {
    fn default() -> Self {
        Self {
            autorun: true,
            // was_run_last: false,
            execution_time: RDuration::from_secs(0),
        }
    }
}

#[repr(C)]
// #[derive(StableAbi, Clone)]
#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct LogicSystem {
    // f: LogicSystemFnType,
    pub f: extern "C" fn(scene: &mut Scene, runner: &mut RunnerState),
    // autorun: bool,
    pub name: ROption<RString>,
}
impl LogicSystem {
    pub fn new(f: extern "C" fn(scene: &mut Scene, runner: &mut RunnerState)) -> Self {
        Self { f, name: RNone }
    }
    #[must_use]
    pub fn with_name(self, name: &str) -> Self {
        Self {
            f: self.f,
            name: RSome(name.to_string().into()),
        }
    }
}

#[repr(C)]
// #[derive(StableAbi, Clone)]
#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct EventSystem {
    pub f: extern "C" fn(scene: &mut Scene, runner: &mut RunnerState, event: &Event) -> bool,
    pub name: ROption<RString>,
}
impl EventSystem {
    pub fn new(f: extern "C" fn(scene: &mut Scene, runner: &mut RunnerState, event: &Event) -> bool) -> Self {
        Self { f, name: RNone }
    }
    #[must_use]
    pub fn with_name(self, name: &str) -> Self {
        Self {
            f: self.f,
            name: RSome(name.to_string().into()),
        }
    }
}
