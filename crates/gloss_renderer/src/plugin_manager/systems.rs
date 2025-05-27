use crate::{scene::Scene, viewer::GpuResources};

use gloss_hecs::Entity;
use gloss_utils::abi_stable_aliases::std_types::{RDuration, RNone, ROption, ROption::RSome, RString};
#[cfg(not(target_arch = "wasm32"))]
use gloss_utils::abi_stable_aliases::StableAbi;

use super::{gui::window::GuiWindow, plugins::Event, runner::RunnerState};

#[repr(C)]
#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct GuiSystem {
    pub f: extern "C" fn(selected_entity: &ROption<Entity>, scene: &mut Scene) -> GuiWindow,
}
impl GuiSystem {
    pub fn new(f: extern "C" fn(selected_entity: &ROption<Entity>, scene: &mut Scene) -> GuiWindow) -> Self {
        Self { f }
    }
}

#[repr(C)]
#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct SystemMetadata {
    pub autorun: bool,
    pub execution_time: RDuration,
}
impl Default for SystemMetadata {
    fn default() -> Self {
        Self {
            autorun: true,
            execution_time: RDuration::from_secs(0),
        }
    }
}

#[repr(C)]
#[derive(Clone)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct LogicSystem {
    pub f: extern "C" fn(scene: &mut Scene, runner: &mut RunnerState),
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

// We do not need to derive StableAbi and do #[repr(C)] because we do not need this to be FFI safe
// These systems can only be added from within gloss
#[derive(Clone)]
pub struct GpuSystem {
    pub f: fn(scene: &mut Scene, runner: &mut RunnerState, gpu_res: &GpuResources),
    pub name: ROption<RString>,
}

impl GpuSystem {
    pub fn new(f: fn(scene: &mut Scene, runner: &mut RunnerState, gpu_res: &GpuResources)) -> Self {
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
