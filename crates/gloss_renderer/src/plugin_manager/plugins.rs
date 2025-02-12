/// TODO Technically we also need to add [no-mangle] as in here:
/// <https://docs.rust-embedded.org/book/interoperability/rust-with-c.html>
/// However since we only create the mangles by the same rust compiler it should
/// work fine. This no-mangle is mostly when you try to export the function to C
/// which does not mangle the names in the same way as Rust. Currently since we
/// don't target C++, we can mangle the functions as normal and therefore avoid
/// collision in names of function.
use crate::scene::Scene;
use crate::{
    viewer::{GpuResources, Runner},
    viewer_headless::RunnerHeadless,
};

use utils_rs::abi_stable_aliases::std_types::{RDuration, RString, RVec, Tuple2};
#[cfg(not(target_arch = "wasm32"))]
use utils_rs::abi_stable_aliases::StableAbi;

use super::{
    runner::RunnerState,
    systems::{EventSystem, GuiSystem, LogicSystem, SystemMetadata},
};

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub enum Event {
    DroppedFile(RString),
}

pub trait Plugin {
    fn event_systems(&self) -> Vec<EventSystem>;
    fn logic_systems(&self) -> Vec<LogicSystem>;
    fn gui_systems(&self) -> Vec<GuiSystem>;
    fn autorun(&self) -> bool;
}

#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct Plugins {
    pub event_systems: RVec<Tuple2<EventSystem, SystemMetadata>>,
    pub logic_systems: RVec<Tuple2<LogicSystem, SystemMetadata>>,
    pub gui_systems: RVec<Tuple2<GuiSystem, SystemMetadata>>,
}
impl Default for Plugins {
    fn default() -> Self {
        Self {
            event_systems: RVec::new(),
            logic_systems: RVec::new(),
            gui_systems: RVec::new(),
        }
    }
}
impl Plugins {
    pub fn new() -> Self {
        Self {
            event_systems: RVec::new(),
            logic_systems: RVec::new(),
            gui_systems: RVec::new(),
        }
    }
    #[allow(clippy::needless_update)]
    pub fn insert_plugin<T: Plugin + 'static>(&mut self, plugin: &T) {
        for sys in plugin.event_systems().iter() {
            let metadata = SystemMetadata {
                autorun: plugin.autorun(),
                ..Default::default()
            };
            self.event_systems.push(Tuple2(sys.clone(), metadata));
        }
        for sys in plugin.logic_systems().iter() {
            let metadata = SystemMetadata {
                autorun: plugin.autorun(),
                ..Default::default()
            };
            self.logic_systems.push(Tuple2(sys.clone(), metadata));
        }
        for sys in plugin.gui_systems().iter() {
            let metadata = SystemMetadata {
                autorun: plugin.autorun(),
                ..Default::default()
            };
            self.gui_systems.push(Tuple2(sys.clone(), metadata));
        }
    }
    pub fn run_logic_systems(&mut self, gpu_res: &mut GpuResources, scene: &mut Scene, runner: &mut Runner, autorun_flag: bool) {
        let mut runner_state = RunnerState::from(runner);

        // gpu_res.gpu.device().poll(wgpu::Maintain::Wait);

        for system_and_metadata in self.logic_systems.iter_mut() {
            let metadata = &mut system_and_metadata.1;
            let sys = &system_and_metadata.0;
            if metadata.autorun == autorun_flag {
                let func = sys.f;

                //run and time the function
                let now = wasm_timer::Instant::now();
                func(scene, &mut runner_state);
                // gpu_res.gpu.device().poll(wgpu::Maintain::Wait); // Sync
                metadata.execution_time = RDuration::from(now.elapsed());
            }
        }
        runner_state.to(runner);

        if runner_state.request_redraw {
            gpu_res.request_redraw();
        }
    }

    pub fn run_logic_systems_headless(&mut self, scene: &mut Scene, runner: &mut RunnerHeadless, autorun_flag: bool) {
        let mut runner_state = RunnerState::from_headless(runner);
        for system_and_metadata in self.logic_systems.iter_mut() {
            let metadata = &mut system_and_metadata.1;
            let sys = &system_and_metadata.0;
            if metadata.autorun == autorun_flag {
                let func = sys.f;

                //run and time the function
                let now = wasm_timer::Instant::now();
                func(scene, &mut runner_state);
                metadata.execution_time = RDuration::from(now.elapsed());
            }
        }
        runner_state.to_headless(runner);
    }

    pub fn try_handle_event(&self, scene: &mut Scene, runner: &mut Runner, event: &Event) -> bool {
        let mut runner_state = RunnerState::from(runner);
        let mut handled = false;
        for system_and_metadata in self.event_systems.iter() {
            let sys = &system_and_metadata.0;
            let func = sys.f;
            handled |= func(scene, &mut runner_state, event);
        }
        runner_state.to(runner);
        handled
    }
}
