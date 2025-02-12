use std::time::Duration;

use crate::{viewer::Runner, viewer_headless::RunnerHeadless};

use utils_rs::abi_stable_aliases::std_types::RDuration;
#[cfg(not(target_arch = "wasm32"))]
use utils_rs::abi_stable_aliases::StableAbi;

//The runner in the viewer contains event loop and other things that cannot
// cross the ffi barrier. So we make a new one that is slimmer
#[repr(C)]
#[cfg_attr(not(target_arch = "wasm32"), derive(StableAbi))]
pub struct RunnerState {
    pub is_running: bool,
    pub do_render: bool,
    pub request_redraw: bool,
    first_time: bool,
    dt: RDuration, /* delta time since we finished last frame, we store it directly here instead of constantly querying it because different
                    * systems might then get different dt depending on how long they take to run. This dt is set once when doing
                    * viewer.start_frame() */
}
//TODO use a tryinto and tryfrom functions
impl RunnerState {
    pub fn from(runner: &Runner) -> Self {
        Self {
            is_running: runner.is_running,
            do_render: runner.do_render,
            request_redraw: false,
            first_time: runner.first_time,
            dt: RDuration::from(runner.dt()),
        }
    }
    pub fn from_headless(runner: &RunnerHeadless) -> Self {
        Self {
            is_running: runner.is_running,
            do_render: runner.do_render,
            request_redraw: false,
            first_time: runner.first_time,
            dt: RDuration::from(runner.dt()),
        }
    }
    //onyl some of the fields are actually going to be modified and are worth
    // storing back into the runner, at least the pub ones shound be set back into
    // runner
    pub fn to(&self, runner: &mut Runner) {
        runner.is_running = self.is_running;
        runner.do_render = self.do_render;
    }
    pub fn to_headless(&self, runner: &mut RunnerHeadless) {
        runner.is_running = self.is_running;
        runner.do_render = self.do_render;
    }
    pub fn request_redraw(&mut self) {
        self.request_redraw = true;
    }
    pub fn dt(&self) -> Duration {
        Duration::from(self.dt)
    }
    pub fn is_first_time(&self) -> bool {
        self.first_time
    }
}
