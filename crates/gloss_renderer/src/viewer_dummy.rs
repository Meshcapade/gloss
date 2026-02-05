cfg_if::cfg_if! {
    if #[cfg(not(target_arch = "wasm32"))] {
        use crate::config::Config;
        use crate::logger::gloss_setup_logger_from_config;
        use crate::plugin_manager::plugins::{Plugin, Plugins};
        use crate::scene::Scene;
        use crate::set_panic_hook;
        use crate::{camera::Camera, scene::GLOSS_CAM_NAME};
    }
}

use core::time::Duration;
#[allow(unused_imports)]
use log::{error, info, Level};

#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use wasm_bindgen::prelude::*;
use wasm_timer::Instant;

#[derive(Debug)]
#[repr(C)]
pub struct RunnerDummy {
    pub is_running: bool,
    pub first_time: bool,
    time_init: Instant, //time when the init of the viewer has finished
}
impl Default for RunnerDummy {
    fn default() -> Self {
        let time_init = Instant::now();
        Self {
            is_running: false,
            first_time: true,
            time_init,
        }
    }
}
#[allow(unused)]
impl RunnerDummy {
    pub fn time_since_init(&self) -> Duration {
        if self.first_time {
            Duration::ZERO
        } else {
            self.time_init.elapsed()
        }
    }
}

/// `ViewerDummy` is just a container for the Scene but cannot perform any rendering
/// It's mainly used for some operations that require the existance of a scene, like exporting GLTFs
/// In an ideal world `ViewerDummy` shouldn't exist
#[cfg(not(target_arch = "wasm32"))] //wasm cannot compile the run_return() call so we just disable the whole
                                    // dummy viewer
pub struct ViewerDummy {
    pub camera: Camera,
    pub scene: Scene,
    pub plugins: Plugins,
    pub config: Config,
    pub runner: RunnerDummy,
}

#[cfg(not(target_arch = "wasm32"))]
impl ViewerDummy {
    pub fn new(config_path: Option<&str>) -> Self {
        let config = Config::new(config_path);
        Self::new_with_config(&config)
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::missing_panics_doc)]
    pub fn new_with_config(config: &Config) -> Self {
        set_panic_hook();
        if config.core.auto_create_logger {
            gloss_setup_logger_from_config(config);
        }

        //expensive but useful
        re_memory::accounting_allocator::set_tracking_callstacks(config.core.enable_memory_profiling_callstacks);

        let runner = RunnerDummy::default();

        let mut scene = Scene::new();
        let camera = Camera::new(GLOSS_CAM_NAME, &mut scene, false);

        Self {
            camera,
            scene,
            plugins: Plugins::new(),
            config: config.clone(),
            runner,
        }
    }

    pub fn insert_plugin<T: Plugin + 'static>(&mut self, plugin: &T) {
        // self.plugins.insert_plugin(plugin);
        self.plugins.insert_plugin(plugin, &mut self.scene);
    }
    #[allow(clippy::missing_panics_doc)]
    pub fn run_manual_plugins(&mut self) {
        self.plugins.run_logic_systems_dummy(&mut self.scene, &mut self.runner, false);
    }

    pub fn reset_for_first_time(&mut self) {
        self.runner.first_time = true;
    }

    pub fn start_batch_net_sending(&mut self) {
        if let Ok(mut sender) = self.scene.get_resource::<&mut crate::network::SceneSender>() {
            sender.start_frame();
        }
    }
    pub fn end_batch_net_sending(&mut self) {
        if let Ok(mut sender) = self.scene.get_resource::<&mut crate::network::SceneSender>() {
            sender.end_frame();
        }
    }
}
