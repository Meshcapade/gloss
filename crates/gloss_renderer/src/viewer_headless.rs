cfg_if::cfg_if! {
    if #[cfg(not(target_arch = "wasm32"))] {
        use crate::components::{TargetResolution, TargetResolutionUpdate};
        use crate::config::Config;
        use crate::forward_renderer::renderer::Renderer;
        use crate::logger::gloss_setup_logger_from_config;
        use crate::plugin_manager::plugins::{Plugin, Plugins};
        use crate::scene::Scene;
        use crate::set_panic_hook;
        use crate::viewer::supported_backends;
        use crate::{camera::Camera, scene::GLOSS_CAM_NAME};

        use easy_wgpu::gpu::Gpu;
        use easy_wgpu::texture::Texture;

        use log::debug;
        use pollster::FutureExt;
    }
}
// #[cfg(target_arch = "wasm32")]
// use winit::platform::web::EventLoopExtWebSys;

use core::time::Duration;
#[allow(unused_imports)]
use log::{error, info, Level};

#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use wasm_bindgen::prelude::*;
use wasm_timer::Instant;

#[cfg(not(target_arch = "wasm32"))]
use crate::viewer::enumerate_adapters;
#[cfg(not(target_arch = "wasm32"))]
use crate::viewer::get_adapter;

#[derive(Debug)]
#[repr(C)]
pub struct RunnerHeadless {
    pub is_running: bool,
    pub do_render: bool,
    pub first_time: bool,
    pub did_warmup: bool,
    frame_is_started: bool, /* after calling viewer.start_frame() this gets set to true. any subsequent cals to start frame will be ignored if
                             * this frame has already been started. This can happen when using python and doing viewer.update() this will run some
                             * system that does a window.request_redraw which will cause another rerender but this frame has already started so we
                             * don't need to do anything since we are already in the middle of redrawing */
    time_init: Instant,       //time when the init of the viewer has finished
    time_last_frame: Instant, //time when we started the previous frame. So the instant when we called viewer.start_frame()
    dt: Duration,             /* delta time since we finished last frame, we store it directly here instead of constantly querying it because
                               * different systems might then get different dt depending on how long they take to run. This dt is set once when
                               * doing viewer.start_frame() */
}
impl Default for RunnerHeadless {
    fn default() -> Self {
        let time_init = Instant::now();
        let time_last_frame = Instant::now();
        Self {
            is_running: false,
            do_render: true,
            first_time: true,
            did_warmup: false,
            frame_is_started: false,
            time_init,
            time_last_frame,
            dt: Duration::ZERO,
        }
    }
}
#[allow(unused)]
impl RunnerHeadless {
    pub fn time_since_init(&self) -> Duration {
        if self.first_time {
            Duration::ZERO
        } else {
            self.time_init.elapsed()
        }
    }
    pub fn update_dt(&mut self) {
        if self.first_time {
            self.dt = Duration::ZERO;
        } else {
            self.dt = self.time_last_frame.elapsed();
        }
    }
    pub fn dt(&self) -> Duration {
        self.dt
    }
}

/// `ViewerHeadless` performs similarly as [Viewer] with the difference that it
/// can be used when rendering on headless machines. A typical usage is
/// rendering an image using [`ViewerHeadless::update`], recovering the texture
/// using [`ViewerHeadless::get_final_tex`], getting it to CPU using
/// [`Texture::download_to_cpu`] and finally write it to disk using
/// [`ImageBuffer::save`]
#[cfg(not(target_arch = "wasm32"))] //wasm cannot compile the run_return() call so we just disable the whole
                                    // headless viewer
pub struct ViewerHeadless {
    // have the renderer separate
    pub renderer: Renderer,
    pub camera: Camera,
    pub scene: Scene,
    pub plugins: Plugins,
    pub config: Config,
    pub runner: RunnerHeadless,
    // The order of properties in a struct is the order in which items are dropped.
    // wgpu seems to require that the device be dropped last, otherwise there is a resouce
    // leak.
    pub gpu: Gpu,
}

#[cfg(not(target_arch = "wasm32"))]
impl ViewerHeadless {
    pub fn new(width: u32, height: u32, config_path: Option<&str>) -> Self {
        let config = Config::new(config_path);
        Self::new_with_config(width, height, &config)
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::missing_panics_doc)]
    pub fn new_with_config(width: u32, height: u32, config: &Config) -> Self {
        set_panic_hook();
        if config.core.auto_create_logger {
            gloss_setup_logger_from_config(config);
        }

        //expensive but useful
        gloss_memory::accounting_allocator::set_tracking_callstacks(config.core.enable_memory_profiling_callstacks);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: supported_backends(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        //if we are on wasm we cannot enumerate adapter so we skip this at compile time
        cfg_if::cfg_if! {
                if #[cfg(not(target_arch = "wasm32"))]{
                let adapters = enumerate_adapters(&instance);
                info!("Number of possible adapters: {:?}", adapters.len());
                for (i, adapter) in adapters.iter().enumerate() {
                    info!("Adapter option {:?}: {:?}", i + 1, adapter.get_info());
                }
            }
        }
        let adapter = get_adapter(&instance, None);
        info!("Selected adapter: {:?}", adapter.get_info());

        // info!("{:?}", adapter.get_info());
        //TODO if the adapter is not a discrete Nvidia gpu, disable the wgpu to pytorch
        // interop

        //features
        let mut desired_features = wgpu::Features::empty();
        cfg_if::cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))]{
                // println!("compiling with time query");
                desired_features = desired_features.union(wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES));
                desired_features = desired_features.union(wgpu::Features::POLYGON_MODE_POINT);
                desired_features = desired_features.union(wgpu::Features::POLYGON_MODE_LINE);
            }
        }
        let required_features = adapter.features().intersection(desired_features); //only take the features that are actually supported
        info!("enabled features: {required_features:?}");

        //dealing with wasm putting 2048 as maximum texture size
        //https://github.com/gfx-rs/wgpu/discussions/2952
        let max_limits = adapter.limits();
        #[allow(unused_mut)]
        let mut limits_to_request = wgpu::Limits::default();
        if cfg!(target_arch = "wasm32") {
            limits_to_request = wgpu::Limits::downlevel_webgl2_defaults();
        }
        limits_to_request.max_texture_dimension_1d = max_limits.max_texture_dimension_1d;
        limits_to_request.max_texture_dimension_2d = max_limits.max_texture_dimension_2d;
        limits_to_request.max_buffer_size = max_limits.max_buffer_size;

        let mut memory_hints = wgpu::MemoryHints::Performance;
        if cfg!(target_arch = "wasm32") {
            //we usually have issue with running out of memory on wasm, so I would rather
            // optimize for low memory usage
            memory_hints = wgpu::MemoryHints::MemoryUsage;
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features,
                    required_limits: limits_to_request,
                    memory_hints,
                },
                None, // Trace path
            )
            .block_on()
            .expect("A device and queue could not be created. Maybe there's a driver issue on your machine?");

        let runner = RunnerHeadless::default();

        let gpu = Gpu::new(adapter, instance, device, queue);
        let mut scene = Scene::new();
        let camera = Camera::new(GLOSS_CAM_NAME, &mut scene, false);
        let _ = scene.world.insert_one(
            camera.entity,
            TargetResolution {
                width,
                height,
                update_mode: TargetResolutionUpdate::Fixed,
            },
        );
        let renderer = Renderer::new(&gpu, &config.render, None); //moves the device and queue into it and takes ownership of them

        Self {
            gpu,
            renderer,
            camera,
            scene,
            plugins: Plugins::new(),
            config: config.clone(),
            runner,
        }
    }

    pub fn insert_plugin<T: Plugin + 'static>(&mut self, plugin: &T) {
        self.plugins.insert_plugin(plugin);
    }
    #[allow(clippy::missing_panics_doc)]
    pub fn run_manual_plugins(&mut self) {
        self.plugins.run_logic_systems_headless(&mut self.scene, &mut self.runner, false);
    }
    //wasm cannot compile the run_return() call so we just disable this whole
    // function
    pub fn update(&mut self) {
        self.render(None);
    }

    pub fn render_from_cam(&mut self, cam: &mut Camera) {
        //we replace our defaul
        self.render(Some(cam));
    }

    //called at the beggining of the render and sets the time that all systems will
    // use
    pub fn start_frame(&mut self) -> Duration {
        //first time we call this we do a warmup render to initialize everything
        if !self.runner.did_warmup {
            self.runner.did_warmup = true; //has to be put here because warmup actually calls start_frame and we don't
                                           // want an infinite recurrsion
            self.warmup();
            self.warmup();
        }

        self.runner.update_dt();
        debug!("after update dt it is {:?}", self.runner.dt());
        self.runner.time_last_frame = Instant::now();

        self.runner.frame_is_started = true;

        self.runner.dt
    }

    fn render(&mut self, other_cam: Option<&mut Camera>) {
        if self.runner.first_time {
            self.runner.time_init = Instant::now();
        }

        self.plugins.run_logic_systems_headless(&mut self.scene, &mut self.runner, true);

        let dt = self.runner.dt();

        //we render to an internal texture since we have no surface
        // let out_view = self.renderer.rendered_tex().view;

        self.renderer
            .render_to_texture(&self.gpu, other_cam.unwrap_or(&mut self.camera), &mut self.scene, &mut self.config, dt);

        self.runner.first_time = false;
        self.runner.frame_is_started = false;
    }

    pub fn warmup(&mut self) {
        debug!("Starting warmup");
        self.start_frame();
        self.run_manual_plugins(); //auto plugins will run when we do self.render(), but here we also need to run
                                   // the manual ones
        #[cfg(not(target_arch = "wasm32"))] //wasm cannot do update because it needs run_return
        self.update();
        #[cfg(target_arch = "wasm32")] //
        self.render();
        self.reset_for_first_time();
        debug!("finished warmup");
    }

    pub fn reset_for_first_time(&mut self) {
        self.runner.first_time = true;
    }

    pub fn get_final_tex(&self) -> &Texture {
        let tex = self.renderer.rendered_tex();
        tex
    }

    pub fn get_final_depth(&self) -> &Texture {
        let depth = self.renderer.depth_buffer();
        depth
    }

    pub fn set_size(&mut self, width: u32, height: u32) {
        self.camera.set_target_res(width, height, &mut self.scene);
    }
}
