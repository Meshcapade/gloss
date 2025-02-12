// TODO: Investigate canvas creation for WASM; Its not created properly

#[cfg(feature = "with-gui")]
use crate::gui::Gui;
#[cfg(feature = "with-gui")]
use crate::plugin_manager::GuiSystem;
use crate::{
    camera::Camera,
    components::Projection,
    config::Config,
    forward_renderer::{render_passes::blit_pass::BlitPass, renderer::Renderer},
    geom::Geom,
    logger::gloss_setup_logger_from_config,
    plugin_manager::{
        plugins::{Plugin, Plugins},
        systems::{LogicSystem, SystemMetadata},
    },
    scene::{Scene, GLOSS_CAM_NAME},
    set_panic_hook,
};

use easy_wgpu::gpu::Gpu;
#[cfg(feature = "with-gui")]
use egui_winit::EventResponse;
use log::{debug, warn};
use utils_rs::abi_stable_aliases::std_types::{RString, Tuple2};
use winit::{
    dpi::PhysicalSize,
    event::TouchPhase,
    event_loop::{ActiveEventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

use core::time::Duration;
use log::{error, info};
use pollster::FutureExt;
use std::{error::Error, sync::Arc};
use utils_rs::io::FileType;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wasm_timer::Instant;
use winit::application::ApplicationHandler;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

use winit::{
    event::{ElementState, Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

/// All the ``GpuResources`` are kept together to be able to easily recreate
/// them
pub struct GpuResources {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pub renderer: Renderer,

    #[cfg(feature = "with-gui")]
    pub gui: Option<Gui>,
    _blit_pass: BlitPass, // Copies the final rendered texture towards screen for visualization
    pub redraw_requested: bool, /* When we request a redraw we set this to true. If we are to request another redraw, if this is set to true we
                           * ignore the call. We need this because the call to window.redraw_requested() is quite expensive and it can be
                           * called multiple times a frame if there are multiple inputs */

    // The order of properties in a struct is the order in which items are dropped.
    // wgpu seems to require that the device be dropped last, otherwise there is a resource leak
    pub gpu: Gpu,
}
#[allow(clippy::missing_panics_doc)]
#[allow(clippy::too_many_lines)]
#[allow(unused)]
impl GpuResources {
    pub fn new(
        event_loop: &ActiveEventLoop,
        event_loop_proxy: &EventLoopProxy<CustomEvent>,
        canvas_id_parsed: &Option<String>,
        config: &Config,
    ) -> Self {
        // The instance is a handle to our GPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: supported_backends(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let window = Viewer::create_window(event_loop, event_loop_proxy, &canvas_id_parsed.clone()).expect("failed to create initial window");

        let window = Arc::new(window);

        let surface = unsafe { instance.create_surface(window.clone()) }.unwrap();

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
        let adapter = get_adapter(&instance, Some(&surface));
        info!("Selected adapter: {:?}", adapter.get_info());

        //TODO if the adapter is not a discrete Nvidia gpu, disable the wgpu to pytorch
        // interop
        let mut desired_features = wgpu::Features::empty();
        cfg_if::cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))]{
                desired_features = desired_features.union(wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES));
                desired_features = desired_features.union(wgpu::Features::POLYGON_MODE_POINT);
                desired_features = desired_features.union(wgpu::Features::POLYGON_MODE_LINE);
            }
        }
        let required_features = adapter.features().intersection(desired_features); //only take the features that are actually supported
                                                                                   // info!("enabled features: {required_features:?}");

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

        //device and queue
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
        let gpu = Gpu::new(adapter, instance, device, queue);

        //configure surface
        let surface_caps = surface.get_capabilities(gpu.adapter());
        // Shader code assumes we are writing to non-Srgb surface texture and we do
        // tonemapping and gamma-correction manually
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        //get size of canvas or just some dummy value at the beggining until we get a
        // resize event
        let mut size = PhysicalSize::new(1200, 1200);
        #[cfg(target_arch = "wasm32")]
        if let Some(canvas_size) = Viewer::get_html_elem_size(&canvas_id_parsed.as_ref().unwrap()) {
            size = canvas_size.to_physical(window.scale_factor());
        }
        // info!("init: size is {:?}", size);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync, // double-buffered: low-latency, may have tearing
            // present_mode: wgpu::PresentMode::AutoVsync, // triple-buffered: high-latency, no tearing
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(gpu.device(), &surface_config);

        // info!("enable gui is {}", config.core.enable_gui);
        #[cfg(feature = "with-gui")]
        let gui = if config.core.enable_gui {
            let mut gui = Gui::new(&window, &gpu, surface_format);
            gui.hidden = config.core.gui_start_hidden;
            Some(gui)
        } else {
            None
        };

        let renderer = Renderer::new(&gpu, &config.render, Some(surface_format));
        let blit_pass = BlitPass::new(&gpu, &surface_format);

        Self {
            window,
            surface,
            surface_config,
            gpu,
            renderer,
            #[cfg(feature = "with-gui")]
            gui,
            _blit_pass: blit_pass,
            redraw_requested: false,
        }
    }
    pub fn request_redraw(&mut self) {
        if self.redraw_requested {
            debug!("Redraw was already requested, ignoring.");
        } else {
            self.window.request_redraw();
            self.redraw_requested = true;
        }
    }
}

/// The ``Runner`` for managing the event loop and time based scene updates
#[derive(Debug)]
#[repr(C)]
pub struct Runner {
    event_loop: Option<EventLoop<CustomEvent>>, // Keep it as an option so as to remove it from this object and avoid lifetime issues
    event_loop_proxy: EventLoopProxy<CustomEvent>,
    pub autostart: bool, // if this is true, when doing viewer.run we also start actually running the event loop and set the is_running to true
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
#[allow(unused)]
impl Runner {
    #[allow(clippy::missing_panics_doc)]
    #[allow(clippy::new_without_default)]
    pub fn new(canvas_id: &Option<String>) -> Self {
        let event_loop = EventLoop::<CustomEvent>::with_user_event().build().unwrap();
        let event_loop_proxy: EventLoopProxy<CustomEvent> = event_loop.create_proxy();

        #[cfg(target_arch = "wasm32")]
        if let Some(ref canvas_id) = canvas_id {
            Viewer::add_listener_to_canvas_resize(&event_loop, &canvas_id);
            Viewer::add_listener_to_context(&event_loop, &canvas_id);
        }

        let time_init = Instant::now();
        let time_last_frame = Instant::now();
        Self {
            event_loop: Some(event_loop),
            event_loop_proxy,
            autostart: true,
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
    pub fn override_dt(&mut self, new_dt: f32) {
        self.dt = Duration::from_secs_f32(new_dt);
    }
    pub fn dt(&self) -> Duration {
        self.dt
    }
}

/// Viewer encapsulates the window and event managing of the app.
/// The viewer contains a [Window], a [Scene] and a [Renderer].
/// Typically the viewer is used by calling [`Viewer::run`] which automatically
/// spins a rendering loop and processes keyboard and mouse events.
/// The rendering loop can also be created manually by running
/// [`Viewer::update`] in a loop {}
#[repr(C)]
pub struct Viewer {
    // Gpu resources that need to be rebuild every time we lose context or suspend the application
    pub gpu_res: Option<GpuResources>,

    // Cpu resources
    pub camera: Camera, //TODO pull the camera into the scene as just another entity
    pub scene: Scene,

    window_size: winit::dpi::PhysicalSize<u32>, /* We need this because sometimes we need to recreate the window and we need to recreate it with
                                                 * this size */

    canvas_id_parsed: Option<String>,

    pub config: Config,
    pub runner: Runner,
    pub plugins: Plugins,
}

impl Viewer {
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

        // Expensive but useful
        gloss_memory::accounting_allocator::set_tracking_callstacks(config.core.enable_memory_profiling_callstacks);

        let canvas_id_parsed = config.core.canvas_id.as_ref().map(|canvas_id| String::from("#") + canvas_id);

        // Runner stuff
        let runner = Runner::new(&canvas_id_parsed);

        let window_size = winit::dpi::PhysicalSize::new(100, 100);

        let mut scene = Scene::new();
        let camera = Camera::new(GLOSS_CAM_NAME, &mut scene, false); //TODO make it another entity inside the Scene

        Self {
            gpu_res: None,
            runner,
            scene,
            camera,
            plugins: Plugins::new(),
            canvas_id_parsed: canvas_id_parsed.clone(),
            config: config.clone(),
            window_size,
        }
    }

    /// Makes the loop emit resize events when the Web canvas changes size. This
    /// allows the renderer to adjust to the size of the dynamic HTML elements
    #[cfg(target_arch = "wasm32")]
    fn add_listener_to_canvas_resize(event_loop: &EventLoop<CustomEvent>, canvas_id: &str) {
        //https://github.com/rust-windowing/winit/blob/master/examples/custom_events.rs
        // `EventLoopProxy` allows you to dispatch custom events to the main Winit event
        // loop from any thread.
        //make it listen to resizes of the canvas
        // https://github.com/bevyengine/bevy/blob/main/crates/bevy_winit/src/web_resize.rs#L61
        let event_loop_proxy = event_loop.create_proxy();
        let canvas_id = String::from(canvas_id); //copy it internally

        // Function that triggers a custom event for resizing
        let resize = move || {
            if let Some(size) = Viewer::get_html_elem_size(&canvas_id) {
                let event_resize = CustomEvent::Resize(size.width, size.height);
                event_loop_proxy.send_event(event_resize).ok();
            }
        };

        // ensure resize happens on startup
        // resize();

        let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move |_: web_sys::Event| {
            resize();
        }) as Box<dyn FnMut(_)>);
        let window = web_sys::window().unwrap();

        window
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    #[cfg(target_arch = "wasm32")]
    fn add_listener_to_context(event_loop: &EventLoop<CustomEvent>, canvas_id: &str) {
        let canvas_id = String::from(canvas_id); //copy it internally

        let win = web_sys::window().unwrap();
        let doc = win.document().unwrap();
        let element = doc.query_selector(&canvas_id).ok().unwrap().unwrap();
        let canvas = element.dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

        //function that triggers a custom event for contextlost
        let event_loop_proxy = event_loop.create_proxy();
        let context_lost = move || {
            let event = CustomEvent::ContextLost;
            info!("SENDING USER EVENT: context loss");
            event_loop_proxy.send_event(event).ok();
        };

        let event_loop_proxy = event_loop.create_proxy();
        let context_restored = move || {
            let event = CustomEvent::ContextRestored;
            info!("SENDING USER EVENT: context restored");
            event_loop_proxy.send_event(event).ok();
        };

        let closure_lost = wasm_bindgen::closure::Closure::wrap(Box::new(move |_: web_sys::Event| {
            context_lost();
        }) as Box<dyn FnMut(_)>);
        let closure_restored = wasm_bindgen::closure::Closure::wrap(Box::new(move |_: web_sys::Event| {
            context_restored();
        }) as Box<dyn FnMut(_)>);

        //https://developer.mozilla.org/en-US/docs/Web/Events
        //https://rustwasm.github.io/wasm-bindgen/api/web_sys/struct.HtmlCanvasElement.html#method.add_event_listener_with_callback
        canvas
            .add_event_listener_with_callback("webglcontextlost", closure_lost.as_ref().unchecked_ref())
            .unwrap();
        canvas
            .add_event_listener_with_callback("webglcontextrestored", closure_restored.as_ref().unchecked_ref())
            .unwrap();
        closure_lost.forget();
        closure_restored.forget();
    }

    // Queries for the size of an html element like the canvas for which we are
    // drawing to
    #[cfg(target_arch = "wasm32")]
    fn get_html_elem_size(selector: &str) -> Option<winit::dpi::LogicalSize<f32>> {
        //https://github.com/bevyengine/bevy/blob/main/crates/bevy_winit/src/web_resize.rs#L61
        let win = web_sys::window().unwrap();
        let doc = win.document().unwrap();
        let element = doc.query_selector(selector).ok()??;
        let parent_element = element.parent_element()?;
        let rect = parent_element.get_bounding_client_rect();
        return Some(winit::dpi::LogicalSize::new(rect.width() as f32, rect.height() as f32));
    }

    #[cfg(target_arch = "wasm32")]
    pub fn resize_to_canvas(&self) {
        if let Some(size) = Self::get_html_elem_size(&self.canvas_id_parsed.as_ref().unwrap()) {
            //TODO: remove
            warn!("size is {:?}", size);
            let event_resize = CustomEvent::Resize(size.width, size.height);
            self.runner.event_loop_proxy.send_event(event_resize).ok();
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        debug!("resizing new_size is {:?}", new_size);
        let max_2d_size = self.gpu_res.as_ref().unwrap().gpu.limits().max_texture_dimension_2d;
        if new_size.width > 16 && new_size.height > 16 && new_size.width < max_2d_size && new_size.height < max_2d_size {
            let gpu_res = self.gpu_res.as_mut().unwrap();
            //window and surface
            self.window_size = new_size;
            gpu_res.surface_config.width = new_size.width;
            gpu_res.surface_config.height = new_size.height;
            gpu_res.surface.configure(gpu_res.gpu.device(), &gpu_res.surface_config);
            gpu_res.request_redraw();

            //camera aspect ratio
            //camera has not yet been initialized so there is nothing to do
            if self.scene.world.has::<Projection>(self.camera.entity).unwrap() {
                self.camera
                    .set_aspect_ratio(new_size.width as f32 / new_size.height as f32, &mut self.scene);
            }

            //gui
            #[cfg(feature = "with-gui")]
            if let Some(ref mut gui) = gpu_res.gui {
                gui.resize(new_size.width, new_size.height);
            }
        } else {
            error!("trying to resize to unsuported size of {new_size:?}");
        }
        //camera
    }

    //Only a conveneince function so that you don't have to do
    // viewer.gpu_res.request_redraw()
    pub fn request_redraw(&mut self) {
        if let Some(gpu_res) = self.gpu_res.as_mut() {
            gpu_res.request_redraw();
        } else {
            error!("No gpu_res created yet");
        }
    }

    /// Runs only one update of the rendering loop by processing events and
    /// rendering a single frame. Useful for cases when the rendering loop needs
    /// to be created manually. WASM cannot compile this because of internal
    /// calls to `run_return`()
    #[cfg(not(target_arch = "wasm32"))]
    pub fn update(&mut self) {
        self.event_loop_one_iter();
        let _ = self.render();
    }

    /// Processes a custom event that we defined. If it matches one of the
    /// events, returns true
    fn process_custom_resize_events(&mut self, event: &Event<CustomEvent>) -> bool {
        match event {
            Event::UserEvent(CustomEvent::Resize(new_width, new_height)) => {
                debug!("rs: handling resize canvas: {:?}", event);
                let logical_size = winit::dpi::LogicalSize {
                    width: *new_width,
                    height: *new_height,
                };
                // the event should be handled by resizing the logical size so the width and
                // height are floats https://github.com/bevyengine/bevy/blob/eb485b1acc619baaae88d5daca0a311b95886281/crates/bevy_winit/src/web_resize.rs#L34C13-L34C46
                //TODO check if it's needed to get the return value and use it for the
                // self.resize
                let _ = self.gpu_res.as_ref().unwrap().window.request_inner_size(logical_size);

                //no need to resize here. Resizing the canvas with window.set_inner_size will
                // trigger a Window::Resized event sometimes it's needed to
                // resize here, I'm not very sure why but in some cases the Window:resize event
                // is sent AFTER the rendering one
                self.resize(self.gpu_res.as_ref().unwrap().window.inner_size()); //surface requires physical pixels which are not the same as the logical pixels
                                                                                 // for the window

                true //return true so that this branch is taken in the match
            }
            _ => false, //doesn't match any of the events, some other function will need to process this event
        }
    }

    fn process_custom_context_event(&mut self, event: &Event<CustomEvent>, event_loop: &ActiveEventLoop) -> bool {
        match event {
            Event::UserEvent(event) => {
                match event {
                    CustomEvent::ContextLost => {
                        info!("rs: handling context lost");
                        self.suspend();
                        true //return true so that this branch is taken in the
                             // match
                    }
                    CustomEvent::ContextRestored => {
                        info!("rs: handling context restored");
                        self.resume(event_loop);
                        true //return true so that this branch is taken in the
                             // match
                    }
                    _ => false, //doesn't match any of the events, some other function will need to process this event
                }
            }
            _ => false, //doesn't match any of the events, some other function will need to process this event
        }
    }

    #[allow(clippy::collapsible_match)]
    fn process_custom_other_event(&mut self, event: &Event<CustomEvent>, event_loop: &ActiveEventLoop) -> bool {
        match event {
            Event::UserEvent(event) => {
                match event {
                    CustomEvent::ResumeLoop => {
                        info!("rs: handling custom resume loop");
                        self.resume(event_loop);
                        true
                    }
                    CustomEvent::StopLoop => {
                        info!("rs: handling custom stop loop");
                        self.runner.is_running = false;
                        event_loop.exit();
                        true
                    }
                    _ => false, //doesn't match any of the events, some other function will need to process this event
                }
            }
            _ => false, //doesn't match any of the events, some other function will need to process this event
        }
    }

    /// Processes loop-related events like stopping the loop. If it matches one
    /// of the events, returns true
    // #[allow(unused_variables)]
    // fn process_loop_events(&mut self, event: &Event<CustomEvent>, event_loop: &ActiveEventLoop) -> bool {
    //     // Check if gpu_res is initialized
    //     if let Some(gpu_res) = self.gpu_res.as_mut() {
    //         match event {
    //             Event::AboutToWait => {
    //                 // For native, it essentially does pooling
    //                 #[cfg(not(target_arch = "wasm32"))]
    //                 gpu_res.request_redraw();
    //                 true
    //             }
    //             Event::Resumed => {
    //                 debug!("rs: Resumed");
    //                 self.resume(event_loop);
    //                 true
    //             }
    //             Event::Suspended => {
    //                 debug!("rs: Suspended");
    //                 self.suspend();
    //                 true
    //             }
    //             Event::LoopExiting => {
    //                 self.runner.is_running = false;
    //                 true
    //             }
    //             _ => false, // Doesn't match any of the events, some other function will need to process this event
    //         }
    //     } else {
    //         // Handle cases where gpu_res is not yet initialized
    //         match event {
    //             Event::Resumed => {
    //                 debug!("rs: Resumed");
    //                 self.resume(event_loop);
    //                 true
    //             }
    //             Event::LoopExiting => {
    //                 self.runner.is_running = false;
    //                 true
    //             }
    //             _ => false, // If gpu_res is None and no other relevant event, do nothing
    //         }
    //     }
    // }

    /// Processes events like closing, resizing etc. If it matches one of the
    /// events, returns true
    #[allow(clippy::too_many_lines)]
    fn process_window_events(&mut self, event: &WindowEvent, event_loop: &ActiveEventLoop) -> bool {
        match event {
            WindowEvent::RedrawRequested => {
                {
                    let gpu_res = self.gpu_res.as_mut().unwrap();
                    gpu_res.redraw_requested = false;
                }

                if !self.runner.do_render {
                    return false;
                }

                debug!("gloss: render");

                //frame was already started before so there is no need to do another render.
                // Check comment in runner.frame_is_started on why this may happen
                if self.runner.frame_is_started {
                    debug!("the frame was already started, we are ignoring this re-render");
                    return true;
                }
                self.start_frame();

                match self.render() {
                    Ok(()) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        self.resize(self.gpu_res.as_ref().unwrap().window.inner_size());
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        error!("SurfaceError: out of memory");
                        //TODO how to best handle this?
                    }
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => error!("SurfaceError: timeout"),
                }
                debug!("finsihed handing RedrawRequested");
                true
            }
            WindowEvent::Resized(physical_size) => {
                self.resize(*physical_size);
                true
            }
            // WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
            //     // new_inner_size is &mut so w have to dereference it twice
            //     // self.resize(**new_inner_size);
            //     warn!("New scale factor {}", scale_factor);
            //     true
            // }
            WindowEvent::DroppedFile(path_buf) => {
                info!("Dropped file {:?}", path_buf);
                let path = path_buf.to_str().unwrap();

                //do it so that we process the events we accumulated and actually get a proper
                // mouse position for egui
                self.render().ok();

                //Gui tries to handle the drop file event
                #[allow(unused_mut)]
                #[cfg(feature = "with-gui")]
                {
                    let gpu_res = self.gpu_res.as_mut().unwrap();
                    let gui = gpu_res.gui.as_mut().unwrap();
                    if gui.is_hovering() {
                        gui.on_drop(path_buf, &mut self.scene);
                        return true;
                    }
                }

                //Gloss tries to handle the drop file event
                let filetype = match path_buf.extension() {
                    Some(extension) => FileType::find_match(extension.to_str().unwrap_or("")),
                    None => FileType::Unknown,
                };
                match filetype {
                    FileType::Obj | FileType::Ply => {
                        let builder = Geom::build_from_file(path);
                        let name = self.scene.get_unused_name();
                        self.scene.get_or_create_entity(&name).insert_builder(builder);
                        return true;
                    }
                    FileType::Unknown => {
                        info!(
                            "Gloss doesn't know how to handle dropped file {:?}. trying to let plugins handle it",
                            path
                        );
                    }
                }

                //try the plugins to see if they have an event for dropped file
                let event = crate::plugin_manager::Event::DroppedFile(RString::from(path));
                let handled = self.plugins.try_handle_event(&mut self.scene, &mut self.runner, &event);

                if !handled {
                    info!("Neither Gloss nor any of the plugin could load the dropped file {:?}", path);
                    return false;
                }

                true
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Pressed && !event.repeat {
                        #[allow(clippy::single_match)] //we might add more keys afterwards
                        match code {
                            KeyCode::KeyH => {
                                //disable gui
                                #[cfg(feature = "with-gui")]
                                {
                                    if let Some(gpu_res) = self.gpu_res.as_mut() {
                                        if let Some(gui) = gpu_res.gui.as_mut() {
                                            gui.hidden = !gui.hidden;
                                            gpu_res.request_redraw();
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                true
            }

            WindowEvent::CloseRequested {} | WindowEvent::Destroyed {} => {
                event_loop.exit();
                true
            }

            WindowEvent::Occluded(_) => {
                self.camera.reset_all_touch_presses(&mut self.scene);
                true
            }
            _ => false, //doesn't match any of the events, some other function will need to process this event
        }
    }

    /// Processes events like mouse drag, scroll etc. If it matches one of the
    /// events, returns true
    #[allow(clippy::cast_possible_truncation)]
    fn process_input_events(&mut self, event: &WindowEvent) -> bool {
        //camera has not yet been initialized so there is nothing to do
        if !self.camera.is_initialized(&self.scene) {
            return false;
        }

        let gpu_res = self.gpu_res.as_mut().unwrap();

        let consumed = match event {
            WindowEvent::MouseInput { button, state, .. } => {
                // self.camera.mouse_pressed = *state == ElementState::Pressed;
                if *state == ElementState::Pressed {
                    self.camera.mouse_pressed(button, &mut self.scene);
                } else {
                    self.camera.mouse_released(&mut self.scene);
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera.process_mouse_scroll(delta, &mut self.scene);
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.camera.process_mouse_move(
                    position.x as f32,
                    position.y as f32,
                    self.window_size.width,
                    self.window_size.height,
                    &mut self.scene,
                );
                true
            }
            WindowEvent::Touch(touch) => {
                if touch.phase == TouchPhase::Started {
                    self.camera.touch_pressed(touch, &mut self.scene);
                }
                if touch.phase == TouchPhase::Ended || touch.phase == TouchPhase::Cancelled {
                    self.camera.touch_released(touch, &mut self.scene);
                }
                if touch.phase == TouchPhase::Moved {
                    self.camera
                        .process_touch_move(touch, self.window_size.width, self.window_size.height, &mut self.scene);
                }
                true
            }
            _ => false, //doesn't match any of the events, some other function will need to process this event
        };

        if consumed {
            gpu_res.request_redraw();
        }

        consumed
    }

    #[cfg(feature = "with-gui")]
    fn process_gui_events(&mut self, event: &WindowEvent) -> EventResponse {
        let gpu_res = self.gpu_res.as_mut().unwrap();
        //the gui need to take window as reference. To make the borrow checker happy we
        // take ownership of the gui
        if let Some(mut gui) = gpu_res.gui.take() {
            let response = gui.on_event(&gpu_res.window, event);
            gpu_res.gui = Some(gui); //put the gui back
            response
        } else {
            EventResponse {
                repaint: false,
                consumed: false,
            }
        }
    }

    /// Processes all events from the event loop. These are all `WindowEvents`
    fn process_all_events(&mut self, event: &Event<CustomEvent>, event_loop: &ActiveEventLoop) {
        if !self.runner.is_running {
            // If we receive a draw event and the loop isn't running we basically ignore it
            // but we have to notify that we have no more draw event queued
            if let Event::WindowEvent { ref event, window_id: _ } = event {
                if event == &WindowEvent::RedrawRequested {
                    //dropping request redraw event
                    let gpu_res = self.gpu_res.as_mut().unwrap();
                    gpu_res.redraw_requested = false;
                }
            }
            //even if the event loop isn't running we still want to respond to resizes of the canvas because missing this event means that when we actually start the event loop we will render at the wrong size
            if let Event::WindowEvent {
                event: WindowEvent::Resized(physical_size),
                window_id: _,
            } = event
            {
                self.resize(*physical_size);
            }

            return; //TODO wait for like 1ms or something
        }

        // self.process_loop_events(event, event_loop);

        //TODO technically we don't need to check the rest of if we already processed
        // the event as a custom event or a loop event

        match event {
            Event::WindowEvent { ref event, window_id } if *window_id == self.gpu_res.as_ref().unwrap().window.id() => {
                // Now we start doing rendering stuff but we skip them if we don't have yet a
                // correct size for rendering, this can happen inf the wasm canvas wasn't
                // initialized yet which can happen if the user moves to a different tab before
                // the viewer is fully initialized
                if self.window_size.height < 16 || self.window_size.width < 16 {
                    warn!("Skipping rendering and trying again to resize. Window size is {:?}", self.window_size);
                    // If we are on wasm we try to reget the canvas size and request another redraw
                    // and hope that now the canvas is initialized and gives us a proper size
                    #[cfg(target_arch = "wasm32")]
                    self.resize_to_canvas();
                    let gpu_res = self.gpu_res.as_mut().unwrap();
                    gpu_res.request_redraw();
                    return;
                }

                self.process_window_events(event, event_loop); //process this first because it does resizing
                cfg_if::cfg_if! {
                    if #[cfg(feature = "with-gui")]{
                        let res = self.process_gui_events(event);
                        //if the gui consumed the event then we don't pass it to the rest of the input pipeline
                        if res.consumed {
                            self.gpu_res.as_mut().unwrap().request_redraw();
                        } else {
                            self.process_input_events(event);
                        }
                        //HACK his is a hack to deal with the fact that clicking on the egui gizmos triggers a mouse press on the camera and then it gets locked there.
                        // if self.gpu_res.as_ref().unwrap().gui.wants_pointer_input() {
                        if let Some(ref gui) = self.gpu_res.as_mut().unwrap().gui {
                            if gui.wants_pointer_input() {
                                self.camera.mouse_released(&mut self.scene);
                            }
                        }
                    }else{ //if we don't have a gui, we just process input events
                        self.process_input_events(event);
                    }
                }
            }
            _ => {}
        }
    }

    /// Processes one iteration of the event loop. Useful when running the loop
    /// manually using [`Viewer::update`]
    #[allow(unused)]
    #[cfg(not(target_arch = "wasm32"))] //wasm cannot compile the run_return() call so we just disable this whole
                                        // function
    fn event_loop_one_iter(&mut self) {
        // remove the eventloop from self
        // We remove this so that we have ownership over it.
        // https://github.com/bevyengine/bevy/blob/eb485b1acc619baaae88d5daca0a311b95886281/crates/bevy_winit/src/lib.rs#L299
        // https://users.rust-lang.org/t/wrapping-a-winit-window-with-a-struct/40750/6

        use winit::platform::pump_events::EventLoopExtPumpEvents;
        let mut event_loop = self.runner.event_loop.take().unwrap();
        self.runner.is_running = true;
        self.runner.do_render = false; //if we use run_return we don't do the rendering in this event processing and
                                       // we rather do it manually
        let timeout = Some(Duration::ZERO);
        event_loop.pump_app_events(timeout, self);
        // self.process_all_events(event, event_loop);
        // event_loop.pump_events(timeout, |event, event_loop| {
        //     self.process_all_events(&event, event_loop);
        // });

        self.runner.is_running = false;

        //put the event loop back into self
        self.runner.event_loop = Some(event_loop);
    }

    //called at the beggining of the render and sets the time that all systems will
    // use
    pub fn start_frame(&mut self) -> Duration {
        // info!("rs: start frame");
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.gpu_res.is_none() {
                self.event_loop_one_iter(); // TODO: ?
            }
            assert!(self.gpu_res.is_some(), "GPU Res has not been created!");
        }
        // First time we call this we do a warmup render to initialize everything
        if !self.runner.did_warmup {
            self.runner.did_warmup = true; //has to be put here because warmup actually calls start_frame and we don't
                                           // want an infinite recurrsion
            self.warmup();
            self.warmup(); // TODO: for testing
        }

        self.runner.update_dt();
        debug!("after update dt it is {:?}", self.runner.dt());
        self.runner.time_last_frame = Instant::now();

        self.runner.frame_is_started = true;

        self.runner.dt
    }

    /// # Panics
    /// Will panic if the `gpu_resources` have not been created
    /// # Errors
    /// Will return error if the surface texture cannot be adquired
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.runner.frame_is_started {
            error!("The frame was not started so this might contain stale dt. Please use viewer.start_frame() before doing a v.render()");
        }

        //we actually take the init time as being the fist time we render, otherwise
        // the init time would be higher since some other processing might happen
        // between creating the viewer and actually rendering with it
        if self.runner.first_time {
            self.runner.time_init = Instant::now();
        }

        let gpu_res = self.gpu_res.as_mut().unwrap();
        self.plugins.run_logic_systems(gpu_res, &mut self.scene, &mut self.runner, true);

        //get surface texture (which can fail and return an SurfaceError)
        let output = gpu_res.surface.get_current_texture()?;
        let out_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_width = output.texture.width();
        let out_height = output.texture.height();

        //render to_texture of the size of the surface
        let dt = self.runner.dt();

        self.camera.on_window_resize(out_width, out_height, &mut self.scene);

        //TODO return the textured final so we can just plug it into blit pass without
        // doing renderer.rendered_tex
        gpu_res.renderer.render_to_view(
            &out_view,
            // out_width,
            // out_height,
            &gpu_res.gpu,
            &mut self.camera,
            &mut self.scene,
            &mut self.config,
            dt,
        );

        //render gui
        //TODO pass the whole renderer and the scene so we can do gui stuff on them
        #[cfg(feature = "with-gui")]
        if let Some(ref mut gui) = gpu_res.gui {
            gui.render(
                &gpu_res.window,
                &gpu_res.gpu,
                &gpu_res.renderer,
                &self.runner,
                &mut self.scene,
                &self.plugins,
                &mut self.config,
                &out_view,
            );
        }

        //swap
        output.present();

        self.runner.first_time = false;
        self.runner.frame_is_started = false;

        Ok(())
    }

    /// Runs the rendering loop automatically. This function does not return as
    /// it will take full control of the loop. If you need to still have control
    /// over the loop use [`Viewer::update`]
    // #[allow(clippy::missing_panics_doc)]
    // pub fn run(mut self) {
    //     // pub fn run(&'static mut self) {
    //     // has to take self by value because this
    //     // https://www.reddit.com/r/rust/comments/cgdhzb/newbie_question_how_do_i_avoid_breaking_apart_my/
    //     // https://users.rust-lang.org/t/cannot-move-out-of-self-event-loop-which-is-behind-a-mutable-reference/67363/6
    //     // https://stackoverflow.com/questions/76577042/why-do-i-get-a-borrowed-data-escapes-outside-of-method-error
    //     // remove the eventloop from self
    //     // We remove this so that we have ownership over it.
    //     // https://github.com/bevyengine/bevy/blob/eb485b1acc619baaae88d5daca0a311b95886281/crates/bevy_winit/src/lib.rs#L299
    //     // https://users.rust-lang.org/t/wrapping-a-winit-window-with-a-struct/40750/6

    //     let event_loop = self.runner.event_loop.take().unwrap();
    //     self.runner.is_running = self.runner.autostart;

    //     // let _ = event_loop.run_app(&mut self);

    //     cfg_if::cfg_if! {
    //         if #[cfg(not(target_arch = "wasm32"))]{
    //             let _ = event_loop.run_app(&mut self);
    //         }else{
    //             let _ = event_loop.spawn_app(self);
    //         }
    //     }
    // }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(&mut self) {
        let event_loop = self.runner.event_loop.take().unwrap();
        self.runner.is_running = self.runner.autostart;
        let _ = event_loop.run_app(self);
    }

    #[cfg(target_arch = "wasm32")]
    pub fn run(mut self) {
        let event_loop = self.runner.event_loop.take().unwrap();
        self.runner.is_running = self.runner.autostart;
        let _ = event_loop.spawn_app(self);
    }

    /// Same as the run function but uses a static reference to the ``Viewer``.
    /// Useful for web apps when the viewer is a static member.
    #[allow(clippy::missing_panics_doc)]
    #[allow(unreachable_code)] //the lines after the event loop are actually ran on wasm
    pub fn run_static_ref(&'static mut self) {
        let event_loop = self.runner.event_loop.take().unwrap();
        self.runner.is_running = self.runner.autostart;

        // let _ = event_loop.run_app(self);
        cfg_if::cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))]{
                let _ = event_loop.run_app(self);
            }else{
                let _ = event_loop.spawn_app(self);
            }
        }
    }

    /// if the event loop is running we can call this to destroy everything and
    /// be able to call ``viewer.run()`` again and potentially connect to a new
    /// canvas
    #[allow(clippy::missing_panics_doc)]
    pub fn recreate_event_loop(&mut self) {
        self.stop_event_loop();
        self.suspend(); //need to destroy window because we potentially are connecting to new canvas
        let runner = Runner::new(&self.canvas_id_parsed);
        self.runner = runner;
        // self.resume(runner.event_loop.as_ref().unwrap());
        let event_stop = CustomEvent::ResumeLoop;
        self.runner.event_loop_proxy.send_event(event_stop).ok();
    }

    pub fn stop_event_loop(&self) {
        let event_stop = CustomEvent::StopLoop;
        self.runner.event_loop_proxy.send_event(event_stop).ok();
    }

    fn resume(&mut self, event_loop: &ActiveEventLoop) {
        // info!("RS: resume");
        self.runner.is_running = self.runner.autostart;
        if self.gpu_res.is_none() {
            self.gpu_res = Some(GpuResources::new(
                event_loop,
                &self.runner.event_loop_proxy,
                &self.canvas_id_parsed.clone(),
                &self.config,
            ));
            // We set all the components to changed because we want them to be reuploaded to
            // gpu. Don't set them as added because some systems rely on components being
            // added to run only once.
            self.scene.world.set_trackers_changed(); //TODO set it to changed, and push hecs to github

            // self.gpu_res.as_mut().unwrap().window.request_redraw();
            self.gpu_res.as_mut().unwrap().request_redraw();
        }
        #[cfg(target_arch = "wasm32")]
        self.resize_to_canvas()
    }

    pub fn suspend(&mut self) {
        // Drop surface and anything to do with the gl context since it's probable that
        // we lost it https://docs.rs/winit/latest/winit/event/enum.Event.html
        info!("RS: suspend");
        self.runner.is_running = false;
        self.scene.remove_all_gpu_components();
        self.gpu_res.take();
        self.camera.reset_all_touch_presses(&mut self.scene);
    }

    // First time we render will take longer since we have to possibly load a lot of
    // data, textures, smpl_models etc, This can cause the animation time for the
    // second frame to be very long which will look as if we skipped part of the
    // animation. Therefore we do a warmup at the first rendered frame
    pub fn warmup(&mut self) {
        debug!("Starting warmup");
        // self.resume(&self.runner.event_loop);
        self.start_frame();
        self.run_manual_plugins(); //auto plugins will run when we do self.render(), but here we also need to run
                                   // the manual ones

        // We cannot do update() because update requires .take() of the event loop. But
        // the event loop is already running so the .take() will fails
        let _ = self.render();
        self.reset_for_first_time();
        debug!("finished warmup");
    }

    pub fn reset_for_first_time(&mut self) {
        self.runner.first_time = true;
    }

    pub fn add_logic_system(&mut self, sys: LogicSystem) {
        self.plugins.logic_systems.push(Tuple2(sys, SystemMetadata::default()));
    }

    #[cfg(feature = "with-gui")]
    pub fn add_gui_system(&mut self, sys: GuiSystem) {
        self.plugins.gui_systems.push(Tuple2(sys, SystemMetadata::default()));
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn run_manual_plugins(&mut self) {
        {
            // if self.gpu_res.is_some() {
            let gpu_res = self.gpu_res.as_mut().unwrap();
            self.plugins.run_logic_systems(gpu_res, &mut self.scene, &mut self.runner, false);
            // }
        }
    }

    pub fn insert_plugin<T: Plugin + 'static>(&mut self, plugin: &T) {
        self.plugins.insert_plugin(plugin);
    }

    /// # Panics
    /// Will panic if the `gpu_resources` have not been created
    pub fn wait_gpu_finish(&self) -> wgpu::MaintainResult {
        self.gpu_res.as_ref().unwrap().gpu.device().poll(wgpu::Maintain::Wait)
    }

    ////////////////////////////////////////////////////////////////////////////

    fn create_window(
        event_loop: &ActiveEventLoop,
        _event_loop_proxy: &EventLoopProxy<CustomEvent>,
        _canvas_id: &Option<String>,
    ) -> Result<Window, Box<dyn Error>> {
        // TODO read-out activation token.

        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes().with_title("Gloss").with_maximized(true);

        // #[cfg(web_platform)]
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            let window = web_sys::window().unwrap();
            let document = window.document().unwrap();
            let canvas = document
                .query_selector(&_canvas_id.as_ref().unwrap())
                .expect("Cannot query for canvas element.");
            if let Some(canvas) = canvas {
                let canvas = canvas.dyn_into::<web_sys::HtmlCanvasElement>().ok();

                //prevent_default prevents the window for getting web keyboard events like F12 and Ctrl+r https://github.com/rust-windowing/winit/issues/1768
                window_attributes = window_attributes.with_canvas(canvas).with_prevent_default(false).with_append(false)
            } else {
                panic!("Cannot find element: {:?}.", _canvas_id.as_ref().unwrap());
            }
        }

        let window = event_loop.create_window(window_attributes)?;

        Ok(window)
    }
}

impl ApplicationHandler<CustomEvent> for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.resume(event_loop);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: CustomEvent) {
        self.process_custom_context_event(&Event::UserEvent(event), event_loop);
        self.process_custom_resize_events(&Event::UserEvent(event));
        self.process_custom_other_event(&Event::UserEvent(event), event_loop);
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        if window_id != self.gpu_res.as_ref().unwrap().window.id() {
            return;
        }
        self.process_all_events(&Event::WindowEvent { window_id, event }, event_loop);
    }
    #[allow(unused_variables)]
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(gpu_res) = self.gpu_res.as_mut() {
                gpu_res.request_redraw();
            }
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        debug!("Handling Suspended event");
        self.suspend();
    }

    fn exiting(&mut self, event_loop: &ActiveEventLoop) {
        self.runner.is_running = false;
        event_loop.exit();
    }
}

/// We target Vulkan for native and WebGL for wasm. We only use Vulkan for
/// native because that allows us to use the `PyTorch` and wgpu interoperability
pub fn supported_backends() -> wgpu::Backends {
    if cfg!(target_arch = "wasm32") {
        // Web - WebGL is used automatically when wgpu is compiled with `webgl` feature.
        wgpu::Backends::GL
    } else {
        // For Native
        wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::VULKAN | wgpu::Backends::METAL)
    }
}

//get the adapter with the following priorities
//if the WGPU_VISIBLE_DEVICES is set get the adaptor with that id, if not get the adaptor from CUDA_VISIBLE_DEVICES and finally if none are set just get whatever wgpu wants
pub fn get_adapter(instance: &wgpu::Instance, surface: Option<&wgpu::Surface>) -> wgpu::Adapter {
    #[cfg(not(target_arch = "wasm32"))]
    fn remove_from_vec(vec: &mut Vec<wgpu::Adapter>, idx_str: &str) -> wgpu::Adapter {
        let idx = idx_str.split(',').next().unwrap().parse::<usize>().unwrap(); //in the case we pass multiple indexes to CUDA_VISIBLE_DEVICES=0,1,2 we use the first index

        assert!(
            (0..vec.len()).contains(&idx),
            "Tried to index device with idx {} but we only have detected {} devices",
            idx,
            vec.len()
        );

        info!("Selecting adapter with idx {}", idx);
        vec.remove(idx)
    }

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")]{
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: surface,
                    force_fallback_adapter: false,
                })
                .block_on()
                .expect("An adapter could not be found. Maybe there's a driver issue on your machine?")
        }else {
            let mut adapters = enumerate_adapters(instance);

            let wgpu_dev_id = std::env::var("WGPU_VISIBLE_DEVICES");
            let cuda_dev_id = std::env::var("CUDA_VISIBLE_DEVICES");
            match wgpu_dev_id {
                Ok(idx) => remove_from_vec(&mut adapters, &idx),
                Err(_) => match cuda_dev_id {
                    Ok(idx) => remove_from_vec(&mut adapters, &idx),
                    Err(_) => instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            compatible_surface: surface,
                            force_fallback_adapter: false,
                        })
                        .block_on()
                        .expect("An adapter could not be found. Maybe there's a driver issue on your machine?"),
                },
            }
        }
    }
}

//sort adapter so that the GPU ones come first, we don't want the llvmp pipe one (cpu) being somewhere in the middle since it messes with the index based selection
//despite the sorting, we cannot guarantee that the order of adapters is the same as nvidia-smi. In order to find out which adapter is which we probably need to resort to trial-and-and error by starting Gloss on multiple GPUs and check which nvidia-smi on which one it's running
//for more info check: https://github.com/pygfx/wgpu-py/issues/482
#[cfg(not(target_arch = "wasm32"))]
pub fn enumerate_adapters(instance: &wgpu::Instance) -> Vec<wgpu::Adapter> {
    let mut adapters = instance.enumerate_adapters(wgpu::Backends::all());

    //this sorts by the distance of device type to the discretegpu, since the sort is stable the devices that are discrete gpus (distance=0) are put first
    adapters.sort_by_key(|x| (x.get_info().device_type as i32 - wgpu::DeviceType::DiscreteGpu as i32).abs());

    adapters
}

/// Custom resize event emited by canvas resizes on the web.
#[derive(Debug, Clone, Copy)]
pub enum CustomEvent {
    Resize(f32, f32),
    ContextLost,
    ContextRestored,
    ResumeLoop,
    StopLoop,
}
