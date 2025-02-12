use crate::{
    components::ConfigChanges,
    config::{Config, RenderConfig},
};

use crate::forward_renderer::render_passes::{prepass::PrePass, shadow_pass::ShadowPass, upload_pass::UploadPass};

use crate::{camera::Camera, scene::Scene};

use easy_wgpu::{
    framebuffer::{FrameBuffer, FrameBufferBuilder},
    gpu::Gpu,
    texture::{TexParams, Texture},
};

use enum_map::Enum;
use log::debug;

use super::main_pass::MainPass;

#[derive(Debug, Enum)]
pub enum OffscreenTarget {
    //order determines the binding index in the shader
    Color,     //for drawing to offscreen
    MSAAColor, //useful for drawing during MSAA and then resolving to another view
    Depth,
}

///  Contains long-living objects that will stay alive for the whole duration of
/// the Renderer
pub struct RenderData {
    pub framebuffer: FrameBuffer<OffscreenTarget>,
}
impl RenderData {
    pub fn new(gpu: &Gpu, params: &RenderConfig, surface_format: Option<wgpu::TextureFormat>) -> Self {
        //create long-living objects
        let frambuffer_builder = FrameBufferBuilder::<OffscreenTarget>::new(128, 128);
        let depth_texture_usage = if cfg!(target_arch = "wasm32") {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            // because for headless rendering we want to download it to cpu
        };

        let mut offscreen_color_format = wgpu::TextureFormat::Rgba8Unorm;
        if params.offscreen_color_float_tex {
            offscreen_color_format = wgpu::TextureFormat::Rgba32Float;
        }

        let framebuffer = frambuffer_builder
            .add_render_target(
                gpu.device(),
                OffscreenTarget::Color,
                surface_format.unwrap_or(offscreen_color_format),
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, /* because for headless rendering we want to download it
                                                                                         * to cpu */
                TexParams::default(),
            )
            .add_render_target(
                gpu.device(),
                OffscreenTarget::MSAAColor,
                surface_format.unwrap_or(wgpu::TextureFormat::Rgba8Unorm),
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                TexParams {
                    sample_count: params.msaa_nr_samples,
                    ..Default::default()
                },
            )
            .add_render_target(
                gpu.device(),
                OffscreenTarget::Depth,
                wgpu::TextureFormat::Depth32Float,
                depth_texture_usage,
                TexParams {
                    sample_count: params.msaa_nr_samples,
                    ..Default::default()
                },
            )
            .build(gpu.device());

        Self { framebuffer }
    }
}

pub struct RenderPasses {
    pub upload_pass: UploadPass, //uploads from CPU to GPU everything that we need globally like settings, camera parameters, lights, etc.
    shadow_pass: ShadowPass,     //renders depth maps towards all lights
    main_pass: MainPass,
}
impl RenderPasses {
    pub fn new(gpu: &Gpu, params: &RenderConfig, color_target_format: wgpu::TextureFormat, depth_target_format: wgpu::TextureFormat) -> Self {
        let upload_pass = UploadPass::new(gpu, params);
        let shadow_pass = ShadowPass::new(gpu);
        let main_pass = MainPass::new(gpu, params, color_target_format, depth_target_format);
        Self {
            upload_pass,
            shadow_pass,
            main_pass,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, out_view: &wgpu::TextureView, data: &RenderData, gpu: &Gpu, camera: &mut Camera, scene: &mut Scene, config: &mut Config) {
        //update ubos
        let global_uniforms = self.upload_pass.run(gpu, camera, scene, &config.render);

        //render all the geoemtry towards shadow maps
        self.shadow_pass.run(gpu, global_uniforms, scene);

        self.main_pass
            .run(gpu, global_uniforms, &data.framebuffer, out_view, scene, &config.render);
    }
}

/// Renderer is the main point of entry for any rendering functionality. It
/// contains various rendering passes which are executed in sequence. I iterated
/// through the entities that need rendering from the ECS world and calls the
/// appropriate rendering passes depending on which components are present on
/// the entities.
pub struct Renderer {
    //long-living objects
    pub data: RenderData,
    //passes
    prepass: PrePass,
    pub passes: RenderPasses,
}

impl Renderer {
    pub fn new(gpu: &Gpu, params: &RenderConfig, surface_format: Option<wgpu::TextureFormat>) -> Self {
        let data = RenderData::new(gpu, params, surface_format);

        //passes
        // let debug_pass = DebugPass::new(gpu.device(), &surface_format.unwrap());
        let prepass = PrePass::new();

        //the color framebuffer will have the same the same format as the surface if
        // the surface_format.is_some()
        let color_target_format = data.framebuffer.get(OffscreenTarget::Color).unwrap().texture.format();
        let depth_target_format = data.framebuffer.get(OffscreenTarget::Depth).unwrap().texture.format();

        let passes = RenderPasses::new(gpu, params, color_target_format, depth_target_format);

        Self { data, prepass, passes }
    }

    /// Calls the full rendering functionality and writes the final image to a
    /// texture view. Useful when rendering directly to screen and there is no
    /// need to save the texture for later # Panics
    /// This function may panic if the GPU or camera is not available.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view(
        &mut self,
        out_view: &wgpu::TextureView,
        // width: u32,
        // height: u32,
        gpu: &Gpu,
        camera: &mut Camera,
        scene: &mut Scene,
        config: &mut Config,
        _dt: core::time::Duration,
    ) {
        self.begin_frame(gpu, camera, scene, config);

        self.passes.run(out_view, &self.data, gpu, camera, scene, config);

        self.end_frame(scene);
    }

    /// Calls the full rendering functionality and writes the final image to an
    /// internal texture which can be recovered using
    /// [`Renderer::rendered_tex`]. Useful when rendering on a headless machine
    /// and there is no need to render towards a window. Renders the scene
    /// to a texture.
    ///
    /// # Panics
    /// This function may panic if the GPU or camera is not available.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_texture(
        &mut self,
        // width: u32,
        // height: u32,
        gpu: &Gpu,
        camera: &mut Camera,
        scene: &mut Scene,
        config: &mut Config,
        _dt: core::time::Duration,
    ) {
        self.begin_frame(gpu, camera, scene, config);

        let out_view = &self.data.framebuffer.get(OffscreenTarget::Color).unwrap().view;
        self.passes.run(out_view, &self.data, gpu, camera, scene, config);

        self.end_frame(scene);
    }

    fn prepare_for_rendering(&mut self, gpu: &Gpu, camera: &mut Camera, scene: &mut Scene, config: &mut Config) {
        //modify config if needed
        if let Ok(delta) = scene.get_resource::<&ConfigChanges>() {
            config.apply_deltas(&delta);
        }

        //prepass
        self.prepass.run(gpu, camera, scene, config);
    }

    fn begin_frame(&mut self, gpu: &Gpu, camera: &mut Camera, scene: &mut Scene, config: &mut Config) {
        //resize gbuffer and other internal things if necessery if necessery
        let (width, height) = camera.get_target_res(scene);
        self.resize_if_necesary(width, height, gpu);
        self.prepare_for_rendering(gpu, camera, scene, config);
    }

    fn end_frame(&self, scene: &mut Scene) {
        //if we do manual ecs without the bevy system, we need to call clear trackers
        // so that the changed flag gets cleared for the next frame
        scene.world.clear_trackers();
    }

    /// # Panics
    /// Will panic if the framebuffer we render to does not have a target callet
    /// `Gtarget::Final`
    pub fn rendered_tex(&self) -> &Texture {
        self.data.framebuffer.get(OffscreenTarget::Color).unwrap()
    }
    /// # Panics
    /// Will panic if the framebuffer we render to does not have a target callet
    /// `Gtarget::Final`
    pub fn rendered_tex_mut(&mut self) -> &mut Texture {
        self.data.framebuffer.get_mut(OffscreenTarget::Color).unwrap()
    }
    /// # Panics
    /// Will panic if the framebuffer we render to does not have a target callet
    /// `Gtarget::Final`
    pub fn depth_buffer(&self) -> &Texture {
        self.data.framebuffer.get(OffscreenTarget::Depth).unwrap()
    }

    fn resize_if_necesary(&mut self, width: u32, height: u32, gpu: &Gpu) {
        if self.data.framebuffer.width != width || self.data.framebuffer.height != height {
            debug!(
                "resizing framebuffer because it is size {}, {}",
                self.data.framebuffer.width, self.data.framebuffer.height
            );
            self.data.framebuffer.resize(gpu.device(), width, height);
        }
    }
}
