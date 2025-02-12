use crate::config::RenderConfig;
use log::trace;
use nalgebra as na;

use crate::scene::Scene;

use easy_wgpu::gpu::Gpu;

use super::{line_pipeline::LinePipeline, mesh_pipeline::MeshPipeline, point_pipeline::PointPipeline, upload_pass::PerFrameUniforms};

use crate::forward_renderer::{render_passes::pipeline_runner::PipelineRunner, renderer::OffscreenTarget};
use easy_wgpu::framebuffer::FrameBuffer;

/// Render all the meshes from the scene
pub struct MainPass {
    mesh_pipeline: MeshPipeline,
    point_pipeline: PointPipeline,
    line_pipeline: LinePipeline,
}

impl MainPass {
    pub fn new(gpu: &Gpu, params: &RenderConfig, color_target_format: wgpu::TextureFormat, depth_target_format: wgpu::TextureFormat) -> Self {
        let mesh_pipeline = MeshPipeline::new(gpu, params, color_target_format, depth_target_format);
        let point_pipeline = PointPipeline::new(gpu, params, color_target_format, depth_target_format);
        let line_pipeline = LinePipeline::new(gpu, params, color_target_format, depth_target_format);
        Self {
            mesh_pipeline,
            point_pipeline,
            line_pipeline,
        }
    }

    /// # Panics
    /// Will panic if the gbuffer does not have the correct textures that are
    /// needed for the render pass
    #[allow(clippy::too_many_lines)]
    pub fn run(
        &mut self,
        gpu: &Gpu,
        per_frame_uniforms: &PerFrameUniforms,
        offscreen_fb: &FrameBuffer<OffscreenTarget>,
        out_view: &wgpu::TextureView,
        scene: &mut Scene,
        render_params: &RenderConfig,
    ) {
        self.begin_pass();

        //tonemap bg color
        let aces = utils_rs::tonemap::AcesFitted::new();
        let bg_color_vec = render_params.bg_color.fixed_rows::<3>(0).clone_owned();
        let bg_color_tonemapped = aces.tonemap(&bg_color_vec);
        let bg_color_tonemapped_gamma = na::Vector3::from_iterator(bg_color_tonemapped.iter().map(|x| x.powf(1.0 / 2.2)));

        //queries have to live as long as the encoder so we put them outside of the
        // pipelines themselves this is because the encoder needs to have
        // reference to the vertex atributes and those atributes are only alive as long
        // as the query is also alive.
        let mut line_query = self.line_pipeline.prepare(gpu, per_frame_uniforms, scene);
        let mut mesh_query = self.mesh_pipeline.prepare(gpu, per_frame_uniforms, scene);
        let mut point_query = self.point_pipeline.prepare(gpu, per_frame_uniforms, scene);

        //do the actual rendering now
        let mut encoder = gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MainPass Encoder"),
        });
        {
            //check if the depth of color need clearing
            let color_clear_op = wgpu::LoadOp::Clear(wgpu::Color {
                r: f64::from(bg_color_tonemapped_gamma.x),
                g: f64::from(bg_color_tonemapped_gamma.y),
                b: f64::from(bg_color_tonemapped_gamma.z),
                a: f64::from(render_params.bg_color.w),
            });

            let mut selected_out_view = out_view;
            let mut store = wgpu::StoreOp::Store;
            let mut resolve_target = None;

            if render_params.msaa_nr_samples > 1 {
                resolve_target = Some(out_view);
                selected_out_view = &offscreen_fb.get(OffscreenTarget::MSAAColor).unwrap().view;
                store = wgpu::StoreOp::Discard; //No need to store the MSAA
                                                // results, since we never
                                                // sample from them again, we
                                                // just use them for resolving
            }

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[
                    //final
                    Some(wgpu::RenderPassColorAttachment {
                        view: selected_out_view,
                        resolve_target,
                        ops: wgpu::Operations { load: color_clear_op, store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &offscreen_fb.get(OffscreenTarget::Depth).unwrap().view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            //Use the piplines to render to the render targets we specified
            self.mesh_pipeline
                .run(&mut render_pass, per_frame_uniforms, render_params, &mut mesh_query);

            self.point_pipeline
                .run(&mut render_pass, per_frame_uniforms, render_params, &mut point_query);
            self.line_pipeline
                .run(&mut render_pass, per_frame_uniforms, render_params, &mut line_query);
        }
        gpu.queue().submit(Some(encoder.finish()));

        self.end_pass();
    }

    fn begin_pass(&mut self) {
        trace!("begin main_pass");
    }

    fn end_pass(&mut self) {
        trace!("end main_pass");
    }
}
