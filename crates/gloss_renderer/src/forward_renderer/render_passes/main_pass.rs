use crate::config::RenderConfig;
use log::trace;
use nalgebra as na;

use crate::scene::Scene;

use easy_wgpu::gpu::Gpu;

use super::{
    entity_id_pass::EntityIdPass, line_pipeline::LinePipeline, mesh_pipeline::MeshPipeline, outline_pass::OutlinePass, point_pipeline::PointPipeline,
    upload_pass::PerFrameUniforms,
};

use crate::forward_renderer::{render_passes::pipeline_runner::PipelineRunner, renderer::OffscreenTarget};
use easy_wgpu::framebuffer::FrameBuffer;

/// Render all the meshes from the scene
pub struct MainPass {
    mesh_pipeline: MeshPipeline,
    point_pipeline: PointPipeline,
    line_pipeline: LinePipeline,
    outline_pass: OutlinePass,
    entity_id_pass: EntityIdPass,
}

impl MainPass {
    pub fn new(gpu: &Gpu, params: &RenderConfig, color_target_format: wgpu::TextureFormat, depth_target_format: wgpu::TextureFormat) -> Self {
        let mesh_pipeline = MeshPipeline::new(gpu, params, color_target_format, depth_target_format);
        let point_pipeline = PointPipeline::new(gpu, params, color_target_format, depth_target_format);
        let line_pipeline = LinePipeline::new(gpu, params, color_target_format, depth_target_format);
        let outline_pass = OutlinePass::new(gpu, params, color_target_format, depth_target_format);
        let entity_id_pass = EntityIdPass::new(gpu);
        Self {
            mesh_pipeline,
            point_pipeline,
            line_pipeline,
            outline_pass,
            entity_id_pass,
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
        let aces = gloss_utils::tonemap::AcesFitted::new();
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
        let mut outline_query = self.outline_pass.prepare(gpu, per_frame_uniforms, scene);
        let mut entity_id_query = self.entity_id_pass.prepare(gpu, per_frame_uniforms, scene);
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
            // Main pass - to run the mesh, outline, point and line render passes
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Main Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: selected_out_view,
                        resolve_target,
                        ops: wgpu::Operations { load: color_clear_op, store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &offscreen_fb.get(OffscreenTarget::Depth).unwrap().view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Store,
                        }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                self.mesh_pipeline
                    .run(&mut render_pass, per_frame_uniforms, render_params, &mut mesh_query, scene);
                self.point_pipeline
                    .run(&mut render_pass, per_frame_uniforms, render_params, &mut point_query, scene);
                self.line_pipeline
                    .run(&mut render_pass, per_frame_uniforms, render_params, &mut line_query, scene);
                // Run outline pass last so that its able to draw over the line and point passes (grid floor draws over outline otherwise)
                self.outline_pass
                    .run(&mut render_pass, per_frame_uniforms, render_params, &mut outline_query, scene);
            }

            // Run entity id pass if a click happened
            if let Some(camera) = scene.get_current_cam() {
                if camera.is_click(scene) {
                    let mut entity_id_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Entity ID Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &offscreen_fb.get(OffscreenTarget::EntityID).unwrap().view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &offscreen_fb.get(OffscreenTarget::SingleDepth).unwrap().view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Discard,
                            }),
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    self.entity_id_pass
                        .run(&mut entity_id_pass, per_frame_uniforms, render_params, &mut entity_id_query, scene);
                }
            }
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
