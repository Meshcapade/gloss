use easy_wgpu::{
    bind_group::{BindGroupBuilder, BindGroupDesc, BindGroupWrapper},
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    gpu::Gpu,
    pipeline::RenderPipelineDescBuilder,
    texture::Texture,
};
use log::debug;

//shaders
#[include_wgsl_oil::include_wgsl_oil("../../../shaders/blit.wgsl")]
mod shader_code {}

/// Rendering pass which copies from a texture towards another one, resizing if
/// necessary. Useful for copying the final rendered texture towards the screen.
pub struct BlitPass {
    render_pipeline: wgpu::RenderPipeline,
    //per_pass things
    sampler: wgpu::Sampler,
    input_layout: wgpu::BindGroupLayout,
    input_bind_group: Option<BindGroupWrapper>,
}

impl BlitPass {
    pub fn new(gpu: &Gpu, surface_format: &wgpu::TextureFormat) -> Self {
        let input_layout_desc = Self::input_layout_desc();
        let input_layout = input_layout_desc.clone().into_bind_group_layout(gpu.device());

        //render pipeline
        let render_pipeline = RenderPipelineDescBuilder::new()
            .label("blit pipeline")
            .shader_code(shader_code::SOURCE)
            .shader_label("blit_shader")
            .add_bind_group_layout_desc(input_layout_desc) // for the texture we are sampling
            .add_render_target(wgpu::ColorTargetState {
                format: *surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })
            .depth_state(None)
            .multisample(wgpu::MultisampleState::default())
            .build_pipeline(gpu.device());

        //sampler
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            render_pipeline,
            sampler,
            input_layout,
            input_bind_group: None,
        }
    }

    /// # Panics
    /// Will panic if the `input_bind_group` is not created. It should be
    /// created automatically when doing ``run()`` by the `update_bind_group()`
    pub fn run(&mut self, gpu: &Gpu, src_texture: &Texture, out_view: &wgpu::TextureView) {
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Blit Encoder") });

        {
            //update the bind group in case the input_texture changed
            self.update_input_bind_group(gpu, src_texture);

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Blit Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: out_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                render_pass.set_pipeline(&self.render_pipeline);

                render_pass.set_bind_group(0, self.input_bind_group.as_ref().unwrap().bg(), &[]);

                //draw a quad
                render_pass.draw(0..4, 0..1);
            }
        }

        gpu.queue().submit(Some(encoder.finish()));
    }

    fn input_layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new()
            .label("blit_texture_layout")
            //input texture
            .add_entry_tex(wgpu::ShaderStages::FRAGMENT, wgpu::TextureSampleType::Float { filterable: true })
            //sampler
            .add_entry_sampler(wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering)
            .build()
    }

    fn update_input_bind_group(&mut self, gpu: &Gpu, src_texture: &Texture) {
        let entries = BindGroupBuilder::new()
            .add_entry_tex(src_texture)
            .add_entry_sampler(&self.sampler)
            .build_entries();
        let stale = self.input_bind_group.as_ref().map_or(true, |b| b.is_stale(&entries)); //returns true if the bg has not been created or if stale
        if stale {
            debug!("blit bind group is stale, recreating");
            self.input_bind_group = Some(BindGroupDesc::new("blit_bg", entries).into_bind_group_wrapper(gpu.device(), &self.input_layout));
        }
    }
}
