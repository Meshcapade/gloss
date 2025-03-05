use std::collections::HashMap;

use crate::{
    components::{ColorsGPU, ModelMatrix, Name, Renderable, VertsGPU, VisPoints},
    config::RenderConfig,
    forward_renderer::{bind_group_collection::BindGroupCollection, locals::LocalEntData},
    scene::Scene,
};

use easy_wgpu::{
    bind_group::{BindGroupBuilder, BindGroupWrapper},
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
    gpu::Gpu,
    pipeline::RenderPipelineDescBuilder,
    utils::create_empty_group,
};

use gloss_hecs::Entity;

use super::{pipeline_runner::PipelineRunner, upload_pass::PerFrameUniforms};

use encase;
use gloss_utils::numerical::align;

//shaders
#[include_wgsl_oil::include_wgsl_oil("../../../shaders/gbuffer_point_instanced.wgsl")]
mod shader_code {}

/// Render all the meshes from the scene to the `GBuffer`
pub struct PointPipeline {
    render_pipeline: wgpu::RenderPipeline,
    _empty_group: wgpu::BindGroup,
    locals_uniform: Buffer, // a uniform buffer that we suballocate for the locals of every mesh
    locals_bind_groups: LocalsBindGroups,
}

impl PointPipeline {
    /// # Panics
    /// Will panic if the gbuffer does not have the correct textures that are
    /// needed for the pipeline creation
    pub fn new(gpu: &Gpu, params: &RenderConfig, color_target_format: wgpu::TextureFormat, depth_target_format: wgpu::TextureFormat) -> Self {
        //wasm likes everything to be 16 bytes aligned
        const_assert!(std::mem::size_of::<Locals>() % 16 == 0);

        //render pipeline
        let render_pipeline = RenderPipelineDescBuilder::new()
            .label("point_pipeline")
            .shader_code(shader_code::SOURCE)
            .shader_label("point_shader")
            .add_bind_group_layout_desc(PerFrameUniforms::build_layout_desc())
            // .add_bind_group_layout_desc(input_layout_desc) //no need for this because we don't use shadow maps here
            .add_bind_group_layout_desc(LocalsBindGroups::build_layout_desc())
            .add_vertex_buffer_layout(VertsGPU::vertex_buffer_layout_instanced::<0>())
            .add_vertex_buffer_layout(ColorsGPU::vertex_buffer_layout_instanced::<1>())
            .add_render_target(wgpu::ColorTargetState {
                format: color_target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
            .depth_state(Some(wgpu::DepthStencilState {
                format: depth_target_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }))
            .multisample(wgpu::MultisampleState {
                count: params.msaa_nr_samples,
                ..Default::default()
            })
            .build_pipeline(gpu.device());

        let empty_group = create_empty_group(gpu.device());

        let size_bytes = 0x10000;
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM;
        let locals_uniform = Buffer::new_empty(gpu.device(), usage, Some("local_buffer"), size_bytes);

        let locals_bind_groups = LocalsBindGroups::new(gpu);

        Self {
            render_pipeline,
            _empty_group: empty_group,
            locals_uniform,
            locals_bind_groups,
        }
    }
}

impl PipelineRunner for PointPipeline {
    type QueryItems<'a> = (&'a VertsGPU, &'a ColorsGPU, &'a VisPoints, &'a Name);
    type QueryState<'a> = gloss_hecs::QueryBorrow<'a, gloss_hecs::With<Self::QueryItems<'a>, &'a Renderable>>;

    fn query_state(scene: &Scene) -> Self::QueryState<'_> {
        scene.world.query::<Self::QueryItems<'_>>().with::<&Renderable>()
    }

    fn prepare<'a>(&mut self, gpu: &Gpu, _per_frame_uniforms: &PerFrameUniforms, scene: &'a Scene) -> Self::QueryState<'a> {
        self.begin_pass();
        self.update_locals(gpu, scene);
        Self::query_state(scene)
    }

    fn run<'r>(
        &'r mut self,
        render_pass: &mut wgpu::RenderPass<'r>,
        per_frame_uniforms: &'r PerFrameUniforms,
        _render_params: &RenderConfig,
        query_state: &'r mut Self::QueryState<'_>,
    ) {
        //completely skip this if there are no entities to draw
        if query_state.iter().count() == 0 {
            return;
        }

        render_pass.set_pipeline(&self.render_pipeline);

        //global binding
        render_pass.set_bind_group(0, &per_frame_uniforms.bind_group, &[]);
        //No need for the input binding because we don't use shadow maps during point
        // rendering

        for (_id, (verts, colors, vis_points, name)) in query_state.iter() {
            if !vis_points.show_points {
                continue;
            }

            //local bindings
            let (local_bg, offset) = &self.locals_bind_groups.mesh2local_bind[&name.0.clone()];
            render_pass.set_bind_group(1, local_bg.bg(), &[*offset]);
            render_pass.set_vertex_buffer(0, verts.buf.slice(..));
            render_pass.set_vertex_buffer(1, colors.buf.slice(..));
            render_pass.draw(0..6, 0..verts.nr_vertices);
        }
    }

    fn begin_pass(&mut self) {}

    fn update_locals(&mut self, gpu: &Gpu, scene: &Scene) {
        Self::update_locals_inner::<Locals, _>(
            gpu,
            scene,
            &mut self.locals_uniform,
            &mut self.locals_bind_groups,
            &mut Self::query_state(scene),
        );
    }
}

/// Keep in sync with shader `gbuffer_mesh.wgsl`
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct Locals {
    model_matrix: nalgebra::Matrix4<f32>,
    color_type: i32,
    point_color: nalgebra::Vector4<f32>,
    point_size: f32,
    is_point_size_in_world_space: u32,
    zbuffer: u32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    // pad_b: f32,
    // pad_c: f32,
    // pad_d: f32,
}
impl LocalEntData for Locals {
    fn new(entity: Entity, scene: &Scene) -> Self {
        let model_matrix = scene.get_comp::<&ModelMatrix>(&entity).unwrap().0.to_homogeneous();
        let vis_points = scene.get_comp::<&VisPoints>(&entity).unwrap();
        let color_type = vis_points.color_type as i32;
        Locals {
            model_matrix,
            color_type,
            point_color: vis_points.point_color,
            point_size: vis_points.point_size,
            is_point_size_in_world_space: u32::from(vis_points.is_point_size_in_world_space),
            zbuffer: u32::from(vis_points.zbuffer),
            // pad_d: 0.0,
        }
    }
}

struct LocalsBindGroups {
    layout: wgpu::BindGroupLayout,
    pub mesh2local_bind: HashMap<String, (BindGroupWrapper, u32)>,
}
impl BindGroupCollection for LocalsBindGroups {
    fn new(gpu: &Gpu) -> Self {
        Self {
            layout: Self::build_layout_desc().into_bind_group_layout(gpu.device()),
            mesh2local_bind: HashMap::default(),
        }
    }

    fn build_layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new()
            .label("gbuffer_pass_locals_layout")
            //locals
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                true,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<Locals>()).unwrap(), 256))),
            )
            .build()
    }

    fn update_bind_group(&mut self, _entity: Entity, gpu: &Gpu, mesh_name: &str, ubo: &Buffer, offset_in_ubo: u32, _scene: &Scene) {
        let entries = BindGroupBuilder::new().add_entry_buf_chunk::<Locals>(&ubo.buffer).build_entries();

        self.update_if_stale(mesh_name, entries, offset_in_ubo, gpu);
    }
    fn get_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
    fn get_mut_entity2binds(&mut self) -> &mut HashMap<String, (BindGroupWrapper, u32)> {
        &mut self.mesh2local_bind
    }
}
