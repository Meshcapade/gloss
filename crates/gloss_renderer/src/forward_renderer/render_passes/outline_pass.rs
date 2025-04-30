use std::collections::HashMap;

use crate::{
    components::{FacesGPU, ModelMatrix, Name, NormalsGPU, Renderable, VertsGPU, VisMesh, VisOutline},
    config::RenderConfig,
    forward_renderer::{bind_group_collection::BindGroupCollection, locals::LocalEntData},
    scene::Scene,
};
use easy_wgpu::{
    bind_group::{BindGroupBuilder, BindGroupWrapper},
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
};
// use gloss_utils::log;

use easy_wgpu::{gpu::Gpu, utils::create_empty_group};
use gloss_hecs::Entity;

use super::{pipeline_runner::PipelineRunner, upload_pass::PerFrameUniforms};

use easy_wgpu::pipeline::RenderPipelineDescBuilder;

use encase;

use gloss_utils::numerical::align;

//shaders
#[include_wgsl_oil::include_wgsl_oil("../../../shaders/outline.wgsl")]
mod shader_code {}

/// Render all the meshes from the scene to the `GBuffer`
pub struct OutlinePass {
    render_pipeline: wgpu::RenderPipeline,
    _empty_group: wgpu::BindGroup,
    locals_uniform: Buffer,
    locals_bind_groups: LocalsBindGroups,
}

impl OutlinePass {
    /// # Panics
    /// Will panic if the gbuffer does not have the correct textures that are
    /// needed for the pipeline creation
    pub fn new(gpu: &Gpu, params: &RenderConfig, color_target_format: wgpu::TextureFormat, depth_target_format: wgpu::TextureFormat) -> Self {
        //wasm likes everything to be 16 bytes aligned
        const_assert!(std::mem::size_of::<Locals>() % 16 == 0);

        // Create the render pipeline for outlines
        let render_pipeline = RenderPipelineDescBuilder::new()
            .label("outline_pass")
            .shader_code(shader_code::SOURCE)
            .shader_label("outline_shader")
            .add_bind_group_layout_desc(PerFrameUniforms::build_layout_desc())
            .add_bind_group_layout_desc(LocalsBindGroups::build_layout_desc())
            .add_vertex_buffer_layout(VertsGPU::vertex_buffer_layout::<0>())
            .add_vertex_buffer_layout(NormalsGPU::vertex_buffer_layout::<1>())
            .add_render_target(wgpu::ColorTargetState {
                format: color_target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
            .depth_state(Some(wgpu::DepthStencilState {
                format: depth_target_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater, // Always pass depth test
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::NotEqual, // Only draw where stencil is not 1
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Keep,
                    },
                    back: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::NotEqual, // Only draw where stencil is not 1
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Keep,
                    },
                    read_mask: 0xFF,
                    write_mask: 0x00, // Don't write to stencil
                },
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
impl PipelineRunner for OutlinePass {
    type QueryItems<'a> = (
        &'a VertsGPU,
        &'a FacesGPU,
        &'a NormalsGPU,
        &'a VisMesh,
        &'a Name,
        &'a ModelMatrix,
        &'a VisOutline,
    );
    type QueryState<'a> = gloss_hecs::QueryBorrow<'a, gloss_hecs::With<Self::QueryItems<'a>, &'a Renderable>>;

    fn query_state(scene: &Scene) -> Self::QueryState<'_> {
        scene.world.query::<Self::QueryItems<'_>>().with::<&Renderable>()
    }

    fn prepare<'a>(&mut self, gpu: &Gpu, _per_frame_uniforms: &PerFrameUniforms, scene: &'a Scene) -> Self::QueryState<'a> {
        self.begin_pass();

        self.update_locals(gpu, scene);

        //update the bind group in case the input_texture or the shadowmaps changed
        // self.update_input_bind_group(gpu, scene, per_frame_uniforms);

        Self::query_state(scene)
    }

    /// # Panics
    /// Will panic if the input bind groups are not created
    #[allow(clippy::too_many_lines)]
    fn run<'r>(
        &'r mut self,
        render_pass: &mut wgpu::RenderPass<'r>,
        per_frame_uniforms: &'r PerFrameUniforms,
        _render_params: &RenderConfig,
        query_state: &'r mut Self::QueryState<'_>,
        _scene: &Scene,
    ) {
        //completely skip this if there are no entities to draw
        if query_state.iter().count() == 0 {
            return;
        }

        // Draw outlines using stencil test
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &per_frame_uniforms.bind_group, &[]);
        render_pass.set_stencil_reference(1); // Use 1 as the reference value

        for (_id, (verts, faces, normals, vis_mesh, name, _model_matrix, vis_outline)) in query_state.iter() {
            if !vis_mesh.show_mesh || !vis_outline.show_outline {
                continue;
            }

            let (local_bg, offset) = &self.locals_bind_groups.mesh2local_bind[&name.0.clone()];

            render_pass.set_bind_group(1, local_bg.bg(), &[*offset]);

            render_pass.set_vertex_buffer(0, verts.buf.slice(..));
            render_pass.set_vertex_buffer(1, normals.buf.slice(..));
            render_pass.set_index_buffer(faces.buf.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..faces.nr_triangles * 3, 0, 0..1);
        }
    }

    fn begin_pass(&mut self) {}

    /// update the local information that need to be sent to the gpu for each
    /// mesh like te model matrix
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

/// Keep in sync with shader `outline.wgsl`
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct Locals {
    model_matrix: nalgebra::Matrix4<f32>,
    outline_color: nalgebra::Vector4<f32>,
    outline_width: f32,
    is_floor: u32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    // pad_b: f32,
    pad_c: f32,
    pad_d: f32,
}
impl LocalEntData for Locals {
    fn new(entity: Entity, scene: &Scene) -> Self {
        let model_matrix = scene.get_comp::<&ModelMatrix>(&entity).unwrap().0.to_homogeneous();

        // Get outline properties from VisOutline component
        let Ok(vis_outline) = scene.get_comp::<&VisOutline>(&entity) else {
            // Default values if entity doesn't have VisOutline
            return Locals {
                model_matrix,
                outline_color: nalgebra::Vector4::new(1.0, 0.5, 0.0, 1.0), // Default orange
                outline_width: 0.0,                                        // No outline by default
                is_floor: 0,
                pad_c: 0.0,
                pad_d: 0.0,
            };
        };

        let is_floor = if let Some(floor) = scene.get_floor() {
            floor.entity == entity
        } else {
            false
        };
        let is_floor = u32::from(is_floor);

        Locals {
            model_matrix,
            outline_color: vis_outline.outline_color,
            outline_width: vis_outline.outline_width,
            is_floor,
            pad_c: 0.0,
            pad_d: 0.0,
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

    //keep as associated function so we can call it in the pipeline creation
    // without and object
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
