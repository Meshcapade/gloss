#![allow(dead_code)] //needed to supress warning of the check function of encase. To be removed after encase 0.11

use std::collections::HashMap;

use crate::{
    components::{FacesGPU, ModelMatrix, Name, Renderable, VertsGPU, VisMesh},
    config::RenderConfig,
    forward_renderer::{bind_group_collection::BindGroupCollection, locals::LocalEntData},
    scene::Scene,
    selector::Selector,
};
use easy_wgpu::{
    bind_group::{BindGroupBuilder, BindGroupWrapper},
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
};

use easy_wgpu::{gpu::Gpu, utils::create_empty_group};
use gloss_hecs::Entity;

use super::{pipeline_runner::PipelineRunner, upload_pass::PerFrameUniforms};

use easy_wgpu::pipeline::RenderPipelineDescBuilder;

use encase;

use gloss_utils::numerical::align;

//shaders
#[include_wgsl_oil::include_wgsl_oil("../../../shaders/entity_id.wgsl")]
mod shader_code {}

/// Render all the meshes from the scene to the `GBuffer`
pub struct EntityIdPass {
    render_pipeline: wgpu::RenderPipeline,
    _empty_group: wgpu::BindGroup,
    locals_uniform: Buffer,
    locals_bind_groups: LocalsBindGroups,
}

impl EntityIdPass {
    /// # Panics
    /// Will panic if the gbuffer does not have the correct textures that are
    /// needed for the pipeline creation
    pub fn new(gpu: &Gpu) -> Self {
        //wasm likes everything to be 16 bytes aligned
        const_assert!(std::mem::size_of::<Locals>() % 16 == 0);
        /* ---------------------------------- NOTE ---------------------------------- */
        // We explicitly set the format to `Rgba8Unorm` because we want to render the entity id to the screen for the selector
        // Using depth format `Depth16Unorm` since its half the size of `Depth32Float`
        // We cannot use the original depth buffer since that one is multi-sampled and we cant have one render target here
        // be multi-sampled and the other not. We do not want to be multi-sampling the main target here since they are ID's
        /* -------------------------------------------------------------------------- */

        // Create the render pipeline for outlines
        // The depth and Rgba8Unorm target are explicitly set since they are not prone to change
        // We choose Rgba8Unorm as the target format due to WebGL restrictions on Firefox
        // Firefox expects target to be Rgba8Unorm (RGBA/UNSIGNED_BYTE) or Rgba32Uint (RGBA_INTEGER/UNSIGNED_INT)
        let render_pipeline = RenderPipelineDescBuilder::new()
            .label("ent_id_pass")
            .shader_code(shader_code::SOURCE)
            .shader_label("ent_id_shader")
            .add_bind_group_layout_desc(PerFrameUniforms::build_layout_desc())
            .add_bind_group_layout_desc(LocalsBindGroups::build_layout_desc())
            .add_vertex_buffer_layout(VertsGPU::vertex_buffer_layout::<0>())
            .add_render_target(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
            .depth_state(Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth16Unorm,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }))
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

impl PipelineRunner for EntityIdPass {
    type QueryItems<'a> = (&'a VertsGPU, &'a FacesGPU, &'a Name, &'a ModelMatrix, &'a VisMesh);
    type QueryState<'a> = gloss_hecs::QueryBorrow<'a, gloss_hecs::With<Self::QueryItems<'a>, &'a Renderable>>;

    fn query_state(scene: &Scene) -> Self::QueryState<'_> {
        scene.world.query::<Self::QueryItems<'_>>().with::<&Renderable>()
    }
    fn prepare<'a>(&mut self, gpu: &Gpu, _per_frame_uniforms: &PerFrameUniforms, scene: &'a Scene) -> Self::QueryState<'a> {
        self.begin_pass();
        self.update_locals(gpu, scene);
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
        scene: &Scene,
    ) {
        //completely skip this if there are no entities to draw or if the selector is not active
        let selector_is_active = scene.has_resource::<Selector>();

        if query_state.iter().count() == 0 || !selector_is_active {
            return;
        }

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &per_frame_uniforms.bind_group, &[]);

        for (_id, (verts, faces, name, _model_matrix, _vis_mesh)) in query_state.iter() {
            let (local_bg, offset) = &self.locals_bind_groups.mesh2local_bind[&name.0.clone()];

            render_pass.set_bind_group(1, local_bg.bg(), &[*offset]);
            render_pass.set_vertex_buffer(0, verts.buf.buffer.slice(..));
            render_pass.set_index_buffer(faces.buf.buffer.slice(..), wgpu::IndexFormat::Uint32);
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

/// Keep in sync with shader `ent_id.wgsl`
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct Locals {
    model_matrix: nalgebra::Matrix4<f32>,
    entity_id: u32,
    pad_a: f32,
    pad_b: f32,
    pad_c: f32,
}
impl LocalEntData for Locals {
    fn new(entity: Entity, scene: &Scene) -> Self {
        let model_matrix = scene.get_comp::<&ModelMatrix>(&entity).unwrap().0.to_homogeneous();
        assert!(entity.id() < 256, "A maximum of 256 entities are allowed!");
        Locals {
            model_matrix,
            entity_id: entity.id(),
            pad_a: 0.0,
            pad_b: 0.0,
            pad_c: 0.0,
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
            .label("ent_id_pass_locals_layout")
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
