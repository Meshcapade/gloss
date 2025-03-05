extern crate nalgebra as na;

use crate::{
    components::{FacesGPU, LightEmit, ModelMatrix, Name, PosLookat, Renderable, ShadowCaster, ShadowMap, ShadowMapDirty, VertsGPU, VisMesh},
    scene::Scene,
};
use easy_wgpu::{
    bind_group::{BindGroupBuilder, BindGroupDesc, BindGroupWrapper},
    bind_group_layout::{BindGroupLayoutBuilder, BindGroupLayoutDesc},
    buffer::Buffer,
    gpu::Gpu,
    pipeline::RenderPipelineDescBuilder,
};
use gloss_hecs::{Changed, CommandBuffer, Entity, Query, QueryBorrow};
use gloss_utils::numerical::{align, align_usz};
use log::debug;
use std::collections::HashMap;

/// Shadow pass which renders depth maps for each of the lights. Only updates
/// the depth map when the vertices on the gpu have changed.
pub struct ShadowPass {
    render_pipeline: wgpu::RenderPipeline,
    locals_uniform: LocalsUniform,                // a uniform buffer that we suballocate for the locals of every mesh
    iterator_light_uniform: IteratorLightUniform, //a uniform buffer that we allocate only an index to use for indexing into our locals
    command_buffer: CommandBuffer,
}
use super::upload_pass::PerFrameUniforms;

//shaders
#[include_wgsl_oil::include_wgsl_oil("../../../shaders/shadow_map.wgsl")]
mod shader_code {}

impl ShadowPass {
    pub fn new(gpu: &Gpu) -> Self {
        //render pipeline
        let render_pipeline = RenderPipelineDescBuilder::new()
            .label("shadow pass pipeline")
            .shader_code(shader_code::SOURCE)
            .add_bind_group_layout_desc(PerFrameUniforms::build_layout_desc())
            .add_bind_group_layout_desc(IteratorLightUniform::layout_desc())
            .add_bind_group_layout_desc(LocalsUniform::layout_desc())
            .add_vertex_buffer_layout(VertsGPU::vertex_buffer_layout::<0>())
            .depth_state(Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }))
            .multisample(wgpu::MultisampleState::default())
            .build_pipeline(gpu.device());

        let locals_uniform = LocalsUniform::new(gpu);

        let iterator_light_uniform = IteratorLightUniform::new(gpu);

        let command_buffer = CommandBuffer::new();

        Self {
            render_pipeline,
            locals_uniform,
            iterator_light_uniform,
            command_buffer,
        }
    }

    pub fn run(&mut self, gpu: &Gpu, per_frame_uniforms: &PerFrameUniforms, scene: &mut Scene) {
        self.begin_pass();

        self.render_shadows(gpu, per_frame_uniforms, scene);

        self.end_pass(scene);
    }

    fn render_shadows(&mut self, gpu: &Gpu, per_frame_uniforms: &PerFrameUniforms, scene: &Scene) {
        let shadow_map_requires_update = self.check_shadow_maps_dirty(scene);
        debug!("shadow_map_requires_update {}", shadow_map_requires_update);
        if !shadow_map_requires_update {
            return; //nothing to do
        }

        let mut query_all_renderables = scene.world.query::<&Renderable>();
        let mut query_meshes_for_shadow = scene.world.query::<(&VertsGPU, &FacesGPU, &VisMesh)>().with::<&Renderable>();

        //upload to gpu the local information for each mesh like model matrix
        self.update_locals(gpu, &mut query_all_renderables, scene);

        //check for every light if it needs update and if it does, render all the
        // meshes towards it's shadowmap
        for (entity_light, shadow_map) in &mut scene.world.query::<&ShadowMap>().with::<(&LightEmit, &ShadowCaster)>() {
            //do the actual rendering now
            let mut encoder = gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Shadow pass encoder"),
            });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Shadow Pass"),
                    color_attachments: &[
                        //depth moments
                        // Some(wgpu::RenderPassColorAttachment {
                        //     view: &shadow_map.tex_depth_moments.view,
                        //     resolve_target: None,
                        //     ops: wgpu::Operations {
                        //         load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        //         store: true,
                        //     },
                        // }),
                    ],
                    // depth_stencil_attachment: None,
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &shadow_map.tex_depth.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                render_pass.set_pipeline(&self.render_pipeline);

                //gloal binding
                render_pass.set_bind_group(0, &per_frame_uniforms.bind_group, &[]);

                //iterator the light, we need to pass this idx into a buffer so we can access
                // it in a shader
                let light_name = scene.get_comp::<&Name>(&entity_light).unwrap().0.clone();
                let light_idx = per_frame_uniforms.light2idx_ubo[&light_name];
                self.iterator_light_uniform.set(gpu, light_idx); //writes to gpu
                render_pass.set_bind_group(1, self.iterator_light_uniform.bind_group.bg(), &[]);

                for (entity_mesh, (verts, faces, vis)) in query_meshes_for_shadow.iter() {
                    if !vis.show_mesh {
                        continue;
                    }

                    //local bindings
                    let name = scene.get_comp::<&Name>(&entity_mesh).unwrap().0.clone();
                    let (local_bg, offset) = &self.locals_uniform.mesh2local_bind[&name];
                    render_pass.set_bind_group(2, local_bg.bg(), &[*offset]);
                    // println!("Rendering mesh to shadow map {}", name);

                    render_pass.set_vertex_buffer(0, verts.buf.slice(..));
                    render_pass.set_index_buffer(faces.buf.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..faces.nr_triangles * 3, 0, 0..1);
                }

                //TODO render points
            }
            debug!("shadow encoder");
            gpu.queue().submit(Some(encoder.finish()));
        }
    }

    fn begin_pass(&mut self) {
        // self.locals_uniform.buffer.reset_chunks_offset();
        self.locals_uniform.buffer.reset_chunks_offset_if_necessary();
    }

    fn end_pass(&mut self, scene: &mut Scene) {
        //remove all dirty components from entities
        {
            let mut query_all_renderables = scene.world.query::<&Renderable>();
            for (id, _comps) in query_all_renderables.iter() {
                self.command_buffer.remove_one::<ShadowMapDirty>(id);
            }
        }

        self.command_buffer.run_on(&mut scene.world);
    }

    fn check_shadow_maps_dirty(&self, scene: &Scene) -> bool {
        let mut query_for_shadow = scene
            .world
            .query::<(Option<&ShadowMapDirty>, Changed<VertsGPU>, Changed<ModelMatrix>)>()
            .with::<&Renderable>();

        //check if this light has had any changes in the scene and therefore we might
        // need to update the shadow map. we do this check first because if the
        // scene has not changed we don't even need to start a encoder or a render pass
        let mut shadow_map_requires_update = false;
        //meshes
        for (_entity_mesh, (shadow_map_opt, changed_verts, changed_model_matrix)) in query_for_shadow.iter() {
            shadow_map_requires_update = shadow_map_requires_update | changed_verts | changed_model_matrix | shadow_map_opt.is_some();
        }

        //check if the light shadow casting component has changed, like it was disabled
        // or the resolution has changed
        let mut query_for_lights = scene.world.query::<Changed<ShadowCaster>>();
        for (_ent, changed_shadow_casting) in query_for_lights.iter() {
            shadow_map_requires_update |= changed_shadow_casting;
        }

        //check if the renderable component has been added or changed in which case we
        // need to update the shadows
        let mut query_for_renderable = scene.world.query::<Changed<Renderable>>();
        for (_entity_mesh, changed_renderable) in query_for_renderable.iter() {
            shadow_map_requires_update |= changed_renderable;
        }

        //if ANY entity has their renderable component removed, then we also update
        // shadows
        if !scene.world.removed::<Renderable>().is_empty() {
            shadow_map_requires_update = true;
        }

        //if any light has moves we also update shadows
        let mut query_for_lights = scene.world.query::<(Changed<PosLookat>,)>().with::<&LightEmit>();
        for (_entity_mesh, (changed_poslookat,)) in query_for_lights.iter() {
            shadow_map_requires_update |= changed_poslookat;
        }

        shadow_map_requires_update
    }

    fn update_locals<Q: Query>(&mut self, gpu: &Gpu, query_state: &mut QueryBorrow<'_, Q>, scene: &Scene) {
        // Update the local binding groups for the meshes we render. We do it in two
        // passes because the binding group cannot be created and consumed in the same
        // loop
        for (id, _comps) in query_state.iter() {
            let name = scene.get_comp::<&Name>(&id).unwrap().0.clone();

            //upload local stuff
            let locals = Locals::new(id, scene);
            let offset_in_ubo = self.locals_uniform.buffer.push_cpu_chunk_aligned::<Locals>(&locals);

            //chekc if we need to recreate bind group (for example when any of the textures
            // of the meshes have changed)
            self.locals_uniform.update_bind_group(id, gpu, &name, offset_in_ubo, scene);
        }
        self.locals_uniform.buffer.upload_from_cpu_chunks(gpu.queue()); //important to upload everything to gpu at the end
    }
}

/// Keep in sync with shader
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct Locals {
    model_matrix: nalgebra::Matrix4<f32>,
}
impl Locals {
    pub fn new(entity: Entity, scene: &Scene) -> Self {
        let model_matrix = scene.get_comp::<&ModelMatrix>(&entity).unwrap().0.to_homogeneous();
        Locals { model_matrix }
    }
}

struct LocalsUniform {
    pub buffer: Buffer,
    layout: wgpu::BindGroupLayout,
    pub mesh2local_bind: HashMap<String, (BindGroupWrapper, u32)>,
}
impl LocalsUniform {
    pub fn new(gpu: &Gpu) -> Self {
        let size_bytes = 0x10000;
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM;
        let buffer = Buffer::new_empty(gpu.device(), usage, Some("local_buffer"), size_bytes);

        let layout = Self::layout_desc().into_bind_group_layout(gpu.device());

        Self {
            buffer,
            layout,
            mesh2local_bind: HashMap::default(),
        }
    }

    //keep as associated function so we can call it in the pipeline creation
    // without and object
    pub fn layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new()
            .label("shadow_pass_locals_layout")
            //Locals
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                true,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<Locals>()).unwrap(), 256))),
            )
            .build()
    }

    pub fn update_bind_group(
        &mut self,
        _entity: Entity,
        gpu: &Gpu,
        // offset: &u32,
        mesh_name: &str,
        offset_in_ubo: u32,
        _scene: &Scene,
    ) {
        let entries = BindGroupBuilder::new().add_entry_buf_chunk::<Locals>(&self.buffer.buffer).build_entries();

        //if there is no entry for the bind group or if the current one is stale, we
        // recreate it
        if !self.mesh2local_bind.contains_key(mesh_name) || self.mesh2local_bind[mesh_name].0.is_stale(&entries) {
            debug!("shadowmap_local_bg_recreating");
            let bg = BindGroupDesc::new("shadow_local_bg", entries).into_bind_group_wrapper(gpu.device(), &self.layout);
            let bg_and_offset = (bg, offset_in_ubo);
            self.mesh2local_bind.insert(mesh_name.to_string(), bg_and_offset);
        }

        //sometimes just the offset of the bind group changes so we also make sure to
        // update this.
        self.mesh2local_bind.entry(mesh_name.to_string()).and_modify(|r| r.1 = offset_in_ubo);
    }
}

/// Keep in sync with shader
#[repr(C)]
#[derive(Clone, Copy, encase::ShaderType)]
struct IteratorLight {
    light_idx: u32,
    //wasm needs padding to 16 bytes https://github.com/gfx-rs/wgpu/issues/2932
    pad_0: u32,
    pad_1: u32,
    pad_2: u32,
}
impl IteratorLight {
    pub fn new(light_idx: u32) -> Self {
        IteratorLight {
            light_idx,
            pad_0: 0,
            pad_1: 0,
            pad_2: 0,
        }
    }
}

struct IteratorLightUniform {
    pub buffer: Buffer,
    pub bind_group: BindGroupWrapper,
}
impl IteratorLightUniform {
    pub fn new(gpu: &Gpu) -> Self {
        let size_bytes = align_usz(std::mem::size_of::<IteratorLight>(), 256);
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM;
        let buffer = Buffer::new_empty(gpu.device(), usage, Some("iterator_light_buffer"), size_bytes);

        let layout = Self::layout_desc().into_bind_group_layout(gpu.device());

        //can build it only once and leave it like this because we won't reallocate the
        // buffer
        let bind_group = BindGroupBuilder::new().add_entry_buf(&buffer.buffer).build(gpu.device(), &layout);

        Self {
            buffer,
            // layout,
            bind_group,
        }
    }

    pub fn set(&mut self, gpu: &Gpu, idx: u32) {
        let iterator = IteratorLight::new(idx);
        self.buffer.reset_chunks_offset();
        self.buffer.push_cpu_chunk_packed(&iterator);
        self.buffer.upload_from_cpu_chunks(gpu.queue());
    }

    //keep as associated function so we can call it in the pipeline creation
    // without and object
    pub fn layout_desc() -> BindGroupLayoutDesc {
        BindGroupLayoutBuilder::new()
            .label("shadow_pass_iterator_layout")
            //InteratorLight
            .add_entry_uniform(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                false,
                wgpu::BufferSize::new(u64::from(align(u32::try_from(std::mem::size_of::<IteratorLight>()).unwrap(), 256))),
            )
            .build()
    }
}
